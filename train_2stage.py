import argparse
import os
import random
import logging
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention_processor import XFormersAttnProcessor

from models.integrated_champ_model import IntegratedChampModel
from models.m2p_encoder import M2PEncoder, ArcFaceLoss, PoseLoss
from models.guidance_encoder import GuidanceMusicEncoder
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.reference_control import ReferenceControlWriter, ReferenceControlReader

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import warnings
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="2-Stage CHAMP Training: M2PEncoder + StableAnimator")
    
    # Training stage
    parser.add_argument("--training_stage", type=str, default="A", choices=["A", "B", "C"],
                        help="Training stage: A (M2P pretrain), B (Image), C (Motion)")
    parser.add_argument("--auto_progress", action="store_true",
                        help="Automatically progress through training stages")
    
    # Model configuration
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained model")
    parser.add_argument("--m2p_config", type=str, default=None,
                        help="Path to M2P encoder config")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./champ_2stage_output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--sequence_length", type=int, default=24, help="Video sequence length")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="LR scheduler cycles")
    parser.add_argument("--lr_power", type=float, default=1.0, help="LR scheduler power")
    
    # Stage-specific learning rates
    parser.add_argument("--stage_a_lr", type=float, default=1e-4, help="Stage A learning rate")
    parser.add_argument("--stage_b_lr", type=float, default=1e-4, help="Stage B learning rate")
    parser.add_argument("--stage_c_lr", type=float, default=5e-5, help="Stage C learning rate")
    parser.add_argument("--m2p_lr_stage_b", type=float, default=1e-5, help="M2P LR in Stage B")
    
    # Loss weights
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0, help="Diffusion loss weight")
    parser.add_argument("--pose_loss_weight", type=float, default=0.5, help="Pose loss weight")
    parser.add_argument("--face_loss_weight", type=float, default=0.1, help="Face loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=0.1, help="Identity loss weight")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.1, help="Temporal loss weight")
    
    # Guidance configuration
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--use_m2p_guidance", action="store_true", default=True,
                        help="Use M2P predictions as guidance")
    parser.add_argument("--scheduled_sampling", action="store_true",
                        help="Use scheduled sampling for guidance")
    
    # Data configuration
    parser.add_argument("--train_data_dir", type=str, default=None, help="Training data directory")
    parser.add_argument("--validation_data_dir", type=str, default=None, help="Validation data directory")
    
    # Logging and saving
    parser.add_argument("--logging_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb"],
                        help="Reporting backend")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Checkpoint saving frequency")
    parser.add_argument("--validation_steps", type=int, default=100, help="Validation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Model saving frequency")
    
    # Stage progression
    parser.add_argument("--stage_a_steps", type=int, default=5000, help="Steps for Stage A")
    parser.add_argument("--stage_b_steps", type=int, default=10000, help="Steps for Stage B")
    parser.add_argument("--stage_c_steps", type=int, default=15000, help="Steps for Stage C")
    
    return parser.parse_args()

def create_m2p_encoder(args):
    """Create M2P encoder"""
    m2p_encoder = M2PEncoder(
        music_input_dim=4800,
        hidden_dim=1024,
        pose_channels=134,
        face_embed_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    return m2p_encoder

def create_guidance_encoders():
    """Create guidance encoders"""
    guidance_encoder_group = {}
    
    # Pose guidance encoder
    pose_encoder = GuidanceMusicEncoder(
        guidance_embedding_channels=1280,
        guidance_input_channels=134,
        block_out_channels=(16, 32, 96, 256),
        attention_num_heads=12
    )
    guidance_encoder_group['pose'] = pose_encoder
    
    return guidance_encoder_group

def create_integrated_model(args, device):
    """Create integrated CHAMP model"""
    
    # Create M2P encoder
    m2p_encoder = create_m2p_encoder(args)
    
    # Load pretrained UNet models
    reference_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        use_safetensors=True
    )
    
    denoising_unet = UNet3DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        use_safetensors=True
    )
    
    # Create reference control components
    reference_control_writer = ReferenceControlWriter()
    reference_control_reader = ReferenceControlReader()
    
    # Create guidance encoders
    guidance_encoder_group = create_guidance_encoders()
    
    # Create VAE and CLIP encoders (will be frozen)
    vae_encoder = None
    vae_decoder = None
    clip_encoder = None
    arcface_encoder = None
    
    try:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae"
        )
        vae_encoder = vae.encoder
        vae_decoder = vae.decoder
        vae_encoder.requires_grad_(False)
        vae_decoder.requires_grad_(False)
    except:
        print("Warning: Could not load VAE, using dummy encoders")
    
    try:
        clip_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        clip_encoder.requires_grad_(False)
    except:
        print("Warning: Could not load CLIP, using dummy encoder")
    
    # Create integrated model
    integrated_model = IntegratedChampModel(
        m2p_encoder=m2p_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        guidance_encoder_group=guidance_encoder_group,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        clip_encoder=clip_encoder,
        arcface_encoder=arcface_encoder
    )
    
    return integrated_model.to(device)

def create_loss_functions(args):
    """Create loss functions for different stages"""
    pose_loss = PoseLoss(
        heatmap_weight=1.0,
        coordinate_weight=0.5,
        temporal_weight=0.1
    )
    
    face_loss = ArcFaceLoss(margin=0.5, scale=64.0)
    
    return pose_loss, face_loss

def create_optimizer(model, args, stage):
    """Create optimizer for specific training stage"""
    if stage == "A":
        lr = args.stage_a_lr
        params = model.get_trainable_parameters("A")
    elif stage == "B":
        lr = args.stage_b_lr
        params = model.get_trainable_parameters("B")
    elif stage == "C":
        lr = args.stage_c_lr
        params = model.get_trainable_parameters("C")
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    # Apply different learning rates for M2P encoder in Stage B
    if stage == "B":
        m2p_params = []
        other_params = []
        for param in params:
            if 'm2p_encoder' in param.name:
                m2p_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': m2p_params, 'lr': args.m2p_lr_stage_b},
            {'params': other_params, 'lr': lr}
        ], betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)
    else:
        optimizer = torch.optim.AdamW(
            params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08
        )
    
    return optimizer

def create_scheduler(optimizer, args, num_training_steps):
    """Create learning rate scheduler"""
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=num_training_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    return lr_scheduler

def create_noise_scheduler():
    """Create noise scheduler for diffusion training"""
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    return noise_scheduler

def training_function(args):
    """
    Main training function with stage progression
    
    Args:
        args: Training arguments and configuration
        
    Returns:
        None
        
    Function: Orchestrates the complete 2-stage training pipeline (M2P pretrain -> Combined training)
    """
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir,
    )
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
    
    # Create integrated model
    device = accelerator.device
    integrated_model = create_integrated_model(args, device)
    
    # Create loss functions
    pose_loss_fn, face_loss_fn = create_loss_functions(args)
    
    # Create noise scheduler
    noise_scheduler = create_noise_scheduler()
    
    # Set initial training stage
    current_stage = args.training_stage
    integrated_model.set_training_stage(current_stage)
    
    # Create optimizer for current stage
    optimizer = create_optimizer(integrated_model, args, current_stage)
    
    # Calculate training steps for current stage
    if current_stage == "A":
        max_steps = args.stage_a_steps
    elif current_stage == "B":
        max_steps = args.stage_b_steps
    elif current_stage == "C":
        max_steps = args.stage_c_steps
    else:
        max_steps = args.max_train_steps or 1000
    
    # Create learning rate scheduler
    lr_scheduler = create_scheduler(optimizer, args, max_steps)
    
    # Prepare for training
    integrated_model, optimizer, lr_scheduler = accelerator.prepare(
        integrated_model, optimizer, lr_scheduler
    )
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        integrated_model.train()
        
        for step in range(max_steps):
            # Generate dummy data for demonstration
            # In real implementation, you would load actual training data
            batch_size = args.train_batch_size
            seq_len = args.sequence_length
            
            # Create dummy music features
            music_features = torch.randn(batch_size, seq_len, 4800).to(device)
            
            # Create dummy reference image
            reference_image = torch.randn(batch_size, 3, args.resolution, args.resolution).to(device)
            
            # Create dummy target pose heatmap
            target_pose_heatmap = torch.randint(0, 134, (batch_size, seq_len)).to(device)
            
            # Create dummy target face embedding
            target_face_embed = torch.randn(batch_size, 512).to(device)
            target_face_embed = F.normalize(target_face_embed, p=2, dim=1)
            
            # Forward pass based on current stage
            if current_stage == "A":
                # Stage A: M2P pretrain
                outputs = integrated_model(
                    music_features=music_features,
                    target_pose_heatmap=target_pose_heatmap,
                    target_face_embed=target_face_embed
                )
                
                # Calculate losses
                pose_loss = pose_loss_fn(
                    outputs['pose_logits'],
                    target_pose_heatmap
                )
                
                face_loss = face_loss_fn(
                    outputs['face_embed'],
                    torch.arange(batch_size).to(device)  # Dummy labels
                )
                
                total_loss = (args.pose_loss_weight * pose_loss + 
                            args.face_loss_weight * face_loss)
                
            elif current_stage in ["B", "C"]:
                # Stage B/C: Image/Motion training
                # Create noisy latents
                dummy_latents = torch.randn(batch_size, 4, seq_len, 64, 64).to(device)
                noise = torch.randn_like(dummy_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                       (batch_size,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(dummy_latents, noise, timesteps)
                
                # Forward pass
                outputs = integrated_model(
                    music_features=music_features,
                    reference_image=reference_image,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    use_m2p_guidance=args.use_m2p_guidance,
                    guidance_scale=args.guidance_scale
                )
                
                # Calculate diffusion loss
                diffusion_loss = F.mse_loss(outputs['model_pred'], dummy_latents)
                
                # Calculate pose loss
                pose_loss = pose_loss_fn(
                    outputs['pose_heatmap'],
                    target_pose_heatmap
                )
                
                # Calculate identity loss (simplified)
                id_loss = F.mse_loss(outputs['face_embed'], target_face_embed)
                
                # Calculate temporal loss for Stage C
                temporal_loss = 0.0
                if current_stage == "C" and seq_len > 1:
                    # Simple temporal consistency loss
                    temp_diff = outputs['model_pred'][:, :, 1:] - outputs['model_pred'][:, :, :-1]
                    temporal_loss = torch.mean(torch.abs(temp_diff))
                
                # Combine losses
                total_loss = (args.diffusion_loss_weight * diffusion_loss +
                            args.pose_loss_weight * pose_loss +
                            args.id_loss_weight * id_loss +
                            args.temporal_loss_weight * temporal_loss)
            
            # Backward pass
            with accelerator.accumulate(integrated_model):
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(integrated_model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                log_dict = {
                    "total_loss": total_loss.detach().float(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": global_step,
                    "stage": current_stage
                }
                
                # Add stage-specific losses
                if current_stage == "A":
                    log_dict.update({
                        "pose_loss": pose_loss.detach().float(),
                        "face_loss": face_loss.detach().float()
                    })
                elif current_stage in ["B", "C"]:
                    log_dict.update({
                        "diffusion_loss": diffusion_loss.detach().float(),
                        "pose_loss": pose_loss.detach().float(),
                        "id_loss": id_loss.detach().float()
                    })
                    if current_stage == "C":
                        log_dict["temporal_loss"] = temporal_loss.detach().float()
                
                accelerator.log(log_dict, step=global_step)
            
            # Stage progression
            if args.auto_progress and global_step >= max_steps:
                if current_stage == "A":
                    # Progress to Stage B
                    current_stage = "B"
                    max_steps = args.stage_b_steps
                    integrated_model.set_training_stage(current_stage)
                    
                    # Create new optimizer for Stage B
                    optimizer = create_optimizer(integrated_model, args, current_stage)
                    lr_scheduler = create_scheduler(optimizer, args, max_steps)
                    
                    # Prepare for new stage
                    integrated_model, optimizer, lr_scheduler = accelerator.prepare(
                        integrated_model, optimizer, lr_scheduler
                    )
                    
                    # Reset progress bar
                    progress_bar.close()
                    progress_bar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
                    global_step = 0
                    
                    print(f"Progressed to Stage B: Image training")
                    
                elif current_stage == "B":
                    # Progress to Stage C
                    current_stage = "C"
                    max_steps = args.stage_c_steps
                    integrated_model.set_training_stage(current_stage)
                    
                    # Create new optimizer for Stage C
                    optimizer = create_optimizer(integrated_model, args, current_stage)
                    lr_scheduler = create_scheduler(optimizer, args, max_steps)
                    
                    # Prepare for new stage
                    integrated_model, optimizer, lr_scheduler = accelerator.prepare(
                        integrated_model, optimizer, lr_scheduler
                    )
                    
                    # Reset progress bar
                    progress_bar.close()
                    progress_bar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
                    global_step = 0
                    
                    print(f"Progressed to Stage C: Motion training")
                    
                elif current_stage == "C":
                    # Training complete
                    print("All stages completed!")
                    break
            
            # Validation
            if global_step % args.validation_steps == 0:
                integrated_model.eval()
                with torch.no_grad():
                    # Perform validation here
                    pass
                integrated_model.train()
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    stage_dir = f"{args.output_dir}/stage_{current_stage}"
                    os.makedirs(stage_dir, exist_ok=True)
                    accelerator.save_state(f"{stage_dir}/checkpoint-{global_step}")
            
            # Check if training is complete for current stage
            if global_step >= max_steps:
                break
        
        progress_bar.close()
        
        # Check if all stages are complete
        if current_stage == "C" and global_step >= max_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.save_state(f"{args.output_dir}/final")
    
    accelerator.end_training()

def main():
    """
    Main entry point for 2-stage training
    
    Args:
        Command line arguments for training configuration
        
    Returns:
        None
        
    Function: Parses arguments and launches the training pipeline
    """
    args = parse_args()
    training_function(args)

if __name__ == "__main__":
    main()
