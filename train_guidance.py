import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention_processor import XFormersAttnProcessor

from models.champ_model import ChampModel
from models.guidance_encoder import GuidanceMusicEncoder
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.reference_control import ReferenceControlWriter, ReferenceControlReader
from datasets.guidance_dataset import create_guidance_dataloader

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
import warnings
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CHAMP model with guidance")
    
    # Model configuration
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--revision", type=str, default=None, help="Revision of pretrained model identifier")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model")
    
    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./champ_guidance_output",
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for training")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--scale_lr", action="store_true", help="Scale the learning rate by the number of GPUs")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type to use")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler")
    
    # Guidance configuration
    parser.add_argument("--guidance_types", type=str, nargs="+", default=["music", "pose"],
                        help="Types of guidance to use during training")
    parser.add_argument("--guidance_input_channels", type=int, nargs="+", default=[1, 134],
                        help="Input channels for each guidance type")
    parser.add_argument("--guidance_embedding_channels", type=int, default=1280,
                        help="Output embedding channels for guidance encoders")
    
    # Dataset configuration
    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data")
    parser.add_argument("--validation_data_dir", type=str, default=None, help="A folder containing the validation data")
    parser.add_argument("--train_data_format", type=str, default="video", choices=["video", "image"],
                        help="Format of training data")
    parser.add_argument("--sequence_length", type=int, default=16, help="Number of frames in sequence")
    
    # Logging and saving
    parser.add_argument("--logging_dir", type=str, default="logs", help="Directory for storing logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision")
    parser.add_argument("--report_to", type=str, default=None, choices=["tensorboard", "wandb"],
                        help="The integration to report the results and logs to")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help="Save a checkpoint of the training state every X updates")
    parser.add_argument("--validation_steps", type=int, default=100,
                        help="Run validation every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    
    return parser.parse_args()

def create_guidance_encoders(guidance_types, guidance_input_channels, guidance_embedding_channels):
    """Create guidance encoders for different guidance types"""
    guidance_encoder_group = {}
    
    for guidance_type, input_channels in zip(guidance_types, guidance_input_channels):
        if guidance_type == "music":
            # Music guidance encoder
            guidance_encoder = GuidanceMusicEncoder(
                guidance_embedding_channels=guidance_embedding_channels,
                guidance_input_channels=input_channels,
                block_out_channels=(16, 32, 96, 256),
                attention_num_heads=12
            )
        elif guidance_type == "pose":
            # Pose guidance encoder (you can customize this)
            guidance_encoder = GuidanceMusicEncoder(
                guidance_embedding_channels=guidance_embedding_channels,
                guidance_input_channels=input_channels,
                block_out_channels=(16, 32, 96, 256),
                attention_num_heads=12
            )
        else:
            # Default guidance encoder
            guidance_encoder = GuidanceMusicEncoder(
                guidance_embedding_channels=guidance_embedding_channels,
                guidance_input_channels=input_channels,
                block_out_channels=(16, 32, 96, 256),
                attention_num_heads=12
            )
        
        guidance_encoder_group[guidance_type] = guidance_encoder
    
    return guidance_encoder_group

def create_champ_model(pretrained_model_path, guidance_encoder_group, device):
    """Create CHAMP model with guidance encoders"""
    
    # Load pretrained UNet models
    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        use_safetensors=True
    )
    
    denoising_unet = UNet3DConditionModel.from_pretrained(
        pretrained_model_path,
        subfolder="unet",
        use_safetensors=True
    )
    
    # Create reference control components
    reference_control_writer = ReferenceControlWriter()
    reference_control_reader = ReferenceControlReader()
    
    # Create CHAMP model
    champ_model = ChampModel(
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        reference_control_writer=reference_control_writer,
        reference_control_reader=reference_control_reader,
        guidance_encoder_group=guidance_encoder_group
    )
    
    return champ_model.to(device)

def create_noise_scheduler():
    """Create noise scheduler for training"""
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )
    return noise_scheduler

def create_vae():
    """Create VAE for encoding/decoding"""
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    )
    return vae

def create_clip_processor():
    """Create CLIP processor for text/image encoding"""
    clip_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    return clip_processor

def create_clip_model():
    """Create CLIP model for image encoding"""
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    return clip_model

def training_function(args):
    """Main training function"""
    
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
    
    # Create guidance encoders
    guidance_encoder_group = create_guidance_encoders(
        args.guidance_types,
        args.guidance_input_channels,
        args.guidance_embedding_channels
    )
    
    # Create CHAMP model
    device = accelerator.device
    champ_model = create_champ_model(
        args.pretrained_model_name_or_path,
        guidance_encoder_group,
        device
    )
    
    # Create noise scheduler
    noise_scheduler = create_noise_scheduler()
    
    # Create VAE
    vae = create_vae()
    vae.requires_grad_(False)
    
    # Create CLIP components
    clip_processor = create_clip_processor()
    clip_model = create_clip_model()
    clip_model.requires_grad_(False)
    
    # Create dataloader
    if args.train_data_dir:
        train_dataloader = create_guidance_dataloader(
            data_root=args.train_data_dir,
            guidance_types=args.guidance_types,
            guidance_input_channels=args.guidance_input_channels,
            batch_size=args.train_batch_size,
            resolution=args.resolution,
            sequence_length=args.sequence_length,
            num_workers=4,
            shuffle=True
        )
    else:
        # Create dummy dataloader for testing
        train_dataloader = None
    
    # Calculate training steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader) if train_dataloader else 1000
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        champ_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Prepare for training
    if train_dataloader:
        champ_model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            champ_model, optimizer, lr_scheduler, train_dataloader
        )
    else:
        champ_model, optimizer, lr_scheduler = accelerator.prepare(
            champ_model, optimizer, lr_scheduler
        )
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        champ_model.train()
        
        if train_dataloader:
            for step, batch in enumerate(train_dataloader):
                # Extract data from batch
                video_tensor = batch['video']  # [B, T, C, H, W]
                guidance_condition = batch['guidance_condition']  # [B, T, total_C, H, W]
                reference_image = batch['reference_image']  # [B, C, H, W]
                
                # Encode reference image with CLIP
                with torch.no_grad():
                    clip_inputs = clip_processor(images=reference_image, return_tensors="pt")
                    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
                    clip_image_embeds = clip_model(**clip_inputs).image_embeds
                
                # Encode video with VAE
                with torch.no_grad():
                    # Reshape video tensor for VAE: [B, T, C, H, W] -> [B*T, C, H, W]
                    video_flat = video_tensor.view(-1, video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4])
                    video_latents = vae.encode(video_flat).latent_dist.sample()
                    video_latents = video_latents * vae.config.scaling_factor
                    # Reshape back: [B*T, C, H, W] -> [B, T, C, H, W]
                    video_latents = video_latents.view(video_tensor.shape[0], video_tensor.shape[1], -1, video_latents.shape[2], video_latents.shape[3])
                
                # Encode reference image with VAE
                with torch.no_grad():
                    ref_latents = vae.encode(reference_image).latent_dist.sample()
                    ref_latents = ref_latents * vae.config.scaling_factor
                
                # Add noise to video latents
                noise = torch.randn_like(video_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (video_latents.shape[0],), device=device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(video_latents, noise, timesteps)
                
                # Forward pass through CHAMP model
                with accelerator.accumulate(champ_model):
                    model_pred = champ_model(
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        ref_image_latents=ref_latents,
                        clip_image_embeds=clip_image_embeds,
                        multi_guidance_cond=guidance_condition,
                        uncond_fwd=False
                    )
                    
                    # Calculate loss
                    target = video_latents
                    loss = F.mse_loss(model_pred, target)
                    
                    # Backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(champ_model.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    accelerator.log({
                        "train_loss": loss.detach().float(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step
                    }, step=global_step)
                
                # Validation
                if global_step % args.validation_steps == 0:
                    champ_model.eval()
                    with torch.no_grad():
                        # Perform validation here
                        pass
                    champ_model.train()
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        accelerator.save_state(f"{args.output_dir}/checkpoint-{global_step}")
                
                # Check if training is complete
                if global_step >= args.max_train_steps:
                    break
        else:
            # Dummy training loop for testing
            for step in range(args.max_train_steps):
                # Generate dummy data
                batch_size = args.train_batch_size
                noisy_latents = torch.randn(batch_size, 4, args.sequence_length, 64, 64).to(device)
                timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
                ref_image_latents = torch.randn(batch_size, 4, 1, 64, 64).to(device)
                clip_image_embeds = torch.randn(batch_size, 77, 1280).to(device)
                
                # Create guidance conditions
                total_channels = sum(args.guidance_input_channels)
                guidance_condition = torch.randn(batch_size, args.sequence_length, total_channels, 64, 64).to(device)
                
                # Forward pass
                with accelerator.accumulate(champ_model):
                    model_pred = champ_model(
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        ref_image_latents=ref_image_latents,
                        clip_image_embeds=clip_image_embeds,
                        multi_guidance_cond=guidance_condition,
                        uncond_fwd=False
                    )
                    
                    # Calculate loss
                    target = torch.randn_like(model_pred)
                    loss = F.mse_loss(model_pred, target)
                    
                    # Backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(champ_model.parameters(), 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % 10 == 0:
                    accelerator.log({
                        "train_loss": loss.detach().float(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step
                    }, step=global_step)
                
                # Check if training is complete
                if global_step >= args.max_train_steps:
                    break
        
        progress_bar.close()
    
    # Save final model
    if accelerator.is_main_process:
        accelerator.save_state(f"{args.output_dir}/final")
    
    accelerator.end_training()

def main():
    args = parse_args()
    training_function(args)

if __name__ == "__main__":
    main()
