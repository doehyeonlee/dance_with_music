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
    parser.add_argument("--guidance_input_path", type=str, default=None,
                        help="Path to external guidance input (e.g., pose videos, motion capture)")
    parser.add_argument("--guidance_weight_m2p", type=float, default=0.7,
                        help="Weight for M2P guidance when combining with external guidance")
    parser.add_argument("--guidance_weight_external", type=float, default=0.3,
                        help="Weight for external guidance when combining with M2P guidance")
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
    
    # Pretrained model paths
    parser.add_argument("--reference_unet_path", type=str, default=None,
                        help="Path to a pre-trained reference UNet checkpoint")
    parser.add_argument("--denoising_unet_path", type=str, default=None,
                        help="Path to a pre-trained denoising UNet checkpoint")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to a pre-trained VAE checkpoint")
    parser.add_argument("--clip_path", type=str, default=None,
                        help="Path to a pre-trained CLIP checkpoint")
    parser.add_argument("--arcface_path", type=str, default=None,
                        help="Path to a pre-trained ArcFace checkpoint")
    
    # Component-specific checkpoints
    parser.add_argument("--m2p_checkpoint_path", type=str, default=None,
                        help="Path to a pre-trained M2P encoder checkpoint")
    parser.add_argument("--guidance_encoder_path", type=str, default=None,
                        help="Path to pre-trained guidance encoder checkpoints")
    
    # Checkpoint management
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--save_checkpoint_format", type=str, default="pt", 
                        choices=["pt", "safetensors"],
                        help="Format for saving checkpoints")
    
    return parser.parse_args()

def create_m2p_encoder(args):
    """Create M2P encoder with dimension validation"""
    m2p_encoder = M2PEncoder(
        music_input_dim=4800,
        hidden_dim=1024,
        pose_channels=134,
        face_embed_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    # Print encoder architecture and dimensions
    print("=== M2P Encoder Architecture ===")
    print(f"Music Input: [B, T, 4800]")
    print(f"Hidden Features: [B, T, 1024]")
    print(f"Pose Output: [B, T, 134]")
    print(f"Face Output: [B, 512]")
    print(f"Total Parameters: {sum(p.numel() for p in m2p_encoder.parameters()):,}")
    
    return m2p_encoder

def validate_dimensions(m2p_encoder, batch_size=2, seq_len=24):
    """Validate M2P encoder dimensions with dummy data"""
    print("\n=== Dimension Validation ===")
    
    # Create dummy music features
    music_features = torch.randn(batch_size, seq_len, 4800)
    print(f"Input music features: {music_features.shape}")
    
    # Forward pass
    with torch.no_grad():
        pose_logits, face_embed = m2p_encoder(music_features)
    
    print(f"Output pose logits: {pose_logits.shape}")
    print(f"Output face embedding: {face_embed.shape}")
    
    # Validate dimensions
    assert pose_logits.shape == (batch_size, seq_len, 134), f"Pose shape mismatch: {pose_logits.shape}"
    assert face_embed.shape == (batch_size, 512), f"Face shape mismatch: {face_embed.shape}"
    
    print("✓ All dimensions validated successfully!")
    
    # Test pose heatmap conversion
    pose_heatmap = m2p_encoder.get_pose_heatmap(pose_logits)
    print(f"Pose heatmap: {pose_heatmap.shape}")
    
    # Test coordinate conversion
    pose_coords = m2p_encoder.get_pose_coordinates(pose_logits)
    print(f"Pose coordinates: {pose_coords.shape}")
    
    # Print detailed dimension flow
    print_dimension_flow(m2p_encoder, batch_size, seq_len)
    
    return True

def print_dimension_flow(m2p_encoder, batch_size=2, seq_len=24):
    """Print detailed dimension flow through M2P encoder"""
    print("\n" + "="*60)
    print("M2P ENCODER DIMENSION FLOW")
    print("="*60)
    
    # Create dummy input
    music_features = torch.randn(batch_size, seq_len, 4800)
    
    print(f"1. Input Music Features: {music_features.shape}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Music feature dimension: 4800")
    
    # Step through each layer
    with torch.no_grad():
        # Music projection
        x = m2p_encoder.music_proj(music_features)
        print(f"2. After Music Projection: {x.shape}")
        print(f"   - Hidden dimension: {x.size(-1)}")
        
        # Add positional encoding
        pos_encoding = m2p_encoder._get_pos_encoding(seq_len, m2p_encoder.hidden_dim)
        x_with_pos = x + pos_encoding.to(x.device)
        print(f"3. After Positional Encoding: {x_with_pos.shape}")
        
        # Transformer (simplified - just show input/output)
        print(f"4. Transformer Input: {x_with_pos.shape}")
        print(f"   - Transformer layers: {m2p_encoder.transformer.num_layers}")
        print(f"   - Attention heads: {m2p_encoder.transformer.layers[0].self_attn.num_heads}")
        
        # Pose head
        pose_logits = m2p_encoder.pose_head(x_with_pos)
        print(f"5. Pose Head Output: {pose_logits.shape}")
        print(f"   - Pose channels: {pose_logits.size(-1)}")
        
        # Face head
        face_features = x_with_pos.mean(dim=1)  # Mean pooling over time
        print(f"6. Face Features (after pooling): {face_features.shape}")
        
        face_embed = m2p_encoder.face_head(face_features)
        print(f"7. Face Head Output: {face_embed.shape}")
        print(f"   - Face embedding dimension: {face_embed.size(-1)}")
    
    print("\n" + "="*60)
    print("DIMENSION SUMMARY")
    print("="*60)
    print(f"Input:  [B={batch_size}, T={seq_len}, 4800] → Music features")
    print(f"Hidden: [B={batch_size}, T={seq_len}, 1024] → Transformer features")
    print(f"Pose:   [B={batch_size}, T={seq_len}, 134]  → Joint predictions")
    print(f"Face:   [B={batch_size}, 512]               → Identity embedding")
    print("="*60)

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
    """Create integrated CHAMP model with improved pretrained model loading"""
    
    # Create M2P encoder
    m2p_encoder = create_m2p_encoder(args)
    
    # Validate M2P encoder dimensions
    validate_dimensions(m2p_encoder, batch_size=2, seq_len=args.sequence_length)
    
    # Load pretrained UNet models with better checkpoint support
    print("=== Loading Pretrained Models ===")
    
    # 1. Reference UNet (2D) - for reference image processing
    try:
        if args.reference_unet_path:
            print(f"Loading Reference UNet from: {args.reference_unet_path}")
            reference_unet = UNet2DConditionModel.from_pretrained(
                args.reference_unet_path,
                use_safetensors=True
            )
        else:
            print(f"Loading Reference UNet from: {args.pretrained_model_name_or_path}")
            reference_unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                use_safetensors=True
            )
        print("✓ Reference UNet loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load Reference UNet: {e}")
        print("Creating new Reference UNet...")
        reference_unet = UNet2DConditionModel.from_config(
            UNet2DConditionModel.config_class.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet"
            )
        )
    
    # 2. Denoising UNet (3D) - for video generation
    try:
        if args.denoising_unet_path:
            print(f"Loading Denoising UNet from: {args.denoising_unet_path}")
            denoising_unet = UNet3DConditionModel.from_pretrained(
                args.denoising_unet_path,
                use_safetensors=True
            )
        else:
            print(f"Loading Denoising UNet from: {args.pretrained_model_name_or_path}")
            denoising_unet = UNet3DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet",
                use_safetensors=True
            )
        print("✓ Denoising UNet loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load Denoising UNet: {e}")
        print("Creating new Denoising UNet...")
        denoising_unet = UNet3DConditionModel.from_config(
            UNet3DConditionModel.config_class.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="unet"
            )
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
    
    # 3. VAE Encoder/Decoder
    try:
        if args.vae_path:
            print(f"Loading VAE from: {args.vae_path}")
            vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_path)
        else:
            print(f"Loading VAE from: {args.pretrained_model_name_or_path}")
            vae = AutoencoderKLTemporalDecoder.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae"
            )
        vae_encoder = vae.encoder
        vae_decoder = vae.decoder
        vae_encoder.requires_grad_(False)
        vae_decoder.requires_grad_(False)
        print("✓ VAE loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load VAE: {e}")
        print("Using dummy VAE encoders")
    
    # 4. CLIP Encoder
    try:
        if args.clip_path:
            print(f"Loading CLIP from: {args.clip_path}")
            clip_encoder = CLIPVisionModelWithProjection.from_pretrained(args.clip_path)
        else:
            print(f"Loading CLIP from: openai/clip-vit-large-patch14")
            clip_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        clip_encoder.requires_grad_(False)
        print("✓ CLIP loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load CLIP: {e}")
        print("Using dummy CLIP encoder")
    
    # 5. ArcFace Encoder (optional)
    if args.arcface_path:
        try:
            print(f"Loading ArcFace from: {args.arcface_path}")
            arcface_encoder = torch.load(args.arcface_path, map_location='cpu')
            arcface_encoder.requires_grad_(False)
            print("✓ ArcFace loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load ArcFace: {e}")
    
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

def save_checkpoint_with_metadata(
    model, 
    optimizer, 
    lr_scheduler, 
    epoch, 
    step, 
    stage, 
    output_dir, 
    metadata=None
):
    """Save checkpoint with comprehensive metadata"""
    
    # Create stage-specific directory
    stage_dir = f"{output_dir}/stage_{stage}"
    os.makedirs(stage_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'stage': stage,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'metadata': metadata or {}
    }
    
    # Save main checkpoint
    checkpoint_path = f"{stage_dir}/checkpoint-{step:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save latest checkpoint
    latest_path = f"{stage_dir}/latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save metadata separately for easy access
    metadata_path = f"{stage_dir}/metadata-{step:06d}.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'step': step,
            'stage': stage,
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.datetime.now().isoformat(),
            **metadata
        }, f, indent=2)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint_with_fallback(
    model, 
    checkpoint_path, 
    device, 
    strict=True
):
    """Load checkpoint with fallback options"""
    
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        if checkpoint_path.endswith('.pt'):
            # PyTorch checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                # Full checkpoint with metadata
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
                print(f"✓ Model loaded successfully (step {checkpoint.get('step', 'unknown')})")
                return checkpoint
            else:
                # Just model weights
                model.load_state_dict(checkpoint, strict=strict)
                print("✓ Model weights loaded successfully")
                return {'model_state_dict': checkpoint}
                
        elif checkpoint_path.endswith('.safetensors'):
            # SafeTensors checkpoint
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path, device=device)
            model.load_state_dict(state_dict, strict=strict)
            print("✓ SafeTensors checkpoint loaded successfully")
            return {'model_state_dict': state_dict}
            
        else:
            # Try to load as directory (diffusers format)
            try:
                model.load_state_dict(
                    torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=device),
                    strict=strict
                )
                print("✓ Diffusers checkpoint loaded successfully")
                return {}
            except:
                raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
                
    except Exception as e:
        print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
        if strict:
            raise e
        return {}

def load_pretrained_components(model, args, device):
    """Load pretrained components for specific stages"""
    
    print("\n=== Loading Pretrained Components ===")
    
    # Stage A: M2P Encoder
    if args.m2p_checkpoint_path:
        try:
            load_checkpoint_with_fallback(
                model.m2p_encoder, 
                args.m2p_checkpoint_path, 
                device, 
                strict=False
            )
            print("✓ M2P Encoder loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load M2P checkpoint: {e}")
    
    # Stage B: Reference UNet
    if args.reference_unet_path and hasattr(model, 'reference_unet'):
        try:
            load_checkpoint_with_fallback(
                model.reference_unet, 
                args.reference_unet_path, 
                device, 
                strict=False
            )
            print("✓ Reference UNet loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load Reference UNet checkpoint: {e}")
    
    # Stage C: Denoising UNet
    if args.denoising_unet_path and hasattr(model, 'denoising_unet'):
        try:
            load_checkpoint_with_fallback(
                model.denoising_unet, 
                args.denoising_unet_path, 
                device, 
                strict=False
            )
            print("✓ Denoising UNet loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load Denoising UNet checkpoint: {e}")
    
    # Guidance Encoders
    if args.guidance_encoder_path:
        try:
            for name, encoder in model.guidance_encoder_group.items():
                encoder_path = f"{args.guidance_encoder_path}/{name}"
                if os.path.exists(encoder_path):
                    load_checkpoint_with_fallback(
                        encoder, 
                        encoder_path, 
                        device, 
                        strict=False
                    )
                    print(f"✓ Guidance Encoder '{name}' loaded from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load guidance encoders: {e}")

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
    
    # Load pretrained components for specific stages
    load_pretrained_components(integrated_model, args, device)
    
    # Create loss functions
    pose_loss_fn, face_loss_fn = create_loss_functions(args)
    
    # Create noise scheduler
    noise_scheduler = create_noise_scheduler()
    
    # Set initial training stage
    current_stage = args.training_stage
    integrated_model.set_training_stage(current_stage)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        try:
            checkpoint = load_checkpoint_with_fallback(
                integrated_model, 
                args.resume_from_checkpoint, 
                device, 
                strict=False
            )
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'step' in checkpoint:
                global_step = checkpoint['step']
            print(f"✓ Resumed training from epoch {start_epoch}, step {global_step}")
        except Exception as e:
            print(f"Warning: Could not resume from checkpoint: {e}")
    
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
    progress_bar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(start_epoch, args.num_train_epochs):
        integrated_model.train()
        
        for step in range(max_steps):
            # Generate dummy data for demonstration
            # In real implementation, you would load actual training data
            batch_size = args.train_batch_size
            seq_len = args.sequence_length
            
            # Create dummy music features: [B, T, 4800]
            music_features = torch.randn(batch_size, seq_len, 4800).to(device)
            
            # Create dummy reference image: [B, 3, H, W]
            reference_image = torch.randn(batch_size, 3, args.resolution, args.resolution).to(device)
            
            # Create dummy target pose data
            if current_stage == "A":
                # Stage A: Use class labels for pose [B, T]
                target_pose_labels = torch.randint(0, 134, (batch_size, seq_len)).to(device)
            else:
                # Stage B/C: Use pose heatmaps [B, T, 134]
                target_pose_heatmap = torch.randn(batch_size, seq_len, 134).to(device)
            
            # Create dummy target face embedding: [B, 512]
            target_face_embed = torch.randn(batch_size, 512).to(device)
            target_face_embed = F.normalize(target_face_embed, p=2, dim=1)
            
            # Load external guidance if specified
            external_guidance = None
            if args.guidance_input_path and os.path.exists(args.guidance_input_path):
                try:
                    # Load external guidance (e.g., pose videos, motion capture)
                    # This is a placeholder - implement based on your guidance format
                    external_guidance = torch.randn(batch_size, seq_len, 134).to(device)
                    print(f"✓ External guidance loaded: {external_guidance.shape}")
                except Exception as e:
                    print(f"Warning: Could not load external guidance: {e}")
                    external_guidance = None
            
            # Forward pass based on current stage
            if current_stage == "A":
                # Stage A: M2P pretrain
                print(f"Stage A - Input music: {music_features.shape}")
                
                outputs = integrated_model(
                    music_features=music_features,
                    target_pose_heatmap=target_pose_labels,
                    target_face_embed=target_face_embed
                )
                
                print(f"Stage A - Output pose: {outputs['pose_logits'].shape}")
                print(f"Stage A - Output face: {outputs['face_embed'].shape}")
                
                # Calculate losses
                pose_loss = pose_loss_fn(
                    outputs['pose_logits'],  # [B, T, 134]
                    target_pose_labels       # [B, T]
                )
                
                face_loss = face_loss_fn(
                    outputs['face_embed'],   # [B, T, 512] - now per-frame
                    torch.arange(batch_size).to(device)  # Dummy labels
                )
                
                total_loss = (args.pose_loss_weight * pose_loss + 
                            args.face_loss_weight * face_loss)
                
            elif current_stage in ["B", "C"]:
                # Stage B/C: Image/Motion training
                print(f"Stage {current_stage} - Input music: {music_features.shape}")
                print(f"Stage {current_stage} - Input reference: {reference_image.shape}")
                if external_guidance is not None:
                    print(f"Stage {current_stage} - External guidance: {external_guidance.shape}")
                
                # Create noisy latents: [B, 4, T, H, W]
                dummy_latents = torch.randn(batch_size, 4, seq_len, 64, 64).to(device)
                noise = torch.randn_like(dummy_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                       (batch_size,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(dummy_latents, noise, timesteps)
                
                # Forward pass with guidance input
                outputs = integrated_model(
                    music_features=music_features,
                    reference_image=reference_image,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    guidance_input=external_guidance,  # Pass external guidance
                    use_m2p_guidance=args.use_m2p_guidance,
                    guidance_scale=args.guidance_scale
                )
                
                print(f"Stage {current_stage} - Output model_pred: {outputs['model_pred'].shape}")
                print(f"Stage {current_stage} - Output pose: {outputs['pose_heatmap'].shape}")
                print(f"Stage {current_stage} - Output face: {outputs['face_embed'].shape}")
                if 'guidance_cond' in outputs:
                    print(f"Stage {current_stage} - Combined guidance: {outputs['guidance_cond'].shape}")
                
                # Calculate diffusion loss
                diffusion_loss = F.mse_loss(outputs['model_pred'], dummy_latents)
                
                # Calculate pose loss
                pose_loss = pose_loss_fn(
                    outputs['pose_heatmap'],  # [B, T, 134]
                    target_pose_heatmap       # [B, T, 134]
                )
                
                # Calculate identity loss (simplified)
                id_loss = F.mse_loss(outputs['face_embed'], target_face_embed.unsqueeze(1).expand(-1, seq_len, -1))
                
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
            
            # Save checkpoint with improved function
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    metadata = {
                        'stage': current_stage,
                        'total_loss': total_loss.detach().float().item(),
                        'learning_rate': lr_scheduler.get_last_lr()[0],
                        'model_parameters': sum(p.numel() for p in integrated_model.parameters()),
                        'trainable_parameters': sum(p.numel() for p in integrated_model.parameters() if p.requires_grad)
                    }
                    
                    save_checkpoint_with_metadata(
                        integrated_model, 
                        optimizer, 
                        lr_scheduler, 
                        epoch, 
                        global_step, 
                        current_stage, 
                        args.output_dir, 
                        metadata
                    )
            
            # Check if training is complete for current stage
            if global_step >= max_steps:
                break
        
        progress_bar.close()
        
        # Check if all stages are complete
        if current_stage == "C" and global_step >= max_steps:
            break
    
    # Save final model
    if accelerator.is_main_process:
        final_metadata = {
            'final_stage': current_stage,
            'total_epochs': epoch + 1,
            'total_steps': global_step,
            'training_completed': True
        }
        
        save_checkpoint_with_metadata(
            integrated_model, 
            optimizer, 
            lr_scheduler, 
            epoch, 
            global_step, 
            'final', 
            args.output_dir, 
            final_metadata
        )
    
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
