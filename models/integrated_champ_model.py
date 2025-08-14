import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .champ_model import ChampModel
from .m2p_encoder import M2PEncoder
from .unet_2d_condition import UNet2DConditionModel
from .unet_3d import UNet3DConditionModel
from .reference_control import ReferenceControlWriter, ReferenceControlReader

class IntegratedChampModel(nn.Module):
    """
    Integrated CHAMP Model combining M2PEncoder with CHAMP
    Supports 2-stage training with different freeze strategies
    """
    
    def __init__(
        self,
        m2p_encoder: M2PEncoder,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        reference_control_writer: ReferenceControlWriter,
        reference_control_reader: ReferenceControlReader,
        guidance_encoder_group: Dict,
        vae_encoder=None,
        vae_decoder=None,
        clip_encoder=None,
        arcface_encoder=None
    ):
        super().__init__()
        
        # Core components
        self.m2p_encoder = m2p_encoder
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.guidance_encoder_group = guidance_encoder_group
        
        # External encoders (will be frozen during training)
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.clip_encoder = clip_encoder
        self.arcface_encoder = arcface_encoder
        
        # Training stage control
        self.current_stage = "A"  # A: M2P pretrain, B: Image stage, C: Motion stage
        
        # Initialize freeze states
        self._setup_freeze_strategy()
    
    def _setup_freeze_strategy(self):
        """Setup initial freeze strategy for Stage A"""
        self._freeze_for_stage_a()
    
    def _freeze_for_stage_a(self):
        """Freeze strategy for Stage A: M2P pretrain only"""
        # Freeze all except M2P encoder
        for name, param in self.named_parameters():
            if 'm2p_encoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Freeze external encoders
        if self.vae_encoder is not None:
            for param in self.vae_encoder.parameters():
                param.requires_grad = False
        
        if self.vae_decoder is not None:
            for param in self.vae_decoder.parameters():
                param.requires_grad = False
        
        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
        
        if self.arcface_encoder is not None:
            for param in self.arcface_encoder.parameters():
                param.requires_grad = False
    
    def _freeze_for_stage_b(self):
        """Freeze strategy for Stage B: Image stage"""
        # Freeze VAE and CLIP encoders
        if self.vae_encoder is not None:
            for param in self.vae_encoder.parameters():
                param.requires_grad = False
        
        if self.vae_decoder is not None:
            for param in self.vae_decoder.parameters():
                param.requires_grad = False
        
        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
        
        if self.arcface_encoder is not None:
            for param in self.arcface_encoder.parameters():
                param.requires_grad = False
        
        # Train guidance paths, UNet, ReferenceNet
        for name, param in self.named_parameters():
            if any(x in name for x in ['guidance_encoder', 'denoising_unet', 'reference_unet']):
                param.requires_grad = True
            elif 'm2p_encoder' in name:
                # M2P encoder: start frozen, then low LR
                param.requires_grad = False
        
        # M2P encoder: very low LR unfreezing
        for name, param in self.m2p_encoder.named_parameters():
            param.requires_grad = True
    
    def _freeze_for_stage_c(self):
        """Freeze strategy for Stage C: Motion/Temporal stage"""
        # Freeze everything except temporal modules
        for name, param in self.named_parameters():
            if any(x in name for x in ['motion_module', 'temporal', 'time_embedding']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Keep external encoders frozen
        if self.vae_encoder is not None:
            for param in self.vae_encoder.parameters():
                param.requires_grad = False
        
        if self.vae_decoder is not None:
            for param in self.vae_decoder.parameters():
                param.requires_grad = False
        
        if self.clip_encoder is not None:
            for param in self.clip_encoder.parameters():
                param.requires_grad = False
        
        if self.arcface_encoder is not None:
            for param in self.arcface_encoder.parameters():
                param.requires_grad = False
    
    def set_training_stage(self, stage: str):
        """Set training stage and apply corresponding freeze strategy"""
        self.current_stage = stage
        
        if stage == "A":
            self._freeze_for_stage_a()
        elif stage == "B":
            self._freeze_for_stage_b()
        elif stage == "C":
            self._freeze_for_stage_c()
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        
        print(f"Switched to training stage {stage}")
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        """Print which parameters are trainable"""
        trainable_params = []
        frozen_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        print(f"Trainable parameters ({len(trainable_params)}):")
        for name in trainable_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(trainable_params) > 10:
            print(f"  ... and {len(trainable_params) - 10} more")
        
        print(f"Frozen parameters ({len(frozen_params)}):")
        for name in frozen_params[:10]:  # Show first 10
            print(f"  - {name}")
        if len(frozen_params) > 10:
            print(f"  ... and {len(frozen_params) - 10} more")
    
    def forward_stage_a(
        self,
        music_features: torch.Tensor,
        target_pose_heatmap: Optional[torch.Tensor] = None,
        target_face_embed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage A: M2P pretrain
        """
        # M2P encoder forward pass
        pose_logits, face_embed = self.m2p_encoder(music_features)
        
        # Convert to heatmap
        pose_heatmap = self.m2p_encoder.get_pose_heatmap(pose_logits)
        
        return {
            'pose_logits': pose_logits,
            'pose_heatmap': pose_heatmap,
            'face_embed': face_embed,
            'target_pose_heatmap': target_pose_heatmap,
            'target_face_embed': target_face_embed
        }
    
    def forward_stage_b(
        self,
        music_features: torch.Tensor,
        reference_image: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        use_m2p_guidance: bool = True,
        guidance_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage B: Image stage
        """
        # Get M2P predictions
        pose_logits, face_embed = self.m2p_encoder(music_features)
        pose_heatmap = self.m2p_encoder.get_pose_heatmap(pose_logits)
        
        # Encode reference image with CLIP
        if self.clip_encoder is not None:
            clip_embeds = self.clip_encoder(reference_image)
        else:
            # Use face embedding as fallback
            clip_embeds = face_embed.unsqueeze(1)  # [B, 1, embed_dim]
        
        # Encode reference image with VAE
        if self.vae_encoder is not None:
            ref_latents = self.vae_encoder(reference_image).latent_dist.sample()
            ref_latents = ref_latents * self.vae_encoder.config.scaling_factor
        else:
            # Dummy reference latents
            ref_latents = torch.randn_like(noisy_latents[:, :1])
        
        # Create guidance condition
        if use_m2p_guidance:
            # Use M2P pose heatmap as guidance
            guidance_cond = pose_heatmap.unsqueeze(1) * guidance_scale  # [B, 1, T, pose_channels]
        else:
            # Use random guidance
            guidance_cond = torch.randn_like(pose_heatmap.unsqueeze(1))
        
        # CHAMP forward pass
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=guidance_cond,
            encoder_hidden_states=clip_embeds,
        ).sample
        
        return {
            'model_pred': model_pred,
            'pose_heatmap': pose_heatmap,
            'face_embed': face_embed,
            'ref_latents': ref_latents,
            'clip_embeds': clip_embeds
        }
    
    def forward_stage_c(
        self,
        music_features: torch.Tensor,
        reference_image: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage C: Motion/Temporal stage
        """
        # Similar to Stage B but with temporal focus
        return self.forward_stage_b(
            music_features=music_features,
            reference_image=reference_image,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            use_m2p_guidance=True,
            guidance_scale=guidance_scale
        )
    
    def forward(
        self,
        music_features: torch.Tensor,
        reference_image: Optional[torch.Tensor] = None,
        noisy_latents: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        target_pose_heatmap: Optional[torch.Tensor] = None,
        target_face_embed: Optional[torch.Tensor] = None,
        use_m2p_guidance: bool = True,
        guidance_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass that routes to appropriate stage
        
        Args:
            music_features: Input music features [B, T, 4800]
            reference_image: Reference face image [B, 3, H, W]
            noisy_latents: Noisy latents for diffusion [B, C, T, H, W]
            timesteps: Diffusion timesteps [B]
            target_pose_heatmap: Target pose heatmaps [B, T, 134, H, W]
            target_face_embed: Target face embeddings [B, 512]
            use_m2p_guidance: Whether to use M2P guidance
            guidance_scale: Guidance strength multiplier
            
        Returns:
            output_dict: Stage-specific output dictionary
            
        Function: Routes input to appropriate training stage (A/B/C) based on current_stage
        """
        if self.current_stage == "A":
            return self.forward_stage_a(
                music_features=music_features,
                target_pose_heatmap=target_pose_heatmap,
                target_face_embed=target_face_embed
            )
        elif self.current_stage == "B":
            return self.forward_stage_b(
                music_features=music_features,
                reference_image=reference_image,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                use_m2p_guidance=use_m2p_guidance,
                guidance_scale=guidance_scale
            )
        elif self.current_stage == "C":
            return self.forward_stage_c(
                music_features=music_features,
                reference_image=reference_image,
                noisy_latents=noisy_latents,
                timesteps=timesteps,
                guidance_scale=guidance_scale
            )
        else:
            raise ValueError(f"Unknown training stage: {self.current_stage}")
    
    def get_trainable_parameters(self, stage: str = None) -> List[nn.Parameter]:
        """Get trainable parameters for current stage or specified stage"""
        if stage is None:
            stage = self.current_stage
        
        if stage == "A":
            return list(self.m2p_encoder.parameters())
        elif stage == "B":
            params = []
            for name, param in self.named_parameters():
                if any(x in name for x in ['guidance_encoder', 'denoising_unet', 'reference_unet', 'm2p_encoder']):
                    params.append(param)
            return params
        elif stage == "C":
            params = []
            for name, param in self.named_parameters():
                if any(x in name for x in ['motion_module', 'temporal', 'time_embedding']):
                    params.append(param)
            return params
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    
    def get_frozen_parameters(self, stage: str = None) -> List[nn.Parameter]:
        """Get frozen parameters for current stage or specified stage"""
        if stage is None:
            stage = self.current_stage
        
        all_params = list(self.named_parameters())
        trainable_params = set(self.get_trainable_parameters(stage))
        
        frozen_params = []
        for name, param in all_params:
            if param not in trainable_params:
                frozen_params.append(param)
        
        return frozen_params
