import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel


class ChampModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        reference_control_reader,
        guidance_encoder_group,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet

        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

        self.guidance_types = []
        self.guidance_input_channels = []

        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        multi_guidance_cond,
        uncond_fwd: bool = False,
    ):
        """
        Forward pass of CHAMP Model
        
        Args:
            noisy_latents: Input noisy latents [B, C, T, H, W]
            timesteps: Diffusion timesteps [B]
            ref_image_latents: Reference image latents [B, C, H, W]
            clip_image_embeds: CLIP image embeddings [B, C, H, W]
            multi_guidance_cond: Multiple guidance conditions [B, total_channels]
            uncond_fwd: Whether to run unconditional forward pass
            
        Returns:
            model_pred: Predicted denoised sample [B, C, T, H, W]
            
        Function: Processes multiple guidance types and generates video with reference control
        """
        guidance_cond_group = torch.split(
            multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        guidance_fea_lst = []
        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=guidance_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred
