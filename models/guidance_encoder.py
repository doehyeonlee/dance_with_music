from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from models.motion_module import zero_module
from models.resnet import InflatedConv3d, InflatedGroupNorm
from models.attention import TemporalBasicTransformerBlock
from models.transformer_3d import Transformer3DModel


class GuidanceMusicEncoder(ModelMixin):
    def __init__(
        self,
        guidance_embedding_channels: int,
        guidance_input_channels: int = 1,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        attention_num_heads: int = 12,
    ):
        super().__init__()
        self.guidance_input_channels = guidance_input_channels

        self.transform_in=nn.Linear(in_features=4800,out_features=4096)
        # 4800 = in features: 

        self.conv_in = InflatedConv3d(
            guidance_input_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]

            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.attentions.append(
                Transformer3DModel(
                    attention_num_heads,
                    channel_in // attention_num_heads,
                    channel_in,
                    norm_num_groups=1,
                    unet_use_cross_frame_attention=False,
                    unet_use_temporal_attention=False,
                )
            )

            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=1
                )
            )
            self.attentions.append(
                Transformer3DModel(
                    attention_num_heads,
                    channel_out // attention_num_heads,
                    channel_out,
                    norm_num_groups=32,
                    unet_use_cross_frame_attention=False,
                    unet_use_temporal_attention=False,
                )
            )
        
        self.conv_pose_out = InflatedConv3d(
                guidance_embedding_channels,
                134,
                kernel_size=7,
                padding=3,
        )
        
        self.conv_out = InflatedConv3d(
                block_out_channels[-1],
                guidance_embedding_channels,
                kernel_size=3,
                padding=1,
        )

    def forward(self, condition):
        b,c,l=condition.shape
        condition = self.transform_in(condition)
        embedding = self.conv_in(condition.view(b,1,-1,64,64)) #torch.Size([1, 3, 2, 512, 512]) --> torch.Size([1, 16, 2, 512, 512])
        embedding = F.silu(embedding)

        for iter_num, (block,attn) in enumerate(zip(self.blocks,self.attentions)):
            #print('step:', iter_num)
            embedding = block(embedding)
            embedding = F.silu(embedding)
            embedding = attn(embedding)[-1]

        body_embedding = self.conv_out(embedding)  
        pose_prediction = self.conv_pose_out(body_embedding)
        pose_shape = pose_prediction.shape
        heatmap = pose_prediction.view(pose_shape[0], pose_shape[1], pose_shape[2],-1)
        heatmap = F.softmax(heatmap/0.1,dim=-1)
        heatmap = heatmap.view(*pose_shape)

        return body_embedding, heatmap
