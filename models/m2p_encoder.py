import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class M2PEncoder(nn.Module):
    """
    Music-to-Pose Encoder with Face Embedding
    Generates pose heatmaps and face embeddings from music features
    """
    
    def __init__(
        self,
        music_input_dim: int = 4800,
        hidden_dim: int = 1024,
        pose_channels: int = 134,
        face_embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.music_input_dim = music_input_dim
        self.hidden_dim = hidden_dim
        self.pose_channels = pose_channels
        self.face_embed_dim = face_embed_dim
        
        # Music feature processing
        self.music_proj = nn.Sequential(
            nn.Linear(music_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pose prediction head
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pose_channels)
        )
        
        # Face embedding head
        self.face_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, face_embed_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        music_features: torch.Tensor,
        return_face_embed: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of M2P Encoder
        
        Args:
            music_features: Input music features [B, T, 4800]
            return_face_embed: Whether to return face embedding
            
        Returns:
            pose_logits: Pose prediction logits [B, T, pose_channels]
            face_embed: Face embedding [B, 512] or None
            
        Function: Converts music features to pose predictions and face embeddings
        """
        """
        Args:
            music_features: [B, T, music_input_dim] - Music features over time
            return_face_embed: Whether to return face embedding
            
        Returns:
            pose_logits: [B, T, pose_channels] - Pose prediction logits
            face_embed: [B, face_embed_dim] - Face embedding (if return_face_embed=True)
        """
        batch_size, seq_len, _ = music_features.shape
        
        # Project music features
        x = self.music_proj(music_features)  # [B, T, hidden_dim]
        
        # Add positional encoding
        pos_encoding = self._get_pos_encoding(seq_len, self.hidden_dim).to(x.device)
        x = x + pos_encoding
        
        # Apply transformer
        x = self.transformer(x)  # [B, T, hidden_dim]
        
        # Pose prediction
        pose_logits = self.pose_head(x)  # [B, T, pose_channels]
        
        # Face embedding (use mean pooling over time)
        if return_face_embed:
            face_embed = self.face_head(x.mean(dim=1))  # [B, face_embed_dim]
            # L2 normalization for ArcFace compatibility
            face_embed = F.normalize(face_embed, p=2, dim=1)
            return pose_logits, face_embed
        else:
            return pose_logits, None
    
    def _get_pos_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Generate positional encoding"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def get_pose_heatmap(self, pose_logits: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Convert pose logits to heatmap using softmax"""
        # Apply temperature scaling and softmax
        heatmap = F.softmax(pose_logits / temperature, dim=-1)
        return heatmap
    
    def get_pose_coordinates(self, pose_logits: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Convert pose logits to coordinates using soft-argmax"""
        heatmap = self.get_pose_heatmap(pose_logits, temperature)
        
        # Soft-argmax: weighted average of positions
        batch_size, seq_len, num_joints = heatmap.shape
        heatmap = heatmap.view(batch_size, seq_len, num_joints, -1)  # Reshape for spatial dimensions
        
        # Create coordinate grid
        h, w = int(math.sqrt(num_joints)), int(math.sqrt(num_joints))
        if h * w != num_joints:
            # If not perfect square, pad or truncate
            h = w = int(math.ceil(math.sqrt(num_joints)))
            heatmap = heatmap[:, :, :num_joints, :]
        
        # Generate coordinate grid
        y_coords = torch.arange(h, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, h, 1)
        x_coords = torch.arange(w, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, 1, w)
        
        # Weighted average
        y_coord = (heatmap * y_coords).sum(dim=(3, 4))  # [B, T, num_joints]
        x_coord = (heatmap * x_coords).sum(dim=(3, 4))  # [B, T, num_joints]
        
        coordinates = torch.stack([x_coord, y_coord], dim=-1)  # [B, T, num_joints, 2]
        return coordinates


class ArcFaceLoss(nn.Module):
    """ArcFace loss for face embedding training"""
    
    def __init__(self, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        ArcFace Loss calculation
        
        Args:
            embeddings: Face embeddings [B, 512]
            labels: Identity labels [B]
            
        Returns:
            loss: ArcFace loss value
            
        Function: Applies ArcFace margin to improve face recognition quality
        """
        """
        Args:
            embeddings: [B, embed_dim] - Normalized face embeddings
            labels: [B] - Class labels
            
        Returns:
            loss: ArcFace loss value
        """
        # Compute cosine similarity
        cos_theta = F.linear(embeddings, embeddings)  # [B, B]
        
        # Apply margin
        cos_theta_m = cos_theta - self.margin
        
        # Create one-hot labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        # Apply margin only to positive pairs
        cos_theta_m = torch.where(one_hot.bool(), cos_theta_m, cos_theta)
        
        # Scale and compute loss
        logits = cos_theta_m * self.scale
        loss = F.cross_entropy(logits, labels)
        
        return loss


class PoseLoss(nn.Module):
    """Combined loss for pose prediction"""
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        coordinate_weight: float = 0.5,
        temporal_weight: float = 0.1
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coordinate_weight = coordinate_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        target_heatmap: torch.Tensor,
        target_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pose Loss calculation
        
        Args:
            pred_logits: Predicted pose logits [B, T, pose_channels]
            target_heatmap: Target pose heatmaps [B, T, 134, H, W]
            target_coords: Target pose coordinates [B, T, 134, 2] (optional)
            
        Returns:
            total_loss: Combined pose loss value
            
        Function: Calculates heatmap, coordinate, and temporal consistency losses
        """
        """
        Args:
            pred_logits: [B, T, pose_channels] - Predicted pose logits
            target_heatmap: [B, T, pose_channels] - Target heatmap
            target_coords: [B, T, pose_channels, 2] - Target coordinates (optional)
            
        Returns:
            total_loss: Combined pose loss
        """
        # Heatmap loss (Cross-entropy)
        heatmap_loss = F.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            target_heatmap.view(-1)
        )
        
        total_loss = self.heatmap_weight * heatmap_loss
        
        # Coordinate loss (if coordinates provided)
        if target_coords is not None:
            pred_coords = self._get_pose_coordinates(pred_logits)
            coord_loss = F.mse_loss(pred_coords, target_coords)
            total_loss += self.coordinate_weight * coord_loss
        
        # Temporal consistency loss
        if pred_logits.size(1) > 1:
            temp_loss = self._temporal_consistency_loss(pred_logits)
            total_loss += self.temporal_weight * temp_loss
        
        return total_loss
    
    def _get_pose_coordinates(self, logits: torch.Tensor) -> torch.Tensor:
        """Helper to get coordinates from logits"""
        # This is a simplified version - in practice, you'd use the M2PEncoder method
        heatmap = F.softmax(logits, dim=-1)
        # Convert to coordinates (simplified)
        return heatmap.mean(dim=-1, keepdim=True).expand(-1, -1, -1, 2)
    
    def _temporal_consistency_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Temporal consistency loss"""
        # Compute difference between consecutive frames
        diff = logits[:, 1:] - logits[:, :-1]
        return torch.mean(torch.abs(diff))
