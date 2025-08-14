import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class M2PEncoder(nn.Module):
    """
    Music-to-Pose Encoder with Face Embedding
    Generates pose heatmaps and face embeddings from music features
    
    Architecture:
    Music Features [B, T, 4800] → Hidden Features [B, T, 1024] → 
    Pose Output [B, T, 134] + Face Output [B, T, 512]
    
    Data Format (based on P00002):
    - Face images: [B, T, 1, 640, 360] - Grayscale face images
    - Pose images: [B, T, 3, 640, 360] - RGB pose skeleton images
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
        
        # Music feature processing with dimension reduction
        self.music_proj = nn.Sequential(
            nn.Linear(music_input_dim, hidden_dim * 2),  # 4800 → 2048
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),      # 2048 → 1024
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
        
        # Pose prediction head with dimension mapping
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),     # 1024 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), # 512 → 256
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, pose_channels)    # 256 → 134
        )
        
        # Face embedding head with dimension mapping
        # Modified to output per-frame embeddings instead of single embedding
        self.face_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),     # 1024 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, face_embed_dim)  # 512 → 512
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
            pose_logits: Pose prediction logits [B, T, 134]
            face_embed: Face embedding [B, T, 512] or None
            
        Dimension Flow:
        1. Input: [B, T, 4800] - Music features
        2. Projection: [B, T, 4800] → [B, T, 1024]
        3. Transformer: [B, T, 1024] → [B, T, 1024]
        4. Pose Head: [B, T, 1024] → [B, T, 134]
        5. Face Head: [B, T, 1024] → [B, T, 512] (per-frame embeddings)
        """
        batch_size, seq_len, _ = music_features.shape
        
        # Validate input dimensions
        if music_features.size(-1) != self.music_input_dim:
            raise ValueError(f"Expected music features dimension {self.music_input_dim}, got {music_features.size(-1)}")
        
        # Project music features: [B, T, 4800] → [B, T, 1024]
        x = self.music_proj(music_features)  # [B, T, hidden_dim]
        
        # Add positional encoding
        pos_encoding = self._get_pos_encoding(seq_len, self.hidden_dim).to(x.device)
        x = x + pos_encoding
        
        # Apply transformer: [B, T, 1024] → [B, T, 1024]
        x = self.transformer(x)  # [B, T, hidden_dim]
        
        # Pose prediction: [B, T, 1024] → [B, T, 134]
        pose_logits = self.pose_head(x)  # [B, T, pose_channels]
        
        # Face embedding: [B, T, 1024] → [B, T, 512]
        if return_face_embed:
            # Generate per-frame face embeddings (no pooling)
            face_embed = self.face_head(x)  # [B, T, 512]
            
            # L2 normalization for each frame
            face_embed = F.normalize(face_embed, p=2, dim=-1)
            
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
        """
        Convert pose logits to heatmap using softmax
        
        Args:
            pose_logits: [B, T, 134] - Pose prediction logits
            temperature: Temperature for softmax scaling
            
        Returns:
            heatmap: [B, T, 134] - Normalized pose probabilities
        """
        # Apply temperature scaling and softmax
        heatmap = F.softmax(pose_logits / temperature, dim=-1)
        return heatmap
    
    def get_pose_coordinates(self, pose_logits: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Convert pose logits to coordinates using soft-argmax
        
        Args:
            pose_logits: [B, T, 134] - Pose prediction logits
            temperature: Temperature for softmax scaling
            
        Returns:
            coordinates: [B, T, 134, 2] - X,Y coordinates for each joint
        """
        heatmap = self.get_pose_heatmap(pose_logits, temperature)
        
        # For 134 joints, we'll create a reasonable spatial layout
        # Assuming joints can be arranged in a grid-like structure
        num_joints = heatmap.size(-1)
        
        # Create a reasonable grid size (e.g., 12x12 = 144, but we have 134)
        grid_size = int(math.ceil(math.sqrt(num_joints)))
        
        # Pad or truncate to fit the grid
        if grid_size * grid_size > num_joints:
            # Pad with zeros
            padding_size = grid_size * grid_size - num_joints
            heatmap_padded = F.pad(heatmap, (0, padding_size), value=0)
        else:
            heatmap_padded = heatmap[:, :, :grid_size * grid_size]
        
        # Reshape to spatial dimensions: [B, T, 134] → [B, T, 134, H, W]
        heatmap_spatial = heatmap_padded.view(heatmap.size(0), heatmap.size(1), -1, grid_size, grid_size)
        
        # Generate coordinate grid
        y_coords = torch.arange(grid_size, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, grid_size, 1)
        x_coords = torch.arange(grid_size, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, 1, grid_size)
        
        # Weighted average to get coordinates
        y_coord = (heatmap_spatial * y_coords).sum(dim=(3, 4))  # [B, T, num_joints]
        x_coord = (heatmap_spatial * x_coords).sum(dim=(3, 4))  # [B, T, num_joints]
        
        # Stack coordinates: [B, T, 134, 2]
        coordinates = torch.stack([x_coord, y_coord], dim=-1)
        
        # Return only the original number of joints
        return coordinates[:, :, :num_joints, :]
    
    def get_output_dimensions(self) -> dict:
        """Get output dimensions for debugging and validation"""
        return {
            'pose_output': (self.pose_channels,),  # 134
            'face_output': (self.face_embed_dim,),  # 512
            'hidden_dim': self.hidden_dim,  # 1024
            'music_input_dim': self.music_input_dim  # 4800
        }


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
            pred_logits: Predicted pose logits [B, T, 134]
            target_heatmap: Target pose heatmaps [B, T, 134] or [B, T, 134, H, W]
            target_coords: Target pose coordinates [B, T, 134, 2] (optional)
            
        Returns:
            total_loss: Combined pose loss value
            
        Dimension Handling:
        - pred_logits: [B, T, 134] - Raw logits from M2P encoder
        - target_heatmap: [B, T, 134] - Class labels or [B, T, 134, H, W] - Spatial heatmaps
        - target_coords: [B, T, 134, 2] - X,Y coordinates for each joint
        """
        batch_size, seq_len, num_joints = pred_logits.shape
        
        # Validate dimensions
        if pred_logits.size(-1) != 134:
            raise ValueError(f"Expected 134 pose channels, got {pred_logits.size(-1)}")
        
        # Handle different target formats
        if target_heatmap.dim() == 3:  # [B, T, 134] - Class labels
            # Cross-entropy loss for classification
            heatmap_loss = F.cross_entropy(
                pred_logits.view(-1, num_joints),  # [B*T, 134]
                target_heatmap.view(-1)            # [B*T]
            )
        elif target_heatmap.dim() == 5:  # [B, T, 134, H, W] - Spatial heatmaps
            # Convert logits to spatial heatmaps and compute MSE
            pred_heatmap = F.softmax(pred_logits, dim=-1)  # [B, T, 134]
            
            # Reshape target to match prediction format
            target_flat = target_heatmap.view(batch_size, seq_len, num_joints, -1)  # [B, T, 134, H*W]
            target_flat = target_flat.mean(dim=-1)  # [B, T, 134] - Average over spatial dimensions
            
            heatmap_loss = F.mse_loss(pred_heatmap, target_flat)
        else:
            raise ValueError(f"Unsupported target_heatmap dimensions: {target_heatmap.shape}")
        
        total_loss = self.heatmap_weight * heatmap_loss
        
        # Coordinate loss (if coordinates provided)
        if target_coords is not None:
            # Validate coordinate dimensions
            if target_coords.shape[-1] != 2:
                raise ValueError(f"Expected 2D coordinates, got {target_coords.shape[-1]}")
            
            # Get predicted coordinates from logits
            pred_coords = self._get_pose_coordinates(pred_logits)  # [B, T, 134, 2]
            
            # Ensure dimensions match
            if pred_coords.shape != target_coords.shape:
                raise ValueError(f"Coordinate shape mismatch: pred {pred_coords.shape} vs target {target_coords.shape}")
            
            coord_loss = F.mse_loss(pred_coords, target_coords)
            total_loss += self.coordinate_weight * coord_loss
        
        # Temporal consistency loss
        if seq_len > 1:
            temp_loss = self._temporal_consistency_loss(pred_logits)
            total_loss += self.temporal_weight * temp_loss
        
        return total_loss
    
    def _get_pose_coordinates(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Helper to get coordinates from logits
        
        Args:
            logits: [B, T, 134] - Pose prediction logits
            
        Returns:
            coordinates: [B, T, 134, 2] - X,Y coordinates for each joint
        """
        # Convert logits to probabilities
        heatmap = F.softmax(logits, dim=-1)  # [B, T, 134]
        
        # For 134 joints, create a reasonable spatial layout
        num_joints = heatmap.size(-1)
        grid_size = int(math.ceil(math.sqrt(num_joints)))
        
        # Pad heatmap to fit grid
        if grid_size * grid_size > num_joints:
            padding_size = grid_size * grid_size - num_joints
            heatmap_padded = F.pad(heatmap, (0, padding_size), value=0)
        else:
            heatmap_padded = heatmap[:, :, :grid_size * grid_size]
        
        # Reshape to spatial dimensions
        heatmap_spatial = heatmap_padded.view(heatmap.size(0), heatmap.size(1), -1, grid_size, grid_size)
        
        # Generate coordinate grid
        y_coords = torch.arange(grid_size, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, grid_size, 1)
        x_coords = torch.arange(grid_size, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, 1, grid_size)
        
        # Weighted average to get coordinates
        y_coord = (heatmap_spatial * y_coords).sum(dim=(3, 4))  # [B, T, num_joints]
        x_coord = (heatmap_spatial * x_coords).sum(dim=(3, 4))  # [B, T, num_joints]
        
        # Stack coordinates and return only original joints
        coordinates = torch.stack([x_coord, y_coord], dim=-1)  # [B, T, num_joints, 2]
        return coordinates[:, :, :num_joints, :]
    
    def _temporal_consistency_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Temporal consistency loss
        
        Args:
            logits: [B, T, 134] - Pose prediction logits
            
        Returns:
            temporal_loss: Scalar temporal consistency loss
        """
        # Compute difference between consecutive frames
        diff = logits[:, 1:] - logits[:, :-1]  # [B, T-1, 134]
        
        # L2 norm of temporal differences
        temporal_loss = torch.mean(torch.norm(diff, p=2, dim=-1))
        
        return temporal_loss
