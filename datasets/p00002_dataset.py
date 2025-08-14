import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any
import json
from pathlib import Path

class P00002Dataset(Dataset):
    """
    Dataset for P00002 face and pose sequences
    
    Data Format:
    - Face images: [T, 1, 640, 360] - Grayscale face images
    - Pose images: [T, 3, 640, 360] - RGB pose skeleton images
    - Music features: [T, 4800] - Extracted music features (placeholder)
    
    Expected directory structure:
    P00002/
    ├── faces/          # frame_0.png, frame_1.png, ...
    ├── poses/          # frame_0.png, frame_1.png, ...
    ├── images/         # Original video frames
    └── music_features/ # Extracted music features (optional)
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 24,
        stride: int = 1,
        transform=None,
        load_music_features: bool = False,
        music_feature_dim: int = 4800
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.load_music_features = load_music_features
        self.music_feature_dim = music_feature_dim
        
        # Get all frame files
        self.face_dir = self.data_dir / "faces"
        self.pose_dir = self.data_dir / "poses"
        self.image_dir = self.data_dir / "images"
        
        # Collect frame files
        self.frame_files = self._collect_frame_files()
        
        # Create sequence indices
        self.sequence_indices = self._create_sequence_indices()
        
        print(f"P00002 Dataset initialized:")
        print(f"  - Data directory: {data_dir}")
        print(f"  - Total frames: {len(self.frame_files)}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Stride: {stride}")
        print(f"  - Total sequences: {len(self.sequence_indices)}")
    
    def _collect_frame_files(self) -> list:
        """Collect all available frame files"""
        frame_files = []
        
        if self.face_dir.exists():
            # Get face frame files
            face_files = sorted([f for f in self.face_dir.glob("frame_*.png")])
            frame_files = [f.stem for f in face_files]  # Remove extension
        
        return frame_files
    
    def _create_sequence_indices(self) -> list:
        """Create indices for sequences"""
        sequences = []
        
        for i in range(0, len(self.frame_files) - self.sequence_length + 1, self.stride):
            sequence = self.frame_files[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load image and convert to tensor"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default conversion to tensor
            image = torch.from_numpy(np.array(image)).float()
            
            # Normalize to [0, 1]
            if image.dtype == torch.uint8:
                image = image / 255.0
            
            # Add channel dimension if grayscale
            if image.dim() == 2:
                image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
            elif image.dim() == 3:
                image = image.permute(2, 0, 1)  # [C, H, W]
        
        return image
    
    def _load_face_sequence(self, frame_names: list) -> torch.Tensor:
        """Load face image sequence"""
        face_images = []
        
        for frame_name in frame_names:
            face_path = self.face_dir / f"{frame_name}.png"
            face_image = self._load_image(face_path)
            face_images.append(face_image)
        
        # Stack into sequence: [T, C, H, W]
        face_sequence = torch.stack(face_images, dim=0)
        return face_sequence
    
    def _load_pose_sequence(self, frame_names: list) -> torch.Tensor:
        """Load pose image sequence"""
        pose_images = []
        
        for frame_name in frame_names:
            pose_path = self.pose_dir / f"{frame_name}.png"
            pose_image = self._load_image(pose_path)
            pose_images.append(pose_image)
        
        # Stack into sequence: [T, C, H, W]
        pose_sequence = torch.stack(pose_images, dim=0)
        return pose_sequence
    
    def _load_music_features(self, frame_names: list) -> torch.Tensor:
        """Load or generate music features (placeholder)"""
        if self.load_music_features and (self.data_dir / "music_features").exists():
            # TODO: Implement actual music feature loading
            # For now, generate random features
            pass
        
        # Generate dummy music features
        # In real implementation, you would load pre-extracted features
        music_features = torch.randn(len(frame_names), self.music_feature_dim)
        return music_features
    
    def __len__(self) -> int:
        return len(self.sequence_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence of face and pose images"""
        frame_names = self.sequence_indices[idx]
        
        # Load face sequence: [T, 1, 640, 360]
        face_sequence = self._load_face_sequence(frame_names)
        
        # Load pose sequence: [T, 3, 640, 360]
        pose_sequence = self._load_pose_sequence(frame_names)
        
        # Load music features: [T, 4800]
        music_features = self._load_music_features(frame_names)
        
        # Create sample
        sample = {
            'face_sequence': face_sequence,      # [T, 1, 640, 360]
            'pose_sequence': pose_sequence,      # [T, 3, 640, 360]
            'music_features': music_features,    # [T, 4800]
            'frame_names': frame_names,          # List of frame names
            'sequence_length': self.sequence_length
        }
        
        return sample
    
    def get_sequence_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific sequence"""
        frame_names = self.sequence_indices[idx]
        
        info = {
            'sequence_index': idx,
            'frame_names': frame_names,
            'sequence_length': len(frame_names),
            'start_frame': frame_names[0],
            'end_frame': frame_names[-1],
            'face_files': [f"{name}.png" for name in frame_names],
            'pose_files': [f"{name}.png" for name in frame_names]
        }
        
        return info


def create_p00002_dataloader(
    data_dir: str,
    batch_size: int = 4,
    sequence_length: int = 24,
    stride: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> torch.utils.data.DataLoader:
    """Create DataLoader for P00002 dataset"""
    
    dataset = P00002Dataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        stride=stride,
        **kwargs
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    dataset = P00002Dataset("../P00002", sequence_length=16, stride=2)
    
    print(f"\nDataset test:")
    print(f"Dataset length: {len(dataset)}")
    
    # Get first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    # Get sequence info
    info = dataset.get_sequence_info(0)
    print(f"\nSequence info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
