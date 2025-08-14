import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset
import cv2

class GuidanceDataset(Dataset):
    """
    Dataset class for training CHAMP model with guidance data
    Supports multiple guidance types: music, pose, etc.
    """
    
    def __init__(
        self,
        data_root: str,
        guidance_types: List[str],
        resolution: int = 512,
        sequence_length: int = 16,
        guidance_input_channels: List[int] = None,
        transform=None,
        cache_dir: Optional[str] = None
    ):
        self.data_root = Path(data_root)
        self.guidance_types = guidance_types
        self.resolution = resolution
        self.sequence_length = sequence_length
        self.guidance_input_channels = guidance_input_channels or [1] * len(guidance_types)
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Validate guidance types and channels
        if len(guidance_types) != len(self.guidance_input_channels):
            raise ValueError("guidance_types and guidance_input_channels must have the same length")
        
        # Find all video files
        self.video_files = self._find_video_files()
        
        # Load guidance data
        self.guidance_data = self._load_guidance_data()
        
        print(f"Found {len(self.video_files)} video files")
        print(f"Guidance types: {guidance_types}")
        print(f"Guidance channels: {self.guidance_input_channels}")
    
    def _find_video_files(self) -> List[Path]:
        """Find all video files in the data directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.data_root.rglob(f"*{ext}"))
        
        return sorted(video_files)
    
    def _load_guidance_data(self) -> Dict[str, Dict]:
        """Load guidance data for each video"""
        guidance_data = {}
        
        for video_file in self.video_files:
            video_id = video_file.stem
            guidance_data[video_id] = {}
            
            # Load pose guidance data
            if 'pose' in self.guidance_types:
                pose_dir = self.data_root / 'poses' / video_id
                if pose_dir.exists():
                    guidance_data[video_id]['pose'] = self._load_pose_data(pose_dir)
            
            # Load music guidance data
            if 'music' in self.guidance_types:
                music_dir = self.data_root / 'music_features' / video_id
                if music_dir.exists():
                    guidance_data[video_id]['music'] = self._load_music_data(music_dir)
            
            # Load reference image
            ref_image_path = self.data_root / 'reference_images' / f"{video_id}.jpg"
            if ref_image_path.exists():
                guidance_data[video_id]['reference'] = str(ref_image_path)
        
        return guidance_data
    
    def _load_pose_data(self, pose_dir: Path) -> Dict:
        """Load pose data from directory"""
        pose_files = sorted(pose_dir.glob("*.json"))
        pose_data = []
        
        for pose_file in pose_files:
            with open(pose_file, 'r') as f:
                pose_info = json.load(f)
                pose_data.append(pose_info)
        
        return pose_data
    
    def _load_music_data(self, music_dir: Path) -> Dict:
        """Load music features from directory"""
        music_files = sorted(music_dir.glob("*.npy"))
        music_data = []
        
        for music_file in music_files:
            music_features = np.load(music_file)
            music_data.append(music_features)
        
        return music_data
    
    def _extract_frames(self, video_path: Path, num_frames: int) -> List[Image.Image]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames:
            # If video is shorter than required, repeat frames
            frame_indices = list(range(total_frames)) * (num_frames // total_frames + 1)
            frame_indices = frame_indices[:num_frames]
        else:
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_pil = frame_pil.resize((self.resolution, self.resolution))
                frames.append(frame_pil)
            else:
                # If frame reading fails, use a black frame
                black_frame = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
                frames.append(black_frame)
        
        cap.release()
        return frames
    
    def _create_guidance_condition(self, video_id: str, frame_indices: List[int]) -> torch.Tensor:
        """Create guidance condition tensor for given frames"""
        guidance_conditions = []
        
        for guidance_type, input_channels in zip(self.guidance_types, self.guidance_input_channels):
            if guidance_type == 'pose':
                pose_data = self.guidance_data[video_id].get('pose', [])
                if pose_data:
                    # Extract pose features for the specified frames
                    pose_features = []
                    for idx in frame_indices:
                        if idx < len(pose_data):
                            # Convert pose keypoints to feature tensor
                            pose_feature = self._pose_to_feature(pose_data[idx], input_channels)
                        else:
                            # Use zero tensor if frame index is out of range
                            pose_feature = torch.zeros(input_channels, self.resolution, self.resolution)
                        pose_features.append(pose_feature)
                    
                    # Stack frames and add temporal dimension
                    pose_tensor = torch.stack(pose_features, dim=0)  # [T, C, H, W]
                    pose_tensor = pose_tensor.unsqueeze(0)  # [1, T, C, H, W]
                    guidance_conditions.append(pose_tensor)
                else:
                    # Use random pose features if no pose data
                    pose_tensor = torch.randn(1, self.sequence_length, input_channels, 
                                           self.resolution, self.resolution)
                    guidance_conditions.append(pose_tensor)
            
            elif guidance_type == 'music':
                music_data = self.guidance_data[video_id].get('music', [])
                if music_data:
                    # Extract music features for the specified frames
                    music_features = []
                    for idx in frame_indices:
                        if idx < len(music_data):
                            music_feature = torch.from_numpy(music_data[idx]).float()
                            # Reshape to match expected dimensions
                            if music_feature.dim() == 1:
                                music_feature = music_feature.view(input_channels, self.resolution, self.resolution)
                            else:
                                music_feature = music_feature[:input_channels, :self.resolution, :self.resolution]
                        else:
                            music_feature = torch.zeros(input_channels, self.resolution, self.resolution)
                        music_features.append(music_feature)
                    
                    # Stack frames and add temporal dimension
                    music_tensor = torch.stack(music_features, dim=0)  # [T, C, H, W]
                    music_tensor = music_tensor.unsqueeze(0)  # [1, T, C, H, W]
                    guidance_conditions.append(music_tensor)
                else:
                    # Use random music features if no music data
                    music_tensor = torch.randn(1, self.sequence_length, input_channels, 
                                            self.resolution, self.resolution)
                    guidance_conditions.append(music_tensor)
            
            else:
                # Default guidance type
                default_tensor = torch.randn(1, self.sequence_length, input_channels, 
                                          self.resolution, self.resolution)
                guidance_conditions.append(default_tensor)
        
        # Concatenate all guidance conditions along channel dimension
        multi_guidance_cond = torch.cat(guidance_conditions, dim=2)  # [1, T, total_C, H, W]
        
        return multi_guidance_cond
    
    def _pose_to_feature(self, pose_info: Dict, num_channels: int) -> torch.Tensor:
        """Convert pose keypoints to feature tensor"""
        # This is a simplified conversion - you may need to customize based on your pose format
        if 'keypoints' in pose_info:
            keypoints = pose_info['keypoints']
            # Convert keypoints to heatmap or feature representation
            # For now, create a simple feature tensor
            feature = torch.zeros(num_channels, self.resolution, self.resolution)
            
            # Place keypoints on the feature map
            for i, kp in enumerate(keypoints[:num_channels]):
                if len(kp) >= 2:
                    x, y = int(kp[0] * self.resolution), int(kp[1] * self.resolution)
                    if 0 <= x < self.resolution and 0 <= y < self.resolution:
                        feature[i, y, x] = 1.0
            
            return feature
        else:
            # Return random features if no keypoints
            return torch.randn(num_channels, self.resolution, self.resolution)
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path = self.video_files[idx]
        video_id = video_path.stem
        
        # Extract frames from video
        frames = self._extract_frames(video_path, self.sequence_length)
        
        # Convert frames to tensors
        frame_tensors = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            frame_tensor = torch.from_numpy(np.array(frame)).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
            frame_tensors.append(frame_tensor)
        
        # Stack frames
        video_tensor = torch.stack(frame_tensors, dim=0)  # [T, C, H, W]
        
        # Create guidance condition
        frame_indices = list(range(self.sequence_length))
        guidance_condition = self._create_guidance_condition(video_id, frame_indices)
        
        # Load reference image if available
        reference_tensor = None
        if 'reference' in self.guidance_data[video_id]:
            ref_image_path = self.guidance_data[video_id]['reference']
            ref_image = Image.open(ref_image_path).convert('RGB')
            ref_image = ref_image.resize((self.resolution, self.resolution))
            if self.transform:
                ref_image = self.transform(ref_image)
            reference_tensor = torch.from_numpy(np.array(ref_image)).float() / 255.0
            reference_tensor = reference_tensor.permute(2, 0, 1)  # HWC -> CHW
        else:
            # Use first frame as reference if no reference image
            reference_tensor = frame_tensors[0]
        
        return {
            'video': video_tensor,  # [T, C, H, W]
            'guidance_condition': guidance_condition,  # [1, T, total_C, H, W]
            'reference_image': reference_tensor,  # [C, H, W]
            'video_id': video_id
        }

def create_guidance_dataloader(
    data_root: str,
    guidance_types: List[str],
    guidance_input_channels: List[int],
    batch_size: int = 1,
    resolution: int = 512,
    sequence_length: int = 16,
    num_workers: int = 4,
    shuffle: bool = True
):
    """Create a dataloader for guidance training"""
    from torch.utils.data import DataLoader
    
    dataset = GuidanceDataset(
        data_root=data_root,
        guidance_types=guidance_types,
        resolution=resolution,
        sequence_length=sequence_length,
        guidance_input_channels=guidance_input_channels
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
