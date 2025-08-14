import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import random

class M2PDataset(Dataset):
    """
    Dataset for M2P (Music-to-Pose) training
    Loads music features, pose annotations, reference images, and face embeddings
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sequence_length: int = 150,  # 5초 * 30fps = 150프레임
        resolution: int = 512,
        pose_resolution: int = 64,
        music_feature_dim: int = 4800,
        face_embed_dim: int = 512,
        augment: bool = True
    ):
        self.data_root = data_root
        self.split = split
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.pose_resolution = pose_resolution
        self.music_feature_dim = music_feature_dim
        self.face_embed_dim = face_embed_dim
        self.augment = augment
        
        # 데이터 경로 설정
        self.split_dir = os.path.join(data_root, split)
        
        # 비디오 목록 수집
        self.video_list = self._collect_videos()
        
        # 메타데이터 로드
        self.metadata = self._load_metadata()
        
        print(f"Loaded {len(self.video_list)} videos from {split} split")
    
    def _collect_videos(self) -> List[str]:
        """비디오 목록 수집"""
        video_list = []
        
        if os.path.exists(self.split_dir):
            for item in os.listdir(self.split_dir):
                item_path = os.path.join(self.split_dir, item)
                if os.path.isdir(item_path):
                    # 필수 파일들이 있는지 확인
                    required_files = [
                        "video.mp4",
                        "music_features.npy",
                        "pose_annotations.json",
                        "reference_image.jpg"
                    ]
                    
                    if all(os.path.exists(os.path.join(item_path, f)) for f in required_files):
                        video_list.append(item)
        
        return sorted(video_list)
    
    def _load_metadata(self) -> Dict:
        """메타데이터 로드"""
        metadata_path = os.path.join(self.data_root, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load data sample for training
        
        Args:
            idx: Sample index
            
        Returns:
            sample_dict: Dictionary containing all data tensors
            
        Function: Loads and preprocesses music features, pose annotations, reference images, and face embeddings
        """
        video_id = self.video_list[idx]
        video_dir = os.path.join(self.split_dir, video_id)
        
        # 1. 음악 특징 로드
        music_features = self._load_music_features(video_dir)
        
        # 2. 포즈 어노테이션 로드
        pose_data = self._load_pose_annotations(video_dir)
        
        # 3. 참조 이미지 로드
        reference_image = self._load_reference_image(video_dir)
        
        # 4. 얼굴 임베딩 로드
        face_embedding = self._load_face_embedding(video_dir)
        
        # 5. 비디오 프레임 로드 (Stage B/C에서 사용)
        video_frames = self._load_video_frames(video_dir)
        
        # 데이터 증강
        if self.augment and self.split == "train":
            music_features, pose_data, reference_image = self._augment_data(
                music_features, pose_data, reference_image
            )
        
        # 텐서로 변환
        return {
            'video_id': video_id,
            'music_features': torch.from_numpy(music_features).float(),
            'pose_heatmap': torch.from_numpy(pose_data['heatmap']).float(),
            'pose_coordinates': torch.from_numpy(pose_data['coordinates']).float(),
            'reference_image': torch.from_numpy(reference_image).float(),
            'face_embedding': torch.from_numpy(face_embedding).float(),
            'video_frames': torch.from_numpy(video_frames).float(),
            'pose_confidence': torch.from_numpy(pose_data['confidence']).float()
        }
    
    def _load_music_features(self, video_dir: str) -> np.ndarray:
        """
        Load music features from numpy file
        
        Args:
            video_dir: Directory containing video data
            
        Returns:
            features: Music features array [T, 4800]
            
        Function: Loads pre-extracted music features and adjusts sequence length
        """
        music_path = os.path.join(video_dir, "music_features.npy")
        features = np.load(music_path)
        
        # 차원 확인 및 조정
        if features.shape[1] != self.music_feature_dim:
            raise ValueError(f"Expected music features dim {self.music_feature_dim}, got {features.shape[1]}")
        
        # 시퀀스 길이 조정
        if features.shape[0] < self.sequence_length:
            # 패딩
            padding = np.zeros((self.sequence_length - features.shape[0], self.music_feature_dim))
            features = np.vstack([features, padding])
        elif features.shape[0] > self.sequence_length:
            # 샘플링
            indices = np.linspace(0, features.shape[0]-1, self.sequence_length, dtype=int)
            features = features[indices]
        
        return features  # [T, 4800]
    
    def _load_pose_annotations(self, video_dir: str) -> Dict[str, np.ndarray]:
        """포즈 어노테이션 로드"""
        pose_path = os.path.join(video_dir, "pose_annotations.json")
        
        with open(pose_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)
        
        # 포즈 데이터를 히트맵과 좌표로 변환
        heatmaps = []
        coordinates = []
        confidences = []
        
        poses = pose_data['poses']
        
        # 시퀀스 길이 조정 (5초 * 30fps = 150프레임 기준)
        if len(poses) < self.sequence_length:
            # 마지막 포즈로 패딩
            last_pose = poses[-1] if poses else {"keypoints": [{"x": 0, "y": 0, "confidence": 0}] * 134}
            while len(poses) < self.sequence_length:
                poses.append(last_pose)
        elif len(poses) > self.sequence_length:
            # 균등 샘플링 (150프레임으로 압축)
            indices = np.linspace(0, len(poses)-1, self.sequence_length, dtype=int)
            poses = [poses[i] for i in indices]
        
        for pose in poses:
            # 히트맵 생성
            heatmap = self._pose_to_heatmap(pose['keypoints'])
            heatmaps.append(heatmap)
            
            # 좌표 추출
            coords = np.array([[kp['x'], kp['y']] for kp in pose['keypoints']])
            coordinates.append(coords)
            
            # 신뢰도 추출
            conf = np.array([kp['confidence'] for kp in pose['keypoints']])
            confidences.append(conf)
        
        return {
            'heatmap': np.array(heatmaps),  # [T, 134, H, W]
            'coordinates': np.array(coordinates),  # [T, 134, 2]
            'confidence': np.array(confidences)  # [T, 134]
        }
    
    def _pose_to_heatmap(self, keypoints: List[Dict]) -> np.ndarray:
        """포즈 키포인트를 히트맵으로 변환"""
        heatmap = np.zeros((134, self.pose_resolution, self.pose_resolution))
        
        for joint_idx, keypoint in enumerate(keypoints):
            if joint_idx >= 134:  # 134개 관절 제한
                break
                
            x, y = keypoint['x'], keypoint['y']
            confidence = keypoint['confidence']
            
            # 좌표를 히트맵 인덱스로 변환
            x_idx = int(x * self.pose_resolution / self.resolution)
            y_idx = int(y * self.pose_resolution / self.resolution)
            
            if 0 <= x_idx < self.pose_resolution and 0 <= y_idx < self.pose_resolution:
                # 가우시안 히트맵 생성
                heatmap[joint_idx] = self._create_gaussian_heatmap(
                    x_idx, y_idx, confidence
                )
        
        return heatmap
    
    def _create_gaussian_heatmap(self, x: int, y: int, confidence: float, sigma: float = 2.0) -> np.ndarray:
        """가우시안 히트맵 생성"""
        heatmap = np.zeros((self.pose_resolution, self.pose_resolution))
        
        for i in range(self.pose_resolution):
            for j in range(self.pose_resolution):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                heatmap[i, j] = confidence * np.exp(-(dist**2) / (2 * sigma**2))
        
        return heatmap
    
    def _load_reference_image(self, video_dir: str) -> np.ndarray:
        """참조 이미지 로드"""
        ref_path = os.path.join(video_dir, "reference_image.jpg")
        
        # PIL로 이미지 로드
        image = Image.open(ref_path).convert('RGB')
        
        # 리사이즈
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        
        # numpy 배열로 변환 및 정규화
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # [H, W, C] -> [C, H, W] 변환
        image_array = np.transpose(image_array, (2, 0, 1))
        
        return image_array  # [3, H, W]
    
    def _load_face_embedding(self, video_dir: str) -> np.ndarray:
        """얼굴 임베딩 로드"""
        face_path = os.path.join(video_dir, "face_embedding.npy")
        
        if os.path.exists(face_path):
            embedding = np.load(face_path)
        else:
            # 얼굴 임베딩이 없는 경우 랜덤 생성
            embedding = np.random.randn(self.face_embed_dim)
            embedding = embedding / np.linalg.norm(embedding)
        
        # 차원 확인
        if embedding.shape[0] != self.face_embed_dim:
            raise ValueError(f"Expected face embedding dim {self.face_embed_dim}, got {embedding.shape[0]}")
        
        return embedding  # [512]
    
    def _load_video_frames(self, video_dir: str) -> np.ndarray:
        """비디오 프레임 로드"""
        video_path = os.path.join(video_dir, "video.mp4")
        
        if not os.path.exists(video_path):
            # 비디오가 없는 경우 더미 프레임 생성
            dummy_frames = np.random.rand(self.sequence_length, self.resolution, self.resolution, 3)
            return dummy_frames
        
        # OpenCV로 비디오 로드
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            # 비디오 로드 실패 시 더미 프레임
            dummy_frames = np.random.rand(self.sequence_length, self.resolution, self.resolution, 3)
            return dummy_frames
        
        # 균등하게 프레임 샘플링
        frame_indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # BGR to RGB 변환
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 리사이즈
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                
                # 정규화 [0, 1]
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
            else:
                # 프레임 로드 실패 시 검은색 프레임
                black_frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.float32)
                frames.append(black_frame)
        
        cap.release()
        
        return np.array(frames)  # [T, H, W, C]
    
    def _augment_data(self, music_features: np.ndarray, pose_data: Dict, reference_image: np.ndarray) -> Tuple:
        """데이터 증강"""
        # 음악 특징에 노이즈 추가
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, music_features.shape)
            music_features = music_features + noise
        
        # 포즈 데이터에 약간의 변형
        if random.random() < 0.3:
            # 좌표에 약간의 랜덤 오프셋
            offset = np.random.normal(0, 2, pose_data['coordinates'].shape)
            pose_data['coordinates'] = pose_data['coordinates'] + offset
            
            # 신뢰도에 약간의 변형
            conf_noise = np.random.normal(0, 0.05, pose_data['confidence'].shape)
            pose_data['confidence'] = np.clip(pose_data['confidence'] + conf_noise, 0, 1)
        
        # 참조 이미지에 약간의 변형
        if random.random() < 0.3:
            # 밝기 조정
            brightness = 1.0 + np.random.normal(0, 0.1)
            reference_image = np.clip(reference_image * brightness, 0, 1)
            
            # 대비 조정
            contrast = 1.0 + np.random.normal(0, 0.1)
            reference_image = np.clip((reference_image - 0.5) * contrast + 0.5, 0, 1)
        
        return music_features, pose_data, reference_image
    
    def get_video_info(self, idx: int) -> Dict:
        """비디오 정보 반환"""
        video_id = self.video_list[idx]
        video_dir = os.path.join(self.split_dir, video_id)
        
        # 메타데이터에서 정보 추출
        video_info = self.metadata.get(video_id, {})
        
        return {
            'video_id': video_id,
            'duration': video_info.get('duration', 0.0),
            'fps': video_info.get('fps', 24),
            'dance_style': video_info.get('dance_style', 'unknown'),
            'music_genre': video_info.get('music_genre', 'unknown'),
            'difficulty': video_info.get('difficulty', 'medium')
        }


def create_m2p_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 1,
    sequence_length: int = 150,  # 5초 * 30fps = 150프레임
    resolution: int = 512,
    pose_resolution: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True
):
    """M2P 데이터로더 생성"""
    dataset = M2PDataset(
        data_root=data_root,
        split=split,
        sequence_length=sequence_length,
        resolution=resolution,
        pose_resolution=pose_resolution,
        augment=augment
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


# 데이터셋 사용 예시
if __name__ == "__main__":
    # 데이터셋 테스트
    dataset = M2PDataset(
        data_root="./dataset",
        split="train",
        sequence_length=150,  # 5초 * 30fps
        resolution=512
    )
    
    # 첫 번째 샘플 로드
    sample = dataset[0]
    
    print("Sample keys:", sample.keys())
    print("Music features shape:", sample['music_features'].shape)
    print("Pose heatmap shape:", sample['pose_heatmap'].shape)
    print("Reference image shape:", sample['reference_image'].shape)
    print("Face embedding shape:", sample['face_embedding'].shape)
    print("Video frames shape:", sample['video_frames'].shape)
    
    # 데이터로더 테스트
    dataloader = create_m2p_dataloader(
        data_root="./dataset",
        batch_size=2,
        sequence_length=150  # 5초 * 30fps
    )
    
    for batch in dataloader:
        print("Batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        break
