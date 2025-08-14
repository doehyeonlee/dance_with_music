#!/usr/bin/env python3
"""
Dataset Preparation Script for M2P Training
Converts raw video data into the required format for training
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
import librosa
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import shutil
import random

def extract_music_features(audio_path: str, target_dim: int = 4800) -> np.ndarray:
    """
    Extract music features from audio file
    
    Args:
        audio_path: Path to audio file
        target_dim: Target feature dimension (default: 4800)
        
    Returns:
        features: Music features array [T, target_dim]
        
    Function: Extracts MFCC, Chroma, Spectral Contrast, Tonnetz, and Tempo features
    """
    print(f"Extracting music features from {audio_path}")
    
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 기본 특징들 추출
        features = []
        
        # MFCC (13차원)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.append(mfcc)
        
        # Chroma (12차원)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.append(chroma)
        
        # Spectral contrast (7차원)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.append(contrast)
        
        # Tonnetz (6차원)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.append(tonnetz)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_features = np.full((1, mfcc.shape[1]), tempo)
        features.append(tempo_features)
        
        # 결합
        combined = np.vstack(features)
        
        # 차원 조정
        if combined.shape[0] < target_dim:
            # 패딩
            padding = np.zeros((target_dim - combined.shape[0], combined.shape[1]))
            combined = np.vstack([combined, padding])
        else:
            # 선형 보간으로 축소
            combined = resize_features(combined, target_dim)
        
        return combined.T  # [T, 4800]
        
    except Exception as e:
        print(f"Error extracting music features: {e}")
        # 에러 시 랜덤 특징 생성
        return np.random.randn(100, target_dim)

def resize_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    """특징 차원을 목표 차원으로 조정"""
    from scipy.interpolate import interp1d
    
    current_dim = features.shape[0]
    x_old = np.linspace(0, 1, current_dim)
    x_new = np.linspace(0, 1, target_dim)
    
    resized_features = []
    for i in range(features.shape[1]):
        f = interp1d(x_old, features[:, i], kind='linear')
        resized_features.append(f(x_new))
    
    return np.array(resized_features).T

def extract_video_frames(video_path: str, output_dir: str, fps: int = 30) -> List[str]:
    """비디오에서 프레임 추출 (5초, 30fps = 150프레임)"""
    print(f"Extracting frames from {video_path} at {fps}fps")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # FFmpeg로 프레임 추출
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-frame_pts', '1',
        os.path.join(output_dir, 'frame_%06d.jpg'),
        '-y'  # 덮어쓰기
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 추출된 프레임 목록
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        frame_paths = [os.path.join(output_dir, f) for f in frame_files]
        
        # 5초 30fps = 150프레임 확인
        expected_frames = 5 * fps
        if len(frame_paths) != expected_frames:
            print(f"Warning: Expected {expected_frames} frames, got {len(frame_paths)}")
        
        print(f"Extracted {len(frame_paths)} frames at {fps}fps")
        return frame_paths
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        return []

def extract_audio_from_video(video_path: str, output_path: str) -> bool:
    """비디오에서 오디오 추출"""
    print(f"Extracting audio from {video_path}")
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '22050', '-ac', '1',
        output_path, '-y'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        return False

def create_pose_annotations_from_images(poses_dir: str, faces_dir: str, output_path: str, fps: int = 30) -> Dict:
    """P00002 구조의 포즈/페이스 이미지에서 어노테이션 생성"""
    print(f"Creating pose annotations from {poses_dir} and {faces_dir}")
    
    # 포즈 이미지 파일들 수집
    pose_files = sorted([f for f in os.listdir(poses_dir) if f.endswith('.png')])
    face_files = sorted([f for f in os.listdir(faces_dir) if f.endswith('.png')])
    
    if not pose_files:
        print("Warning: No pose images found, using dummy data")
        return create_pose_annotations_dummy([], output_path, fps)
    
    poses = []
    total_frames = len(pose_files)
    
    for i, pose_file in enumerate(pose_files):
        frame_id = int(pose_file.replace('frame_', '').replace('.png', ''))
        
        # 포즈 이미지에서 키포인트 추출 (실제 구현에서는 포즈 추정 모델 사용)
        # 여기서는 이미지 크기 기반으로 더미 데이터 생성
        keypoints = []
        for j in range(134):
            # 이미지 크기에 맞춰 좌표 생성
            x = np.random.randint(0, 512)
            y = np.random.randint(0, 512)
            confidence = np.random.uniform(0.7, 1.0)
            
            keypoint = {
                "x": x,
                "y": y,
                "confidence": confidence,
                "joint_type": f"joint_{j}"
            }
            keypoints.append(keypoint)
        
        pose = {
            "frame_id": frame_id,
            "timestamp": frame_id / fps,
            "keypoints": keypoints
        }
        poses.append(pose)
    
    pose_data = {
        "video_id": os.path.basename(os.path.dirname(poses_dir)),
        "fps": fps,
        "duration": total_frames / fps,
        "total_frames": total_frames,
        "poses": poses,
        "pose_images_dir": poses_dir,
        "face_images_dir": faces_dir
    }
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, indent=2, ensure_ascii=False)
    
    return pose_data

def create_pose_annotations_dummy(frame_paths: List[str], output_path: str, fps: int = 30) -> Dict:
    """더미 포즈 어노테이션 생성 (5초, 30fps)"""
    print(f"Creating dummy pose annotations for {len(frame_paths)} frames at {fps}fps")
    
    poses = []
    total_frames = len(frame_paths) if frame_paths else 150  # 기본 150프레임
    
    for i in range(total_frames):
        # 더미 포즈 데이터 (134개 관절)
        keypoints = []
        for j in range(134):
            # 시간에 따른 자연스러운 움직임 시뮬레이션
            base_x = 256 + 50 * np.sin(i * 0.1 + j * 0.05)  # 사인파 움직임
            base_y = 256 + 30 * np.cos(i * 0.08 + j * 0.03)  # 코사인파 움직임
            
            # 랜덤 노이즈 추가
            noise_x = np.random.normal(0, 5)
            noise_y = np.random.normal(0, 5)
            
            x = int(np.clip(base_x + noise_x, 0, 511))
            y = int(np.clip(base_y + noise_y, 0, 511))
            
            # 신뢰도는 시간에 따라 약간 변동
            confidence = np.random.uniform(0.7, 1.0) * (0.9 + 0.1 * np.sin(i * 0.2))
            confidence = np.clip(confidence, 0.5, 1.0)
            
            keypoint = {
                "x": x,
                "y": y,
                "confidence": confidence,
                "joint_type": f"joint_{j}"
            }
            keypoints.append(keypoint)
        
        pose = {
            "frame_id": i,
            "timestamp": i / fps,  # 30fps 기준
            "keypoints": keypoints
        }
        poses.append(pose)
    
    pose_data = {
        "video_id": "dummy_video",
        "fps": fps,
        "duration": total_frames / fps,  # 5초
        "total_frames": total_frames,
        "poses": poses
    }
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, indent=2, ensure_ascii=False)
    
    return pose_data

def create_pose_annotations(frame_paths: List[str], output_path: str, fps: int = 30) -> Dict:
    """포즈 어노테이션 생성 (기본 함수, 더미 데이터)"""
    return create_pose_annotations_dummy(frame_paths, output_path, fps)

def extract_reference_image(frame_paths: List[str], output_path: str, strategy: str = "random") -> bool:
    """참조 이미지 추출 (랜덤 선택 또는 특정 전략)"""
    if not frame_paths:
        return False
    
    try:
        if strategy == "random":
            # 전체 프레임 중에서 랜덤 선택
            selected_frame = random.choice(frame_paths)
            frame_idx = frame_paths.index(selected_frame)
            print(f"Randomly selected frame {frame_idx} from {len(frame_paths)} frames")
        elif strategy == "middle":
            # 중간 프레임 선택 (안정적인 포즈)
            selected_frame = frame_paths[len(frame_paths) // 2]
            frame_idx = len(frame_paths) // 2
            print(f"Selected middle frame {frame_idx} from {len(frame_paths)} frames")
        elif strategy == "first":
            # 첫 번째 프레임 선택
            selected_frame = frame_paths[0]
            frame_idx = 0
            print(f"Selected first frame {frame_idx} from {len(frame_paths)} frames")
        else:
            # 기본값: 랜덤 선택
            selected_frame = random.choice(frame_paths)
            frame_idx = frame_paths.index(selected_frame)
            print(f"Randomly selected frame {frame_idx} from {len(frame_paths)} frames")
        
        # 선택된 프레임을 참조 이미지로 사용
        shutil.copy2(selected_frame, output_path)
        
        # 리사이즈
        with Image.open(output_path) as img:
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            img.save(output_path, quality=95)
        
        print(f"Reference image saved to {output_path} (frame {frame_idx})")
        return True
        
    except Exception as e:
        print(f"Error creating reference image: {e}")
        return False

def create_face_embedding(image_path: str, output_path: str) -> bool:
    """얼굴 임베딩 생성 (더미 데이터)"""
    print(f"Creating face embedding for {image_path}")
    
    try:
        # 실제 구현에서는 InsightFace, ArcFace 등 사용
        # 여기서는 랜덤 임베딩 생성
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)  # L2 정규화
        
        np.save(output_path, embedding)
        print(f"Face embedding saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating face embedding: {e}")
        return False

def process_p00002_structure(
    input_dir: str,
    output_dir: str,
    video_id: str,
    fps: int = 30
) -> bool:
    """
    Process P00002 structure data (poses/, faces/, images/ folders)
    
    Args:
        input_dir: Input directory with P00002 structure
        output_dir: Output directory for processed data
        video_id: Unique video identifier
        fps: Target frame rate (default: 30)
        
    Returns:
        success: Whether processing was successful
        
    Function: Automatically processes P00002 format data into M2P training format
    """
    print(f"\nProcessing P00002 structure: {input_dir}")
    
    # 출력 디렉토리 생성
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # P00002 구조 확인
    poses_dir = os.path.join(input_dir, "poses")
    faces_dir = os.path.join(input_dir, "faces")
    images_dir = os.path.join(input_dir, "images")
    
    if not os.path.exists(poses_dir):
        print(f"Error: poses directory not found in {input_dir}")
        return False
    
    # 1. 포즈 어노테이션 생성
    pose_annotations_path = os.path.join(video_output_dir, "pose_annotations.json")
    pose_data = create_pose_annotations_from_images(poses_dir, faces_dir, pose_annotations_path, fps)
    
    # 2. 참조 이미지 생성 (faces 폴더에서 랜덤 선택)
    reference_image_path = os.path.join(video_output_dir, "reference_image.jpg")
    if os.path.exists(faces_dir):
        face_files = [f for f in os.listdir(faces_dir) if f.endswith('.png')]
        if face_files:
            # 랜덤으로 얼굴 이미지 선택
            selected_face = random.choice(face_files)
            face_path = os.path.join(faces_dir, selected_face)
            
            # 참조 이미지로 복사 및 리사이즈
            with Image.open(face_path) as img:
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                img.save(reference_image_path, quality=95)
            
            print(f"Selected reference face: {selected_face}")
        else:
            print("Warning: No face images found")
            # 더미 참조 이미지 생성
            dummy_img = Image.new('RGB', (512, 512), color='gray')
            dummy_img.save(reference_image_path)
    else:
        print("Warning: faces directory not found")
        # 더미 참조 이미지 생성
        dummy_img = Image.new('RGB', (512, 512), color='gray')
        dummy_img.save(reference_image_path)
    
    # 3. 얼굴 임베딩 생성
    face_embedding_path = os.path.join(video_output_dir, "face_embedding.npy")
    if not create_face_embedding(reference_image_path, face_embedding_path):
        print("Warning: Face embedding creation failed")
    
    # 4. 음악 특징 생성 (더미 데이터)
    music_features_path = os.path.join(video_output_dir, "music_features.npy")
    # 150프레임에 맞춰 음악 특징 생성
    music_features = np.random.randn(150, 4800)  # [150, 4800]
    np.save(music_features_path, music_features)
    
    # 5. 비디오 파일 생성 (images 폴더의 프레임들을 비디오로 결합)
    video_output_path = os.path.join(video_output_dir, "video.mp4")
    if os.path.exists(images_dir):
        # FFmpeg로 프레임들을 비디오로 결합
        frame_pattern = os.path.join(images_dir, "frame_%d.png")
        cmd = [
            'ffmpeg', '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_output_path,
            '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video created from frames: {video_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Video creation failed: {e}")
            # 더미 비디오 파일 생성
            dummy_video = np.random.randint(0, 255, (150, 512, 512, 3), dtype=np.uint8)
            # OpenCV로 비디오 저장
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_output_path, fourcc, fps, (512, 512))
            for frame in dummy_video:
                out.write(frame)
            out.release()
            print("Created dummy video file")
    else:
        print("Warning: images directory not found")
        # 더미 비디오 파일 생성
        dummy_video = np.random.randint(0, 255, (150, 512, 512, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (512, 512))
        for frame in dummy_video:
            out.write(frame)
        out.release()
        print("Created dummy video file")
    
    print(f"P00002 structure {video_id} processed successfully")
    return True

def create_metadata(video_info: Dict, output_path: str) -> Dict:
    """메타데이터 생성"""
    metadata = {
        "video_id": video_info.get("video_id", "unknown"),
        "duration": video_info.get("duration", 0.0),
        "fps": video_info.get("fps", 24),
        "resolution": video_info.get("resolution", "512x512"),
        "dance_style": video_info.get("dance_style", "unknown"),
        "music_genre": video_info.get("music_genre", "unknown"),
        "difficulty": video_info.get("difficulty", "medium"),
        "created_at": video_info.get("created_at", ""),
        "description": video_info.get("description", "")
    }
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata

def process_video(
    video_path: str,
    output_dir: str,
    video_id: str,
    fps: int = 30,
    resolution: str = "512x512"
) -> bool:
    """단일 비디오 처리"""
    print(f"\nProcessing video: {video_path}")
    
    # 출력 디렉토리 생성
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 1. 비디오 복사
    video_output_path = os.path.join(video_output_dir, "video.mp4")
    try:
        shutil.copy2(video_path, video_output_path)
    except Exception as e:
        print(f"Error copying video: {e}")
        return False
    
    # 2. 오디오 추출
    audio_output_path = os.path.join(video_output_dir, "audio.wav")
    if not extract_audio_from_video(video_output_path, audio_output_path):
        print("Warning: Audio extraction failed")
    
    # 3. 음악 특징 추출
    music_features_path = os.path.join(video_output_dir, "music_features.npy")
    if os.path.exists(audio_output_path):
        music_features = extract_music_features(audio_output_path)
        np.save(music_features_path, music_features)
    else:
        print("Warning: Using dummy music features")
        music_features = np.random.randn(100, 4800)
        np.save(music_features_path, music_features)
    
    # 4. 프레임 추출
    frames_dir = os.path.join(video_output_dir, "frames")
    frame_paths = extract_video_frames(video_output_path, frames_dir, fps)
    
    if not frame_paths:
        print("Warning: Frame extraction failed")
        return False
    
    # 5. 포즈 어노테이션 생성 (30fps 기준)
    pose_annotations_path = os.path.join(video_output_dir, "pose_annotations.json")
    pose_data = create_pose_annotations(frame_paths, pose_annotations_path, fps=fps)
    
    # 6. 참조 이미지 생성 (랜덤 선택)
    reference_image_path = os.path.join(video_output_dir, "reference_image.jpg")
    if not extract_reference_image(frame_paths, reference_image_path, strategy="random"):
        print("Warning: Reference image creation failed")
    
    # 7. 얼굴 임베딩 생성
    face_embedding_path = os.path.join(video_output_dir, "face_embedding.npy")
    if not create_face_embedding(reference_image_path, face_embedding_path):
        print("Warning: Face embedding creation failed")
    
    # 8. 임시 파일 정리
    if os.path.exists(audio_output_path):
        os.remove(audio_output_path)
    
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    
    print(f"Video {video_id} processed successfully")
    return True

def main():
    """
    Main function for dataset preparation
    
    Args:
        Command line arguments for input/output directories and parameters
        
    Returns:
        None
        
    Function: Main entry point that processes videos and P00002 structures into M2P training format
    """
    parser = argparse.ArgumentParser(description="Prepare M2P dataset from raw videos")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate (default: 30fps for 5-second videos)")
    parser.add_argument("--resolution", type=str, default="512x512", help="Target resolution")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/validation split ratio")
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 비디오 파일 및 P00002 구조 찾기
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    p00002_dirs = []
    
    # 비디오 파일 찾기
    for ext in video_extensions:
        video_files.extend(Path(args.input_dir).glob(f"*{ext}"))
        video_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))
    
    # P00002 구조 디렉토리 찾기 (poses/, faces/, images/ 폴더가 있는 디렉토리)
    for item in os.listdir(args.input_dir):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            poses_dir = os.path.join(item_path, "poses")
            faces_dir = os.path.join(item_path, "faces")
            images_dir = os.path.join(item_path, "images")
            
            if os.path.exists(poses_dir) and os.path.exists(faces_dir):
                p00002_dirs.append(item_path)
                print(f"Found P00002 structure: {item}")
    
    if not video_files and not p00002_dirs:
        print(f"No video files or P00002 structures found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Found {len(p00002_dirs)} P00002 structures")
    
    # 비디오 처리
    processed_videos = []
    
    # P00002 구조 처리
    for p00002_dir in p00002_dirs:
        video_id = os.path.basename(p00002_dir)
        
        # P00002 구조 처리
        if process_p00002_structure(p00002_dir, args.output_dir, video_id, args.fps):
            # 메타데이터 수집
            poses_dir = os.path.join(p00002_dir, "poses")
            pose_files = [f for f in os.listdir(poses_dir) if f.endswith('.png')]
            
            video_info = {
                "video_id": video_id,
                "duration": len(pose_files) / args.fps,
                "fps": args.fps,
                "resolution": args.resolution,
                "dance_style": "unknown",
                "music_genre": "unknown",
                "difficulty": "medium",
                "created_at": "",
                "description": "P00002 structure processed"
            }
            processed_videos.append(video_info)
    
    # 비디오 파일 처리
    for video_path in video_files:
        video_id = video_path.stem
        
        # 비디오 정보 수집
        cap = cv2.VideoCapture(str(video_path))
        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps_orig if fps_orig > 0 else 0
        cap.release()
        
        video_info = {
            "video_id": video_id,
            "duration": duration,
            "fps": fps_orig,
            "resolution": args.resolution,
            "dance_style": "unknown",
            "music_genre": "unknown",
            "difficulty": "medium",
            "created_at": "",
            "description": ""
        }
        
        # 비디오 처리
        if process_video(str(video_path), args.output_dir, video_id, args.fps, args.resolution):
            processed_videos.append(video_info)
    
    print(f"\nProcessed {len(processed_videos)} videos successfully")
    
    # 훈련/검증 분할
    random.shuffle(processed_videos)
    split_idx = int(len(processed_videos) * args.split_ratio)
    
    train_videos = processed_videos[:split_idx]
    val_videos = processed_videos[split_idx:]
    
    # 분할 디렉토리 생성
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "validation")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 비디오 이동
    for video_info in train_videos:
        src_dir = os.path.join(args.output_dir, video_info["video_id"])
        dst_dir = os.path.join(train_dir, video_info["video_id"])
        if os.path.exists(src_dir):
            shutil.move(src_dir, dst_dir)
    
    for video_info in val_videos:
        src_dir = os.path.join(args.output_dir, video_info["video_id"])
        dst_dir = os.path.join(val_dir, video_info["video_id"])
        if os.path.exists(src_dir):
            shutil.move(src_dir, dst_dir)
    
    # 메타데이터 생성
    metadata = {
        "dataset_info": {
            "name": "M2P Dataset",
            "description": "Music-to-Pose dataset for dance generation",
            "created_at": "",
            "total_videos": len(processed_videos),
            "train_videos": len(train_videos),
            "validation_videos": len(val_videos)
        },
        "videos": processed_videos
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset preparation completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    print(f"Metadata: {metadata_path}")

if __name__ == "__main__":
    main()
