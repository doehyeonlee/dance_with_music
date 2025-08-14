# M2P Dataset Preparation Guide

이 문서는 **M2P (Music-to-Pose)** 훈련을 위한 데이터셋 준비 방법을 설명합니다.

## 📁 데이터셋 구조

### **최종 데이터셋 구조**
```
dataset/
├── train/
│   ├── video_001/
│   │   ├── video.mp4                    # 원본 비디오 (24fps, 512x512)
│   │   ├── music_features.npy           # 음악 특징 (T, 4800)
│   │   ├── pose_annotations.json        # 포즈 어노테이션 (T, 134)
│   │   ├── reference_image.jpg          # 참조 이미지 (512x512)
│   │   └── face_embedding.npy           # ArcFace 임베딩 (512)
│   ├── video_002/
│   │   └── ...
│   └── ...
├── validation/
│   ├── video_101/
│   │   └── ...
│   └── ...
└── metadata.json                         # 데이터셋 메타데이터
```

### **필수 파일 설명**

#### **1. video.mp4**
- **형식**: MP4 비디오
- **해상도**: 512x512 (훈련용)
- **프레임률**: 24fps
- **길이**: 최소 1초 이상 (24프레임 이상)

#### **2. music_features.npy**
- **형식**: NumPy 배열 (.npy)
- **차원**: `[T, 4800]` (T = 시퀀스 길이)
- **내용**: 음악의 다양한 특징을 결합
  - MFCC (13차원)
  - Chroma (12차원)
  - Spectral Contrast (7차원)
  - Tonnetz (6차원)
  - Tempo (1차원)
  - 기타 특징들 (총 4800차원)

#### **3. pose_annotations.json**
- **형식**: JSON 파일
- **구조**: 각 프레임별 포즈 키포인트 정보
- **키포인트**: 134개 관절 (COCO + 추가 관절)
- **정보**: x, y 좌표, 신뢰도, 관절 타입

#### **4. reference_image.jpg**
- **형식**: JPEG 이미지
- **해상도**: 512x512
- **용도**: 얼굴 ID 참조, 스타일 참조
- **내용**: 춤추는 사람의 얼굴이 명확히 보이는 이미지

#### **5. face_embedding.npy**
- **형식**: NumPy 배열 (.npy)
- **차원**: `[512]`
- **내용**: ArcFace 호환 얼굴 임베딩
- **특징**: L2 정규화된 512차원 벡터

## 🚀 데이터셋 준비 방법

### **방법 1: 자동 스크립트 사용 (권장)**

#### **1. 스크립트 실행**
```bash
cd dance_with_music/scripts

# 기본 실행
python prepare_dataset.py \
    --input_dir /path/to/raw/videos \
    --output_dir /path/to/output/dataset

# 상세 옵션
python prepare_dataset.py \
    --input_dir /path/to/raw/videos \
    --output_dir /path/to/output/dataset \
    --fps 24 \
    --resolution 512x512 \
    --split_ratio 0.8
```

#### **2. 스크립트 기능**
- **비디오 처리**: 다양한 형식 지원 (.mp4, .avi, .mov, .mkv, .wmv)
- **오디오 추출**: 비디오에서 오디오 분리
- **음악 특징 추출**: librosa를 사용한 자동 특징 추출
- **프레임 추출**: FFmpeg를 사용한 고품질 프레임 추출
- **포즈 어노테이션**: 더미 데이터 생성 (실제 구현에서는 MediaPipe/OpenPose 사용)
- **참조 이미지**: 첫 번째 프레임에서 자동 추출
- **얼굴 임베딩**: 더미 데이터 생성 (실제 구현에서는 InsightFace 사용)
- **자동 분할**: 훈련/검증 데이터 자동 분할

### **방법 2: 수동 준비**

#### **1. 음악 특징 추출**
```python
import librosa
import numpy as np

def extract_music_features(audio_path, target_dim=4800):
    # 오디오 로드
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 특징 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_features = np.full((1, mfcc.shape[1]), tempo)
    
    # 결합
    features = np.vstack([mfcc, chroma, contrast, tonnetz, tempo_features])
    
    # 차원 조정
    if features.shape[0] < target_dim:
        padding = np.zeros((target_dim - features.shape[0], features.shape[1]))
        features = np.vstack([features, padding])
    
    return features.T  # [T, 4800]

# 사용 예시
features = extract_music_features("audio.wav")
np.save("music_features.npy", features)
```

#### **2. 포즈 어노테이션 생성**
```python
import json
import numpy as np

def create_pose_annotations(video_path, output_path):
    # MediaPipe 또는 OpenPose로 포즈 추정
    # 여기서는 더미 데이터 생성
    
    poses = []
    for frame_id in range(24):  # 24프레임 가정
        keypoints = []
        for joint_id in range(134):
            keypoint = {
                "x": np.random.randint(0, 512),
                "y": np.random.randint(0, 512),
                "confidence": np.random.uniform(0.5, 1.0),
                "joint_type": f"joint_{joint_id}"
            }
            keypoints.append(keypoint)
        
        pose = {
            "frame_id": frame_id,
            "timestamp": frame_id / 24.0,
            "keypoints": keypoints
        }
        poses.append(pose)
    
    pose_data = {
        "video_id": "video_001",
        "fps": 24,
        "duration": 1.0,
        "poses": poses
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pose_data, f, indent=2, ensure_ascii=False)

# 사용 예시
create_pose_annotations("video.mp4", "pose_annotations.json")
```

#### **3. 얼굴 임베딩 생성**
```python
import numpy as np
from insightface.app import FaceAnalysis

def extract_face_embedding(image_path, output_path):
    # InsightFace 모델 로드
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 이미지에서 얼굴 감지
    faces = app.get(image_path)
    
    if len(faces) > 0:
        # 첫 번째 얼굴의 임베딩 추출
        embedding = faces[0].embedding  # [512]
        
        # L2 정규화
        embedding = embedding / np.linalg.norm(embedding)
        
        np.save(output_path, embedding)
        return True
    else:
        # 얼굴이 감지되지 않은 경우 랜덤 임베딩
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        np.save(output_path, embedding)
        return False

# 사용 예시
extract_face_embedding("reference_image.jpg", "face_embedding.npy")
```

## 📋 데이터 품질 요구사항

### **비디오 품질**
- **해상도**: 최소 512x512 (높을수록 좋음)
- **프레임률**: 24fps 이상
- **길이**: 최소 1초, 권장 3-10초
- **내용**: 춤추는 사람이 명확히 보이는 영상

### **음악 품질**
- **오디오 품질**: 44.1kHz 이상
- **노이즈**: 최소화
- **길이**: 비디오와 동일
- **장르**: 다양한 장르 포함 권장

### **포즈 어노테이션 품질**
- **정확도**: 높은 신뢰도 (>0.8)
- **일관성**: 프레임 간 연속성 유지
- **완성도**: 134개 관절 모두 포함

### **참조 이미지 품질**
- **얼굴**: 명확하게 보이는 얼굴
- **해상도**: 512x512
- **조명**: 균등한 조명
- **각도**: 정면 또는 약간의 측면

## 🔧 필요한 도구들

### **Python 패키지**
```bash
pip install librosa opencv-python pillow numpy scipy
```

### **시스템 도구**
- **FFmpeg**: 비디오/오디오 처리
- **CUDA**: GPU 가속 (선택사항)

### **포즈 추정 모델 (선택사항)**
- **MediaPipe**: Google의 경량 포즈 추정
- **OpenPose**: CMU의 정확한 포즈 추정
- **HRNet**: 고품질 포즈 추정

### **얼굴 인식 모델 (선택사항)**
- **InsightFace**: 고품질 얼굴 임베딩
- **ArcFace**: 정확한 얼굴 인식
- **FaceNet**: Google의 얼굴 인식

## 📊 데이터셋 검증

### **자동 검증 스크립트**
```python
from datasets.m2p_dataset import M2PDataset

# 데이터셋 로드 테스트
dataset = M2PDataset(
    data_root="./dataset",
    split="train",
    sequence_length=24,
    resolution=512
)

# 첫 번째 샘플 확인
sample = dataset[0]
print("Sample keys:", sample.keys())
print("Music features shape:", sample['music_features'].shape)
print("Pose heatmap shape:", sample['pose_heatmap'].shape)
print("Reference image shape:", sample['reference_image'].shape)
print("Face embedding shape:", sample['face_embedding'].shape)
print("Video frames shape:", sample['video_frames'].shape)
```

### **수동 검증 체크리스트**
- [ ] 모든 비디오가 정상적으로 로드되는가?
- [ ] 음악 특징이 올바른 차원(4800)을 가지는가?
- [ ] 포즈 어노테이션이 모든 프레임에 대해 존재하는가?
- [ ] 참조 이미지가 명확한 얼굴을 보여주는가?
- [ ] 얼굴 임베딩이 512차원인가?
- [ ] 훈련/검증 분할이 적절한가?

## 🚨 문제 해결

### **일반적인 문제들**

#### **1. 메모리 부족**
```bash
# 배치 크기 줄이기
--batch_size 1

# 시퀀스 길이 줄이기
--sequence_length 16

# 해상도 줄이기
--resolution 256x256
```

#### **2. 포즈 추정 실패**
- **MediaPipe 사용**: 더 안정적이고 빠름
- **신뢰도 임계값 조정**: 낮은 신뢰도 키포인트 필터링
- **프레임 건너뛰기**: 문제가 있는 프레임 제외

#### **3. 얼굴 감지 실패**
- **이미지 품질 향상**: 조명, 각도 개선
- **여러 얼굴**: 가장 큰 얼굴 선택
- **얼굴이 없는 경우**: 랜덤 임베딩 생성

#### **4. 음악 특징 추출 실패**
- **오디오 품질**: 노이즈 제거, 정규화
- **파일 형식**: WAV, MP3 등 지원 형식 사용
- **길이**: 너무 짧은 오디오는 패딩

## 📈 데이터셋 확장 팁

### **다양성 증가**
- **춤 스타일**: 다양한 춤 장르 포함
- **음악 장르**: 팝, 힙합, 클래식 등
- **난이도**: 초급, 중급, 고급
- **인종/성별**: 다양한 인구통계학적 특성

### **품질 향상**
- **고해상도**: 1024x1024 이상
- **고프레임률**: 60fps 이상
- **다중 각도**: 여러 카메라 각도
- **다중 조명**: 다양한 조명 조건

### **메타데이터 풍부화**
- **춤 스타일**: 정확한 춤 장르 분류
- **음악 정보**: 아티스트, 앨범, 년도
- **기술적 정보**: 카메라 설정, 조명 정보
- **태그**: 감정, 분위기, 테마

이 가이드를 따라 데이터셋을 준비하면 M2P 모델 훈련을 위한 고품질 데이터를 확보할 수 있습니다.
