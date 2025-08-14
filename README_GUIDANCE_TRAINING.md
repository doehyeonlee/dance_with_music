# CHAMP Model Guidance Training

이 문서는 CHAMP 모델에 guidance를 반영하여 훈련을 진행하는 방법을 설명합니다.

## 개요

CHAMP 모델은 reference image와 다양한 guidance 조건을 결합하여 자연스러운 pose sequence를 생성하는 모델입니다. 이 훈련 시스템은 다음과 같은 guidance 타입을 지원합니다:

- **Music Guidance**: 음악의 리듬과 템포에 맞춘 춤 동작 생성
- **Pose Guidance**: 특정 pose 스타일이나 제약 조건을 반영한 동작 생성
- **Custom Guidance**: 사용자 정의 guidance 타입 추가 가능

## 파일 구조

```
dance_with_music/
├── train_guidance.py          # 메인 훈련 스크립트
├── datasets/
│   └── guidance_dataset.py    # Guidance 데이터셋 클래스
├── command_train_guidance.sh  # Linux/Mac 훈련 스크립트
├── command_train_guidance.bat # Windows 훈련 스크립트
└── README_GUIDANCE_TRAINING.md
```

## 데이터 준비

### 1. 데이터 디렉토리 구조

```
data/
├── videos/                    # 훈련용 비디오 파일들
│   ├── dance_001.mp4
│   ├── dance_002.mp4
│   └── ...
├── poses/                     # Pose guidance 데이터
│   ├── dance_001/
│   │   ├── frame_000.json
│   │   ├── frame_001.json
│   │   └── ...
│   └── dance_002/
│       └── ...
├── music_features/            # Music guidance 데이터
│   ├── dance_001/
│   │   ├── frame_000.npy
│   │   ├── frame_001.npy
│   │   └── ...
│   └── dance_002/
│       └── ...
└── reference_images/          # Reference 이미지들
    ├── dance_001.jpg
    ├── dance_002.jpg
    └── ...
```

### 2. 데이터 형식

#### Pose Data (JSON)
```json
{
    "keypoints": [
        [x1, y1, confidence1],
        [x2, y2, confidence2],
        ...
    ],
    "pose_type": "dance",
    "frame_id": 0
}
```

#### Music Features (NPY)
- Shape: `(feature_dim,)` 또는 `(feature_dim, height, width)`
- Feature dimension은 `guidance_input_channels`와 일치해야 함

#### Reference Images
- Format: JPG, PNG
- Resolution: 훈련 시 지정한 resolution과 일치해야 함

## 훈련 실행

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# CUDA 설정 (GPU 사용 시)
export CUDA_VISIBLE_DEVICES=0
```

### 2. 훈련 실행

#### Linux/Mac
```bash
chmod +x command_train_guidance.sh
./command_train_guidance.sh
```

#### Windows
```cmd
command_train_guidance.bat
```

#### 직접 실행
```bash
python train_guidance.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_dir ./champ_guidance_output \
    --train_data_dir ./data \
    --resolution 512 \
    --sequence_length 16 \
    --train_batch_size 1 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --guidance_types music pose \
    --guidance_input_channels 1 134 \
    --guidance_embedding_channels 1280
```

## 주요 파라미터

### Guidance 관련
- `--guidance_types`: 사용할 guidance 타입들 (예: "music pose")
- `--guidance_input_channels`: 각 guidance 타입의 입력 채널 수
- `--guidance_embedding_channels`: Guidance encoder의 출력 임베딩 차원

### 훈련 관련
- `--resolution`: 입력 이미지 해상도
- `--sequence_length`: 비디오 시퀀스 길이 (프레임 수)
- `--train_batch_size`: 배치 크기
- `--learning_rate`: 학습률
- `--num_train_epochs`: 훈련 에포크 수

## 훈련 과정

### 1. Forward Pass
1. **Reference Processing**: Reference image를 CLIP과 VAE로 인코딩
2. **Video Encoding**: 훈련 비디오를 VAE로 인코딩하여 latents 생성
3. **Guidance Processing**: 각 guidance 타입을 해당 encoder로 처리
4. **CHAMP Forward**: Guidance가 반영된 CHAMP 모델의 forward pass

### 2. Loss 계산
- **MSE Loss**: 모델 예측과 실제 video latents 간의 차이
- **Guidance Loss**: 필요시 추가적인 guidance 관련 loss 추가 가능

### 3. Backward Pass
- Gradient 계산 및 모델 파라미터 업데이트
- Gradient clipping으로 훈련 안정성 확보

## 모니터링

### 1. TensorBoard
```bash
tensorboard --logdir logs
```

### 2. 주요 메트릭
- `train_loss`: 훈련 손실
- `lr`: 학습률
- `epoch`: 현재 에포크
- `step`: 현재 스텝

## 체크포인트 및 모델 저장

- **Checkpoint**: `--save_steps`마다 저장
- **Final Model**: 훈련 완료 후 최종 모델 저장
- **Output Directory**: `--output_dir`에 모든 결과 저장

## 커스터마이징

### 1. 새로운 Guidance 타입 추가

```python
# guidance_encoder.py에 새로운 encoder 클래스 추가
class CustomGuidanceEncoder(nn.Module):
    def __init__(self, guidance_embedding_channels, guidance_input_channels):
        super().__init__()
        # Custom encoder implementation
        pass
    
    def forward(self, condition):
        # Custom forward pass
        pass

# train_guidance.py의 create_guidance_encoders 함수 수정
def create_guidance_encoders(guidance_types, guidance_input_channels, guidance_embedding_channels):
    guidance_encoder_group = {}
    
    for guidance_type, input_channels in zip(guidance_types, guidance_input_channels):
        if guidance_type == "custom":
            guidance_encoder = CustomGuidanceEncoder(
                guidance_embedding_channels=guidance_embedding_channels,
                guidance_input_channels=input_channels
            )
        # ... 기존 코드 ...
    
    return guidance_encoder_group
```

### 2. Loss 함수 커스터마이징

```python
# train_guidance.py의 loss 계산 부분 수정
def custom_loss(model_pred, target, guidance_condition):
    # 기본 MSE loss
    mse_loss = F.mse_loss(model_pred, target)
    
    # 추가적인 guidance loss
    guidance_loss = calculate_guidance_loss(model_pred, guidance_condition)
    
    # Loss 결합
    total_loss = mse_loss + 0.1 * guidance_loss
    
    return total_loss
```

## 문제 해결

### 1. 메모리 부족
- `--train_batch_size` 감소
- `--gradient_accumulation_steps` 증가
- `--mixed_precision fp16` 사용

### 2. 훈련 불안정
- `--learning_rate` 감소
- `--lr_warmup_steps` 증가
- Gradient clipping 값 조정

### 3. Guidance 효과 부족
- Guidance encoder 구조 검토
- Loss 가중치 조정
- Guidance 데이터 품질 확인

## 성능 최적화

### 1. 데이터 로딩
- `--num_workers` 증가
- SSD 사용으로 I/O 성능 향상
- 데이터 캐싱 활용

### 2. 모델 최적화
- Mixed precision training 사용
- Gradient checkpointing 활성화
- XFormers attention processor 사용

## 참고 자료

- [CHAMP Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
