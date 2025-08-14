# M2P Encoder 차원 변환 가이드

## 개요

M2P (Music-to-Pose) Encoder는 음악 특징을 포즈 히트맵과 얼굴 임베딩으로 변환하는 핵심 모듈입니다. 이 문서는 입력부터 출력까지의 차원 변환 과정을 상세히 설명합니다.

## 차원 변환 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                    M2P ENCODER ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [B, T, 4800]                                           │
│     ↓                                                           │
│  Music Projection: 4800 → 2048 → 1024                          │
│     ↓                                                           │
│  Positional Encoding: [B, T, 1024]                             │
│     ↓                                                           │
│  Transformer Encoder: [B, T, 1024] → [B, T, 1024]             │
│     ↓                                                           │
│  ┌─────────────────────┬─────────────────────────────────────┐ │
│  │   Pose Head         │           Face Head                 │ │
│  │   1024 → 512 → 256 → 134 │  1024 → 512                   │ │
│  │   [B, T, 134]      │          [B, 512]                   │ │
│  └─────────────────────┴─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 상세 차원 변환

### 1. 입력 단계
- **음악 특징**: `[B, T, 4800]`
  - `B`: 배치 크기 (기본값: 2)
  - `T`: 시퀀스 길이 (기본값: 24)
  - `4800`: 음악 특징 차원 (MFCC, Chroma, Spectral Contrast, Tonnetz, Tempo)

### 2. 음악 특징 투영
```python
# 4800 → 2048 → 1024
self.music_proj = nn.Sequential(
    nn.Linear(4800, 2048),      # 첫 번째 투영
    nn.LayerNorm(2048),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(2048, 1024),      # 두 번째 투영
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Dropout(0.1)
)
```
- **출력**: `[B, T, 1024]`

### 3. 위치 인코딩
- **입력**: `[B, T, 1024]`
- **위치 인코딩**: `[T, 1024]` (sinusoidal)
- **결과**: `[B, T, 1024]`

### 4. Transformer 인코더
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=1024,           # 입력/출력 차원
    nhead=8,                # 어텐션 헤드 수
    dim_feedforward=4096,   # 피드포워드 차원 (1024 * 4)
    dropout=0.1,
    activation='gelu',
    batch_first=True
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
```
- **입력**: `[B, T, 1024]`
- **출력**: `[B, T, 1024]`

### 5. 포즈 예측 헤드
```python
self.pose_head = nn.Sequential(
    nn.Linear(1024, 512),       # 1024 → 512
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 256),        # 512 → 256
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 134)         # 256 → 134 (포즈 관절 수)
)
```
- **입력**: `[B, T, 1024]`
- **출력**: `[B, T, 134]`

### 6. 얼굴 임베딩 헤드
```python
self.face_head = nn.Sequential(
    nn.Linear(1024, 512),       # 1024 → 512
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(512, 512)         # 512 → 512 (최종 얼굴 임베딩)
)
```
- **입력**: `[B, T, 1024]` → 시간 차원 평균 풀링 → `[B, 1024]`
- **출력**: `[B, 512]`

## 출력 형태

### 포즈 히트맵
- **차원**: `[B, T, 134]`
- **의미**: 각 시점에서 134개 관절의 확률 분포
- **변환 방법**: 
  ```python
  pose_heatmap = F.softmax(pose_logits / temperature, dim=-1)
  ```

### 포즈 좌표
- **차원**: `[B, T, 134, 2]`
- **의미**: 각 시점에서 134개 관절의 (x, y) 좌표
- **변환 방법**: soft-argmax를 사용하여 히트맵을 좌표로 변환

### 얼굴 임베딩
- **차원**: `[B, 512]`
- **의미**: 배치 내 각 샘플의 얼굴 정체성 표현
- **정규화**: L2 정규화로 ArcFace 손실과 호환

## 차원 검증

### 입력 검증
```python
if music_features.size(-1) != self.music_input_dim:
    raise ValueError(f"Expected music features dimension {self.music_input_dim}, got {music_features.size(-1)}")
```

### 출력 검증
```python
assert pose_logits.shape == (batch_size, seq_len, 134)
assert face_embed.shape == (batch_size, 512)
```

## 사용 예시

### 기본 사용
```python
# M2P Encoder 생성
m2p_encoder = M2PEncoder(
    music_input_dim=4800,
    hidden_dim=1024,
    pose_channels=134,
    face_embed_dim=512
)

# 입력 데이터
music_features = torch.randn(2, 24, 4800)  # [B=2, T=24, 4800]

# 순전파
pose_logits, face_embed = m2p_encoder(music_features)

# 출력 확인
print(f"Pose: {pose_logits.shape}")      # torch.Size([2, 24, 134])
print(f"Face: {face_embed.shape}")       # torch.Size([2, 512])
```

### 포즈 변환
```python
# 히트맵으로 변환
pose_heatmap = m2p_encoder.get_pose_heatmap(pose_logits)
print(f"Heatmap: {pose_heatmap.shape}")  # torch.Size([2, 24, 134])

# 좌표로 변환
pose_coords = m2p_encoder.get_pose_coordinates(pose_logits)
print(f"Coordinates: {pose_coords.shape}")  # torch.Size([2, 24, 134, 2])
```

## 주의사항

1. **입력 차원**: 음악 특징은 반드시 4800차원이어야 합니다.
2. **배치 처리**: 배치 크기와 시퀀스 길이는 유연하게 설정 가능합니다.
3. **디바이스**: 모든 텐서는 동일한 디바이스에 있어야 합니다.
4. **메모리**: 긴 시퀀스의 경우 메모리 사용량에 주의하세요.

## 성능 최적화

1. **혼합 정밀도**: `torch.float16` 사용으로 메모리 절약
2. **그래디언트 체크포인팅**: 긴 시퀀스에서 메모리 효율성 향상
3. **배치 크기**: GPU 메모리에 맞게 배치 크기 조정

## 문제 해결

### 차원 불일치 오류
```python
# 오류: Expected music features dimension 4800, got 4096
# 해결: 음악 특징 추출 시 4800차원으로 맞춰주세요
```

### 메모리 부족
```python
# 해결: 배치 크기나 시퀀스 길이를 줄이세요
# 또는 그래디언트 체크포인팅을 사용하세요
```

### 학습 불안정
```python
# 해결: 학습률을 낮추거나 드롭아웃을 조정하세요
# 또는 배치 정규화를 추가하세요
```
