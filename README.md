# 2-Stage CHAMP Training: M2PEncoder + StableAnimator

이 문서는 **M2PEncoder + StableAnimator**를 통합하여 2-stage 학습을 진행하는 방법을 설명합니다. CHAMP 논문의 학습 절차를 참고하여 재구성된 시스템입니다.

## 🎯 학습 전략 개요

### **핵심 아이디어**
1. **Stage A**: M2PEncoder 단독 사전학습으로 음악→포즈 변환 안정화
2. **Stage B**: 이미지 중심 학습으로 ID/포즈 조건 수용 능력 향상
3. **Stage C**: 모션/시간 일관성 중심으로 Temporal 모듈 최적화

### **학습 흐름**
```
Stage A (M2P Pretrain) → Stage B (Image Training) → Stage C (Motion Training)
     ↓                        ↓                        ↓
음악→포즈 안정화        Guidance/UNet 적응        Temporal 모듈 최적화
```

## 📁 파일 구조

```
dance_with_music/
├── models/
│   ├── m2p_encoder.py              # Music-to-Pose Encoder
│   ├── integrated_champ_model.py    # 통합 CHAMP 모델
│   ├── guidance_encoder.py          # Guidance Encoder
│   └── ...
├── train_2stage.py                  # 2-stage 훈련 메인 스크립트
├── command_train_2stage.sh          # Linux/Mac 실행 스크립트
├── command_train_2stage.bat         # Windows 실행 스크립트
└── README_2STAGE_TRAINING.md        # 이 문서
```

## 🚀 학습 실행 방법

### **1. 자동 단계 진행 (권장)**
```bash
# Linux/Mac
chmod +x command_train_2stage.sh
./command_train_2stage.sh

# Windows
command_train_2stage.bat
```

### **2. 수동 단계별 실행**
```bash
# Stage A: M2P Pretrain
python train_2stage.py --training_stage A --stage_a_steps 5000

# Stage B: Image Training
python train_2stage.py --training_stage B --stage_b_steps 10000

# Stage C: Motion Training
python train_2stage.py --training_stage C --stage_c_steps 15000
```

## 📊 단계별 상세 학습 전략

### **Stage A: M2PEncoder 단독 사전학습**

#### **목표**
- 음악→포즈 히트맵/좌표 안정적 예측
- ArcFace 호환 얼굴 임베딩 생성

#### **입력/출력**
- **입력**: 오디오 윈도우 `A`, 참조 얼굴 이미지 `I_ref`
- **GT**: DWpose 히트맵/좌표, ArcFace 임베딩 (512-d, L2-norm)

#### **손실 함수**
```python
L_total = L_heat + L_face

# Pose 히트맵 손실
L_heat = CrossEntropy(pose_logits, target_heatmap) + 
          L1(soft_argmax(pred), target_coordinates)

# Face 임베딩 손실
L_face = 1 - cos(E_pred, E_ArcFace)  # 또는 L2
```

#### **Freeze 전략**
- **Freeze**: ArcFace (완전 고정), VAE, CLIP
- **Train**: M2PEncoder만 학습

#### **학습률**: `1e-4`

---

### **Stage B: 결합 이미지 단계**

#### **목표**
- StableAnimator UNet이 **ID/포즈 조건을 수용**하도록 적응
- Guidance 경로와 ReferenceNet 최적화

#### **입력/출력**
- **입력**: `face_emb`(M2P), `heatmap`(M2P), 참조 이미지
- **출력**: 조건부 비디오 생성

#### **손실 함수**
```python
L_total = L_diff + L_id + L_heat

# 확산 손실
L_diff = MSE(model_pred, target_latents)

# Identity 손실 (얼굴 마스크 가중)
L_id = ArcFace/CLIP-ID loss (face_mask_weighted)

# Pose 손실
L_heat = CrossEntropy(pose_pred, target_pose)
```

#### **Freeze 전략**
- **Freeze**: VAE 인코더/디코더, CLIP 이미지 인코더
- **Train**: Guidance 경로, Denoising UNet, ReferenceNet
- **M2PEncoder**: 처음에는 **고정** → 수렴 후 **낮은 LR**로 풀기

#### **학습률**: 
- Guidance/UNet: `1e-4`
- M2PEncoder: `1e-5` (후반부)

---

### **Stage C: 모션 단계**

#### **목표**
- 시간 일관성/리듬 강화
- Temporal Attention/모듈 최적화

#### **입력/출력**
- **입력**: T=24~150 프레임 시퀀스, `face_emb`, `heatmap`(M2P)
- **출력**: 시간 일관성 있는 비디오

#### **손실 함수**
```python
L_total = L_diff + L_temp + L_id

# 확산 손실
L_diff = MSE(model_pred, target_latents)

# Temporal 일관성 손실
L_temp = Flow/차분 정합 손실

# Identity 손실 (얼굴 가중↑)
L_id = Face-weighted identity loss
```

#### **Freeze 전략**
- **Freeze**: Stage B에서 학습한 Guidance/UNet/ReferenceNet
- **Train**: Temporal 모듈만 학습
- **M2PEncoder**: 기본 **고정**, 필요시 **소LR**로 동결 해제

#### **학습률**: `5e-5`

## 🔧 주요 파라미터 설정

### **학습률 설정**
```bash
--stage_a_lr 1e-4        # Stage A: M2P pretrain
--stage_b_lr 1e-4        # Stage B: Image training
--stage_c_lr 5e-5        # Stage C: Motion training
--m2p_lr_stage_b 1e-5   # M2P LR in Stage B
```

### **손실 가중치**
```bash
--diffusion_loss_weight 1.0    # 확산 손실
--pose_loss_weight 0.5         # 포즈 손실
--face_loss_weight 0.1         # 얼굴 손실
--id_loss_weight 0.1           # Identity 손실
--temporal_loss_weight 0.1     # Temporal 손실
```

### **단계별 스텝 수**
```bash
--stage_a_steps 5000    # M2P pretrain
--stage_b_steps 10000   # Image training
--stage_c_steps 15000   # Motion training
```

## 🎛️ Freeze 매트릭스 (모듈별)

| 모듈 | Stage A | Stage B | Stage C |
|------|---------|---------|---------|
| **M2PEncoder** | 🟢 Train | 🟡 Low LR | 🔴 Freeze |
| **ArcFace** | 🔴 Freeze | 🔴 Freeze | 🔴 Freeze |
| **Pose Adapter/Control** | — | 🟢 Train | 🔴 Freeze |
| **UNet(Spatial)** | — | 🟢 Train | 🔴 Freeze |
| **Temporal 모듈** | — | — | 🟢 Train |
| **ReferenceNet** | — | 🟢 Train | 🔴 Freeze |
| **VAE Enc/Dec** | — | 🔴 Freeze | 🔴 Freeze |
| **CLIP Image Encoder** | — | 🔴 Freeze | 🔴 Freeze |

**🟢 Train**: 활성 학습, **🟡 Low LR**: 낮은 학습률, **🔴 Freeze**: 동결

## 💡 실전 운용 팁

### **1. Scheduled Sampling**
```python
# Stage B→C 전환 시
if stage == "B" and step > stage_b_steps * 0.8:
    # 점진적으로 M2P 예측으로 치환
    guidance_ratio = min(1.0, (step - stage_b_steps * 0.8) / (stage_b_steps * 0.2))
    guidance = guidance_ratio * m2p_pred + (1 - guidance_ratio) * gt_pose
```

### **2. Condition Scale 스케줄**
```python
# 학습 초반 포즈 강제력 점진적 상승
guidance_scale = min(1.5, 0.5 + step / (total_steps * 0.3))
```

### **3. Detach 활용**
```python
# 초기 E2E 결합 시 역전파 차단
with torch.no_grad():
    m2p_output = m2p_encoder(music_features)
    guidance_condition = m2p_output['pose_heatmap'].detach()
```

### **4. ID 안정화**
```python
# 얼굴 마스크 가중
face_mask = create_face_mask(reference_image)
id_loss = id_loss * face_mask * 3.0  # 얼굴 중심 가중치
```

### **5. Temporal 배치 전략**
```python
# T=24로 빠른 수렴 → 점차 T↑
sequence_length = min(150, 24 + epoch // 10)
```

## 🔍 모니터링 및 검증

### **TensorBoard 메트릭**
```bash
tensorboard --logdir logs
```

### **주요 추적 지표**
- **Stage A**: `pose_loss`, `face_loss`
- **Stage B**: `diffusion_loss`, `pose_loss`, `id_loss`
- **Stage C**: `diffusion_loss`, `temporal_loss`, `id_loss`

### **검증 전략**
- **Stage A**: Pose 정확도, Face 임베딩 품질
- **Stage B**: 이미지 품질, ID 유지, 포즈 정확도
- **Stage C**: 시간 일관성, 모션 자연스러움

## 🚨 문제 해결

### **메모리 부족**
```bash
--train_batch_size 1
--gradient_accumulation_steps 8
--mixed_precision fp16
```

### **훈련 불안정**
```bash
--learning_rate 5e-5        # LR 감소
--lr_warmup_steps 1000      # Warmup 증가
--gradient_clip_norm 0.5    # Gradient clipping
```

### **Guidance 효과 부족**
```bash
--guidance_scale 1.5        # Guidance 강도 증가
--pose_loss_weight 1.0      # Pose loss 가중치 증가
```

### **Temporal 학습 어려움**
```bash
--temporal_loss_weight 0.2  # Temporal loss 가중치 증가
--sequence_length 48         # 시퀀스 길이 조정
```

## 📈 성능 최적화

### **데이터 로딩**
```bash
--num_workers 8              # Worker 수 증가
--pin_memory true            # GPU 메모리 활용
--prefetch_factor 2          # 프리페치 최적화
```

### **모델 최적화**
```bash
--mixed_precision fp16       # Mixed precision
--gradient_checkpointing     # 메모리 절약
--use_xformers               # XFormers attention
```

## 🔄 단계별 체크포인트 관리

### **체크포인트 구조**
```
champ_2stage_output/
├── stage_A/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
├── stage_B/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
├── stage_C/
│   ├── checkpoint-1000/
│   ├── checkpoint-2000/
│   └── ...
└── final/
```

### **체크포인트 로딩**
```python
# 특정 단계 체크포인트 로딩
accelerator.load_state("champ_2stage_output/stage_B/checkpoint-5000")

# 다음 단계로 진행
model.set_training_stage("C")
```

## 🎯 최종 정리

### **핵심 학습 순서**
1. **M2PEncoder 안정화** (Stage A)
2. **Guidance/UNet 적응** (Stage B, VAE/CLIP 동결)
3. **Temporal 모듈 최적화** (Stage C, 나머지 동결)
4. **필요시 저LR 합미세조정**

### **성공 요인**
- **단계별 명확한 목표** 설정
- **적절한 Freeze 전략** 적용
- **단계별 학습률** 조정
- **손실 가중치** 균형
- **체계적인 검증** 및 모니터링

이 2-stage 학습 시스템을 통해 **M2PEncoder + StableAnimator**가 효과적으로 통합되어, 음악에 맞춘 자연스러운 춤 동작을 생성할 수 있게 됩니다.
