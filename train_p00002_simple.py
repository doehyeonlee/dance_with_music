import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

from models.m2p_encoder import M2PEncoder, PoseLoss, ArcFaceLoss
from datasets.p00002_dataset import P00002Dataset

class SimpleIntegratedModel(nn.Module):
    """간단한 통합 모델 (M2P Encoder만 포함)"""
    
    def __init__(self, m2p_encoder):
        super().__init__()
        self.m2p_encoder = m2p_encoder
        self.current_stage = "A"
    
    def forward(self, music_features, target_pose_heatmap=None, target_face_embed=None):
        """Forward pass - M2P Encoder만 사용"""
        pose_logits, face_embed = self.m2p_encoder(music_features)
        
        return {
            'pose_logits': pose_logits,
            'face_embed': face_embed,
            'target_pose_heatmap': target_pose_heatmap,
            'target_face_embed': target_face_embed
        }
    
    def set_training_stage(self, stage):
        """훈련 스테이지 설정"""
        self.current_stage = stage
        print(f"훈련 스테이지 설정: {stage}")

def create_simple_model():
    """간단한 모델 생성"""
    
    # M2P Encoder 생성
    m2p_encoder = M2PEncoder(
        music_input_dim=4800,
        hidden_dim=1024,
        pose_channels=134,
        face_embed_dim=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    # 통합 모델 생성
    model = SimpleIntegratedModel(m2p_encoder)
    
    return model

def train_single_sample(model, sample, device, num_epochs=100):
    """하나의 샘플로 훈련"""
    
    print("=== 단일 샘플 훈련 시작 ===")
    print(f"Face sequence shape: {sample['face_sequence'].shape}")
    print(f"Pose sequence shape: {sample['pose_sequence'].shape}")
    print(f"Music features shape: {sample['music_features'].shape}")
    
    # 모델을 훈련 모드로 설정
    model.train()
    
    # Stage A로 설정 (M2P Encoder만 훈련)
    model.set_training_stage("A")
    
    # 옵티마이저 생성 (M2P Encoder만 훈련)
    optimizer = torch.optim.AdamW(model.m2p_encoder.parameters(), lr=1e-4)
    
    # Loss 함수 생성
    pose_loss_fn = PoseLoss(heatmap_weight=1.0, coordinate_weight=0.5, temporal_weight=0.1)
    face_loss_fn = ArcFaceLoss(margin=0.5, scale=64.0)
    
    # 데이터를 device로 이동
    face_sequence = sample['face_sequence'].unsqueeze(0).to(device)  # [1, T, C, H, W]
    pose_sequence = sample['pose_sequence'].unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # Music features를 0으로 설정 (사용자 요청)
    music_features = torch.zeros_like(sample['music_features']).unsqueeze(0).to(device)  # [1, T, 4800]
    
    # Pose sequence를 target으로 변환
    # [1, T, C, H, W] -> [1, T, 134] (간단한 변환)
    target_pose = pose_sequence.mean(dim=(3, 4))  # 공간 차원 평균
    target_pose = target_pose[:, :, 0:134]  # 134 채널로 제한
    
    # Face embedding target 생성 (간단한 변환)
    target_face = face_sequence.mean(dim=(2, 3, 4))  # 공간 및 시간 차원 평균
    target_face = target_face[:, 0:512]  # 512 차원으로 제한
    target_face = F.normalize(target_face, p=2, dim=1)
    
    print(f"Target pose shape: {target_pose.shape}")
    print(f"Target face shape: {target_face.shape}")
    
    # 차원 문제 해결: target_pose를 134 차원으로 확장
    if target_pose.shape[-1] < 134:
        # 3차원을 134차원으로 확장
        target_pose_expanded = torch.zeros(1, args.sequence_length, 134).to(device)
        target_pose_expanded[:, :, :target_pose.shape[-1]] = target_pose
        target_pose = target_pose_expanded
    
    # target_face를 512 차원으로 확장
    if target_face.shape[-1] < 512:
        target_face_expanded = torch.zeros(1, 512).to(device)
        target_face_expanded[:, :target_face.shape[-1]] = target_face
        target_face = target_face_expanded
    
    print(f"Adjusted Target pose shape: {target_pose.shape}")
    print(f"Adjusted Target face shape: {target_face.shape}")
    
    # 훈련 루프
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            music_features=music_features,
            target_pose_heatmap=target_pose,
            target_face_embed=target_face
        )
        
        # Loss 계산 - 차원 맞춤
        # PoseLoss는 [B, T, 134] 형태의 target을 기대하지만, 
        # 실제로는 pose_logits의 차원과 맞춰야 함
        pose_logits = outputs['pose_logits']  # [1, T, 134]
        
        # PoseLoss에 맞는 형태로 target 변환
        # target_pose를 pose_logits와 같은 차원으로 맞춤
        if pose_logits.shape != target_pose.shape:
            # target_pose를 pose_logits와 같은 차원으로 확장
            target_pose_expanded = target_pose.expand_as(pose_logits)
        else:
            target_pose_expanded = target_pose
        
        pose_loss = pose_loss_fn(
            pose_logits,           # [1, T, 134]
            target_pose_expanded   # [1, T, 134]
        )
        
        # ArcFaceLoss는 [B, 512] 형태의 target을 기대하므로 첫 번째 프레임만 사용
        face_loss = face_loss_fn(
            outputs['face_embed'][:, 0, :],   # [1, 512] - 첫 번째 프레임
            torch.tensor([0]).to(device)      # 더미 라벨
        )
        
        total_loss = pose_loss + 0.1 * face_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # 로깅
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {total_loss.item():.6f} "
                  f"(Pose: {pose_loss.item():.6f}, Face: {face_loss.item():.6f})")
    
    print("=== 훈련 완료 ===")
    return model

def main():
    parser = argparse.ArgumentParser(description="P00002 데이터로 단일 샘플 훈련")
    parser.add_argument("--data_dir", type=str, default="../P00002", help="P00002 데이터 디렉토리")
    parser.add_argument("--sequence_length", type=int, default=16, help="시퀀스 길이")
    parser.add_argument("--num_epochs", type=int, default=100, help="훈련 에포크 수")
    parser.add_argument("--output_dir", type=str, default="./p00002_training_output", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터셋 생성
    dataset = P00002Dataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        stride=1
    )
    
    print(f"데이터셋 크기: {len(dataset)}")
    
    # 첫 번째 샘플만 사용
    sample = dataset[0]
    print(f"선택된 샘플: {dataset.get_sequence_info(0)}")
    
    # 모델 생성
    model = create_simple_model().to(device)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"훈련 가능한 파라미터 수: {trainable_params:,}")
    
    # 단일 샘플로 훈련
    trained_model = train_single_sample(model, sample, device, args.num_epochs)
    
    # 훈련된 모델 저장
    output_path = os.path.join(args.output_dir, "trained_model.pt")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'sample_info': dataset.get_sequence_info(0),
        'training_config': vars(args)
    }, output_path)
    
    print(f"훈련된 모델이 저장되었습니다: {output_path}")
    
    # 테스트: 훈련된 모델로 추론
    print("\n=== 훈련된 모델 테스트 ===")
    trained_model.eval()
    
    with torch.no_grad():
        # 동일한 입력으로 추론
        test_outputs = trained_model(
            music_features=torch.zeros(1, args.sequence_length, 4800).to(device),
            target_pose_heatmap=None,
            target_face_embed=None
        )
        
        print(f"테스트 출력 - Pose: {test_outputs['pose_logits'].shape}")
        print(f"테스트 출력 - Face: {test_outputs['face_embed'].shape}")
        
        # 테스트용 target 생성 (훈련과 동일한 형태)
        test_target_pose = torch.randn(1, args.sequence_length, 134).to(device)
        test_target_face = torch.randn(1, 512).to(device)
        test_target_face = F.normalize(test_target_face, p=2, dim=1)
        
        # Loss 계산 - 차원 맞춤
        test_pose_logits = test_outputs['pose_logits']
        if test_pose_logits.shape != test_target_pose.shape:
            test_target_pose_expanded = test_target_pose.expand_as(test_pose_logits)
        else:
            test_target_pose_expanded = test_target_pose
        
        test_pose_loss = PoseLoss()(test_pose_logits, test_target_pose_expanded)
        test_face_loss = ArcFaceLoss()(test_outputs['face_embed'][:, 0, :], torch.tensor([0]).to(device))
        test_total_loss = test_pose_loss + 0.1 * test_face_loss
        
        print(f"테스트 Loss: {test_total_loss.item():.6f}")

if __name__ == "__main__":
    main()
