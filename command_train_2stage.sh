#!/bin/bash

# 2-Stage CHAMP Training Script: M2PEncoder + StableAnimator

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
PRETRAINED_MODEL_PATH="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="./champ_2stage_output"
RESOLUTION=512
SEQUENCE_LENGTH=24
BATCH_SIZE=1
NUM_EPOCHS=100

# Stage-specific learning rates
STAGE_A_LR=1e-4      # M2P pretrain
STAGE_B_LR=1e-4      # Image stage
STAGE_C_LR=5e-5      # Motion stage
M2P_LR_STAGE_B=1e-5  # M2P LR in Stage B

# Stage progression steps
STAGE_A_STEPS=5000   # M2P pretrain steps
STAGE_B_STEPS=10000  # Image stage steps
STAGE_C_STEPS=15000  # Motion stage steps

# Loss weights
DIFFUSION_LOSS_WEIGHT=1.0
POSE_LOSS_WEIGHT=0.5
FACE_LOSS_WEIGHT=0.1
ID_LOSS_WEIGHT=0.1
TEMPORAL_LOSS_WEIGHT=0.1

# Guidance configuration
GUIDANCE_SCALE=1.0
USE_M2P_GUIDANCE=true

echo "Starting 2-Stage CHAMP Training..."
echo "Stage A: M2P Pretrain (${STAGE_A_STEPS} steps)"
echo "Stage B: Image Training (${STAGE_B_STEPS} steps)"
echo "Stage C: Motion Training (${STAGE_C_STEPS} steps)"

# Run 2-stage training with auto progression
python train_2stage.py \
    --training_stage A \
    --auto_progress \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --resolution $RESOLUTION \
    --sequence_length $SEQUENCE_LENGTH \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --stage_a_lr $STAGE_A_LR \
    --stage_b_lr $STAGE_B_LR \
    --stage_c_lr $STAGE_C_LR \
    --m2p_lr_stage_b $M2P_LR_STAGE_B \
    --stage_a_steps $STAGE_A_STEPS \
    --stage_b_steps $STAGE_B_STEPS \
    --stage_c_steps $STAGE_C_STEPS \
    --diffusion_loss_weight $DIFFUSION_LOSS_WEIGHT \
    --pose_loss_weight $POSE_LOSS_WEIGHT \
    --face_loss_weight $FACE_LOSS_WEIGHT \
    --id_loss_weight $ID_LOSS_WEIGHT \
    --temporal_loss_weight $TEMPORAL_LOSS_WEIGHT \
    --guidance_scale $GUIDANCE_SCALE \
    --use_m2p_guidance \
    --gradient_accumulation_steps 4 \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --save_steps 500 \
    --validation_steps 100 \
    --checkpointing_steps 500 \
    --mixed_precision fp16 \
    --report_to tensorboard \
    --logging_dir logs

echo "2-Stage training completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "Logs saved in: logs/"
