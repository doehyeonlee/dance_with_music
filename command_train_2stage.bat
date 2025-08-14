@echo off
REM 2-Stage CHAMP Training Script: M2PEncoder + StableAnimator (Windows)

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Training parameters
set PRETRAINED_MODEL_PATH=runwayml/stable-diffusion-v1-5
set OUTPUT_DIR=./champ_2stage_output
set RESOLUTION=512
set SEQUENCE_LENGTH=24
set BATCH_SIZE=1
set NUM_EPOCHS=100

REM Stage-specific learning rates
set STAGE_A_LR=1e-4
set STAGE_B_LR=1e-4
set STAGE_C_LR=5e-5
set M2P_LR_STAGE_B=1e-5

REM Stage progression steps
set STAGE_A_STEPS=5000
set STAGE_B_STEPS=10000
set STAGE_C_STEPS=15000

REM Loss weights
set DIFFUSION_LOSS_WEIGHT=1.0
set POSE_LOSS_WEIGHT=0.5
set FACE_LOSS_WEIGHT=0.1
set ID_LOSS_WEIGHT=0.1
set TEMPORAL_LOSS_WEIGHT=0.1

REM Guidance configuration
set GUIDANCE_SCALE=1.0

echo Starting 2-Stage CHAMP Training...
echo Stage A: M2P Pretrain (%STAGE_A_STEPS% steps)
echo Stage B: Image Training (%STAGE_B_STEPS% steps)
echo Stage C: Motion Training (%STAGE_C_STEPS% steps)

REM Run 2-stage training with auto progression
python train_2stage.py ^
    --training_stage A ^
    --auto_progress ^
    --pretrained_model_name_or_path %PRETRAINED_MODEL_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --resolution %RESOLUTION% ^
    --sequence_length %SEQUENCE_LENGTH% ^
    --train_batch_size %BATCH_SIZE% ^
    --num_train_epochs %NUM_EPOCHS% ^
    --stage_a_lr %STAGE_A_LR% ^
    --stage_b_lr %STAGE_B_LR% ^
    --stage_c_lr %STAGE_C_LR% ^
    --m2p_lr_stage_b %M2P_LR_STAGE_B% ^
    --stage_a_steps %STAGE_A_STEPS% ^
    --stage_b_steps %STAGE_B_STEPS% ^
    --stage_c_steps %STAGE_C_STEPS% ^
    --diffusion_loss_weight %DIFFUSION_LOSS_WEIGHT% ^
    --pose_loss_weight %POSE_LOSS_WEIGHT% ^
    --face_loss_weight %FACE_LOSS_WEIGHT% ^
    --id_loss_weight %ID_LOSS_WEIGHT% ^
    --temporal_loss_weight %TEMPORAL_LOSS_WEIGHT% ^
    --guidance_scale %GUIDANCE_SCALE% ^
    --use_m2p_guidance ^
    --gradient_accumulation_steps 4 ^
    --lr_scheduler cosine ^
    --lr_warmup_steps 500 ^
    --save_steps 500 ^
    --validation_steps 100 ^
    --checkpointing_steps 500 ^
    --mixed_precision fp16 ^
    --report_to tensorboard ^
    --logging_dir logs

echo 2-Stage training completed!
echo Checkpoints saved in: %OUTPUT_DIR%
echo Logs saved in: logs/

pause
