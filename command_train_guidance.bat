@echo off
REM Training script for CHAMP model with guidance (Windows)

REM Set environment variables
set CUDA_VISIBLE_DEVICES=0
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Training parameters
set PRETRAINED_MODEL_PATH=runwayml/stable-diffusion-v1-5
set OUTPUT_DIR=./champ_guidance_output
set TRAIN_DATA_DIR=./data
set RESOLUTION=512
set SEQUENCE_LENGTH=16
set BATCH_SIZE=1
set NUM_EPOCHS=100
set LEARNING_RATE=1e-4

REM Guidance configuration
set GUIDANCE_TYPES=music pose
set GUIDANCE_INPUT_CHANNELS=1 134
set GUIDANCE_EMBEDDING_CHANNELS=1280

REM Run training
python train_guidance.py ^
    --pretrained_model_name_or_path %PRETRAINED_MODEL_PATH% ^
    --output_dir %OUTPUT_DIR% ^
    --train_data_dir %TRAIN_DATA_DIR% ^
    --resolution %RESOLUTION% ^
    --sequence_length %SEQUENCE_LENGTH% ^
    --train_batch_size %BATCH_SIZE% ^
    --num_train_epochs %NUM_EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --guidance_types %GUIDANCE_TYPES% ^
    --guidance_input_channels %GUIDANCE_INPUT_CHANNELS% ^
    --guidance_embedding_channels %GUIDANCE_EMBEDDING_CHANNELS% ^
    --gradient_accumulation_steps 4 ^
    --lr_scheduler constant ^
    --lr_warmup_steps 500 ^
    --save_steps 500 ^
    --validation_steps 100 ^
    --checkpointing_steps 500 ^
    --mixed_precision fp16 ^
    --report_to tensorboard ^
    --logging_dir logs

pause
