#!/bin/bash

# Training script for CHAMP model with guidance

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
PRETRAINED_MODEL_PATH="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="./champ_guidance_output"
TRAIN_DATA_DIR="./data"  # Set your data directory path
RESOLUTION=512
SEQUENCE_LENGTH=16
BATCH_SIZE=1
NUM_EPOCHS=100
LEARNING_RATE=1e-4

# Guidance configuration
GUIDANCE_TYPES=("music" "pose")
GUIDANCE_INPUT_CHANNELS=(1 134)
GUIDANCE_EMBEDDING_CHANNELS=1280

# Run training
python train_guidance.py \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --train_data_dir $TRAIN_DATA_DIR \
    --resolution $RESOLUTION \
    --sequence_length $SEQUENCE_LENGTH \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --guidance_types ${GUIDANCE_TYPES[@]} \
    --guidance_input_channels ${GUIDANCE_INPUT_CHANNELS[@]} \
    --guidance_embedding_channels $GUIDANCE_EMBEDDING_CHANNELS \
    --gradient_accumulation_steps 4 \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --save_steps 500 \
    --validation_steps 100 \
    --checkpointing_steps 500 \
    --mixed_precision fp16 \
    --report_to tensorboard \
    --logging_dir logs
