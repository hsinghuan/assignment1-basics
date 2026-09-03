#!/bin/bash
set -e

TRAIN_DATA_PATH="$1"
VAL_DATA_PATH="$2"


uv run python -m scripts.train \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --device mps \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 16 \
    --d_ff 1344 \
    --num_layers 4 \
    --rope_theta 10000 \
    --train_steps 10000 \
    --train_batch_size 16 \
    --eval_steps 50 \
    --eval_batch_size 16 \
    --steps_per_eval 200 \
    --steps_per_log 20 \
    --steps_per_save 1000 \
    --max_learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --warmup_steps 500 \
    --cosine_cycle_steps 10000 \
    --max_grad_norm 1.0 \
    --run_name mps \
    --use_wandb