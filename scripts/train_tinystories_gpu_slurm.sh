#!/bin/bash

#SBATCH -p gh -N 1 -n 1 -t 2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

TRAIN_DATA_PATH="$1"
VAL_DATA_PATH="$2"
MAX_LR="${MAX_LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_NAME="tinystories_bs${TRAIN_BATCH_SIZE}_lr${MAX_LR}"

uv run python -m scripts.train \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --device cuda \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 16 \
    --d_ff 1344 \
    --num_layers 4 \
    --rope_theta 10000 \
    --train_steps 10000 \
    --train_batch_size "$BATCH_SIZE" \
    --eval_steps 50 \
    --eval_batch_size "$BATCH_SIZE" \
    --steps_per_eval 200 \
    --steps_per_log 20 \
    --steps_per_save 1000 \
    --max_learning_rate "$MAX_LR" \
    --min_learning_rate "$MIN_LR" \
    --warmup_steps 500 \
    --cosine_cycle_steps 10000 \
    --max_grad_norm 1.0 \
    --run_name "$RUN_NAME" \
    --use_wandb