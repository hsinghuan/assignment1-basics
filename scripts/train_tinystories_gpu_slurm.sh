#!/bin/bash

#SBATCH -p gh -N 1 -n 1 -t 4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e

TRAIN_DATA_PATH="$1"
VAL_DATA_PATH="$2"
MAX_LR="${MAX_LR:-1e-3}"
MIN_LR="${MIN_LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
TRAIN_STEPS="${TRAIN_STEPS:-10000}"
WARMUP_STEPS=$(( TRAIN_STEPS / 20 ))
COSINE_CYCLE_STEPS=$TRAIN_STEPS
STEPS_PER_SAVE=$(( TRAIN_STEPS / 10 ));  (( STEPS_PER_SAVE < 1 )) && STEPS_PER_SAVE=1
STEPS_PER_EVAL=$(( TRAIN_STEPS / 50 ));  (( STEPS_PER_EVAL < 1 )) && STEPS_PER_EVAL=1
STEPS_PER_LOG=$(( TRAIN_STEPS / 500 ));  (( STEPS_PER_LOG < 1 )) && STEPS_PER_LOG=1
RUN_NAME="tinystories_bs${BATCH_SIZE}_lr${MAX_LR}"

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
    --train_steps "$TRAIN_STEPS" \
    --train_batch_size "$BATCH_SIZE" \
    --eval_steps 50 \
    --eval_batch_size "$BATCH_SIZE" \
    --steps_per_eval "$STEPS_PER_EVAL" \
    --steps_per_log "$STEPS_PER_LOG" \
    --steps_per_save "$STEPS_PER_SAVE" \
    --max_learning_rate "$MAX_LR" \
    --min_learning_rate "$MIN_LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --cosine_cycle_steps "$COSINE_CYCLE_STEPS" \
    --max_grad_norm 1.0 \
    --run_name "$RUN_NAME" \
    --use_wandb
