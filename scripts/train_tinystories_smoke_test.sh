#!/bin/bash
set -e   # stop on first error — good for a smoke test

TRAIN_DATA_PATH="$1"
VAL_DATA_PATH="${2:-$1}"   # default val = train file, fine for a smoke test

uv run python -m scripts.train \
    --train_data_path "$TRAIN_DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --device cpu \
    --vocab_size 10000 \
    --context_length 128 \
    --d_model 64 \
    --num_heads 4 \
    --d_ff 256 \
    --num_layers 2 \
    --rope_theta 10000 \
    --train_steps 50 \
    --train_batch_size 8 \
    --eval_steps 5 \
    --eval_batch_size 8 \
    --steps_per_eval 20 \
    --steps_per_log 5 \
    --steps_per_save 25 \
    --max_learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --warmup_steps 10 \
    --cosine_cycle_steps 50 \
    --max_grad_norm 1.0
