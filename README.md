# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Report

My write-up for the assignment problems (accounting, learning-rate and batch-size
experiments, etc.) is in [REPORT.md](./REPORT.md).

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

The tests connect to the implementation in [cs336_basics/](./cs336_basics/)
through the adapters in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Training

### 1. Train the BPE tokenizer

```sh
uv run python -m scripts.train_bpe_tinystories --input_path data/TinyStoriesV2-GPT4-train.txt
```

This writes `tinystories_bpe_vocab.json` and `tinystories_bpe_merges.json` to `tokenizer/`
(not tracked in git).

### 2. Tokenize the data

Run once for the train split and once for the valid split:

```sh
uv run python -m scripts.tokenize_tinystories \
    --data_path data/TinyStoriesV2-GPT4-train.txt \
    --vocab_filepath tokenizer/tinystories_bpe_vocab.json \
    --merges_filepath tokenizer/tinystories_bpe_merges.json
```

This writes `uint16` token IDs to `data_tokenized/<name>_tokenized.bin`, which the training
script loads with `np.memmap`.

### 3. Train the language model

The training entry point is [scripts/train.py](./scripts/train.py). Two launcher scripts
wrap it with the TinyStories configuration (4 layers, `d_model` 512, 16 heads, context 256):

- **Apple Silicon (mps):** [scripts/train_tinystories_mps.sh](./scripts/train_tinystories_mps.sh)

  ```sh
  bash scripts/train_tinystories_mps.sh \
      data_tokenized/TinyStoriesV2-GPT4-train_tokenized.bin \
      data_tokenized/TinyStoriesV2-GPT4-valid_tokenized.bin
  ```

- **Slurm GPU cluster (cuda):** [scripts/train_tinystories_gpu_slurm.sh](./scripts/train_tinystories_gpu_slurm.sh).
  Hyperparameters can be overridden with environment variables (`MAX_LR`, `MIN_LR`,
  `BATCH_SIZE`, `TRAIN_STEPS`). Warmup, cosine decay length, and the eval/log/save
  intervals are derived from `TRAIN_STEPS`, and the run name is built from the batch size
  and learning rate. Create `logs/` first, since Slurm writes job output there.

  ```sh
  mkdir -p logs
  BATCH_SIZE=128 TRAIN_STEPS=10000 MAX_LR=4e-3 MIN_LR=4e-4 \
      sbatch -A <allocation> scripts/train_tinystories_gpu_slurm.sh \
      data_tokenized/TinyStoriesV2-GPT4-train_tokenized.bin \
      data_tokenized/TinyStoriesV2-GPT4-valid_tokenized.bin
  ```

Checkpoints (`ckpt_<step>.pt` and `best.pt`) are saved under `result/<run_name>_<timestamp>/`.
Pass `--resume_from <checkpoint>` to [scripts/train.py](./scripts/train.py) to resume a run.
Add `--use_wandb` to log metrics to Weights & Biases (requires `wandb login` or `WANDB_API_KEY`).

