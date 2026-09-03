import os
import argparse
import numpy as np
import torch
import wandb
from pathlib import Path
import time
from tqdm import tqdm
from cs336_basics.model.language_model import TransformerLanguageModel
from cs336_basics.optimizer import AdamW, learning_rate_schedule, gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.serialization import save_checkpoint, load_checkpoint

@torch.no_grad()
def validation(model: torch.nn.Module, val_data: np.ndarray, eval_steps: int, eval_batch_size: int, context_length: int, device: str):
    model.eval()
    val_loss = 0
    for _ in range(1, eval_steps + 1):
        x, y = get_batch(val_data, eval_batch_size, context_length, device)
        logits = model(x)
        val_loss += cross_entropy(logits, y)
    val_loss /= eval_steps
    model.train()
    return val_loss 



if __name__ == "__main__":
    # specify args: hyper-parameters, data path, checkpoint path, wandb logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint path to resume training")
    parser.add_argument("--output_root", type=str, default="result")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--train_steps", type=int, help="Number of training steps")
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--eval_steps", type=int, help="Number of eval steps")
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--steps_per_eval", type=int, default=100, help="Validate every n training steps")
    parser.add_argument("--steps_per_log", type=int, default=10, help="Log every n training steps")
    parser.add_argument("--steps_per_save", type=int, default=1000, help="Save every n training steps")
    parser.add_argument("--max_learning_rate", type=float)
    parser.add_argument("--min_learning_rate", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--cosine_cycle_steps", type=int)
    parser.add_argument("--max_grad_norm", type=float)


    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=1e-2)


    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--rope_theta", type=float)
    

    args = parser.parse_args()

    # Initialize wandb project
    if args.use_wandb:
        run = wandb.init(
            project="lm_from_scratch_hw1",
            config=vars(args)
        )

    

    # Memory-efficient loading of large training and validation datasets with np.memmap
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

    # Instantiate model and optimizer
    model = TransformerLanguageModel(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32)

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.max_learning_rate,
        betas= (args.beta1, args.beta2),
        weight_decay = args.weight_decay
    )


    # Create output/checkpoint directory
    if args.resume_from is not None:
        ckpt_step, historical_best_val_loss = load_checkpoint(args.resume_from, model, optimizer, return_historical_best_val_loss=True)
        start_step = ckpt_step + 1
        if historical_best_val_loss is None:
            best_val_loss = np.inf
        else:
            best_val_loss = historical_best_val_loss
        output_dir = Path(args.resume_from).parent
    else:
        if args.run_name is None:
            run_name = str(time.time())
        else:
            run_name = args.run_name + "_" + str(time.time())
        output_dir = Path(args.output_root) / run_name
        if not os.path.exists(args.output_root):
            os.makedirs(args.output_root)
        os.makedirs(output_dir)
        start_step = 1
        best_val_loss = np.inf


    model.train()
    # Train model
    pbar = tqdm(range(start_step, args.train_steps + 1),
                total=args.train_steps, initial=start_step - 1)
    for step in pbar:
        x, y = get_batch(train_data, args.train_batch_size, args.context_length, device=args.device)

        # Forward
        logits = model(x)

        lr = learning_rate_schedule(step, args.max_learning_rate, args.min_learning_rate, args.warmup_steps, args.cosine_cycle_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        train_loss = cross_entropy(logits, y)
        train_loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # Log train metrics
        if step % args.steps_per_log == 0:
            train_loss_value = train_loss.item()
            print(f"train/step: {step} train/loss: {train_loss_value:.3f} train/lr: {lr:.3f}")
            if args.use_wandb:
                run.log({"train/loss": train_loss_value,
                        "train/lr": lr,
                        "train/step": step}, step=step)
            pbar.set_postfix(loss=train_loss_value, lr=lr)

        # Validation
        if step % args.steps_per_eval == 0:
            val_loss = validation(model, val_data, args.eval_steps, args.eval_batch_size, args.context_length, args.device)
            # Log val loss
            val_loss_value = val_loss.item()
            print(f"val/loss: {val_loss_value:.3f}")
            if args.use_wandb:
                run.log({"val/loss": val_loss_value}, step=step)

            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                save_checkpoint(model, optimizer, step, Path(output_dir) / "best.pt", historical_best_val_loss=best_val_loss)


        # Regular checkpointing
        if step % args.steps_per_save == 0:
            save_checkpoint(model, optimizer, step, Path(output_dir) / f"ckpt_{step}.pt", historical_best_val_loss=best_val_loss)


    if args.use_wandb:
        run.finish()