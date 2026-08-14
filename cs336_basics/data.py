import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str = "cpu"):
    total_num_samples = len(x) - context_length
    rng = np.random.default_rng()
    sampled_idx_arr = rng.choice(total_num_samples, batch_size)
    input_starts = sampled_idx_arr[:, None]
    offsets = np.arange(context_length)[None, :]
    
    inputs = torch.as_tensor(x[input_starts + offsets].astype(np.int64), dtype=torch.long, device=device)
    targets = torch.as_tensor(x[input_starts + offsets + 1].astype(np.int64), dtype=torch.long, device=device)

    return (inputs, targets)