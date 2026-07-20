from typing import Tuple, Optional, Callable, Iterable
import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        
        assert lr >= 0
        assert 0 <= betas[0] < 1 and 0 <= betas[1] < 1
        assert eps > 0
        assert weight_decay >= 0
            
        defaults = {"lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay
                   }
        super().__init__(params, defaults)

    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                if t == 0: # lazy initialization
                    state["m"] = torch.zeros_like(p.data) # 1st moment vector initialization
                    state["v"] = torch.zeros_like(p.data) # 2nd moment vector initialization
                    t += 1

                lr_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                p.data = p.data - lr * weight_decay * p.data # 2 FLOPs per param
                state["m"] = betas[0] * state["m"] + (1 - betas[0]) * p.grad.data # 3 FLOPs per param
                state["v"] = betas[1] * state["v"] + (1 - betas[1]) * p.grad.data**2 # 4 FLOPs per param
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps) # 5 FLOPs per param
                state["t"] = t + 1
        
        return loss


def learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        learning_rate = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        learning_rate = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        learning_rate = min_learning_rate
    
    return learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = list(parameters)
    grad_norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm += (torch.linalg.norm(param.grad)) ** 2
    grad_norm = torch.sqrt(grad_norm)

    if grad_norm > max_l2_norm:
        for param in parameters:
            if param.grad is None:
                continue
            param.grad *= (max_l2_norm / (grad_norm + 1e-6))