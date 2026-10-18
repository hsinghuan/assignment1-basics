import torch
import typing
import os

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    historical_best_val_loss: float = None):
    obj = {"model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
           "iteration": iteration,
           "historical_best_val_loss": historical_best_val_loss}
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    return_historical_best_val_loss: bool = False,
                    map_location: str | torch.device | None = None):
    obj = torch.load(src, weights_only=False, map_location=map_location)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    if return_historical_best_val_loss:
        return obj["iteration"], obj["historical_best_val_loss"]
    else:
        return obj["iteration"]