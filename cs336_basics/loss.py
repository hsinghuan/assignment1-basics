import torch
from einops import reduce, rearrange

def cross_entropy(logits: torch.Tensor, targets: torch.LongTensor):
    # logits shape: ... vocab size (... could be batch, seq_len)
    # targets shape: ... (... could be batch, seq_len)
    # cross entropy per sample per position:
    # - \log (\exp(logits[i]) / \sum_{j=1}^d \exp(logits[j])) = - logits[i] + \log (\sum_{j=1}^d \exp(logits[j]))

    logits_max = reduce(logits, "... vocab -> ... 1", "max")
    logits = logits - logits_max # ... vocab or batch_size seq_len vocab
    log_sum_exp = torch.log(reduce(torch.exp(logits), "... vocab -> ... 1", "sum"))
    targets = rearrange(targets, "... -> ... 1")
    ce_loss = - torch.gather(logits, -1, targets) + log_sum_exp

    return ce_loss.mean()
