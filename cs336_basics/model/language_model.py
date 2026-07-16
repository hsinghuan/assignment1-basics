import torch
import torch.nn as nn
from cs336_basics.model.module import TransformerBlock, Embedding, RoPE, RMSNorm, Linear

class TransformerLanguageModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        rope = RoPE(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: shape: batch_size seq_len
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(self.ln_final(x))
        return x