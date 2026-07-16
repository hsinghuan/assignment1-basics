import math
import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange, repeat

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        assert in_features > 0 and out_features > 0

        mean = 0
        std = math.sqrt(2 / (in_features + out_features))
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(W, mean=mean, std=std, a=-3*std, b=3*std)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        assert num_embeddings > 0 and embedding_dim > 0

        E = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(E, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(E)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert token_ids.dtype == torch.long
        output = self.weight[token_ids]
        return output
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(torch.square(x), "... d -> ... 1", "mean") + self.eps) # FLOPS: batch size x seq len x (d_model + d_model - 1 + 1 + 1 + 1), negligible
        output = einsum(x, self.weight, "... d, d -> ... d") / rms # FLOPs: batch size x seq len x (d_model + d_model) -> element-wise, negligible
        return output.to(in_dtype)

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))
        # FLOPs: w1 + silu + w3 + * + w2
        #       = 2 x seq_len x d_model x d_ff + 4 x seq_len x d_ff + 2 x seq_len x d_model x d_ff + seq_len x d_ff + 2 x seq_len x d_model x d_ff
        #       = 6 x seq_len x d_model x d_ff + 5 x seq_len x d_ff


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        num_rotate = d_k // 2
        
        # construct num_rotate (d_k / 2) rotation frequencies
        freq = torch.tensor([1 / (self.theta ** (2 * k / d_k)) for k in range(num_rotate)], device=device, dtype=torch.float32) # shape: (num_rotate,)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) # shape: (max_seq_len,)
        angles = einsum(positions, freq, "max_seq_len, num_rotate -> max_seq_len num_rotate") # Precomputed angles from outer product, shape: (max_seq_len, num_rotate)
        cos_table = torch.cos(angles) # shape: (max_seq_len, num_rotate)
        sin_table = torch.sin(angles) # shape: (max_seq_len, num_rotate)
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: shape (..., seq_len, d_k)
        # token_positions: shape (..., seq_len)

        # Rearrange the input x into pairwise view
        x = rearrange(x, "... seq_len (num_rotate rotate_dim) -> ... seq_len num_rotate rotate_dim", rotate_dim=2) # row-major: num_rotate = j // 2, rotate_dim = j % 2

        # For each 2d vector v,
        # v'[0] = cos(\theta) v[0] - sin(\theta) v[1]
        # v'[1] = sin(\theta) v[0] + cos(\theta) v[1]
        
        cos_table_lookup = self.cos_table[token_positions] # indexing first axis of cos_table with token_positions -> (..., seq_len, num_rotate)
        sin_table_lookup = self.sin_table[token_positions] # indexing first axis of sin_table with token_positions -> (..., seq_len, num_rotate)

        x0 = cos_table_lookup * x[..., 0] - sin_table_lookup * x[..., 1] # Shape: ... seq_len num_rotate, FLOPs: 3 x seq_len x num_rotate = 3 x seq_len x d_k / 2 (per head)
        x1 = sin_table_lookup * x[..., 0] + cos_table_lookup * x[..., 1] # Shape: ... seq_len num_rotate, FLOPs: 3 x seq_len x num_rotate = 3 x seq_len x d_k / 2 (per head)
        output = rearrange([x0, x1], "rotate_dim ... seq_len num_rotate -> ... seq_len (num_rotate rotate_dim)") # Stack and rearrange back to shape: ... seq_len d_k, also row major
        
        return output
        

def softmax(x: torch.Tensor, i: int):
    assert i < len(x.size())
    softmax_max = torch.max(x, dim=i, keepdim=True)[0] # torch.max returns (values, indices)
    x = torch.exp(x - softmax_max)
    return x / x.sum(dim=i, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    # Q shape: (batch_size, ..., n_queries, d_k)
    # K shape: (batch_size, ..., n_keys, d_k)
    # V shape: (batch_size, ..., n_keys, d_v)
    # mask shape: (n_queries, n_keys)

    d_k = Q.size(-1)

    presoftmax = einsum(Q, K, "... n_queries d_k, ... n_keys d_k -> ... n_queries n_keys") / math.sqrt(d_k) # FLOPs: 2 x n_queries x n_keys x d_k
    if mask is not None:
        presoftmax = presoftmax.masked_fill(~mask, -torch.inf)
    postsoftmax = softmax(presoftmax, i=-1) # FLOPs: ~ 5 x n_keys x n_queries
    output = einsum(postsoftmax, V, "... n_queries n_keys, ... n_keys d_v -> ... n_queries d_v") # FLOPs: 2 x n_queries x n_keys x d_v
    return output


class CausalMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE = None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.d_k = d_k
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        # x shape: ..., seq_len, d_model
        x_device = x.device
        seq_len = x.size(-2)
        Q = rearrange(self.q_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.d_k) # FLOPS: 2 x seq_len x d_model ** 2
        K = rearrange(self.k_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.d_k) # FLOPS: 2 x seq_len x d_model ** 2
        V = rearrange(self.v_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k=self.d_k) # FLOPS: 2 x seq_len x d_model ** 2
        
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x_device)
            Q = self.rope(Q, token_positions) # FLOPS: 3 x seq_len x d_k x num_heads = 3 x seq_len x d_model -> negligible
            K = self.rope(K, token_positions) # FLOPS: 3 x seq_len x d_k x num_heads = 3 x seq_len x d_model -> negligible

        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x_device, dtype=torch.bool), diagonal=1)
        output = scaled_dot_product_attention(Q, K, V, mask=mask) # FLOPS: 4 x d_k x seq_len ** 2 x num_heads + 5 x seq_len ** 2 x num_heads = 4 x seq_len ** 2 x d_model + 5 x seq_len ** 2 x num_heads
        output = rearrange(output, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        output = self.output_proj(output) # FLOPS: 2 x seq_len x d_model ** 2
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE = None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiheadAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))

        return x
    


# if __name__ == "__main__":
    # rope = RoPE(theta=10000.0, d_k=4, max_seq_len=8)
    # x = torch.arange(4).float().reshape(1, 1, 4)        # (batch=1, seq=1, d_k=4)
    # pos = torch.tensor([0])                              # position 0
    # print(torch.allclose(rope(x, pos), x))                  # should be True

    # t = torch.randn(2, 3, 4)
    # for d in range(3):
    #     out = softmax(t, d)
    #     print(torch.allclose(out.sum(dim=d), torch.ones_like(out.sum(dim=d))))  # all True?
