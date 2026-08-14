import torch
from cs336_basics.model.module import softmax

@torch.no_grad()
def generate(model: torch.nn.Module,
             input_ids: torch.Tensor,
             max_new_tokens: int = 256,
             temperature: float = 1.0,
             top_p: float = 1.0,
             eos_token_id: int = 256,
             context_length: int = None):

    assert input_ids.dim() == 1 # (seq_len,)

    model.eval()
    if context_length is None:
        context_length = model.context_length
    
    for _ in range(max_new_tokens):
        model_input = input_ids[-context_length:]
        next_token_logits = model(model_input)[-1,:] # (vocab_size,)

        # temperature scaling
        if temperature == 0.:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token_logits /= temperature

            # softmax
            probs = softmax(next_token_logits, i=-1) # (vocab_size,)

            # top-p sampling, sort from high p to low p
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1) # (vocab_size,)
            top_p_indices = sorted_indices[:(cumsum < top_p).sum() + 1] # include the first token that crosses top_p
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask[top_p_indices] = True
            probs = probs.masked_fill(~mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        if next_token.item() == eos_token_id:
            break

    return input_ids

