import argparse
import torch
from cs336_basics.model.language_model import TransformerLanguageModel
from cs336_basics.optimizer import AdamW
from cs336_basics.serialization import load_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.generate import generate


def main(args):
    # Construct TransformerLanguageModel w/ same config
    print("Load model and tokenizer")
    model = TransformerLanguageModel(
            d_model=512,
            num_heads=16,
            d_ff=1344,
            vocab_size=10000,
            context_length=256,
            num_layers=4,
            rope_theta=10000,
            device="mps",
            dtype=torch.float32)
    
    # Load model checkpoint
    null_optimizer = AdamW(
            params=model.parameters(),
            lr=1e-3,
            betas= (0.9, 0.999),
            weight_decay = 1e-2
        )
    load_checkpoint(args.checkpoint_path, model, null_optimizer, map_location="mps")
    tinystories_tokenizer = Tokenizer.from_files(args.tinystories_vocab_filepath, args.tinystories_merges_filepath, special_tokens=["<|endoftext|>"])
    eos_token_id = tinystories_tokenizer.encode("<|endoftext|>")[0]

    # Start sampling
    print("Start generating")
    generated_ids = generate(model, torch.tensor([eos_token_id], dtype=torch.long, device="mps"), temperature=0.5, top_p=0.9, eos_token_id=eos_token_id)
    generated_str = tinystories_tokenizer.decode(generated_ids.tolist())
    print(generated_str)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--tinystories_vocab_filepath", type=str, default=None)
    parser.add_argument("--tinystories_merges_filepath", type=str, default=None)
    args = parser.parse_args()
    main(args)