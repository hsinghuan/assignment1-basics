import os
import argparse
import cProfile
import pstats
import time
import json
from cs336_basics.tokenizer import train_bpe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="tokenizer")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--cprofile", action="store_true")

    args = parser.parse_args()

    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()

    time_start_train_bpe = time.time()
    vocab, merges = train_bpe(input_path=args.input_path,
                              vocab_size=args.vocab_size,
                              special_tokens=["<|endoftext|>"],
                              num_processes=args.num_processes)
    
    time_end_train_bpe = time.time()

    print(f"Total Train BPE Time: {(time_end_train_bpe - time_start_train_bpe) / 60} minutes")
    
    if args.cprofile:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime")
        stats.print_stats(30)


    longest_token = None
    longest_token_len = 0
    for id, token in vocab.items():
        if len(token) > longest_token_len:
            longest_token = token
            longest_token_len = len(token)

    print(f"Longest token: {longest_token}")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "tinystories_bpe_vocab.json"), "w") as f:
        serializable_vocab = {
            token_id: list(token_bytes)
            for token_id, token_bytes in vocab.items()
        }
        json.dump(serializable_vocab, f)

    with open(os.path.join(args.output_dir, "tinystories_bpe_merges.json"), "w") as f:
        serializable_merges = [
            [list(left), list(right)]
            for left, right in merges
        ]
        json.dump(serializable_merges, f)