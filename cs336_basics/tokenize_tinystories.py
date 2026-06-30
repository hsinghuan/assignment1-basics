import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vocab_filepath", type=str, required=True)
    parser.add_argument("--merges_filepath", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data_tokenized")
    args = parser.parse_args()

    
    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merges_filepath, special_tokens=["<|endoftext|>"])

    data_path = Path(args.data_path)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = Path(args.output_dir) / Path(data_path.stem + "_tokenized.bin")
    
    token_count = 0
    print("Start tokenizing")
    with open(data_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "wb") as f_out:
            buffer = []
            for token_id in tokenizer.encode_iterable(tqdm(f_in, unit="line")):
                buffer.append(token_id)
                token_count += 1
                if len(buffer) >= 1000000:
                    arr = np.array(buffer, dtype=np.uint16)
                    arr.tofile(f_out)
                    buffer = []
            
            if buffer:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(f_out)
                
    print(f"Finish tokenizing to {output_path}")
    print(f"Total number of tokens: {token_count}")