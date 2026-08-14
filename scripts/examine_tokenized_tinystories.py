import argparse
import numpy as np
from cs336_basics.tokenizer import Tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_data_path", type=str, required=True)
    parser.add_argument("--vocab_filepath", type=str, required=True)
    parser.add_argument("--merges_filepath", type=str, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merges_filepath, special_tokens=["<|endoftext|>"])

    end_of_text_id = tokenizer.vocab_reverse[b"<|endoftext|>"]
    token_ids_to_decode = []
    with open(args.tokenized_data_path, "rb") as f:
        first_doc_found = False
        while True:
            chunk_token_ids = np.fromfile(f, dtype=np.uint16, count=10000)
            if len(chunk_token_ids) == 0:
                break # EOF
            
            for token_id in chunk_token_ids:
                if token_id == end_of_text_id:
                    first_doc_found = True
                    break # First document found
                token_ids_to_decode.append(int(token_id))
            
            if first_doc_found:
                break

    decoded_text = tokenizer.decode(token_ids_to_decode)
    print(decoded_text)