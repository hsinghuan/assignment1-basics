import os
import json
import regex as re
from typing import Iterable, Iterator
from cs336_basics.tokenizer.train_bpe import PAT

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens_escaped = [re.escape(token) for token in self.special_tokens]
            self.special_tokens_split_pattern = "|".join(special_tokens_escaped)


        self.merge_to_rank = dict()
        for i, merge in enumerate(merges):
            self.merge_to_rank[merge] = i

        self.vocab_reverse = {token: i for i, token in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        if os.path.exists(vocab_filepath):
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                vocab_json = json.load(f)
            vocab = {
                int(token_id): bytes(byte_values)
                for token_id, byte_values in vocab_json.items()
            }
        else:
            raise FileNotFoundError(f"{vocab_filepath} does not exist.")

        if os.path.exists(merges_filepath):
            with open(merges_filepath, "r", encoding="utf-8") as f:
                merges_json = json.load(f)
            merges = [
                (bytes(left), bytes(right))
                for left, right in merges_json
            ]
        else:
            raise FileNotFoundError(f"{merges_filepath} does not exist.")

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids = []

        if self.special_tokens:
            chunk_str_list = re.split(f"({self.special_tokens_split_pattern})", text)
        else:
            chunk_str_list = [text]

        for chunk_str in chunk_str_list:
            if not chunk_str:
                continue
            if chunk_str in self.special_tokens:
                token_ids.append(self.vocab_reverse[chunk_str.encode("utf-8")])
                continue
            # pretokenize
            matches = re.finditer(PAT, chunk_str)
            for match in matches:
                pretoken = match.group()
                pretoken_bytestr = pretoken.encode("utf-8")
                pretoken_tokens = [pretoken_bytestr[i:i+1] for i in range(len(pretoken_bytestr))]

                while True:
                    highest_rank = float('inf')
                    highest_rank_token_pair = None
                    for i in range(len(pretoken_tokens)-1):
                        token_pair = tuple(pretoken_tokens[i:i+2])
                        if token_pair in self.merge_to_rank and self.merge_to_rank[token_pair] < highest_rank:
                            highest_rank = self.merge_to_rank[token_pair]
                            highest_rank_token_pair = token_pair
                    
                    if highest_rank_token_pair:
                        merged_pretoken_tokens = []
                        i = 0
                        while i < len(pretoken_tokens) - 1:
                            if tuple(pretoken_tokens[i:i+2]) == highest_rank_token_pair:
                                merged_pretoken_tokens.append(b''.join(pretoken_tokens[i:i+2]))
                                i += 2
                            else:
                                merged_pretoken_tokens.append(pretoken_tokens[i])
                                i += 1
                        if i == len(pretoken_tokens) - 1:
                            merged_pretoken_tokens.append(pretoken_tokens[i])
                        pretoken_tokens = merged_pretoken_tokens
                    else:
                        # no more merges left
                        break

                for token in pretoken_tokens:
                    token_ids.append(self.vocab_reverse[token])


        return token_ids

            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:

        decoded_bytes = b''.join([self.vocab[id] for id in ids])

        return decoded_bytes.decode("utf-8", errors="replace")

