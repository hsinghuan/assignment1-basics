import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
import time

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_chunk(input_path: str, start: int, end: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_str = f.read(end - start).decode("utf-8", errors="ignore")

    # Sort special tokens based on length so if a longer one contains another, it will be matched first
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    # Split each chunk by special token because we don't want to calculate byte pairs across special tokens
    if special_tokens:
        # Escape first because special token <|endoftext|> contains |
        special_tokens_escaped = [re.escape(token) for token in special_tokens]
        pattern = "|".join(special_tokens_escaped)
        chunk_str_list = re.split(pattern, chunk_str)
    else:
        chunk_str_list = [chunk_str]

    # Pretokenize and calculate pretoken frequencies
    pretoken_freq_dict = defaultdict(int)
    for chunk_substr in chunk_str_list:
        matches = re.finditer(PAT, chunk_substr)
        for match in matches:
             pretoken = match.group()
             pretoken_freq_dict[pretoken] += 1

    return pretoken_freq_dict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    time_start = time.time()
    # Parallelized pre-tokenization
    ## Read file
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    ## Pre-tokenize chunks in parallel
    with Pool(processes=num_processes) as pool:
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        chunk_pretoken_freq_dict_list = pool.starmap(pretokenize_chunk, tasks)

    time_end_pretokenization = time.time()
    print(f"Pretokenization Time: {round((time_end_pretokenization - time_start) / 60, 3)} minutes")

    pretoken_tokens_dict = dict() # key: int, pretoken ID; value: tuple of byte str, the current token sequence
    pretoken_freq_dict = defaultdict(int) # key: int, pretoken ID; value: frequency of a pretoken's occurrence
    pretoken_str_id_dict = dict() # key: str, string format of a pretoken, value: int, pretoken ID

    # Get the global statistics from the chunk-wise pretoken frequency dictionaries
    for chunk_pretoken_freq_dict in chunk_pretoken_freq_dict_list:
        for pretoken, freq in chunk_pretoken_freq_dict.items():
            if pretoken in pretoken_str_id_dict:
                pretoken_freq_dict[pretoken_str_id_dict[pretoken]] += freq
            else:
                pretoken_id = len(pretoken_tokens_dict)
                pretoken_tokens = pretoken.encode("utf-8")
                pretoken_tokens_dict[pretoken_id] = tuple([pretoken_tokens[i:i+1] for i in range(len(pretoken_tokens))])
                pretoken_freq_dict[pretoken_id] = freq
                pretoken_str_id_dict[pretoken] = pretoken_id
    
    time_end_pretoken_stats = time.time()
    print(f"Pretoken Statistics Computation Time: {round((time_end_pretoken_stats - time_end_pretokenization) / 60, 3)} minutes")

    # Initialize vocabulary dictionary
    vocab = dict()
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # Initialize merge rule
    merges = []

    token_pair_freq_dict = defaultdict(int) # key: Tuple[bytes, bytes], token pairs; value: int, frequency of token pair occurrence
    # token_pair_pretoken_ids_dict: A mapping for quickly looking at the pretokens that contain a token pair for efficient merging
    token_pair_pretoken_ids_dict = defaultdict(set) # key: Tuple[bytes, bytes], token pairs; value: set(int), a set of pretoken ids of pretokens containing the token pair

    # Initialize token_pair_freq_dict and token_pair_pretoken_ids_dict
    for pretoken_id, pretoken_freq in pretoken_freq_dict.items():
        pretoken_tokens = pretoken_tokens_dict[pretoken_id]
        for i in range(len(pretoken_tokens) - 1):
            token_pair = pretoken_tokens[i:i+2]
            token_pair_freq_dict[token_pair] += pretoken_freq
            token_pair_pretoken_ids_dict[token_pair].add(pretoken_id)
    
    time_start_merge = time.time()
    print(f"Token Pair/Vocab/Merge Data Structure Initialization Time: {round((time_start_merge - time_end_pretoken_stats) / 60, 3)} minutes")

    vocab_size_before_merge = len(vocab)
    num_merges_upper_bound = vocab_size - vocab_size_before_merge
    with tqdm(total=num_merges_upper_bound, desc="BPE Training") as pbar:
        while len(vocab) < vocab_size and len(token_pair_freq_dict) > 0:
            # Find the token pair with the highest number of occurrences
            max_freq = 0
            max_freq_token_pair = None
            # for token_pair in sorted(token_pair_freq_dict.keys()):
            for token_pair in token_pair_freq_dict.keys():
                freq = token_pair_freq_dict[token_pair]
                if freq > max_freq or (freq == max_freq and token_pair > max_freq_token_pair):
                    max_freq_token_pair = token_pair
                    max_freq = freq
            
            # Update merge rule and vocabulary
            merges.append(max_freq_token_pair)
            vocab[len(vocab)] = (b''.join(max_freq_token_pair))

            # Merge tokens in pretokens containing the max_freq_token_pair
            pretoken_ids_to_update = token_pair_pretoken_ids_dict[max_freq_token_pair].copy()
            for pretoken_id in pretoken_ids_to_update:
                pretoken_tokens = pretoken_tokens_dict[pretoken_id]

                # calculate old adjacent token pair frequencies/contributions for this pretoken
                pretoken_token_pair_freq = defaultdict(int)
                k = 0
                while k < len(pretoken_tokens) - 1:
                    pretoken_token_pair_freq[pretoken_tokens[k:k+2]] += pretoken_freq_dict[pretoken_id]
                    k += 1

                # subtract old adjacent token pair contributions from global counts
                # remove old adjacent token pair contributions to pretoken ids
                for token_pair, freq in pretoken_token_pair_freq.items():
                    token_pair_freq_dict[token_pair] -= freq
                    assert token_pair_freq_dict[token_pair] >= 0
                    if token_pair_freq_dict[token_pair] == 0:
                        del token_pair_freq_dict[token_pair]
                    token_pair_pretoken_ids_dict[token_pair].remove(pretoken_id)
                    if len(token_pair_pretoken_ids_dict[token_pair]) == 0:
                        del token_pair_pretoken_ids_dict[token_pair]

                # construct new token sequence after merging
                new_pretoken_tokens = []
                i = 0
                while i < len(pretoken_tokens) - 1:
                    if pretoken_tokens[i:i+2] == max_freq_token_pair:
                        new_pretoken_tokens.append(b''.join(max_freq_token_pair)) # build new pretoken token sequence
                        i += 2
                    else:
                        new_pretoken_tokens.append(pretoken_tokens[i])
                        i += 1
                if i == len(pretoken_tokens) - 1:
                    new_pretoken_tokens.append(pretoken_tokens[i])
                new_pretoken_tokens = tuple(new_pretoken_tokens)
                
                
                # add new adjacent token pair contributions
                j = 0
                while j < len(new_pretoken_tokens) - 1:
                    token_pair_freq_dict[new_pretoken_tokens[j:j+2]] += pretoken_freq_dict[pretoken_id]
                    token_pair_pretoken_ids_dict[new_pretoken_tokens[j:j+2]].add(pretoken_id)
                    j += 1
                
                pretoken_tokens_dict[pretoken_id] = tuple(new_pretoken_tokens)

            assert max_freq_token_pair not in token_pair_freq_dict
            assert max_freq_token_pair not in token_pair_pretoken_ids_dict

            pbar.update(1)

    time_end_merge = time.time()
    print(f"Merge Time: {round((time_end_merge - time_start_merge) / 60, 3)} minutes")

    return vocab, merges

# if __name__ == "__main__":
#     train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 0, ["<|endoftext|>"])