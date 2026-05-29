
text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"


def bpe_example(text: str, total_rounds: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Pre-tokenization: list of pre-tokenized strings
    pretokens = text.split(" ")

    # Get frequency table of tokens of pre-tokens
    # Key: tuple of byte-string tokens, representing a pretoken
    # Value: frequency of the pretoken
    pretoken_freq_dict = dict()
    for pretoken in pretokens:
        pretoken_bytes = pretoken.encode("utf-8")
        pretoken_tokens = tuple([pretoken_bytes[i:i+1] for i in range(len(pretoken_bytes))])
        pretoken_freq_dict[pretoken_tokens] = pretoken_freq_dict.get(pretoken_tokens, 0) + 1
    
    print("pretoken freq dict", pretoken_freq_dict)

    # Initialize the vocabulary dictionary to be returned
    # Key: integer, token ID
    # Value: bytes
    vocab = dict()
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    vocab[len(vocab)] = "<|endoftext|>".encode("utf-8")

    # Initialize merge rules
    merges = []

    round = 0

    while round < total_rounds:
        # Initialize token pair frequency table to determine which to merge
        token_pair_freq_dict = dict()
        
        # Calculate the frequency of a token pair by its occurrence in the pretokens
        for pretoken_tokens, pretoken_freq in pretoken_freq_dict.items():
            for i in range(len(pretoken_tokens) - 1):
                token_pair = pretoken_tokens[i:i+2]
                token_pair_freq_dict[token_pair] = token_pair_freq_dict.get(token_pair, 0) + pretoken_freq
        
        # Find the token pair with the highest number of occurrences
        max_freq = 0
        max_freq_token_pair = None
        for token_pair in sorted(token_pair_freq_dict.keys()):
            freq = token_pair_freq_dict[token_pair]
            if freq >= max_freq:
                max_freq_token_pair = token_pair
                max_freq = freq

        # Update merge rule and vocabulary
        merges.append(max_freq_token_pair)
        vocab[len(vocab)] = (b''.join(max_freq_token_pair))

        # Merge tokens in pre-tokens
        new_pretoken_freq_dict = dict()
        for pretoken_tokens in pretoken_freq_dict:
            new_pretoken_tokens = []
            i = 0
            while i < len(pretoken_tokens) - 1:
                if pretoken_tokens[i:i+2] == max_freq_token_pair:
                    new_pretoken_tokens.append(b''.join(max_freq_token_pair))
                    i += 2
                else:
                    new_pretoken_tokens.append(pretoken_tokens[i])
                    i += 1
            if i == len(pretoken_tokens) - 1:
                new_pretoken_tokens.append(pretoken_tokens[i])
            
            new_pretoken_freq_dict[tuple(new_pretoken_tokens)] = pretoken_freq_dict[pretoken_tokens]
        assert sum(pretoken_freq_dict.values()) == sum(new_pretoken_freq_dict.values())
        pretoken_freq_dict = new_pretoken_freq_dict
        del new_pretoken_freq_dict

        round += 1
    
    print(token_pair_freq_dict)
    

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = bpe_example(text, 6)
    for k, v in vocab.items():
        print(k, v)
    print("merges", merges)