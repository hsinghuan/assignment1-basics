import argparse
import random
import regex as re
from multiprocessing import Pool
from cs336_basics.tokenizer import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer


def split_documents(input_path: str, start: int, end: int):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_str = f.read(end - start).decode("utf-8", errors="ignore")

    # Escape first because special token <|endoftext|> contains |
    pattern = re.escape("<|endoftext|>")
    chunk_str_list = re.split(f"({pattern})", chunk_str)
    
    return chunk_str_list


def sample_documents(input_path: str, num_docs: int = 10, num_processes: int = 4):

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    ## Pre-tokenize chunks in parallel
    with Pool(processes=num_processes) as pool:
        tasks = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
        chunk_str_list = pool.starmap(split_documents, tasks) # list of lists

    split_str_list = [split_str for chunk in chunk_str_list for split_str in chunk]

    i = 0
    documents = []
    doc = []
    while i < len(split_str_list):
        split_str = split_str_list[i]
        if split_str == '<|endoftext|>':
            doc_str = "".join(doc)
            if doc_str:
                documents.append(doc_str)
            doc = []
        else:
            doc.append(split_str)

        i += 1

    if doc: # if last document does not end with '<|endoftext|>', append final document
        doc_str = "".join(doc)
        if doc_str:
            documents.append(doc_str)

    assert num_docs <= len(documents), f"Documents to be sampled ({num_docs}) are more than documents available ({len(documents)})."

    random.seed(42)
    sampled_indices = random.sample(range(len(documents)), k=num_docs)
    sampled_documents = [documents[i] for i in sampled_indices]

    return sampled_documents
        

def estimate_compression_ratio(input_path, tokenizer, num_docs, num_processes):
    sampled_documents = sample_documents(input_path, num_docs=num_docs, num_processes=num_processes)
    
    total_num_bytes = 0
    total_num_tokens = 0
    for doc in sampled_documents:
        total_num_bytes += len(doc.encode("utf-8"))
        total_num_tokens += len(tokenizer.encode(doc))


    compression_ratio = total_num_bytes / total_num_tokens

    return compression_ratio

def estimate_tokenizer_throughput(input_path, tokenizer, num_docs, num_processes):
    sampled_documents = sample_documents(input_path, num_docs=num_docs, num_processes=num_processes)
    total_num_bytes = 0
    for doc in sampled_documents:
        total_num_bytes += len(doc.encode("utf-8"))
    
    import time
    time_start_encode = time.time()
    for doc in sampled_documents:
        _ = tokenizer.encode(doc)
    time_end_encode = time.time()
    
    throughput = total_num_bytes / (time_end_encode - time_start_encode)

    return throughput

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinystories_input_path", type=str, required=True)
    parser.add_argument("--owt_input_path", type=str, required=True)
    parser.add_argument("--tinystories_vocab_filepath", type=str, required=True)
    parser.add_argument("--tinystories_merges_filepath", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=4)
    args = parser.parse_args()

    
    tinystories_tokenizer = Tokenizer.from_files(args.tinystories_vocab_filepath, args.tinystories_merges_filepath, special_tokens=["<|endoftext|>"])

    tinystories_tinystories_compression_ratio = estimate_compression_ratio(args.tinystories_input_path, tinystories_tokenizer, num_docs=10, num_processes=args.num_processes)
    print(f"Compression ratio of using tinystories tokenizer to encode tinystories: {round(tinystories_tinystories_compression_ratio, 3)}")

    tinystories_owt_compression_ratio = estimate_compression_ratio(args.owt_input_path, tinystories_tokenizer, num_docs=10, num_processes=args.num_processes)
    print(f"Compression ratio of using tinystories tokenizer to encode OpenWebText: {tinystories_owt_compression_ratio}")


    throughput = estimate_tokenizer_throughput(args.tinystories_input_path, tinystories_tokenizer, num_docs=10, num_processes=args.num_processes)
    print(f"Tokenizer throughput: {round(throughput, 3)}")