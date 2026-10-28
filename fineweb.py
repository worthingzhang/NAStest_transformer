"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Tokenizes downloaded local data and saves data shards to disk.
Run simply as:
$ python3 fineweb.py
Reads the completed Parquet files from the existing download cache under /tmp.
Keeps 30 shards (100M validation + 2.9B training tokens) in "edu_fineweb3B".
After success, deletes that private cache; the uint16 output is about 6 GB.
"""

import os
import multiprocessing as mp
import shutil
from glob import glob
import numpy as np
import tiktoken

# Share a writable tokenizer cache with train.py, outside the disposable data cache.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken"))

# Reuse the interrupted download; keep any Arrow conversion cache on /tmp too.
download_cache = "/tmp/fineweb-10b-0wr5c827"
os.environ["HF_HUB_CACHE"] = os.path.join(download_cache, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(download_cache, "datasets")
os.environ["HF_XET_CACHE"] = os.path.join(download_cache, "xet")
print(f"Using existing download cache: {download_cache}", flush=True)
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb3B"
shard_size = int(1e8) # 100M tokens per shard
max_shards = 30 # keep 3B tokens in total; shard 0 is validation

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Snapshot paths contain completed downloads, not the .incomplete files in blobs.
parquet_files = sorted(glob(os.path.join(
    download_cache, "hub/datasets--HuggingFaceFW--fineweb-edu",
    "snapshots/*/sample/10BT/*.parquet",
)))
if not parquet_files:
    raise FileNotFoundError(f"No downloaded Parquet files found in {download_cache}")
print(f"Reading {len(parquet_files)} local Parquet files; keeping {max_shards * shard_size:,} tokens.", flush=True)
fw = load_dataset(
    "parquet", data_files=parquet_files, split="train", columns=["text"],
    cache_dir=os.path.join(download_cache, "datasets"),
)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar.close()
            progress_bar = None
            if shard_index == max_shards:
                token_count = 0 # do not write a 31st, partial shard below
                break
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

# Delete only this preparation's private cache, after all requested shards are saved.
if shard_index != max_shards:
    raise RuntimeError(f"Source ended before {max_shards} full shards; cache kept at {download_cache}")
del fw
shutil.rmtree(download_cache)
print(f"Done: {max_shards} shards saved to {DATA_CACHE_DIR}; private cache deleted.")
