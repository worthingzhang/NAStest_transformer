"""Tokenize local FineWeb-Edu sample-10BT Parquet files into GPT-2 shards.

Run from the project directory: python3 -u fineweb.py
Use --parquet-dir to select a different local copy. Only the text column is
read, one bounded batch at a time; no full-dataset Arrow cache is created.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import tiktoken


ROOT = Path(__file__).resolve().parent
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
DEFAULT_SOURCE = (ROOT / "log/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu"
                  / "snapshots" / REVISION / "sample/10BT")
enc = None


def init_tokenizer():
    global enc
    enc = tiktoken.get_encoding("gpt2")


def tokenize(text):
    # Keep empty documents: each source row contributes one leading EOT.
    if not isinstance(text, str):
        raise ValueError("Expected a string in the Parquet text column")
    tokens = [enc.eot_token]
    tokens.extend(enc.encode_ordinary(text))
    return np.asarray(tokens, dtype=np.uint16)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "edu_fineweb10B")
    parser.add_argument("--workers", type=int, default=min(32, max(1, (os.cpu_count() or 1) // 2)))
    parser.add_argument("--batch-size", type=int, default=8192, help="Maximum documents held per batch")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Tokens per shard")
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.shard_size) < 1:
        parser.error("workers, batch-size and shard-size must be positive")

    files = sorted(args.parquet_dir.glob("*.parquet"))
    if not files:
        parser.error(f"No local Parquet files in {args.parquet_dir}; provide --parquet-dir")
    source = []
    for path in files:
        parquet = pq.ParquetFile(path)
        if "text" not in parquet.schema_arrow.names:
            raise ValueError(f"Missing text column: {path}")
        source.append({"name": path.name, "size_bytes": path.stat().st_size,
                       "rows": parquet.metadata.num_rows, "cache_blob": path.resolve().name})
        parquet.close()
    expected_documents = sum(item["rows"] for item in source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    # Exclusive creation also prevents two preparation jobs using the same output.
    lock = args.output_dir / ".prepare.lock"
    with lock.open("x") as stream:
        stream.write(f"{os.getpid()}\n")

    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / "log/tiktoken_cache"))
    started = time.time()
    manifest_path = args.output_dir / "preparation.json"
    manifest = {
        "status": "running", "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT", "source_directory": str(args.parquet_dir.resolve()),
        "source_files": source, "expected_documents": expected_documents,
        "tokenizer": "tiktoken/gpt2", "tiktoken_version": tiktoken.__version__,
        "document_prefix": 50256, "encoding": "encode_ordinary", "dtype": "uint16",
        "shard_size": args.shard_size, "validation_shard_index": 0,
        "workers": args.workers, "batch_size": args.batch_size,
        "documents": 0, "total_tokens": 0, "shards": [],
    }
    write_json(manifest_path, manifest)
    print(f"Source: {args.parquet_dir.resolve()}", flush=True)
    print(f"Files: {len(files)}, documents: {expected_documents:,}, workers: {args.workers}", flush=True)
    print(f"Output: {args.output_dir.resolve()}", flush=True)

    def save_shard(tokens):
        index = len(manifest["shards"])
        split = "val" if index == 0 else "train"
        name = f"edufineweb_{split}_{index:06d}.npy"
        path = args.output_dir / name
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("xb") as stream:
            np.save(stream, tokens, allow_pickle=False)
        temporary.replace(path)
        manifest["shards"].append({"name": name, "split": split, "tokens": len(tokens)})
        print(f"Saved {name}: {len(tokens):,} tokens", flush=True)

    try:
        init_tokenizer()
        if enc.n_vocab != 50257 or enc.eot_token != 50256:
            raise ValueError("Unexpected GPT-2 vocabulary")
        # Spawn avoids inheriting Arrow's thread state. Each worker loads GPT-2 once.
        with mp.get_context("spawn").Pool(args.workers, initializer=init_tokenizer) as pool:
            buffer = np.empty(args.shard_size, dtype=np.uint16)
            used = 0
            last_report = started
            for file_index, path in enumerate(files):
                print(f"Reading {file_index + 1}/{len(files)}: {path.name}", flush=True)
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(batch_size=args.batch_size,
                                                       columns=["text"], use_threads=False):
                        # map preserves row order; finishing each batch bounds prefetch/results.
                        results = pool.map(tokenize, batch.column(0).to_pylist(), chunksize=16)
                        for tokens in results:
                            offset = 0
                            while offset < len(tokens):
                                take = min(args.shard_size - used, len(tokens) - offset)
                                buffer[used:used + take] = tokens[offset:offset + take]
                                used += take
                                offset += take
                                if used == args.shard_size:
                                    save_shard(buffer)
                                    used = 0
                            manifest["documents"] += 1
                            manifest["total_tokens"] += len(tokens)
                        now = time.time()
                        if now - last_report >= 30:
                            manifest["elapsed_seconds"] = now - started
                            write_json(manifest_path, manifest)
                            rate = manifest["total_tokens"] / (now - started)
                            print(f"Progress: {manifest['documents']:,}/{expected_documents:,} documents, "
                                  f"{manifest['total_tokens']:,} tokens, {rate:,.0f} tokens/s", flush=True)
                            last_report = now
            if used:
                save_shard(buffer[:used])
        if manifest["documents"] != expected_documents:
            raise RuntimeError("Processed document count differs from source metadata")
        if sum(item["tokens"] for item in manifest["shards"]) != manifest["total_tokens"]:
            raise RuntimeError("Shard token count differs from tokenizer output")
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = time.time() - started
        write_json(manifest_path, manifest)
        print(f"Complete: {manifest['documents']:,} documents, {manifest['total_tokens']:,} tokens, "
              f"{len(manifest['shards'])} shards", flush=True)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["elapsed_seconds"] = time.time() - started
        write_json(manifest_path, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
