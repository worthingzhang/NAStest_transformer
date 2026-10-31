"""只读训练分片，生成三组实验共用的词频、观测 token 和几何拟合划分。

示例（需要用户另行执行，不会由训练入口自动扫描数据）：
    python3 build_freq_buckets.py --data-dir edu_fineweb10B

计数包含 EOT；分桶排除 EOT 和填充词表行。这里统计的是完整语料词频，
不是某个训练检查点之前的实际输入出现次数。验证分片只记录元数据，不参与计数。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


VOCAB_SIZE = 50304
TOKENIZER_VOCAB_SIZE = 50257
EOT_ID = 50256


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path, *, json_file=False):
    """先写同目录临时文件，再原子替换；不留下看似完整的半文件。"""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as file:
            temporary = Path(file.name)
            if json_file:
                file.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
            else:
                torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def inspect_shards(data_dir, split):
    """与当前 DataLoaderLite 一样按文件名排序；只读取 NPY 头部。"""
    paths = sorted(path for path in data_dir.iterdir() if split in path.name and path.suffix == ".npy")
    if not paths:
        raise ValueError(f"没有找到 {split} 分片：{data_dir}")
    records = []
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 1 or array.dtype.kind not in "iu" or array.size == 0:
            raise ValueError(f"需要非空一维整数 token 数组：{path}")
        stat = path.stat()
        records.append({
            "name": path.name,
            "tokens": int(array.size),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "dtype": str(array.dtype),
        })
        del array
    return records


def count_train_tokens(data_dir, records, chunk_tokens):
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for record in records:
        path = data_dir / record["name"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, array.size, chunk_tokens):
            chunk = np.asarray(array[start:start + chunk_tokens], dtype=np.int64)
            if chunk.min() < 0 or chunk.max() >= TOKENIZER_VOCAB_SIZE:
                raise ValueError(f"训练分片含非法 GPT-2 token ID：{path}, offset={start}")
            counts += np.bincount(chunk, minlength=VOCAB_SIZE)
        stat = path.stat()
        if stat.st_size != record["size_bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"计数期间分片被修改：{path}")
        print(f"counted {record['name']}: {record['tokens']:,} tokens", flush=True)
        del array
    expected = sum(record["tokens"] for record in records)
    if int(counts.sum()) != expected:
        raise RuntimeError("计数之和与训练语料 token 总量不一致")
    return counts


def deranged_ids(ids, rng):
    """在随机循环上平移一位，保证每个位置都不会配到自己。"""
    if len(ids) < 2:
        raise ValueError("打乱对照至少需要两个 token")
    order = rng.permutation(len(ids))
    result = np.empty_like(ids)
    result[order] = ids[np.roll(order, 1)]
    return result


def frequency_matched_ids(ids, counts, rng, bin_width=0.5):
    """按 log10 词频分桶后错配；单例桶与相邻频率桶合并，避免自配。

    这是近似频率匹配。极端高频 token 可能没有接近的配对，分析时仍应查看
    每对 token 的实际计数，而不能把这个对照称作严格相同词频。
    """
    order = np.argsort(counts[ids], kind="stable")
    log_counts = np.log10(counts[ids])
    labels = np.floor(log_counts[order] / bin_width).astype(np.int64)
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    bins = list(np.split(order, boundaries))
    index = 0
    while index < len(bins):
        if len(bins[index]) != 1:
            index += 1
            continue
        if len(bins) == 1:
            raise ValueError("频率匹配对照至少需要两个 token")
        if index == 0:
            neighbor = 1
        elif index == len(bins) - 1:
            neighbor = index - 1
        else:
            value = log_counts[bins[index][0]]
            left_gap = value - log_counts[bins[index - 1][-1]]
            right_gap = log_counts[bins[index + 1][0]] - value
            neighbor = index - 1 if left_gap <= right_gap else index + 1
        first, second = sorted((index, neighbor))
        bins[first] = np.concatenate((bins[first], bins[second]))
        del bins[second]
        index = max(0, first)
    result = np.empty_like(ids)
    for positions in bins:
        result[positions] = deranged_ids(ids[positions], rng)
    return result


def build_groups(counts, args):
    rng = np.random.default_rng(args.seed)
    ordinary_ids = np.arange(EOT_ID, dtype=np.int64)
    seen = ordinary_ids[counts[ordinary_ids] > 0]
    # 相同词频时用 token ID 打破平局，使结果不依赖排序实现。
    descending = seen[np.lexsort((seen, -counts[seen]))]
    rare_candidates = ordinary_ids[counts[ordinary_ids] >= args.min_rare_count]
    ascending = rare_candidates[np.lexsort((rare_candidates, counts[rare_candidates]))]
    required = max(args.high_count, args.fit_count + args.test_count, args.uniform_count)
    if len(seen) < required or len(ascending) < args.rare_count:
        raise ValueError(
            f"可用 token 不足：已见普通 token={len(seen)}, 需要至少 {required}; "
            f"count>={args.min_rare_count} 的 token={len(ascending)}, 低频桶需要 {args.rare_count}"
        )
    geometry_pool = descending[:args.fit_count + args.test_count]
    split = rng.permutation(geometry_pool)
    groups = {
        "high": descending[:args.high_count],
        "rare": ascending[:args.rare_count],
        "uniform": rng.choice(seen, size=args.uniform_count, replace=False),
        "geometry_fit": split[:args.fit_count],
        "geometry_test": split[args.fit_count:],
    }
    permutations = {}
    for name, ids in groups.items():
        permutations[name] = {
            "shuffled": torch.from_numpy(deranged_ids(ids, rng).copy()),
            "frequency_matched": torch.from_numpy(frequency_matched_ids(ids, counts, rng).copy()),
        }
    null_fits = [torch.from_numpy(deranged_ids(groups["geometry_fit"], rng).copy()) for _ in range(3)]
    return {
        "schema_version": 1,
        "seed": args.seed,
        "vocab_size": VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "eot_id": EOT_ID,
        "token_ids": torch.from_numpy(np.unique(np.concatenate(list(groups.values())))),
        "groups": {name: torch.from_numpy(ids.copy()) for name, ids in groups.items()},
        "permutations": permutations,
        "fit_null_permutations": null_fits,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("edu_fineweb10B"))
    parser.add_argument("--out-dir", type=Path, default=Path("log/embedding_head/analysis_data"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-rare-count", type=int, default=20)
    parser.add_argument("--high-count", type=int, default=500)
    parser.add_argument("--rare-count", type=int, default=2000)
    parser.add_argument("--uniform-count", type=int, default=1024)
    parser.add_argument("--fit-count", type=int, default=2000)
    parser.add_argument("--test-count", type=int, default=500)
    parser.add_argument("--chunk-tokens", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.seed < 0 or args.min_rare_count < 1 or args.chunk_tokens < 1:
        parser.error("seed 不能为负，min-rare-count 和 chunk-tokens 必须为正")
    if min(args.high_count, args.rare_count, args.uniform_count, args.fit_count, args.test_count) < 2:
        parser.error("每个观测集合至少需要 2 个 token，以构造无自配对照")
    return args


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")
    # 不覆盖任何已存在目录，避免三组实验在中途切换到不同的观测 token。
    if out_dir.exists():
        raise FileExistsError(f"输出目录已存在，请复用其产物或选择新目录：{out_dir}")
    train_shards = inspect_shards(data_dir, "train")
    val_shards = inspect_shards(data_dir, "val")
    overlap = {item["name"] for item in train_shards} & {item["name"] for item in val_shards}
    if overlap:
        raise ValueError(f"文件名同时匹配 train 和 val，无法隔离训练/验证数据：{sorted(overlap)}")
    counts = count_train_tokens(data_dir, train_shards, args.chunk_tokens)
    groups = build_groups(counts, args)
    out_dir.mkdir(parents=True, exist_ok=False)
    counts_path = out_dir / "token_counts.pt"
    groups_path = out_dir / "token_groups.pt"
    atomic_save(torch.from_numpy(counts), counts_path)
    groups["counts_sha256"] = file_sha256(counts_path)
    atomic_save(groups, groups_path)
    manifest = {
        "schema_version": 1,
        "data_dir": str(data_dir),
        "tokenizer": "gpt2",
        "vocab_size": VOCAB_SIZE,
        "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
        "eot_id": EOT_ID,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "total_train_tokens": sum(item["tokens"] for item in train_shards),
        "total_val_tokens": sum(item["tokens"] for item in val_shards),
        "sha256": {
            "token_counts.pt": groups["counts_sha256"],
            "token_groups.pt": file_sha256(groups_path),
        },
        "sampling": {
            "seed": args.seed,
            "min_rare_count": args.min_rare_count,
            "group_sizes": {name: len(ids) for name, ids in groups["groups"].items()},
            "selected_unique_tokens": len(groups["token_ids"]),
            "rare_count_min": int(counts[groups["groups"]["rare"].numpy()].min()),
            "rare_count_max": int(counts[groups["groups"]["rare"].numpy()].max()),
            "frequency_bin_log10_width": 0.5,
            "singleton_rule": "merge with nearest adjacent frequency bin before derangement",
            "fit_null_count": len(groups["fit_null_permutations"]),
            "excluded_analysis_ids": "EOT 50256 and padding 50257..50303; unseen tokens excluded",
            "count_scope": "all train shard tokens, including EOT; not actual prefix exposure",
        },
    }
    atomic_save(manifest, out_dir / "manifest.json", json_file=True)
    print(f"saved {len(groups['token_ids'])} unique observed token IDs to {out_dir}")


if __name__ == "__main__":
    main()
