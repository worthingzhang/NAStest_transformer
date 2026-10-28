"""从原版 FineWeb-Edu 数据源流式准备小规模 GPT-2 训练数据。

在项目目录执行：
    python3 prepare_small_data.py
    python3 prepare_small_data.py --train-tokens 20000000 --val-tokens 2000000

默认保存 1000 万训练 token 和 100 万验证 token，uint16 数组约占 22 MB。
输出目录仍叫 edu_fineweb10B，是为了兼容 train.py，不代表实际有 10B token。
需要 numpy、tiktoken、huggingface_hub、pyarrow、tqdm，以及可访问数据源的网络。
其中 huggingface_hub 和 pyarrow 通常已随 datasets 一起安装。

流程：流式读取文章 -> GPT-2 分词 -> 添加文档分隔符 -> 保存一维 .npy 数组。
每个集合保存一个分片；内存主要由当前集合的 token 数组和读取缓冲决定。
两边使用不同的文档记录，达到 token 上限时丢弃最后一篇文档的剩余部分。
这是按读取顺序截取的小样本，不保证代表完整数据集，也不额外去除重复文本。

自定义数量时，每个分片至少需要 B * T * 进程数 + 1 个 token，才能供当前
DataLoaderLite 取出一组批次。训练时的 B、T、步数仍需在 train.py 中配置。

数据源：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
参考：https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
"""

import argparse
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as parquet
import tiktoken
from huggingface_hub import HfFileSystem
from tqdm import tqdm


DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "edu_fineweb10B"


def stream_documents():
    """逐批读取 sample-10BT 对应的远程 Parquet 文件，仅获取 text 列。"""
    filesystem = HfFileSystem()
    pattern = f"datasets/{DATASET_ID}/sample/10BT/*.parquet"
    filenames = sorted(filesystem.glob(pattern))
    if not filenames:
        raise RuntimeError(f"没有找到数据文件：{pattern}")

    for filename in filenames:
        # 范围请求和有上限的内存缓存，不先把整个 Parquet 文件下载到磁盘。
        with filesystem.open(
            filename, "rb", block_size=1024 * 1024,
            cache_type="blockcache", cache_options={"maxblocks": 8},
        ) as remote_file:
            # 同步读取并显式关闭迭代器，避免当前环境中 datasets 的 Parquet
            # 流提前停止后在解释器退出时触发 PyGILState_Release 错误。
            with parquet.ParquetFile(remote_file, pre_buffer=False) as reader:
                with closing(reader.iter_batches(
                    batch_size=128, columns=["text"], use_threads=False,
                )) as batches:
                    for batch in batches:
                        for text in batch.column(0).to_pylist():
                            yield {"text": text}


def token_count(value):
    """至少保留两个 token，才能构造输入和下一 token 目标。"""
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError("token 数量必须至少为 2")
    return count


def write_split(documents, enc, target_tokens, filename, split):
    """消费同一个文档迭代器，写出指定长度的 token 分片。"""
    buffer = np.empty(target_tokens, dtype=np.uint16)
    position = 0
    document_count = 0

    with tqdm(total=target_tokens, desc=split, unit="token") as progress:
        while position < target_tokens:
            try:
                document = next(documents)
            except StopIteration as exc:
                raise RuntimeError(
                    f"数据源提前结束：{split} 需要 {target_tokens:,} 个 token，"
                    f"实际只收集到 {position:,} 个。"
                ) from exc

            text = document["text"]
            if not text.strip():
                continue

            # 与原版一致：每篇文章前加 <|endoftext|>，正文使用 encode_ordinary。
            # GPT-2 的 token ID 范围为 0..50256，可以安全地保存为 uint16。
            token_ids = [enc.eot_token] + enc.encode_ordinary(text)
            count = min(len(token_ids), target_tokens - position)
            buffer[position : position + count] = token_ids[:count]
            position += count
            document_count += 1
            progress.update(count)

            # 达到上限后，不把这篇文章剩余的 token 带入下一个集合。

    np.save(filename, buffer, allow_pickle=False)
    print(f"已处理 {split}：{position:,} token，{document_count:,} 篇非空文档")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--train-tokens", type=token_count, default=10_000_000,
        help="训练集 token 数量（默认：10000000）",
    )
    parser.add_argument(
        "--val-tokens", type=token_count, default=1_000_000,
        help="验证集 token 数量（默认：1000000）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="输出目录；自定义后需同步修改 train.py 的 data_root",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    # 不覆盖旧数据，也不混合不同规模的分片。
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        parser.error(f"输出路径必须不存在或为空目录：{output_dir}")

    total_tokens = args.train_tokens + args.val_tokens
    print(f"数据源：{DATASET_ID} / {DATASET_CONFIG}")
    print(f"输出：{output_dir}")
    print(f"目标：{total_tokens:,} token，数组约 {total_tokens * 2 / 1e6:.1f} MB")
    print("正在初始化分词器与数据流；首次访问需要联网。", flush=True)

    enc = tiktoken.get_encoding("gpt2")
    # 只创建一次迭代器：验证集用前面的文档，训练集从后续文档继续。
    documents = stream_documents()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # 先在相邻临时目录中处理；两份数据都成功后再整体重命名。
    # 普通异常或 Ctrl+C 会清理临时文件，不留下可被误用的半成品。
    with TemporaryDirectory(
        prefix=f".{output_dir.name}-preparing-", dir=output_dir.parent
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        try:
            for index, (split, count) in enumerate(
                [("val", args.val_tokens), ("train", args.train_tokens)]
            ):
                filename = staging_dir / f"edufineweb_{split}_{index:06d}.npy"
                write_split(documents, enc, count, filename, split)
        finally:
            # 即使中途报错，也在解释器退出前关闭远程文件和读取器。
            documents.close()

        # 同一文件系统内重命名目录；目标目录若已被其他进程写入，会报错。
        staging_dir.replace(output_dir)

    print(f"数据准备完成：{output_dir}")
    print("训练前请确认 train.py 中的 data_root 指向此目录。")


if __name__ == "__main__":
    main()
