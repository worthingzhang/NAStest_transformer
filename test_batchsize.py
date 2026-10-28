"""独立的 GPT-2 batch size 探索入口，保持 train.py 原样。

模型定义是当前 train.py 的副本；没有 import train，避免触发原脚本的顶层训练。
固定 T=1024，A = total_batch_tokens / (B * T * world_size)。
benchmark 模式固定有效 batch，比较显存与吞吐；train 模式按 token 预算比较验证 loss。
每次运行保存参数、源码快照/哈希、数据哈希、环境、逐步指标、结果，不复用旧目录。

示例（先指定你被分配的 GPU）：
CUDA_VISIBLE_DEVICES=2,3,4,6 torchrun --standalone --nproc_per_node=4 \
    test_batchsize.py --batch-size 8
CUDA_VISIBLE_DEVICES=2,3,4,6 torchrun --standalone --nproc_per_node=4 \
    test_batchsize.py --mode train --batch-size 8 --total-batch-tokens 65536

benchmark 默认预热 10 次参数更新，测量 50 次；不评估或保存模型。
train 默认累计处理约 3 遍训练数据，每 1048576 token 评估固定 262144 token。
--save-checkpoints 可保存 best/last；这些文件不包含完整断点恢复功能。
性能结论只适用于相同模型、精度、序列长度、GPU 分配和通信环境。
"""

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
#-----------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT=1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
            # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            # output projection
            y = self.c_proj(y)
            return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# ---------------------------------------------------------------------------
# 实验工具：以下实现与基线训练入口独立。
import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import traceback

import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

master_process = True
SEQUENCE_LENGTH = 1024


class DataLoaderLite:
    """各 rank 共用全局游标，仅在取数据时加入 rank 偏移。"""

    def __init__(self, data_dir, B, T, rank, world_size, split):
        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.global_batch_tokens = B * T * world_size
        self.shards = sorted(Path(data_dir).glob(f"*{split}*.npy"))
        if not self.shards:
            raise ValueError(f"没有找到 {split} 数据：{data_dir}")
        self.lengths = []
        for path in self.shards:
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
                raise ValueError(f"需要一维整数 token 数组：{path}")
            if data.size < self.global_batch_tokens + 1:
                raise ValueError(f"分片太小，至少需要 {self.global_batch_tokens + 1} token：{path}")
            self.lengths.append(data.size)
        self.total_tokens = sum(self.lengths)
        self.usable_tokens = sum(
            ((n - 1) // self.global_batch_tokens) * self.global_batch_tokens
            for n in self.lengths
        )
        self.reset()

    def _load_shard(self):
        # 与原脚本一致：当前分片放在 CPU，转换为 torch.long。
        array = np.load(self.shards[self.current_shard], allow_pickle=False)
        self.tokens = torch.tensor(array.astype(np.int32), dtype=torch.long)

    def reset(self):
        self.current_shard = 0
        self.global_position = 0
        self.cycles = 0
        self._load_shard()

    def next_batch(self):
        if self.global_position + self.global_batch_tokens + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.cycles += int(self.current_shard == 0)
            self.global_position = 0
            self._load_shard()
        start = self.global_position + self.rank * self.B * self.T
        buf = self.tokens[start : start + self.B * self.T + 1]
        x, y = buf[:-1].view(self.B, self.T), buf[1:].view(self.B, self.T)
        self.global_position += self.global_batch_tokens
        return x, y


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["benchmark", "train"], default="benchmark")
    parser.add_argument("--batch-size", type=positive_int, default=8, help="每张卡的 B")
    parser.add_argument("--total-batch-tokens", type=positive_int, default=None, help="全局 Q；benchmark 默认 131072，train 默认 65536")
    parser.add_argument("--warmup-steps", type=positive_int, default=10, help="性能测量前的预热步数")
    parser.add_argument("--measure-steps", type=positive_int, default=50)
    parser.add_argument("--train-passes", type=float, default=3.0, help="按 token 数计算的等效遍数")
    parser.add_argument("--max-steps", type=positive_int, help="覆盖 train 模式的训练步数")
    parser.add_argument("--eval-every-tokens", type=positive_int, default=1048576)
    parser.add_argument("--val-tokens", type=positive_int, default=262144)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-checkpoints", action="store_true", help="train 模式保存 best.pt 和 last.pt")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "edu_fineweb10B")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "log" / "batchsize")
    args = parser.parse_args()
    if not math.isfinite(args.train_passes) or args.train_passes <= 0:
        parser.error("--train-passes 必须为有限正数")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须为有限正数")
    if args.mode == "benchmark" and args.max_steps is not None:
        parser.error("--max-steps 仅用于 train；性能测试请设置 --measure-steps")
    if args.total_batch_tokens is None:
        args.total_batch_tokens = 131072 if args.mode == "benchmark" else 65536
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(run_dir, row):
    if master_process:
        with (run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def gpu_command(*arguments):
    result = subprocess.run(
        ["nvidia-smi", *arguments], capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def learning_rate(step, max_steps, max_lr, benchmark):
    if benchmark:
        return max_lr
    warmup = max(1, math.ceil(0.05 * max_steps))
    if step < warmup:
        return max_lr * (step + 1) / warmup
    ratio = min(1.0, (step - warmup) / max(1, max_steps - 1 - warmup))
    return max_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * ratio)))


@torch.no_grad()
def evaluate(model, loader, device, steps, distributed):
    model.eval()
    loader.reset()
    loss_sum = torch.zeros((), device=device)
    for _ in range(steps):
        x, y = loader.next_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x.to(device), y.to(device))
        loss_sum += loss.detach() / steps
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
    result = loss_sum.item()
    if not math.isfinite(result):
        raise FloatingPointError(f"验证 loss 非有限：{result}")
    return result


def train_step(model, optimizer, loader, device, accum_steps, lr, distributed):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_sum = torch.zeros((), device=device)
    for micro_step in range(accum_steps):
        # 前向和反向都放在 no_sync 作用域；只在最后一个小批次同步梯度。
        context = model.no_sync() if distributed and micro_step < accum_steps - 1 else nullcontext()
        with context:
            x, y = loader.next_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x.to(device), y.to(device))
            loss = loss / accum_steps
            loss_sum += loss.detach()
            loss.backward()
    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_sum, norm


def run(args):
    global master_process
    if not torch.cuda.is_available():
        raise RuntimeError("这个实验入口需要 CUDA GPU")
    distributed = int(os.environ.get("RANK", "-1")) >= 0
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_process = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=2), device_id=device)
    run_dir = None
    try:
        B, T, Q = args.batch_size, SEQUENCE_LENGTH, args.total_batch_tokens
        micro_tokens = B * T * world_size
        if Q % micro_tokens:
            raise ValueError(f"Q={Q} 必须能被 B*T*W={micro_tokens} 整除")
        accum_steps = Q // micro_tokens
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if master_process else None
        if distributed:
            stamps = [stamp]
            dist.broadcast_object_list(stamps, src=0)
            stamp = stamps[0]
        run_dir = args.output_root.resolve() / f"{args.mode}_B{B}_Q{Q}_W{world_size}_{stamp}"
        if master_process:
            run_dir.mkdir(parents=True, exist_ok=False)
            print(f"RUN_DIR={run_dir}", flush=True)
        if distributed:
            dist.barrier()

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        train_loader = DataLoaderLite(args.data_dir, B, T, rank, world_size, "train")
        val_loader = None
        if args.mode == "train":
            val_loader = DataLoaderLite(args.data_dir, B, T, rank, world_size, "val")
            if args.val_tokens % micro_tokens or args.val_tokens > val_loader.usable_tokens:
                raise ValueError("验证 token 预算必须能被 B*T*W 整除，且不能超过可用验证数据")
            if args.eval_every_tokens % Q:
                raise ValueError("验证间隔 token 数必须能被 Q 整除")
        benchmark = args.mode == "benchmark"
        max_steps = (
            args.warmup_steps + args.measure_steps if benchmark else
            args.max_steps or math.ceil(args.train_passes * train_loader.total_tokens / Q)
        )
        raw_model = GPT(GPTConfig(vocab_size=50304))
        raw_model.to(device)
        model = DDP(raw_model, device_ids=[local_rank]) if distributed else raw_model
        optimizer = raw_model.configure_optimizers(0.1, args.lr, "cuda")

        if master_process:
            script = Path(__file__).resolve()
            baseline = script.with_name("train.py")
            settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            settings.update({
                "T": T, "world_size": world_size, "grad_accum_steps": accum_steps,
                "max_steps": max_steps, "train_tokens": train_loader.total_tokens,
                "usable_train_tokens_per_cycle": train_loader.usable_tokens,
                "model_config": vars(raw_model.config),
                "model_parameters": sum(p.numel() for p in raw_model.parameters()),
                "precision": "bf16 autocast; FP32 model and AdamW state",
                "compile": False, "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": sys.version, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "seed": args.seed, "script_sha256": sha256(script),
                "baseline_sha256": sha256(baseline) if baseline.exists() else None,
                "launch_args": sys.argv,
                "gpu_inventory": gpu_command("--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"),
                "gpu_topology": gpu_command("topo", "-m"),
                "communication_env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
                "data": [{"file": str(p.resolve()), "tokens": n, "sha256": sha256(p)}
                         for p, n in zip(train_loader.shards, train_loader.lengths)],
            })
            if val_loader is not None:
                settings["validation_data"] = [
                    {"file": str(p.resolve()), "tokens": n, "sha256": sha256(p)}
                    for p, n in zip(val_loader.shards, val_loader.lengths)
                ]
            write_json(run_dir / "config.json", settings)
            (run_dir / "source.py").write_bytes(script.read_bytes())
            write_json(run_dir / "result.json", {"status": "running"})
            print(f"B={B} T={T} W={world_size} A={accum_steps} Q={Q} steps={max_steps}", flush=True)
        if distributed:
            dist.barrier()

        measured = []
        benchmark_start = None
        best_val = float("inf")
        last_val = None
        if not benchmark:
            last_val = evaluate(model, val_loader, device, args.val_tokens // micro_tokens, distributed)
            record(run_dir, {"kind": "validation", "step": 0, "tokens_seen": 0, "loss": last_val})
            if master_process:
                print(f"validation step=0 loss={last_val:.6f}", flush=True)

        for step in range(max_steps):
            if distributed:
                dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if benchmark and step == args.warmup_steps:
                benchmark_start = time.perf_counter()
            start = time.perf_counter()
            lr = learning_rate(step, max_steps, args.lr, benchmark)
            loss, norm = train_step(model, optimizer, train_loader, device, accum_steps, lr, distributed)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            stats = torch.tensor([
                elapsed, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(),
            ], device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            elapsed, allocated, reserved = stats.tolist()
            loss_value, norm_value = loss.item(), norm.item()
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise FloatingPointError(f"非有限 loss/gradient：{loss_value}, {norm_value}")
            row = {
                "kind": "train", "step": step + 1, "tokens_seen": (step + 1) * Q,
                "equivalent_passes": (step + 1) * Q / train_loader.total_tokens,
                "measured": benchmark and step >= args.warmup_steps,
                "loss": loss_value, "lr": lr, "grad_norm": norm_value,
                "seconds": elapsed, "tokens_per_second": Q / elapsed,
                "peak_allocated_gib": allocated / 1024**3,
                "peak_reserved_gib": reserved / 1024**3,
            }
            record(run_dir, row)
            if row["measured"]:
                measured.append(row)
            if master_process and (step == 0 or (step + 1) % 10 == 0 or step + 1 == max_steps):
                print(f"step {step + 1}/{max_steps} loss={loss_value:.4f} "
                      f"tok/s={Q / elapsed:.0f} allocated={allocated / 1024**3:.2f} GiB "
                      f"reserved={reserved / 1024**3:.2f} GiB", flush=True)

            completed = step + 1
            if not benchmark and (completed * Q % args.eval_every_tokens == 0 or completed == max_steps):
                last_val = evaluate(model, val_loader, device, args.val_tokens // micro_tokens, distributed)
                improved = last_val < best_val
                best_val = min(best_val, last_val)
                record(run_dir, {"kind": "validation", "step": completed, "tokens_seen": completed * Q, "loss": last_val})
                if master_process:
                    print(f"validation step={completed} loss={last_val:.6f}", flush=True)
                    if args.save_checkpoints and (improved or completed == max_steps):
                        checkpoint = {
                            "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "config": vars(raw_model.config), "step": completed,
                            "tokens_seen": completed * Q, "val_loss": last_val,
                            "settings": settings,
                        }
                        if improved:
                            torch.save(checkpoint, run_dir / "best.pt")
                        if completed == max_steps:
                            torch.save(checkpoint, run_dir / "last.pt")

        if benchmark:
            torch.cuda.synchronize()
            wall_seconds = torch.tensor(time.perf_counter() - benchmark_start, device=device)
            if distributed:
                dist.all_reduce(wall_seconds, op=dist.ReduceOp.MAX)
            durations = [row["seconds"] for row in measured]
            result = {
                "status": "completed", "mode": args.mode, "B": B, "T": T, "W": world_size,
                "Q": Q, "A": accum_steps, "warmup_steps": args.warmup_steps,
                "measured_steps": len(measured), "measured_tokens": len(measured) * Q,
                "tokens_per_second": len(measured) * Q / sum(durations),
                "end_to_end_tokens_per_second": len(measured) * Q / wall_seconds.item(),
                "median_step_seconds": statistics.median(durations),
                "p95_step_seconds": float(np.percentile(durations, 95)),
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in measured),
                "peak_reserved_gib": max(row["peak_reserved_gib"] for row in measured),
                "last_train_loss": loss_value,
                "timing": "tokens_per_second excludes barriers, metric reductions and log writes; end_to_end includes them",
            }
        else:
            result = {
                "status": "completed", "mode": args.mode, "B": B, "T": T, "W": world_size,
                "Q": Q, "A": accum_steps, "steps": max_steps, "tokens_seen": max_steps * Q,
                "equivalent_passes": max_steps * Q / train_loader.total_tokens,
                "best_validation_loss": best_val, "last_validation_loss": last_val,
            }
        if master_process:
            write_json(run_dir / "result.json", result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        if run_dir is not None and run_dir.exists():
            write_json(run_dir / f"error_rank_{rank}.json", {
                "status": "failed", "rank": rank, "type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(),
            })
            if master_process:
                write_json(run_dir / "result.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())

