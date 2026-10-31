"""Embedding / LM Head 实验：保持 train.py 的模型与显式训练循环。

四张物理卡 0、4、5、6 的启动示例（不会由本文件自动启动其他进程）：
CUDA_VISIBLE_DEVICES=0,4,5,6 torchrun --standalone --nproc_per_node=4 \
    train_experiment.py --mode tied --run-dir log/embedding_head/seed_1337/tied/run1

先运行 build_freq_buckets.py 准备固定观测集合。--stop-after 只提前停止，
不缩短学习率计划。恢复时使用 --resume checkpoints/last.pt 和新的 --run-dir。
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dynamics_tracker import DynamicsTracker

os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken"))


# 模型数学与当前 train.py 相同；实验期间固定此副本，不引入新的模型结构。
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 8 * config.n_embd // 3
        self.c_fc = nn.Linear(config.n_embd, hidden_dim)
        self.c_gate = nn.Linear(config.n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd, eps=1e-5),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # named_parameters 自动去重 tied 参数；解绑必须在创建 optimizer 前完成。
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95),
                                 eps=1e-8, fused=fused and device_type == "cuda")


def set_experiment_mode(model, mode, head_seed):
    """先构造共同 tied 初值，再仅更改 U；独立 generator 不消耗训练 RNG。"""
    if mode not in {"tied", "untied_clone", "untied_random"}:
        raise ValueError(f"Unknown experiment mode: {mode}")
    if mode != "tied":
        model.lm_head.weight = nn.Parameter(model.transformer.wte.weight.detach().clone())
        if mode == "untied_random":
            generator = torch.Generator(device="cpu").manual_seed(head_seed)
            with torch.no_grad():
                model.lm_head.weight.normal_(mean=0.0, std=0.02, generator=generator)
    E, U = model.transformer.wte.weight, model.lm_head.weight
    assert (E is U) == (mode == "tied")
    assert (E.data_ptr() == U.data_ptr()) == (mode == "tied")


class DataLoaderLite:
    """与基线相同的全局游标、顺序分片读取；添加显式目录和恢复游标。"""
    def __init__(self, B, T, rank, world_size, data_dir, split):
        self.B, self.T = B, T
        self.rank, self.world_size = rank, world_size
        self.shards = sorted(p for p in Path(data_dir).glob("*.npy") if split in p.name)
        if not self.shards:
            raise ValueError(f"No {split} shards in {data_dir}")
        sizes = []
        for path in self.shards:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1 or array.dtype.kind not in "iu":
                raise ValueError(f"Expected one-dimensional integer tokens: {path}")
            if array.size < B * T * world_size + 1:
                raise ValueError(f"Shard too short for a global microbatch: {path}")
            sizes.append(array.size)
        self.num_tokens = sum(sizes)
        self.reset()

    def _load(self):
        array = np.load(self.shards[self.current_shard], allow_pickle=False)
        self.tokens = torch.tensor(array.astype(np.int32), dtype=torch.long)

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self._load()

    def next_batch(self):
        start = self.current_position + self.B * self.T * self.rank
        buf = self.tokens[start:start + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.world_size
        if self.current_position + self.B * self.T * self.world_size + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = 0
            self._load()
        return x, y

    def state_dict(self):
        return {"current_shard": self.current_shard, "current_position": self.current_position}

    def load_state_dict(self, state):
        self.current_shard = int(state["current_shard"])
        self.current_position = int(state["current_position"])
        if not 0 <= self.current_shard < len(self.shards):
            raise ValueError("Checkpoint shard index is out of bounds")
        self._load()
        stride = self.B * self.T * self.world_size
        if self.current_position < 0 or self.current_position % stride != 0:
            raise ValueError("Checkpoint loader position is invalid")
        if self.current_position + stride + 1 > len(self.tokens):
            raise ValueError("Checkpoint loader position cannot supply a global batch")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def check_analysis_data(data_dir, groups_path):
    """不重复扫描十亿 token；核对预处理材料哈希和所有分片元数据。"""
    directory = groups_path.parent
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("vocab_size") != GPTConfig.vocab_size:
        raise ValueError("Analysis manifest schema or vocabulary does not match this model")
    if groups_path.name != "token_groups.pt":
        raise ValueError("--groups must point to token_groups.pt in its manifest directory")
    for name in ("token_groups.pt", "token_counts.pt"):
        if file_sha256(directory / name) != manifest["sha256"][name]:
            raise ValueError(f"Analysis artifact hash mismatch: {name}")
    for split in ("train", "val"):
        actual = sorted(p for p in data_dir.glob("*.npy") if split in p.name)
        expected = manifest[f"{split}_shards"]
        if [p.name for p in actual] != [item["name"] for item in expected]:
            raise ValueError(f"{split} shards differ from the frequency manifest")
        for path, item in zip(actual, expected):
            stat = path.stat()
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]
                    or array.size != item["tokens"] or str(array.dtype) != item["dtype"]):
                raise ValueError(f"Shard changed since frequency preprocessing: {path}")
    return manifest


def initialization_fingerprint(model):
    """初始化证据：三组 E/主干哈希应一致，Clone 的 E/U 哈希应相等。"""
    trunk = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name in {"transformer.wte.weight", "lm_head.weight"}:
            continue
        trunk.update(name.encode())
        trunk.update(parameter.detach().contiguous().numpy().tobytes())
    return {
        "scope": "common initialization, before loading any resume checkpoint",
        "trunk_sha256": trunk.hexdigest(),
        "E_sha256": hashlib.sha256(model.transformer.wte.weight.detach().numpy().tobytes()).hexdigest(),
        "U_sha256": hashlib.sha256(model.lm_head.weight.detach().numpy().tobytes()).hexdigest(),
    }


def get_lr(step, args):
    if step < args.warmup_steps:
        return args.max_lr * (step + 1) / args.warmup_steps
    ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * min(ratio, 1.0)))
    return args.min_lr + coefficient * (args.max_lr - args.min_lr)


def autocast(device):
    # 与基线 CUDA BF16 相同；CPU 只供未来显式验证使用，不会自动退回 CPU 长训。
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, loader, batches, device, ddp):
    model.eval()
    loader.reset()
    loss_sum = torch.zeros((), device=device)
    for _ in range(batches):
        x, y = loader.next_batch()
        with autocast(device):
            logits, loss = model(x.to(device), y.to(device))
        loss_sum += loss.detach() / batches
        del logits, loss
    if ddp:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
    return loss_sum.item()


@torch.no_grad()
def evaluate_hellaswag(raw_model, device, rank, world_size, ddp):
    # 延迟导入：禁用此辅助评估时不需要分词缓存或网络请求。
    from hellaswag import iterate_examples, render_example
    raw_model.eval()
    correct, total = 0, 0
    for index, example in enumerate(iterate_examples("val")):
        if index % world_size != rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with autocast(device):
            logits, _ = raw_model(tokens)
        losses = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                 tokens[:, 1:].contiguous().view(-1), reduction="none")
        losses = losses.view(tokens.size(0), -1)
        mask = mask[:, 1:]
        prediction = ((losses * mask).sum(1) / mask.sum(1)).argmin().item()
        correct += int(prediction == label)
        total += 1
    counts = torch.tensor([correct, total], dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    correct, total = counts.tolist()
    return {"hellaswag_accuracy": correct / total, "correct": correct, "total": total}


def rng_state(device):
    # 将 numpy 状态转成基础类型和 tensor，使 checkpoint 可用 weights_only=True 加载。
    name, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [name, torch.from_numpy(keys.astype(np.int64)), position, has_gauss, cached_gauss],
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), position, has_gauss, cached_gauss))
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(raw_model, optimizer, loader, completed_steps, settings, device,
                    rank, world_size, ddp, run_dir, elapsed_seconds):
    # 所有 rank 都必须进入此函数；只有 rank 0 写文件。
    state = rng_state(device)
    states = [None] * world_size if rank == 0 else None
    if ddp:
        dist.gather_object(state, states, dst=0)
    else:
        states = [state]
    if rank == 0:
        checkpoint = {
            "schema_version": 1, "model": raw_model.state_dict(),
            "config": asdict(raw_model.config), "experiment_mode": settings["mode"],
            "step": completed_steps, "completed_steps": completed_steps,
            "tokens_seen": completed_steps * settings["total_batch_tokens"],
            "optimizer": optimizer.state_dict(), "loader_state": loader.state_dict(),
            "rng_by_rank": states, "settings": settings, "run_dir": str(run_dir),
            "elapsed_seconds": elapsed_seconds,
        }
        atomic_save(run_dir / "checkpoints" / "last.pt", checkpoint)
    if ddp:
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tied", "untied_clone", "untied_random"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="新建的独立运行目录，不覆盖已有结果")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_ROOT", "edu_fineweb10B")))
    parser.add_argument("--groups", type=Path, default=Path("log/embedding_head/analysis_data/token_groups.pt"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--head-seed", type=int, default=None, help="默认 seed+1；只控制 random 组 U 初始化")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--total-batch-tokens", type=int, default=524288)
    parser.add_argument("--max-steps", type=int, default=19073, help="完整训练及 LR 衰减计划的长度")
    parser.add_argument("--stop-after", type=int, default=None, help="在累计完成这些更新后停止，不改变 LR")
    parser.add_argument("--warmup-steps", type=int, default=715)
    parser.add_argument("--max-lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-tokens", type=int, default=1310720, help="固定验证 token 预算；默认四卡20批")
    parser.add_argument("--snapshot-every", type=int, default=250)
    parser.add_argument("--grad-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--hellaswag-every", type=int, default=5000, help="0禁用；否则在阶段点和本次结束评估")
    parser.add_argument("--resume", type=Path, help="本入口保存的完整 last.pt；需提供新的 run-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    positive = ("batch_size", "sequence_length", "total_batch_tokens", "max_steps", "val_every",
                "val_tokens", "snapshot_every", "grad_every", "checkpoint_every")
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("batch、步数、记录间隔和 token 预算必须为正数")
    if not 0 <= args.warmup_steps < args.max_steps:
        parser.error("需要 0 <= warmup-steps < max-steps")
    if not 0 < args.min_lr <= args.max_lr or args.weight_decay < 0:
        parser.error("需要 0 < min-lr <= max-lr 且 weight-decay >= 0")
    if args.sequence_length > GPTConfig.block_size or args.hellaswag_every < 0:
        parser.error("sequence-length 不能超过 1024；hellaswag-every 不能为负")
    if args.stop_after is not None and not 0 < args.stop_after <= args.max_steps:
        parser.error("stop-after 必须在 1 到 max-steps 之间")
    args.head_seed = args.seed + 1 if args.head_seed is None else args.head_seed
    args.run_dir, args.data_dir, args.groups = (p.resolve() for p in (args.run_dir, args.data_dir, args.groups))
    return args


def main():
    args = parse_args()
    args.owns_run_dir = False
    ddp = int(os.environ.get("RANK", -1)) != -1
    rank = int(os.environ["RANK"]) if ddp else 0
    local_rank = int(os.environ["LOCAL_RANK"]) if ddp else 0
    world_size = int(os.environ["WORLD_SIZE"]) if ddp else 1
    if ddp and args.device != "cuda":
        raise ValueError("DDP uses NCCL and requires --device cuda")
    if args.device == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError("CUDA unavailable or LOCAL_RANK exceeds visible GPUs")
        # CUDA_VISIBLE_DEVICES=0,4,5,6 后，cuda:0..3 分别对应物理 0,4,5,6。
        # 不能在这里将 LOCAL_RANK 再映射为物理编号，否则会选择错误设备。
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if ddp:
        dist.init_process_group(backend="nccl", device_id=device)
    try:
        run_training(args, device, rank, local_rank, world_size, ddp)
    except Exception as error:
        # 拒绝复用旧 run-dir 时，连错误日志也不能写进别人的已有结果。
        if args.owns_run_dir:
            atomic_json(args.run_dir / f"error_rank_{rank}.json",
                        {"rank": rank, "error": repr(error), "status": "failed"})
            if rank == 0:
                atomic_json(args.run_dir / "result.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


def run_training(args, device, rank, local_rank, world_size, ddp):
    master = rank == 0
    B, T, Q = args.batch_size, args.sequence_length, args.total_batch_tokens
    global_microbatch = B * T * world_size
    if Q % global_microbatch or args.val_tokens % global_microbatch:
        raise ValueError("total-batch-tokens 和 val-tokens 必须能被 B*T*WORLD_SIZE 整除")
    accum_steps = Q // global_microbatch
    manifest = check_analysis_data(args.data_dir, args.groups)
    model_config = GPTConfig()
    source_dir = Path(__file__).resolve().parent
    source_names = ("train_experiment.py", "dynamics_tracker.py", "build_freq_buckets.py",
                    "analyze_dynamics.py", "inference.py", "train.py", "fineweb.py", "hellaswag.py")
    source_hashes = {name: file_sha256(source_dir / name) for name in source_names}
    training = {name: getattr(args, name) for name in (
        "batch_size", "sequence_length", "max_steps", "warmup_steps", "max_lr", "min_lr",
        "weight_decay", "val_tokens", "val_every", "snapshot_every", "grad_every",
        "hellaswag_every", "checkpoint_every")}
    training.update(world_size=world_size, grad_accum_steps=accum_steps,
                    precision="bf16_autocast" if device.type == "cuda" else "fp32",
                    optimizer="AdamW", betas=[0.9, 0.95], eps=1e-8, clip_norm=1.0, compile=False)
    settings = {
        "schema_version": 1, "mode": args.mode, "seed": args.seed, "head_seed": args.head_seed,
        "total_batch_tokens": Q, "model_config": asdict(model_config), "training": training,
        "data_dir": str(args.data_dir), "groups_path": str(args.groups),
        "groups_sha256": manifest["sha256"]["token_groups.pt"],
        "manifest_sha256": file_sha256(args.groups.parent / "manifest.json"),
        "source_sha256": source_hashes, "parent_run_dir": None, "resume_step": 0,
        "stop_after": args.stop_after, "run_dir": str(args.run_dir),
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "cuda": torch.version.cuda, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                        "device": str(device), "device_name": torch.cuda.get_device_name(device)
                        if device.type == "cuda" else "CPU"},
    }
    checkpoint = None
    start_step = 0
    previous_elapsed = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        for key in ("optimizer", "loader_state", "rng_by_rank", "settings", "completed_steps"):
            if key not in checkpoint:
                raise ValueError(f"Resume needs a full experiment checkpoint, missing {key}")
        old = checkpoint["settings"]
        for key in ("mode", "seed", "head_seed", "total_batch_tokens", "model_config", "training",
                    "groups_sha256", "manifest_sha256"):
            if settings[key] != old[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        for name in ("train_experiment.py", "dynamics_tracker.py"):
            if source_hashes[name] != old["source_sha256"][name]:
                raise ValueError(f"Resume training source changed: {name}")
        if len(checkpoint["rng_by_rank"]) != world_size:
            raise ValueError("Resume requires the original world size")
        start_step = int(checkpoint["completed_steps"])
        settings.update(parent_run_dir=checkpoint["run_dir"], resume_step=start_step,
                        resume_checkpoint=str(args.resume.resolve()))
        previous_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
    stop_step = args.max_steps if args.stop_after is None else args.stop_after
    if not 0 <= start_step < stop_step:
        raise ValueError("stop-after/max-steps must be greater than the checkpoint's completed steps")

    # 只由 rank 0 创建目录；已有目录一律不复用，恢复也保留原始轨迹。
    error = [None]
    if master:
        try:
            args.run_dir.mkdir(parents=True, exist_ok=False)
            args.owns_run_dir = True
            (args.run_dir / "source").mkdir()
            (args.run_dir / "checkpoints").mkdir()
            for name in source_names:
                shutil.copy2(source_dir / name, args.run_dir / "source" / name)
            shutil.copy2(args.groups.parent / "manifest.json", args.run_dir / "data_manifest.json")
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_dir,
                                 capture_output=True, text=True, check=False)
            settings["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
            atomic_json(args.run_dir / "config.json", settings)
            atomic_json(args.run_dir / "result.json", {"status": "running", "completed_steps": start_step})
        except Exception as exception:
            error[0] = repr(exception)
    if ddp:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    args.owns_run_dir = True

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    train_loader = DataLoaderLite(B, T, rank, world_size, args.data_dir, "train")
    val_loader = DataLoaderLite(B, T, rank, world_size, args.data_dir, "val")
    model = GPT(model_config)
    set_experiment_mode(model, args.mode, args.head_seed)
    if master:
        fingerprint = initialization_fingerprint(model)
        if args.mode != "untied_random" and fingerprint["E_sha256"] != fingerprint["U_sha256"]:
            raise RuntimeError("Tied/clone initialization must have identical E/U values")
        atomic_json(args.run_dir / "initialization.json", fingerprint)
    if checkpoint is not None:
        if checkpoint.get("experiment_mode") != args.mode or checkpoint["config"] != asdict(model_config):
            raise ValueError("Checkpoint model metadata does not match experiment settings")
        if args.mode == "tied" and not torch.equal(checkpoint["model"]["transformer.wte.weight"],
                                                  checkpoint["model"]["lm_head.weight"]):
            raise ValueError("Tied checkpoint contains inconsistent E/U values")
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    optimizer = raw_model.configure_optimizers(args.weight_decay, args.max_lr, device.type)
    resume_rng = None
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_loader.load_state_dict(checkpoint["loader_state"])
        resume_rng = checkpoint["rng_by_rank"][rank]
        del checkpoint
    E, U = raw_model.transformer.wte.weight, raw_model.lm_head.weight
    assert (E is U) == (args.mode == "tied")
    assert any(p is U for group in optimizer.param_groups for p in group["params"])
    if args.hellaswag_every:
        from hellaswag import download
        if master:
            download("val")
        if ddp:
            dist.barrier()
    tracker = DynamicsTracker(args.run_dir, args.groups, args.snapshot_every, args.grad_every) if master else None
    if master:
        print(f"mode={args.mode}, visible GPUs={os.getenv('CUDA_VISIBLE_DEVICES', 'all')}, "
              f"world_size={world_size}, B={B}, T={T}, accumulation={accum_steps}, Q={Q}")
        print(f"parameters={sum(p.numel() for p in raw_model.parameters()):,}; run_dir={args.run_dir}")
        tracker.snapshot(raw_model, start_step, start_step * Q, force=True)
    if ddp:
        dist.barrier()
    wall_start = time.perf_counter()
    # 验证位置统一为已完成的更新数；最后一次验证一定在最后一次更新之后。
    val_loss = evaluate(model, val_loader, args.val_tokens // global_microbatch, device, ddp)
    if master:
        tracker.record({"event": "val", "completed_steps": start_step, "tokens_seen": start_step * Q,
                        "val_loss": val_loss, "val_tokens": args.val_tokens})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # 恢复 RNG 放在导入、缓存准备、验证等初始化工作之后，紧邻下一次训练。
    if resume_rng is not None:
        restore_rng(resume_rng, device)

    for step in range(start_step, stop_step):
        completed = step + 1
        if ddp:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        iteration_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        for micro_step in range(accum_steps):
            x, y = train_loader.next_batch()
            # no_sync 必须同时包住 forward 和 backward，最后一次正常同步。
            sync = model.no_sync() if ddp and micro_step < accum_steps - 1 else nullcontext()
            with sync:
                with autocast(device):
                    logits, loss = model(x.to(device), y.to(device))
                loss = loss / accum_steps
                loss_sum += loss.detach()
                loss.backward()
                del logits, loss
        if ddp:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        before_logging = time.perf_counter()
        gradient_record = None
        if master:
            gradient_record = tracker.gradients(raw_model, completed, step * Q, args.mode == "tied")
        if ddp:
            dist.barrier()  # rank 0 采集期间其他 rank 等待；开销单独记录。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_logging = time.perf_counter()
        grad_seconds = after_logging - before_logging
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_end = time.perf_counter()
        train_seconds = update_end - iteration_start - grad_seconds
        # 完整分布式更新耗时取最慢 rank，不将验证/快照计入训练吞吐。
        duration = torch.tensor(train_seconds, dtype=torch.float64, device=device)
        if ddp:
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        train_seconds = duration.item()
        norm_value = norm.item()
        if master:
            tracker.record({
                "event": "train", "completed_steps": completed, "tokens_seen": completed * Q,
                "lr": lr, "train_loss": loss_sum.item(), "global_grad_norm": norm_value,
                "clip_coefficient": min(1.0, 1.0 / (norm_value + 1e-6)),
                "train_seconds": train_seconds, "gradient_logging_seconds": grad_seconds,
                "tokens_per_second": Q / train_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda" else None,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30
                if device.type == "cuda" else None,
                "memory_scope": "rank0_since_initial_validation",
            })
            if gradient_record is not None:
                gradient_record.update(lr=lr, clip_coefficient=min(1.0, 1.0 / (norm_value + 1e-6)))
                tracker.record(gradient_record)
            print(f"step {completed:5d} | loss {loss_sum.item():.6f} | lr {lr:.4e} | "
                  f"norm {norm_value:.4f} | tok/s {Q / train_seconds:.0f}", flush=True)
            snapshot_start = time.perf_counter()
            path = tracker.snapshot(raw_model, completed, completed * Q, force=completed == stop_step)
            if path is not None:
                tracker.record({"event": "snapshot", "completed_steps": completed,
                                "tokens_seen": completed * Q,
                                "snapshot_seconds": time.perf_counter() - snapshot_start})
        # 采集只在 rank 0；下一次 collective 前让各 rank 对齐。
        if ddp:
            dist.barrier()
        if completed % args.val_every == 0 or completed == stop_step:
            val_start = time.perf_counter()
            val_loss = evaluate(model, val_loader, args.val_tokens // global_microbatch, device, ddp)
            if master:
                tracker.record({"event": "val", "completed_steps": completed,
                                "tokens_seen": completed * Q, "val_loss": val_loss,
                                "val_tokens": args.val_tokens, "val_seconds": time.perf_counter() - val_start})
        if args.hellaswag_every and (completed % args.hellaswag_every == 0 or completed == stop_step):
            hella_start = time.perf_counter()
            # 各 rank 的题数可不相等，使用 raw_model 避免 DDP forward collective 不配对。
            result = evaluate_hellaswag(raw_model, device, rank, world_size, ddp)
            if master:
                tracker.record({"event": "hellaswag", "completed_steps": completed,
                                "tokens_seen": completed * Q, **result,
                                "hellaswag_seconds": time.perf_counter() - hella_start})
        if completed % args.checkpoint_every == 0 or completed == stop_step:
            save_checkpoint(raw_model, optimizer, train_loader, completed, settings, device,
                            rank, world_size, ddp, args.run_dir,
                            previous_elapsed + time.perf_counter() - wall_start)
        if master:
            tracker.record({"event": "timing", "completed_steps": completed,
                            "tokens_seen": completed * Q,
                            "iteration_seconds": time.perf_counter() - iteration_start,
                            "elapsed_seconds": previous_elapsed + time.perf_counter() - wall_start})

    if master:
        # 小一些的 model-only 文件供推理；恢复训练应使用 last.pt。
        atomic_save(args.run_dir / "checkpoints" / "model_final.pt", {
            "schema_version": 1, "model": raw_model.state_dict(), "config": asdict(model_config),
            "experiment_mode": args.mode, "step": stop_step, "completed_steps": stop_step,
            "tokens_seen": stop_step * Q, "val_loss": val_loss,
        })
        atomic_json(args.run_dir / "result.json", {
            "status": "complete" if stop_step == args.max_steps else "stopped",
            "completed_steps": stop_step, "tokens_seen": stop_step * Q,
            "val_loss": val_loss, "elapsed_seconds": previous_elapsed + time.perf_counter() - wall_start,
            "resume_checkpoint": str(args.run_dir / "checkpoints" / "last.pt"),
        })
    if ddp:
        dist.barrier()


if __name__ == "__main__":
    main()
