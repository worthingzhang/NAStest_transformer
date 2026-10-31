"""训练期采集：只保存标量和固定 token 行，不在 GPU 训练循环里做几何拟合。

由 rank 0 创建。梯度统计在最后一次累积 backward/DDP 同步后、clip 前调用；
权重快照在 optimizer.step 后调用。所有快照使用 FP32，保留微小位移。
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch


EARLY_STEPS = frozenset((0, 1, 2, 5, 10, 20, 50, 100))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_json(value):
    """JSON 不接受 NaN/Infinity；无定义的余弦也用 null 表示。"""
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as file:
            temporary = Path(file.name)
            torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def gradient_stats(e_grad, u_grad, tied):
    """row_rms = sqrt(mean_i ||g_i||²)，方便比较大小不同的 token 桶。"""
    result = dict.fromkeys(("e_norm", "u_norm", "e_row_rms", "u_row_rms", "cosine", "shared_norm", "shared_row_rms"))
    if tied:
        norm = torch.linalg.vector_norm(e_grad).item()
        result.update(shared_norm=norm, shared_row_rms=norm / math.sqrt(e_grad.shape[0]))
        return result
    e_norm = torch.linalg.vector_norm(e_grad).item()
    u_norm = torch.linalg.vector_norm(u_grad).item()
    cosine = None
    if e_norm > 0 and u_norm > 0:
        cosine = (torch.dot(e_grad.reshape(-1), u_grad.reshape(-1)) / (e_norm * u_norm)).item()
    result.update(
        e_norm=e_norm,
        u_norm=u_norm,
        e_row_rms=e_norm / math.sqrt(e_grad.shape[0]),
        u_row_rms=u_norm / math.sqrt(u_grad.shape[0]),
        cosine=cosine,
    )
    return result


class DynamicsTracker:
    def __init__(self, run_dir, groups_path, snapshot_every=250, grad_every=100):
        if snapshot_every < 1 or grad_every < 1:
            raise ValueError("snapshot_every 和 grad_every 必须为正")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.run_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.artifact_hash = file_sha256(groups_path)
        artifact = torch.load(groups_path, map_location="cpu", weights_only=True)
        if artifact.get("schema_version") != 1:
            raise ValueError("不支持的 token_groups.pt schema_version")
        self.token_ids = artifact["token_ids"]
        self.groups = artifact["groups"]
        if self.token_ids.dtype != torch.int64 or self.token_ids.ndim != 1 or not self.token_ids.numel():
            raise ValueError("token_ids 必须是非空一维 int64 Tensor")
        if not torch.equal(self.token_ids, torch.unique(self.token_ids, sorted=True)):
            raise ValueError("token_ids 必须升序且不重复")
        if int(self.token_ids.min()) < 0 or int(self.token_ids.max()) >= artifact["eot_id"]:
            raise ValueError("观测 token 包含非法、特殊或填充 ID")
        for name, ids in self.groups.items():
            if ids.dtype != torch.int64 or ids.ndim != 1 or not ids.numel():
                raise ValueError(f"非法 token 分组：{name}")
            if not torch.isin(ids, self.token_ids).all().item():
                raise ValueError(f"分组 ID 不在快照观测集合中：{name}")
        self.snapshot_every = snapshot_every
        self.grad_every = grad_every

    def record(self, record):
        """每行独立 JSON；即时关闭文件，减少训练意外结束时丢失的记录。"""
        with self.metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(clean_json(record), ensure_ascii=False, allow_nan=False) + "\n")

    def should_snapshot(self, completed_steps, final=False):
        return final or completed_steps in EARLY_STEPS or completed_steps % self.snapshot_every == 0

    @torch.no_grad()
    def snapshot(self, model, completed_steps, tokens_seen, force=False):
        if not self.should_snapshot(completed_steps, final=force):
            return None
        if completed_steps < 0 or tokens_seen < 0:
            raise ValueError("快照的完成步数和累计 token 数不能为负")
        embedding = model.transformer.wte.weight
        head = model.lm_head.weight
        ids = self.token_ids.to(embedding.device)
        snapshot = {
            "schema_version": 1,
            "completed_steps": completed_steps,
            "tokens_seen": tokens_seen,
            "token_ids": self.token_ids.clone(),
            "E": embedding.index_select(0, ids).to(device="cpu", dtype=torch.float32),
            "U": head.index_select(0, ids).to(device="cpu", dtype=torch.float32),
            "groups_sha256": self.artifact_hash,
        }
        path = self.snapshot_dir / f"step_{completed_steps:07d}.pt"
        if path.exists():
            previous = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("schema_version", "completed_steps", "tokens_seen", "groups_sha256"):
                if previous.get(key) != snapshot[key]:
                    raise ValueError(f"已有快照元数据不一致，拒绝覆盖：{path} ({key})")
            for key in ("token_ids", "E", "U"):
                if not torch.equal(previous[key], snapshot[key]):
                    raise ValueError(f"已有快照张量不一致，拒绝覆盖：{path} ({key})")
            return path
        atomic_torch_save(snapshot, path)
        return path

    @torch.no_grad()
    def gradients(self, model, for_update, tokens_seen, tied):
        if for_update < 1 or tokens_seen < 0:
            raise ValueError("for_update 从 1 开始，tokens_seen 不能为负")
        if for_update not in EARLY_STEPS and for_update % self.grad_every != 0:
            return None
        embedding = model.transformer.wte.weight
        head = model.lm_head.weight
        if (embedding is head) != tied:
            raise ValueError("传入的 tied 模式与模型实际参数共享关系不同")
        if embedding.grad is None or head.grad is None:
            raise ValueError("需要在 backward 完成后、梯度清零前采集梯度")
        # 当前模型使用 FP32 参数/梯度。避免为全词表创建乘法临时矩阵。
        e_grad = embedding.grad.detach()
        u_grad = head.grad.detach()
        result = {
            "event": "gradient",
            "for_update": for_update,
            "completed_steps": for_update - 1,
            "tokens_seen": tokens_seen,
            "tied": tied,
            "scope": "pre_clip; global includes all model vocabulary rows including padding",
            "global": gradient_stats(e_grad, u_grad, tied),
            "groups": {},
        }
        for name in ("high", "rare", "uniform", "geometry_test"):
            ids = self.groups[name].to(e_grad.device)
            e_rows = e_grad.index_select(0, ids)
            u_rows = e_rows if tied else u_grad.index_select(0, ids)
            result["groups"][name] = gradient_stats(e_rows, u_rows, tied)
        return clean_json(result)
