"""离线分析 Embedding / LM Head 快照；不会导入训练入口或启动训练。

运行示例见 experiment_design/IMPLEMENTATION.md。原始快照保持不变，派生的
CSV、PNG、PDF 写到各 run 的 analysis/；多个 run 可另外生成对比图。
geometry_test 是几何拟合留出的 token 身份，不是语言模型未见过的 token。
固定窗口漂移衡量参数净位移，不是单步更新，也不是区间内走过的总路程。
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch


REPORT_GROUPS = ("high", "rare", "uniform", "geometry_test")
VALUE_NAMES = (
    "e_norm", "u_norm", "abs_log_norm_ratio", "matched_cosine",
    "shuffled_cosine", "frequency_matched_cosine", "matched_minus_shuffled",
    "matched_minus_frequency_matched",
)
IDENTITY_COLUMNS = ("run", "mode", "seed", "completed_steps", "tokens_seen")
TOKEN_COLUMNS = (
    *IDENTITY_COLUMNS, "group", "token_id", "corpus_count",
    "shuffled_token_id", "shuffled_corpus_count", "frequency_matched_token_id",
    "frequency_matched_corpus_count", *VALUE_NAMES,
)
GROUP_COLUMNS = (
    *IDENTITY_COLUMNS, "group", "n_tokens",
    *(f"{name}_{stat}" for name in VALUE_NAMES for stat in ("mean", "median", "p10", "p90", "valid_count")),
)
DRIFT_COLUMNS = (
    *IDENTITY_COLUMNS, "group", "from_step", "interval_steps", "interval_tokens",
    "e_relative_net_displacement", "u_relative_net_displacement",
    "e_absolute_net_displacement", "u_absolute_net_displacement",
    "e_row_relative_median", "u_row_relative_median", "net_direction_cosine",
)
GEOMETRY_COLUMNS = (
    *IDENTITY_COLUMNS, "fit_tokens", "test_tokens", "valid_fit_tokens", "valid_test_tokens",
    "raw_test_cosine", "aligned_test_cosine", "aligned_shuffled_test_cosine",
    "aligned_frequency_shuffled_test_cosine", "fit_null_cosine_mean", "fit_null_cosine_min",
    "fit_null_cosine_max", "fit_null_cosines", "rsa_pearson", "rsa_shuffled_pearson",
    "rsa_frequency_shuffled_pearson", "status",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def scalar(value):
    """CSV 空字段 / JSON null 表示无定义；不会把无效余弦记成 0。"""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def csv_row(writer, record):
    writer.writerow({key: scalar(value) for key, value in record.items()})


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else None


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if not valid.size:
        return dict(mean=None, median=None, p10=None, p90=None, valid_count=0)
    p10, median, p90 = np.percentile(valid, [10, 50, 90])
    return dict(mean=float(valid.mean()), median=float(median), p10=float(p10),
                p90=float(p90), valid_count=int(valid.size))


def row_cosine(left, right):
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    result = np.full(left.shape[0], np.nan)
    np.divide(numerator, denominator, out=result, where=valid)
    return np.clip(result, -1.0, 1.0)


def unit_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(norms) & (norms > 0)
    result = np.zeros_like(matrix)
    np.divide(matrix, norms[:, None], out=result, where=valid[:, None])
    return result, valid


def pearson(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return None
    x, y = left[valid], right[valid]
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0 or not np.isfinite(denominator):
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def rsa(left, right):
    left, left_valid = unit_rows(left)
    right, right_valid = unit_rows(right)
    valid = left_valid & right_valid
    if valid.sum() < 3:
        return None
    left, right = left[valid], right[valid]
    upper = np.triu_indices(left.shape[0], k=1)
    return pearson((left @ left.T)[upper], (right @ right.T)[upper])


def tensor_ids(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.int64:
        raise ValueError(f"{name} 必须是一维 int64 Tensor")
    result = value.cpu().numpy()
    if not result.size or np.unique(result).size != result.size:
        raise ValueError(f"{name} 不能为空或包含重复 token ID")
    return result


def load_artifacts(groups_path, counts_path, permuted_fits):
    artifact = torch.load(groups_path, map_location="cpu", weights_only=True)
    counts = torch.load(counts_path, map_location="cpu", weights_only=True)
    if artifact.get("schema_version") != 1:
        raise ValueError("不支持的 token_groups schema_version")
    if not isinstance(counts, torch.Tensor) or counts.ndim != 1 or counts.dtype != torch.int64:
        raise ValueError("token_counts.pt 必须是一维 int64 Tensor")
    if counts.numel() != artifact["vocab_size"] or (counts < 0).any().item():
        raise ValueError("词频长度与词表不一致，或存在负计数")
    expected_counts_hash = artifact.get("counts_sha256")
    if expected_counts_hash is None or expected_counts_hash != sha256_file(counts_path):
        raise ValueError("token_counts.pt 与分组文件记录的 counts_sha256 不一致")
    groups_hash = sha256_file(groups_path)
    manifest_path = groups_path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"不支持的 analysis_data manifest schema_version：{manifest_path}")
    if (manifest["sha256"]["token_groups.pt"] != groups_hash
            or manifest["sha256"]["token_counts.pt"] != expected_counts_hash):
        raise ValueError("分析分组/计数文件与 manifest 的 SHA256 不一致")
    ids = tensor_ids(artifact["token_ids"], "token_ids")
    if not np.all(ids[:-1] < ids[1:]):
        raise ValueError("token_ids 必须严格递增")
    if ids.min() < 0 or ids.max() >= counts.numel():
        raise ValueError("token_ids 超出词表")
    if np.any(ids >= artifact["tokenizer_vocab_size"]) or np.any(ids == artifact["eot_id"]):
        raise ValueError("分析 token 不应包含 EOT 或填充词表行")
    if np.any(counts.numpy()[ids] == 0):
        raise ValueError("分析 token 必须在训练语料中出现过")
    positions = {int(token): index for index, token in enumerate(ids)}
    groups, permutations = {}, {}
    for name in (*REPORT_GROUPS, "geometry_fit"):
        group_ids = tensor_ids(artifact["groups"][name], f"groups.{name}")
        try:
            groups[name] = np.asarray([positions[int(token)] for token in group_ids])
        except KeyError as error:
            raise ValueError(f"分组 {name} 包含不在 token_ids 中的 ID") from error
        permutations[name] = {}
        for kind in ("shuffled", "frequency_matched"):
            permuted = tensor_ids(artifact["permutations"][name][kind], f"{name}.{kind}")
            if not np.array_equal(np.sort(permuted), np.sort(group_ids)) or np.any(permuted == group_ids):
                raise ValueError(f"{name}.{kind} 必须是本组 token 的无自配排列")
            permutations[name][kind] = np.asarray([positions[int(token)] for token in permuted])
    if np.intersect1d(groups["geometry_fit"], groups["geometry_test"]).size:
        raise ValueError("Procrustes fit/test token 身份必须不相交")
    stored_nulls = artifact["fit_null_permutations"]
    if permuted_fits > len(stored_nulls):
        raise ValueError(f"只预存了 {len(stored_nulls)} 个 fit 乱配，不能请求 {permuted_fits} 个")
    fit_ids = ids[groups["geometry_fit"]]
    nulls = []
    for index, value in enumerate(stored_nulls[:permuted_fits]):
        permuted = tensor_ids(value, f"fit_null_permutations[{index}]")
        if not np.array_equal(np.sort(permuted), np.sort(fit_ids)) or np.any(permuted == fit_ids):
            raise ValueError("fit null 必须是 fit token 的无自配排列")
        nulls.append(np.asarray([positions[int(token)] for token in permuted]))
    return dict(ids=ids, counts=counts.numpy(), groups=groups, permutations=permutations,
                nulls=nulls, groups_sha256=groups_hash, manifest_sha256=sha256_file(manifest_path),
                counts_sha256=expected_counts_hash)


def run_signature(config):
    return {name: config[name] for name in (
        "mode", "seed", "head_seed", "total_batch_tokens", "model_config", "training",
        "groups_sha256", "manifest_sha256",
    )}


def history_sources(run_dir, ceiling=None, stack=()):
    """恢复使用新目录：父 run 只取 checkpoint 及以前，子 run 的同一步优先。"""
    run_dir = Path(run_dir).resolve()
    if run_dir in stack:
        raise ValueError(f"parent_run_dir 出现循环引用：{run_dir}")
    config = read_json(run_dir / "config.json")
    if config.get("schema_version") != 1:
        raise ValueError(f"不支持的 run config schema_version：{run_dir}")
    sources = []
    if config.get("parent_run_dir"):
        parent_dir = Path(config["parent_run_dir"])
        if not parent_dir.is_absolute():
            parent_dir = run_dir / parent_dir
        resume_step = int(config["resume_step"])
        if resume_step < 0:
            raise ValueError("resume_step 不能为负数")
        parent_ceiling = resume_step if ceiling is None else min(ceiling, resume_step)
        sources = history_sources(parent_dir, parent_ceiling, (*stack, run_dir))
        if run_signature(sources[-1][2]) != run_signature(config):
            raise ValueError(f"恢复前后的初始化、模型、训练或分析数据配置不一致：{run_dir}")
    sources.append((run_dir, ceiling, config))
    return sources


def validate_sources(sources, artifacts):
    for directory, _, config in sources:
        for key in ("groups_sha256", "manifest_sha256"):
            if config.get(key) != artifacts[key]:
                raise ValueError(f"run 配置的 {key} 与当前分析数据不一致：{directory}")
        if int(config["total_batch_tokens"]) < 1 or int(config["model_config"]["n_embd"]) < 1:
            raise ValueError(f"run 的全局 batch 或模型维度无效：{directory}")


def validate_comparison(histories):
    """对比图只接受相同训练条件；不同 seed 允许，但同 seed 的公共初值应相同。"""
    keys = ("training", "model_config", "total_batch_tokens", "manifest_sha256", "groups_sha256")
    reference = histories[0][-1][2]
    initializations = {}
    for sources in histories:
        directory, _, config = sources[-1]
        for key in keys:
            if config[key] != reference[key]:
                raise ValueError(f"跨 run 对比要求 {key} 相同：{directory}")
        initialization_path = directory / "initialization.json"
        if initialization_path.exists():
            fingerprint = read_json(initialization_path)
            common = {key: fingerprint[key] for key in ("trunk_sha256", "E_sha256")}
            seed = config["seed"]
            if seed in initializations and initializations[seed] != common:
                raise ValueError(f"seed={seed} 的主干或 E 初始权重不一致：{directory}")
            initializations[seed] = common
        else:
            print(f"Initialization fingerprint unavailable: {directory}", flush=True)


def collect_history(sources):
    snapshots, records = {}, {}
    for directory, ceiling, _ in sources:
        for path in sorted((directory / "snapshots").glob("step_*.pt")):
            match = re.fullmatch(r"step_(\d+)\.pt", path.name)
            if match is None:
                raise ValueError(f"快照文件名应为 step_<完成步数>.pt：{path}")
            step = int(match.group(1))
            if ceiling is None or step <= ceiling:
                snapshots[step] = path
        metrics_path = directory / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    event = record["event"]
                    # checkpoint 后的下一次梯度仍标 completed_steps=resume_step，须按更新序号截断。
                    step = int(record["for_update"] if event == "gradient" else record["completed_steps"])
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f"无效指标记录 {metrics_path}:{line_number}") from error
                if ceiling is None or step <= ceiling:
                    records[(event, step)] = record
    if not snapshots:
        raise ValueError("没有找到可分析的 snapshots/step_*.pt")
    return dict(sorted(snapshots.items())), sorted(records.values(), key=lambda row: (row["tokens_seen"], row["event"]))


def load_snapshot(path, expected_step, artifacts, total_batch_tokens):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema_version") != 1 or data.get("groups_sha256") != artifacts["groups_sha256"]:
        raise ValueError(f"快照 schema 或分组 SHA256 不匹配：{path}")
    if int(data["completed_steps"]) != expected_step:
        raise ValueError(f"快照内部步数与文件名不一致：{path}")
    if int(data["tokens_seen"]) != expected_step * total_batch_tokens:
        raise ValueError(f"快照 tokens_seen 与完成步数及全局 batch 不一致：{path}")
    if not np.array_equal(tensor_ids(data["token_ids"], "snapshot.token_ids"), artifacts["ids"]):
        raise ValueError(f"快照 token ID 或顺序不匹配：{path}")
    matrices = []
    for name in ("E", "U"):
        value = data[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.dtype != torch.float32:
            raise ValueError(f"快照 {name} 必须是二维 FP32 Tensor：{path}")
        if value.shape[0] != artifacts["ids"].size:
            raise ValueError(f"快照 {name} 行数不匹配：{path}")
        # FP32 落盘保留细微漂移；离线归一化、差分与 SVD 用 float64。
        matrices.append(value.numpy().astype(np.float64))
    if matrices[0].shape != matrices[1].shape:
        raise ValueError(f"E/U 形状不一致：{path}")
    return dict(completed_steps=expected_step, tokens_seen=int(data["tokens_seen"]),
                E=matrices[0], U=matrices[1])


def group_values(snapshot, group, artifacts):
    indices = artifacts["groups"][group]
    e, u = snapshot["E"][indices], snapshot["U"][indices]
    e_norm, u_norm = np.linalg.norm(e, axis=1), np.linalg.norm(u, axis=1)
    log_ratio = np.full(indices.size, np.nan)
    valid = np.isfinite(e_norm) & np.isfinite(u_norm) & (e_norm > 0) & (u_norm > 0)
    log_ratio[valid] = np.abs(np.log(e_norm[valid]) - np.log(u_norm[valid]))
    matched = row_cosine(e, u)
    shuffled = row_cosine(e, snapshot["U"][artifacts["permutations"][group]["shuffled"]])
    frequency = row_cosine(e, snapshot["U"][artifacts["permutations"][group]["frequency_matched"]])
    return dict(e_norm=e_norm, u_norm=u_norm, abs_log_norm_ratio=log_ratio,
                matched_cosine=matched, shuffled_cosine=shuffled,
                frequency_matched_cosine=frequency, matched_minus_shuffled=matched - shuffled,
                matched_minus_frequency_matched=matched - frequency)


def relative_displacement(before, after):
    # 不完整/无效矩阵不产生“看似正常”的 Frobenius 总量。
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return None, None, None
    delta = after - before
    absolute = float(np.linalg.norm(delta))
    original = float(np.linalg.norm(before))
    row_original = np.linalg.norm(before, axis=1)
    row_relative = np.full(before.shape[0], np.nan)
    np.divide(np.linalg.norm(delta, axis=1), row_original, out=row_relative, where=row_original > 0)
    return absolute / original if original > 0 else None, absolute, summary(row_relative)["median"]


def drift_values(before, after, group, artifacts):
    indices = artifacts["groups"][group]
    e0, e1 = before["E"][indices], after["E"][indices]
    u0, u1 = before["U"][indices], after["U"][indices]
    e_relative, e_absolute, e_median = relative_displacement(e0, e1)
    u_relative, u_absolute, u_median = relative_displacement(u0, u1)
    direction = row_cosine((e1 - e0).reshape(1, -1), (u1 - u0).reshape(1, -1))[0]
    return dict(group=group, from_step=before["completed_steps"],
                interval_steps=after["completed_steps"] - before["completed_steps"],
                interval_tokens=after["tokens_seen"] - before["tokens_seen"],
                e_relative_net_displacement=e_relative, u_relative_net_displacement=u_relative,
                e_absolute_net_displacement=e_absolute, u_absolute_net_displacement=u_absolute,
                e_row_relative_median=e_median, u_row_relative_median=u_median,
                net_direction_cosine=scalar(direction))


def fit_rotation(left, right):
    left, left_valid = unit_rows(left)
    right, right_valid = unit_rows(right)
    valid = left_valid & right_valid
    # 至少要求有效拟合行数超过维度；这并不保证充分性，仍依赖留出集和乱配对照。
    if valid.sum() <= left.shape[1]:
        return None, int(valid.sum())
    u, _, vh = np.linalg.svd(left[valid].T @ right[valid], full_matrices=False)
    return u @ vh, int(valid.sum())


def geometry_values(snapshot, artifacts):
    fit, test = artifacts["groups"]["geometry_fit"], artifacts["groups"]["geometry_test"]
    e, u = snapshot["E"], snapshot["U"]
    test_e, test_u = e[test], u[test]
    rotation, valid_fit = fit_rotation(e[fit], u[fit])
    valid_test = np.isfinite(row_cosine(test_e, test_u))
    record = {key: None for key in GEOMETRY_COLUMNS if key not in IDENTITY_COLUMNS}
    record.update(fit_tokens=int(fit.size), test_tokens=int(test.size),
                  valid_fit_tokens=valid_fit, valid_test_tokens=int(valid_test.sum()),
                  raw_test_cosine=finite_mean(row_cosine(test_e, test_u)),
                  rsa_pearson=rsa(test_e, test_u))
    shuffled_u = u[artifacts["permutations"]["geometry_test"]["shuffled"]]
    frequency_u = u[artifacts["permutations"]["geometry_test"]["frequency_matched"]]
    record.update(rsa_shuffled_pearson=rsa(test_e, shuffled_u),
                  rsa_frequency_shuffled_pearson=rsa(test_e, frequency_u))
    if rotation is None:
        record["status"] = "insufficient_valid_fit_rows"
        return record
    aligned = test_e @ rotation
    null_cosines = []
    for null_fit in artifacts["nulls"]:
        null_rotation, _ = fit_rotation(e[fit], u[null_fit])
        value = finite_mean(row_cosine(test_e @ null_rotation, test_u)) if null_rotation is not None else None
        null_cosines.append(value)
    finite_nulls = [value for value in null_cosines if value is not None]
    record.update(aligned_test_cosine=finite_mean(row_cosine(aligned, test_u)),
                  aligned_shuffled_test_cosine=finite_mean(row_cosine(aligned, shuffled_u)),
                  aligned_frequency_shuffled_test_cosine=finite_mean(row_cosine(aligned, frequency_u)),
                  fit_null_cosine_mean=finite_mean(finite_nulls),
                  fit_null_cosine_min=min(finite_nulls) if finite_nulls else None,
                  fit_null_cosine_max=max(finite_nulls) if finite_nulls else None,
                  fit_null_cosines=json.dumps(null_cosines, allow_nan=False), status="ok")
    return record


def curve(axis, rows, field, label, **kwargs):
    points = [(row["tokens_seen"] / 1e9, row.get(field)) for row in rows]
    points = [(x, y) for x, y in points if y is not None and math.isfinite(y)]
    if points:
        x, y = zip(*points)
        axis.plot(x, y, label=label, **kwargs)


def finish_figure(figure, axes, directory, name):
    for axis in np.asarray(axes, dtype=object).reshape(-1):
        axis.set_xlabel("Training tokens (billions)")
        axis.grid(alpha=0.2)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize="small")
        elif not axis.lines:
            axis.text(0.5, 0.5, "No eligible measurements", ha="center", va="center",
                      transform=axis.transAxes, color="gray")
    figure.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        destination = directory / f"{name}.{suffix}"
        temporary = directory / f".{name}.tmp.{suffix}"
        figure.savefig(temporary, dpi=160)
        os.replace(temporary, destination)


def plot_run(result, plt):
    directory, metrics = result["directory"] / "analysis" / "plots", result["metrics"]
    title = result["label"]
    figure, axis = plt.subplots(figsize=(8, 4))
    for event, field in (("train", "train_loss"), ("val", "val_loss")):
        curve(axis, [row for row in metrics if row["event"] == event], field, event)
    axis.set(title=title, ylabel="Cross-entropy loss")
    finish_figure(figure, [axis], directory, "loss")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, group in zip(axes.flat, REPORT_GROUPS):
        rows = [row for row in result["groups"] if row["group"] == group]
        for field, label in (("matched_cosine_mean", "matched"), ("shuffled_cosine_mean", "shuffled"),
                             ("frequency_matched_cosine_mean", "frequency shuffled"),
                             ("matched_minus_frequency_matched_mean", "matched - frequency shuffled")):
            curve(axis, rows, field, label)
        band = [row for row in rows if row["matched_cosine_p10"] is not None]
        if band:
            axis.fill_between([row["tokens_seen"] / 1e9 for row in band],
                              [row["matched_cosine_p10"] for row in band],
                              [row["matched_cosine_p90"] for row in band], alpha=0.12,
                              label="matched token p10-p90")
        axis.set(title=group, ylabel="Cosine / cosine gap")
    figure.suptitle(title)
    finish_figure(figure, axes, directory, "cosine")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    for column, group in enumerate(REPORT_GROUPS):
        rows = [row for row in result["groups"] if row["group"] == group]
        curve(axes[0, column], rows, "e_norm_mean", "E")
        curve(axes[0, column], rows, "u_norm_mean", "U")
        curve(axes[1, column], rows, "abs_log_norm_ratio_mean", "mean")
        curve(axes[1, column], rows, "abs_log_norm_ratio_median", "median")
        axes[0, column].set(title=group, ylabel="Mean row L2 norm")
        axes[1, column].set(ylabel="Absolute log norm ratio")
    figure.suptitle(title)
    finish_figure(figure, axes, directory, "norms")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
    for field, label in (("raw_test_cosine", "raw"), ("aligned_test_cosine", "aligned"),
                         ("aligned_shuffled_test_cosine", "test shuffled"),
                         ("aligned_frequency_shuffled_test_cosine", "test frequency shuffled"),
                         ("fit_null_cosine_mean", "refitted fit-shuffle null")):
        curve(axes[0, 0], result["geometry"], field, label)
    for field, label in (("rsa_pearson", "matched"), ("rsa_shuffled_pearson", "shuffled"),
                         ("rsa_frequency_shuffled_pearson", "frequency shuffled")):
        curve(axes[0, 1], result["geometry"], field, label)
    axes[0, 0].set(title="Procrustes: held-out token identities", ylabel="Test cosine")
    axes[0, 1].set(title="RSA: held-out token identities", ylabel="Pearson correlation")
    figure.suptitle(title)
    finish_figure(figure, axes, directory, "geometry")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, group in zip(axes.flat, REPORT_GROUPS):
        rows = [row for row in result["drift"] if row["group"] == group]
        curve(axis, rows, "e_relative_net_displacement", "E")
        curve(axis, rows, "u_relative_net_displacement", "U")
        axis.set(title=group, ylabel=f"Relative net displacement ({result['drift_steps']} updates)")
    figure.suptitle(title)
    finish_figure(figure, axes, directory, "drift")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    gradients = [row for row in metrics if row["event"] == "gradient"]
    for axis, group in zip(axes.flat, ("global", *REPORT_GROUPS)):
        rows = []
        for record in gradients:
            values = record.get("global", {}) if group == "global" else record.get("groups", {}).get(group, {})
            rows.append(dict(tokens_seen=record["tokens_seen"], **values))
        for field, label in (("e_row_rms", "E"), ("u_row_rms", "U"), ("shared_row_rms", "shared")):
            curve(axis, rows, field, label)
        axis.set(title=group, ylabel="Pre-clip gradient row RMS")
    curve(axes.flat[-1], [row for row in metrics if row["event"] == "train"], "lr", "learning rate")
    axes.flat[-1].set(title="Learning rate", ylabel="LR")
    figure.suptitle(title)
    finish_figure(figure, axes, directory, "gradients_and_lr")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for group in ("global", *REPORT_GROUPS):
        rows = []
        for record in gradients:
            values = record.get("global", {}) if group == "global" else record.get("groups", {}).get(group, {})
            rows.append(dict(tokens_seen=record["tokens_seen"], cosine=values.get("cosine")))
        curve(axis, rows, "cosine", group)
    axis.set(title="Pre-clip E/U gradient cosine (untied only)", ylabel="Gradient cosine")
    finish_figure(figure, [axis], directory, "gradient_cosine")
    plt.close(figure)


def plot_comparison(results, directory, plt):
    """各模式使用固定颜色；同模式不同 seed/run 仍有独立标签。"""
    colors = {"tied": "tab:blue", "untied_clone": "tab:orange", "untied_random": "tab:green"}
    figure, axis = plt.subplots(figsize=(9, 5))
    for result in results:
        rows = [row for row in result["metrics"] if row["event"] == "val"]
        curve(axis, rows, "val_loss", result["label"], color=colors.get(result["mode"]))
    axis.set(title="Validation loss", ylabel="Cross-entropy loss")
    finish_figure(figure, [axis], directory, "validation_comparison")
    plt.close(figure)
    for kind in ("cosine_gap", "drift"):
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
        for axis, group in zip(axes.flat, REPORT_GROUPS):
            for result in results:
                color = colors.get(result["mode"])
                if kind == "cosine_gap":
                    rows = [row for row in result["groups"] if row["group"] == group]
                    curve(axis, rows, "matched_minus_frequency_matched_mean", result["label"], color=color)
                else:
                    rows = [row for row in result["drift"] if row["group"] == group]
                    curve(axis, rows, "e_relative_net_displacement", result["label"] + " E", color=color)
                    curve(axis, rows, "u_relative_net_displacement", result["label"] + " U", color=color, linestyle="--")
            axis.set(title=group, ylabel="Matched - frequency-shuffled cosine" if kind == "cosine_gap" else "Relative net displacement")
        finish_figure(figure, axes, directory, kind + "_comparison")
        plt.close(figure)


def analyze_run(run_dir, artifacts, args):
    sources = history_sources(run_dir)
    validate_sources(sources, artifacts)
    directory, _, config = sources[-1]
    snapshots, metrics = collect_history(sources)
    output_dir = directory / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"{config['mode']}/seed_{config['seed']}/{directory.name}"
    result = dict(directory=directory, label=label, mode=config["mode"], metrics=metrics,
                  groups=[], geometry=[], drift=[], drift_steps=args.drift_steps)
    paths = {
        "token_metrics": TOKEN_COLUMNS, "token_groups": GROUP_COLUMNS,
        "drift": DRIFT_COLUMNS, "geometry": GEOMETRY_COLUMNS,
    }
    handles, writers = {}, {}
    try:
        for name, columns in paths.items():
            handles[name] = (output_dir / f".{name}.tmp.csv").open("w", encoding="utf-8", newline="")
            writers[name] = csv.DictWriter(handles[name], fieldnames=columns)
            writers[name].writeheader()
        for step, path in snapshots.items():
            snapshot = load_snapshot(path, step, artifacts, int(config["total_batch_tokens"]))
            if snapshot["E"].shape[1] != int(config["model_config"]["n_embd"]):
                raise ValueError(f"快照维度与 model_config 不一致：{path}")
            identity = dict(run=str(directory), mode=config["mode"], seed=config["seed"],
                            completed_steps=step, tokens_seen=snapshot["tokens_seen"])
            for group in REPORT_GROUPS:
                indices = artifacts["groups"][group]
                values = group_values(snapshot, group, artifacts)
                for row_index, token_index in enumerate(indices):
                    token_id = int(artifacts["ids"][token_index])
                    record = dict(identity, group=group, token_id=token_id,
                                  corpus_count=int(artifacts["counts"][token_id]))
                    for kind in ("shuffled", "frequency_matched"):
                        paired_index = artifacts["permutations"][group][kind][row_index]
                        paired_id = int(artifacts["ids"][paired_index])
                        record[f"{kind}_token_id"] = paired_id
                        record[f"{kind}_corpus_count"] = int(artifacts["counts"][paired_id])
                    record.update({key: value[row_index] for key, value in values.items()})
                    csv_row(writers["token_metrics"], record)
                group_record = dict(identity, group=group, n_tokens=int(indices.size))
                for key, value in values.items():
                    group_record.update({f"{key}_{stat}": number for stat, number in summary(value).items()})
                csv_row(writers["token_groups"], group_record)
                result["groups"].append(group_record)
            if step - args.drift_steps in snapshots:
                before = load_snapshot(snapshots[step - args.drift_steps], step - args.drift_steps,
                                       artifacts, int(config["total_batch_tokens"]))
                for group in REPORT_GROUPS:
                    drift_record = dict(identity, **drift_values(before, snapshot, group, artifacts))
                    csv_row(writers["drift"], drift_record)
                    result["drift"].append(drift_record)
                del before
            if step in (min(snapshots), max(snapshots)) or step % args.geometry_every == 0:
                geometry_record = dict(identity, **geometry_values(snapshot, artifacts))
                csv_row(writers["geometry"], geometry_record)
                result["geometry"].append(geometry_record)
                print(f"{label}: analyzed completed_steps={step}, tokens_seen={snapshot['tokens_seen']}", flush=True)
            del snapshot
    finally:
        for handle in handles.values():
            handle.close()
    # 只有 CSV 全部计算完成才替换旧派生结果；原始材料始终只读。
    for name in paths:
        os.replace(output_dir / f".{name}.tmp.csv", output_dir / f"{name}.csv")
    result["manifest"] = dict(
        schema_version=1, status="complete", run_dir=str(directory),
        mode=config["mode"], seed=config["seed"], total_batch_tokens=config["total_batch_tokens"],
        model_config=config["model_config"],
        source_runs=[dict(run_dir=str(path), through_step=ceiling) for path, ceiling, _ in sources],
        groups_sha256=artifacts["groups_sha256"], counts_sha256=artifacts["counts_sha256"],
        manifest_sha256=artifacts["manifest_sha256"],
        analyzer_sha256=sha256_file(__file__), snapshot_count=len(snapshots),
        first_step=min(snapshots), last_step=max(snapshots), drift_steps=args.drift_steps,
        geometry_every=args.geometry_every, permuted_fits=args.permuted_fits,
        interpretation="Geometry tests hold out token identities from fitting, not from LM training; drift is fixed-window net displacement.",
    )
    return result


def write_json_atomic(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="一个或多个实验输出目录")
    parser.add_argument("--groups", type=Path, default=Path("log/embedding_head/analysis_data/token_groups.pt"))
    parser.add_argument("--counts", type=Path, default=None, help="默认读取 groups 同目录的 token_counts.pt")
    parser.add_argument("--compare-dir", type=Path, help="可选：将跨 run 对比图写入此目录")
    parser.add_argument("--drift-steps", type=int, default=250, help="只比较恰好相隔这些 optimizer updates 的快照")
    parser.add_argument("--geometry-every", type=int, default=1000, help="每隔这些 updates 计算几何指标，另包含首末快照")
    parser.add_argument("--permuted-fits", type=int, default=3, help="使用预处理时固定保存的 fit 乱配数量")
    args = parser.parse_args()
    if min(args.drift_steps, args.geometry_every, args.permuted_fits) < 1:
        parser.error("drift-steps、geometry-every、permuted-fits 必须为正数")
    if len({path.resolve() for path in args.run_dir}) != len(args.run_dir):
        parser.error("run-dir 不能重复")
    if args.counts is None:
        args.counts = args.groups.parent / "token_counts.pt"
    # 无显示服务器也能离线画图；依赖检查在写入分析结果之前完成。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts = load_artifacts(args.groups, args.counts, args.permuted_fits)
    histories = [history_sources(run_dir) for run_dir in args.run_dir]
    for sources in histories:
        validate_sources(sources, artifacts)
    if args.compare_dir:
        validate_comparison(histories)
    results = []
    for run_dir in args.run_dir:
        analysis_dir = run_dir.resolve() / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        # 重算失败时，不能让上次的 complete 状态冒充本次结果。
        write_json_atomic(analysis_dir / "manifest.json", dict(schema_version=1, status="running"))
        try:
            result = analyze_run(run_dir, artifacts, args)
            plot_run(result, plt)
            write_json_atomic(analysis_dir / "manifest.json", result["manifest"])
        except Exception as error:
            write_json_atomic(analysis_dir / "manifest.json", dict(schema_version=1, status="failed", error=str(error)))
            raise
        results.append(result)
        print(f"Saved analysis: {result['directory'] / 'analysis'}", flush=True)
    if args.compare_dir:
        plot_comparison(results, args.compare_dir, plt)
        write_json_atomic(args.compare_dir / "manifest.json", dict(
            schema_version=1, runs=[str(result["directory"]) for result in results],
            groups_sha256=artifacts["groups_sha256"], drift_steps=args.drift_steps,
        ))
        print(f"Saved comparison: {args.compare_dir}", flush=True)


if __name__ == "__main__":
    main()
