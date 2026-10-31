"""独立的 GPT 推理实验文件，不导入或执行 train.py。

运行普通推理与测速：python3 inference.py --mode no_cache
测试训练结果：加上 --checkpoint log/full_swiglu_rmsnorm/model_final.pt。
KV cache 入口留给你实现：generate_with_kv_cache。
不指定 --checkpoint 时仍使用随机权重，便于单独做推理实现实验。
支持 train_experiment.py 的 tied / untied 检查点，按保存的模式恢复参数共享关系。
"""

import argparse
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# 与训练共用可写的词表缓存，避免推理时重新联网下载。
os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken"))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x,layer_past=None,use_cache=False):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 如果有历史KV缓存，在序列维度dim=2上拼接
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v) if use_cache else None

        is_causal = (layer_past is None) and (T > 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y),present


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 8 * config.n_embd // 3
        self.c_fc = nn.Linear(config.n_embd, hidden_dim)
        self.c_gate = nn.Linear(config.n_embd, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = F.silu(self.c_gate(x)) * self.c_fc(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x, layer_past=None, use_cache=False):
        attn_out,present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x,present


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
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
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx,kv_cache=None,use_cache=False):
        # 仅返回 logits，不接收训练 targets，也不计算 loss。
        B, T = idx.size()
        #计算已经缓存的历史序列past_length
        past_length = kv_cache[0][0].size(2) if kv_cache is not None else 0
        assert past_length +T <= self.config.block_size, (
            f"Cannot forward sequence of length {past_length +T}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(past_length, past_length + T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        new_kv_cache = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = kv_cache[i] if kv_cache is not None else None
            x, present = block(x, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                new_kv_cache.append(present)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits,new_kv_cache
        else:
            return logits


@torch.inference_mode()
def generate_no_cache(model, tokens, max_new_tokens, top_k=50, vocab_size=50257):
    """每步输入完整前缀，返回包含 prompt 和新增 token 的序列。"""
    if tokens.size(1) == 0:
        raise ValueError("prompt 不能为空")
    if max_new_tokens < 1 or top_k < 1 or vocab_size < 1:
        raise ValueError("max_new_tokens、top_k 和 vocab_size 必须大于 0")
    if tokens.size(1) + max_new_tokens > model.config.block_size:
        raise ValueError("prompt 长度 + max_new_tokens 不能超过 block_size")

    xgen = tokens
    for _ in range(max_new_tokens):
        logits = model(xgen)
        # 训练词表填充到了 50304；只采样 tokenizer 能解码的 token。
        logits = logits[:, -1, :vocab_size]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        xgen = torch.cat((xgen, xcol), dim=1)
    return xgen

@torch.inference_mode()
def generate_with_kv_cache(model, tokens, max_new_tokens, top_k=50, vocab_size=50257):
    """TODO：由你实现，输入和返回格式与 generate_no_cache 保持一致。"""
    if tokens.size(1) == 0:
        raise ValueError("prompt 不能为空")
    if max_new_tokens < 1 or top_k < 1 or vocab_size < 1:
        raise ValueError("max_new_tokens、top_k 和 vocab_size 必须大于 0")
    if tokens.size(1) + max_new_tokens > model.config.block_size:
        raise ValueError("prompt 长度 + max_new_tokens 不能超过 block_size")

    xgen = tokens
    kv_cache = None # 初始化kc_cache缓存为空
    #prefill the cache with the prompt tokens
    idx_input = xgen
    for _ in range (max_new_tokens):
        logits ,kv_cache = model(idx_input,kv_cache,use_cache=True)
        logits = logits[:,-1, :vocab_size]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        xgen = torch.cat((xgen, xcol), dim=1)
        #讲下轮循环喂给模型的输入idx_input       
        idx_input = xcol
    return xgen


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def benchmark(model, tokens, generate_fn, max_new_tokens, warmup=1, runs=3, **kwargs):
    """测量完整生成耗时，包含首轮 prompt 计算、后续生成与采样。

    不计模型初始化、分词和文本解码；吞吐量只统计新增 token。
    """
    if warmup < 0 or runs < 1:
        raise ValueError("warmup 不能小于 0，runs 必须大于 0")
    model.eval()
    # CUDA 沿用 train.py 生成时的 BF16 autocast；CPU/MPS 使用 FP32。
    with torch.autocast(
        device_type=tokens.device.type,
        dtype=torch.bfloat16,
        enabled=tokens.device.type == "cuda",
    ):
        for _ in range(warmup):
            generate_fn(model, tokens, max_new_tokens, **kwargs)
        elapsed = []
        for _ in range(runs):
            synchronize(tokens.device)
            start = time.perf_counter()
            output = generate_fn(model, tokens, max_new_tokens, **kwargs)
            synchronize(tokens.device)
            elapsed.append(time.perf_counter() - start)

    mean_seconds = sum(elapsed) / runs
    generated_tokens = tokens.size(0) * max_new_tokens
    print(f"平均生成耗时: {mean_seconds:.4f} s ({runs} 次，不含 {warmup} 次预热)")
    print(f"吞吐量（整个 batch 的新增 token）: {generated_tokens / mean_seconds:.2f} tokens/s")
    return output


def load_checkpoint_model(checkpoint_path):
    """在 CPU 恢复权重及共享关系，同时兼容旧版 GPTConfig 检查点。"""
    # 旧 train.py 保存的类名是 __main__.GPTConfig；无需导入训练脚本。
    # 新实验检查点只用字典、基础类型和 Tensor，可保持 weights_only=True。
    with torch.serialization.safe_globals([(GPTConfig, "__main__.GPTConfig")]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if isinstance(config, dict):
        config = GPTConfig(**config)
    if not isinstance(config, GPTConfig):
        raise ValueError("检查点 config 必须是 GPTConfig 或对应字段的字典")

    experiment_mode = checkpoint.get("experiment_mode", "tied")
    if experiment_mode not in {"tied", "untied_clone", "untied_random"}:
        raise ValueError(f"未知的检查点实验模式: {experiment_mode!r}")
    state = checkpoint["model"]
    saved_e = state["transformer.wte.weight"]
    saved_u = state["lm_head.weight"]
    tied = experiment_mode == "tied"
    if tied and not torch.equal(saved_e, saved_u):
        raise ValueError(
            "检查点声明 tied（旧检查点缺少 experiment_mode 时默认 tied），"
            "但 E/U 权重不同；拒绝把两套权重加载到同一 Parameter。"
        )

    model = GPT(config)
    if not tied:
        # 必须在 load_state_dict 前解除共享；strict=True 不检查参数是否共享。
        model.lm_head.weight = nn.Parameter(model.transformer.wte.weight.detach().clone())
    model.load_state_dict(state, strict=True)
    e = model.transformer.wte.weight
    u = model.lm_head.weight
    if (e is u) != tied or (e.data_ptr() == u.data_ptr()) != tied:
        raise RuntimeError("恢复后的 E/U 参数共享关系与检查点模式不一致")
    if not torch.equal(e.detach(), saved_e) or not torch.equal(u.detach(), saved_u):
        raise RuntimeError("恢复后的 E/U 权重与检查点不一致")

    # 不把 optimizer 等训练状态带入推理流程，函数返回后释放其 CPU 内存。
    info = {
        "step": checkpoint.get("completed_steps", checkpoint.get("step")),
        "experiment_mode": experiment_mode,
    }
    return model, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no_cache", "kv_cache"), default="no_cache")
    parser.add_argument("--checkpoint", help="train.py 或 train_experiment.py 保存的 .pt 路径")
    parser.add_argument("--prompt", default="Hello, I'm a language model,")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default=None, help="例如 cpu、cuda:0；默认自动选择")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.warmup < 0 or args.runs < 1:
        parser.error("batch-size、max-new-tokens、runs 必须大于 0，warmup 不能小于 0")

    device = args.device
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    device = torch.device(device)
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(args.prompt)
    checkpoint_info = None
    if args.checkpoint:
        model, checkpoint_info = load_checkpoint_model(args.checkpoint)
    else:
        model = GPT(GPTConfig(vocab_size=50304))  # 与 train.py 实际实例化的配置一致。
    if not prompt_tokens or len(prompt_tokens) + args.max_new_tokens > model.config.block_size:
        parser.error("prompt 不能为空，且 prompt 长度 + max-new-tokens 不能超过 block_size")
    model.to(device).eval()
    if checkpoint_info is not None:
        print(f"已加载训练权重: {args.checkpoint}，checkpoint step={checkpoint_info['step']}，"
              f"experiment_mode={checkpoint_info['experiment_mode']}")
    else:
        print("当前使用随机初始化权重，仅用于跑通推理和测速，生成文本没有训练效果。")
    print(f"device={device}, mode={args.mode}, batch_size={args.batch_size}, "
          f"prompt_tokens={len(prompt_tokens)}, max_new_tokens={args.max_new_tokens}")

    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(args.batch_size, 1)
    generate_fn = generate_no_cache if args.mode == "no_cache" else generate_with_kv_cache
    output = benchmark(
        model, tokens, generate_fn, args.max_new_tokens,
        warmup=args.warmup, runs=args.runs, vocab_size=enc.n_vocab,
    )
    for i, sequence in enumerate(output.tolist()):
        print(f"sample {i}: {enc.decode(sequence)}")


if __name__ == "__main__":
    main()
