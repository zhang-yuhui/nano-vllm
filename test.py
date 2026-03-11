from pathlib import Path
import time
import os
from dotenv import load_dotenv
# has to before import torch
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")

import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    from flash_attn import flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("flash-attn package not found – skipping GPU flash attention benchmark.")


def load_decode_extension() -> None:
    root = Path(__file__).resolve().parent
    build_dir = root / "build" / "decode_ext"
    matches = sorted(build_dir.glob("decode_ext*.so"))
    print(build_dir)
    if not matches:
        raise FileNotFoundError(
            f"Missing decode extension under {build_dir}. "
            f"Build it first with: python {root / 'build_decode.py'}"
        )
    torch.ops.load_library(str(matches[0]))


load_decode_extension()


# ---------------------------------------------------------------------------
# Attention implementations
# ---------------------------------------------------------------------------

def run_decode_attention(output: torch.Tensor) -> None:
    torch.ops.sgl_kernel.decode_attention_cpu(
        q,
        k_buffer,
        v_buffer,
        output,
        key,
        value,
        loc,
        attn_logits,
        req_to_token,
        b_req_idx,
        b_seq_len,
        sm_scale,
        logit_cap,
    )


def _run_sdpa_forward_decode(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q, start_kv = 0, 0
    for seq_idx in range(seq_lens.shape[0]):
        seq_len_q = 1
        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + seq_len_q
        end_kv = start_kv + seq_len_kv

        per_req_query = query[:, start_q:end_q, :]
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        per_req_out = (
            scaled_dot_product_attention(
                per_req_query.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out
        start_q, start_kv = end_q, end_kv

    return output


def run_sdpa_attention(output: torch.Tensor) -> None:
    _run_sdpa_forward_decode(
        q,
        output,
        k_buffer,
        v_buffer,
        req_to_token,
        b_req_idx,
        b_seq_len,
        scaling=sm_scale,
        enable_gqa=enable_gqa,
    )


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------

def profile_latency(
    fn,
    label: str,
    out_device: torch.device,
    warmup: int = 10,
    iters: int = 100,
) -> torch.Tensor:
    """
    Profiles `fn(output_tensor)` on the given device.
    Uses CUDA synchronisation barriers when profiling on GPU so that
    host-side timings accurately capture kernel completion.
    """
    is_cuda = out_device.type == "cuda"
    out = torch.zeros(B, H_Q, D_V, dtype=dtype, device=out_device)

    # Warmup
    for _ in range(warmup):
        fn(out)
    if is_cuda:
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(out)
        if is_cuda:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1e3)

    t = torch.tensor(latencies_ms, dtype=torch.float64)
    print(f"\n[{label}]  warmup={warmup}, iters={iters}")
    print(
        f"  latency_ms "
        f"avg={t.mean().item():.3f} "
        f"min={t.min().item():.3f} "
        f"p50={t.quantile(0.50).item():.3f} "
        f"p90={t.quantile(0.90).item():.3f} "
        f"p99={t.quantile(0.99).item():.3f} "
        f"max={t.max().item():.3f}"
    )
    return t


# ---------------------------------------------------------------------------
# Setup – CPU tensors (sgl_kernel + SDPA)
# ---------------------------------------------------------------------------
device = torch.device("cpu")
dtype = torch.float16
# batch, num_q_heads, num_kv_heads, head_dimention
B, H_Q, H_KV, D, D_V = 1, 32, 8, 128, 128

seq_len = 8192
total_tokens = B * seq_len
sm_scale = 1.0 / (D**0.5)
logit_cap = 0.0
num_kv_splits = 8
enable_gqa = H_Q != H_KV

#  [1, 32, 128]
q = torch.randn(B, H_Q, D, dtype=dtype, device=device)
# [1, ]
k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)

key = torch.randn(B, H_KV, D, dtype=dtype)
value = torch.randn(B, H_KV, D_V, dtype=dtype)
loc = torch.randint(0, 10, (B,)).to(torch.int64)

k_buffer[loc] = key
v_buffer[loc] = value

req_to_token = (
    torch.arange(total_tokens, device=device)
    .reshape(B, seq_len)
    .to(torch.int32)
)
b_req_idx = torch.arange(B, device=device).to(torch.int64)
b_seq_len = torch.full((B,), seq_len, device=device).to(torch.int64)

attn_logits = torch.empty(
    (B, H_Q, num_kv_splits, D_V + 1),
    dtype=torch.float32,
    device=device,
)

# Non-contiguous tensors (matches the test harness)
k_buffer = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
v_buffer = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)
q = q.transpose(0, 1).contiguous().transpose(0, 1)
key = key.transpose(0, 1).contiguous().transpose(0, 1)
value = value.transpose(0, 1).contiguous().transpose(0, 1)

# ---------------------------------------------------------------------------
# Setup – GPU tensors (flash attention)
#
# flash_attn_with_kvcache layout:
#   q          : (batch, seqlen_q, nheads,   headdim)
#   k/v_cache  : (batch, seqlen_k, nheads_k, headdim)
#   output     : (batch, seqlen_q, nheads,   headdim)
#
# For decode seqlen_q == 1, so we unsqueeze dim-1 on q and squeeze it back
# on the output.  The per-batch contiguous KV cache is built by gathering
# the paged CPU buffer through req_to_token once during setup.
# ---------------------------------------------------------------------------
if HAS_FLASH_ATTN and torch.cuda.is_available():
    device_gpu = torch.device("cuda")

    # q: (B, H_Q, D) -> (B, 1, H_Q, D)
    q_fa = q.to(device_gpu).unsqueeze(1).contiguous()

    # Materialise the paged KV cache as a dense per-batch tensor on GPU
    k_cache_fa = torch.zeros(B, seq_len, H_KV, D,   dtype=torch.float16, device=device_gpu)
    v_cache_fa = torch.zeros(B, seq_len, H_KV, D_V, dtype=torch.float16, device=device_gpu)
    for b in range(B):
        tokens = req_to_token[b, :seq_len]          # (seq_len,)
        k_cache_fa[b] = k_buffer[tokens].to(device_gpu)   # (seq_len, H_KV, D)
        v_cache_fa[b] = v_buffer[tokens].to(device_gpu)   # (seq_len, H_KV, D_V)

    cache_seqlens_gpu = b_seq_len.to(device_gpu).to(torch.int32)

    def run_flash_attention(output: torch.Tensor) -> None:
        # out: (B, 1, H_Q, D_V)
        out = flash_attn_with_kvcache(
            q_fa,
            k_cache_fa,
            v_cache_fa,
            cache_seqlens=cache_seqlens_gpu,
            softmax_scale=sm_scale,
        )
        output.copy_(out.squeeze(1))  # -> (B, H_Q, D_V)

# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------
o_sgl  = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
o_sdpa = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

run_decode_attention(o_sgl)
run_sdpa_attention(o_sdpa)

cos_sim = torch.nn.functional.cosine_similarity(
    o_sgl.flatten().float(), o_sdpa.flatten().float(), dim=0
)
print(f"\nCorrectness — sgl_kernel vs SDPA  cosine_sim={cos_sim.item():.6f}")
torch.testing.assert_close(o_sgl, o_sdpa, atol=3e-2, rtol=1e-6)
print("  ✓ outputs match within tolerance")

if HAS_FLASH_ATTN and torch.cuda.is_available():
    o_fa = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device_gpu)
    run_flash_attention(o_fa)
    o_fa_cpu = o_fa.cpu()

    cos_sim_fa = torch.nn.functional.cosine_similarity(
        o_sgl.flatten().float(), o_fa_cpu.flatten().float(), dim=0
    )
    print(f"\nCorrectness — sgl_kernel vs flash_attn  cosine_sim={cos_sim_fa.item():.6f}")
    torch.testing.assert_close(o_sgl, o_fa_cpu, atol=3e-2, rtol=1e-6)
    print("  ✓ outputs match within tolerance")

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
print(
    f"\nConfig: B={B} H_Q={H_Q} H_KV={H_KV} D={D} D_V={D_V} "
    f"seq_len={seq_len} dtype={dtype}"
)
print(
    "NOTE: sgl_kernel and torch SDPA run on CPU; "
    "flash_attn runs on GPU – cross-device numbers reflect wall-clock "
    "latency including any dispatch overhead, not raw compute throughput."
)

sgl_times  = profile_latency(run_decode_attention, "sgl_kernel  (CPU)", device)
sdpa_times = profile_latency(run_sdpa_attention,   "torch SDPA  (CPU)", device)

cpu_speedup = sdpa_times.mean() / sgl_times.mean()
print(
    f"\n  [CPU] SDPA / sgl_kernel = {cpu_speedup:.2f}x  "
    f"({'sgl_kernel faster' if cpu_speedup > 1 else 'SDPA faster'})"
)

if HAS_FLASH_ATTN and torch.cuda.is_available():
    fa_times = profile_latency(run_flash_attention, "flash_attn  (GPU)", device_gpu)

    print(
        f"\n  [cross-device] sgl_kernel / flash_attn = "
        f"{sgl_times.mean() / fa_times.mean():.2f}x  "
        f"({'GPU faster' if fa_times.mean() < sgl_times.mean() else 'CPU sgl_kernel faster'})"
    )
    print(
        f"  [cross-device] SDPA       / flash_attn = "
        f"{sdpa_times.mean() / fa_times.mean():.2f}x  "
        f"({'GPU faster' if fa_times.mean() < sdpa_times.mean() else 'CPU SDPA faster'})"
    )