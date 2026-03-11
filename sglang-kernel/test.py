from pathlib import Path
import time

import torch


def load_decode_extension() -> None:
    root = Path(__file__).resolve().parent
    build_dir = root / "build" / "decode_ext"
    matches = sorted(build_dir.glob("decode_ext*.so"))
    print(build_dir)
    if not matches:
        raise FileNotFoundError(
            f"Missing decode extension under {build_dir}. Build it first with: python {root / 'build_decode.py'}"
        )
    torch.ops.load_library(str(matches[0]))


load_decode_extension()


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


def profile_decode_latency(warmup: int = 10, iters: int = 100) -> None:
    for _ in range(warmup):
        run_decode_attention(o)

    latencies_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        run_decode_attention(o)
        latencies_ms.append((time.perf_counter() - start) * 1e3)

    latency_tensor = torch.tensor(latencies_ms, dtype=torch.float64)
    print(f"warmup={warmup}, iters={iters}")
    print(
        "latency_ms "
        f"avg={latency_tensor.mean().item():.3f} "
        f"min={latency_tensor.min().item():.3f} "
        f"p50={latency_tensor.quantile(0.50).item():.3f} "
        f"p90={latency_tensor.quantile(0.90).item():.3f} "
        f"p99={latency_tensor.quantile(0.99).item():.3f} "
        f"max={latency_tensor.max().item():.3f}"
    )

device = torch.device("cpu")
dtype = torch.float16
B, H_Q, H_KV, D, D_V = 1, 32, 8, 128, 128


seq_len = 8192
total_tokens = B * seq_len
sm_scale = 1.0 / (D**0.5)
logit_cap = 0.0
num_kv_splits = 8
enable_gqa = H_Q != H_KV

# q represents the new token being generated, one per batch
q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

# k_buffer and v_buffer represent all previous tokens
k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)

key = torch.randn(B, H_KV, D, dtype=dtype)
value = torch.randn(B, H_KV, D_V, dtype=dtype)
loc = torch.randint(0, 10, (B,)).to(torch.int64)

# set kv cache
k_buffer[loc] = key
v_buffer[loc] = value

# o will have the same shape as q
o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

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

# k_buffer, v_buffer, query, key and value supports non-contiguous tensors
k_buffer = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
v_buffer = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)
q = q.transpose(0, 1).contiguous().transpose(0, 1)
key = key.transpose(0, 1).contiguous().transpose(0, 1)
value = value.transpose(0, 1).contiguous().transpose(0, 1)

run_decode_attention(o)
profile_decode_latency()
