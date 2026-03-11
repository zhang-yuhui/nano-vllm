"""
Profiling script for nano-vllm Attention.forward() in decoding phase.

Simulates a single sequence generating one new token against an existing KV cache.

Configurable variables (edit these):
  KV_CACHE_MB  -- total KV cache allocation in megabytes
  USE_CPU      -- True  → sequence lives in CPU KV cache (context.cpu_kv_cache=True,
                           context.cache_locations=[BlockLocation.CPU])
                  False → sequence lives in GPU KV cache
  SEQ_LEN      -- number of tokens already stored in the KV cache
"""
import os
from dotenv import load_dotenv
# has to before import torch
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES")
import math
import sys
import time
import types
from tqdm import tqdm
import torch

# The top-level nanovllm/__init__.py does `from nanovllm.llm import LLM` which
# transitively requires tqdm.  We only need the attention layer and context
# utilities, so we register a lightweight stub for the top-level package before
# any submodule is imported.  The stub exposes __path__ so Python can still
# discover and import subpackages normally.
# _pkg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanovllm")
# _nanovllm_stub = types.ModuleType("nanovllm")
# _nanovllm_stub.__path__ = [_pkg_root]
# _nanovllm_stub.__package__ = "nanovllm"
# sys.modules.setdefault("nanovllm", _nanovllm_stub)

from nanovllm.layers.attention import Attention          # noqa: E402
from nanovllm.utils.context import set_context, reset_context  # noqa: E402
from nanovllm.engine.block_location import BlockLocation  # noqa: E402

# ── Configurable parameters ────────────────────────────────────────────────────
KV_CACHE_MB = 1024*9    # total KV cache size in megabytes
USE_CPU     = False  # True → CPU KV cache path; False → GPU KV cache path
SEQ_LEN     = 8000*100  # tokens already in the KV cache; decode adds one more
PROFILE = False
PROFILE_EXPORT_ITERS = 10
# ── Model architecture constants (Qwen 0.6/4B) ────────────────────────────────
# attention_cpu() has hard-coded assertions for these values:
#   q.shape in {(32, 128), (16, 128)}, k_cache.shape[1] == 8, num_groups in {2, 4}
NUM_HEADS    = 16
NUM_KV_HEADS = 8
HEAD_DIM     = 128
DTYPE        = torch.float16

# ── Block-size constants (must match engine config) ────────────────────────────
GPU_BLOCK_SIZE = 256   # tokens per GPU KV cache block
CPU_BLOCK_SIZE = 1024 * 8 *100  # tokens per CPU KV cache block

# ── Profiling settings ─────────────────────────────────────────────────────────
WARMUP_ITERS  = 5
PROFILE_ITERS = 20


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_num_blocks(kv_cache_mb: float, block_size: int) -> int:
    """Return the number of paged-attention blocks that fit in kv_cache_mb MB."""
    element_bytes = 2  # float16
    bytes_per_block = 2 * block_size * NUM_KV_HEADS * HEAD_DIM * element_bytes  # K + V
    total_bytes = kv_cache_mb * 1024 ** 2
    return max(1, int(total_bytes // bytes_per_block))


def build_context_tensors(seq_len: int, use_cpu: bool, block_size: int):
    """
    Construct all tensors that set_context() needs for a single decode step.

    The block table is assigned sequentially [0, 1, ..., num_used_blocks-1].
    The new token (token index seq_len) maps to the slot at:
        last_block_id * block_size + (seq_len % block_size)
    """
    total_tokens   = seq_len + 1                          # includes the new decode token
    num_used_blocks = math.ceil(total_tokens / block_size)
    block_table    = list(range(num_used_blocks))

    new_token_slot = block_table[-1] * block_size + (seq_len % block_size)

    if use_cpu:
        # New token is written to the CPU KV cache; GPU slot is -1 (no-op in Triton kernel)
        slot_mapping     = torch.tensor([-1],              dtype=torch.long,  device="cuda")
        slot_mapping_cpu = torch.tensor([new_token_slot],  dtype=torch.long,  device="cpu")
        # Block tables: GPU path gets all -1 (unused); CPU path gets the real IDs
        block_tables     = torch.full((1, num_used_blocks), -1, dtype=torch.int32, device="cuda")
        block_tables_cpu = torch.tensor([block_table],        dtype=torch.int32)
        # Context lengths: GPU sees 0 (no GPU sequence); CPU sees full length
        context_lens     = torch.tensor([0],            dtype=torch.int32, device="cuda")
        context_lens_cpu = torch.tensor([total_tokens], dtype=torch.int32)
        cache_locations  = [BlockLocation.CPU]
    else:
        # New token is written to the GPU KV cache; CPU slot is -1 (unused)
        slot_mapping     = torch.tensor([new_token_slot], dtype=torch.long,  device="cuda")
        slot_mapping_cpu = torch.tensor([-1],             dtype=torch.long,  device="cpu")
        block_tables     = torch.tensor([block_table],    dtype=torch.int32, device="cuda")
        block_tables_cpu = torch.full((1, num_used_blocks), -1, dtype=torch.int32)
        context_lens     = torch.tensor([total_tokens], dtype=torch.int32, device="cuda")
        context_lens_cpu = torch.tensor([0],            dtype=torch.int32)
        cache_locations  = [BlockLocation.GPU]

    return (
        slot_mapping, slot_mapping_cpu,
        block_tables, block_tables_cpu,
        context_lens, context_lens_cpu,
        cache_locations,
        total_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    block_size = CPU_BLOCK_SIZE if USE_CPU else GPU_BLOCK_SIZE
    num_blocks = compute_num_blocks(KV_CACHE_MB, block_size)
    max_tokens = num_blocks * block_size

    print("=" * 60)
    print("nano-vllm Attention decode profiler")
    print("=" * 60)
    print(f"  {'KV cache size':<28}: {KV_CACHE_MB} MB")
    print(f"  {'Sequence location':<28}: {'CPU' if USE_CPU else 'GPU'}")
    print(f"  {'Block size':<28}: {block_size} tokens")
    print(f"  {'Num KV cache blocks':<28}: {num_blocks}")
    print(f"  {'Max cache capacity':<28}: {max_tokens} tokens")
    print(f"  {'Sequence length (cached)':<28}: {SEQ_LEN} tokens")
    print(f"  {'Total tokens after decode':<28}: {SEQ_LEN + 1} tokens")
    print()

    if SEQ_LEN + 1 > max_tokens:
        raise ValueError(
            f"SEQ_LEN={SEQ_LEN} requires {SEQ_LEN + 1} tokens but the cache "
            f"only holds {max_tokens} tokens ({KV_CACHE_MB} MB, {block_size}-token blocks). "
            f"Increase KV_CACHE_MB or decrease SEQ_LEN."
        )

    # ── Build Attention module ─────────────────────────────────────────────────
    attn = Attention(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        scale=1.0 / math.sqrt(HEAD_DIM),
        num_kv_heads=NUM_KV_HEADS,
    ).cuda()

    # ── Allocate KV cache tensors and wire into the module ─────────────────────
    if USE_CPU:
        # CPU KV cache: the real store; pin_memory for fast DMA transfers
        attn.k_cache_cpu = torch.zeros(
            num_blocks, CPU_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
            dtype=DTYPE, pin_memory=True,
        )
        attn.v_cache_cpu = torch.zeros_like(attn.k_cache_cpu)

        # GPU KV cache: minimal placeholder so `if k_cache.numel()` is True,
        # allowing the Triton store kernel to run (it will see slot=-1 and skip).
        attn.k_cache = torch.zeros(
            1, GPU_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
            dtype=DTYPE, device="cuda",
        )
        attn.v_cache = torch.zeros_like(attn.k_cache)
    else:
        # GPU KV cache: the real store
        attn.k_cache = torch.zeros(
            num_blocks, GPU_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
            dtype=DTYPE, device="cuda",
        )
        attn.v_cache = torch.zeros_like(attn.k_cache)
        # CPU caches remain as empty tensors (set in Attention.__init__)

    _elem = 2  # bytes per float16 element
    gpu_cache_mb  = (attn.k_cache.numel() + attn.v_cache.numel()) * _elem / 1024**2
    cpu_cache_mb  = (attn.k_cache_cpu.numel() + attn.v_cache_cpu.numel()) * _elem / 1024**2
    total_cache_mb = gpu_cache_mb + cpu_cache_mb
    print(f"  {'Actual GPU KV cache':<28}: {gpu_cache_mb:.2f} MB  "
          f"(shape {list(attn.k_cache.shape)}, K+V)")
    print(f"  {'Actual CPU KV cache':<28}: {cpu_cache_mb:.2f} MB  "
          f"(shape {list(attn.k_cache_cpu.shape)}, K+V)")
    print(f"  {'Total KV cache allocated':<28}: {total_cache_mb:.2f} MB")
    filled_mb = 2 * SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * _elem / 1024**2
    print(f"  {'KV cache in use (SEQ_LEN)':<28}: {filled_mb:.4f} MB  "
          f"({SEQ_LEN} tokens × {NUM_KV_HEADS} heads × {HEAD_DIM} dim × K+V × fp16)")
    print()

    # ── Input tensors (single decode step: one new token) ─────────────────────
    # q: (1, num_heads, head_dim) — query for the single new token
    # k, v: (1, num_kv_heads, head_dim) — key/value projections of the new token
    q = torch.randn(1, NUM_HEADS,    HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

    # ── Build context tensors ─────────────────────────────────────────────────
    (
        slot_mapping, slot_mapping_cpu,
        block_tables, block_tables_cpu,
        context_lens, context_lens_cpu,
        cache_locations,
        total_tokens,
    ) = build_context_tensors(SEQ_LEN, USE_CPU, block_size)

    # Pre-populate the KV cache with random data so attention has something to attend to
    print("Pre-populating KV cache with random data...")
    num_used_blocks = math.ceil((SEQ_LEN + 1) / block_size)
    if USE_CPU:
        for blk_idx in tqdm(range(num_used_blocks - 1)):  # full blocks only; last block filled by store
            fill_len = min(CPU_BLOCK_SIZE, SEQ_LEN - blk_idx * CPU_BLOCK_SIZE)
            if fill_len > 0:
                attn.k_cache_cpu[blk_idx, :fill_len] = torch.randn(
                    fill_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
                attn.v_cache_cpu[blk_idx, :fill_len] = torch.randn(
                    fill_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    else:
        for blk_idx in range(num_used_blocks - 1):
            fill_len = min(GPU_BLOCK_SIZE, SEQ_LEN - blk_idx * GPU_BLOCK_SIZE)
            if fill_len > 0:
                attn.k_cache[blk_idx, :fill_len] = torch.randn(
                    fill_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
                attn.v_cache[blk_idx, :fill_len] = torch.randn(
                    fill_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    print("Done.\n")

    # ── Run function ───────────────────────────────────────────────────────────
    def run_once():
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping,
            slot_mapping_cpu=slot_mapping_cpu,
            context_lens=context_lens,
            context_lens_cpu=context_lens_cpu,
            block_tables=block_tables,
            block_tables_cpu=block_tables_cpu,
            cache_locations=cache_locations,
            cpu_kv_cache=USE_CPU,
        )
        with torch.no_grad():
            out = attn(q, k, v)
        reset_context()
        return out

    # ── Warmup ─────────────────────────────────────────────────────────────────
    print(f"Warming up ({WARMUP_ITERS} iterations)...")
    for _ in range(WARMUP_ITERS):
        run_once()
    torch.cuda.synchronize()
    print("Done.\n")

    # ── Torch profiler (first 10 iterations) ──────────────────────────────────
    if PROFILE:
        print(f"Running torch.profiler on first {PROFILE_EXPORT_ITERS} iterations...")
        profile_output = "profile_attention_cpu.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(PROFILE_EXPORT_ITERS):
                run_once()
            torch.cuda.synchronize()

        prof.export_chrome_trace(profile_output)
        print(f"Profiler trace exported → {profile_output}\n")

    # ── Timed profiling ────────────────────────────────────────────────────────
    print(f"Profiling ({PROFILE_ITERS} iterations)...")
    start = time.perf_counter()
    for _ in tqdm(range(PROFILE_ITERS)):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms    = elapsed / PROFILE_ITERS * 1000
    total_ms  = elapsed * 1000

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  {'Iterations':<28}: {PROFILE_ITERS}")
    print(f"  {'Total elapsed':<28}: {total_ms:.2f} ms")
    print(f"  {'Average latency':<28}: {avg_ms:.3f} ms")
    print(f"  {'Throughput':<28}: {1000 / avg_ms:.1f} decode steps/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
