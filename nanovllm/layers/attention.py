import torch
from torch import nn
import triton
import triton.language as tl
from itertools import count
from torch import Tensor
from time import sleep

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton kernel for efficiently storing key-value pairs into the KV cache.
    
    This kernel implements the core memory management for attention in vLLM:
    - Each sequence position needs to store its key and value vectors
    - The KV cache is a pre-allocated memory pool for all sequences
    - slot_mapping determines where each position's KV should be stored
    
    Args:
        key_ptr: Pointer to the key tensor data
        value_ptr: Pointer to the value tensor data  
        k_cache_ptr: Pointer to the key cache storage
        v_cache_ptr: Pointer to the value cache storage
        slot_mapping_ptr: Mapping from sequence position to cache slot
        D: Total dimension size (num_heads * head_dim)
    """
    idx = tl.program_id(0)  # Current sequence position being processed
    slot = tl.load(slot_mapping_ptr + idx)  # Get cache slot for this position
    if slot == -1: return  # Skip if no valid slot (e.g., padding token)
    
    # Calculate memory offsets for this position's key and value
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # Load the key and value vectors for this position
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # Calculate cache storage offsets and store the vectors
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    Python wrapper function that calls the Triton kernel to store KV cache.
    
    This function handles the interface between PyTorch tensors and the Triton kernel:
    - Validates tensor memory layout (contiguous in last dimension)
    - Extracts tensor metadata (shapes, strides)
    - Launches the Triton kernel with proper grid configuration
    
    The slot_mapping is crucial - it tells us where in the cache each sequence position
    should store its key and value vectors. This enables efficient memory management
    across multiple sequences of different lengths.
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    # Validate memory layout assumptions for efficient access
    assert key.stride(-1) == 1 and value.stride(-1) == 1  # Contiguous in last dim
    assert key.stride(1) == head_dim and value.stride(1) == head_dim  # Proper head stride
    assert k_cache.stride(1) == D and v_cache.stride(1) == D  # Cache layout
    assert slot_mapping.numel() == N  # One slot mapping per sequence position
    
    # Launch Triton kernel with N parallel threads (one per sequence position)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    Multi-head attention implementation for vLLM with KV caching.
    
    This attention layer implements two key optimizations from vLLM:
    1. KV Caching: Store previously computed key-value pairs to avoid recomputation
    2. PagedAttention: Use a block-based memory management system for efficient batching
    
    The attention mechanism supports both prefill (processing new tokens) and decode 
    (generating next tokens) phases with different FlashAttention kernels.
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads      # Number of attention heads
        self.head_dim = head_dim        # Dimension of each attention head
        self.scale = scale             # Scaling factor (usually 1/sqrt(head_dim))
        self.num_kv_heads = num_kv_heads # Number of KV heads (for GQA/MQA)
        
        # Initialize empty KV caches - will be allocated by the engine
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_cache_cpu = {}
        self.v_cache_cpu = {}
        self.counter = count()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Forward pass of the attention layer.
        
        This implements the core attention computation with two distinct phases:
        
        1. PREFILL PHASE: Process new input tokens
           - Uses flash_attn_varlen_func for variable-length sequences
           - Supports prefix caching (reusing cached KV from previous requests)
           - Computes attention for all new tokens at once
        
        2. DECODE PHASE: Generate next token
           - Uses flash_attn_with_kvcache for incremental generation
           - Only processes the new query token
           - Reuses all previously cached key-value pairs
        
        Args:
            q: Query tensor [batch_size, num_heads, head_dim] (decode) or [total_tokens, num_heads, head_dim] (prefill)
            k: Key tensor [batch_size, num_heads, head_dim] (decode) or [total_tokens, num_heads, head_dim] (prefill)  
            v: Value tensor [batch_size, num_heads, head_dim] (decode) or [total_tokens, num_heads, head_dim] (prefill)
        
        Returns:
            o: Output tensor with attention results
        """
        # Get execution context containing batch metadata and memory mappings
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        cache_infos = context.cache_infos
        
        # Store new key-value pairs in cache if cache is allocated
        if k_cache.numel() and v_cache.numel():
            cpu_idx = [i for (i, cache_location) in enumerate(cache_infos) if cache_location != -1]
            if len(cpu_idx) == 0:
                pass
            elif context.is_prefill:
                # store cache also in cpu
                for idx in cpu_idx:
                    start = context.cu_seqlens_k[idx]
                    end = context.cu_seqlens_k[idx + 1]
                    k_cache_cpu = k[start: end].to("cpu")
                    v_cache_cpu = v[start: end].to("cpu")
                    self.store_kv_cache_cpu(k_cache_cpu, v_cache_cpu, [cache_infos[idx]], prefill=True)
                k_cache_cpu = k[cpu_idx]
            
            # decoding
            else:
                k_cpu = k[cpu_idx].to("cpu")
                v_cpu = v[cpu_idx].to("cpu")
                self.store_kv_cache_cpu(k_cpu, v_cpu, [i for i in cache_infos if i != -1], prefill=False)

            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            # if context.is_prefill:
            #     print("prefill--------")
        
        if context.is_prefill:
            # PREFILL: Processing new input tokens (first forward pass)
            if context.block_tables is not None:    # prefix cache case
                # Reuse cached KV from previous requests (prefix caching optimization)
                k, v = k_cache, v_cache
            
            # Use variable-length FlashAttention for prefill
            # This handles sequences of different lengths efficiently
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            #print(f"cu seq len: {context.cu_seqlens_k.to('cpu')}")
        else:    # DECODE: Generating next token
            # Use KV-cache optimized FlashAttention for decode
            # Only processes the new query token, reuses all cached KV

            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
            #print(f"k_cache shape decoding: {k_cache.shape}, q shape: {q.shape}, o shape: {o.shape}")

            # load cpu cache and calculate attention
            o_cpu = torch.empty(0, 1, q.shape[-2], k.shape[-1]).to(o.device)
            for i, cache_info in enumerate(cache_infos):
                if cache_info == -1:
                    continue
                k_cpu, v_cpu = self.load_kv_cache_cpu(cache_info)
                assert torch.equal(k_cpu[-1], k[i].to("cpu")), k_cpu.shape[0]
                o_one = self.attention_cpu(q[i].to("cpu"), k_cpu, v_cpu, self.scale).to(o.device)
                #print(f"virefication of attention for seq {i}: {torch.allclose(o[i], o_one, rtol=1e-2, atol=1e-2)}")
                o_cpu = torch.cat([o_cpu, o_one.unsqueeze(0)], dim=0)
                o[i] = o_one
                

            #print(f"cpu attention shape: {o_cpu.shape}")

            # o shape: [seq_len, 1, num_q_heads, head_dim]
        return o
    
    def store_kv_cache_cpu(self, k: Tensor, v: Tensor, slot_mapping: list[int], prefill: bool):
        """
        k_cache: [seq_len, num_heads, head_dim]
        k: same
        """
        assert len(slot_mapping) > 0
        for i, id in enumerate(slot_mapping):
            if id not in self.k_cache_cpu:
                assert prefill, id
                self.k_cache_cpu[id] = k
                self.v_cache_cpu[id] = v
            else:
                assert not prefill
                self.k_cache_cpu[id] = torch.cat([self.k_cache_cpu[id], k[i].unsqueeze(0)], dim=0)
                self.v_cache_cpu[id] = torch.cat([self.v_cache_cpu[id], v[i].unsqueeze(0)], dim=0)

    def load_kv_cache_cpu(self, id):
        if id not in self.k_cache_cpu:
            return None
        return self.k_cache_cpu[id], self.v_cache_cpu[id]

    def attention_cpu(self, q: torch.Tensor, k_cache, v_cache, softmax_scale) -> torch.Tensor:
        """
        do attention for only 1 seq
        q: [1, num_heads (16), head_dim (128)]
        k_cache_cpu: [seq_len, num_heads (8), head_dim(128)]
        output: [hum_heads, head_dim], no flatten now
        """
        # extract k and v from cache

                # Verify shapes match expectations
        assert q.shape == (16, 128)
        assert k_cache.shape[1] == 8  # Or is it [1, seq_len, 8, 128]?
        num_q_heads = q.shape[-2]
        head_dim = q.shape[-1]
        num_kv_heads = k_cache.shape[-2]

        num_groups = num_q_heads // num_kv_heads # should be 2

        assert num_groups == 2
        assert k_cache.shape[-1] == v_cache.shape[-1] == head_dim

        # Reshape Q for grouped attention
        # [1, 16, 128] -> [8, 2, 128]
        q = q.view(num_kv_heads, num_groups, head_dim).float()
        
        # Reshape K, V
        # [total_len, 8, 128] -> [8, total_len, 128]
        k = k_cache.transpose(-3, -2).float()
        v = v_cache.transpose(-3, -2).float()
        
        # Compute attention scores
        # [8, 2, 128] @ [8, 128, total_len] -> [8, 2, total_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        
        # Softmax
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # [8, 2, total_len]
        
        # Weighted sum
        # [8, 2, total_len] @ [8, total_len, 128] -> [8, 2, 128]
        output = torch.matmul(attn_weights, v)
        
        # Reshape output
        # [8, 2, 128] -> [16, 128]
        output = output.reshape(1, num_q_heads, head_dim).bfloat16()
        return output