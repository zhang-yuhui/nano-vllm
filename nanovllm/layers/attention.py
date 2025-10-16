import torch
from torch import nn
import triton
import triton.language as tl

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
        
        # Store new key-value pairs in cache if cache is allocated
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
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
        else:    # DECODE: Generating next token
            # Use KV-cache optimized FlashAttention for decode
            # Only processes the new query token, reuses all cached KV
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o