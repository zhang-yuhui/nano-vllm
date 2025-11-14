import torch
from torch import nn
import triton
import triton.language as tl
from torch import Tensor
from time import perf_counter, sleep

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.engine.block_location import BlockLocation
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
    assert slot_mapping.numel() == N, f"slot_mapping.numel() != N, {slot_mapping.numel()} != {N}"  # One slot mapping per sequence position
    
    # Launch Triton kernel with N parallel threads (one per sequence position)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_cpu(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    CPU version of store_kvcache.
    
    Simple sequential implementation that stores key-value pairs into the KV cache
    based on slot mapping. Unlike the Triton version, this operates on CPU tensors
    and processes each position sequentially without parallelization.
    
    Args:
        key: Key tensor of shape (N, num_heads, head_dim)
        value: Value tensor of shape (N, num_heads, head_dim)
        k_cache: Key cache storage of shape (num_blocks, block_size, num_heads, head_dim)
        v_cache: Value cache storage of shape (num_blocks, block_size, num_heads, head_dim)
        slot_mapping: Mapping from sequence position to cache slot (absolute position), shape (N,)
    """
    N, num_heads, head_dim = key.shape
    block_size = k_cache.shape[1]  # Get block_size from cache shape
    
    for idx in range(N):
        slot = slot_mapping[idx].item()
        if slot == -1:
            continue  # Skip invalid slots (e.g., padding tokens)
        
        # Convert absolute slot to block index and position within block
        block_idx = slot // block_size
        pos_in_block = slot % block_size
        
        # Store key and value at the correct block and position
        k_cache[block_idx, pos_in_block] = key[idx]
        v_cache[block_idx, pos_in_block] = value[idx]

def load_kvcache_cpu(k_cache: torch.Tensor, v_cache: torch.Tensor, block_table: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU version of load_kvcache for a single sequence.
    
    Reconstructs the full key-value sequence from block-based storage.
    
    Args:
        k_cache: Key cache storage of shape (num_blocks, block_size, num_heads, head_dim)
        v_cache: Value cache storage of shape (num_blocks, block_size, num_heads, head_dim)
        block_table: Block indices for this sequence, shape (num_blocks_used,)
        seq_len: Total length of the sequence to reconstruct
        
    Returns:
        Tuple of (k, v) tensors with shape (seq_len, num_heads, head_dim)
    """
    block_size = k_cache.shape[1]  # Get block_size from cache shape
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    
    assert len(block_table) * block_size >= seq_len, f"Not enough blocks: {len(block_table)} * {block_size} < {seq_len}"
    
    # Allocate output tensors
    k = torch.empty(seq_len, num_kv_heads, head_dim, device='cpu', dtype=k_cache.dtype)
    v = torch.empty(seq_len, num_kv_heads, head_dim, device='cpu', dtype=v_cache.dtype)
    
    # Copy data from blocks to output
    output_pos = 0
    for i in range(len(block_table)):
        block_idx = block_table[i].item()
        
        # Determine how many tokens to copy from this block
        tokens_in_this_block = min(block_size, seq_len - output_pos)
        
        # Copy the tokens from cache to output
        k[output_pos:output_pos + tokens_in_this_block] = k_cache[block_idx, :tokens_in_this_block]
        v[output_pos:output_pos + tokens_in_this_block] = v_cache[block_idx, :tokens_in_this_block]
        
        output_pos += tokens_in_this_block
        
        if output_pos >= seq_len:
            break
    
    assert output_pos == seq_len, f"Loaded {output_pos} tokens but expected {seq_len}"
    return k, v

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
        self.k_cache_cpu = torch.tensor([])
        self.v_cache_cpu = torch.tensor([])
        
        # Create CUDA streams once for reuse (avoid overhead of creating on every forward pass)
        self.gpu_store_stream = torch.cuda.Stream()
        self.cpu_store_stream = torch.cuda.Stream()
        self.cpu_attn_stream = torch.cuda.Stream()
        self.gpu_attn_stream = torch.cuda.Stream()

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
        k_cache_cpu, v_cache_cpu = self.k_cache_cpu, self.v_cache_cpu
        
        # Store new key-value pairs in cache if cache is allocated
        if k_cache.numel() and v_cache.numel():
            # if this batch has cpu kv cache, store to cpu
            if context.cpu_kv_cache:
                # seperate cache for gpu and cpu
                # k_gpu, v_gpu = k[context.slot_mapping != -1], v[context.slot_mapping != -1]
                # k_cpu, v_cpu = k[context.slot_mapping_cpu != -1], v[context.slot_mapping_cpu != -1]

                # Reuse pre-created streams to avoid overhead
                # Launch GPU cache store in parallel stream
                with torch.cuda.stream(self.gpu_store_stream):
                    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
                
                # Launch CPU transfer and store in parallel stream
                with torch.cuda.stream(self.cpu_store_stream):
                    # Use non_blocking=True for async transfer
                    k_cpu = k.to("cpu", non_blocking=True)
                    v_cpu = v.to("cpu", non_blocking=True)
                    
                    # Note: We need to synchronize before CPU store since it's a CPU operation
                    # that depends on the async transfer completing
                    self.cpu_store_stream.synchronize()
                    store_kvcache_cpu(k_cpu, v_cpu, k_cache_cpu, v_cache_cpu, context.slot_mapping_cpu)
                
                # Both streams will synchronize implicitly at next CUDA operation
                # or explicitly synchronize here if needed for correctness
                self.gpu_store_stream.synchronize()
                self.cpu_store_stream.synchronize()
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            # // store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # PREFILL: Processing new input tokens (first forward pass)
            if context.block_tables is not None:    # prefix cache case
                # Reuse cached KV from previous requests (prefix caching optimization)
                #! assume that no prefix cache for now
                k, v = k_cache, v_cache
            
            # concat two block tables
            # block table is [batch, num_blocks]
            # Merge per-sequence rows from GPU/CPU tables into one CUDA tensor
            #! since no prefix cache for now, the block table should be None
            if not context.cpu_kv_cache and context.block_tables is not None and context.block_tables_cpu is not None:
                batch_size = len(context.cache_locations)
                max_blocks = context.block_tables.shape[1]
                block_tables_merged = torch.empty(batch_size, max_blocks, dtype=torch.int32, device='cuda')
                
                for i, loc in enumerate(context.cache_locations):
                    if loc == BlockLocation.GPU:
                        block_tables_merged[i] = context.block_tables[i]
                    else:
                        block_tables_merged[i] = context.block_tables_cpu[i].to('cuda', non_blocking=True)
            else:
                block_tables_merged = context.block_tables

            # Use variable-length FlashAttention for prefill
            # This handles sequences of different lengths efficiently
            # since this is prefill, we can use block table merged and no need to load kv cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=block_tables_merged)
        # DECODE: Generating next token
        else:    
            if not context.cpu_kv_cache:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
            # use gpu kv cache and cpu kv cache
            else:
                # seperate block table and context lens for gpu and cpu
                block_tables_gpu = context.block_tables
                block_tables_cpu = context.block_tables_cpu
                context_lens_gpu = context.context_lens
                context_lens_cpu = context.context_lens_cpu
                #print(context.block_tables, context.block_tables_cpu, context.context_lens, context.context_lens_cpu)
                # o shape: [seq_len, 1, num_q_heads, head_dim]
                #print(context.cache_locations)
                o = torch.empty(len(context.cache_locations), 1, q.shape[-2], k.shape[-1], device=q.device, dtype=q.dtype)

                # Reuse pre-created CUDA streams for parallel execution
                # Launch CPU attention in a separate stream
                with torch.cuda.stream(self.cpu_attn_stream):
                    # load kv cache and calculate attention for cpu sequences
                    for i, loc in enumerate(context.cache_locations):
                        if loc == BlockLocation.GPU:
                            continue
                        # load kv cache for cpu sequence
                        k_cpu, v_cpu = load_kvcache_cpu(k_cache_cpu, v_cache_cpu, block_tables_cpu[i], context_lens_cpu[i])
                        # calculate attention for cpu sequence
                        o_cpu = self.attention_cpu(q[i].to("cpu"), k_cpu, v_cpu, self.scale)
                        # Transfer result to GPU asynchronously
                        o[i] = o_cpu.to(o.device, non_blocking=True)

                # Launch GPU flash attention in parallel stream
                with torch.cuda.stream(self.gpu_attn_stream):
                    # flash attention for gpu sequences
                    o_gpu = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                                cache_seqlens=context_lens_gpu, block_table=block_tables_gpu, 
                                                softmax_scale=self.scale, causal=True)

                    # concat attention result for gpu sequences
                    for i, loc in enumerate(context.cache_locations):
                        if loc == BlockLocation.GPU:
                            o[i] = o_gpu[i]
                
                # Synchronize both streams before returning
                self.cpu_attn_stream.synchronize()
                self.gpu_attn_stream.synchronize()
            #print(o.dtype)

        return o

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
        q = q.view(num_kv_heads, num_groups, head_dim)
        
        # Reshape K, V
        # [total_len, 8, 128] -> [8, total_len, 128]
        k = k_cache.transpose(-3, -2)
        v = v_cache.transpose(-3, -2)
        
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
        output = output.reshape(1, num_q_heads, head_dim)
        return output