# KV Cache Memory Allocation & Access Pattern

## 1. WHERE KV CACHE IS DEFINED

### Step 1: Initial Definition in Attention Module (`nanovllm/layers/attention.py`)
```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        # Initialize empty KV caches - will be allocated by the engine
        self.k_cache = self.v_cache = torch.tensor([])  # <-- DEFINED HERE (empty initially)
        self.k_cache_cpu = {}
        self.v_cache_cpu = {}
        self.counter = count()
```

**Location**: `nanovllm/layers/attention.py`, line ~111
- `self.k_cache` and `self.v_cache` are **placeholder attributes** initialized as empty tensors
- These are instance attributes that will be replaced later

### Step 2: Memory Allocation in ModelRunner (`nanovllm/engine/model_runner.py`)
```python
def allocate_kv_cache(self):
    """Allocate KV cache memory for efficient attention computation."""
    config = self.config
    hf_config = config.hf_config
    
    # Calculate cache size based on available GPU memory
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    
    # ALLOCATE the massive GPU memory buffer
    self.kv_cache = torch.empty(
        2,                              # [K_cache, V_cache]
        hf_config.num_hidden_layers,    # one per layer
        config.num_kvcache_blocks,      # number of blocks
        self.block_size,                # tokens per block
        num_kv_heads,                   # number of KV heads
        head_dim                        # dimension per head
    )  # <-- MAIN ALLOCATION (GPU memory)
```

**Location**: `nanovllm/engine/model_runner.py`, line ~155

**Memory Layout**:
```
self.kv_cache shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]

Example with real numbers:
- 2 (K and V)
- 28 (layers in Qwen-3)
- 256 (number of cache blocks, depends on GPU memory)
- 16 (block_size - tokens per block)
- 8 (num_kv_heads after tensor parallelism)
- 128 (head_dim)

Total memory = 2 * 28 * 256 * 16 * 8 * 128 * 2 bytes (bfloat16)
             ≈ 37 GB for a single GPU
```

## 2. HOW KV CACHE ACCESSES ALLOCATED MEMORY

### Step 3: Assign Cache References to Attention Modules

```python
def allocate_kv_cache(self):
    # ... allocation code above ...
    
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            # ASSIGN references to specific layers
            module.k_cache = self.kv_cache[0, layer_id]  # K cache for this layer
            module.v_cache = self.kv_cache[1, layer_id]  # V cache for this layer
            layer_id += 1
```

**Location**: `nanovllm/engine/model_runner.py`, line ~162-167

**What's happening**:
- Iterates through all modules in the model (Qwen3ForCausalLM)
- Finds all `Attention` modules (which have `k_cache` and `v_cache` attributes)
- **Replaces the empty placeholder tensors** with slices of the main `self.kv_cache` buffer

**Result after assignment**:
```
attention_module_layer_0.k_cache --> self.kv_cache[0, 0, :, :, :, :]
attention_module_layer_0.v_cache --> self.kv_cache[1, 0, :, :, :, :]

attention_module_layer_1.k_cache --> self.kv_cache[0, 1, :, :, :, :]
attention_module_layer_1.v_cache --> self.kv_cache[1, 1, :, :, :, :]

... (repeat for all layers)

attention_module_layer_27.k_cache --> self.kv_cache[0, 27, :, :, :, :]
attention_module_layer_27.v_cache --> self.kv_cache[1, 27, :, :, :, :]
```

All these slices **point to the same underlying GPU memory buffer**.

### Step 4: During Forward Pass - Store KV in Cache

```python
# In Attention.forward() at nanovllm/layers/attention.py, line ~151
def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache  # Access the assigned cache
    
    if k_cache.numel() and v_cache.numel():  # Check if cache was allocated
        # Store new K,V pairs into cache
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

**The `store_kvcache()` function** (Triton kernel):
```python
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride, 
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)  # Current sequence position
    slot = tl.load(slot_mapping_ptr + idx)  # Which cache slot to use
    
    # Load K,V for this position
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # Store into the allocated cache at the slot location
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)  # <-- Writes to self.kv_cache[0, layer_id]
    tl.store(v_cache_ptr + cache_offsets, value)  # <-- Writes to self.kv_cache[1, layer_id]
```

### Step 5: During Decode - Read from Cache

```python
# In Attention.forward() - decode phase at nanovllm/layers/attention.py, line ~176
else:  # DECODE phase
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,  # <-- Reads from allocated cache
        v_cache,  # <-- Reads from allocated cache
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True
    )
```

The FlashAttention kernel directly accesses the GPU memory stored in `k_cache` and `v_cache`.

## 3. MEMORY ACCESS FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                     ModelRunner.__init__()                      │
│                                                                 │
│  1. Create model: self.model = Qwen3ForCausalLM(hf_config)     │
│     └─ Creates 28 layers, each with Attention module            │
│        └─ Each Attention has: k_cache = torch.tensor([])        │
│           (empty placeholder)                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    allocate_kv_cache()                          │
│                                                                 │
│  2. Allocate GPU memory:                                        │
│     self.kv_cache = torch.empty(2, 28, 256, 16, 8, 128)        │
│     (GPU memory allocated)                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Assign cache slices to modules                     │
│                                                                 │
│  3. For each layer:                                            │
│     layer.attention.k_cache = self.kv_cache[0, layer_id]       │
│     layer.attention.v_cache = self.kv_cache[1, layer_id]       │
│                                                                 │
│     These are VIEWS (references) to the same GPU memory!       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Attention.forward()                          │
│                                                                 │
│  4. Use cache:                                                  │
│     - PREFILL: store_kvcache(k, v, self.k_cache, self.v_cache) │
│     - DECODE:  flash_attn_with_kvcache(..., self.k_cache, ...) │
│                                                                 │
│     All operations use the views pointing to the GPU buffer     │
└─────────────────────────────────────────────────────────────────┘
```

## 4. KEY INSIGHTS

### Why This Design?

1. **Separation of Concerns**:
   - Model layers don't know about memory management
   - ModelRunner (engine) handles allocation and lifecycle

2. **Unified Memory Buffer**:
   - All layers share one large buffer instead of individual allocations
   - More efficient GPU memory usage
   - Better cache coherency

3. **Dynamic Allocation**:
   - Cache size calculated based on available GPU memory
   - Can adapt to different GPU sizes

4. **Tensor Views (Not Copies)**:
   - `module.k_cache = self.kv_cache[0, layer_id]` creates a **view**, not a copy
   - Both point to same underlying GPU memory
   - Changes to one reflect in the other
   - Zero-copy overhead

### Memory Access Path

```
Attention.forward()
  │
  ├─ self.k_cache (a view)
  │   └─ points to self.kv_cache[0, layer_id]
  │       └─ points to GPU memory buffer
  │
  └─ store_kvcache() or flash_attn_with_kvcache()
      └─ uses the memory location
```

All operations happen on the **same GPU memory**, just accessed through different tensor views!
