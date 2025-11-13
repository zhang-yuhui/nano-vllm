# KV Cache Allocation - Detailed Variable Breakdown

## Code Section Being Analyzed
```python
def allocate_kv_cache(self):
    """Allocate KV cache memory for efficient attention computation."""
    config = self.config
    hf_config = config.hf_config

    # Get current GPU memory status
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
    config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    assert config.num_kvcache_blocks > 0
    if config.cpu_kv_cache:
        assert config.cpu_block_size > 0 and config.num_cpu_blocks > 0
    self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

---

## Variable Definitions with Real Examples

### 1. Memory Status Variables

#### `free`, `total`
```python
free, total = torch.cuda.mem_get_info()
```
**What they are:**
- `total`: Total GPU VRAM available on the device
- `free`: Currently available (unused) GPU VRAM

**Example:**
```
For an NVIDIA A100 GPU (80GB):
total = 80 * 1024 * 1024 * 1024 bytes = 85,899,345,920 bytes
free = 75 * 1024 * 1024 * 1024 bytes = 80,530,636,800 bytes (if 5GB is in use)
```

---

#### `used`
```python
used = total - free
```
**What it is:**
- Currently allocated/used GPU memory

**Example:**
```
used = 85,899,345,920 - 80,530,636,800 = 5,368,709,120 bytes (5 GB in use)
```

---

#### `peak`
```python
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
```
**What it is:**
- The **maximum amount of memory ever allocated** during the current session
- Not the current allocation, but the highest watermark

**Why it matters:**
- GPU memory fragmentation: peak memory might be higher than current usage due to freed memory leaving gaps
- We reserve space for potential peak re-allocation

**Example:**
```
During model loading and initialization:
peak = 40 * 1024 * 1024 * 1024 bytes = 42,949,672,960 bytes (40 GB was allocated at peak)

But currently:
current = 5,368,709,120 bytes (only 5 GB now, rest was freed)

Without accounting for peak, we'd run out of memory when re-allocating to peak levels
```

---

#### `current`
```python
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
```
**What it is:**
- The **currently allocated** GPU memory right now

**Example:**
```
current = 5,368,709,120 bytes (5 GB currently in use by model weights, etc.)
```

---

### 2. Model Architecture Variables

#### `num_key_value_heads` (from config)
```python
hf_config.num_key_value_heads  # From HuggingFace config
```
**What it is:**
- Total number of key-value attention heads across **all GPUs** in the cluster
- For models with Grouped Query Attention (GQA) or Multi-Query Attention (MQA)

**Example for Qwen-3 with 8 GPUs (tensor parallelism):**
```
hf_config.num_key_value_heads = 8  # Qwen-3 config value (GQA: fewer KV heads than Q heads)
```

---

#### `num_kv_heads`
```python
num_kv_heads = hf_config.num_key_value_heads // self.world_size
```
**What it is:**
- Number of KV heads **on this specific GPU** after tensor parallelism sharding
- Divides the KV heads among all GPUs

**Example:**
```
If running on GPU rank 0 out of 8 GPUs:
self.world_size = 8
hf_config.num_key_value_heads = 8

num_kv_heads = 8 // 8 = 1  # This GPU handles 1 KV head
```

**Why it matters:**
- In tensor parallelism, each GPU only stores a portion of the KV cache
- This reduces per-GPU memory requirements

---

#### `head_dim`
```python
head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
```
**What it is:**
- Dimension of each attention head
- Either explicitly set in config, or calculated from hidden_size and total num_heads

**Example for Qwen-3:**
```
If not explicitly set in config:
hf_config.hidden_size = 4096
hf_config.num_attention_heads = 32

head_dim = 4096 // 32 = 128 dimensions per head
```

**Breakdown of 128-dim head:**
```
[head_dim = 128 scalar values]
e.g., Q vector for one head: [0.15, 0.42, -0.8, ..., 0.21]  (128 values)
```

---

#### `hf_config.num_hidden_layers`
```python
hf_config.num_hidden_layers
```
**What it is:**
- Number of transformer layers in the model

**Example for Qwen-3:**
```
hf_config.num_hidden_layers = 28  # Qwen-3 has 28 transformer layers
```

---

#### `self.block_size`
```python
self.block_size = config.kvcache_block_size
```
**What it is:**
- Number of tokens stored per KV cache block
- Part of "paged attention" - memory is organized into fixed-size blocks

**Example:**
```
self.block_size = 16  # Each block holds 16 tokens' KV pairs

This enables:
- Efficient memory management (like virtual memory/paging in OS)
- Flexible sequence allocation without contiguous memory
```

---

#### `hf_config.torch_dtype`
```python
hf_config.torch_dtype.itemsize
```
**What it is:**
- Size in bytes of a single tensor element
- Depends on the precision/dtype of the model

**Example:**
```
For different precisions:
torch.float32.itemsize = 4 bytes (32 bits)
torch.float16.itemsize = 2 bytes (16 bits)
torch.bfloat16.itemsize = 2 bytes (16 bits)

Example: Qwen-3 uses bfloat16
hf_config.torch_dtype = torch.bfloat16
hf_config.torch_dtype.itemsize = 2 bytes per element
```

---

### 3. Memory Calculation Variables

#### `block_bytes`
```python
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
```
**What it is:**
- Memory required (in bytes) to store **one block of K,V cache** across **all layers**

**Breakdown:**
- `2`: K cache and V cache (2 separate buffers)
- `hf_config.num_hidden_layers`: Memory for one block in each transformer layer
- `self.block_size`: Tokens per block
- `num_kv_heads`: Number of KV heads on this GPU
- `head_dim`: Dimension per head
- `torch_dtype.itemsize`: Bytes per element

**Example Calculation (Qwen-3, 1 GPU, bfloat16):**
```
block_bytes = 2 
            * 28 (layers)
            * 16 (tokens per block)
            * 1 (KV heads on this GPU)
            * 128 (dimensions per head)
            * 2 (bytes for bfloat16)

block_bytes = 2 * 28 * 16 * 1 * 128 * 2
            = 114,688 bytes
            ≈ 112 KB per block

This means: storing one 16-token block for all 28 layers requires 112 KB
```

---

#### `config.gpu_memory_utilization`
```python
config.gpu_memory_utilization
```
**What it is:**
- Target fraction of GPU memory to use for KV cache (e.g., 0.9 = 90%)
- Must leave headroom for model weights, activations, etc.

**Example:**
```
config.gpu_memory_utilization = 0.9  # Use up to 90% of GPU VRAM

For 80GB A100:
target_memory = 80 * 0.9 = 72 GB reserved for KV cache and other allocations
```

---

#### `config.num_kvcache_blocks`
```python
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```
**What it is:**
- **Number of cache blocks** to allocate based on available GPU memory

**Breakdown of the formula:**
```
Available memory for cache = (total * gpu_memory_utilization) - used - peak + current
                           
- total * gpu_memory_utilization: Target memory budget (e.g., 90% of 80GB = 72GB)
- used: Currently used memory (need to subtract to avoid conflicts)
- peak: Subtract peak because of fragmentation risk
- + current: Add back current (it's part of "used")

Number of blocks = Available memory // block_bytes
```

**Example Calculation (A100 80GB, Qwen-3):**
```
total = 85,899,345,920 bytes (80 GB)
config.gpu_memory_utilization = 0.9
used = 5,368,709,120 bytes (5 GB)
peak = 42,949,672,960 bytes (40 GB peak during loading)
current = 5,368,709,120 bytes (5 GB current)
block_bytes = 114,688 bytes (from previous calculation)

Available = (85,899,345,920 * 0.9) - 5,368,709,120 - 42,949,672,960 + 5,368,709,120
         = 77,309,411,328 - 5,368,709,120 - 42,949,672,960 + 5,368,709,120
         = 34,359,738,368 bytes (32 GB)

config.num_kvcache_blocks = 34,359,738,368 // 114,688
                          = 299,680 blocks
```

**What this means:**
```
With 299,680 blocks:
- Each block holds 16 tokens
- Total tokens cacheable = 299,680 * 16 = 4,794,880 tokens
- At bfloat16, this requires ~32 GB per GPU
```

---

### 4. The Main KV Cache Tensor

#### `self.kv_cache`
```python
self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                            self.block_size, num_kv_heads, head_dim)
```
**What it is:**
- The main GPU memory buffer storing all KV pairs for all layers

**Shape dimensions:**
```
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]

Position 0: 2 (Index 0 = K cache, Index 1 = V cache)
Position 1: num_hidden_layers (28 for Qwen-3)
Position 2: num_kvcache_blocks (299,680 blocks in our example)
Position 3: block_size (16 tokens per block)
Position 4: num_kv_heads (1 for single-GPU, 8 for 8-GPU)
Position 5: head_dim (128)
```

**Example shape (Qwen-3, single GPU, bfloat16):**
```
self.kv_cache shape: [2, 28, 299680, 16, 1, 128]

Memory size = 2 * 28 * 299680 * 16 * 1 * 128 * 2 bytes
            = 34,359,738,368 bytes
            ≈ 32 GB
```

**How it's used internally:**
```
self.kv_cache[0, :, :, :, :, :] → K cache for all blocks, all layers
self.kv_cache[1, :, :, :, :, :] → V cache for all blocks, all layers

self.kv_cache[0, 5, :, :, :, :] → K cache for layer 5, all blocks
self.kv_cache[0, 5, 100, :, :, :] → K cache for layer 5, block 100 (16 tokens)
```

---

### 5. Layer Assignment

#### `layer_id`
```python
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```
**What it is:**
- Counter tracking which layer we're currently assigning cache to

**Example execution:**
```
Iteration 1:
  layer_id = 0
  module = Attention layer 0
  module.k_cache = self.kv_cache[0, 0]  (K cache for layer 0)
  module.v_cache = self.kv_cache[1, 0]  (V cache for layer 0)
  layer_id = 1

Iteration 2:
  layer_id = 1
  module = Attention layer 1
  module.k_cache = self.kv_cache[0, 1]  (K cache for layer 1)
  module.v_cache = self.kv_cache[1, 1]  (V cache for layer 1)
  layer_id = 2

... (repeat 28 times for Qwen-3)

Final state:
  layer_id = 28 (all 28 layers assigned)
```

---

## Complete Example: Full Allocation Scenario

### Model Configuration
```
Model: Qwen-3
GPU: Single NVIDIA A100 (80GB)
Tensor Parallelism: 1 GPU
Precision: bfloat16
```

### Calculated Values
```
1. Memory Stats:
   total = 85,899,345,920 bytes (80 GB)
   free = 75,530,636,800 bytes (70.4 GB available)
   used = 5,368,709,120 bytes (5 GB in use)
   peak = 42,949,672,960 bytes (40 GB peak)
   current = 5,368,709,120 bytes (5 GB current)

2. Model Architecture:
   hf_config.num_hidden_layers = 28
   hf_config.num_key_value_heads = 8
   hf_config.num_attention_heads = 32
   hf_config.hidden_size = 4096
   
3. GPU-Specific Values:
   self.world_size = 1 (single GPU)
   num_kv_heads = 8 // 1 = 8
   head_dim = 4096 // 32 = 128
   self.block_size = 16
   hf_config.torch_dtype = torch.bfloat16
   hf_config.torch_dtype.itemsize = 2

4. Cache Calculation:
   config.gpu_memory_utilization = 0.9
   block_bytes = 2 * 28 * 16 * 8 * 128 * 2 = 917,504 bytes (~897 KB)
   
   Available = (85,899,345,920 * 0.9) - 5,368,709,120 - 42,949,672,960 + 5,368,709,120
             = 77,309,411,328 - 42,949,672,960
             = 34,359,738,368 bytes (32 GB)
   
   config.num_kvcache_blocks = 34,359,738,368 // 917,504 = 37,449 blocks

5. KV Cache Allocation:
   self.kv_cache = torch.empty(2, 28, 37449, 16, 8, 128)
   
   Total memory = 2 * 28 * 37449 * 16 * 8 * 128 * 2 bytes
                = 34,359,738,368 bytes ≈ 32 GB

6. Layer Assignment:
   For each of 28 layers:
     layer.attention.k_cache → self.kv_cache[0, layer_idx]
     layer.attention.v_cache → self.kv_cache[1, layer_idx]
```

### What Can Be Cached
```
Total tokens = 37,449 blocks * 16 tokens/block = 599,184 tokens per GPU

Practical scenarios:
- Single long sequence: Can cache up to 599K tokens
- Batch of 8 sequences: Can cache ~75K tokens per sequence
- Batch of 32 sequences: Can cache ~19K tokens per sequence
```

---

## Key Insights

### Why Each Variable Matters

| Variable | Purpose | Example |
|----------|---------|---------|
| `total` | GPU capacity limit | 80 GB |
| `used` | Memory already allocated | 5 GB (model weights) |
| `peak` | Fragmentation safety margin | 40 GB (reserved) |
| `current` | Account for freed memory | 5 GB |
| `num_kv_heads` | Per-GPU KV head count | 8 (single GPU) / 1 (8-GPU TP) |
| `head_dim` | Head vector size | 128 dimensions |
| `block_size` | Paged attention granularity | 16 tokens per block |
| `block_bytes` | Memory per block per layer | 897 KB |
| `gpu_memory_utilization` | Budget percentage | 90% of total |
| `num_kvcache_blocks` | Total allocatable blocks | 37,449 blocks |

### Memory Hierarchy
```
GPU VRAM (80 GB)
├── Model Weights: ~34 GB (Qwen-3 weights)
├── KV Cache: ~32 GB (allocated by allocate_kv_cache)
├── Activations: ~10 GB (during forward pass)
└── Buffer: ~4 GB (safety margin)
```

This allocation strategy maximizes KV cache while preventing OOM errors!
