<!-- d6fb9bc9-5651-49b3-992f-b8f66b41c629 bea4d818-ec69-4d7b-b79f-2edd16c565b1 -->
# Implement CPU+GPU context mappings in ModelRunner

## Files to edit

- `nanovllm/engine/model_runner.py`

## Changes

### 1) Prepare separate block tables per device

- Update `prepare_block_tables(seqs)` to build and return both:
  - `block_tables_gpu`: padded table for sequences whose `seq.cache_location == BlockLocation.GPU` (rows for CPU seqs filled with -1)
  - `block_tables_cpu`: padded table for sequences whose `seq.cache_location == BlockLocation.CPU` (rows for GPU seqs filled with -1)
- Keep output shapes as `[batch, max_blocks]` each and dtype `int32`.
- Explicitly create GPU tables on CUDA; CPU tables on CPU.

### 2) Prefill: split slot mapping into GPU and CPU

- In `prepare_prefill`:
  - Build `slot_mapping_gpu` over all new tokens, mapping GPU sequences to GPU KV slots; fill `-1` for tokens belonging to CPU sequences.
  - Build `slot_mapping_cpu` over all new tokens, mapping CPU sequences to CPU KV slots; fill `-1` for tokens belonging to GPU sequences.
  - Compute `block_tables_gpu, block_tables_cpu = prepare_block_tables(seqs)` when prefix cache is present.
  - Compute `cache_infos = [seq.cache_info for seq in seqs]` and `cache_locations = [seq.cache_location for seq in seqs]`.
  - Use named arguments in `set_context` to avoid ordering errors, passing both GPU and CPU fields:
    - `is_prefill=True`
    - `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`
    - `slot_mapping=slot_mapping_gpu` (CUDA), `slot_mapping_cpu=slot_mapping_cpu` (CPU device)
    - `block_tables=block_tables_gpu` (CUDA), `block_tables_cpu=block_tables_cpu` (CPU device)
    - `cache_infos=cache_infos`, `cache_locations=cache_locations`

### 3) Decode: provide lens/slot/tables for both devices

- In `prepare_decode`:
  - Build per-sequence `slot_mapping_gpu` and `slot_mapping_cpu` (length = batch size):
    - GPU seq: set GPU slot id; set CPU `-1`.
    - CPU seq: set CPU slot id using `self.cpu_block_size`; set GPU `-1`.
  - Build `context_lens_gpu` and `context_lens_cpu` (length = batch size):
    - GPU seq: `context_lens_gpu = len(seq)`, `context_lens_cpu = 0`.
    - CPU seq: `context_lens_cpu = len(seq)`, `context_lens_gpu = 0`.
  - Get `block_tables_gpu, block_tables_cpu = prepare_block_tables(seqs)`.
  - Pass all via named args in `set_context`:
    - `is_prefill=False`
    - `slot_mapping=slot_mapping_gpu` (CUDA), `slot_mapping_cpu=slot_mapping_cpu` (CPU)
    - `context_lens=context_lens_gpu` (CUDA), `context_lens_cpu=context_lens_cpu` (CPU)
    - `block_tables=block_tables_gpu` (CUDA), `block_tables_cpu=block_tables_cpu` (CPU)
    - `cache_infos` and `cache_locations` lists as above.

### 4) Safety and correctness

- Explicitly place CPU tensors on `device="cpu"` (default device is set to CUDA in `ModelRunner.__init__`). Use `pin_memory=True` where helpful; do NOT call `.cuda()` on CPU variants.
- Always create GPU tensors with `.cuda(non_blocking=True)`.
- Replace current positional `set_context` usage with named args in both prefill and decode (fixes existing mismatch bug).
- Import and use `BlockLocation` in `model_runner.py` where needed.
- Do not alter CUDA graph capture paths.

## Example snippets (concise)

- Decode slot mapping (per-seq):
```python
if seq.cache_location == BlockLocation.CPU:
    slot_mapping_gpu.append(-1)
    slot_mapping_cpu.append(seq.block_table[-1] * self.cpu_block_size + seq.last_block_num_tokens - 1)
    context_lens_gpu.append(0)
    context_lens_cpu.append(len(seq))
else:
    slot_mapping_gpu.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
    slot_mapping_cpu.append(-1)
    context_lens_gpu.append(len(seq))
    context_lens_cpu.append(0)
```

- Prefill per-token mapping (new tokens only):
```python
if seq.cache_location == BlockLocation.CPU:
    # fill GPU mapping with -1s, CPU with proper indices
    slot_mapping_gpu.extend([-1] * num_new)
    slot_mapping_cpu.extend(range(start_cpu, end_cpu))
else:
    slot_mapping_gpu.extend(range(start_gpu, end_gpu))
    slot_mapping_cpu.extend([-1] * num_new)
```


## Todos

- prepare-block-tables: Return per-device tensors
- prefill-context: Add cpu/gpu slot maps and block tables; pass named args
- decode-context: Add cpu/gpu slot maps, lens, and block tables; pass named args

### To-dos

- [ ] Return GPU and CPU block tables with per-row masking by device
- [ ] Build slot_mapping_{gpu,cpu}, set block_tables, pass named set_context
- [ ] Build slot_mapping/context_lens for both devices; pass named set_context