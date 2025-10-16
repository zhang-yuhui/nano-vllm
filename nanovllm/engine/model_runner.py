# ModelRunner: A distributed model inference runner for vLLM with tensor parallelism
# This class manages model execution across multiple GPUs, KV cache allocation,
# and CUDA graph optimization for efficient large language model inference

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """Initialize the ModelRunner with distributed setup.

        Args:
            config: Configuration object containing model and runtime parameters
            rank: GPU rank in the distributed setup (0 for main process)
            event: Synchronization event(s) for inter-process communication
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # Size of each KV cache block
        self.enforce_eager = config.enforce_eager    # Whether to force eager execution (no CUDA graphs)
        self.world_size = config.tensor_parallel_size  # Number of GPUs for tensor parallelism
        self.rank = rank  # Current GPU rank
        self.event = event  # Synchronization event for shared memory communication

        # Initialize distributed process group for multi-GPU communication
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # Set current GPU device

        # Save and set default data types for consistent tensor operations
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)  # Use model's precision (e.g., float16)
        torch.set_default_device("cuda")  # Set default device to GPU

        # Load model and prepare for inference
        self.model = Qwen3ForCausalLM(hf_config)  # Initialize the transformer model
        load_model(self.model, config.model)      # Load model weights from disk
        self.sampler = Sampler()  # Initialize token sampler for generation

        # Prepare model and memory for inference
        self.warmup_model()       # Warm up model to initialize GPU memory patterns
        self.allocate_kv_cache()  # Allocate KV cache for efficient attention computation

        # Capture CUDA graphs for optimized inference (unless in eager mode)
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Restore original defaults for CPU operations
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Set up shared memory for inter-process communication in multi-GPU setup
        if self.world_size > 1:
            if rank == 0:
                # Main process creates shared memory for command passing
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # Wait for other processes
            else:
                # Worker processes wait for shared memory creation then connect
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # Start worker loop to receive commands

    def exit(self):
        """Clean up resources and shut down the model runner."""
        if self.world_size > 1:
            self.shm.close()  # Close shared memory connection
            dist.barrier()    # Synchronize all processes
            if self.rank == 0:
                self.shm.unlink()  # Remove shared memory object (main process only)
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # Clean up CUDA graphs
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
        dist.destroy_process_group()  # Shut down distributed process group

    def loop(self):
        """Main worker loop for non-main processes in multi-GPU setup.
        Continuously reads commands from shared memory and executes them.
        """
        while True:
            method_name, args = self.read_shm()  # Read command from shared memory
            self.call(method_name, *args)        # Execute the requested method
            if method_name == "exit":            # Exit condition
                break

    def read_shm(self):
        """Read a command from shared memory (worker processes only)."""
        assert self.world_size > 1 and self.rank > 0  # Only workers read from shared memory
        self.event.wait()  # Wait for main process to write data
        n = int.from_bytes(self.shm.buf[0:4], "little")  # Read data length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # Deserialize command and args
        self.event.clear()  # Clear event to signal data was read
        return method_name, args

    def write_shm(self, method_name, *args):
        """Write a command to shared memory (main process only)."""
        assert self.world_size > 1 and self.rank == 0  # Only main process writes to shared memory
        data = pickle.dumps([method_name, *args])  # Serialize command and arguments
        n = len(data)  # Get data length
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # Write length prefix
        self.shm.buf[4:n+4] = data  # Write serialized data
        for event in self.event:
            event.set()  # Signal all worker processes that data is ready

    def call(self, method_name, *args):
        """Execute a method call, potentially across multiple processes."""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # Broadcast command to workers
        method = getattr(self, method_name, None)  # Get method by name
        return method(*args)  # Execute the method

    def warmup_model(self):
        """Warm up the model by running a forward pass to initialize GPU memory patterns.
        This helps ensure consistent performance and memory usage during actual inference.
        """
        torch.cuda.empty_cache()  # Clear unused GPU memory
        torch.cuda.reset_peak_memory_stats()  # Reset memory statistics
        # Calculate number of sequences for warmup based on max batch constraints
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # Create dummy sequences filled with padding tokens for warmup
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)  # Run prefill pass for warmup
        torch.cuda.empty_cache()  # Clear cache after warmup

    def allocate_kv_cache(self):
        """Allocate KV cache memory for efficient attention computation.
        KV cache stores key/value pairs from previous tokens to avoid recomputation.
        """
        config = self.config
        hf_config = config.hf_config

        # Get current GPU memory status
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Calculate KV cache parameters for this GPU in tensor parallel setup
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # Split heads across GPUs

        # Calculate memory needed per KV cache block (2 for K/V, layers, block_size, heads, head_dim, bytes_per_element)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                      num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize)

        # Calculate how many blocks can fit in remaining GPU memory
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "Not enough memory for KV cache"

        # Allocate KV cache tensor: [2, layers, blocks, block_size, kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                   self.block_size, num_kv_heads, hf_config.head_dim)

        # Assign KV cache slices to each attention layer in the model
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # Assign K cache (index 0) and V cache (index 1) for this layer
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """Prepare block tables for sequences by padding to equal length.
        Block tables track which KV cache blocks are used for each sequence.
        """
        max_len = max(len(seq.block_table) for seq in seqs)  # Find longest block table
        # Pad all block tables to same length with -1 (indicating unused blocks)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Prepare input data for prefill phase (processing new tokens in prompt).
        This phase computes attention for all tokens in new input sequences.
        """
        input_ids = []      # Token IDs for new input
        positions = []      # Position indices for each token
        cu_seqlens_q = [0]  # Cumulative sequence lengths for queries (starts with 0)
        cu_seqlens_k = [0]  # Cumulative sequence lengths for keys (starts with 0)
        max_seqlen_q = 0    # Maximum query sequence length
        max_seqlen_k = 0    # Maximum key sequence length
        slot_mapping = []   # Mapping of tokens to KV cache slots
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)  # Total sequence length
            # Extract tokens that aren't yet cached (new tokens to process)
            input_ids.extend(seq[seq.num_cached_tokens:])
            # Create position indices for new tokens
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens  # Number of query tokens
            seqlen_k = seqlen  # Number of key tokens (includes cached + new)

            # Update cumulative lengths for efficient attention computation
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:    # Skip during warmup (no block allocation)
                continue

            # Map new tokens to KV cache slots for storage
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size  # Start slot for this block
                if i != seq.num_blocks - 1:
                    end = start + self.block_size  # Full block
                else:
                    end = start + seq.last_block_num_tokens  # Partial last block
                slot_mapping.extend(list(range(start, end)))  # Add slot indices

        # Use block tables if we have cached context (prefix cache scenario)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        # Convert to GPU tensors with async transfer for better performance
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # Set context for efficient attention computation with FlashAttention
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Prepare input data for decode phase (generating next tokens).
        In this phase, we process one token per sequence and use cached context.
        """
        input_ids = []      # Last token from each sequence
        positions = []      # Position of last token in each sequence
        slot_mapping = []   # KV cache slot for storing new token
        context_lens = []   # Context length for each sequence

        for seq in seqs:
            input_ids.append(seq.last_token)  # Most recently generated token
            positions.append(len(seq) - 1)    # Position = sequence length - 1
            context_lens.append(len(seq))     # Full sequence length as context
            # Calculate slot for new token: last block + offset within block
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # Convert to GPU tensors with async transfer
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # Set context for decode phase (single token per sequence)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """Prepare sampling parameters (temperature) for token generation."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)  # Sampling temperature for each sequence
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Run model inference with optional CUDA graph optimization.

        Uses CUDA graphs for small batch sizes in decode phase for better performance,
        falls back to eager execution for large batches or prefill phase.
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Use eager execution for prefill, forced eager mode, or large batches
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # Use CUDA graph optimization for small decode batches
            bs = input_ids.size(0)  # Batch size
            context = get_context()  # Get attention context

            # Select appropriate pre-captured graph based on batch size
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars  # Pre-allocated graph variables

            # Copy input data to graph variables
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)  # Reset slot mapping
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()  # Reset context lengths
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            graph.replay()  # Execute the pre-captured computation graph
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Main inference entry point - prepare data, run model, and sample tokens."""
        # Prepare input data based on phase (prefill vs decode)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # Prepare sampling parameters (main process only, since sampling is not parallelized)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # Run model inference
        logits = self.run_model(input_ids, positions, is_prefill)
        # Sample next tokens from logits (main process only)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()  # Clear attention context for next iteration
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Capture CUDA computation graphs for optimized inference.

        Pre-captures model forward passes for different batch sizes to avoid
        kernel launch overhead during decode phase. This significantly improves
        performance for small batch sizes.
        """
        config = self.config
        hf_config = config.hf_config

        # Maximum batch size for graph capture (limited by sequences and 512)
        max_bs = min(self.config.max_num_seqs, 512)
        # Maximum number of KV cache blocks needed for max sequence length
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # Pre-allocate tensors for graph capture (decode phase format)
        input_ids = torch.zeros(max_bs, dtype=torch.int64)      # Token IDs
        positions = torch.zeros(max_bs, dtype=torch.int64)      # Position indices
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)   # KV cache slots
        context_lens = torch.zeros(max_bs, dtype=torch.int32)   # Context lengths
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # Block tables
        outputs = torch.zeros(max_bs, hf_config.hidden_size)    # Model outputs

        # Define batch sizes for which to capture graphs: powers of 2 + multiples of 16
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}      # Dictionary to store captured graphs
        self.graph_pool = None  # Memory pool for graph allocations

        # Capture graphs in reverse order (largest to smallest batch size)
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()  # Create new CUDA graph

            # Set context for this specific batch size
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                       block_tables=block_tables[:bs])

            # Warmup run (not captured) to initialize memory
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Capture the actual computation graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # Initialize memory pool with first graph
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            # Store captured graph for this batch size
            self.graphs[bs] = graph
            torch.cuda.synchronize()  # Ensure capture is complete
            reset_context()  # Clear context for next iteration

        # Store graph variables for efficient replay during inference
        self.graph_vars = dict(
            input_ids=input_ids,        # Input token IDs (mutable during replay)
            positions=positions,        # Position indices (mutable during replay)
            slot_mapping=slot_mapping,  # KV cache slot mapping (mutable during replay)
            context_lens=context_lens,  # Context lengths (mutable during replay)
            block_tables=block_tables,  # Block tables (mutable during replay)
            outputs=outputs,           # Model outputs (mutable during replay)
        )
