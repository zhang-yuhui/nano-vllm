"""
LLM Engine - Core orchestrator for the simplified vLLM inference system.

This module implements the main engine that coordinates:
1. Multi-process tensor parallelism for distributed inference
2. Request scheduling and batching
3. Model execution and token generation
4. Output collection and decoding

Architecture:
- Spawns multiple worker processes for tensor parallel execution
- Uses a scheduler to batch requests efficiently (continuous batching)
- Delegates model execution to ModelRunner instances
- Manages the full lifecycle from prompt to completion
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """
        Initialize the LLM Engine with tensor parallelism support.

        Args:
            model: Model name or path to load
            **kwargs: Configuration parameters (max_num_tokens, max_num_seqs, etc.)

        Initialization steps:
        1. Parse config parameters from kwargs
        2. Spawn worker processes for tensor parallelism (ranks 1 to N-1)
        3. Initialize main ModelRunner (rank 0) that coordinates workers
        4. Load tokenizer and configure end-of-sequence token
        5. Initialize scheduler for request batching
        """
        # Extract only valid Config fields from kwargs
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # Setup tensor parallel workers (if tensor_parallel_size > 1)
        self.ps = []  # Worker processes
        self.events = []  # Synchronization events for workers
        ctx = mp.get_context("spawn")  # Use spawn to avoid CUDA context issues

        # Spawn worker processes for ranks 1 to N-1
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # Initialize main ModelRunner on rank 0 (coordinates all workers)
        self.model_runner = ModelRunner(config, 0, self.events)

        # Load tokenizer and configure EOS token
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # Initialize scheduler for continuous batching
        self.scheduler = Scheduler(config)

        # Register cleanup function to terminate workers on exit
        atexit.register(self.exit)

    def exit(self):
        """
        Cleanup function to gracefully shutdown all worker processes.

        Steps:
        1. Send exit signal to all workers via ModelRunner
        2. Delete the main ModelRunner to release GPU memory
        3. Wait for all worker processes to terminate
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        Add a new generation request to the engine.

        Args:
            prompt: Input text (will be tokenized) or pre-tokenized input IDs
            sampling_params: Parameters controlling generation (temperature, top_p, max_tokens, etc.)

        Process:
        1. Tokenize prompt if it's a string
        2. Create a Sequence object to track this request's state
        3. Add sequence to scheduler's waiting queue
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        Execute one iteration of the inference loop.

        This is the core of the continuous batching algorithm:

        Returns:
            outputs: List of (seq_id, token_ids) for completed sequences
            num_tokens: Number of tokens processed (positive=prefill, negative=decode)

        Algorithm steps:
        1. Scheduler selects sequences to run in this iteration
           - Batches waiting sequences for prefill (first token generation)
           - Batches running sequences for decode (subsequent tokens)
           - Uses KV cache block manager to ensure memory constraints

        2. ModelRunner executes the model forward pass
           - Prefill: Process all prompt tokens at once (parallel)
           - Decode: Generate one token per sequence (auto-regressive)
           - All tensor parallel workers execute synchronously

        3. Scheduler postprocesses results
           - Updates sequence states with new tokens
           - Checks stopping conditions (max_tokens, EOS, stop strings)
           - Manages KV cache blocks (allocate/free)

        4. Collect finished sequences and compute throughput metrics
        """
        # Scheduler decides which sequences to run (continuous batching)
        seqs, is_prefill = self.scheduler.schedule()

        # Execute model inference (coordinates all tensor parallel workers)
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # Update sequence states and manage KV cache
        self.scheduler.postprocess(seqs, token_ids)

        # Collect completed sequences for return
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # Track tokens processed: positive for prefill, negative for decode (for throughput calculation)
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """
        Check if all requests have completed.

        Returns:
            True if no sequences are waiting or running, False otherwise
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        High-level API for batch text generation.

        Args:
            prompts: List of input texts or pre-tokenized inputs
            sampling_params: Single SamplingParams or list (one per prompt)
            use_tqdm: Whether to show progress bar with throughput metrics

        Returns:
            List of dicts with "text" and "token_ids" for each prompt

        Algorithm:
        1. Add all prompts as requests to the scheduler
        2. Run continuous batching loop until all requests complete
           - Each step() processes a batch (mixed prefill + decode)
           - Dynamically reorders sequences for optimal batching
           - Tracks throughput for prefill and decode separately
        3. Collect and decode all outputs in original order
        4. Return decoded texts with token IDs

        This demonstrates vLLM's key innovation: continuous batching
        - Unlike traditional batching (wait for all to finish), sequences
          can complete at different times
        - New sequences can start while others are still generating
        - Maximizes GPU utilization and throughput
        """
        # Initialize progress bar
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # Broadcast single sampling_params to all prompts if needed
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Add all requests to scheduler queue
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # Storage for completed outputs (indexed by seq_id)
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # Main generation loop (continuous batching)
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            # Calculate and display throughput metrics
            if use_tqdm:
                if num_tokens > 0:  # Prefill step
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:  # Decode step (num_tokens is negative)
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # Collect completed sequences
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Sort outputs by original request order (seq_id)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # Decode token IDs to text
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        if use_tqdm:
            pbar.close()
        return outputs
 