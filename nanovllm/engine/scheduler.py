from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    Scheduler for managing sequence execution in vLLM.
    
    The scheduler implements a two-phase execution model:
    1. PREFILL: Process new input sequences (initial forward pass)
    2. DECODE: Generate next tokens for running sequences (incremental generation)
    
    Key responsibilities:
    - Manage sequence queues (waiting vs running)
    - Allocate/deallocate KV cache blocks via BlockManager
    - Implement preemption for memory management
    - Batch sequences efficiently while respecting memory limits
    - Handle sequence completion and cleanup
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs              # Maximum sequences per batch
        self.max_num_batched_tokens = config.max_num_batched_tokens  # Max tokens per batch
        self.eos = config.eos                                # End-of-sequence token ID
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # Two main queues for sequence management
        self.waiting: deque[Sequence] = deque()  # New sequences waiting for prefill
        self.running: deque[Sequence] = deque()  # Sequences currently generating tokens

    def is_finished(self):
        """Check if all sequences have completed (no waiting or running sequences)."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Add a new sequence to the waiting queue for prefill processing."""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        Schedule sequences for execution in the next batch.
        
        Returns:
            tuple: (scheduled_sequences, is_prefill_phase)
            - scheduled_sequences: List of sequences to process
            - is_prefill_phase: True if this is a prefill batch, False if decode batch
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        """
        [1, 2, 3]
        [3]
        """
        # PHASE 1: PREFILL - Process new sequences from waiting queue
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # Check if we can fit this sequence in the current batch
            # Two constraints: token limit and memory availability
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or 
                not self.block_manager.can_allocate(seq)):
                break
            
            # Allocate resources and move sequence to running state
            num_seqs += 1
            self.block_manager.allocate(seq)  # Allocate KV cache blocks
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # Count new tokens (excluding cached)
            seq.status = SequenceStatus.RUNNING
            if seq.seq_id == 1:
                seq.cache_location = 'cpu'
                seq.cache_info = 1
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # If we scheduled prefill sequences, return them
        if scheduled_seqs:
            return scheduled_seqs, True

        # PHASE 2: DECODE - Generate next tokens for running sequences
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # Check if we can append a new token to this sequence
            # If not, we may need to preempt other sequences to free memory
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Preempt the last running sequence to free memory
                    self.preempt(self.running.pop())
                else:
                    # No other sequences to preempt, preempt current sequence
                    self.preempt(seq)
                    break
            else:
                # Successfully can append to this sequence
                num_seqs += 1
                self.block_manager.may_append(seq)  # Reserve space for next token
                scheduled_seqs.append(seq)
        
        # Ensure we scheduled at least one sequence for decode
        assert scheduled_seqs
        
        # Put scheduled sequences back in running queue (maintaining order)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        Preempt a running sequence to free memory for other sequences.
        
        Preemption is a key memory management technique in vLLM:
        - Moves sequence back to waiting queue
        - Deallocates its KV cache blocks
        - Allows other sequences to use the freed memory
        
        The sequence will need to be re-processed from scratch when scheduled again.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)  # Free KV cache blocks
        self.waiting.appendleft(seq)        # Put back in waiting queue (higher priority)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        Post-process generated tokens and handle sequence completion.
        
        Args:
            seqs: List of sequences that were processed
            token_ids: List of generated token IDs (one per sequence)
        
        Returns:
            list[bool]: List indicating which sequences finished (not used in this implementation)
        """
        status = []
        for seq, token_id in zip(seqs, token_ids):
            # Add the generated token to the sequence
            seq.append_token(token_id)
            
            # Check if sequence should be finished
            if ((not seq.ignore_eos and token_id == self.eos) or  # Generated EOS token
                seq.num_completion_tokens == seq.max_tokens):      # Reached max length
                
                # Mark sequence as finished and clean up resources
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # Free KV cache blocks
                self.running.remove(seq)            # Remove from running queue

                # Append status
                status.append(True)
            else:
                status.append(False)
        
        return status