from collections import deque
import xxhash
import numpy as np
from typing import Optional

from nanovllm.engine.block_location import BlockLocation
from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(
        self, 
        num_gpu_blocks: int, 
        gpu_block_size: int,
        num_cpu_blocks: int = 0,
        cpu_block_size: Optional[int] = None
    ):
        # GPU blocks
        self.gpu_block_size = gpu_block_size
        self.gpu_blocks: list[Block] = [Block(i) for i in range(num_gpu_blocks)]
        self.gpu_hash_to_block_id: dict[int, int] = dict()
        self.free_gpu_block_ids: deque[int] = deque(range(num_gpu_blocks))
        self.used_gpu_block_ids: set[int] = set()
        
        # CPU blocks
        self.cpu_block_size = cpu_block_size or gpu_block_size
        self.cpu_blocks: list[Block] = [Block(i) for i in range(num_cpu_blocks)]
        self.cpu_hash_to_block_id: dict[int, int] = dict()
        self.free_cpu_block_ids: deque[int] = deque(range(num_cpu_blocks))
        self.used_cpu_block_ids: set[int] = set()
        
        # Default block size (for compatibility)
        self.block_size = self.gpu_block_size

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _get_blocks(self, location: BlockLocation) -> list[Block]:
        return self.gpu_blocks if location == BlockLocation.GPU else self.cpu_blocks
    
    def _get_block_size(self, location: BlockLocation) -> int:
        return self.gpu_block_size if location == BlockLocation.GPU else self.cpu_block_size
    
    def _get_hash_table(self, location: BlockLocation) -> dict[int, int]:
        return self.gpu_hash_to_block_id if location == BlockLocation.GPU else self.cpu_hash_to_block_id
    
    def _get_free_ids(self, location: BlockLocation) -> deque[int]:
        return self.free_gpu_block_ids if location == BlockLocation.GPU else self.free_cpu_block_ids
    
    def _get_used_ids(self, location: BlockLocation) -> set[int]:
        return self.used_gpu_block_ids if location == BlockLocation.GPU else self.used_cpu_block_ids

    def _allocate_block(self, block_id: int, location: BlockLocation) -> Block:
        blocks = self._get_blocks(location)
        free_ids = self._get_free_ids(location)
        used_ids = self._get_used_ids(location)
        
        block = blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        free_ids.remove(block_id)
        used_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int, location: BlockLocation):
        blocks = self._get_blocks(location)
        free_ids = self._get_free_ids(location)
        used_ids = self._get_used_ids(location)
        
        assert blocks[block_id].ref_count == 0
        used_ids.remove(block_id)
        free_ids.append(block_id)

    def can_allocate(self, seq: Sequence, location: BlockLocation = BlockLocation.GPU) -> bool:
        """Check if sequence can be allocated on specified device"""
        free_ids = self._get_free_ids(location)
        # if location == BlockLocation.CPU:
        #     print(len(free_ids))
        return len(free_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence, location: BlockLocation = BlockLocation.GPU):
        """Allocate sequence on specified device"""
        assert not seq.block_table
        assert self.can_allocate(seq, location)
        
        # Set sequence location
        seq.cache_location = location
        
        blocks = self._get_blocks(location)
        block_size = self._get_block_size(location)
        hash_table = self._get_hash_table(location)
        free_ids = self._get_free_ids(location)
        used_ids = self._get_used_ids(location)
        
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == block_size else -1
            block_id = hash_table.get(h, -1)
            
            if block_id == -1 or blocks[block_id].token_ids != token_ids:
                cache_miss = True
                
            if cache_miss:
                block_id = free_ids[0]
                block = self._allocate_block(block_id, location)
            else:
                seq.num_cached_tokens += block_size
                if block_id in used_ids:
                    block = blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id, location)
                    
            if h != -1:
                block.update(h, token_ids)
                hash_table[h] = block_id
                
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Deallocate sequence from its current location"""
        if not hasattr(seq, 'cache_location') or seq.cache_location is None:
            # Backward compatibility - assume GPU
            seq.cache_location = BlockLocation.GPU
        
        location = seq.cache_location
        blocks = self._get_blocks(location)
        
        for block_id in reversed(seq.block_table):
            block = blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id, location)
                
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.cache_location = None

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append to sequence at its current location"""
        if not hasattr(seq, 'cache_location') or seq.cache_location is None:
            seq.cache_location = BlockLocation.GPU
        
        location = seq.cache_location
        block_size = self._get_block_size(location)
        free_ids = self._get_free_ids(location)
        
        needs_new_block = (len(seq) % block_size == 1)
        return len(free_ids) >= needs_new_block

    def may_append(self, seq: Sequence):
        """Append tokens to sequence at its current location"""
        if not hasattr(seq, 'cache_location') or seq.cache_location is None:
            seq.cache_location = BlockLocation.GPU
        
        location = seq.cache_location
        blocks = self._get_blocks(location)
        block_size = self._get_block_size(location)
        hash_table = self._get_hash_table(location)
        free_ids = self._get_free_ids(location)
        
        block_table = seq.block_table
        last_block = blocks[block_table[-1]]
        
        if len(seq) % block_size == 1:
            # Need new block
            assert last_block.hash != -1
            block_id = free_ids[0]
            self._allocate_block(block_id, location)
            block_table.append(block_id)
            
        elif len(seq) % block_size == 0:
            # Just filled last block
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            hash_table[h] = last_block.block_id
        else:
            # Partially filled block
            assert last_block.hash == -1

    # =============== Status Methods ===============
    
    def is_full(self, location: BlockLocation) -> bool:
        """Check if specified device memory is full"""
        free_ids = self._get_free_ids(location)
        return len(free_ids) == 0
    
    def is_gpu_full(self) -> bool:
        """Check if GPU memory is full"""
        return self.is_full(BlockLocation.GPU)

    def is_cpu_full(self) -> bool:
        """Check if CPU memory is full"""
        return self.is_full(BlockLocation.CPU)

    def get_usage(self, location: BlockLocation) -> tuple[int, int]:
        """Returns (used_blocks, total_blocks) for specified device"""
        used_ids = self._get_used_ids(location)
        blocks = self._get_blocks(location)
        return len(used_ids), len(blocks)
    
    def get_gpu_usage(self) -> tuple[int, int]:
        """Returns (used_blocks, total_blocks) for GPU"""
        return self.get_usage(BlockLocation.GPU)

    def get_cpu_usage(self) -> tuple[int, int]:
        """Returns (used_blocks, total_blocks) for CPU"""
        return self.get_usage(BlockLocation.CPU)

    def num_free_blocks(self, location: BlockLocation) -> int:
        """Get number of free blocks on specified device"""
        free_ids = self._get_free_ids(location)
        return len(free_ids)
    
    def num_free_gpu_blocks(self) -> int:
        """Get number of free GPU blocks"""
        return self.num_free_blocks(BlockLocation.GPU)

    def num_free_cpu_blocks(self) -> int:
        """Get number of free CPU blocks"""
        return self.num_free_blocks(BlockLocation.CPU)