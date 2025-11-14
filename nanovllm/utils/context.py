from dataclasses import dataclass
import torch
from typing import Literal

from nanovllm.engine.block_location import BlockLocation


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    slot_mapping_cpu: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    context_lens_cpu: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    block_tables_cpu: torch.Tensor | None = None
    cache_locations: list[BlockLocation] | None = None
    cpu_kv_cache: bool = False


_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, 
    cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, 
    slot_mapping=None, slot_mapping_cpu=None, 
    context_lens=None, context_lens_cpu=None, 
    block_tables=None, block_tables_cpu=None, 
    cache_locations=None, cpu_kv_cache=False):
    
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                        slot_mapping, slot_mapping_cpu, 
                        context_lens, context_lens_cpu, 
                        block_tables, block_tables_cpu, 
                        cache_locations, cpu_kv_cache, )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
