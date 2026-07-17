"""
Memory Module for FinMemory Trading Agent System

This module provides the hierarchical memory system with importance-based allocation
and reflection-based updates as described in Section 4.1 of the FinMemory paper.

Components:
- HierarchicalMemory: Three-layer memory system (shallow, intermediate, deep)
- MemoryAllocator: Dynamic importance-based memory allocation
- ReflectionModule: Experience-based memory updates from trading outcomes
"""

from finmemory.memory.hierarchical_memory import (
    HierarchicalMemory,
    MemoryItem,
    ReflectionNote,
    create_hierarchical_memory,
)

from finmemory.memory.memory_allocator import (
    MemoryAllocator,
    AllocationResult,
    AllocationStats,
    create_memory_allocator,
)

from finmemory.memory.reflection_module import (
    ReflectionModule,
    DecisionOutcome,
    create_reflection_module,
)

__version__ = "1.0.0"

__all__ = [
    # Hierarchical Memory
    "HierarchicalMemory",
    "MemoryItem",
    "ReflectionNote",
    "create_hierarchical_memory",
    
    # Memory Allocator
    "MemoryAllocator",
    "AllocationResult",
    "AllocationStats",
    "create_memory_allocator",
    
    # Reflection Module
    "ReflectionModule",
    "DecisionOutcome",
    "create_reflection_module",
    
    # Version
    "__version__",
]