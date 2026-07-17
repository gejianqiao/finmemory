"""
Hierarchical Memory System for FinMemory Trading Agent.

Implements three-layer memory architecture (shallow, intermediate, deep) with
importance-based allocation and reflection-based updates as described in
Section 4.1 and Figure 2 of the FinMemory paper.

Memory Layers:
- Shallow Memory: Recent, low-importance information (capacity: 10 items)
- Intermediate Memory: Medium-importance insights (capacity: 30 items)
- Deep Memory: High-importance strategic information (capacity: 100 items)
- Reflection Memory: Experiential learnings from past decisions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryItem:
    """
    Represents a single item stored in hierarchical memory.
    
    Attributes:
        content: The actual information content (text, metrics, insights)
        importance_score: Importance rating 1-10 assigned by preprocessing or an active agent
        source: Origin of information (e.g., 'company_news', 'macro_news', '10-K', '10-Q')
        timestamp: When the information was added to memory
        ticker: Related stock ticker symbol
        metadata: Additional context (sentiment, relevance, etc.)
        access_count: Number of times this item has been accessed
        last_accessed: Timestamp of last access
        reinforcement_score: Score from reflection mechanism (-10 to +10)
    """
    content: str
    importance_score: int
    source: str
    timestamp: datetime
    ticker: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    reinforcement_score: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory item to dictionary for serialization."""
        return {
            'content': self.content,
            'importance_score': self.importance_score,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.ticker,
            'metadata': self.metadata,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'reinforcement_score': self.reinforcement_score,
            'effective_importance': self.get_effective_importance()
        }
    
    def get_effective_importance(self) -> int:
        """
        Calculate effective importance considering reinforcement.
        
        Returns:
            Effective importance score (1-10, clamped)
        """
        effective = self.importance_score + (self.reinforcement_score / 10)
        return max(1, min(10, int(round(effective))))
    
    def access(self) -> None:
        """Record an access to this memory item."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def reinforce(self, delta: int) -> None:
        """
        Apply reinforcement from reflection mechanism.
        
        Args:
            delta: Reinforcement change (-10 to +10)
        """
        self.reinforcement_score = max(-10, min(10, self.reinforcement_score + delta))
        logger.debug(f"Memory item reinforced: delta={delta}, new_score={self.reinforcement_score}")


@dataclass
class ReflectionNote:
    """
    Represents a reflection note generated after trading decisions.
    
    Attributes:
        decision_outcome: Result of the decision (profit/loss percentage)
        decision_type: Type of decision (buy/sell/hold)
        influenced_by: List of memory item indices that influenced the decision
        lesson_learned: Extracted lesson or pattern to remember/avoid
        timestamp: When the reflection was created
        ticker: Related stock ticker
        severity: Severity of outcome (-10 to +10, negative=loss, positive=profit)
    """
    decision_outcome: float
    decision_type: str
    influenced_by: List[int]
    lesson_learned: str
    timestamp: datetime
    ticker: str
    severity: int = 0
    
    def __post_init__(self):
        """Calculate severity based on outcome."""
        if self.decision_outcome > 5:
            self.severity = min(10, int(self.decision_outcome))
        elif self.decision_outcome < -5:
            self.severity = max(-10, int(self.decision_outcome))
        else:
            self.severity = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reflection note to dictionary."""
        return {
            'decision_outcome': self.decision_outcome,
            'decision_type': self.decision_type,
            'influenced_by': self.influenced_by,
            'lesson_learned': self.lesson_learned,
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.ticker,
            'severity': self.severity
        }


class HierarchicalMemory:
    """
    Hierarchical memory system with three layers and reflection mechanism.
    
    Implements the memory architecture from FinMemory paper Section 4.1:
    - Shallow Memory: capacity=10, importance 1-4
    - Intermediate Memory: capacity=30, importance 5-7
    - Deep Memory: capacity=100, importance 8-10
    - Reflection Memory: experiential learnings
    
    Memory Allocation Algorithm:
    1. Analysis agents assign importance score (1-10)
    2. Score >= 8: Store in Deep Memory
    3. Score 5-7: Store in Intermediate Memory
    4. Score <= 4: Store in Shallow Memory
    5. When memory full: Remove oldest item from that layer (FIFO)
    
    Attributes:
        shallow_capacity: Maximum items in shallow memory (default: 10)
        intermediate_capacity: Maximum items in intermediate memory (default: 30)
        deep_capacity: Maximum items in deep memory (default: 100)
        reflection_capacity: Maximum reflection notes (default: 50)
    """
    
    def __init__(
        self,
        shallow_capacity: int = 10,
        intermediate_capacity: int = 30,
        deep_capacity: int = 100,
        reflection_capacity: int = 50,
        ticker: str = "UNKNOWN"
    ):
        """
        Initialize hierarchical memory system.
        
        Args:
            shallow_capacity: Max items in shallow memory (low importance)
            intermediate_capacity: Max items in intermediate memory (medium importance)
            deep_capacity: Max items in deep memory (high importance)
            reflection_capacity: Max reflection notes to retain
            ticker: Stock ticker this memory system is for
        """
        self.ticker = ticker
        self.shallow_capacity = shallow_capacity
        self.intermediate_capacity = intermediate_capacity
        self.deep_capacity = deep_capacity
        self.reflection_capacity = reflection_capacity
        
        # Initialize memory layers as deques for efficient FIFO operations
        self.shallow_memory: deque[MemoryItem] = deque(maxlen=shallow_capacity)
        self.intermediate_memory: deque[MemoryItem] = deque(maxlen=intermediate_capacity)
        self.deep_memory: deque[MemoryItem] = deque(maxlen=deep_capacity)
        self.reflection_memory: deque[ReflectionNote] = deque(maxlen=reflection_capacity)
        
        # Track global item indices for reference by agents
        self._next_index = 0
        self._item_index_map: Dict[int, Tuple[str, int]] = {}  # index -> (layer, position)
        
        # Statistics tracking
        self.stats = {
            'items_added': 0,
            'items_evicted': 0,
            'reflections_generated': 0,
            'total_accesses': 0
        }
        
        logger.info(f"HierarchicalMemory initialized for {ticker}: "
                   f"shallow={shallow_capacity}, intermediate={intermediate_capacity}, "
                   f"deep={deep_capacity}, reflection={reflection_capacity}")
    
    def add_item(
        self,
        content: str,
        importance_score: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Add a new item to appropriate memory layer based on importance.
        
        Allocation Algorithm:
        - Score >= 8: Deep Memory
        - Score 5-7: Intermediate Memory
        - Score <= 4: Shallow Memory
        
        Args:
            content: Information content to store
            importance_score: Importance rating 1-10
            source: Origin of information ('company_news', 'macro_news', '10-K', '10-Q')
            metadata: Additional context (sentiment, relevance, etc.)
            timestamp: When the information was created (default: now)
        
        Returns:
            int: Global index assigned to this memory item
        
        Raises:
            ValueError: If importance_score not in range 1-10
        """
        if not 1 <= importance_score <= 10:
            raise ValueError(f"Importance score must be 1-10, got {importance_score}")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        
        # Create memory item
        item = MemoryItem(
            content=content,
            importance_score=importance_score,
            source=source,
            timestamp=timestamp,
            ticker=self.ticker,
            metadata=metadata
        )
        
        # Determine target layer based on importance
        if importance_score >= 8:
            target_layer = 'deep'
            target_memory = self.deep_memory
            evicted = len(target_memory) >= self.deep_capacity
        elif importance_score >= 5:
            target_layer = 'intermediate'
            target_memory = self.intermediate_memory
            evicted = len(target_memory) >= self.intermediate_capacity
        else:
            target_layer = 'shallow'
            target_memory = self.shallow_memory
            evicted = len(target_memory) >= self.shallow_capacity
        
        # Check if eviction will occur
        if evicted:
            old_item = target_memory[0] if target_memory else None
            if old_item:
                # Remove from index map
                for idx, (layer, pos) in list(self._item_index_map.items()):
                    if layer == target_layer and pos == 0:
                        del self._item_index_map[idx]
                        break
                self.stats['items_evicted'] += 1
                logger.debug(f"Evicted item from {target_layer} memory: {old_item.content[:50]}...")
        
        # Add item to target layer
        target_memory.append(item)
        
        # Assign global index and update index map
        item_index = self._next_index
        self._next_index += 1
        self._item_index_map[item_index] = (target_layer, len(target_memory) - 1)
        
        # Update statistics
        self.stats['items_added'] += 1
        
        logger.debug(f"Added item to {target_layer} memory (importance={importance_score}): "
                    f"{content[:50]}... [index={item_index}]")
        
        return item_index
    
    def add_reflection(
        self,
        decision_outcome: float,
        decision_type: str,
        influenced_by: List[int],
        lesson_learned: str,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Add a reflection note based on trading decision outcome.
        
        Reflection Update Mechanism:
        1. Record decision outcome (profit/loss)
        2. Identify which memory items influenced decision
        3. If outcome positive: Reinforce those memory indices
        4. If outcome negative: Add reflection note to avoid similar patterns
        
        Args:
            decision_outcome: Profit/loss percentage from the decision
            decision_type: Type of decision ('buy', 'sell', 'hold')
            influenced_by: List of memory item indices that influenced decision
            lesson_learned: Extracted lesson or pattern
            timestamp: When reflection was created (default: now)
        
        Returns:
            int: Index of the reflection note
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create reflection note
        reflection = ReflectionNote(
            decision_outcome=decision_outcome,
            decision_type=decision_type,
            influenced_by=influenced_by,
            lesson_learned=lesson_learned,
            timestamp=timestamp,
            ticker=self.ticker
        )
        
        # Add to reflection memory
        self.reflection_memory.append(reflection)
        
        # Update statistics
        self.stats['reflections_generated'] += 1
        
        # Apply reinforcement to influenced memory items
        if decision_outcome > 0:
            # Positive outcome: reinforce used memory items
            for item_idx in influenced_by:
                self._reinforce_item(item_idx, delta=2)
            logger.info(f"Positive outcome ({decision_outcome:.2f}%): reinforced {len(influenced_by)} memory items")
        elif decision_outcome < 0:
            # Negative outcome: add reflection and slightly penalize
            for item_idx in influenced_by:
                self._reinforce_item(item_idx, delta=-1)
            logger.warning(f"Negative outcome ({decision_outcome:.2f}%): added reflection, penalized {len(influenced_by)} memory items")
        
        logger.debug(f"Added reflection note: {lesson_learned[:50]}...")
        
        return len(self.reflection_memory) - 1
    
    def _reinforce_item(self, item_index: int, delta: int) -> bool:
        """
        Apply reinforcement to a specific memory item.
        
        Args:
            item_index: Global index of memory item
            delta: Reinforcement change (-10 to +10)
        
        Returns:
            bool: True if reinforcement applied successfully
        """
        if item_index not in self._item_index_map:
            logger.warning(f"Cannot reinforce: item index {item_index} not found")
            return False
        
        layer, position = self._item_index_map[item_index]
        
        # Get the appropriate memory layer
        if layer == 'shallow':
            memory = self.shallow_memory
        elif layer == 'intermediate':
            memory = self.intermediate_memory
        else:  # deep
            memory = self.deep_memory
        
        # Check if position is still valid (may have been evicted)
        if position >= len(memory):
            logger.warning(f"Cannot reinforce: position {position} evicted from {layer} memory")
            return False
        
        # Apply reinforcement
        memory[position].reinforce(delta)
        return True
    
    def get_working_memory(self, include_reflections: bool = True) -> Dict[str, Any]:
        """
        Get combined working memory from all layers for decision agents.
        
        Args:
            include_reflections: Whether to include reflection notes
        
        Returns:
            Dict containing:
                - shallow_items: List of shallow memory items
                - intermediate_items: List of intermediate memory items
                - deep_items: List of deep memory items
                - reflection_notes: List of reflection notes (if included)
                - total_items: Total count across all layers
                - layer_distribution: Count per layer
        """
        # Convert deques to lists and record accesses
        shallow_items = []
        for item in self.shallow_memory:
            item.access()
            self.stats['total_accesses'] += 1
            shallow_items.append(item.to_dict())
        
        intermediate_items = []
        for item in self.intermediate_memory:
            item.access()
            self.stats['total_accesses'] += 1
            intermediate_items.append(item.to_dict())
        
        deep_items = []
        for item in self.deep_memory:
            item.access()
            self.stats['total_accesses'] += 1
            deep_items.append(item.to_dict())
        
        working_memory = {
            'shallow_items': shallow_items,
            'intermediate_items': intermediate_items,
            'deep_items': deep_items,
            'total_items': len(shallow_items) + len(intermediate_items) + len(deep_items),
            'layer_distribution': {
                'shallow': len(shallow_items),
                'intermediate': len(intermediate_items),
                'deep': len(deep_items)
            }
        }
        
        if include_reflections:
            reflection_notes = [note.to_dict() for note in self.reflection_memory]
            working_memory['reflection_notes'] = reflection_notes
            working_memory['reflection_count'] = len(reflection_notes)
        
        return working_memory
    
    def get_item_by_index(self, item_index: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific memory item by its global index.
        
        Args:
            item_index: Global index of the memory item
        
        Returns:
            Dict representation of memory item, or None if not found
        """
        if item_index not in self._item_index_map:
            return None
        
        layer, position = self._item_index_map[item_index]
        
        # Get the appropriate memory layer
        if layer == 'shallow':
            memory = self.shallow_memory
        elif layer == 'intermediate':
            memory = self.intermediate_memory
        else:  # deep
            memory = self.deep_memory
        
        # Check if position is still valid
        if position >= len(memory):
            return None
        
        # Access the item and return
        item = memory[position]
        item.access()
        self.stats['total_accesses'] += 1
        
        return item.to_dict()
    
    def get_recent_items(self, n: int = 10, layer: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the n most recently added items from specified layer or all layers.
        
        Args:
            n: Number of items to retrieve
            layer: Specific layer ('shallow', 'intermediate', 'deep') or None for all
        
        Returns:
            List of memory item dictionaries, ordered by recency (newest first)
        """
        items = []
        
        if layer is None or layer == 'shallow':
            for item in list(self.shallow_memory)[-n:]:
                item.access()
                items.append(item.to_dict())
        
        if layer is None or layer == 'intermediate':
            for item in list(self.intermediate_memory)[-n:]:
                item.access()
                items.append(item.to_dict())
        
        if layer is None or layer == 'deep':
            for item in list(self.deep_memory)[-n:]:
                item.access()
                items.append(item.to_dict())
        
        # Sort by timestamp (newest first) and limit to n
        items.sort(key=lambda x: x['timestamp'], reverse=True)
        return items[:n]
    
    def get_items_by_source(self, source: str, layer: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all memory items from a specific source (e.g., 'company_news', '10-K').
        
        Args:
            source: Source type to filter by
            layer: Specific layer to search, or None for all layers
        
        Returns:
            List of matching memory item dictionaries
        """
        items = []
        layers_to_search = []
        
        if layer is None:
            layers_to_search = [
                ('shallow', self.shallow_memory),
                ('intermediate', self.intermediate_memory),
                ('deep', self.deep_memory)
            ]
        else:
            if layer == 'shallow':
                layers_to_search = [('shallow', self.shallow_memory)]
            elif layer == 'intermediate':
                layers_to_search = [('intermediate', self.intermediate_memory)]
            elif layer == 'deep':
                layers_to_search = [('deep', self.deep_memory)]
        
        for layer_name, memory in layers_to_search:
            for item in memory:
                if item.source == source:
                    item.access()
                    items.append(item.to_dict())
        
        return items
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        
        Returns:
            Dict containing usage statistics and capacity information
        """
        return {
            'ticker': self.ticker,
            'capacities': {
                'shallow': self.shallow_capacity,
                'intermediate': self.intermediate_capacity,
                'deep': self.deep_capacity,
                'reflection': self.reflection_capacity
            },
            'current_sizes': {
                'shallow': len(self.shallow_memory),
                'intermediate': len(self.intermediate_memory),
                'deep': len(self.deep_memory),
                'reflection': len(self.reflection_memory)
            },
            'utilization': {
                'shallow': len(self.shallow_memory) / self.shallow_capacity,
                'intermediate': len(self.intermediate_memory) / self.intermediate_capacity,
                'deep': len(self.deep_memory) / self.deep_capacity,
                'reflection': len(self.reflection_memory) / self.reflection_capacity
            },
            'stats': self.stats.copy(),
            'total_indexed_items': len(self._item_index_map)
        }
    
    def clear_layer(self, layer: str) -> int:
        """
        Clear all items from a specific memory layer.
        
        Args:
            layer: Layer to clear ('shallow', 'intermediate', 'deep', 'reflection')
        
        Returns:
            int: Number of items cleared
        """
        if layer == 'shallow':
            count = len(self.shallow_memory)
            self.shallow_memory.clear()
        elif layer == 'intermediate':
            count = len(self.intermediate_memory)
            self.intermediate_memory.clear()
        elif layer == 'deep':
            count = len(self.deep_memory)
            self.deep_memory.clear()
        elif layer == 'reflection':
            count = len(self.reflection_memory)
            self.reflection_memory.clear()
        else:
            raise ValueError(f"Unknown layer: {layer}")
        
        # Clean up index map for cleared layer
        if layer != 'reflection':
            self._item_index_map = {
                idx: (l, p) for idx, (l, p) in self._item_index_map.items()
                if l != layer
            }
        
        logger.info(f"Cleared {count} items from {layer} memory")
        return count
    
    def reset(self) -> None:
        """
        Reset all memory layers to empty state.
        """
        self.shallow_memory.clear()
        self.intermediate_memory.clear()
        self.deep_memory.clear()
        self.reflection_memory.clear()
        self._item_index_map.clear()
        self._next_index = 0
        self.stats = {
            'items_added': 0,
            'items_evicted': 0,
            'reflections_generated': 0,
            'total_accesses': 0
        }
        logger.info(f"HierarchicalMemory reset for {self.ticker}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire memory state to dictionary for serialization.
        
        Returns:
            Dict containing complete memory state
        """
        return {
            'ticker': self.ticker,
            'shallow_memory': [item.to_dict() for item in self.shallow_memory],
            'intermediate_memory': [item.to_dict() for item in self.intermediate_memory],
            'deep_memory': [item.to_dict() for item in self.deep_memory],
            'reflection_memory': [note.to_dict() for note in self.reflection_memory],
            'statistics': self.get_statistics(),
            'item_index_map': {str(k): v for k, v in self._item_index_map.items()}
        }


def create_hierarchical_memory(
    ticker: str = "UNKNOWN",
    shallow_capacity: int = 10,
    intermediate_capacity: int = 30,
    deep_capacity: int = 100,
    reflection_capacity: int = 50
) -> HierarchicalMemory:
    """
    Factory function to create a configured HierarchicalMemory instance.
    
    Args:
        ticker: Stock ticker this memory system is for
        shallow_capacity: Max items in shallow memory (default: 10)
        intermediate_capacity: Max items in intermediate memory (default: 30)
        deep_capacity: Max items in deep memory (default: 100)
        reflection_capacity: Max reflection notes (default: 50)
    
    Returns:
        HierarchicalMemory: Configured memory instance
    """
    return HierarchicalMemory(
        ticker=ticker,
        shallow_capacity=shallow_capacity,
        intermediate_capacity=intermediate_capacity,
        deep_capacity=deep_capacity,
        reflection_capacity=reflection_capacity
    )
