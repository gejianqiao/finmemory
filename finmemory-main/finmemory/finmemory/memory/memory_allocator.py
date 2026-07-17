"""
Memory Allocator Module for FinMemory Trading System.

This module implements dynamic importance-based memory allocation as described in
Section 4.1 and Figure 2 of the FinMemory paper. It routes information items to
appropriate memory layers (shallow, intermediate, deep) based on importance scores.

Key Features:
- Importance score analysis (1-10 scale)
- Layer assignment based on score thresholds
- Capacity-aware allocation with overflow handling
- Integration with hierarchical memory system
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from finmemory.utils.logger import get_logger
from finmemory.memory.hierarchical_memory import HierarchicalMemory

logger = get_logger(__name__)


@dataclass
class AllocationResult:
    """Result of memory allocation decision."""
    success: bool
    layer: str  # 'shallow', 'intermediate', 'deep', 'rejected'
    importance_score: int
    reason: str
    item_index: Optional[int] = None
    evicted_item: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'success': self.success,
            'layer': self.layer,
            'importance_score': self.importance_score,
            'reason': self.reason,
            'item_index': self.item_index,
            'evicted_item': self.evicted_item is not None if self.evicted_item else None
        }


@dataclass
class AllocationStats:
    """Statistics about memory allocation patterns."""
    total_allocations: int = 0
    shallow_allocations: int = 0
    intermediate_allocations: int = 0
    deep_allocations: int = 0
    rejected_allocations: int = 0
    average_importance: float = 0.0
    importance_distribution: Dict[int, int] = None
    
    def __post_init__(self):
        if self.importance_distribution is None:
            self.importance_distribution = {i: 0 for i in range(1, 11)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_allocations': self.total_allocations,
            'shallow_allocations': self.shallow_allocations,
            'intermediate_allocations': self.intermediate_allocations,
            'deep_allocations': self.deep_allocations,
            'rejected_allocations': self.rejected_allocations,
            'average_importance': round(self.average_importance, 2),
            'importance_distribution': self.importance_distribution.copy()
        }


class MemoryAllocator:
    """
    Dynamic importance-based memory allocator for FinMemory system.
    
    Routes information items to appropriate memory layers based on importance scores:
    - Score 1-4: Shallow Memory (recent, low-importance)
    - Score 5-7: Intermediate Memory (medium-importance insights)
    - Score 8-10: Deep Memory (high-importance, strategic information)
    
    Attributes:
        memory: Reference to HierarchicalMemory instance for storage
        stats: Allocation statistics tracker
        layer_thresholds: Dict defining score ranges for each layer
    """
    
    def __init__(
        self,
        memory: HierarchicalMemory,
        shallow_threshold: int = 4,
        intermediate_threshold: int = 7,
        deep_threshold: int = 10,
        min_importance: int = 1,
        max_importance: int = 10
    ):
        """
        Initialize MemoryAllocator.
        
        Args:
            memory: HierarchicalMemory instance to manage allocations for
            shallow_threshold: Maximum score for shallow layer (default: 4)
            intermediate_threshold: Maximum score for intermediate layer (default: 7)
            deep_threshold: Maximum score for deep layer (default: 10)
            min_importance: Minimum valid importance score (default: 1)
            max_importance: Maximum valid importance score (default: 10)
        """
        self.memory = memory
        self.layer_thresholds = {
            'shallow': (min_importance, shallow_threshold),
            'intermediate': (shallow_threshold + 1, intermediate_threshold),
            'deep': (intermediate_threshold + 1, deep_threshold)
        }
        self.min_importance = min_importance
        self.max_importance = max_importance
        self.stats = AllocationStats()
        
        logger.info(
            f"MemoryAllocator initialized with thresholds: "
            f"shallow={self.layer_thresholds['shallow']}, "
            f"intermediate={self.layer_thresholds['intermediate']}, "
            f"deep={self.layer_thresholds['deep']}"
        )
    
    def allocate(
        self,
        content: str,
        importance_score: int,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AllocationResult:
        """
        Allocate a memory item to the appropriate layer based on importance score.
        
        Args:
            content: The content/information to store in memory
            importance_score: Score from 1-10 determining layer assignment
            source: Source of the information (e.g., 'company_news', '10k_filing')
            metadata: Optional additional metadata for the memory item
            
        Returns:
            AllocationResult with allocation decision and details
            
        Algorithm:
            1. Validate importance score (must be 1-10)
            2. Determine target layer based on score thresholds
            3. Check if target layer has capacity
            4. If full, evict oldest item (FIFO)
            5. Add item to target layer
            6. Update allocation statistics
        """
        # Validate importance score
        if not self._validate_importance_score(importance_score):
            logger.warning(f"Invalid importance score {importance_score}, rejecting allocation")
            self.stats.rejected_allocations += 1
            return AllocationResult(
                success=False,
                layer='rejected',
                importance_score=importance_score,
                reason=f"Invalid importance score: {importance_score} (must be {self.min_importance}-{self.max_importance})"
            )
        
        # Determine target layer
        target_layer = self._determine_layer(importance_score)
        logger.debug(f"Importance score {importance_score} maps to layer: {target_layer}")
        
        # Check capacity and evict if necessary
        evicted_item = None
        layer_capacity = self._get_layer_capacity(target_layer)
        layer_size = self._get_layer_size(target_layer)
        
        if layer_size >= layer_capacity:
            logger.info(
                f"Layer '{target_layer}' at capacity ({layer_size}/{layer_capacity}), "
                f"evicting oldest item"
            )
            evicted_item = self._evict_oldest_from_layer(target_layer)
        
        # Add item to memory
        try:
            item_index = self.memory.add_item(
                content=content,
                importance_score=importance_score,
                source=source,
                metadata=metadata
            )
            
            # Update statistics
            self._update_stats(importance_score, target_layer)
            
            result = AllocationResult(
                success=True,
                layer=target_layer,
                importance_score=importance_score,
                reason=f"Allocated to {target_layer} memory (score={importance_score})",
                item_index=item_index,
                evicted_item=evicted_item
            )
            
            logger.info(
                f"Memory allocation successful: layer={target_layer}, "
                f"score={importance_score}, index={item_index}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add item to memory: {e}")
            self.stats.rejected_allocations += 1
            return AllocationResult(
                success=False,
                layer='rejected',
                importance_score=importance_score,
                reason=f"Memory storage failed: {str(e)}"
            )
    
    def allocate_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[AllocationResult]:
        """
        Allocate multiple memory items in batch.
        
        Args:
            items: List of dicts with keys: content, importance_score, source, metadata
            
        Returns:
            List of AllocationResult for each item
        """
        results = []
        for item in items:
            result = self.allocate(
                content=item.get('content', ''),
                importance_score=item.get('importance_score', 5),
                source=item.get('source', 'unknown'),
                metadata=item.get('metadata')
            )
            results.append(result)
        
        logger.info(
            f"Batch allocation complete: {len([r for r in results if r.success])}/"
            f"{len(items)} items allocated successfully"
        )
        
        return results
    
    def analyze_importance(
        self,
        content: str,
        content_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Analyze content and assign importance score (1-10).
        
        This is a heuristic-based scorer. In production, this could be replaced
        with an LLM-based importance assessor.
        
        Args:
            content: The content to analyze
            content_type: Type of content ('news', 'filing', 'macro', 'price_signal')
            context: Optional context for scoring (e.g., market volatility)
            
        Returns:
            Importance score from 1-10
            
        Scoring Heuristics:
            - Earnings/financial metrics: +2-3 points
            - Major corporate actions (M&A, CEO change): +3-4 points
            - Macro economic data: +1-2 points
            - Regular news updates: +0-1 points
            - High volatility context: +1 point
        """
        base_score = 5  # Neutral starting point
        
        # Content type adjustments
        type_scores = {
            '10k_filing': 8,      # Annual reports are high importance
            '10q_filing': 7,      # Quarterly reports are medium-high
            'earnings': 8,        # Earnings announcements are critical
            'company_news': 5,    # Regular company news is medium
            'macro_news': 4,      # Macro news is medium-low relevance
            'price_signal': 6,    # Price signals are medium importance
            'analyst_rating': 6,  # Analyst ratings are medium
            'sec_filing': 7,      # SEC filings are important
            'default': 5
        }
        
        base_score = type_scores.get(content_type, type_scores['default'])
        
        # Content-based adjustments
        content_lower = content.lower()
        
        # High-impact keywords
        high_impact_keywords = [
            'merger', 'acquisition', 'bankruptcy', 'ceo', 'cfo', 'resign',
            'lawsuit', 'investigation', 'fraud', 'recall', 'warning',
            'earnings beat', 'earnings miss', 'revenue surge', 'profit surge',
            'guidance raise', 'guidance cut', 'dividend', 'split'
        ]
        
        for keyword in high_impact_keywords:
            if keyword in content_lower:
                base_score = min(base_score + 1, 10)
                break
        
        # Context adjustments
        if context:
            if context.get('high_volatility', False):
                base_score = min(base_score + 1, 10)
            if context.get('earnings_season', False) and content_type in ['company_news', '10q_filing']:
                base_score = min(base_score + 1, 10)
        
        # Ensure score is in valid range
        final_score = max(self.min_importance, min(self.max_importance, base_score))
        
        logger.debug(
            f"Importance analysis: type={content_type}, base={base_score}, "
            f"final={final_score}"
        )
        
        return final_score
    
    def get_layer_for_score(self, importance_score: int) -> str:
        """
        Get the target memory layer for an importance score.
        
        Args:
            importance_score: Score from 1-10
            
        Returns:
            Layer name: 'shallow', 'intermediate', or 'deep'
        """
        if not self._validate_importance_score(importance_score):
            return 'rejected'
        return self._determine_layer(importance_score)
    
    def get_allocation_stats(self) -> AllocationStats:
        """
        Get current allocation statistics.
        
        Returns:
            AllocationStats with current allocation metrics
        """
        # Update average importance
        if self.stats.total_allocations > 0:
            total_score = sum(
                score * count 
                for score, count in self.stats.importance_distribution.items()
            )
            self.stats.average_importance = total_score / self.stats.total_allocations
        
        return self.stats
    
    def reset_stats(self):
        """Reset allocation statistics."""
        self.stats = AllocationStats()
        logger.info("Memory allocation statistics reset")
    
    def _validate_importance_score(self, score: int) -> bool:
        """Validate that importance score is in valid range."""
        return isinstance(score, int) and self.min_importance <= score <= self.max_importance
    
    def _determine_layer(self, importance_score: int) -> str:
        """Determine target layer based on importance score."""
        if importance_score <= self.layer_thresholds['shallow'][1]:
            return 'shallow'
        elif importance_score <= self.layer_thresholds['intermediate'][1]:
            return 'intermediate'
        else:
            return 'deep'
    
    def _get_layer_capacity(self, layer: str) -> int:
        """Get capacity for a specific layer."""
        capacities = {
            'shallow': self.memory.shallow_capacity,
            'intermediate': self.memory.intermediate_capacity,
            'deep': self.memory.deep_capacity
        }
        return capacities.get(layer, 0)
    
    def _get_layer_size(self, layer: str) -> int:
        """Get current size of a specific layer."""
        sizes = {
            'shallow': len(self.memory.shallow_memory),
            'intermediate': len(self.memory.intermediate_memory),
            'deep': len(self.memory.deep_memory)
        }
        return sizes.get(layer, 0)
    
    def _evict_oldest_from_layer(self, layer: str) -> Optional[Dict[str, Any]]:
        """
        Get the oldest item from a specific layer (preview only).
        
        Args:
            layer: Layer to evict from
            
        Returns:
            Oldest item as dict, or None if layer was empty
        """
        if layer == 'shallow' and len(self.memory.shallow_memory) > 0:
            return self.memory.shallow_memory[0].to_dict()
        elif layer == 'intermediate' and len(self.memory.intermediate_memory) > 0:
            return self.memory.intermediate_memory[0].to_dict()
        elif layer == 'deep' and len(self.memory.deep_memory) > 0:
            return self.memory.deep_memory[0].to_dict()
        
        return None
    
    def _update_stats(self, importance_score: int, layer: str):
        """Update allocation statistics after successful allocation."""
        self.stats.total_allocations += 1
        self.stats.importance_distribution[importance_score] += 1
        
        if layer == 'shallow':
            self.stats.shallow_allocations += 1
        elif layer == 'intermediate':
            self.stats.intermediate_allocations += 1
        elif layer == 'deep':
            self.stats.deep_allocations += 1


def create_memory_allocator(
    memory: HierarchicalMemory,
    shallow_threshold: int = 4,
    intermediate_threshold: int = 7
) -> MemoryAllocator:
    """
    Factory function to create a configured MemoryAllocator instance.
    
    Args:
        memory: HierarchicalMemory instance to manage
        shallow_threshold: Maximum score for shallow layer (default: 4)
        intermediate_threshold: Maximum score for intermediate layer (default: 7)
        
    Returns:
        MemoryAllocator: Configured allocator instance
    """
    return MemoryAllocator(
        memory=memory,
        shallow_threshold=shallow_threshold,
        intermediate_threshold=intermediate_threshold
    )
