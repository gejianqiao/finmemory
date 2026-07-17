"""
Reflection Module for Hierarchical Memory System.

This module implements the reflection mechanism that updates memory based on
trading decision outcomes. It records profit/loss from trades, identifies which
memory items influenced decisions, and generates reflection notes to improve
future decision-making.

Paper Reference: Section 4.1, Figure 2 (Reflection Memory)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionOutcome:
    """
    Dataclass representing the outcome of a trading decision.
    
    Attributes:
        timestep: Trading day index when decision was made
        date: Actual date of decision
        ticker: Stock ticker symbol
        direction: Decision direction (buy/sell/hold)
        quantity: Number of shares traded
        entry_price: Price at which trade was executed
        exit_price: Price at which position was closed (if applicable)
        pnl: Profit/loss from this decision
        pnl_pct: Percentage return from this decision
        memory_indices: List of memory item indices that influenced decision
        outcome_type: Classification of outcome (positive/negative/neutral)
        reflection_generated: Whether reflection note was created
    """
    timestep: int
    date: datetime
    ticker: str
    direction: str
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    memory_indices: List[int] = field(default_factory=list)
    outcome_type: str = "neutral"  # positive, negative, neutral
    reflection_generated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestep": self.timestep,
            "date": self.date.isoformat() if self.date else None,
            "ticker": self.ticker,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "memory_indices": self.memory_indices,
            "outcome_type": self.outcome_type,
            "reflection_generated": self.reflection_generated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionOutcome":
        """Create from dictionary."""
        date_str = data.get("date")
        if date_str and isinstance(date_str, str):
            data["date"] = datetime.fromisoformat(date_str)
        return cls(**data)


@dataclass
class ReflectionNote:
    """
    Dataclass representing a reflection note generated from trading experience.
    
    Attributes:
        note_id: Unique identifier for this reflection
        timestep: When reflection was generated
        date: Actual date of reflection
        ticker: Stock ticker symbol
        outcome_type: Type of outcome that triggered reflection
        pnl_impact: P&L that triggered this reflection
        triggered_by_indices: Memory indices that led to the outcome
        lesson_learned: Text description of the lesson
        action_recommendation: Suggested action for similar future situations
        confidence: Confidence level in this reflection (0-1)
        reinforcement_value: Value to add to memory items (+2 positive, -1 negative)
    """
    note_id: int
    timestep: int
    date: datetime
    ticker: str
    outcome_type: str
    pnl_impact: float
    triggered_by_indices: List[int]
    lesson_learned: str
    action_recommendation: str
    confidence: float = 0.5
    reinforcement_value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "note_id": self.note_id,
            "timestep": self.timestep,
            "date": self.date.isoformat() if self.date else None,
            "ticker": self.ticker,
            "outcome_type": self.outcome_type,
            "pnl_impact": self.pnl_impact,
            "triggered_by_indices": self.triggered_by_indices,
            "lesson_learned": self.lesson_learned,
            "action_recommendation": self.action_recommendation,
            "confidence": self.confidence,
            "reinforcement_value": self.reinforcement_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionNote":
        """Create from dictionary."""
        date_str = data.get("date")
        if date_str and isinstance(date_str, str):
            data["date"] = datetime.fromisoformat(date_str)
        return cls(**data)


class ReflectionModule:
    """
    Reflection module for experience-based memory updates.
    
    This module implements the reflection mechanism described in the FinMemory paper,
    which learns from trading outcomes to improve future decisions. After each
    trading decision, it:
    
    1. Records the decision outcome (profit/loss)
    2. Identifies which memory items influenced the decision
    3. Reinforces memory items for positive outcomes (+2)
    4. Generates reflection notes for negative outcomes
    5. Updates reflection_memory_index in decision agents
    
    The reflection system helps the agent avoid repeating mistakes and reinforce
    successful patterns over time.
    
    Attributes:
        ticker: Stock ticker symbol
        outcomes: List of all decision outcomes
        reflections: List of all reflection notes
        positive_outcome_count: Count of positive outcomes
        negative_outcome_count: Count of negative outcomes
        neutral_outcome_count: Count of neutral outcomes
        total_pnl: Cumulative P&L from all tracked decisions
        reflection_threshold_positive: P&L % threshold for positive reflection
        reflection_threshold_negative: P&L % threshold for negative reflection
    """
    
    def __init__(
        self,
        ticker: str = "UNKNOWN",
        reflection_threshold_positive: float = 5.0,
        reflection_threshold_negative: float = -5.0
    ):
        """
        Initialize reflection module.
        
        Args:
            ticker: Stock ticker symbol
            reflection_threshold_positive: P&L % threshold for positive outcomes (>5%)
            reflection_threshold_negative: P&L % threshold for negative outcomes (<-5%)
        """
        self.ticker = ticker
        self.outcomes: List[DecisionOutcome] = []
        self.reflections: List[ReflectionNote] = []
        
        self.positive_outcome_count = 0
        self.negative_outcome_count = 0
        self.neutral_outcome_count = 0
        self.total_pnl = 0.0
        
        self.reflection_threshold_positive = reflection_threshold_positive
        self.reflection_threshold_negative = reflection_threshold_negative
        
        self._next_note_id = 0
        
        logger.info(f"ReflectionModule initialized for {ticker}")
    
    def record_outcome(
        self,
        timestep: int,
        date: datetime,
        direction: str,
        quantity: int,
        entry_price: float,
        exit_price: Optional[float],
        pnl: float,
        pnl_pct: float,
        memory_indices: List[int]
    ) -> DecisionOutcome:
        """
        Record a trading decision outcome.
        
        This method records the outcome of a trading decision and determines
        whether to generate a reflection note based on the P&L.
        
        Args:
            timestep: Trading day index
            date: Actual date of decision
            direction: Decision direction (buy/sell/hold)
            quantity: Number of shares traded
            entry_price: Entry price
            exit_price: Exit price (None if position still open)
            pnl: Profit/loss in dollars
            pnl_pct: Profit/loss as percentage
            memory_indices: Memory item indices that influenced decision
            
        Returns:
            DecisionOutcome: The recorded outcome
        """
        # Determine outcome type
        if pnl_pct >= self.reflection_threshold_positive:
            outcome_type = "positive"
            self.positive_outcome_count += 1
        elif pnl_pct <= self.reflection_threshold_negative:
            outcome_type = "negative"
            self.negative_outcome_count += 1
        else:
            outcome_type = "neutral"
            self.neutral_outcome_count += 1
        
        # Create outcome record
        outcome = DecisionOutcome(
            timestep=timestep,
            date=date,
            ticker=self.ticker,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            memory_indices=memory_indices.copy(),
            outcome_type=outcome_type
        )
        
        self.outcomes.append(outcome)
        self.total_pnl += pnl
        
        logger.debug(
            f"Recorded outcome: {direction} {quantity} shares, "
            f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%), type: {outcome_type}"
        )
        
        # Generate reflection if needed
        if outcome_type == "negative":
            self._generate_reflection(outcome)
            outcome.reflection_generated = True
        elif outcome_type == "positive":
            self._reinforce_memories(outcome)
        
        return outcome
    
    def _generate_reflection(self, outcome: DecisionOutcome) -> ReflectionNote:
        """
        Generate a reflection note from a negative outcome.
        
        This creates a lesson-learned note that will help avoid similar
        mistakes in the future.
        
        Args:
            outcome: The negative outcome to reflect on
            
        Returns:
            ReflectionNote: Generated reflection note
        """
        # Generate lesson learned based on outcome characteristics
        lesson = self._generate_lesson(outcome)
        recommendation = self._generate_recommendation(outcome)
        
        # Calculate confidence based on magnitude of loss
        confidence = min(0.9, 0.5 + abs(outcome.pnl_pct) / 20.0)
        
        # Create reflection note
        note = ReflectionNote(
            note_id=self._next_note_id,
            timestep=outcome.timestep,
            date=outcome.date,
            ticker=self.ticker,
            outcome_type=outcome.outcome_type,
            pnl_impact=outcome.pnl,
            triggered_by_indices=outcome.memory_indices,
            lesson_learned=lesson,
            action_recommendation=recommendation,
            confidence=confidence,
            reinforcement_value=-1.0  # Negative reinforcement for associated memories
        )
        
        self.reflections.append(note)
        self._next_note_id += 1
        
        logger.info(
            f"Generated reflection note #{note.note_id}: {lesson[:100]}..."
        )
        
        return note
    
    def _generate_lesson(self, outcome: DecisionOutcome) -> str:
        """
        Generate a lesson learned from the outcome.
        
        Args:
            outcome: The outcome to analyze
            
        Returns:
            str: Lesson learned text
        """
        direction = outcome.direction
        pnl_pct = outcome.pnl_pct
        
        if direction == "buy":
            if pnl_pct < -20:
                lesson = (
                    f"Severe loss ({pnl_pct:.1f}%) from buy decision. "
                    f"Consider: Was there insufficient risk analysis? "
                    f"Were warning signs in market signals ignored? "
                    f"Review memory items that suggested this buy opportunity."
                )
            elif pnl_pct < -10:
                lesson = (
                    f"Significant loss ({pnl_pct:.1f}%) from buy decision. "
                    f"Consider: Was position size too large for the risk? "
                    f"Were there conflicting signals in the analysis? "
                    f"Review whether CVaR constraints were properly applied."
                )
            else:
                lesson = (
                    f"Moderate loss ({pnl_pct:.1f}%) from buy decision. "
                    f"Consider: Was timing suboptimal? "
                    f"Were there better entry points available? "
                    f"Review technical indicators and sentiment at entry."
                )
        elif direction == "sell":
            if pnl_pct < -20:
                lesson = (
                    f"Severe loss ({pnl_pct:.1f}%) from sell decision. "
                    f"Consider: Was this a premature exit? "
                    f"Did price recover after selling? "
                    f"Review whether sell signal was based on temporary volatility."
                )
            elif pnl_pct < -10:
                lesson = (
                    f"Significant loss ({pnl_pct:.1f}%) from sell decision. "
                    f"Consider: Was this panic selling? "
                    f"Were fundamentals actually deteriorating? "
                    f"Review whether sell was reaction to noise vs. signal."
                )
            else:
                lesson = (
                    f"Moderate loss ({pnl_pct:.1f}%) from sell decision. "
                    f"Consider: Could holding longer have been better? "
                    f"Review whether exit timing aligned with strategy."
                )
        else:  # hold
            lesson = (
                f"Opportunity cost from hold decision ({pnl_pct:.1f}% vs potential). "
                f"Consider: Was this excessive caution? "
                f"Were there clear signals that were ignored? "
                f"Review whether hold was due to uncertainty or valid analysis."
            )
        
        return lesson
    
    def _generate_recommendation(self, outcome: DecisionOutcome) -> str:
        """
        Generate an action recommendation for similar future situations.
        
        Args:
            outcome: The outcome to analyze
            
        Returns:
            str: Action recommendation text
        """
        direction = outcome.direction
        pnl_pct = outcome.pnl_pct
        
        if direction == "buy" and pnl_pct < -10:
            return (
                "For similar buy signals in the future: "
                "1) Verify at least 2 independent bullish indicators, "
                "2) Check that CVaR-based position sizing is conservative, "
                "3) Look for confirmation in multiple timeframes, "
                "4) Consider waiting for pullback if momentum is overextended."
            )
        elif direction == "sell" and pnl_pct < -10:
            return (
                "For similar sell signals in the future: "
                "1) Distinguish between temporary volatility and trend change, "
                "2) Check if fundamentals have actually deteriorated, "
                "3) Consider partial position reduction instead of full exit, "
                "4) Set stop-loss levels before entering positions."
            )
        elif direction == "hold" and pnl_pct < -5:
            return (
                "For similar hold decisions in the future: "
                "1) Review whether hold was due to analysis paralysis, "
                "2) Check if clear signals were present but ignored, "
                "3) Consider smaller test positions when uncertain, "
                "4) Set clear criteria for action vs. inaction."
            )
        else:
            return (
                "Review this decision pattern and compare with successful trades. "
                "Identify key differences in market conditions, signals, and timing."
            )
    
    def _reinforce_memories(self, outcome: DecisionOutcome) -> None:
        """
        Reinforce memory items associated with positive outcomes.
        
        This method would typically call back to the hierarchical memory system
        to increase the reinforcement score of memory items that led to
        successful decisions.
        
        Args:
            outcome: The positive outcome
        """
        # Note: Actual reinforcement happens in HierarchicalMemory
        # This is a placeholder for the callback mechanism
        logger.debug(
            f"Reinforcing {len(outcome.memory_indices)} memory items "
            f"for positive outcome (+{outcome.pnl_pct:.1f}%)"
        )
    
    def get_outcomes(
        self,
        outcome_type: Optional[str] = None,
        min_pnl_pct: Optional[float] = None,
        max_pnl_pct: Optional[float] = None
    ) -> List[DecisionOutcome]:
        """
        Get filtered list of decision outcomes.
        
        Args:
            outcome_type: Filter by outcome type (positive/negative/neutral)
            min_pnl_pct: Minimum P&L percentage filter
            max_pnl_pct: Maximum P&L percentage filter
            
        Returns:
            List[DecisionOutcome]: Filtered outcomes
        """
        filtered = self.outcomes
        
        if outcome_type is not None:
            filtered = [o for o in filtered if o.outcome_type == outcome_type]
        
        if min_pnl_pct is not None:
            filtered = [o for o in filtered if o.pnl_pct >= min_pnl_pct]
        
        if max_pnl_pct is not None:
            filtered = [o for o in filtered if o.pnl_pct <= max_pnl_pct]
        
        return filtered
    
    def get_reflections(
        self,
        outcome_type: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[ReflectionNote]:
        """
        Get filtered list of reflection notes.
        
        Args:
            outcome_type: Filter by outcome type
            min_confidence: Minimum confidence threshold
            
        Returns:
            List[ReflectionNote]: Filtered reflections
        """
        filtered = self.reflections
        
        if outcome_type is not None:
            filtered = [r for r in filtered if r.outcome_type == outcome_type]
        
        if min_confidence is not None:
            filtered = [r for r in filtered if r.confidence >= min_confidence]
        
        return filtered
    
    def get_memory_indices_to_reinforce(
        self,
        min_confidence: float = 0.5
    ) -> Dict[int, float]:
        """
        Get memory indices that should be reinforced based on reflections.
        
        This aggregates reinforcement values across all reflections for each
        memory index, providing a net reinforcement score.
        
        Args:
            min_confidence: Minimum confidence threshold for reflections
            
        Returns:
            Dict[int, float]: Map of memory_index -> total reinforcement value
        """
        reinforcement_map: Dict[int, float] = {}
        
        for reflection in self.reflections:
            if reflection.confidence >= min_confidence:
                for idx in reflection.triggered_by_indices:
                    if idx not in reinforcement_map:
                        reinforcement_map[idx] = 0.0
                    reinforcement_map[idx] += reflection.reinforcement_value
        
        return reinforcement_map
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get reflection module statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        total_outcomes = len(self.outcomes)
        
        # Calculate outcome distribution
        if total_outcomes > 0:
            positive_ratio = self.positive_outcome_count / total_outcomes
            negative_ratio = self.negative_outcome_count / total_outcomes
            neutral_ratio = self.neutral_outcome_count / total_outcomes
        else:
            positive_ratio = negative_ratio = neutral_ratio = 0.0
        
        # Calculate average P&L by outcome type
        positive_outcomes = [o for o in self.outcomes if o.outcome_type == "positive"]
        negative_outcomes = [o for o in self.outcomes if o.outcome_type == "negative"]
        
        avg_positive_pnl = (
            sum(o.pnl for o in positive_outcomes) / len(positive_outcomes)
            if positive_outcomes else 0.0
        )
        avg_negative_pnl = (
            sum(o.pnl for o in negative_outcomes) / len(negative_outcomes)
            if negative_outcomes else 0.0
        )
        
        return {
            "ticker": self.ticker,
            "total_outcomes": total_outcomes,
            "positive_outcomes": self.positive_outcome_count,
            "negative_outcomes": self.negative_outcome_count,
            "neutral_outcomes": self.neutral_outcome_count,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "neutral_ratio": neutral_ratio,
            "total_pnl": self.total_pnl,
            "avg_positive_pnl": avg_positive_pnl,
            "avg_negative_pnl": avg_negative_pnl,
            "total_reflections": len(self.reflections),
            "reflection_threshold_positive": self.reflection_threshold_positive,
            "reflection_threshold_negative": self.reflection_threshold_negative
        }
    
    def get_recent_reflections(self, n: int = 10) -> List[ReflectionNote]:
        """
        Get the most recent reflection notes.
        
        Args:
            n: Number of recent reflections to return
            
        Returns:
            List[ReflectionNote]: Recent reflections
        """
        return self.reflections[-n:]
    
    def clear(self) -> None:
        """Clear all outcomes and reflections."""
        self.outcomes.clear()
        self.reflections.clear()
        self.positive_outcome_count = 0
        self.negative_outcome_count = 0
        self.neutral_outcome_count = 0
        self.total_pnl = 0.0
        self._next_note_id = 0
        
        logger.info("ReflectionModule cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert reflection module state to dictionary.
        
        Returns:
            Dict[str, Any]: Serialized state
        """
        return {
            "ticker": self.ticker,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "reflections": [r.to_dict() for r in self.reflections],
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionModule":
        """
        Create ReflectionModule from dictionary.
        
        Args:
            data: Serialized state dictionary
            
        Returns:
            ReflectionModule: Deserialized module
        """
        module = cls(
            ticker=data.get("ticker", "UNKNOWN"),
            reflection_threshold_positive=data.get("reflection_threshold_positive", 5.0),
            reflection_threshold_negative=data.get("reflection_threshold_negative", -5.0)
        )
        
        module.outcomes = [
            DecisionOutcome.from_dict(o) for o in data.get("outcomes", [])
        ]
        module.reflections = [
            ReflectionNote.from_dict(r) for r in data.get("reflections", [])
        ]
        
        # Update counters
        for outcome in module.outcomes:
            if outcome.outcome_type == "positive":
                module.positive_outcome_count += 1
            elif outcome.outcome_type == "negative":
                module.negative_outcome_count += 1
            else:
                module.neutral_outcome_count += 1
            module.total_pnl += outcome.pnl
        
        # Update next note ID
        if module.reflections:
            module._next_note_id = max(r.note_id for r in module.reflections) + 1
        
        return module


def create_reflection_module(
    ticker: str = "UNKNOWN",
    reflection_threshold_positive: float = 5.0,
    reflection_threshold_negative: float = -5.0
) -> ReflectionModule:
    """
    Factory function to create a configured ReflectionModule instance.
    
    Args:
        ticker: Stock ticker symbol
        reflection_threshold_positive: P&L % threshold for positive reflection
        reflection_threshold_negative: P&L % threshold for negative reflection
        
    Returns:
        ReflectionModule: Configured reflection module
    """
    return ReflectionModule(
        ticker=ticker,
        reflection_threshold_positive=reflection_threshold_positive,
        reflection_threshold_negative=reflection_threshold_negative
    )
