"""
FinMemory: A Position-Aware Trading Agent System for Real Financial Markets

A position-aware LLM trading agent system that explicitly models and manages
continuous positions through dual-agent decision architecture (Direction Agent +
Quantity/Risk Agent) and multi-timescale reward signals (1-day, 7-day, 30-day).

Paper: "FinMemory: A Position-Aware Trading Agent System for Real Financial Markets"
Core Contribution: Position-aware trading with hierarchical memory, CVaR-based
risk management, and multi-timescale reward design.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FinMemory Research Team"
__description__ = "Position-Aware Trading Agent System for Real Financial Markets"

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__description__",
]

# Note: Submodules should be imported explicitly
# Example usage:
#   from finmemory.agents.analysis import MarketContextAgent
#   from finmemory.environment import TradingEnvironment
#   from finmemory.risk import CVaRCalculator
#   from finmemory.memory import HierarchicalMemory
