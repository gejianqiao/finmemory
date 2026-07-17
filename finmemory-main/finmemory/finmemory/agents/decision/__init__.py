"""
Decision Agents Module for FinMemory Trading System

This module exports the dual-agent decision architecture:
- DirectionAgent: Makes strategic buy/sell/hold decisions
- QuantityRiskAgent: Determines position sizing with CVaR constraints

The separation of direction and quantity decisions enables independent
risk management and prevents overconfidence from violating risk constraints.
"""

from finmemory.agents.decision.direction_agent import (
    DirectionAgent,
    create_direction_agent
)

from finmemory.agents.decision.quantity_risk_agent import (
    QuantityRiskAgent,
    create_quantity_risk_agent
)

__version__ = "1.0.0"

__all__ = [
    # Direction Agent
    "DirectionAgent",
    "create_direction_agent",
    
    # Quantity & Risk Agent
    "QuantityRiskAgent",
    "create_quantity_risk_agent",
]
