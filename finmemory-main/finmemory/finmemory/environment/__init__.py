"""
Environment module for FinMemory position-aware trading system.

This module provides the core trading environment infrastructure including:
- Position-aware trading environment with continuous position management
- Position state tracking and trade execution
- Multi-timescale reward calculation
- Risk metrics computation (CR, SR, MDD, Calmar, CVaR)
"""

from finmemory.environment.trading_env import (
    TradingEnvironment,
    TradingState,
    TradeResult,
    EpisodeResult,
    create_trading_environment
)

from finmemory.environment.position_manager import (
    PositionManager,
    PositionState
)

from finmemory.environment.reward_calculator import (
    RewardCalculator,
    TrendSignals,
    RewardResult,
    create_reward_calculator
)

from finmemory.environment.risk_metrics import (
    RiskMetricsCalculator,
    RiskMetricsResult,
    calculate_risk_metrics,
    calculate_cvar_for_position_sizing,
    calculate_max_position_size
)

__version__ = "1.0.0"

__all__ = [
    # Trading Environment
    "TradingEnvironment",
    "TradingState",
    "TradeResult",
    "EpisodeResult",
    "create_trading_environment",
    
    # Position Management
    "PositionManager",
    "PositionState",
    
    # Reward Calculation
    "RewardCalculator",
    "TrendSignals",
    "RewardResult",
    "create_reward_calculator",
    
    # Risk Metrics
    "RiskMetricsCalculator",
    "RiskMetricsResult",
    "calculate_risk_metrics",
    "calculate_cvar_for_position_sizing",
    "calculate_max_position_size",
    
    # Module version
    "__version__"
]
