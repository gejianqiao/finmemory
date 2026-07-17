"""
Risk Management Module for FinMemory Trading Agent System.

This module provides CVaR-based risk calculations, position sizing with risk constraints,
and real-time exposure tracking for position-aware trading decisions.

Components:
- CVaRCalculator: Conditional Value at Risk calculations with rolling window support
- PositionSizer: CVaR-constrained position sizing engine
- ExposureTracker: Real-time portfolio risk exposure monitoring
"""

from finmemory.risk.cvar_calculator import (
    CVaRCalculator,
    CVaRResult,
    create_cvar_calculator,
    calculate_cvar_for_position_sizing,
)

from finmemory.risk.position_sizer import (
    PositionSizer,
    PositionSizeResult,
    create_position_sizer,
)

from finmemory.risk.exposure_tracker import (
    ExposureTracker,
    PositionExposure,
    ExposureSnapshot,
    ExposureAlert,
    create_exposure_tracker,
)

__version__ = "1.0.0"

__all__ = [
    # CVaR Calculator
    "CVaRCalculator",
    "CVaRResult",
    "create_cvar_calculator",
    "calculate_cvar_for_position_sizing",
    # Position Sizer
    "PositionSizer",
    "PositionSizeResult",
    "create_position_sizer",
    # Exposure Tracker
    "ExposureTracker",
    "PositionExposure",
    "ExposureSnapshot",
    "ExposureAlert",
    "create_exposure_tracker",
]
