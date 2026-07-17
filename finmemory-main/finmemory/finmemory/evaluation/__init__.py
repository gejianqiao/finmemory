"""
Evaluation module for the FinMemory trading agent system.

This module provides comprehensive performance evaluation metrics,
baseline implementations, and ablation study utilities for assessing
trading strategy performance.

Components:
- metrics.py: Core evaluation metrics (CR%, SR, MDD%, Calmar, CVaR)
- baselines/: 10 baseline implementations (LLM, DRL, rule-based)
- ablation_studies.py: Component ablation experiment utilities
"""

from finmemory.evaluation.metrics import (
    MetricsResult,
    MetricsCalculator,
    calculate_metrics,
    compare_strategies,
    generate_table1_results,
    generate_ablation_table,
)

__version__ = "1.0.0"

__all__ = [
    # Core metrics classes
    "MetricsResult",
    "MetricsCalculator",
    # Metric calculation functions
    "calculate_metrics",
    "compare_strategies",
    # Table generation functions
    "generate_table1_results",
    "generate_ablation_table",
]
