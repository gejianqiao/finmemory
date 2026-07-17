"""
Experiments module for the FinMemory trading agent system.

This module contains experiment runners for:
- Main evaluation experiments (Table 1 reproduction)
- Ablation studies (Table 2 reproduction)
- Sensitivity analysis (Figure 3 reproduction)
"""

from finmemory.experiments.main_experiment import (
    MainExperiment,
    run_main_experiment,
    ExperimentConfig,
    ExperimentResult,
    StockResult
)

from finmemory.experiments.ablation_experiment import (
    AblationExperiment,
    run_ablation_experiment,
    AblationConfig,
    AblationResult
)

from finmemory.experiments.sensitivity_analysis import (
    SensitivityAnalysis,
    run_sensitivity_analysis,
    TimescaleConfig,
    SensitivityResult
)

__version__ = "1.0.0"

__all__ = [
    # Main experiment
    "MainExperiment",
    "run_main_experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "StockResult",
    
    # Ablation experiment
    "AblationExperiment",
    "run_ablation_experiment",
    "AblationConfig",
    "AblationResult",
    
    # Sensitivity analysis
    "SensitivityAnalysis",
    "run_sensitivity_analysis",
    "TimescaleConfig",
    "SensitivityResult",
]
