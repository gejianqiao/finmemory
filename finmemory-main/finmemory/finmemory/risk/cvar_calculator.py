"""
CVaR Calculator Module for FinMemory Trading System.

This module implements Conditional Value at Risk (CVaR) calculations with rolling windows
for position sizing constraints. CVaR (also known as Expected Shortfall) measures the
expected loss in the worst-case tail of the return distribution.

Paper Reference: Section 4.2.2, Appendix B.2
Formula: CVaR_α = E[returns | returns ≤ VaR_α] where α=0.95
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CVaRResult:
    """Dataclass storing CVaR calculation results."""
    
    cvar: float
    """CVaR value (positive decimal representing loss magnitude)"""
    
    var: float
    """VaR value at the same confidence level"""
    
    confidence_level: float
    """Confidence level used (e.g., 0.95 for 95%)"""
    
    window_size: int
    """Number of returns used in calculation"""
    
    tail_returns: List[float]
    """Returns in the tail (≤ VaR)"""
    
    tail_count: int
    """Number of returns in the tail"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'cvar': self.cvar,
            'var': self.var,
            'confidence_level': self.confidence_level,
            'window_size': self.window_size,
            'tail_count': self.tail_count,
            'tail_mean': np.mean(self.tail_returns) if self.tail_returns else 0.0,
            'tail_std': np.std(self.tail_returns) if len(self.tail_returns) > 1 else 0.0
        }
    
    def summary(self) -> str:
        """Return human-readable summary string."""
        return (
            f"CVaR({self.confidence_level*100:.0f}%): {self.cvar*100:.2f}% | "
            f"VaR: {self.var*100:.2f}% | "
            f"Tail returns: {self.tail_count}/{self.window_size}"
        )


class CVaRCalculator:
    """
    Calculates Conditional Value at Risk (CVaR) with rolling window support.
    
    CVaR (Expected Shortfall) measures the expected loss given that the loss
    exceeds the VaR threshold. It provides a more conservative risk estimate
    than VaR alone by considering the severity of tail losses.
    
    Formula (95% confidence):
        Step 1: Collect rolling window of past returns (default: 20 days)
        Step 2: Calculate VaR_α = percentile(returns, (1-α)×100)
                VaR_95 = 5th percentile of returns (worst 5% cutoff)
        Step 3: Calculate CVaR_α = E[returns | returns ≤ VaR_α]
                CVaR_95 = average of all returns in the worst 5% tail
        Step 4: Return CVaR as positive value for sizing calculation
    
    Attributes:
        confidence_level: Confidence level for CVaR (default: 0.95)
        window_size: Rolling window size in days (default: 20)
        min_samples: Minimum samples required for calculation (default: 5)
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        window_size: int = 20,
        min_samples: int = 5
    ):
        """
        Initialize CVaR calculator.
        
        Args:
            confidence_level: Confidence level for CVaR (0.90, 0.95, or 0.99)
                Default is 0.95 (95% confidence) as per paper specification
            window_size: Rolling window size in days for historical returns
                Default is 20 days as per paper specification
            min_samples: Minimum number of samples required for calculation
                Returns 0.0 if fewer samples available
            
        Raises:
            ValueError: If confidence_level not in (0.90, 0.95, 0.99)
            ValueError: If window_size < min_samples
        """
        # Validate confidence level
        if confidence_level not in [0.90, 0.95, 0.99]:
            logger.warning(
                f"Unusual confidence level {confidence_level}. "
                f"Recommended: 0.90, 0.95, or 0.99"
            )
        
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"Confidence level must be between 0 and 1, got {confidence_level}"
            )
        
        # Validate window size
        if window_size < min_samples:
            raise ValueError(
                f"Window size ({window_size}) must be >= min_samples ({min_samples})"
            )
        
        self.confidence_level = confidence_level
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Calculate tail probability (e.g., 0.05 for 95% confidence)
        self.tail_probability = 1.0 - confidence_level
        
        logger.info(
            f"CVaRCalculator initialized: confidence={confidence_level*100:.0f}%, "
            f"window={window_size} days, min_samples={min_samples}"
        )
    
    def calculate_cvar(
        self,
        returns: List[float],
        use_rolling: bool = True
    ) -> CVaRResult:
        """
        Calculate CVaR from a list of returns.
        
        Args:
            returns: List of historical returns (as decimals, e.g., -0.02 for -2%)
            use_rolling: If True, use only the last window_size returns
                        If False, use all provided returns
        
        Returns:
            CVaRResult: Object containing CVaR, VaR, and diagnostic information
        
        Raises:
            ValueError: If insufficient data for calculation
        """
        if not returns or len(returns) == 0:
            raise ValueError("Returns list cannot be empty")
        
        # Apply rolling window if requested
        if use_rolling and len(returns) > self.window_size:
            effective_returns = returns[-self.window_size:]
        else:
            effective_returns = returns
        
        # Check minimum samples
        if len(effective_returns) < self.min_samples:
            logger.warning(
                f"Insufficient data for CVaR: {len(effective_returns)} < {self.min_samples}. "
                f"Returning 0.0"
            )
            return CVaRResult(
                cvar=0.0,
                var=0.0,
                confidence_level=self.confidence_level,
                window_size=len(effective_returns),
                tail_returns=[],
                tail_count=0
            )
        
        # Convert to numpy array for efficient calculation
        returns_array = np.array(effective_returns, dtype=float)
        
        # Calculate VaR (Value at Risk)
        # VaR_α = percentile(returns, (1-α)×100)
        # For 95% confidence: VaR = 5th percentile
        var_percentile = self.tail_probability * 100
        var = np.percentile(returns_array, var_percentile)
        
        # Calculate CVaR (Conditional Value at Risk)
        # CVaR_α = E[returns | returns ≤ VaR_α]
        # Average of all returns in the worst (1-α)% tail
        tail_mask = returns_array <= var
        tail_returns = returns_array[tail_mask].tolist()
        
        if len(tail_returns) == 0:
            # Edge case: no returns in tail (all returns > VaR)
            # Use VaR as conservative estimate for CVaR
            logger.debug(
                f"No returns in tail (all > VaR={var:.4f}). Using VaR as CVaR."
            )
            cvar = abs(var)
            tail_returns = []
        else:
            # CVaR is the mean of tail losses (returned as positive value)
            cvar = abs(np.mean(tail_returns))
        
        result = CVaRResult(
            cvar=cvar,
            var=abs(var),  # Return VaR as positive value
            confidence_level=self.confidence_level,
            window_size=len(effective_returns),
            tail_returns=tail_returns,
            tail_count=len(tail_returns)
        )
        
        logger.debug(
            f"CVaR calculated: {result.summary()} from {len(effective_returns)} returns"
        )
        
        return result
    
    def calculate_rolling_cvar(
        self,
        returns: List[float]
    ) -> List[CVaRResult]:
        """
        Calculate CVaR for each point in time using expanding window.
        
        This is useful for analyzing how CVaR evolves over time and for
        backtesting position sizing strategies.
        
        Args:
            returns: List of historical returns in chronological order
        
        Returns:
            List[CVaRResult]: CVaR result for each timestep (starting from min_samples)
        """
        if not returns or len(returns) < self.min_samples:
            logger.warning(
                f"Insufficient data for rolling CVaR: {len(returns) if returns else 0} < {self.min_samples}"
            )
            return []
        
        results = []
        
        # Calculate CVaR for each expanding window
        for i in range(self.min_samples, len(returns) + 1):
            window_returns = returns[:i]
            result = self.calculate_cvar(window_returns, use_rolling=False)
            results.append(result)
        
        logger.info(
            f"Calculated {len(results)} rolling CVaR values from {len(returns)} returns"
        )
        
        return results
    
    def calculate_cvar_from_prices(
        self,
        prices: List[float],
        use_log_returns: bool = True
    ) -> CVaRResult:
        """
        Calculate CVaR directly from price series.
        
        Convenience method that converts prices to returns before calculating CVaR.
        
        Args:
            prices: List of historical prices in chronological order
            use_log_returns: If True, use log returns; if False, use simple returns
                Log returns: log(p[t] / p[t-1])
                Simple returns: (p[t] - p[t-1]) / p[t-1]
        
        Returns:
            CVaRResult: CVaR calculation result
        """
        if not prices or len(prices) < 2:
            raise ValueError("Need at least 2 prices to calculate returns")
        
        # Convert prices to returns
        prices_array = np.array(prices, dtype=float)
        
        if use_log_returns:
            # Log returns: log(p[t] / p[t-1])
            returns = np.diff(np.log(prices_array)).tolist()
        else:
            # Simple returns: (p[t] - p[t-1]) / p[t-1]
            returns = np.diff(prices_array) / prices_array[:-1]
            returns = returns.tolist()
        
        return self.calculate_cvar(returns, use_rolling=True)
    
    def get_cvar_for_position_sizing(
        self,
        returns: List[float]
    ) -> float:
        """
        Get CVaR value specifically for position sizing calculations.
        
        This is the primary interface used by the position_sizer module.
        Returns CVaR as a positive decimal value representing the expected
        loss in the worst (1-α)% of cases.
        
        Args:
            returns: List of historical returns (as decimals)
        
        Returns:
            float: CVaR as positive decimal (e.g., 0.025 for 2.5% expected tail loss)
                  Returns 0.0 if insufficient data
        """
        try:
            result = self.calculate_cvar(returns, use_rolling=True)
            return result.cvar
        except Exception as e:
            logger.error(f"Error calculating CVaR for position sizing: {e}")
            return 0.0
    
    def get_statistics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive risk statistics for a return series.
        
        Args:
            returns: List of historical returns
        
        Returns:
            Dict containing: cvar, var, mean, std, min, max, skewness, kurtosis
        """
        if not returns or len(returns) < self.min_samples:
            return {
                'cvar': 0.0,
                'var': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        returns_array = np.array(returns, dtype=float)
        
        # Calculate CVaR
        cvar_result = self.calculate_cvar(returns, use_rolling=True)
        
        # Calculate additional statistics
        stats = {
            'cvar': cvar_result.cvar,
            'var': cvar_result.var,
            'mean': float(np.mean(returns_array)),
            'std': float(np.std(returns_array)),
            'min': float(np.min(returns_array)),
            'max': float(np.max(returns_array)),
            'skewness': float(self._calculate_skewness(returns_array)),
            'kurtosis': float(self._calculate_kurtosis(returns_array))
        }
        
        logger.debug(f"Risk statistics calculated: {stats}")
        
        return stats
    
    @staticmethod
    def _calculate_skewness(returns: np.ndarray) -> float:
        """Calculate skewness of return distribution."""
        if len(returns) < 3:
            return 0.0
        
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        
        if std == 0:
            return 0.0
        
        skew = np.sum(((returns - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return float(skew)
    
    @staticmethod
    def _calculate_kurtosis(returns: np.ndarray) -> float:
        """Calculate excess kurtosis of return distribution."""
        if len(returns) < 4:
            return 0.0
        
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        
        if std == 0:
            return 0.0
        
        kurt = (np.sum(((returns - mean) / std) ** 4) / n) - 3.0
        return float(kurt)


def create_cvar_calculator(
    confidence_level: float = 0.95,
    window_size: int = 20,
    min_samples: int = 5
) -> CVaRCalculator:
    """
    Factory function to create a CVaRCalculator with standard settings.
    
    Args:
        confidence_level: Confidence level for CVaR (default: 0.95)
        window_size: Rolling window size in days (default: 20)
        min_samples: Minimum samples required (default: 5)
    
    Returns:
        CVaRCalculator: Configured calculator instance
    
    Example:
        >>> calculator = create_cvar_calculator()
        >>> returns = [-0.02, 0.01, -0.03, 0.02, -0.01, ...]
        >>> result = calculator.calculate_cvar(returns)
        >>> print(f"CVaR: {result.cvar*100:.2f}%")
    """
    return CVaRCalculator(
        confidence_level=confidence_level,
        window_size=window_size,
        min_samples=min_samples
    )


def calculate_cvar_for_position_sizing(
    returns: List[float],
    window_size: int = 20,
    confidence_level: float = 0.95
) -> float:
    """
    Convenience function to calculate CVaR for position sizing.
    
    This is the most common use case - getting a CVaR value to constrain
    position sizes based on tail risk.
    
    Args:
        returns: List of historical returns (as decimals)
        window_size: Rolling window size (default: 20 days)
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        float: CVaR as positive decimal for position sizing calculation
              Returns 0.0 if insufficient data
    
    Example:
        >>> returns = [-0.02, 0.01, -0.03, 0.02, -0.01, 0.015, -0.025, ...]
        >>> cvar = calculate_cvar_for_position_sizing(returns)
        >>> max_position = (account_value * 0.1) / cvar  # 10% max loss tolerance
    """
    try:
        calculator = CVaRCalculator(
            confidence_level=confidence_level,
            window_size=window_size,
            min_samples=5
        )
        return calculator.get_cvar_for_position_sizing(returns)
    except Exception as e:
        logger.error(f"Error in convenience CVaR calculation: {e}")
        return 0.0
