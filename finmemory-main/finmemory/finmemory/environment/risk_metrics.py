"""
Risk Metrics Calculator for FinMemory Trading System.

This module implements all evaluation metrics specified in the FinMemory paper:
- Cumulative Return (CR%)
- Sharpe Ratio (SR)
- Maximum Drawdown (MDD%)
- Calmar Ratio
- CVaR (95% Conditional Value at Risk)

These metrics are used for both evaluation (Component 7) and risk management (Component 3).
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetricsResult:
    """Dataclass storing complete risk metrics calculation results."""
    
    cumulative_return: float  # CR% as decimal (e.g., 0.6215 for 62.15%)
    cumulative_return_pct: float  # CR% as percentage (e.g., 62.15)
    sharpe_ratio: float  # Annualized Sharpe Ratio
    max_drawdown: float  # MDD as decimal (e.g., 0.4234 for 42.34%)
    max_drawdown_pct: float  # MDD as percentage (e.g., 42.34)
    calmar_ratio: float  # Calmar Ratio
    cvar_95: float  # 95% CVaR as decimal (positive value)
    cvar_95_pct: float  # 95% CVaR as percentage
    var_95: float  # 95% VaR as decimal
    annualized_return: float  # Annualized return as decimal
    volatility: float  # Annualized volatility (std dev)
    total_trading_days: int  # Number of trading days in period
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            'cumulative_return': self.cumulative_return,
            'cumulative_return_pct': self.cumulative_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'calmar_ratio': self.calmar_ratio,
            'cvar_95': self.cvar_95,
            'cvar_95_pct': self.cvar_95_pct,
            'var_95': self.var_95,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'total_trading_days': self.total_trading_days
        }
    
    def summary(self) -> str:
        """Return human-readable summary string."""
        return (
            f"CR: {self.cumulative_return_pct:.2f}%, "
            f"SR: {self.sharpe_ratio:.2f}, "
            f"MDD: {self.max_drawdown_pct:.2f}%, "
            f"Calmar: {self.calmar_ratio:.2f}, "
            f"CVaR(95%): {self.cvar_95_pct:.2f}%"
        )


class RiskMetricsCalculator:
    """
    Calculator for comprehensive risk and performance metrics.
    
    Implements all evaluation metrics from Section 5 and Appendix B of the FinMemory paper.
    Used for both strategy evaluation and risk management constraints.
    
    Attributes:
        returns: Array of daily returns (as decimals, e.g., 0.01 for 1%)
        account_values: Array of account values over time
        risk_free_rate: Annual risk-free rate (default 2%)
        trading_days_per_year: Number of trading days per year (default 252)
    """
    
    def __init__(
        self,
        returns: Optional[np.ndarray] = None,
        account_values: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        trading_days_per_year: int = 252
    ):
        """
        Initialize risk metrics calculator.
        
        Args:
            returns: Array of daily returns (as decimals). Can be None if account_values provided.
            account_values: Array of account values over time. Can be None if returns provided.
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days_per_year: Number of trading days per year (default 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.daily_rf_rate = risk_free_rate / trading_days_per_year
        
        # Process input data
        if returns is not None:
            self.returns = np.array(returns, dtype=np.float64)
            if account_values is None:
                # Reconstruct account values from returns (assuming starting value of 1.0)
                self.account_values = np.cumprod(1 + self.returns)
            else:
                self.account_values = np.array(account_values, dtype=np.float64)
        elif account_values is not None:
            self.account_values = np.array(account_values, dtype=np.float64)
            # Calculate returns from account values
            self.returns = np.diff(self.account_values) / self.account_values[:-1]
        else:
            raise ValueError("Either returns or account_values must be provided")
        
        # Validate data
        if len(self.returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        logger.debug(f"Initialized RiskMetricsCalculator with {len(self.returns)} daily returns")
    
    def calculate_cumulative_return(self) -> float:
        """
        Calculate Cumulative Return (CR%).
        
        Formula: CR = (V_final - V_initial) / V_initial
        
        Returns:
            Cumulative return as decimal (e.g., 0.6215 for 62.15%)
        """
        if len(self.account_values) == 0:
            return 0.0
        
        initial_value = self.account_values[0]
        final_value = self.account_values[-1]
        
        if initial_value == 0:
            logger.warning("Initial account value is zero, returning 0 for CR")
            return 0.0
        
        cr = (final_value - initial_value) / initial_value
        logger.debug(f"Cumulative Return: {cr:.4f} ({cr*100:.2f}%)")
        return cr
    
    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio (SR).
        
        Formula: SR = (R_p - R_f) / σ_p × √252
        where:
            R_p = mean daily return
            R_f = risk-free rate (2%/252 daily)
            σ_p = standard deviation of daily returns
        
        Returns:
            Annualized Sharpe Ratio
        """
        if len(self.returns) < 2:
            logger.warning("Insufficient data for Sharpe Ratio calculation (< 2 returns)")
            return 0.0
        
        # Calculate mean daily return
        mean_return = np.mean(self.returns)
        
        # Calculate standard deviation of daily returns
        std_return = np.std(self.returns, ddof=1)  # Sample std dev
        
        if std_return == 0:
            logger.warning("Zero standard deviation, returning 0 for Sharpe Ratio")
            return 0.0
        
        # Calculate daily Sharpe ratio and annualize
        daily_sharpe = (mean_return - self.daily_rf_rate) / std_return
        annualized_sharpe = daily_sharpe * np.sqrt(self.trading_days_per_year)
        
        logger.debug(f"Sharpe Ratio: {annualized_sharpe:.4f} (mean_return={mean_return:.6f}, std={std_return:.6f})")
        return annualized_sharpe
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate Maximum Drawdown (MDD%).
        
        Formula: MDD = max((peak_value - trough_value) / peak_value)
        
        Returns:
            Maximum drawdown as decimal (e.g., 0.4234 for 42.34%)
        """
        if len(self.account_values) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(self.account_values)
        
        # Calculate drawdown at each point
        drawdowns = (running_max - self.account_values) / running_max
        
        # Handle division by zero (if running_max contains zeros)
        drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get maximum drawdown
        mdd = np.max(drawdowns)
        
        logger.debug(f"Maximum Drawdown: {mdd:.4f} ({mdd*100:.2f}%)")
        return mdd
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.
        
        Formula: Calmar = Annualized Return / |MDD|
        where Annualized Return = (V_final / V_initial)^(252/trading_days) - 1
        
        Returns:
            Calmar Ratio (higher is better)
        """
        if len(self.account_values) < 2:
            logger.warning("Insufficient data for Calmar Ratio calculation")
            return 0.0
        
        # Calculate annualized return
        initial_value = self.account_values[0]
        final_value = self.account_values[-1]
        trading_days = len(self.account_values)
        
        if initial_value == 0 or trading_days == 0:
            logger.warning("Invalid initial value or trading days for Calmar Ratio")
            return 0.0
        
        annualized_return = (final_value / initial_value) ** (self.trading_days_per_year / trading_days) - 1
        
        # Calculate MDD
        mdd = self.calculate_max_drawdown()
        
        if mdd == 0:
            logger.warning("Zero MDD, returning 0 for Calmar Ratio")
            return 0.0
        
        calmar = annualized_return / abs(mdd)
        
        logger.debug(f"Calmar Ratio: {calmar:.4f} (ann_return={annualized_return:.4f}, mdd={mdd:.4f})")
        return calmar
    
    def calculate_cvar(self, confidence_level: float = 0.95, window: Optional[int] = None) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) at specified confidence level.
        
        Formula (95% confidence):
            Step 1: Collect returns (optionally within rolling window)
            Step 2: Calculate VaR_α = percentile(returns, (1-α)×100)
            Step 3: Calculate CVaR_α = E[returns | returns ≤ VaR_α]
        
        Args:
            confidence_level: Confidence level α (default 0.95 for 95%)
            window: Rolling window size (default None for full history)
        
        Returns:
            CVaR as positive decimal value (e.g., 0.05 for 5% tail loss)
        """
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
        
        # Use rolling window if specified
        if window is not None and window < len(self.returns):
            returns_subset = self.returns[-window:]
        else:
            returns_subset = self.returns
        
        if len(returns_subset) == 0:
            logger.warning("No returns available for CVaR calculation")
            return 0.0
        
        # Calculate VaR (Value at Risk)
        # For 95% confidence, we want the 5th percentile
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_subset, var_percentile)
        
        # Calculate CVaR (average of returns in the tail)
        tail_returns = returns_subset[returns_subset <= var]
        
        if len(tail_returns) == 0:
            # Edge case: no returns in tail, use VaR as CVaR
            logger.debug("No returns in tail, using VaR as CVaR")
            cvar = var
        else:
            cvar = np.mean(tail_returns)
        
        # Return as positive value (CVaR represents loss magnitude)
        cvar_positive = abs(cvar)
        
        logger.debug(f"CVaR({confidence_level*100:.0f}%): {cvar_positive:.4f} ({cvar_positive*100:.2f}%), VaR: {var:.4f}")
        return cvar_positive
    
    def calculate_var(self, confidence_level: float = 0.95, window: Optional[int] = None) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.
        
        Args:
            confidence_level: Confidence level α (default 0.95 for 95%)
            window: Rolling window size (default None for full history)
        
        Returns:
            VaR as positive decimal value
        """
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
        
        # Use rolling window if specified
        if window is not None and window < len(self.returns):
            returns_subset = self.returns[-window:]
        else:
            returns_subset = self.returns
        
        if len(returns_subset) == 0:
            logger.warning("No returns available for VaR calculation")
            return 0.0
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_subset, var_percentile)
        
        # Return as positive value
        var_positive = abs(var)
        
        logger.debug(f"VaR({confidence_level*100:.0f}%): {var_positive:.4f} ({var_positive*100:.2f}%)")
        return var_positive
    
    def calculate_annualized_return(self) -> float:
        """
        Calculate annualized return.
        
        Formula: Annualized Return = (V_final / V_initial)^(252/trading_days) - 1
        
        Returns:
            Annualized return as decimal
        """
        if len(self.account_values) < 2:
            return 0.0
        
        initial_value = self.account_values[0]
        final_value = self.account_values[-1]
        trading_days = len(self.account_values)
        
        if initial_value == 0 or trading_days == 0:
            return 0.0
        
        annualized = (final_value / initial_value) ** (self.trading_days_per_year / trading_days) - 1
        return annualized
    
    def calculate_volatility(self) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).
        
        Returns:
            Annualized volatility as decimal
        """
        if len(self.returns) < 2:
            return 0.0
        
        daily_volatility = np.std(self.returns, ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(self.trading_days_per_year)
        
        return annualized_volatility
    
    def calculate_all_metrics(self) -> RiskMetricsResult:
        """
        Calculate all risk metrics and return as RiskMetricsResult.
        
        Returns:
            RiskMetricsResult dataclass with all metrics
        """
        logger.info("Calculating all risk metrics")
        
        cr = self.calculate_cumulative_return()
        sr = self.calculate_sharpe_ratio()
        mdd = self.calculate_max_drawdown()
        calmar = self.calculate_calmar_ratio()
        cvar = self.calculate_cvar(confidence_level=0.95)
        var = self.calculate_var(confidence_level=0.95)
        ann_return = self.calculate_annualized_return()
        vol = self.calculate_volatility()
        
        result = RiskMetricsResult(
            cumulative_return=cr,
            cumulative_return_pct=cr * 100,
            sharpe_ratio=sr,
            max_drawdown=mdd,
            max_drawdown_pct=mdd * 100,
            calmar_ratio=calmar,
            cvar_95=cvar,
            cvar_95_pct=cvar * 100,
            var_95=var,
            var_95_pct=var * 100,
            annualized_return=ann_return,
            annualized_return_pct=ann_return * 100,
            volatility=vol,
            volatility_pct=vol * 100,
            total_trading_days=len(self.returns)
        )
        
        logger.info(f"Risk metrics calculated: {result.summary()}")
        return result


def calculate_risk_metrics(
    returns: Optional[np.ndarray] = None,
    account_values: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
) -> RiskMetricsResult:
    """
    Convenience function to calculate all risk metrics.
    
    Args:
        returns: Array of daily returns (as decimals)
        account_values: Array of account values over time
        risk_free_rate: Annual risk-free rate (default 2%)
        trading_days_per_year: Number of trading days per year (default 252)
    
    Returns:
        RiskMetricsResult with all calculated metrics
    """
    calculator = RiskMetricsCalculator(
        returns=returns,
        account_values=account_values,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year
    )
    return calculator.calculate_all_metrics()


def calculate_cvar_for_position_sizing(
    returns: np.ndarray,
    window: int = 20,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate CVaR specifically for position sizing constraints.
    
    This function implements the CVaR calculation from Section 4.2.2 and Appendix B.2,
    using a rolling window of past returns to constrain position sizes.
    
    Args:
        returns: Array of historical daily returns
        window: Rolling window size (default 20 days as per paper)
        confidence_level: Confidence level (default 0.95 for 95%)
    
    Returns:
        CVaR as positive decimal value for position sizing calculation
    """
    if len(returns) == 0:
        logger.warning("No returns provided for CVaR position sizing")
        return 0.01  # Default conservative CVaR
    
    # Use most recent window of returns
    if window < len(returns):
        recent_returns = returns[-window:]
    else:
        recent_returns = returns
    
    calculator = RiskMetricsCalculator(returns=recent_returns)
    cvar = calculator.calculate_cvar(confidence_level=confidence_level, window=None)
    
    # Ensure minimum CVaR to avoid division by zero in position sizing
    if cvar < 0.001:
        cvar = 0.001
        logger.debug(f"CVaR too small, using minimum value: {cvar}")
    
    return cvar


def calculate_max_position_size(
    account_value: float,
    cvar: float,
    max_exposure_ratio: float = 0.1,
    current_price: float = 1.0
) -> int:
    """
    Calculate maximum position size based on CVaR constraint.
    
    Formula: Max position size = (account_value × max_exposure_ratio) / CVaR
    
    This implements the position sizing constraint from Section 4.2.2.
    
    Args:
        account_value: Current account value in dollars
        cvar: CVaR as decimal (e.g., 0.05 for 5%)
        max_exposure_ratio: Maximum loss tolerance ratio (default 0.1 for 10%)
        current_price: Current asset price per share
    
    Returns:
        Maximum position size as integer (rounded down for safety)
    """
    if account_value <= 0:
        logger.warning(f"Invalid account value: {account_value}")
        return 0
    
    if cvar <= 0:
        logger.warning(f"Invalid CVaR: {cvar}, using default")
        cvar = 0.01  # Default 1% CVaR
    
    # Calculate maximum dollar exposure
    max_dollar_exposure = account_value * max_exposure_ratio
    
    # Calculate maximum position size in shares
    # Position size × price × CVaR <= max_dollar_exposure
    # Position size <= max_dollar_exposure / (price × CVaR)
    max_position = max_dollar_exposure / (current_price * cvar)
    
    # Round down for safety (conservative approach)
    max_position_int = int(np.floor(max_position))
    
    logger.debug(
        f"Max position size: {max_position_int} shares "
        f"(account=${account_value:.2f}, CVaR={cvar:.4f}, price=${current_price:.2f})"
    )
    
    return max_position_int
