"""
Evaluation Metrics Module for FinMemory Trading Agent System.

This module implements comprehensive performance evaluation metrics including:
- Cumulative Return (CR%)
- Sharpe Ratio (SR)
- Maximum Drawdown (MDD%)
- Calmar Ratio
- CVaR (95%)
- Additional risk-adjusted metrics

These metrics are used for comparing FinMemory against baselines and conducting
ablation studies as specified in Section 5 of the paper.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from finmemory.utils.logger import get_logger
from finmemory.risk.cvar_calculator import calculate_cvar_for_position_sizing


logger = get_logger(__name__)


@dataclass
class MetricsResult:
    """
    Dataclass storing complete evaluation metrics results.
    
    Attributes:
        ticker: Stock ticker symbol
        period_start: Start date of evaluation period
        period_end: End date of evaluation period
        initial_capital: Starting capital amount
        final_value: Final portfolio value
        cumulative_return: Total return as decimal (e.g., 0.6215 for 62.15%)
        cumulative_return_pct: Total return as percentage (e.g., 62.15)
        sharpe_ratio: Risk-adjusted return metric (annualized)
        max_drawdown: Maximum peak-to-trough decline as decimal
        max_drawdown_pct: Maximum drawdown as percentage
        calmar_ratio: Annualized return / |MDD|
        cvar_95: 95% Conditional Value at Risk (positive decimal)
        var_95: 95% Value at Risk (positive decimal)
        annualized_return: Annualized return as decimal
        volatility: Annualized volatility (standard deviation of returns)
        total_trading_days: Number of trading days in evaluation
        total_trades: Total number of trades executed
        win_rate: Percentage of profitable trades
        avg_win: Average profit on winning trades
        avg_loss: Average loss on losing trades
        profit_factor: Gross profit / Gross loss
        max_consecutive_wins: Maximum consecutive winning trades
        max_consecutive_losses: Maximum consecutive losing trades
        avg_daily_return: Mean daily return
        std_daily_return: Standard deviation of daily returns
        skewness: Skewness of return distribution
        kurtosis: Kurtosis of return distribution
    """
    ticker: str
    period_start: str
    period_end: str
    initial_capital: float
    final_value: float
    cumulative_return: float
    cumulative_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    cvar_95: float
    var_95: float
    annualized_return: float
    volatility: float
    total_trading_days: int
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_daily_return: float
    std_daily_return: float
    skewness: float
    kurtosis: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics result to dictionary."""
        return {
            'ticker': self.ticker,
            'period_start': self.period_start,
            'period_end': self.period_end,
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'cumulative_return': self.cumulative_return,
            'cumulative_return_pct': self.cumulative_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'calmar_ratio': self.calmar_ratio,
            'cvar_95': self.cvar_95,
            'var_95': self.var_95,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'total_trading_days': self.total_trading_days,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'avg_daily_return': self.avg_daily_return,
            'std_daily_return': self.std_daily_return,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }
    
    def summary(self) -> str:
        """Generate human-readable summary of metrics."""
        return (
            f"\n{'='*60}\n"
            f"Performance Metrics Summary: {self.ticker}\n"
            f"Period: {self.period_start} to {self.period_end}\n"
            f"{'='*60}\n"
            f"Returns:\n"
            f"  Cumulative Return:  {self.cumulative_return_pct:>8.2f}%\n"
            f"  Annualized Return:  {self.annualized_return*100:>8.2f}%\n"
            f"  Avg Daily Return:   {self.avg_daily_return*100:>8.4f}%\n"
            f"\nRisk Metrics:\n"
            f"  Sharpe Ratio:       {self.sharpe_ratio:>8.2f}\n"
            f"  Max Drawdown:       {self.max_drawdown_pct:>8.2f}%\n"
            f"  Calmar Ratio:       {self.calmar_ratio:>8.2f}\n"
            f"  Volatility (Ann.):  {self.volatility*100:>8.2f}%\n"
            f"  CVaR (95%):         {self.cvar_95*100:>8.2f}%\n"
            f"\nTrading Statistics:\n"
            f"  Total Trades:       {self.total_trades:>8d}\n"
            f"  Win Rate:           {self.win_rate*100:>8.2f}%\n"
            f"  Profit Factor:      {self.profit_factor:>8.2f}\n"
            f"  Avg Win:            {self.avg_win*100:>8.2f}%\n"
            f"  Avg Loss:           {self.avg_loss*100:>8.2f}%\n"
            f"{'='*60}\n"
        )


class MetricsCalculator:
    """
    Comprehensive metrics calculator for evaluating trading strategy performance.
    
    Implements all evaluation metrics specified in Section 5 and Appendix B
    of the FinMemory paper, including return metrics, risk metrics, and
    trading statistics.
    
    Args:
        returns: Array of daily returns (as decimals, e.g., 0.01 for 1%)
        account_values: Array of daily account values
        trades: List of trade dictionaries with 'pnl' or 'pnl_pct' keys
        risk_free_rate: Annual risk-free rate (default: 0.02 for 2%)
        trading_days_per_year: Trading days per year (default: 252)
    
    Example:
        >>> calculator = MetricsCalculator(returns, account_values, trades)
        >>> metrics = calculator.calculate_all_metrics(
        ...     ticker="TSLA",
        ...     period_start="2025-03-01",
        ...     period_end="2025-09-30",
        ...     initial_capital=100000
        ... )
        >>> print(metrics.summary())
    """
    
    def __init__(
        self,
        returns: Union[List[float], np.ndarray],
        account_values: Union[List[float], np.ndarray],
        trades: Optional[List[Dict[str, Any]]] = None,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252
    ):
        """
        Initialize metrics calculator with return series and account values.
        
        Args:
            returns: Daily returns as decimals (e.g., [0.01, -0.005, 0.02])
            account_values: Daily account values (e.g., [100000, 101000, 100500])
            trades: Optional list of trade records with P&L information
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days_per_year: Trading days per year (default 252)
        """
        self.returns = np.array(returns, dtype=np.float64)
        self.account_values = np.array(account_values, dtype=np.float64)
        self.trades = trades if trades is not None else []
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        
        # Validate inputs
        if len(self.returns) == 0:
            raise ValueError("Returns array cannot be empty")
        if len(self.account_values) == 0:
            raise ValueError("Account values array cannot be empty")
        if len(self.returns) != len(self.account_values) - 1:
            logger.warning(
                f"Returns length ({len(self.returns)}) doesn't match "
                f"account_values length ({len(self.account_values)}) - 1"
            )
        
        # Pre-calculate daily risk-free rate
        self.daily_rf_rate = risk_free_rate / trading_days_per_year
        
        logger.debug(
            f"Initialized MetricsCalculator with {len(self.returns)} returns, "
            f"{len(self.account_values)} account values, {len(self.trades)} trades"
        )
    
    def calculate_cumulative_return(self) -> float:
        """
        Calculate cumulative return over the evaluation period.
        
        Formula: CR = (V_final - V_initial) / V_initial
        
        Returns:
            Cumulative return as decimal (e.g., 0.6215 for 62.15%)
        """
        if len(self.account_values) < 2:
            return 0.0
        
        initial_value = self.account_values[0]
        final_value = self.account_values[-1]
        
        if initial_value <= 0:
            logger.warning("Initial account value is zero or negative")
            return 0.0
        
        cumulative_return = (final_value - initial_value) / initial_value
        
        logger.debug(f"Cumulative return calculated: {cumulative_return:.4f}")
        return cumulative_return
    
    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe Ratio (annualized).
        
        Formula: SR = (R_p - R_f) / σ_p × √252
        where R_p = mean daily return, R_f = daily risk-free rate, σ_p = std dev
        
        Returns:
            Annualized Sharpe Ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate mean daily return
        mean_return = np.mean(self.returns)
        
        # Calculate standard deviation of returns
        std_return = np.std(self.returns, ddof=1)  # Sample std dev
        
        if std_return == 0 or np.isnan(std_return):
            logger.warning("Standard deviation of returns is zero or NaN")
            return 0.0
        
        # Calculate excess return over risk-free rate
        excess_return = mean_return - self.daily_rf_rate
        
        # Annualize Sharpe Ratio
        sharpe = (excess_return / std_return) * np.sqrt(self.trading_days_per_year)
        
        logger.debug(f"Sharpe ratio calculated: {sharpe:.4f}")
        return sharpe
    
    def calculate_max_drawdown(self) -> float:
        """
        Calculate Maximum Drawdown (peak-to-trough decline).
        
        Formula: MDD = max((peak_value - trough_value) / peak_value)
        
        Returns:
            Maximum drawdown as decimal (e.g., 0.4234 for 42.34%)
        """
        if len(self.account_values) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(self.account_values)
        
        # Calculate drawdown at each point
        drawdowns = (running_max - self.account_values) / running_max
        
        # Handle division by zero
        drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Find maximum drawdown
        max_drawdown = np.max(drawdowns)
        
        logger.debug(f"Maximum drawdown calculated: {max_drawdown:.4f}")
        return max_drawdown
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio (annualized return / |MDD|).
        
        Formula: Calmar = Annualized Return / |Maximum Drawdown|
        
        Returns:
            Calmar Ratio
        """
        # Calculate annualized return
        annualized_return = self.calculate_annualized_return()
        
        # Calculate maximum drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        if max_drawdown == 0 or np.isnan(max_drawdown):
            logger.warning("Maximum drawdown is zero or NaN, cannot calculate Calmar ratio")
            return 0.0
        
        calmar = annualized_return / abs(max_drawdown)
        
        logger.debug(f"Calmar ratio calculated: {calmar:.4f}")
        return calmar
    
    def calculate_annualized_return(self) -> float:
        """
        Calculate annualized return.
        
        Formula: Annualized = (V_final / V_initial)^(252/trading_days) - 1
        
        Returns:
            Annualized return as decimal
        """
        if len(self.account_values) < 2:
            return 0.0
        
        initial_value = self.account_values[0]
        final_value = self.account_values[-1]
        trading_days = len(self.account_values) - 1
        
        if initial_value <= 0 or trading_days <= 0:
            return 0.0
        
        # Calculate annualized return
        annualized = (final_value / initial_value) ** (self.trading_days_per_year / trading_days) - 1
        
        logger.debug(f"Annualized return calculated: {annualized:.4f}")
        return annualized
    
    def calculate_volatility(self) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).
        
        Formula: Volatility = std(returns) × √252
        
        Returns:
            Annualized volatility as decimal
        """
        if len(self.returns) < 2:
            return 0.0
        
        # Calculate daily standard deviation
        daily_std = np.std(self.returns, ddof=1)
        
        # Annualize
        annualized_vol = daily_std * np.sqrt(self.trading_days_per_year)
        
        logger.debug(f"Annualized volatility calculated: {annualized_vol:.4f}")
        return annualized_vol
    
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Formula: CVaR_α = E[returns | returns ≤ VaR_α]
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95%)
        
        Returns:
            CVaR as positive decimal (loss magnitude)
        """
        if len(self.returns) < 5:
            logger.warning("Insufficient data for CVaR calculation")
            return 0.0
        
        # Use the CVaR calculator from risk module
        cvar = calculate_cvar_for_position_sizing(
            returns=self.returns,
            window_size=len(self.returns),  # Use all available data
            confidence_level=confidence_level
        )
        
        logger.debug(f"CVaR ({confidence_level*100:.0f}%) calculated: {cvar:.4f}")
        return cvar
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Formula: VaR_α = percentile(returns, (1-α)×100)
        
        Args:
            confidence_level: Confidence level (default 0.95 for 95%)
        
        Returns:
            VaR as positive decimal (loss magnitude)
        """
        if len(self.returns) < 5:
            logger.warning("Insufficient data for VaR calculation")
            return 0.0
        
        # Calculate VaR as percentile
        percentile = (1 - confidence_level) * 100
        var = np.percentile(self.returns, percentile)
        
        # Return as positive value (loss magnitude)
        var_positive = abs(var) if var < 0 else 0.0
        
        logger.debug(f"VaR ({confidence_level*100:.0f}%) calculated: {var_positive:.4f}")
        return var_positive
    
    def calculate_trading_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive trading statistics from trade history.
        
        Returns:
            Dictionary containing:
            - total_trades: Total number of trades
            - win_rate: Percentage of profitable trades
            - avg_win: Average profit on winning trades
            - avg_loss: Average loss on losing trades
            - profit_factor: Gross profit / Gross loss
            - max_consecutive_wins: Maximum consecutive winning trades
            - max_consecutive_losses: Maximum consecutive losing trades
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        # Extract P&L from trades
        pnls = []
        for trade in self.trades:
            if 'pnl_pct' in trade:
                pnls.append(trade['pnl_pct'])
            elif 'pnl' in trade:
                # Convert absolute P&L to percentage if possible
                if 'entry_value' in trade and trade['entry_value'] > 0:
                    pnls.append(trade['pnl'] / trade['entry_value'])
                else:
                    pnls.append(0.0)
            else:
                pnls.append(0.0)
        
        pnls = np.array(pnls)
        
        # Calculate statistics
        total_trades = len(pnls)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
        
        # Profit factor: gross profit / gross loss
        gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(pnls > 0)
        max_consecutive_losses = self._calculate_max_consecutive(pnls < 0)
        
        stats = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor if not np.isinf(profit_factor) else 0.0,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
        
        logger.debug(f"Trading statistics calculated: {total_trades} trades, {win_rate*100:.1f}% win rate")
        return stats
    
    def _calculate_max_consecutive(self, boolean_array: np.ndarray) -> int:
        """
        Calculate maximum consecutive True values in boolean array.
        
        Args:
            boolean_array: Array of boolean values
        
        Returns:
            Maximum consecutive True count
        """
        if len(boolean_array) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in boolean_array:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def calculate_return_statistics(self) -> Dict[str, float]:
        """
        Calculate statistical properties of return distribution.
        
        Returns:
            Dictionary containing:
            - avg_daily_return: Mean daily return
            - std_daily_return: Standard deviation of daily returns
            - skewness: Skewness of return distribution
            - kurtosis: Kurtosis of return distribution
        """
        if len(self.returns) < 2:
            return {
                'avg_daily_return': 0.0,
                'std_daily_return': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        avg_daily = np.mean(self.returns)
        std_daily = np.std(self.returns, ddof=1)
        
        # Calculate skewness
        if std_daily > 0:
            skewness = np.mean(((self.returns - avg_daily) / std_daily) ** 3)
        else:
            skewness = 0.0
        
        # Calculate kurtosis (excess kurtosis)
        if std_daily > 0:
            kurtosis = np.mean(((self.returns - avg_daily) / std_daily) ** 4) - 3
        else:
            kurtosis = 0.0
        
        stats = {
            'avg_daily_return': avg_daily,
            'std_daily_return': std_daily,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        logger.debug(f"Return statistics: avg={avg_daily:.6f}, std={std_daily:.6f}, skew={skewness:.4f}, kurt={kurtosis:.4f}")
        return stats
    
    def calculate_all_metrics(
        self,
        ticker: str,
        period_start: str,
        period_end: str,
        initial_capital: float
    ) -> MetricsResult:
        """
        Calculate all evaluation metrics and return comprehensive result.
        
        Args:
            ticker: Stock ticker symbol
            period_start: Start date of evaluation period (YYYY-MM-DD)
            period_end: End date of evaluation period (YYYY-MM-DD)
            initial_capital: Starting capital amount
        
        Returns:
            MetricsResult dataclass with all calculated metrics
        """
        logger.info(f"Calculating all metrics for {ticker} from {period_start} to {period_end}")
        
        # Calculate return metrics
        cumulative_return = self.calculate_cumulative_return()
        annualized_return = self.calculate_annualized_return()
        
        # Calculate risk metrics
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = self.calculate_calmar_ratio()
        volatility = self.calculate_volatility()
        cvar_95 = self.calculate_cvar(0.95)
        var_95 = self.calculate_var(0.95)
        
        # Calculate trading statistics
        trading_stats = self.calculate_trading_statistics()
        
        # Calculate return distribution statistics
        return_stats = self.calculate_return_statistics()
        
        # Get final account value
        final_value = self.account_values[-1] if len(self.account_values) > 0 else initial_capital
        
        # Create result object
        result = MetricsResult(
            ticker=ticker,
            period_start=period_start,
            period_end=period_end,
            initial_capital=initial_capital,
            final_value=final_value,
            cumulative_return=cumulative_return,
            cumulative_return_pct=cumulative_return * 100,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown * 100,
            calmar_ratio=calmar_ratio,
            cvar_95=cvar_95,
            var_95=var_95,
            annualized_return=annualized_return,
            volatility=volatility,
            total_trading_days=len(self.account_values) - 1,
            total_trades=trading_stats['total_trades'],
            win_rate=trading_stats['win_rate'],
            avg_win=trading_stats['avg_win'],
            avg_loss=trading_stats['avg_loss'],
            profit_factor=trading_stats['profit_factor'],
            max_consecutive_wins=trading_stats['max_consecutive_wins'],
            max_consecutive_losses=trading_stats['max_consecutive_losses'],
            avg_daily_return=return_stats['avg_daily_return'],
            std_daily_return=return_stats['std_daily_return'],
            skewness=return_stats['skewness'],
            kurtosis=return_stats['kurtosis']
        )
        
        logger.info(
            f"Metrics calculation complete for {ticker}: "
            f"CR={result.cumulative_return_pct:.2f}%, "
            f"SR={result.sharpe_ratio:.2f}, "
            f"MDD={result.max_drawdown_pct:.2f}%"
        )
        
        return result


def calculate_metrics(
    returns: Union[List[float], np.ndarray],
    account_values: Union[List[float], np.ndarray],
    trades: Optional[List[Dict[str, Any]]] = None,
    ticker: str = "UNKNOWN",
    period_start: str = "",
    period_end: str = "",
    initial_capital: float = 100000.0,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252
) -> MetricsResult:
    """
    Convenience function to calculate all metrics in one call.
    
    Args:
        returns: Daily returns as decimals
        account_values: Daily account values
        trades: Optional list of trade records
        ticker: Stock ticker symbol
        period_start: Start date (YYYY-MM-DD)
        period_end: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        trading_days_per_year: Trading days per year
    
    Returns:
        MetricsResult with all calculated metrics
    
    Example:
        >>> metrics = calculate_metrics(
        ...     returns=daily_returns,
        ...     account_values=account_values,
        ...     trades=trade_history,
        ...     ticker="TSLA",
        ...     period_start="2025-03-01",
        ...     period_end="2025-09-30",
        ...     initial_capital=100000
        ... )
        >>> print(metrics.summary())
    """
    calculator = MetricsCalculator(
        returns=returns,
        account_values=account_values,
        trades=trades,
        risk_free_rate=risk_free_rate,
        trading_days_per_year=trading_days_per_year
    )
    
    return calculator.calculate_all_metrics(
        ticker=ticker,
        period_start=period_start,
        period_end=period_end,
        initial_capital=initial_capital
    )


def compare_strategies(
    metrics_list: List[MetricsResult],
    primary_metric: str = "cumulative_return_pct"
) -> pd.DataFrame:
    """
    Compare multiple strategies and rank by primary metric.
    
    Args:
        metrics_list: List of MetricsResult objects from different strategies
        primary_metric: Metric to use for ranking (default: cumulative_return_pct)
    
    Returns:
        DataFrame with strategies ranked by primary metric
    
    Example:
        >>> comparison = compare_strategies(
        ...     [finmemory_metrics, finmem_metrics, a2c_metrics],
        ...     primary_metric="sharpe_ratio"
        ... )
        >>> print(comparison)
    """
    if not metrics_list:
        return pd.DataFrame()
    
    # Convert to list of dicts
    metrics_dicts = [m.to_dict() for m in metrics_list]
    
    # Create DataFrame
    df = pd.DataFrame(metrics_dicts)
    
    # Sort by primary metric (descending)
    if primary_metric in df.columns:
        df = df.sort_values(by=primary_metric, ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
    else:
        logger.warning(f"Primary metric '{primary_metric}' not found in metrics")
        df['rank'] = range(1, len(df) + 1)
    
    # Select key columns for display
    display_cols = [
        'rank', 'ticker', 'cumulative_return_pct', 'sharpe_ratio',
        'max_drawdown_pct', 'calmar_ratio', 'total_trades', 'win_rate'
    ]
    
    available_cols = [col for col in display_cols if col in df.columns]
    
    return df[available_cols]


def generate_table1_results(
    results_dict: Dict[str, Dict[str, MetricsResult]]
) -> pd.DataFrame:
    """
    Generate Table 1 results format from the FinMemory paper.
    
    Args:
        results_dict: Nested dict with structure:
            {method_name: {ticker: MetricsResult}}
            Example: {"FinMemory": {"TSLA": metrics, "AAPL": metrics}, ...}
    
    Returns:
        DataFrame formatted like Table 1 in the paper
    
    Example:
        >>> table1 = generate_table1_results({
        ...     "FinMemory": {"TSLA": tsla_metrics, "AAPL": aapl_metrics},
        ...     "FINMEM": {"TSLA": tsla_finmem, "AAPL": aapl_finmem}
        ... })
    """
    rows = []
    
    for method, tickers in results_dict.items():
        for ticker, metrics in tickers.items():
            row = {
                'Method': method,
                'Ticker': ticker,
                'CR%': metrics.cumulative_return_pct,
                'SR': metrics.sharpe_ratio,
                'MDD%': metrics.max_drawdown_pct,
                'Calmar': metrics.calmar_ratio,
                'CVaR%': metrics.cvar_95 * 100,
                'Annualized Return%': metrics.annualized_return * 100,
                'Volatility%': metrics.volatility * 100,
                'Total Trades': metrics.total_trades,
                'Win Rate%': metrics.win_rate * 100
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by method and ticker
    if not df.empty:
        df = df.sort_values(['Method', 'Ticker']).reset_index(drop=True)
    
    return df


def generate_ablation_table(
    full_metrics: MetricsResult,
    ablation_metrics: Dict[str, MetricsResult]
) -> pd.DataFrame:
    """
    Generate ablation study table (Table 2 format).
    
    Args:
        full_metrics: Metrics from full FinMemory model
        ablation_metrics: Dict of ablated model metrics
            Example: {"w/o MTR": metrics, "w/o QRA": metrics, "w/o MSP": metrics}
    
    Returns:
        DataFrame showing performance degradation from ablations
    """
    rows = []
    
    # Add full model
    rows.append({
        'Configuration': 'Full FinMemory',
        'CR%': full_metrics.cumulative_return_pct,
        'SR': full_metrics.sharpe_ratio,
        'MDD%': full_metrics.max_drawdown_pct,
        'Calmar': full_metrics.calmar_ratio,
        'Degradation%': 0.0
    })
    
    # Add ablated versions
    for config, metrics in ablation_metrics.items():
        degradation = (
            (full_metrics.cumulative_return_pct - metrics.cumulative_return_pct)
            / full_metrics.cumulative_return_pct * 100
            if full_metrics.cumulative_return_pct != 0 else 0.0
        )
        
        rows.append({
            'Configuration': config,
            'CR%': metrics.cumulative_return_pct,
            'SR': metrics.sharpe_ratio,
            'MDD%': metrics.max_drawdown_pct,
            'Calmar': metrics.calmar_ratio,
            'Degradation%': degradation
        })
    
    df = pd.DataFrame(rows)
    
    # Sort to put full model first
    df = df.sort_values('Degradation%').reset_index(drop=True)
    
    return df


# Export public API
__all__ = [
    'MetricsResult',
    'MetricsCalculator',
    'calculate_metrics',
    'compare_strategies',
    'generate_table1_results',
    'generate_ablation_table'
]
