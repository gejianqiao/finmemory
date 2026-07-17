"""
Exposure Tracker Module for FinMemory Trading System.

This module implements real-time risk exposure monitoring for the FinMemory position-aware
trading agent system. It tracks portfolio exposure across multiple dimensions including
absolute exposure, relative exposure, concentration risk, and CVaR-based risk metrics.

Key Features:
- Real-time exposure tracking across multiple positions
- Concentration risk measurement (Herfindahl-Herfindahl Index)
- Exposure ratio monitoring vs. CVaR constraints
- Historical exposure tracking for analysis
- Alert generation for exposure threshold breaches

Paper Reference: Section 4.2.2 (CVaR-Based Position Sizing), Appendix B.2
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from finmemory.utils.logger import get_logger
from finmemory.risk.cvar_calculator import CVaRCalculator, calculate_cvar_for_position_sizing

logger = get_logger(__name__)


@dataclass
class PositionExposure:
    """
    Dataclass representing exposure metrics for a single position.
    
    Attributes:
        ticker: Stock ticker symbol
        position_size: Number of shares held
        current_price: Current market price
        market_value: Total market value (position_size × current_price)
        average_cost: Average cost basis per share
        cost_basis: Total cost basis (position_size × average_cost)
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_pct: Unrealized P&L as percentage
        weight: Portfolio weight (market_value / total_portfolio_value)
        cvar_contribution: Position's contribution to portfolio CVaR
        exposure_ratio: Position value / account value
    """
    ticker: str
    position_size: int
    current_price: float
    market_value: float
    average_cost: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    cvar_contribution: float
    exposure_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'position_size': self.position_size,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'average_cost': self.average_cost,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'weight': self.weight,
            'cvar_contribution': self.cvar_contribution,
            'exposure_ratio': self.exposure_ratio
        }


@dataclass
class ExposureSnapshot:
    """
    Dataclass representing a point-in-time exposure snapshot.
    
    Attributes:
        timestamp: Snapshot timestamp
        account_value: Total account value
        total_exposure: Sum of all position market values
        cash: Cash holdings
        exposure_ratio: Total exposure / account value
        cash_ratio: Cash / account value
        num_positions: Number of active positions
        largest_position_weight: Weight of largest position
        concentration_hhi: Herfindahl-Herfindahl Index for concentration
        positions: List of PositionExposure objects
        alerts: List of alert messages
    """
    timestamp: datetime
    account_value: float
    total_exposure: float
    cash: float
    exposure_ratio: float
    cash_ratio: float
    num_positions: int
    largest_position_weight: float
    concentration_hhi: float
    positions: List[PositionExposure]
    alerts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'account_value': self.account_value,
            'total_exposure': self.total_exposure,
            'cash': self.cash,
            'exposure_ratio': self.exposure_ratio,
            'cash_ratio': self.cash_ratio,
            'num_positions': self.num_positions,
            'largest_position_weight': self.largest_position_weight,
            'concentration_hhi': self.concentration_hhi,
            'positions': [p.to_dict() for p in self.positions],
            'alerts': self.alerts
        }


@dataclass
class ExposureAlert:
    """
    Dataclass representing an exposure-related alert.
    
    Attributes:
        timestamp: Alert timestamp
        alert_type: Type of alert (e.g., 'high_exposure', 'concentration', 'cvar_breach')
        severity: Alert severity ('low', 'medium', 'high', 'critical')
        message: Alert message
        ticker: Affected ticker (if applicable)
        current_value: Current metric value
        threshold: Threshold that was breached
    """
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    ticker: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'ticker': self.ticker,
            'current_value': self.current_value,
            'threshold': self.threshold
        }


class ExposureTracker:
    """
    Real-time risk exposure monitoring system for FinMemory.
    
    This class tracks portfolio exposure across multiple dimensions:
    - Absolute exposure (total market value of positions)
    - Relative exposure (exposure / account value)
    - Concentration risk (Herfindahl-Herfindahl Index)
    - CVaR-based risk exposure
    - Per-position exposure limits
    
    The tracker generates alerts when exposure thresholds are breached,
    enabling proactive risk management.
    
    Attributes:
        max_exposure_ratio: Maximum allowed exposure ratio (default 0.9 = 90%)
        max_single_position_ratio: Maximum weight for single position (default 0.3 = 30%)
        concentration_threshold: HHI threshold for concentration alert (default 0.25)
        cvar_confidence_level: CVaR confidence level (default 0.95 = 95%)
        cvar_window_size: Rolling window size for CVaR calculation (default 20 days)
    """
    
    def __init__(
        self,
        max_exposure_ratio: float = 0.9,
        max_single_position_ratio: float = 0.3,
        concentration_threshold: float = 0.25,
        cvar_confidence_level: float = 0.95,
        cvar_window_size: int = 20,
        alert_history_size: int = 100
    ):
        """
        Initialize ExposureTracker with risk parameters.
        
        Args:
            max_exposure_ratio: Maximum allowed total exposure / account value (0.0-1.0)
            max_single_position_ratio: Maximum allowed weight for single position (0.0-1.0)
            concentration_threshold: HHI threshold for concentration alert (0.0-1.0)
            cvar_confidence_level: CVaR confidence level (0.90, 0.95, or 0.99)
            cvar_window_size: Rolling window size for CVaR calculation (days)
            alert_history_size: Maximum number of alerts to retain in history
        """
        self.max_exposure_ratio = max_exposure_ratio
        self.max_single_position_ratio = max_single_position_ratio
        self.concentration_threshold = concentration_threshold
        self.cvar_confidence_level = cvar_confidence_level
        self.cvar_window_size = cvar_window_size
        
        self._alert_history: List[ExposureAlert] = []
        self._alert_history_size = alert_history_size
        self._snapshot_history: List[ExposureSnapshot] = []
        self._cvar_calculator = CVaRCalculator(
            confidence_level=cvar_confidence_level,
            window_size=cvar_window_size
        )
        
        logger.info(
            f"ExposureTracker initialized: max_exposure={max_exposure_ratio}, "
            f"max_single={max_single_position_ratio}, cvar_confidence={cvar_confidence_level}"
        )
    
    def calculate_position_exposure(
        self,
        ticker: str,
        position_size: int,
        current_price: float,
        average_cost: float,
        account_value: float,
        position_returns: Optional[List[float]] = None
    ) -> PositionExposure:
        """
        Calculate exposure metrics for a single position.
        
        Args:
            ticker: Stock ticker symbol
            position_size: Number of shares held
            current_price: Current market price
            average_cost: Average cost basis per share
            account_value: Total account value
            position_returns: Historical returns for this position (for CVaR calculation)
        
        Returns:
            PositionExposure: Calculated exposure metrics
        """
        # Calculate basic metrics
        market_value = position_size * current_price
        cost_basis = position_size * average_cost
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0
        
        # Calculate portfolio weight
        weight = market_value / account_value if account_value > 0 else 0.0
        
        # Calculate exposure ratio (same as weight for long-only)
        exposure_ratio = weight
        
        # Calculate CVaR contribution if returns available
        cvar_contribution = 0.0
        if position_returns is not None and len(position_returns) >= 5:
            cvar = calculate_cvar_for_position_sizing(
                position_returns,
                window_size=self.cvar_window_size,
                confidence_level=self.cvar_confidence_level
            )
            # CVaR contribution = position weight × CVaR
            cvar_contribution = weight * cvar
        
        return PositionExposure(
            ticker=ticker,
            position_size=position_size,
            current_price=current_price,
            market_value=market_value,
            average_cost=average_cost,
            cost_basis=cost_basis,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            weight=weight,
            cvar_contribution=cvar_contribution,
            exposure_ratio=exposure_ratio
        )
    
    def calculate_concentration_hhi(self, weights: List[float]) -> float:
        """
        Calculate Herfindahl-Herfindahl Index (HHI) for portfolio concentration.
        
        HHI = Σ(weight_i²) for all positions
        - HHI = 1.0: Single position (maximum concentration)
        - HHI = 1/n: Equal-weight portfolio with n positions (minimum concentration)
        - HHI > 0.25: High concentration (alert threshold)
        
        Args:
            weights: List of position weights (should sum to <= 1.0)
        
        Returns:
            float: HHI value (0.0 to 1.0)
        """
        if not weights or len(weights) == 0:
            return 0.0
        
        # HHI = sum of squared weights
        hhi = sum(w ** 2 for w in weights if w > 0)
        return hhi
    
    def check_exposure_limits(
        self,
        exposure_ratio: float,
        position_weights: List[float],
        cvar_exposure: Optional[float] = None
    ) -> List[ExposureAlert]:
        """
        Check if exposure metrics breach predefined limits.
        
        Args:
            exposure_ratio: Total exposure / account value
            position_weights: List of individual position weights
            cvar_exposure: Portfolio CVaR exposure (optional)
        
        Returns:
            List[ExposureAlert]: List of triggered alerts
        """
        alerts = []
        timestamp = datetime.now()
        
        # Check total exposure ratio
        if exposure_ratio > self.max_exposure_ratio:
            alert = ExposureAlert(
                timestamp=timestamp,
                alert_type='high_exposure',
                severity='high',
                message=f"Total exposure ratio {exposure_ratio:.2%} exceeds limit {self.max_exposure_ratio:.2%}",
                current_value=exposure_ratio,
                threshold=self.max_exposure_ratio
            )
            alerts.append(alert)
            logger.warning(alert.message)
        
        # Check single position concentration
        for i, weight in enumerate(position_weights):
            if weight > self.max_single_position_ratio:
                alert = ExposureAlert(
                    timestamp=timestamp,
                    alert_type='concentration',
                    severity='medium',
                    message=f"Position weight {weight:.2%} exceeds single-position limit {self.max_single_position_ratio:.2%}",
                    current_value=weight,
                    threshold=self.max_single_position_ratio
                )
                alerts.append(alert)
                logger.warning(alert.message)
        
        # Check portfolio concentration (HHI)
        if position_weights:
            hhi = self.calculate_concentration_hhi(position_weights)
            if hhi > self.concentration_threshold:
                alert = ExposureAlert(
                    timestamp=timestamp,
                    alert_type='portfolio_concentration',
                    severity='medium',
                    message=f"Portfolio concentration HHI {hhi:.4f} exceeds threshold {self.concentration_threshold:.4f}",
                    current_value=hhi,
                    threshold=self.concentration_threshold
                )
                alerts.append(alert)
                logger.warning(alert.message)
        
        return alerts
    
    def take_snapshot(
        self,
        account_value: float,
        cash: float,
        positions: Dict[str, Dict[str, Any]],
        position_returns: Optional[Dict[str, List[float]]] = None
    ) -> ExposureSnapshot:
        """
        Take a complete exposure snapshot of the portfolio.
        
        Args:
            account_value: Total account value
            cash: Cash holdings
            positions: Dict of position data {ticker: {position_size, current_price, average_cost}}
            position_returns: Dict of historical returns {ticker: [returns]}
        
        Returns:
            ExposureSnapshot: Complete exposure snapshot
        """
        timestamp = datetime.now()
        
        # Calculate position exposures
        position_exposures: List[PositionExposure] = []
        total_exposure = 0.0
        weights: List[float] = []
        
        for ticker, pos_data in positions.items():
            position_size = pos_data.get('position_size', 0)
            current_price = pos_data.get('current_price', 0.0)
            average_cost = pos_data.get('average_cost', 0.0)
            
            if position_size > 0 and current_price > 0:
                returns = position_returns.get(ticker, []) if position_returns else None
                pos_exposure = self.calculate_position_exposure(
                    ticker=ticker,
                    position_size=position_size,
                    current_price=current_price,
                    average_cost=average_cost,
                    account_value=account_value,
                    position_returns=returns
                )
                position_exposures.append(pos_exposure)
                total_exposure += pos_exposure.market_value
                weights.append(pos_exposure.weight)
        
        # Calculate portfolio metrics
        exposure_ratio = total_exposure / account_value if account_value > 0 else 0.0
        cash_ratio = cash / account_value if account_value > 0 else 0.0
        num_positions = len(position_exposures)
        largest_position_weight = max(weights) if weights else 0.0
        concentration_hhi = self.calculate_concentration_hhi(weights)
        
        # Check for limit breaches
        alerts = self.check_exposure_limits(
            exposure_ratio=exposure_ratio,
            position_weights=weights
        )
        
        # Store alerts in history
        self._add_alerts(alerts)
        
        # Create snapshot
        snapshot = ExposureSnapshot(
            timestamp=timestamp,
            account_value=account_value,
            total_exposure=total_exposure,
            cash=cash,
            exposure_ratio=exposure_ratio,
            cash_ratio=cash_ratio,
            num_positions=num_positions,
            largest_position_weight=largest_position_weight,
            concentration_hhi=concentration_hhi,
            positions=position_exposures,
            alerts=[a.message for a in alerts]
        )
        
        # Store snapshot in history
        self._snapshot_history.append(snapshot)
        
        logger.debug(
            f"Exposure snapshot: exposure_ratio={exposure_ratio:.2%}, "
            f"positions={num_positions}, hhi={concentration_hhi:.4f}"
        )
        
        return snapshot
    
    def _add_alerts(self, alerts: List[ExposureAlert]) -> None:
        """Add alerts to history with size limit."""
        self._alert_history.extend(alerts)
        
        # Trim history if exceeds limit
        if len(self._alert_history) > self._alert_history_size:
            self._alert_history = self._alert_history[-self._alert_history_size:]
    
    def get_alert_history(
        self,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 50
    ) -> List[ExposureAlert]:
        """
        Retrieve alert history with optional filtering.
        
        Args:
            severity: Filter by severity ('low', 'medium', 'high', 'critical')
            alert_type: Filter by alert type
            limit: Maximum number of alerts to return
        
        Returns:
            List[ExposureAlert]: Filtered alert history
        """
        filtered = self._alert_history
        
        if severity is not None:
            filtered = [a for a in filtered if a.severity == severity]
        
        if alert_type is not None:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        
        return filtered[-limit:]
    
    def get_snapshot_history(self, limit: int = 100) -> List[ExposureSnapshot]:
        """
        Retrieve snapshot history.
        
        Args:
            limit: Maximum number of snapshots to return
        
        Returns:
            List[ExposureSnapshot]: Recent snapshots
        """
        return self._snapshot_history[-limit:]
    
    def calculate_portfolio_cvar(
        self,
        portfolio_returns: List[float],
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calculate portfolio-level CVaR.
        
        Args:
            portfolio_returns: Historical portfolio returns
            confidence_level: Override default confidence level
        
        Returns:
            float: Portfolio CVaR (as positive value representing loss)
        """
        conf_level = confidence_level or self.cvar_confidence_level
        result = self._cvar_calculator.calculate_cvar(
            portfolio_returns,
            use_rolling=False
        )
        return result.cvar
    
    def get_exposure_summary(self, snapshot: Optional[ExposureSnapshot] = None) -> Dict[str, Any]:
        """
        Get human-readable exposure summary.
        
        Args:
            snapshot: Use specific snapshot, or take new one if None
        
        Returns:
            Dict[str, Any]: Exposure summary dictionary
        """
        if snapshot is None and self._snapshot_history:
            snapshot = self._snapshot_history[-1]
        
        if snapshot is None:
            return {'error': 'No snapshot available'}
        
        # Determine exposure status
        if snapshot.exposure_ratio > self.max_exposure_ratio:
            status = 'OVER_EXPOSED'
        elif snapshot.exposure_ratio > self.max_exposure_ratio * 0.8:
            status = 'HIGH_EXPOSURE'
        else:
            status = 'NORMAL'
        
        # Determine concentration status
        if snapshot.concentration_hhi > self.concentration_threshold:
            concentration_status = 'CONCENTRATED'
        else:
            concentration_status = 'DIVERSIFIED'
        
        return {
            'status': status,
            'concentration_status': concentration_status,
            'account_value': snapshot.account_value,
            'total_exposure': snapshot.total_exposure,
            'exposure_ratio': snapshot.exposure_ratio,
            'cash_ratio': snapshot.cash_ratio,
            'num_positions': snapshot.num_positions,
            'largest_position_weight': snapshot.largest_position_weight,
            'concentration_hhi': snapshot.concentration_hhi,
            'active_alerts': len(snapshot.alerts),
            'timestamp': snapshot.timestamp.isoformat()
        }
    
    def reset(self) -> None:
        """Reset tracker state (clear history)."""
        self._alert_history.clear()
        self._snapshot_history.clear()
        logger.info("ExposureTracker reset")


def create_exposure_tracker(
    max_exposure_ratio: float = 0.9,
    max_single_position_ratio: float = 0.3,
    concentration_threshold: float = 0.25,
    cvar_confidence_level: float = 0.95,
    cvar_window_size: int = 20
) -> ExposureTracker:
    """
    Factory function to create configured ExposureTracker instance.
    
    Args:
        max_exposure_ratio: Maximum allowed total exposure ratio (default 0.9)
        max_single_position_ratio: Maximum single position weight (default 0.3)
        concentration_threshold: HHI concentration threshold (default 0.25)
        cvar_confidence_level: CVaR confidence level (default 0.95)
        cvar_window_size: CVaR rolling window size (default 20 days)
    
    Returns:
        ExposureTracker: Configured exposure tracker instance
    """
    return ExposureTracker(
        max_exposure_ratio=max_exposure_ratio,
        max_single_position_ratio=max_single_position_ratio,
        concentration_threshold=concentration_threshold,
        cvar_confidence_level=cvar_confidence_level,
        cvar_window_size=cvar_window_size
    )
