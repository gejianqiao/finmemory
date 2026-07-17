"""
Position Sizer Module for FinMemory Trading System.

This module implements CVaR-constrained position sizing logic that determines
optimal position sizes based on tail risk constraints, available capital, and
trading direction. It enforces risk discipline by limiting position sizes to
prevent catastrophic losses during extreme market events.

Key Features:
- CVaR-based maximum position calculation
- Cash constraint enforcement
- Direction-aware sizing (buy/sell/hold)
- Integer share rounding (conservative floor rounding)
- Real-time exposure monitoring integration

References:
- Paper Section 4.2.2: CVaR-Based Position Sizing
- Appendix B.2: Risk Management Details
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from finmemory.utils.logger import get_logger
from finmemory.risk.cvar_calculator import CVaRCalculator, calculate_cvar_for_position_sizing

logger = get_logger(__name__)


@dataclass
class PositionSizeResult:
    """
    Dataclass storing position sizing calculation results.
    
    Attributes:
        requested_size: Original requested position size from agent
        constrained_size: Final size after applying all constraints
        max_cvar_size: Maximum size allowed by CVaR constraint
        max_cash_size: Maximum size allowed by available cash
        effective_max: The binding constraint (min of CVaR and cash)
        constraint_type: Which constraint was binding ('cvar', 'cash', or 'none')
        utilization_ratio: How much of available capacity was used (0-1)
        risk_exposure: Dollar value of position at current price
        notes: Any warnings or notes about the sizing decision
    """
    requested_size: int
    constrained_size: int
    max_cvar_size: int
    max_cash_size: int
    effective_max: int
    constraint_type: str
    utilization_ratio: float
    risk_exposure: float
    notes: str


class PositionSizer:
    """
    Position sizing engine with CVaR constraints for the FinMemory trading system.
    
    This class determines optimal position sizes by considering:
    1. CVaR-based risk constraints (tail risk limits)
    2. Cash availability constraints (can't buy more than affordable)
    3. Direction constraints (can't sell more than held)
    4. Integer share requirements (round down for safety)
    
    The position sizer acts as an independent risk check on the Quantity/Risk
    Agent's decisions, ensuring that even overconfident LLM outputs don't
    violate risk management principles.
    
    Attributes:
        max_exposure_ratio: Maximum fraction of account value at risk (default 0.1 = 10%)
        cvar_calculator: CVaR calculator instance for tail risk measurement
        confidence_level: CVaR confidence level (default 0.95 = 95%)
        window_size: Rolling window size for CVaR calculation (default 20 days)
    
    Example:
        >>> sizer = PositionSizer(max_exposure_ratio=0.1)
        >>> result = sizer.calculate_position_size(
        ...     direction='buy',
        ...     requested_size=100,
        ...     account_value=100000,
        ...     current_price=250.0,
        ...     returns_history=[-0.02, 0.01, 0.03, ...],  # 20+ days
        ...     current_position=50
        ... )
        >>> print(f"Constrained size: {result.constrained_size}")
    """
    
    def __init__(
        self,
        max_exposure_ratio: float = 0.1,
        confidence_level: float = 0.95,
        window_size: int = 20,
        min_samples: int = 5
    ):
        """
        Initialize PositionSizer with risk parameters.
        
        Args:
            max_exposure_ratio: Maximum fraction of account value allowed at risk.
                               Default 0.1 means max 10% loss tolerance.
            confidence_level: CVaR confidence level (0.95 = 95% confidence).
            window_size: Rolling window size for CVaR calculation (days).
            min_samples: Minimum samples required for CVaR calculation.
        
        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        # Validate parameters
        if not 0.0 < max_exposure_ratio <= 1.0:
            raise ValueError(f"max_exposure_ratio must be in (0, 1], got {max_exposure_ratio}")
        if not 0.5 <= confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in [0.5, 1), got {confidence_level}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        
        self.max_exposure_ratio = max_exposure_ratio
        self.confidence_level = confidence_level
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Initialize CVaR calculator
        self.cvar_calculator = CVaRCalculator(
            confidence_level=confidence_level,
            window_size=window_size,
            min_samples=min_samples
        )
        
        logger.info(
            f"PositionSizer initialized: max_exposure={max_exposure_ratio}, "
            f"CVaR confidence={confidence_level}, window={window_size}d"
        )
    
    def calculate_max_cvar_size(
        self,
        account_value: float,
        returns_history: list
    ) -> int:
        """
        Calculate maximum position size based on CVaR constraint.
        
        Formula:
            CVaR = Conditional Value at Risk (95% confidence, 20-day window)
            Max Loss Tolerance = account_value × max_exposure_ratio
            Max Position Size = Max Loss Tolerance / |CVaR|
        
        This ensures that even in worst-case scenarios (5% tail), the position
        won't cause losses exceeding the maximum exposure ratio.
        
        Args:
            account_value: Total account value in dollars.
            returns_history: List of historical returns (at least min_samples).
        
        Returns:
            Maximum position size in shares (integer, rounded down).
            Returns 0 if CVaR cannot be calculated or is zero.
        
        Example:
            >>> sizer = PositionSizer(max_exposure_ratio=0.1)
            >>> returns = [-0.02, -0.01, 0.01, 0.02, ...]  # 20 days
            >>> max_size = sizer.calculate_max_cvar_size(100000, returns)
            >>> # If CVaR = 0.025 (2.5% daily tail loss):
            >>> # Max Loss = 100000 × 0.1 = 10000
            >>> # Max Position = 10000 / 0.025 = 400,000 dollars
            >>> # If price = $250, max shares = 400000 / 250 = 1600 shares
        """
        if account_value <= 0:
            logger.warning(f"Invalid account value: {account_value}")
            return 0
        
        if len(returns_history) < self.min_samples:
            logger.warning(
                f"Insufficient returns history: {len(returns_history)} < {self.min_samples}"
            )
            return 0
        
        # Calculate CVaR
        cvar_result = self.cvar_calculator.calculate_cvar(
            returns_history,
            use_rolling=False  # Use full history provided
        )
        
        cvar = cvar_result.cvar
        
        # Handle edge case: zero or negative CVaR
        if cvar <= 0:
            logger.warning(f"CVaR is non-positive ({cvar}), using conservative estimate")
            # Use a small default CVaR to allow some positioning
            cvar = 0.01  # 1% conservative estimate
        
        # Calculate maximum dollar exposure
        max_loss_tolerance = account_value * self.max_exposure_ratio
        
        # Calculate maximum position value
        max_position_value = max_loss_tolerance / cvar
        
        # Convert to shares (will be divided by price in calculate_position_size)
        # Return as dollar value for now, actual share count calculated later
        logger.debug(
            f"CVaR constraint: CVaR={cvar:.4f}, max_loss=${max_loss_tolerance:.2f}, "
            f"max_position_value=${max_position_value:.2f}"
        )
        
        # Return max position value (will be converted to shares by caller)
        # For API consistency, return as int representing max shares at $1 price
        return int(max_position_value)
    
    def calculate_max_cash_size(
        self,
        account_value: float,
        current_price: float,
        current_position: int,
        direction: str
    ) -> int:
        """
        Calculate maximum position size based on available cash.
        
        For buy orders: Limited by available cash / current_price
        For sell orders: Limited by current position (can't sell more than held)
        For hold: Returns 0 (no action)
        
        Args:
            account_value: Total account value in dollars.
            current_price: Current stock price per share.
            current_position: Current number of shares held.
            direction: Trading direction ('buy', 'sell', 'hold').
        
        Returns:
            Maximum position size in shares based on cash constraint.
            Returns 0 for 'hold' direction.
        
        Raises:
            ValueError: If direction is invalid.
        """
        if current_price <= 0:
            logger.warning(f"Invalid current price: {current_price}")
            return 0
        
        direction = direction.lower()
        
        if direction == 'hold':
            return 0
        elif direction == 'buy':
            # For buy, we need to know current holdings to calculate available cash
            # This is a simplification - full calculation needs holdings value
            # Assume account_value includes both cash and holdings
            # Available cash = account_value - (current_position × current_price)
            holdings_value = current_position * current_price
            available_cash = max(0, account_value - holdings_value)
            
            max_shares = int(available_cash / current_price)
            logger.debug(
                f"Cash constraint (buy): available=${available_cash:.2f}, "
                f"price=${current_price:.2f}, max_shares={max_shares}"
            )
            return max_shares
        elif direction == 'sell':
            # For sell, maximum is current position (can't sell more than held)
            logger.debug(f"Cash constraint (sell): max_shares={current_position}")
            return current_position
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'buy', 'sell', or 'hold'")
    
    def calculate_position_size(
        self,
        direction: str,
        requested_size: int,
        account_value: float,
        current_price: float,
        returns_history: list,
        current_position: int = 0,
        current_holdings_value: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate constrained position size based on CVaR and cash constraints.
        
        This is the main method for position sizing. It applies multiple constraints:
        1. CVaR constraint: Position value ≤ (account_value × max_exposure_ratio) / CVaR
        2. Cash constraint: Can't buy more than available cash allows
        3. Direction constraint: Can't sell more than currently held
        4. Non-negativity: Position size must be >= 0
        5. Integer constraint: Round down to nearest integer share
        
        Args:
            direction: Trading direction ('buy', 'sell', 'hold').
            requested_size: Original requested position size from agent.
            account_value: Total account value in dollars.
            current_price: Current stock price per share.
            returns_history: Historical returns for CVaR calculation.
            current_position: Current number of shares held.
            current_holdings_value: Current holdings value in dollars (optional).
                                   If None, calculated from current_position × price.
        
        Returns:
            PositionSizeResult with constrained size and constraint diagnostics.
        
        Example:
            >>> sizer = PositionSizer()
            >>> result = sizer.calculate_position_size(
            ...     direction='buy',
            ...     requested_size=500,
            ...     account_value=100000,
            ...     current_price=250.0,
            ...     returns_history=[-0.02, 0.01, ...],  # 20 days
            ...     current_position=100
            ... )
            >>> print(f"Requested: {result.requested_size}")
            >>> print(f"Constrained: {result.constrained_size}")
            >>> print(f"Constraint type: {result.constraint_type}")
        """
        notes = []
        
        # Validate inputs
        if requested_size < 0:
            logger.warning(f"Negative requested size: {requested_size}, setting to 0")
            requested_size = 0
            notes.append("Requested size was negative, set to 0")
        
        if account_value <= 0:
            logger.error(f"Invalid account value: {account_value}")
            return PositionSizeResult(
                requested_size=requested_size,
                constrained_size=0,
                max_cvar_size=0,
                max_cash_size=0,
                effective_max=0,
                constraint_type='invalid_account',
                utilization_ratio=0.0,
                risk_exposure=0.0,
                notes="Invalid account value"
            )
        
        if current_price <= 0:
            logger.error(f"Invalid current price: {current_price}")
            return PositionSizeResult(
                requested_size=requested_size,
                constrained_size=0,
                max_cvar_size=0,
                max_cash_size=0,
                effective_max=0,
                constraint_type='invalid_price',
                utilization_ratio=0.0,
                risk_exposure=0.0,
                notes="Invalid current price"
            )
        
        direction = direction.lower()
        
        # Handle 'hold' direction
        if direction == 'hold':
            return PositionSizeResult(
                requested_size=0,
                constrained_size=0,
                max_cvar_size=0,
                max_cash_size=0,
                effective_max=0,
                constraint_type='hold',
                utilization_ratio=0.0,
                risk_exposure=current_position * current_price,
                notes="Hold direction - no position change"
            )
        
        # Calculate CVaR-based maximum position value
        max_cvar_value = self.calculate_max_cvar_size(account_value, returns_history)
        
        # Convert CVaR max value to shares
        if max_cvar_value > 0 and current_price > 0:
            max_cvar_shares = int(max_cvar_value / current_price)
        else:
            max_cvar_shares = 0
        
        # Calculate cash-based maximum position size
        max_cash_shares = self.calculate_max_cash_size(
            account_value, current_price, current_position, direction
        )
        
        # Determine effective maximum based on direction
        if direction == 'buy':
            # For buy, take minimum of CVaR and cash constraints
            effective_max = min(max_cvar_shares, max_cash_shares)
            
            # Track which constraint is binding
            if max_cvar_shares < max_cash_shares:
                constraint_type = 'cvar'
                notes.append(f"CVaR constraint binding: {max_cvar_shares} < {max_cash_shares} cash")
            elif max_cash_shares < max_cvar_shares:
                constraint_type = 'cash'
                notes.append(f"Cash constraint binding: {max_cash_shares} < {max_cvar_shares} CVaR")
            else:
                constraint_type = 'both'
                notes.append("CVaR and cash constraints equal")
        
        elif direction == 'sell':
            # For sell, maximum is current position (can't sell more than held)
            # CVaR constraint doesn't apply to selling (reduces risk)
            effective_max = min(max_cash_shares, requested_size)
            constraint_type = 'position'
            notes.append(f"Sell constrained by current position: {current_position}")
        
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Apply effective maximum to requested size
        constrained_size = min(requested_size, effective_max)
        
        # Ensure non-negative
        constrained_size = max(0, constrained_size)
        
        # Calculate utilization ratio
        if effective_max > 0:
            utilization_ratio = constrained_size / effective_max
        else:
            utilization_ratio = 0.0
        
        # Calculate risk exposure
        risk_exposure = constrained_size * current_price
        
        # Log results
        logger.info(
            f"Position sizing: direction={direction}, requested={requested_size}, "
            f"constrained={constrained_size}, constraint={constraint_type}, "
            f"utilization={utilization_ratio:.2%}"
        )
        
        return PositionSizeResult(
            requested_size=requested_size,
            constrained_size=constrained_size,
            max_cvar_size=max_cvar_shares,
            max_cash_size=max_cash_shares,
            effective_max=effective_max,
            constraint_type=constraint_type,
            utilization_ratio=utilization_ratio,
            risk_exposure=risk_exposure,
            notes="; ".join(notes) if notes else "No constraints violated"
        )
    
    def validate_position_size(
        self,
        direction: str,
        proposed_size: int,
        account_value: float,
        current_price: float,
        returns_history: list,
        current_position: int
    ) -> Dict[str, Any]:
        """
        Validate a proposed position size against all constraints.
        
        This method checks if a proposed position size violates any constraints
        without modifying it. Useful for pre-validation before trade execution.
        
        Args:
            direction: Trading direction ('buy', 'sell', 'hold').
            proposed_size: Proposed position size to validate.
            account_value: Total account value in dollars.
            current_price: Current stock price per share.
            returns_history: Historical returns for CVaR calculation.
            current_position: Current number of shares held.
        
        Returns:
            Dictionary with validation results:
            - 'valid': bool - Whether proposed size is valid
            - 'violations': list - List of constraint violations
            - 'max_allowed': int - Maximum allowed position size
            - 'recommendation': str - Recommended action
        """
        violations = []
        
        # Calculate constrained size
        result = self.calculate_position_size(
            direction=direction,
            requested_size=proposed_size,
            account_value=account_value,
            current_price=current_price,
            returns_history=returns_history,
            current_position=current_position
        )
        
        # Check for violations
        if proposed_size > result.constrained_size:
            violations.append(
                f"Proposed size ({proposed_size}) exceeds constrained size "
                f"({result.constrained_size})"
            )
        
        if proposed_size < 0:
            violations.append(f"Proposed size is negative: {proposed_size}")
        
        if direction == 'sell' and proposed_size > current_position:
            violations.append(
                f"Cannot sell {proposed_size} shares, only hold {current_position}"
            )
        
        # Determine validity
        valid = len(violations) == 0
        
        # Generate recommendation
        if valid:
            recommendation = "Proposed size is valid and within all constraints"
        else:
            recommendation = f"Reduce size to {result.constrained_size} to meet constraints"
        
        return {
            'valid': valid,
            'violations': violations,
            'max_allowed': result.effective_max,
            'recommended_size': result.constrained_size,
            'constraint_type': result.constraint_type,
            'recommendation': recommendation
        }
    
    def get_cvar_diagnostics(self, returns_history: list) -> Dict[str, Any]:
        """
        Get detailed CVaR diagnostics for monitoring and debugging.
        
        Args:
            returns_history: Historical returns for analysis.
        
        Returns:
            Dictionary with CVaR diagnostics:
            - 'cvar': CVaR value (positive decimal)
            - 'var': VaR value (positive decimal)
            - 'confidence_level': Confidence level used
            - 'window_size': Window size used
            - 'tail_returns': Returns in the tail (worst cases)
            - 'tail_count': Number of returns in tail
            - 'max_loss': Maximum single-day loss in history
            - 'mean_return': Average return in history
        """
        if len(returns_history) < self.min_samples:
            return {
                'error': f'Insufficient data: {len(returns_history)} < {self.min_samples}',
                'cvar': None,
                'var': None
            }
        
        cvar_result = self.cvar_calculator.calculate_cvar(returns_history, use_rolling=False)
        
        import numpy as np
        returns_array = np.array(returns_history)
        
        return {
            'cvar': cvar_result.cvar,
            'var': cvar_result.var,
            'confidence_level': self.confidence_level,
            'window_size': self.window_size,
            'tail_returns': cvar_result.tail_returns,
            'tail_count': cvar_result.tail_count,
            'max_loss': float(np.min(returns_array)),
            'mean_return': float(np.mean(returns_array)),
            'std_return': float(np.std(returns_array))
        }


def create_position_sizer(
    max_exposure_ratio: float = 0.1,
    confidence_level: float = 0.95,
    window_size: int = 20
) -> PositionSizer:
    """
    Factory function to create a configured PositionSizer instance.
    
    Args:
        max_exposure_ratio: Maximum fraction of account value at risk (default 0.1).
        confidence_level: CVaR confidence level (default 0.95).
        window_size: Rolling window size for CVaR calculation (default 20).
    
    Returns:
        PositionSizer: Configured position sizer instance.
    
    Example:
        >>> sizer = create_position_sizer(max_exposure_ratio=0.15, confidence_level=0.99)
        >>> # More conservative: 15% max exposure, 99% confidence
    """
    return PositionSizer(
        max_exposure_ratio=max_exposure_ratio,
        confidence_level=confidence_level,
        window_size=window_size
    )
