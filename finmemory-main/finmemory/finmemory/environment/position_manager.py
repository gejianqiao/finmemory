"""
Position Manager for FinMemory Trading System.

This module implements continuous position state tracking for the position-aware
trading environment. It manages position sizes, average costs, unrealized P&L,
and exposure calculations.

Paper Reference: Section 3.1 (Task Definition), Section 3.2 (Position-Aware Return)
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionState:
    """
    Represents the current state of a position in a single asset.
    
    Attributes:
        ticker: Stock ticker symbol
        position_size: Number of shares held (>= 0, no short selling)
        average_cost: Average cost basis per share
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss from closed positions
        total_invested: Total capital currently invested
    """
    ticker: str
    position_size: int = 0
    average_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_invested: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update current price and recalculate unrealized P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.average_cost) * self.position_size
        self.total_invested = self.position_size * self.average_cost
    
    def get_market_value(self) -> float:
        """Get current market value of position."""
        return self.position_size * self.current_price
    
    def get_return_pct(self) -> float:
        """Get position return as percentage."""
        if self.total_invested == 0:
            return 0.0
        return (self.unrealized_pnl / self.total_invested) * 100


class PositionManager:
    """
    Manages continuous position state tracking for the trading environment.
    
    Handles position updates, P&L calculations, exposure tracking, and
    enforces no short-selling constraint.
    
    Key Formulas:
    - Position update: pos_t = pos_{t-1} + direction × quantity
    - Unrealized P&L: (current_price - average_cost) × position_size
    - Market value: position_size × current_price
    
    Attributes:
        ticker: Stock ticker symbol being tracked
        positions: Dictionary of position states (supports multiple entries if needed)
        total_realized_pnl: Cumulative realized P&L from all trades
        total_trades: Total number of trades executed
    """
    
    def __init__(self, ticker: str, initial_price: float = 0.0):
        """
        Initialize PositionManager.
        
        Args:
            ticker: Stock ticker symbol
            initial_price: Initial price for the asset
        """
        self.ticker = ticker
        self.position = PositionState(ticker=ticker, current_price=initial_price)
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.trade_history: list[Dict] = []
        
        logger.info(f"PositionManager initialized for {ticker} at price ${initial_price:.2f}")
    
    def reset(self, initial_price: float = 0.0) -> None:
        """
        Reset position manager to initial state.
        
        Args:
            initial_price: New initial price for the asset
        """
        self.position = PositionState(ticker=self.ticker, current_price=initial_price)
        self.total_realized_pnl = 0.0
        self.total_trades = 0
        self.trade_history = []
        logger.info(f"PositionManager reset for {ticker} at price ${initial_price:.2f}")
    
    def update_price(self, price: float) -> None:
        """
        Update current market price.
        
        Args:
            price: Current market price
        """
        old_price = self.position.current_price
        self.position.update_price(price)
        
        if old_price != price:
            logger.debug(f"Price updated for {self.ticker}: ${old_price:.2f} -> ${price:.2f}")
    
    def execute_trade(self, direction: str, quantity: int, price: float) -> Dict[str, any]:
        """
        Execute a trade and update position.
        
        Args:
            direction: Trade direction ('buy', 'sell', 'hold')
            quantity: Number of shares to trade
            price: Execution price
            
        Returns:
            Dictionary with trade details and results
            
        Raises:
            ValueError: If trade violates constraints (short selling, insufficient shares)
        """
        trade_result = {
            'ticker': self.ticker,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'success': False,
            'message': '',
            'position_after': 0,
            'cost_basis_change': 0.0,
            'realized_pnl': 0.0
        }
        
        if direction == 'hold' or quantity == 0:
            trade_result['success'] = True
            trade_result['message'] = 'No trade executed (hold)'
            trade_result['position_after'] = self.position.position_size
            return trade_result
        
        try:
            if direction == 'buy':
                self._execute_buy(quantity, price, trade_result)
            elif direction == 'sell':
                self._execute_sell(quantity, price, trade_result)
            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'buy', 'sell', or 'hold'")
            
            # Update price after trade
            self.position.update_price(price)
            
            # Record trade
            self._record_trade(trade_result)
            
            logger.info(f"Trade executed: {direction.upper()} {quantity} shares of {self.ticker} @ ${price:.2f}")
            
        except ValueError as e:
            trade_result['message'] = str(e)
            logger.warning(f"Trade failed: {e}")
            raise
        
        return trade_result
    
    def _execute_buy(self, quantity: int, price: float, trade_result: Dict) -> None:
        """
        Execute a buy order.
        
        Args:
            quantity: Number of shares to buy
            price: Execution price
            trade_result: Dictionary to update with results
        """
        if quantity < 0:
            raise ValueError(f"Buy quantity must be non-negative, got {quantity}")
        
        cost = quantity * price
        
        # Update average cost using weighted average
        old_shares = self.position.position_size
        old_cost_basis = old_shares * self.position.average_cost
        new_shares = old_shares + quantity
        new_cost_basis = old_cost_basis + cost
        
        if new_shares > 0:
            self.position.average_cost = new_cost_basis / new_shares
        
        self.position.position_size = new_shares
        self.total_trades += 1
        
        trade_result['success'] = True
        trade_result['message'] = f'Bought {quantity} shares @ ${price:.2f}'
        trade_result['position_after'] = self.position.position_size
        trade_result['cost_basis_change'] = cost
    
    def _execute_sell(self, quantity: int, price: float, trade_result: Dict) -> None:
        """
        Execute a sell order.
        
        Args:
            quantity: Number of shares to sell
            price: Execution price
            trade_result: Dictionary to update with results
            
        Raises:
            ValueError: If selling more shares than owned (no short selling)
        """
        if quantity < 0:
            raise ValueError(f"Sell quantity must be non-negative, got {quantity}")
        
        if quantity > self.position.position_size:
            raise ValueError(
                f"Cannot sell {quantity} shares, only {self.position.position_size} available "
                f"(no short selling allowed)"
            )
        
        # Calculate realized P&L
        cost_basis = quantity * self.position.average_cost
        proceeds = quantity * price
        realized_pnl = proceeds - cost_basis
        
        # Update position
        self.position.position_size -= quantity
        self.total_realized_pnl += realized_pnl
        
        # If position closed, reset average cost
        if self.position.position_size == 0:
            self.position.average_cost = 0.0
        
        self.total_trades += 1
        
        trade_result['success'] = True
        trade_result['message'] = f'Sold {quantity} shares @ ${price:.2f}'
        trade_result['position_after'] = self.position.position_size
        trade_result['realized_pnl'] = realized_pnl
    
    def _record_trade(self, trade_result: Dict) -> None:
        """Record trade in history."""
        self.trade_history.append(trade_result.copy())
    
    def get_position_size(self) -> int:
        """Get current position size (number of shares)."""
        return self.position.position_size
    
    def get_average_cost(self) -> float:
        """Get average cost basis per share."""
        return self.position.average_cost
    
    def get_market_value(self) -> float:
        """Get current market value of position."""
        return self.position.get_market_value()
    
    def get_unrealized_pnl(self) -> float:
        """Get unrealized profit/loss."""
        return self.position.unrealized_pnl
    
    def get_realized_pnl(self) -> float:
        """Get cumulative realized profit/loss."""
        return self.total_realized_pnl
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.position.unrealized_pnl
    
    def get_return_pct(self) -> float:
        """Get position return as percentage."""
        return self.position.get_return_pct()
    
    def get_exposure(self) -> float:
        """
        Get current dollar exposure (market value of position).
        
        Returns:
            Current market value of holdings
        """
        return self.get_market_value()
    
    def get_exposure_ratio(self, account_value: float) -> float:
        """
        Get exposure as ratio of account value.
        
        Args:
            account_value: Total account value
            
        Returns:
            Exposure ratio (0.0 to 1.0+)
        """
        if account_value <= 0:
            return 0.0
        return self.get_market_value() / account_value
    
    def get_state(self) -> Dict[str, any]:
        """
        Get current position state as dictionary.
        
        Returns:
            Dictionary with all position state information
        """
        return {
            'ticker': self.ticker,
            'position_size': self.position.position_size,
            'average_cost': self.position.average_cost,
            'current_price': self.position.current_price,
            'market_value': self.get_market_value(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'total_pnl': self.get_total_pnl(),
            'return_pct': self.get_return_pct(),
            'total_trades': self.total_trades
        }
    
    def __str__(self) -> str:
        """String representation of position state."""
        return (
            f"Position({self.ticker}): {self.position.position_size} shares @ "
            f"${self.position.average_cost:.2f} (avg), ${self.position.current_price:.2f} (current), "
            f"Value: ${self.get_market_value():.2f}, P&L: ${self.get_total_pnl():.2f}"
        )
