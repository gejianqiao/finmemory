"""
Trading Environment for FinMemory Position-Aware Trading Agent System.

This module implements the core trading environment that simulates real trading
with continuous position management across multi-step episodes. It provides
the interface for agents to make trading decisions and tracks portfolio performance.

Key Features:
- Position-aware return calculation (Eq. 2): R = Σ(pos_t × log(price[t+1] / price[t]))
- Daily timesteps with action execution
- Cash and holdings tracking
- Integration with PositionManager for position state management
- Support for multi-timescale reward calculation
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np

from finmemory.environment.position_manager import PositionManager
from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingState:
    """
    Represents the complete state of the trading environment at a given timestep.
    
    Attributes:
        current_price: Current asset price
        position_size: Current number of shares held
        account_value: Total account value (cash + holdings)
        cash: Available cash balance
        holdings_value: Current market value of holdings
        unrealized_pnl: Unrealized profit/loss on current position
        realized_pnl: Realized profit/loss from closed positions
        return_pct: Percentage return on invested capital
        exposure: Dollar exposure to the asset
        exposure_ratio: Exposure as fraction of account value
        timestep: Current timestep index
        date: Current trading date (if available)
    """
    current_price: float
    position_size: int
    account_value: float
    cash: float
    holdings_value: float
    unrealized_pnl: float
    realized_pnl: float
    return_pct: float
    exposure: float
    exposure_ratio: float
    timestep: int
    date: Optional[str] = None


@dataclass
class TradeResult:
    """
    Represents the result of executing a trade action.
    
    Attributes:
        success: Whether the trade was executed successfully
        direction: Trade direction (buy/sell/hold)
        quantity: Number of shares traded
        price: Execution price
        cost: Total cost of trade (for buys) or proceeds (for sells)
        position_after: Position size after trade
        cash_after: Cash balance after trade
        message: Description of trade result or error
    """
    success: bool
    direction: str
    quantity: int
    price: float
    cost: float
    position_after: int
    cash_after: float
    message: str


@dataclass
class EpisodeResult:
    """
    Represents the complete result of a trading episode.
    
    Attributes:
        initial_capital: Starting account value
        final_value: Ending account value
        cumulative_return: Total return percentage
        total_trades: Number of trades executed
        buy_trades: Number of buy orders
        sell_trades: Number of sell orders
        hold_actions: Number of hold actions
        trades: List of all trade results
        daily_returns: List of daily position-aware returns
        account_values: List of daily account values
        positions: List of daily position sizes
        prices: List of daily prices
    """
    initial_capital: float
    final_value: float
    cumulative_return: float
    total_trades: int
    buy_trades: int
    sell_trades: int
    hold_actions: int
    trades: List[TradeResult]
    daily_returns: List[float]
    account_values: List[float]
    positions: List[int]
    prices: List[float]


class TradingEnvironment:
    """
    Position-aware trading environment for the FinMemory system.
    
    This environment simulates real trading with continuous position management,
    implementing the position-aware return formula from the paper (Eq. 2):
        R_t = position_t × log(price[t+1] / price[t])
    
    The environment supports:
    - Daily timesteps from historical price data
    - Buy/sell/hold actions with quantity specification
    - No short-selling constraint
    - Cash constraint enforcement
    - Position-aware return calculation
    - Complete episode tracking for evaluation
    
    Attributes:
        ticker: Asset ticker symbol
        initial_capital: Starting account value
        prices: Array of historical prices
        dates: Array of trading dates (optional)
        current_step: Current timestep index
        position_manager: Manages position state and trades
        account_value: Current total account value
        cash: Current cash balance
        episode_trades: List of trades in current episode
        episode_returns: List of daily returns in current episode
        episode_account_values: List of account values in current episode
        episode_positions: List of positions in current episode
        episode_prices: List of prices in current episode
    """
    
    def __init__(
        self,
        ticker: str,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1% per trade
    ):
        """
        Initialize the trading environment.
        
        Args:
            ticker: Asset ticker symbol (e.g., 'TSLA', 'AAPL')
            prices: Array of historical prices (length = number of timesteps)
            dates: Optional array of trading dates (same length as prices)
            initial_capital: Starting account value in dollars (default: $100,000)
            transaction_cost: Transaction cost as fraction of trade value (default: 0.1%)
        """
        self.ticker = ticker
        self.prices = np.array(prices, dtype=np.float64)
        self.dates = dates
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Validate inputs
        if len(self.prices) == 0:
            raise ValueError("Prices array cannot be empty")
        if np.any(self.prices <= 0):
            raise ValueError("Prices must be positive")
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        # State variables
        self.current_step = 0
        self.position_manager: Optional[PositionManager] = None
        self.account_value = initial_capital
        self.cash = initial_capital
        
        # Episode tracking
        self.episode_trades: List[TradeResult] = []
        self.episode_returns: List[float] = []
        self.episode_account_values: List[float] = []
        self.episode_positions: List[int] = []
        self.episode_prices: List[float] = []
        
        # Action counters
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        
        logger.info(f"TradingEnvironment initialized for {ticker} with {len(prices)} timesteps")
        logger.info(f"Initial capital: ${initial_capital:,.2f}, Transaction cost: {transaction_cost*100:.2f}%")
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to initial state for a new episode.
        
        Returns:
            Initial state dictionary containing:
            - current_price: First price in the series
            - position_size: 0 (no position)
            - account_value: initial_capital
            - cash: initial_capital
            - timestep: 0
            - date: First date (if available)
        """
        self.current_step = 0
        self.account_value = self.initial_capital
        self.cash = self.initial_capital
        
        # Initialize position manager with first price
        initial_price = float(self.prices[0])
        self.position_manager = PositionManager(
            ticker=self.ticker,
            initial_price=initial_price
        )
        
        # Reset episode tracking
        self.episode_trades = []
        self.episode_returns = []
        self.episode_account_values = [self.initial_capital]
        self.episode_positions = [0]
        self.episode_prices = [initial_price]
        
        # Reset action counters
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        
        # Get initial date
        initial_date = None
        if self.dates is not None and len(self.dates) > 0:
            initial_date = str(self.dates[0])
        
        logger.debug(f"Environment reset for {self.ticker} at step 0, price: ${initial_price:.2f}")
        
        return self.get_state()
    
    def step(
        self,
        action: Dict[str, Any],
        next_price: Optional[float] = None
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one timestep in the environment.
        
        This method:
        1. Executes the trade action (buy/sell/hold with quantity)
        2. Updates position and account state
        3. Calculates position-aware return for the timestep
        4. Returns new state, reward, done flag, and info
        
        Args:
            action: Dictionary containing:
                - direction: 'buy', 'sell', or 'hold'
                - quantity: Number of shares to trade (integer >= 0)
            next_price: Price for next timestep (for return calculation).
                       If None, uses prices[self.current_step + 1] if available.
        
        Returns:
            Tuple of (state, reward, done, info):
            - state: Current environment state (dict)
            - reward: Position-aware return for this timestep (float)
            - done: Whether episode is complete (bool)
            - info: Additional information (dict) containing:
                - trade_result: TradeResult object
                - daily_return: Position-aware log return
                - account_value: Current account value
                - position_size: Current position size
        """
        # Validate action format
        if not isinstance(action, dict):
            raise ValueError(f"Action must be a dictionary, got {type(action)}")
        if 'direction' not in action:
            raise ValueError("Action must contain 'direction' key")
        if 'quantity' not in action:
            raise ValueError("Action must contain 'quantity' key")
        
        direction = action['direction'].lower()
        quantity = int(action['quantity'])
        
        if direction not in ['buy', 'sell', 'hold']:
            raise ValueError(f"Invalid direction: {direction}. Must be 'buy', 'sell', or 'hold'")
        if quantity < 0:
            raise ValueError(f"Quantity must be non-negative, got {quantity}")
        
        # Get current price
        current_price = float(self.prices[self.current_step])
        
        # Execute trade
        trade_result = self._execute_trade(direction, quantity, current_price)
        
        # Update action counters
        if direction == 'buy':
            self.buy_count += 1
        elif direction == 'sell':
            self.sell_count += 1
        else:  # hold
            self.hold_count += 1
        
        # Store trade result
        self.episode_trades.append(trade_result)
        
        # Move to next timestep
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.prices)
        
        # Calculate position-aware return
        if not done:
            # Get next price for return calculation
            if next_price is None:
                next_price = float(self.prices[self.current_step])
            
            # Calculate position-aware return (Eq. 2)
            position_size = self.position_manager.get_position_size()
            if position_size > 0 and current_price > 0:
                log_return = np.log(next_price / current_price)
                daily_return = position_size * log_return
            else:
                daily_return = 0.0
        else:
            daily_return = 0.0
        
        # Store episode metrics
        self.episode_returns.append(daily_return)
        self.episode_account_values.append(self.account_value)
        self.episode_positions.append(self.position_manager.get_position_size())
        self.episode_prices.append(float(self.prices[self.current_step]) if not done else self.episode_prices[-1])
        
        # Get current state
        state = self.get_state()
        
        # Prepare info dictionary
        info = {
            'trade_result': trade_result,
            'daily_return': daily_return,
            'account_value': self.account_value,
            'position_size': self.position_manager.get_position_size(),
            'timestep': self.current_step,
            'done': done,
        }
        
        # Use daily return as reward (can be modified for multi-timescale rewards)
        reward = daily_return
        
        logger.debug(
            f"Step {self.current_step}: {direction} {quantity} shares @ ${current_price:.2f}, "
            f"return: {daily_return:.6f}, account: ${self.account_value:,.2f}"
        )
        
        return state, reward, done, info
    
    def _execute_trade(
        self,
        direction: str,
        quantity: int,
        price: float
    ) -> TradeResult:
        """
        Execute a trade action through the position manager.
        
        Args:
            direction: Trade direction ('buy', 'sell', 'hold')
            quantity: Number of shares to trade
            price: Execution price
        
        Returns:
            TradeResult object with execution details
        """
        # Handle hold action
        if direction == 'hold' or quantity == 0:
            return TradeResult(
                success=True,
                direction='hold',
                quantity=0,
                price=price,
                cost=0.0,
                position_after=self.position_manager.get_position_size(),
                cash_after=self.cash,
                message="Hold action - no trade executed"
            )
        
        # Execute trade through position manager
        trade_info = self.position_manager.execute_trade(direction, quantity, price)
        
        if not trade_info.get('success', False):
            return TradeResult(
                success=False,
                direction=direction,
                quantity=quantity,
                price=price,
                cost=0.0,
                position_after=self.position_manager.get_position_size(),
                cash_after=self.cash,
                message=trade_info.get('error', 'Trade execution failed')
            )
        
        # Calculate trade cost/proceeds including transaction costs
        trade_value = quantity * price
        transaction_fee = trade_value * self.transaction_cost
        
        if direction == 'buy':
            total_cost = trade_value + transaction_fee
            # Validate cash constraint
            if total_cost > self.cash:
                # Adjust quantity to what we can afford
                max_quantity = int(self.cash / (price * (1 + self.transaction_cost)))
                if max_quantity < quantity:
                    logger.warning(
                        f"Cash constraint: requested {quantity}, can afford {max_quantity}"
                    )
                    # Re-execute with adjusted quantity
                    return self._execute_trade(direction, max_quantity, price)
            
            self.cash -= total_cost
            cost = total_cost
            message = f"Bought {quantity} shares @ ${price:.2f} (fee: ${transaction_fee:.2f})"
        
        elif direction == 'sell':
            proceeds = trade_value - transaction_fee
            self.cash += proceeds
            cost = -proceeds  # Negative cost = proceeds
            message = f"Sold {quantity} shares @ ${price:.2f} (fee: ${transaction_fee:.2f})"
        
        else:
            return TradeResult(
                success=False,
                direction=direction,
                quantity=quantity,
                price=price,
                cost=0.0,
                position_after=self.position_manager.get_position_size(),
                cash_after=self.cash,
                message=f"Invalid direction: {direction}"
            )
        
        # Update account value
        self.account_value = self.cash + self.position_manager.get_market_value()
        
        return TradeResult(
            success=True,
            direction=direction,
            quantity=quantity,
            price=price,
            cost=cost,
            position_after=self.position_manager.get_position_size(),
            cash_after=self.cash,
            message=message
        )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current environment state for agent observation.
        
        Returns:
            Dictionary containing:
            - ticker: Asset ticker symbol
            - current_price: Current asset price
            - position_size: Current number of shares held
            - account_value: Total account value
            - cash: Available cash balance
            - holdings_value: Market value of holdings
            - unrealized_pnl: Unrealized profit/loss
            - realized_pnl: Realized profit/loss
            - return_pct: Percentage return
            - exposure: Dollar exposure
            - exposure_ratio: Exposure as fraction of account value
            - timestep: Current timestep index
            - date: Current date (if available)
            - price_history: Last 20 prices (for technical analysis)
            - position_history: Last 20 positions (for pattern recognition)
        """
        if self.position_manager is None:
            # Environment not initialized
            return {
                'ticker': self.ticker,
                'current_price': float(self.prices[0]) if len(self.prices) > 0 else 0.0,
                'position_size': 0,
                'account_value': self.initial_capital,
                'cash': self.initial_capital,
                'holdings_value': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'return_pct': 0.0,
                'exposure': 0.0,
                'exposure_ratio': 0.0,
                'timestep': 0,
                'date': str(self.dates[0]) if self.dates is not None and len(self.dates) > 0 else None,
                'price_history': [],
                'position_history': [],
            }
        
        current_price = float(self.prices[self.current_step])
        
        # Get position state
        position_size = self.position_manager.get_position_size()
        holdings_value = self.position_manager.get_market_value()
        unrealized_pnl = self.position_manager.get_unrealized_pnl()
        realized_pnl = self.position_manager.get_realized_pnl()
        exposure = self.position_manager.get_exposure()
        exposure_ratio = self.position_manager.get_exposure_ratio(self.account_value)
        
        # Calculate return percentage
        return_pct = ((self.account_value - self.initial_capital) / self.initial_capital) * 100
        
        # Get date
        current_date = None
        if self.dates is not None and self.current_step < len(self.dates):
            current_date = str(self.dates[self.current_step])
        
        # Get price history (last 20 timesteps)
        start_idx = max(0, self.current_step - 19)
        price_history = [float(p) for p in self.prices[start_idx:self.current_step + 1]]
        
        # Get position history (last 20 timesteps)
        position_history = self.episode_positions[-20:] if self.episode_positions else [0]
        
        state = {
            'ticker': self.ticker,
            'current_price': current_price,
            'position_size': position_size,
            'account_value': self.account_value,
            'cash': self.cash,
            'holdings_value': holdings_value,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'return_pct': return_pct,
            'exposure': exposure,
            'exposure_ratio': exposure_ratio,
            'timestep': self.current_step,
            'date': current_date,
            'price_history': price_history,
            'position_history': position_history,
        }
        
        return state
    
    def calculate_return(self) -> float:
        """
        Calculate cumulative position-aware return for the episode.
        
        Implements Eq. 2 from the paper:
            R = Σ(pos_t × log(price[t+1] / price[t]))
        
        Returns:
            Cumulative position-aware log return
        """
        if len(self.episode_returns) == 0:
            return 0.0
        
        return sum(self.episode_returns)
    
    def get_episode_result(self) -> EpisodeResult:
        """
        Get complete episode results for evaluation.
        
        Returns:
            EpisodeResult object containing all episode metrics
        """
        cumulative_return = self.calculate_return()
        final_value = self.account_value
        
        return EpisodeResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            cumulative_return=((final_value - self.initial_capital) / self.initial_capital) * 100,
            total_trades=len(self.episode_trades),
            buy_trades=self.buy_count,
            sell_trades=self.sell_count,
            hold_actions=self.hold_count,
            trades=self.episode_trades,
            daily_returns=self.episode_returns.copy(),
            account_values=self.episode_account_values.copy(),
            positions=self.episode_positions.copy(),
            prices=self.episode_prices.copy(),
        )
    
    def get_account_value(self) -> float:
        """Get current total account value."""
        return self.account_value
    
    def get_cash(self) -> float:
        """Get current cash balance."""
        return self.cash
    
    def get_position_size(self) -> int:
        """Get current position size (number of shares)."""
        if self.position_manager is None:
            return 0
        return self.position_manager.get_position_size()
    
    def get_current_price(self) -> float:
        """Get current asset price."""
        return float(self.prices[self.current_step])
    
    def get_timestep(self) -> int:
        """Get current timestep index."""
        return self.current_step
    
    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.current_step >= len(self.prices)
    
    def get_remaining_steps(self) -> int:
        """Get number of remaining timesteps in episode."""
        return max(0, len(self.prices) - self.current_step)
    
    def get_price_at(self, step: int) -> float:
        """
        Get price at a specific timestep.
        
        Args:
            step: Timestep index
        
        Returns:
            Price at specified timestep
        """
        if step < 0 or step >= len(self.prices):
            raise IndexError(f"Step {step} out of range [0, {len(self.prices)-1}]")
        return float(self.prices[step])
    
    def get_prices_in_window(
        self,
        start_step: int,
        end_step: int
    ) -> np.ndarray:
        """
        Get prices in a specified window.
        
        Args:
            start_step: Start timestep (inclusive)
            end_step: End timestep (inclusive)
        
        Returns:
            Array of prices in window
        """
        if start_step < 0:
            start_step = 0
        if end_step >= len(self.prices):
            end_step = len(self.prices) - 1
        if start_step > end_step:
            return np.array([])
        
        return self.prices[start_step:end_step + 1].copy()


def create_trading_environment(
    ticker: str,
    prices: np.ndarray,
    dates: Optional[np.ndarray] = None,
    initial_capital: float = 100000.0,
    transaction_cost: float = 0.001
) -> TradingEnvironment:
    """
    Factory function to create a TradingEnvironment instance.
    
    Args:
        ticker: Asset ticker symbol
        prices: Array of historical prices
        dates: Optional array of trading dates
        initial_capital: Starting account value (default: $100,000)
        transaction_cost: Transaction cost as fraction (default: 0.1%)
    
    Returns:
        Configured TradingEnvironment instance
    """
    return TradingEnvironment(
        ticker=ticker,
        prices=prices,
        dates=dates,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
