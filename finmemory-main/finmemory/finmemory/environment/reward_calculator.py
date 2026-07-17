"""
Multi-Timescale Reward Module for FinMemory Trading System.

This module implements the multi-timescale reward design from Section 4.3 of the paper.
It provides training signals across 1-day, 7-day, and 30-day horizons to align
short-term actions with long-term outcomes.

Formulas (Eq. 3-5):
    Ms_t = price[t+1] - price[t]           # 1-day trend (short-term)
    Mm_t = price[t+7] - price[t]           # 7-day trend (mid-term)
    Ml_t = price[t+30] - price[t]          # 30-day trend (long-term)
    M_t = Ms_t + Mm_t + Ml_t               # Combined multi-timescale signal
    
    if position_t == position_{t-1}:       # No action taken
        reward = -(M_t)²                   # Penalty for inaction during volatility
    else:                                  # Action taken
        reward = position_t × M_t          # Reward proportional to position & trend
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrendSignals:
    """Dataclass representing multi-timescale trend signals."""
    short_term: float  # 1-day trend (Ms_t)
    mid_term: float    # 7-day trend (Mm_t)
    long_term: float   # 30-day trend (Ml_t)
    combined: float    # Combined signal (M_t)
    
    def __post_init__(self):
        """Validate trend signals."""
        if not all(isinstance(x, (int, float)) for x in [self.short_term, self.mid_term, self.long_term, self.combined]):
            raise ValueError("All trend signals must be numeric")


@dataclass
class RewardResult:
    """Dataclass representing reward calculation result."""
    reward: float
    trend_signals: TrendSignals
    reward_type: str  # 'action' or 'inaction'
    position: float
    timestep: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'reward': self.reward,
            'reward_type': self.reward_type,
            'position': self.position,
            'timestep': self.timestep,
            'trend_signals': {
                'short_term': self.trend_signals.short_term,
                'mid_term': self.trend_signals.mid_term,
                'long_term': self.trend_signals.long_term,
                'combined': self.trend_signals.combined
            }
        }


class RewardCalculator:
    """
    Multi-timescale reward calculator for FinMemory trading environment.
    
    Implements the reward formula from Section 4.3 of the paper:
    - Calculates trend signals at 1-day, 7-day, and 30-day horizons
    - Applies different reward logic based on whether action was taken
    - Handles boundary conditions when future prices are unavailable
    
    Attributes:
        prices: Array of historical prices
        dates: Array of corresponding dates
        short_window: Short-term trend window (default: 1 day)
        mid_window: Mid-term trend window (default: 7 days)
        long_window: Long-term trend window (default: 30 days)
        inaction_penalty_scale: Scaling factor for inaction penalty (default: 1.0)
    """
    
    def __init__(
        self,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        short_window: int = 1,
        mid_window: int = 7,
        long_window: int = 30,
        inaction_penalty_scale: float = 1.0
    ):
        """
        Initialize reward calculator with price data.
        
        Args:
            prices: Array of historical prices (length T)
            dates: Optional array of corresponding dates
            short_window: Short-term trend window in days (default: 1)
            mid_window: Mid-term trend window in days (default: 7)
            long_window: Long-term trend window in days (default: 30)
            inaction_penalty_scale: Scaling factor for inaction penalty (default: 1.0)
        """
        self.prices = np.asarray(prices, dtype=np.float64)
        self.dates = dates
        self.short_window = short_window
        self.mid_window = mid_window
        self.long_window = long_window
        self.inaction_penalty_scale = inaction_penalty_scale
        
        self.n_timesteps = len(self.prices)
        
        # Pre-calculate all trend signals for efficiency
        self._trend_signals_cache: Dict[int, TrendSignals] = {}
        self._precalculate_trends()
        
        logger.info(
            f"RewardCalculator initialized with {self.n_timesteps} timesteps, "
            f"windows: short={short_window}, mid={mid_window}, long={long_window}"
        )
    
    def _precalculate_trends(self) -> None:
        """Pre-calculate trend signals for all timesteps."""
        for t in range(self.n_timesteps):
            self._trend_signals_cache[t] = self._calculate_trend_signals(t)
    
    def _calculate_trend_signals(self, timestep: int) -> TrendSignals:
        """
        Calculate multi-timescale trend signals at a given timestep.
        
        Args:
            timestep: Current timestep index
            
        Returns:
            TrendSignals object with short, mid, long, and combined trends
            
        Notes:
            - Handles boundary conditions by using available data
            - If future price unavailable, uses last available price
            - Trends are calculated as price differences (not returns)
        """
        current_price = self.prices[timestep]
        
        # Calculate short-term trend (1-day)
        short_idx = min(timestep + self.short_window, self.n_timesteps - 1)
        short_price = self.prices[short_idx]
        short_term = short_price - current_price
        
        # Calculate mid-term trend (7-day)
        mid_idx = min(timestep + self.mid_window, self.n_timesteps - 1)
        mid_price = self.prices[mid_idx]
        mid_term = mid_price - current_price
        
        # Calculate long-term trend (30-day)
        long_idx = min(timestep + self.long_window, self.n_timesteps - 1)
        long_price = self.prices[long_idx]
        long_term = long_price - current_price
        
        # Combined signal
        combined = short_term + mid_term + long_term
        
        return TrendSignals(
            short_term=short_term,
            mid_term=mid_term,
            long_term=long_term,
            combined=combined
        )
    
    def calculate_reward(
        self,
        timestep: int,
        current_position: float,
        previous_position: float,
        use_squared_penalty: bool = True
    ) -> RewardResult:
        """
        Calculate multi-timescale reward for a trading decision.
        
        Args:
            timestep: Current timestep index
            current_position: Current position size after action
            previous_position: Previous position size before action
            use_squared_penalty: Whether to use squared penalty for inaction (default: True)
            
        Returns:
            RewardResult object with reward value and metadata
            
        Formula:
            M_t = Ms_t + Mm_t + Ml_t  (combined trend signal)
            
            if position_t == position_{t-1}:  # No action
                reward = -(M_t)²  (penalty for inaction during volatility)
            else:  # Action taken
                reward = position_t × M_t  (reward proportional to position & trend)
        """
        if timestep < 0 or timestep >= self.n_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [0, {self.n_timesteps - 1}]")
        
        # Get pre-calculated trend signals
        trend_signals = self._trend_signals_cache[timestep]
        combined_trend = trend_signals.combined
        
        # Determine if action was taken
        action_taken = abs(current_position - previous_position) > 1e-6
        
        if action_taken:
            # Reward proportional to position and trend
            reward = current_position * combined_trend
            reward_type = 'action'
            logger.debug(
                f"Timestep {timestep}: Action reward = {current_position:.4f} × "
                f"{combined_trend:.4f} = {reward:.4f}"
            )
        else:
            # Penalty for inaction during volatility
            if use_squared_penalty:
                reward = -self.inaction_penalty_scale * (combined_trend ** 2)
            else:
                reward = -self.inaction_penalty_scale * abs(combined_trend)
            reward_type = 'inaction'
            logger.debug(
                f"Timestep {timestep}: Inaction penalty = -{combined_trend:.4f}² = {reward:.4f}"
            )
        
        return RewardResult(
            reward=reward,
            trend_signals=trend_signals,
            reward_type=reward_type,
            position=current_position,
            timestep=timestep
        )
    
    def calculate_reward_batch(
        self,
        timesteps: List[int],
        positions: List[float],
        previous_positions: List[float]
    ) -> List[RewardResult]:
        """
        Calculate rewards for a batch of timesteps.
        
        Args:
            timesteps: List of timestep indices
            positions: List of current position sizes
            previous_positions: List of previous position sizes
            
        Returns:
            List of RewardResult objects
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if not (len(timesteps) == len(positions) == len(previous_positions)):
            raise ValueError("All input lists must have the same length")
        
        results = []
        for t, pos, prev_pos in zip(timesteps, positions, previous_positions):
            result = self.calculate_reward(t, pos, prev_pos)
            results.append(result)
        
        return results
    
    def get_trend_at(self, timestep: int) -> TrendSignals:
        """
        Get pre-calculated trend signals at a specific timestep.
        
        Args:
            timestep: Timestep index
            
        Returns:
            TrendSignals object
        """
        if timestep < 0 or timestep >= self.n_timesteps:
            raise ValueError(f"Timestep {timestep} out of range [0, {self.n_timesteps - 1}]")
        return self._trend_signals_cache[timestep]
    
    def get_all_trends(self) -> Dict[int, TrendSignals]:
        """
        Get all pre-calculated trend signals.
        
        Returns:
            Dictionary mapping timestep to TrendSignals
        """
        return self._trend_signals_cache.copy()
    
    def calculate_cumulative_reward(
        self,
        positions: List[float],
        previous_positions: List[float]
    ) -> float:
        """
        Calculate cumulative reward over an episode.
        
        Args:
            positions: List of position sizes at each timestep
            previous_positions: List of previous position sizes at each timestep
            
        Returns:
            Total cumulative reward
        """
        if len(positions) != len(previous_positions):
            raise ValueError("Positions and previous_positions must have same length")
        
        total_reward = 0.0
        for t in range(len(positions)):
            if t < self.n_timesteps:
                result = self.calculate_reward(t, positions[t], previous_positions[t])
                total_reward += result.reward
        
        return total_reward
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistics about trend signals across all timesteps.
        
        Returns:
            Dictionary with statistical measures of trend signals
        """
        short_terms = [ts.short_term for ts in self._trend_signals_cache.values()]
        mid_terms = [ts.mid_term for ts in self._trend_signals_cache.values()]
        long_terms = [ts.long_term for ts in self._trend_signals_cache.values()]
        combined = [ts.combined for ts in self._trend_signals_cache.values()]
        
        return {
            'n_timesteps': self.n_timesteps,
            'short_term': {
                'mean': float(np.mean(short_terms)),
                'std': float(np.std(short_terms)),
                'min': float(np.min(short_terms)),
                'max': float(np.max(short_terms))
            },
            'mid_term': {
                'mean': float(np.mean(mid_terms)),
                'std': float(np.std(mid_terms)),
                'min': float(np.min(mid_terms)),
                'max': float(np.max(mid_terms))
            },
            'long_term': {
                'mean': float(np.mean(long_terms)),
                'std': float(np.std(long_terms)),
                'min': float(np.min(long_terms)),
                'max': float(np.max(long_terms))
            },
            'combined': {
                'mean': float(np.mean(combined)),
                'std': float(np.std(combined)),
                'min': float(np.min(combined)),
                'max': float(np.max(combined))
            }
        }


def create_reward_calculator(
    prices: np.ndarray,
    dates: Optional[np.ndarray] = None,
    short_window: int = 1,
    mid_window: int = 7,
    long_window: int = 30,
    inaction_penalty_scale: float = 1.0
) -> RewardCalculator:
    """
    Factory function to create a RewardCalculator instance.
    
    Args:
        prices: Array of historical prices
        dates: Optional array of corresponding dates
        short_window: Short-term trend window in days (default: 1)
        mid_window: Mid-term trend window in days (default: 7)
        long_window: Long-term trend window in days (default: 30)
        inaction_penalty_scale: Scaling factor for inaction penalty (default: 1.0)
        
    Returns:
        Configured RewardCalculator instance
    """
    return RewardCalculator(
        prices=prices,
        dates=dates,
        short_window=short_window,
        mid_window=mid_window,
        long_window=long_window,
        inaction_penalty_scale=inaction_penalty_scale
    )
