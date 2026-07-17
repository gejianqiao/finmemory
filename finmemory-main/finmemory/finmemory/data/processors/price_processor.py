"""
Price Processor Module for FinMemory Trading System

Processes raw OHLCV price data into normalized returns, technical indicators,
and time series features for use by trading agents and environment.

Author: FinMemory Research Team
Date: 2024
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PriceFeatures:
    """Dataclass storing processed price features for a single timestep."""
    timestamp: pd.Timestamp
    price: float
    log_return: float
    simple_return: float
    volatility: float
    momentum_5d: float
    momentum_10d: float
    momentum_20d: float
    ma_5: float
    ma_10: float
    ma_20: float
    ma_50: float
    rs_14: float
    price_normalized: float
    volume_normalized: float


class PriceProcessor:
    """
    Processes raw OHLCV price data into normalized features and technical indicators.
    
    Handles:
    - Log return and simple return calculations
    - Rolling volatility estimation
    - Moving averages (5, 10, 20, 50 day)
    - Momentum indicators (5, 10, 20 day)
    - Relative strength (14 day)
    - Price and volume normalization
    """
    
    def __init__(
        self,
        ticker: str,
        normalization_window: int = 252,
        volatility_window: int = 20,
        rs_window: int = 14
    ):
        """
        Initialize PriceProcessor with configuration.
        
        Args:
            ticker: Stock ticker symbol
            normalization_window: Window for price/volume normalization (default 252 = 1 trading year)
            volatility_window: Window for rolling volatility calculation (default 20 days)
            rs_window: Window for relative strength calculation (default 14 days)
        """
        self.ticker = ticker
        self.normalization_window = normalization_window
        self.volatility_window = volatility_window
        self.rs_window = rs_window
        
        self._price_data: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._base_price: float = 1.0
        self._base_volume: float = 1.0
        
        logger.info(f"PriceProcessor initialized for {ticker}")
    
    def process(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw OHLCV data into features.
        
        Args:
            price_data: DataFrame with columns [open, high, low, close, volume]
                       indexed by datetime
            
        Returns:
            DataFrame with processed features including returns, indicators, normalized values
        """
        logger.info(f"Processing price data for {self.ticker}: {len(price_data)} rows")
        
        # Data loaders and external callers commonly use Yahoo's title-case
        # OHLCV names; normalize once at this boundary.
        price_data = price_data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

        # Validate input data
        self._validate_price_data(price_data)
        
        # Store reference to raw data
        self._price_data = price_data.copy()
        
        # Calculate base values for normalization
        self._calculate_normalization_base()
        
        # Create features DataFrame
        self._features = pd.DataFrame(index=price_data.index)
        
        # Calculate all features
        self._calculate_returns()
        self._calculate_volatility()
        self._calculate_moving_averages()
        self._calculate_momentum()
        self._calculate_relative_strength()
        self._calculate_normalized_values()
        
        logger.info(f"Feature processing complete for {self.ticker}: {len(self._features)} features")
        
        return self._features
    
    def get_features(self) -> Optional[pd.DataFrame]:
        """Get processed features DataFrame."""
        return self._features
    
    def get_features_at(self, timestamp: pd.Timestamp) -> Optional[PriceFeatures]:
        """
        Get features for a specific timestamp.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            PriceFeatures dataclass or None if timestamp not found
        """
        if self._features is None:
            return None
        
        if timestamp not in self._features.index:
            return None
        
        row = self._features.loc[timestamp]
        
        return PriceFeatures(
            timestamp=timestamp,
            price=row['close'],
            log_return=row['log_return'],
            simple_return=row['simple_return'],
            volatility=row['volatility'],
            momentum_5d=row['momentum_5d'],
            momentum_10d=row['momentum_10d'],
            momentum_20d=row['momentum_20d'],
            ma_5=row['ma_5'],
            ma_10=row['ma_10'],
            ma_20=row['ma_20'],
            ma_50=row['ma_50'],
            rs_14=row['rs_14'],
            price_normalized=row['price_normalized'],
            volume_normalized=row['volume_normalized']
        )
    
    def get_features_in_window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> Optional[pd.DataFrame]:
        """
        Get features for a time window.
        
        Args:
            start: Start timestamp
            end: End timestamp
            
        Returns:
            DataFrame with features in window
        """
        if self._features is None:
            return None
        
        return self._features.loc[start:end]
    
    def get_latest_features(self, n: int = 1) -> Optional[pd.DataFrame]:
        """
        Get most recent n feature rows.
        
        Args:
            n: Number of rows to return
            
        Returns:
            DataFrame with latest n rows
        """
        if self._features is None:
            return None
        
        return self._features.tail(n)
    
    def reset(self):
        """Reset processor state."""
        self._price_data = None
        self._features = None
        self._base_price = 1.0
        self._base_volume = 1.0
        logger.debug(f"PriceProcessor reset for {self.ticker}")
    
    def _validate_price_data(self, data: pd.DataFrame):
        """Validate price data has required columns and no critical issues."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for NaN values in close prices (critical)
        if data['close'].isna().any():
            nan_count = data['close'].isna().sum()
            logger.warning(f"Found {nan_count} NaN values in close prices for {self.ticker}")
        
        # Check for zero or negative prices
        if (data['close'] <= 0).any():
            invalid_count = (data['close'] <= 0).sum()
            logger.warning(f"Found {invalid_count} zero/negative prices for {self.ticker}")
        
        # Check for zero volume
        if (data['volume'] <= 0).any():
            zero_vol_count = (data['volume'] <= 0).sum()
            logger.debug(f"Found {zero_vol_count} zero volume entries for {self.ticker}")
    
    def _calculate_normalization_base(self):
        """Calculate base values for price and volume normalization."""
        if self._price_data is None:
            return
        
        # Use first window's mean as base
        window_data = self._price_data.head(self.normalization_window)
        
        self._base_price = window_data['close'].mean()
        self._base_volume = window_data['volume'].mean()
        
        # Handle edge cases
        if self._base_price <= 0:
            self._base_price = self._price_data['close'].iloc[0]
        if self._base_volume <= 0:
            self._base_volume = self._price_data['volume'].mean()
        
        logger.debug(f"Normalization base for {self.ticker}: price={self._base_price:.2f}, volume={self._base_volume:.2f}")
    
    def _calculate_returns(self):
        """Calculate log returns and simple returns."""
        if self._price_data is None:
            return
        
        close = self._price_data['close']
        
        # Log returns: log(price_t / price_{t-1})
        self._features['log_return'] = np.log(close / close.shift(1))
        
        # Simple returns: (price_t - price_{t-1}) / price_{t-1}
        self._features['simple_return'] = close.pct_change()
        
        # Fill first NaN with 0
        self._features['log_return'] = self._features['log_return'].fillna(0)
        self._features['simple_return'] = self._features['simple_return'].fillna(0)
        
        logger.debug(f"Returns calculated for {self.ticker}")
    
    def _calculate_volatility(self):
        """Calculate rolling volatility (standard deviation of log returns)."""
        if self._features is None:
            return
        
        # Rolling standard deviation of log returns
        self._features['volatility'] = (
            self._features['log_return']
            .rolling(window=self.volatility_window, min_periods=1)
            .std()
        )
        
        # Annualize volatility (assuming 252 trading days)
        self._features['volatility'] = self._features['volatility'] * np.sqrt(252)
        
        # Fill any remaining NaN with first valid value
        self._features['volatility'] = self._features['volatility'].bfill().fillna(0)
        
        logger.debug(f"Volatility calculated for {self.ticker} (window={self.volatility_window})")
    
    def _calculate_moving_averages(self):
        """Calculate moving averages (5, 10, 20, 50 day)."""
        if self._price_data is None:
            return
        
        close = self._price_data['close']
        
        self._features['ma_5'] = close.rolling(window=5, min_periods=1).mean()
        self._features['ma_10'] = close.rolling(window=10, min_periods=1).mean()
        self._features['ma_20'] = close.rolling(window=20, min_periods=1).mean()
        self._features['ma_50'] = close.rolling(window=50, min_periods=1).mean()
        
        logger.debug(f"Moving averages calculated for {self.ticker}")
    
    def _calculate_momentum(self):
        """Calculate momentum indicators (5, 10, 20 day)."""
        if self._price_data is None:
            return
        
        close = self._price_data['close']
        
        # Momentum as price change over period
        self._features['momentum_5d'] = close.pct_change(periods=5)
        self._features['momentum_10d'] = close.pct_change(periods=10)
        self._features['momentum_20d'] = close.pct_change(periods=20)
        
        # Fill NaN values
        self._features['momentum_5d'] = self._features['momentum_5d'].fillna(0)
        self._features['momentum_10d'] = self._features['momentum_10d'].fillna(0)
        self._features['momentum_20d'] = self._features['momentum_20d'].fillna(0)
        
        logger.debug(f"Momentum indicators calculated for {self.ticker}")
    
    def _calculate_relative_strength(self):
        """Calculate 14-day Relative Strength (RS) indicator."""
        if self._price_data is None:
            return
        
        close = self._price_data['close']
        
        # Calculate price changes
        delta = close.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = (-delta).where(delta < 0, 0)
        
        # Calculate average gains and losses over RS window
        avg_gains = gains.rolling(window=self.rs_window, min_periods=1).mean()
        avg_losses = losses.rolling(window=self.rs_window, min_periods=1).mean()
        
        # Calculate RS (avoid division by zero)
        rs = avg_gains / avg_losses.replace(0, np.inf)
        rs = rs.replace([np.inf, -np.inf], 0)
        
        self._features['rs_14'] = rs
        
        logger.debug(f"Relative Strength calculated for {self.ticker} (window={self.rs_window})")
    
    def _calculate_normalized_values(self):
        """Calculate normalized price and volume values."""
        if self._price_data is None or self._features is None:
            return
        
        # Normalize price by base price
        self._features['price_normalized'] = self._price_data['close'] / self._base_price
        
        # Normalize volume by base volume
        self._features['volume_normalized'] = self._price_data['volume'] / self._base_volume
        
        logger.debug(f"Normalized values calculated for {self.ticker}")


def create_price_processor(
    ticker: str,
    normalization_window: int = 252,
    volatility_window: int = 20,
    rs_window: int = 14
) -> PriceProcessor:
    """
    Factory function to create configured PriceProcessor instance.
    
    Args:
        ticker: Stock ticker symbol
        normalization_window: Window for price/volume normalization (default 252)
        volatility_window: Window for rolling volatility (default 20)
        rs_window: Window for relative strength (default 14)
        
    Returns:
        PriceProcessor: Configured processor instance
    """
    return PriceProcessor(
        ticker=ticker,
        normalization_window=normalization_window,
        volatility_window=volatility_window,
        rs_window=rs_window
    )


def process_price_data(
    price_data: pd.DataFrame,
    ticker: str,
    normalization_window: int = 252
) -> pd.DataFrame:
    """
    Convenience function to process price data in one call.
    
    Args:
        price_data: Raw OHLCV DataFrame
        ticker: Stock ticker symbol
        normalization_window: Window for normalization
        
    Returns:
        DataFrame with processed features
    """
    processor = create_price_processor(
        ticker=ticker,
        normalization_window=normalization_window
    )
    
    return processor.process(price_data)
