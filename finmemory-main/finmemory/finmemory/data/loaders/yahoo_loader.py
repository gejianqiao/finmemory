"""
Yahoo Finance Data Loader for FinMemory Trading System.

This module provides functionality to download daily OHLCV (Open, High, Low, Close, Volume)
price data from Yahoo Finance using the yfinance library. It handles stock splits, dividends,
missing data, and saves data to CSV files for offline use.

Author: FinMemory Research Team
Date: 2024
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import yfinance as yf

from finmemory.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class YahooFinanceLoader:
    """
    Loader for downloading historical price data from Yahoo Finance.
    
    Handles daily OHLCV data retrieval with support for:
    - Custom date ranges
    - Multiple tickers
    - Adjusted prices (splits, dividends)
    - Missing data handling
    - Local caching to CSV files
    
    Attributes:
        data_dir: Directory path for storing downloaded data
        cache_enabled: Whether to cache downloaded data to disk
    """
    
    def __init__(
        self,
        data_dir: str = "finmemory/data/raw/prices",
        cache_enabled: bool = True
    ):
        """
        Initialize Yahoo Finance loader.
        
        Args:
            data_dir: Directory to store downloaded price data CSV files
            cache_enabled: If True, save downloaded data to CSV for reuse
        """
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}
        
        # Create data directory if it doesn't exist
        if self.cache_enabled:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Price data directory: {self.data_dir.absolute()}")
    
    def download_single(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Download historical price data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            force_refresh: If True, re-download even if cached data exists
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj_Close, Volume
            
        Raises:
            ValueError: If no data is available for the ticker
        """
        # Check cache first
        cache_file = self.data_dir / f"{ticker}.csv"
        if not force_refresh and cache_file.exists() and ticker in self._cache:
            logger.debug(f"Loading {ticker} from memory cache")
            return self._cache[ticker].copy()
        
        if not force_refresh and cache_file.exists():
            logger.debug(f"Loading {ticker} from disk cache: {cache_file}")
            try:
                df = pd.read_csv(cache_file, parse_dates=['date'])
                df = df.set_index('date', drop=False)
                self._cache[ticker] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}, re-downloading")
        
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
        
        try:
            # Download data using yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            if df.empty:
                raise ValueError(f"No data available for ticker {ticker}")
            
            # Reset index to make Date a column
            df = df.reset_index()
            df.rename(columns={'Date': 'date'}, inplace=True)
            
            # Ensure required columns exist
            required_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Use Adj Close if available, otherwise use Close
            if 'Adj Close' in df.columns:
                df['adj_close'] = df['Adj Close']
            else:
                df['adj_close'] = df['Close']
                logger.warning(f"Adjusted close not available for {ticker}, using Close")
            
            # Select and order columns
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'adj_close', 'Volume']]
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing data (forward fill, then backward fill)
            df = df.ffill().bfill()

            # Keep the trading date both as a column and as the canonical index.
            # Downstream environments consume the index as their timestep date.
            df = df.set_index('date', drop=False)
            df.index.name = 'trading_date'
            
            # Cache the data
            self._cache[ticker] = df.copy()
            
            # Save to disk if caching enabled
            if self.cache_enabled:
                df.to_csv(cache_file, index=False)
                logger.info(f"Saved {ticker} data to {cache_file} ({len(df)} rows)")
            
            logger.info(f"Successfully downloaded {ticker}: {len(df)} trading days")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            raise
    
    def download_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical price data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            force_refresh: If True, re-download all data
            
        Returns:
            Dictionary mapping ticker symbols to DataFrames
        """
        logger.info(f"Downloading data for {len(tickers)} tickers: {tickers}")
        
        results = {}
        failed = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"[{i}/{len(tickers)}] Processing {ticker}")
                df = self.download_single(ticker, start_date, end_date, force_refresh)
                results[ticker] = df
            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")
                failed.append(ticker)
        
        if failed:
            logger.warning(f"Failed to download {len(failed)} tickers: {failed}")
        
        logger.info(f"Download complete: {len(results)} successful, {len(failed)} failed")
        return results
    
    def load_from_cache(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load price data from local cache with optional date filtering.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            Filtered DataFrame with price data
        """
        cache_file = self.data_dir / f"{ticker}.csv"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"No cached data found for {ticker} at {cache_file}")
        
        logger.debug(f"Loading {ticker} from cache: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
        
        # Apply date filters if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        return df
    
    def get_trading_dates(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> List[datetime]:
        """
        Get list of trading dates for a ticker within a date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of trading dates as datetime objects
        """
        df = self.download_single(ticker, start_date, end_date)
        return df['date'].tolist()
    
    def validate_data(
        self,
        df: pd.DataFrame,
        ticker: str,
        max_gap_days: int = 3
    ) -> Tuple[bool, List[str]]:
        """
        Validate price data quality.
        
        Checks for:
        - Missing dates (gaps > max_gap_days)
        - Zero or negative prices
        - Extreme outliers (>50% daily change)
        - Missing values
        
        Args:
            df: DataFrame with price data
            ticker: Ticker symbol for logging
            max_gap_days: Maximum allowed gap between trading days
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            for col, count in missing.items():
                if count > 0:
                    issues.append(f"Missing values in {col}: {count}")
        
        # Check for zero or negative prices
        for col in ['Open', 'High', 'Low', 'Close', 'adj_close']:
            if col in df.columns:
                if (df[col] <= 0).any():
                    count = (df[col] <= 0).sum()
                    issues.append(f"Non-positive {col} values: {count}")
        
        # Check for extreme daily changes (>50%)
        if 'Close' in df.columns and len(df) > 1:
            daily_returns = df['Close'].pct_change().abs()
            extreme = (daily_returns > 0.5).sum()
            if extreme > 0:
                issues.append(f"Extreme daily changes (>50%): {extreme} occurrences")
        
        # Check for gaps in trading dates
        if 'date' in df.columns and len(df) > 1:
            dates = pd.to_datetime(df['date']).sort_values()
            gaps = dates.diff()[1:]  # Skip first NaT
            large_gaps = gaps[gaps > pd.Timedelta(days=max_gap_days)]
            if len(large_gaps) > 0:
                for idx, gap in large_gaps.items():
                    issues.append(f"Gap of {gap.days} days at index {idx}")
        
        is_valid = len(issues) == 0
        if is_valid:
            logger.info(f"Data validation passed for {ticker}")
        else:
            logger.warning(f"Data validation issues for {ticker}: {issues}")
        
        return is_valid, issues
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the most recent closing price for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Latest closing price or None if not available
        """
        try:
            # Try cache first
            if ticker in self._cache:
                return self._cache[ticker]['Close'].iloc[-1]
            
            # Try disk cache
            cache_file = self.data_dir / f"{ticker}.csv"
            if cache_file.exists():
                df = pd.read_csv(cache_file)
                return df['Close'].iloc[-1]
            
            # Download latest data
            stock = yf.Ticker(ticker)
            df = stock.history(period='1d')
            if not df.empty:
                return df['Close'].iloc[-1]
            
            return None
        except Exception as e:
            logger.error(f"Failed to get latest price for {ticker}: {e}")
            return None


def create_yahoo_loader(
    data_dir: str = "finmemory/data/raw/prices",
    cache_enabled: bool = True
) -> YahooFinanceLoader:
    """
    Factory function to create a configured YahooFinanceLoader instance.
    
    Args:
        data_dir: Directory for storing downloaded price data
        cache_enabled: Whether to enable disk caching
        
    Returns:
        Configured YahooFinanceLoader instance
    """
    return YahooFinanceLoader(data_dir=data_dir, cache_enabled=cache_enabled)


def download_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: str = "finmemory/data/raw/prices",
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols to download
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory for storing data
        force_refresh: Whether to force re-download
        
    Returns:
        Dictionary mapping tickers to DataFrames
    """
    loader = create_yahoo_loader(data_dir=data_dir)
    return loader.download_multiple(tickers, start_date, end_date, force_refresh)


if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)
    
    # Download data for FinMemory experiment stocks
    tickers = ["TSLA", "AAPL", "AMZN", "NFLX", "COIN"]
    start_date = "2024-01-01"
    end_date = "2025-09-30"
    
    print(f"Downloading price data for {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    
    loader = create_yahoo_loader()
    data = loader.download_multiple(tickers, start_date, end_date)
    
    for ticker, df in data.items():
        print(f"\n{ticker}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        # Validate data
        is_valid, issues = loader.validate_data(df, ticker)
        if not is_valid:
            print(f"  Validation issues: {issues}")
        else:
            print(f"  Validation: PASSED")
