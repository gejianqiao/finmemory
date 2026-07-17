"""
Finnhub News Data Loader for FinMemory Trading System.

This module provides functionality to fetch company-specific news and macroeconomic
news from the Finnhub API. It implements rate limiting, caching, and data validation
to ensure reliable news data collection within API tier limits.

Paper Reference: Section 4.1 (Market Signal Processing)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from finmemory.utils.logger import get_logger
from finmemory.utils.api_clients import FinnhubClient, APICache

logger = get_logger(__name__)


class FinnhubNewsLoader:
    """
    Loader for fetching company and macroeconomic news from Finnhub API.
    
    Implements rate limiting (60 calls/min, 30k/month free tier), response caching,
    and data validation for reliable news data collection.
    
    Attributes:
        data_dir: Directory for caching news data
        cache: APICache instance for response caching
        client: FinnhubClient for API calls
        cache_enabled: Whether to use disk caching
    """
    
    def __init__(
        self,
        data_dir: str = "data/raw/news",
        cache_enabled: bool = True,
        api_key: Optional[str] = None,
        cache_ttl: int = 3600  # 1 hour default TTL for news
    ):
        """
        Initialize Finnhub news loader.
        
        Args:
            data_dir: Directory to store cached news data
            cache_enabled: Whether to enable disk caching
            api_key: Finnhub API key (uses FINNHUB_API_KEY env var if None)
            cache_ttl: Cache TTL in seconds (default 1 hour for news)
        """
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self.client = FinnhubClient(api_key=api_key)
        self.cache = APICache(ttl_seconds=cache_ttl, max_size=1000)
        
        # Create data directory
        if self.cache_enabled:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FinnhubNewsLoader initialized with data_dir={self.data_dir}")
    
    def _get_cache_path(self, ticker: str, start_date: str, end_date: str) -> Path:
        """Get cache file path for company news."""
        return self.data_dir / f"{ticker}_{start_date}_{end_date}.csv"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load news data from disk cache."""
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, parse_dates=['datetime'])
                logger.debug(f"Loaded {len(df)} news items from cache: {cache_path}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path) -> None:
        """Save news data to disk cache."""
        if self.cache_enabled and len(df) > 0:
            try:
                df.to_csv(cache_path, index=False)
                logger.debug(f"Saved {len(df)} news items to cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
    
    def get_company_news(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch company-specific news for a ticker within date range.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use disk caching
            
        Returns:
            DataFrame with columns: datetime, headline, summary, sentiment,
                                   source, url, related_tickers
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cache_path = self._get_cache_path(ticker, start_date, end_date)
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df
        
        # Check memory cache
        cache_params = {"ticker": ticker, "start_date": start_date, "end_date": end_date}
        cached = self.cache.get("company-news-loader", cache_params)
        if cached is not None:
            logger.debug(f"News loaded from memory cache for {ticker}")
            return pd.DataFrame(cached)
        
        try:
            logger.info(f"Fetching company news for {ticker} from {start_date} to {end_date}")
            
            # Fetch news from Finnhub API
            news_items = self.client.get_company_news(ticker, start_date, end_date)
            
            if not news_items:
                logger.warning(f"No news found for {ticker} in date range")
                df = pd.DataFrame(columns=[
                    'datetime', 'headline', 'summary', 'sentiment',
                    'source', 'url', 'related_tickers'
                ])
            else:
                # Parse news items into DataFrame
                df = self._parse_news_items(news_items, ticker)
                logger.info(f"Fetched {len(df)} news items for {ticker}")
            
            # Cache results
            if use_cache and self.cache_enabled:
                cache_path = self._get_cache_path(ticker, start_date, end_date)
                self._save_to_cache(df, cache_path)
            
            # Memory cache
            self.cache.set("company-news-loader", cache_params, df.to_dict('records'))
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch company news for {ticker}: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame(columns=[
                'datetime', 'headline', 'summary', 'sentiment',
                'source', 'url', 'related_tickers'
            ])
    
    def get_macro_news(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch general macroeconomic/market news.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use disk caching
            
        Returns:
            DataFrame with columns: datetime, headline, summary, sentiment,
                                   source, url, category
        """
        # Check cache first
        if use_cache and self.cache_enabled:
            cache_path = self.data_dir / f"macro_{start_date}_{end_date}.csv"
            cached_df = self._load_from_cache(cache_path)
            if cached_df is not None:
                return cached_df
        
        # Check memory cache
        cache_params = {"start_date": start_date, "end_date": end_date}
        cached = self.cache.get("macro-news-loader", cache_params)
        if cached is not None:
            logger.debug("Macro news loaded from memory cache")
            return pd.DataFrame(cached)
        
        try:
            logger.info(f"Fetching macro news from {start_date} to {end_date}")
            
            # Fetch general news from Finnhub API
            news_items = self.client.get_general_news(topic="general", limit=1000)
            
            if not news_items:
                logger.warning("No macro news found in date range")
                df = pd.DataFrame(columns=[
                    'datetime', 'headline', 'summary', 'sentiment',
                    'source', 'url', 'category'
                ])
            else:
                # Parse news items into DataFrame
                df = self._parse_news_items(news_items, ticker=None)
                if not df.empty:
                    start = pd.Timestamp(start_date, tz="UTC")
                    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
                    datetimes = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
                    df = df.loc[(datetimes >= start) & (datetimes < end)].copy()
                logger.info(f"Fetched {len(df)} macro news items")
            
            # Cache results
            if use_cache and self.cache_enabled:
                cache_path = self.data_dir / f"macro_{start_date}_{end_date}.csv"
                self._save_to_cache(df, cache_path)
            
            # Memory cache
            self.cache.set("macro-news-loader", cache_params, df.to_dict('records'))
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch macro news: {e}")
            return pd.DataFrame(columns=[
                'datetime', 'headline', 'summary', 'sentiment',
                'source', 'url', 'category'
            ])
    
    def _parse_news_items(self, news_items: List[Dict[str, Any]], ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Parse raw news items from Finnhub API into structured DataFrame.
        
        Args:
            news_items: List of raw news items from API
            ticker: Ticker symbol (None for macro news)
            
        Returns:
            Structured DataFrame with standardized columns
        """
        parsed_items = []
        
        for item in news_items:
            try:
                # Convert timestamp to datetime
                timestamp = item.get('datetime')
                if isinstance(timestamp, int):
                    dt = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.now()
                
                # Extract sentiment score (Finnhub provides -1 to 1 or 0 to 1)
                sentiment = item.get('sentiment', None)
                if sentiment is not None:
                    # Normalize to -1 to 1 range if needed
                    if isinstance(sentiment, str):
                        sentiment_map = {'negative': -0.5, 'neutral': 0.0, 'positive': 0.5}
                        sentiment = sentiment_map.get(sentiment.lower(), 0.0)
                else:
                    sentiment = 0.0  # Default neutral if not provided
                
                parsed_item = {
                    'datetime': dt,
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', item.get('headline', '')),
                    'sentiment': sentiment,
                    'source': item.get('source', 'unknown'),
                    'url': item.get('url', ''),
                    'related_tickers': ','.join(item.get('related', [])) if ticker is None else ticker,
                    'category': item.get('category', 'general') if ticker is None else 'company'
                }
                
                parsed_items.append(parsed_item)
                
            except Exception as e:
                logger.warning(f"Failed to parse news item: {e}")
                continue
        
        df = pd.DataFrame(parsed_items)
        
        # Sort by datetime
        if len(df) > 0:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def get_news_for_period(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        include_macro: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch news for multiple tickers and optionally macro news.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            include_macro: Whether to include macroeconomic news
            
        Returns:
            Dictionary mapping ticker (or 'macro') to news DataFrames
        """
        results = {}
        
        # Fetch company news for each ticker
        for ticker in tickers:
            logger.info(f"Fetching news for {ticker}")
            news_df = self.get_company_news(ticker, start_date, end_date)
            results[ticker] = news_df
            logger.info(f"  -> {len(news_df)} news items for {ticker}")
        
        # Fetch macro news if requested
        if include_macro:
            logger.info("Fetching macroeconomic news")
            macro_df = self.get_macro_news(start_date, end_date)
            results['macro'] = macro_df
            logger.info(f"  -> {len(macro_df)} macro news items")
        
        return results
    
    def validate_news_data(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Validate news data quality.
        
        Args:
            df: News DataFrame to validate
            ticker: Ticker symbol for context
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'ticker': ticker,
            'total_items': len(df),
            'is_valid': True,
            'issues': []
        }
        
        if len(df) == 0:
            validation['issues'].append("No news items found")
            return validation
        
        # Check for missing headlines
        missing_headlines = df['headline'].isna().sum()
        if missing_headlines > 0:
            validation['issues'].append(f"{missing_headlines} items missing headlines")
            validation['is_valid'] = False
        
        # Check for missing timestamps
        missing_dates = df['datetime'].isna().sum()
        if missing_dates > 0:
            validation['issues'].append(f"{missing_dates} items missing timestamps")
            validation['is_valid'] = False
        
        # Check date range
        if 'datetime' in df.columns and len(df) > 0:
            min_date = df['datetime'].min()
            max_date = df['datetime'].max()
            validation['date_range'] = {
                'min': str(min_date),
                'max': str(max_date)
            }
        
        # Check sentiment distribution
        if 'sentiment' in df.columns:
            sentiment_stats = {
                'mean': float(df['sentiment'].mean()),
                'std': float(df['sentiment'].std()),
                'min': float(df['sentiment'].min()),
                'max': float(df['sentiment'].max())
            }
            validation['sentiment_stats'] = sentiment_stats
        
        return validation
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return self.client.get_usage_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached news data."""
        self.cache.clear()
        if self.cache_enabled and self.data_dir.exists():
            for file in self.data_dir.glob("*.csv"):
                file.unlink()
            logger.info("Cleared all news cache files")


def create_finnhub_loader(
    data_dir: str = "data/raw/news",
    cache_enabled: bool = True,
    api_key: Optional[str] = None
) -> FinnhubNewsLoader:
    """
    Factory function to create configured FinnhubNewsLoader instance.
    
    Args:
        data_dir: Directory for caching news data
        cache_enabled: Whether to enable disk caching
        api_key: Finnhub API key (uses env var if None)
        
    Returns:
        FinnhubNewsLoader: Configured loader instance
    """
    return FinnhubNewsLoader(
        data_dir=data_dir,
        cache_enabled=cache_enabled,
        api_key=api_key
    )


def download_news_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: str = "data/raw/news",
    api_key: Optional[str] = None,
    include_macro: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download news data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        data_dir: Directory for caching
        api_key: Finnhub API key
        include_macro: Whether to include macroeconomic news
        
    Returns:
        Dictionary mapping ticker to news DataFrames
    """
    loader = create_finnhub_loader(data_dir=data_dir, api_key=api_key)
    return loader.get_news_for_period(tickers, start_date, end_date, include_macro)
