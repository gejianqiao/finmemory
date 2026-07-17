"""
Data Loaders Package for FinMemory Trading Agent System

This package provides data loading utilities for fetching and caching:
- Historical OHLCV price data (Yahoo Finance)
- Company and macroeconomic news (Finnhub API)
- SEC filings 10-K/10-Q (SEC EDGAR API)

All loaders implement dual-layer caching (memory + disk) to minimize API calls
and ensure reliable operation within free tier limits.
"""

from finmemory.data.loaders.yahoo_loader import (
    YahooFinanceLoader,
    create_yahoo_loader,
    download_price_data,
)

from finmemory.data.loaders.finnhub_loader import (
    FinnhubNewsLoader,
    create_finnhub_loader,
    download_news_data,
)

from finmemory.data.loaders.sec_loader import (
    SECFilingLoader,
    create_sec_loader,
    download_sec_filings,
)

__version__ = "1.0.0"

__all__ = [
    # Yahoo Finance Loader
    "YahooFinanceLoader",
    "create_yahoo_loader",
    "download_price_data",
    
    # Finnhub News Loader
    "FinnhubNewsLoader",
    "create_finnhub_loader",
    "download_news_data",
    
    # SEC Filing Loader
    "SECFilingLoader",
    "create_sec_loader",
    "download_sec_filings",
]
