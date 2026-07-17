"""
Data module for FinMemory trading agent system.

This module provides data loading and processing utilities for:
- Historical price data (Yahoo Finance)
- Company and macro news (Finnhub API)
- SEC filings (10-K, 10-Q from EDGAR)
- Price feature extraction and normalization
- Market signal processing and aggregation
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

from finmemory.data.processors.price_processor import (
    PriceProcessor,
    PriceFeatures,
    create_price_processor,
    process_price_data,
)

from finmemory.data.processors.signal_processor import (
    SignalProcessor,
    NewsSignal,
    SignalSnapshot,
    create_signal_processor,
    process_market_signals,
)

__version__ = "1.0.0"

__all__ = [
    # Loaders
    "YahooFinanceLoader",
    "create_yahoo_loader",
    "download_price_data",
    "FinnhubNewsLoader",
    "create_finnhub_loader",
    "download_news_data",
    "SECFilingLoader",
    "create_sec_loader",
    "download_sec_filings",
    # Processors
    "PriceProcessor",
    "PriceFeatures",
    "create_price_processor",
    "process_price_data",
    "SignalProcessor",
    "NewsSignal",
    "SignalSnapshot",
    "create_signal_processor",
    "process_market_signals",
]
