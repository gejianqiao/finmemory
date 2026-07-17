"""
Data Processors Module for FinMemory Trading System.

This module provides data processing utilities for converting raw market data
into structured features and signals for trading agents.

Modules:
    - price_processor: OHLCV data processing, returns, technical indicators
    - signal_processor: News and SEC filing signal processing, sentiment aggregation
"""

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
    # Price Processor
    "PriceProcessor",
    "PriceFeatures",
    "create_price_processor",
    "process_price_data",
    # Signal Processor
    "SignalProcessor",
    "NewsSignal",
    "SignalSnapshot",
    "create_signal_processor",
    "process_market_signals",
]
