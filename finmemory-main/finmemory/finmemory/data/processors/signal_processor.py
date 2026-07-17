"""
Signal Processor Module for FinMemory Trading System.

Processes raw market signals (news, filings) into structured insights with sentiment
scores and relevance metrics for deterministic preprocessing and market context.

Implements:
- News sentiment aggregation and normalization
- Relevance scoring for market signals
- Signal fusion from multiple sources (company news, macro news, SEC filings)
- Time-aligned signal series for trading decisions
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsSignal:
    """
    Dataclass representing a processed news signal.
    
    Attributes:
        timestamp: News publication timestamp
        source: Source type (company_news, macro_news, sec_filing)
        headline: News headline or filing title
        summary: News summary or filing excerpt
        sentiment_score: Normalized sentiment (-1.0 to 1.0)
        relevance_score: Relevance to target ticker (1-10)
        importance_score: Overall importance for memory allocation (1-10)
        ticker: Target ticker symbol
        metadata: Additional metadata (category, tags, etc.)
    """
    timestamp: datetime
    source: str
    headline: str
    summary: str
    sentiment_score: float
    relevance_score: float
    importance_score: float
    ticker: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'headline': self.headline,
            'summary': self.summary,
            'sentiment_score': self.sentiment_score,
            'relevance_score': self.relevance_score,
            'importance_score': self.importance_score,
            'ticker': self.ticker,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsSignal':
        """Create NewsSignal from dictionary."""
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            timestamp=timestamp,
            source=data['source'],
            headline=data['headline'],
            summary=data['summary'],
            sentiment_score=float(data['sentiment_score']),
            relevance_score=float(data['relevance_score']),
            importance_score=float(data['importance_score']),
            ticker=data['ticker'],
            metadata=data.get('metadata', {})
        )


@dataclass
class SignalSnapshot:
    """
    Dataclass representing aggregated signals at a specific timestep.
    
    Attributes:
        timestamp: Snapshot timestamp
        ticker: Target ticker symbol
        company_news_count: Number of company news items
        macro_news_count: Number of macro news items
        sec_filing_count: Number of SEC filings
        avg_sentiment: Average sentiment score across all signals
        sentiment_std: Standard deviation of sentiment scores
        max_importance: Maximum importance score
        signal_intensity: Overall signal intensity (0-1 normalized)
        signals: List of individual news signals
    """
    timestamp: datetime
    ticker: str
    company_news_count: int
    macro_news_count: int
    sec_filing_count: int
    avg_sentiment: float
    sentiment_std: float
    max_importance: float
    signal_intensity: float
    signals: List[NewsSignal] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'ticker': self.ticker,
            'company_news_count': self.company_news_count,
            'macro_news_count': self.macro_news_count,
            'sec_filing_count': self.sec_filing_count,
            'avg_sentiment': self.avg_sentiment,
            'sentiment_std': self.sentiment_std,
            'max_importance': self.max_importance,
            'signal_intensity': self.signal_intensity,
            'signals': [s.to_dict() for s in self.signals]
        }


class SignalProcessor:
    """
    Processes raw market data into structured signals for trading agents.
    
    This processor:
    1. Normalizes sentiment scores from different sources to [-1, 1] range
    2. Calculates relevance scores based on ticker mentions and context
    3. Aggregates signals by time period for efficient agent consumption
    4. Computes signal intensity metrics for market attention measurement
    
    Usage:
        processor = SignalProcessor(ticker='TSLA')
        processor.process_company_news(news_df)
        processor.process_macro_news(macro_df)
        processor.process_sec_filings(filings_list)
        snapshot = processor.get_signal_snapshot(date)
    """
    
    def __init__(
        self,
        ticker: str,
        sentiment_window: int = 5,
        intensity_decay: float = 0.5,
        min_relevance_threshold: float = 3.0
    ):
        """
        Initialize SignalProcessor.
        
        Args:
            ticker: Target ticker symbol
            sentiment_window: Rolling window for sentiment aggregation (days)
            intensity_decay: Decay factor for older signals (0-1)
            min_relevance_threshold: Minimum relevance score to include signal
        """
        self.ticker = ticker
        self.sentiment_window = sentiment_window
        self.intensity_decay = intensity_decay
        self.min_relevance_threshold = min_relevance_threshold
        
        # Signal storage
        self._company_news: List[NewsSignal] = []
        self._macro_news: List[NewsSignal] = []
        self._sec_filings: List[NewsSignal] = []
        self._all_signals: List[NewsSignal] = []
        
        # Time index for snapshots
        self._date_index: Optional[pd.DatetimeIndex] = None
        
        logger.info(f"Initialized SignalProcessor for {ticker}")
    
    def process_company_news(
        self,
        news_df: pd.DataFrame,
        filter_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[NewsSignal]:
        """
        Process company news DataFrame into NewsSignal objects.
        
        Args:
            news_df: DataFrame with columns [timestamp, headline, summary, sentiment]
            filter_results: Optional list of filter agent results with relevance scores
        
        Returns:
            List of processed NewsSignal objects
        """
        if news_df.empty:
            logger.debug("Empty company news DataFrame")
            return []
        
        signals = []
        for idx, row in news_df.iterrows():
            try:
                # Parse timestamp
                timestamp = row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                # Get sentiment (normalize to [-1, 1])
                sentiment = self._normalize_sentiment(row.get('sentiment', 0))
                
                # Get relevance from filter results or default
                relevance = 5.0  # Default medium relevance
                if filter_results and idx < len(filter_results):
                    relevance = filter_results[idx].get('relevance_score', 5.0)
                
                # Check if headline mentions ticker
                headline = str(row.get('headline', ''))
                if self.ticker in headline.upper():
                    relevance = max(relevance, 7.0)  # Boost relevance for direct mentions
                
                # Skip if below relevance threshold
                if relevance < self.min_relevance_threshold:
                    continue
                
                signal = NewsSignal(
                    timestamp=timestamp,
                    source='company_news',
                    headline=headline,
                    summary=str(row.get('summary', '')),
                    sentiment_score=sentiment,
                    relevance_score=relevance,
                    importance_score=relevance,  # Initial importance = relevance
                    ticker=self.ticker,
                    metadata={
                        'category': row.get('category', 'general'),
                        'source_url': row.get('url', '')
                    }
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing news row {idx}: {e}")
                continue
        
        self._company_news.extend(signals)
        self._rebuild_all_signals()
        
        logger.info(f"Processed {len(signals)} company news signals for {self.ticker}")
        return signals
    
    def process_macro_news(
        self,
        news_df: pd.DataFrame,
        filter_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[NewsSignal]:
        """
        Process macroeconomic news DataFrame into NewsSignal objects.
        
        Args:
            news_df: DataFrame with columns [timestamp, headline, summary, sentiment]
            filter_results: Optional list of filter results with relevance classification
        
        Returns:
            List of processed NewsSignal objects
        """
        if news_df.empty:
            logger.debug("Empty macro news DataFrame")
            return []
        
        signals = []
        for idx, row in news_df.iterrows():
            try:
                # Parse timestamp
                timestamp = row.get('timestamp')
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                # Get sentiment
                sentiment = self._normalize_sentiment(row.get('sentiment', 0))
                
                # Get relevance from filter results
                relevance = 3.0  # Default low-medium for macro news
                relevance_class = 'indirect'  # Default classification
                if filter_results and idx < len(filter_results):
                    relevance = filter_results[idx].get('relevance_score', 3.0)
                    relevance_class = filter_results[idx].get(
                        'relevance_category', filter_results[idx].get('relevance_class', 'indirect')
                    )
                
                # Adjust relevance based on classification
                if relevance_class == 'direct':
                    relevance = max(relevance, 7.0)
                elif relevance_class == 'none':
                    continue  # Skip irrelevant macro news
                
                signal = NewsSignal(
                    timestamp=timestamp,
                    source='macro_news',
                    headline=str(row.get('headline', '')),
                    summary=str(row.get('summary', '')),
                    sentiment_score=sentiment,
                    relevance_score=relevance,
                    importance_score=relevance * 0.8,  # Macro news slightly lower importance
                    ticker=self.ticker,
                    metadata={
                        'relevance_class': relevance_class,
                        'macro_category': row.get('category', 'general')
                    }
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing macro news row {idx}: {e}")
                continue
        
        self._macro_news.extend(signals)
        self._rebuild_all_signals()
        
        logger.info(f"Processed {len(signals)} macro news signals for {self.ticker}")
        return signals
    
    def process_sec_filings(
        self,
        filings: List[Dict[str, Any]],
        filter_results: Optional[List[Dict[str, Any]]] = None
    ) -> List[NewsSignal]:
        """
        Process SEC filings into NewsSignal objects.
        
        Args:
            filings: List of filing dictionaries with [filing_date, form_type, text, metrics]
            filter_results: Optional list of filter results with importance scores
        
        Returns:
            List of processed NewsSignal objects
        """
        if not filings:
            logger.debug("Empty SEC filings list")
            return []
        
        signals = []
        for idx, filing in enumerate(filings):
            try:
                # Parse filing date
                filing_date = filing.get('filing_date')
                if isinstance(filing_date, str):
                    filing_date = pd.to_datetime(filing_date)
                elif not isinstance(filing_date, datetime):
                    filing_date = datetime.now()
                
                # Get form type
                form_type = filing.get('form_type', 'UNKNOWN')
                
                # Get importance from filter results
                importance = 6.0  # Default medium-high for SEC filings
                if filter_results and idx < len(filter_results):
                    importance = filter_results[idx].get('importance_score', 6.0)
                
                # Boost importance for 10-K vs 10-Q
                if form_type == '10-K':
                    importance = max(importance, 8.0)
                elif form_type == '10-Q':
                    importance = max(importance, 6.0)
                
                # Extract headline and summary
                headline = f"{self.ticker} {form_type} Filing - {filing.get('fiscal_period', '')}"
                summary = filing.get('summary', filing.get('text', '')[:500])
                
                # Sentiment based on metrics (if available)
                sentiment = 0.0
                metrics = filing.get('metrics', {})
                if metrics:
                    revenue_growth = metrics.get('revenue_growth_yoy', 0)
                    if revenue_growth > 0.1:
                        sentiment = 0.5
                    elif revenue_growth < -0.1:
                        sentiment = -0.5
                
                signal = NewsSignal(
                    timestamp=filing_date,
                    source='sec_filing',
                    headline=headline,
                    summary=summary,
                    sentiment_score=sentiment,
                    relevance_score=importance,
                    importance_score=importance,
                    ticker=self.ticker,
                    metadata={
                        'form_type': form_type,
                        'fiscal_period': filing.get('fiscal_period', ''),
                        'cik': filing.get('cik', ''),
                        'metrics': metrics
                    }
                )
                signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Error processing SEC filing {idx}: {e}")
                continue
        
        self._sec_filings.extend(signals)
        self._rebuild_all_signals()
        
        logger.info(f"Processed {len(signals)} SEC filing signals for {self.ticker}")
        return signals
    
    def set_date_index(self, dates: pd.DatetimeIndex) -> None:
        """
        Set the date index for generating signal snapshots.
        
        Args:
            dates: DatetimeIndex of trading dates
        """
        self._date_index = dates
        logger.debug(f"Set date index with {len(dates)} dates for {self.ticker}")
    
    def get_signal_snapshot(self, date: datetime) -> SignalSnapshot:
        """
        Get aggregated signal snapshot for a specific date.
        
        Args:
            date: Target date for snapshot
        
        Returns:
            SignalSnapshot with aggregated metrics
        """
        # Filter signals up to and including this date
        cutoff = pd.Timestamp(date)
        cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
        
        # Get signals within sentiment window
        window_start = cutoff - timedelta(days=self.sentiment_window)
        
        relevant_signals = [
            s for s in self._all_signals
            if window_start <= (
                pd.Timestamp(s.timestamp).tz_localize('UTC')
                if pd.Timestamp(s.timestamp).tzinfo is None
                else pd.Timestamp(s.timestamp).tz_convert('UTC')
            ) <= cutoff
        ]
        
        # Categorize by source
        company_signals = [s for s in relevant_signals if s.source == 'company_news']
        macro_signals = [s for s in relevant_signals if s.source == 'macro_news']
        sec_signals = [s for s in relevant_signals if s.source == 'sec_filing']
        
        # Calculate aggregated metrics
        if relevant_signals:
            sentiments = [s.sentiment_score for s in relevant_signals]
            avg_sentiment = float(np.mean(sentiments))
            sentiment_std = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
            max_importance = max(s.importance_score for s in relevant_signals)
            
            # Calculate signal intensity with decay
            signal_intensity = self._calculate_signal_intensity(relevant_signals, cutoff)
        else:
            avg_sentiment = 0.0
            sentiment_std = 0.0
            max_importance = 0.0
            signal_intensity = 0.0
        
        snapshot = SignalSnapshot(
            timestamp=date,
            ticker=self.ticker,
            company_news_count=len(company_signals),
            macro_news_count=len(macro_signals),
            sec_filing_count=len(sec_signals),
            avg_sentiment=avg_sentiment,
            sentiment_std=sentiment_std,
            max_importance=max_importance,
            signal_intensity=signal_intensity,
            signals=relevant_signals
        )
        
        logger.debug(f"Generated signal snapshot for {date.date()}: "
                    f"{len(relevant_signals)} signals, intensity={signal_intensity:.2f}")
        
        return snapshot
    
    def get_all_snapshots(self) -> List[SignalSnapshot]:
        """
        Generate signal snapshots for all dates in the index.
        
        Returns:
            List of SignalSnapshot objects for each date
        """
        if self._date_index is None:
            logger.warning("No date index set, cannot generate snapshots")
            return []
        
        snapshots = []
        for date in self._date_index:
            snapshot = self.get_signal_snapshot(date.to_pydatetime())
            snapshots.append(snapshot)
        
        logger.info(f"Generated {len(snapshots)} signal snapshots for {self.ticker}")
        return snapshots
    
    def get_signals_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        source: Optional[str] = None
    ) -> List[NewsSignal]:
        """
        Get signals within a date range, optionally filtered by source.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            source: Optional source filter ('company_news', 'macro_news', 'sec_filing')
        
        Returns:
            List of NewsSignal objects in range
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        signals = [
            s for s in self._all_signals
            if start <= pd.Timestamp(s.timestamp) <= end
        ]
        
        if source:
            signals = [s for s in signals if s.source == source]
        
        # Sort by timestamp
        signals.sort(key=lambda s: s.timestamp)
        
        return signals
    
    def get_sentiment_series(self) -> pd.Series:
        """
        Get time series of daily average sentiment scores.
        
        Returns:
            pandas Series with date index and sentiment values
        """
        if not self._all_signals:
            return pd.Series(dtype=float)
        
        # Group by date and calculate mean sentiment
        data = []
        for signal in self._all_signals:
            date = pd.Timestamp(signal.timestamp).date()
            data.append((date, signal.sentiment_score, signal.importance_score))
        
        df = pd.DataFrame(data, columns=['date', 'sentiment', 'importance'])
        df['date'] = pd.to_datetime(df['date'])
        
        # Weight sentiment by importance
        df['weighted_sentiment'] = df['sentiment'] * df['importance']
        df['total_importance'] = df['importance']
        
        # Group by date
        daily = df.groupby('date').agg({
            'weighted_sentiment': 'sum',
            'total_importance': 'sum'
        }).reset_index()
        
        daily['avg_sentiment'] = daily['weighted_sentiment'] / daily['total_importance']
        daily = daily.set_index('date')['avg_sentiment']
        
        logger.debug(f"Generated sentiment series with {len(daily)} dates")
        return daily
    
    def update_signal_importance(
        self,
        signal_indices: List[int],
        new_importance: float
    ) -> None:
        """
        Update importance scores for specific signals.
        
        Args:
            signal_indices: Indices of signals to update in _all_signals
            new_importance: New importance score (1-10)
        """
        # Clamp importance to valid range
        new_importance = max(1.0, min(10.0, new_importance))
        
        updated = 0
        for idx in signal_indices:
            if 0 <= idx < len(self._all_signals):
                self._all_signals[idx].importance_score = new_importance
                updated += 1
        
        logger.debug(f"Updated importance for {updated} signals to {new_importance}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processed signals.
        
        Returns:
            Dictionary with signal statistics
        """
        if not self._all_signals:
            return {
                'total_signals': 0,
                'company_news': 0,
                'macro_news': 0,
                'sec_filings': 0,
                'avg_sentiment': 0.0,
                'avg_importance': 0.0
            }
        
        sentiments = [s.sentiment_score for s in self._all_signals]
        importances = [s.importance_score for s in self._all_signals]
        
        return {
            'total_signals': len(self._all_signals),
            'company_news': len(self._company_news),
            'macro_news': len(self._macro_news),
            'sec_filings': len(self._sec_filings),
            'avg_sentiment': float(np.mean(sentiments)),
            'sentiment_std': float(np.std(sentiments)),
            'avg_importance': float(np.mean(importances)),
            'max_importance': max(importances),
            'min_importance': min(importances)
        }
    
    def reset(self) -> None:
        """Reset all processed signals."""
        self._company_news = []
        self._macro_news = []
        self._sec_filings = []
        self._all_signals = []
        logger.info(f"Reset SignalProcessor for {self.ticker}")
    
    def _normalize_sentiment(self, sentiment: Any) -> float:
        """
        Normalize sentiment score to [-1, 1] range.
        
        Handles various sentiment formats:
        - Finnhub: -1 to 1 or 0 to 1
        - String labels: 'positive', 'negative', 'neutral'
        - Numeric: any range
        
        Args:
            sentiment: Raw sentiment value
        
        Returns:
            Normalized sentiment in [-1, 1]
        """
        if isinstance(sentiment, str):
            sentiment_lower = sentiment.lower()
            if 'positive' in sentiment_lower or 'bullish' in sentiment_lower:
                return 1.0
            elif 'negative' in sentiment_lower or 'bearish' in sentiment_lower:
                return -1.0
            else:
                return 0.0
        
        try:
            sentiment = float(sentiment)
        except (TypeError, ValueError):
            return 0.0
        
        # Normalize to [-1, 1]
        if sentiment > 1.0:
            # Assume 0-10 scale
            return (sentiment / 10.0) * 2 - 1
        elif sentiment < -1.0:
            # Assume -10 to 10 scale
            return max(-1.0, sentiment / 10.0)
        else:
            # Already in [-1, 1] or [0, 1]
            if sentiment >= 0:
                # Convert [0, 1] to [-1, 1]
                return sentiment * 2 - 1
            else:
                return sentiment
    
    def _calculate_signal_intensity(
        self,
        signals: List[NewsSignal],
        current_date: datetime
    ) -> float:
        """
        Calculate overall signal intensity with time decay.
        
        More recent signals have higher weight. Intensity is normalized to [0, 1].
        
        Args:
            signals: List of signals to consider
            current_date: Current date for decay calculation
        
        Returns:
            Signal intensity score (0-1)
        """
        if not signals:
            return 0.0
        
        total_weight = 0.0
        max_possible_weight = 0.0
        
        current = pd.Timestamp(current_date)
        current = current.tz_localize('UTC') if current.tzinfo is None else current.tz_convert('UTC')
        
        for signal in signals:
            # Calculate days ago
            signal_date = pd.Timestamp(signal.timestamp)
            signal_date = (
                signal_date.tz_localize('UTC')
                if signal_date.tzinfo is None
                else signal_date.tz_convert('UTC')
            )
            days_ago = (current - signal_date).days
            
            if days_ago < 0:
                days_ago = 0
            
            # Exponential decay
            weight = np.exp(-self.intensity_decay * days_ago)
            
            # Weight by importance
            weighted_importance = signal.importance_score * weight
            total_weight += weighted_importance
            
            # Max possible (importance=10, days_ago=0)
            max_possible_weight += 10.0 * weight
        
        if max_possible_weight == 0:
            return 0.0
        
        # Normalize to [0, 1]
        intensity = total_weight / max_possible_weight
        
        return min(1.0, intensity)
    
    def _rebuild_all_signals(self) -> None:
        """Rebuild combined signal list and sort by timestamp."""
        self._all_signals = (
            self._company_news +
            self._macro_news +
            self._sec_filings
        )
        # Sort by timestamp
        self._all_signals.sort(key=lambda s: s.timestamp)


def create_signal_processor(
    ticker: str,
    sentiment_window: int = 5,
    intensity_decay: float = 0.5,
    min_relevance_threshold: float = 3.0
) -> SignalProcessor:
    """
    Factory function to create configured SignalProcessor instance.
    
    Args:
        ticker: Target ticker symbol
        sentiment_window: Rolling window for sentiment aggregation (days)
        intensity_decay: Decay factor for older signals
        min_relevance_threshold: Minimum relevance score to include signal
    
    Returns:
        SignalProcessor: Configured signal processor instance
    """
    return SignalProcessor(
        ticker=ticker,
        sentiment_window=sentiment_window,
        intensity_decay=intensity_decay,
        min_relevance_threshold=min_relevance_threshold
    )


def process_market_signals(
    ticker: str,
    company_news_df: pd.DataFrame,
    macro_news_df: pd.DataFrame,
    sec_filings: List[Dict[str, Any]],
    dates: pd.DatetimeIndex,
    filter_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Tuple[SignalProcessor, List[SignalSnapshot]]:
    """
    Convenience function to process all market signals in one call.
    
    Args:
        ticker: Target ticker symbol
        company_news_df: Company news DataFrame
        macro_news_df: Macro news DataFrame
        sec_filings: List of SEC filing dictionaries
        dates: Trading date index
        filter_results: Optional dict with filter results by source
    
    Returns:
        Tuple of (SignalProcessor, List[SignalSnapshot])
    """
    processor = create_signal_processor(ticker)
    
    # Get filter results by source
    company_filters = None
    macro_filters = None
    sec_filters = None
    if filter_results:
        company_filters = filter_results.get('company_news')
        macro_filters = filter_results.get('macro_news')
        sec_filters = filter_results.get('sec_filings')
    
    # Process all signal sources
    processor.process_company_news(company_news_df, company_filters)
    processor.process_macro_news(macro_news_df, macro_filters)
    processor.process_sec_filings(sec_filings, sec_filters)
    
    # Set date index and generate snapshots
    processor.set_date_index(dates)
    snapshots = processor.get_all_snapshots()
    
    logger.info(f"Processed market signals for {ticker}: "
               f"{len(snapshots)} snapshots, {processor._all_signals} total signals")
    
    return processor, snapshots
