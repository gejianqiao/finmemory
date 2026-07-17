"""
SEC EDGAR Loader for FinMemory Trading System

Downloads and caches SEC filings (10-K annual reports, 10-Q quarterly reports) 
from the SEC EDGAR API for fundamental analysis in the FinMemory trading agent system.

Features:
- Company facts and submission history retrieval
- 10-K and 10-Q filing text extraction
- Dual-layer caching (memory + disk) to minimize API calls
- Data validation and parsing of key financial metrics
- Rate limit compliance (SEC allows 10 requests/second)

Author: FinMemory Research Team
Date: 2024
"""

import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import requests

from finmemory.utils.logger import get_logger
from finmemory.utils.api_clients import SECClient, APICache

logger = get_logger(__name__)


class SECFilingLoader:
    """
    SEC EDGAR filing loader for retrieving 10-K and 10-Q reports.
    
    Implements dual-layer caching (memory + disk) to minimize API calls
    and ensure reliable filing data collection within SEC rate limits.
    
    Attributes:
        data_dir: Directory for storing cached filing data
        cache_enabled: Whether to use caching
        cache_ttl: Cache time-to-live in seconds
        sec_client: SEC API client instance
        _memory_cache: In-memory cache for recent filings
    """
    
    def __init__(
        self,
        data_dir: str = "data/raw/reports",
        cache_enabled: bool = True,
        cache_ttl: int = 86400,  # 24 hours for SEC data
        user_agent: str = "FinMemory Research Agent research@finmemory.example.com"
    ):
        """
        Initialize SEC Filing Loader.
        
        Args:
            data_dir: Directory for storing cached filing data
            cache_enabled: Whether to enable caching (default: True)
            cache_ttl: Cache TTL in seconds (default: 86400 = 24 hours)
            user_agent: User agent string for SEC API (required by EDGAR)
        """
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.sec_client = SECClient(user_agent=user_agent)
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for recent filings
        self._memory_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        logger.info(f"SEC Filing Loader initialized with data_dir={self.data_dir}")
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a company ticker symbol.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            CIK number as string (zero-padded) or None if not found
        """
        cache_key = f"cik_{ticker.upper()}"
        
        # Check memory cache
        if self.cache_enabled and cache_key in self._memory_cache:
            if self._is_cache_valid(cache_key):
                return self._memory_cache[cache_key]
        
        # Check disk cache
        cik_file = self.data_dir / f"{ticker.upper()}_cik.json"
        if self.cache_enabled and cik_file.exists():
            try:
                with open(cik_file, 'r') as f:
                    data = json.load(f)
                    cik = data.get('cik')
                    if cik:
                        self._update_cache(cache_key, cik)
                        return cik
            except Exception as e:
                logger.warning(f"Error reading CIK cache for {ticker}: {e}")
        
        # Resolve ticker through the SEC-maintained ticker registry. Company
        # facts endpoints require a numeric CIK and cannot accept a ticker.
        try:
            cik = self.sec_client._get_cik_from_ticker(ticker)
            if cik:
                self._update_cache(cache_key, cik)
                
                # Save to disk cache
                if self.cache_enabled:
                    with open(cik_file, 'w') as f:
                        json.dump({'cik': cik, 'ticker': ticker}, f)
                
                return cik
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
        
        # Fallback: try common CIK mappings for major companies
        cik_mapping = {
            'AAPL': '0000320193',
            'TSLA': '0001318605',
            'AMZN': '0001018724',
            'NFLX': '0001065280',
            'COIN': '0001679788',
            'MSFT': '0000789019',
            'GOOGL': '0001652044',
            'META': '0001326801',
            'NVDA': '0001045810',
            'SPY': '0000884394'
        }
        
        if ticker.upper() in cik_mapping:
            cik = cik_mapping[ticker.upper()]
            self._update_cache(cache_key, cik)
            if self.cache_enabled:
                with open(cik_file, 'w') as f:
                    json.dump({'cik': cik, 'ticker': ticker}, f)
            return cik
        
        logger.warning(f"Could not find CIK for ticker: {ticker}")
        return None
    
    def get_submissions(self, ticker: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Get filing submission history for a company.
        
        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of submissions to return (default: 100)
            
        Returns:
            List of filing submissions or None if error
        """
        cik = self.get_company_cik(ticker)
        if not cik:
            logger.error(f"Cannot get submissions: CIK not found for {ticker}")
            return None
        
        cache_key = f"submissions_{ticker}_{limit}"
        
        # Check memory cache
        if self.cache_enabled and cache_key in self._memory_cache:
            if self._is_cache_valid(cache_key):
                return self._memory_cache[cache_key]
        
        # Check disk cache
        submissions_file = self.data_dir / f"{ticker}_submissions.json"
        if self.cache_enabled and submissions_file.exists():
            try:
                with open(submissions_file, 'r') as f:
                    data = json.load(f)
                    filings = data.get('filings', [])
                    if filings:
                        self._update_cache(cache_key, filings)
                        return filings[:limit]
            except Exception as e:
                logger.warning(f"Error reading submissions cache for {ticker}: {e}")
        
        # Fetch from SEC API
        try:
            submissions = self.sec_client.get_submissions(cik)
            recent = submissions.get('filings', {}).get('recent', {}) if submissions else {}
            if recent:
                # SEC returns a column-oriented dictionary; convert it into
                # one ordinary dictionary per filing for the filtering code.
                keys = list(recent.keys())
                row_count = len(recent.get('accessionNumber', []))
                recent_filings = [
                    {key: recent[key][i] for key in keys if i < len(recent[key])}
                    for i in range(min(row_count, limit))
                ]
                self._update_cache(cache_key, recent_filings)
                
                # Save to disk cache
                if self.cache_enabled:
                    with open(submissions_file, 'w') as f:
                        json.dump({'filings': recent_filings, 'timestamp': datetime.now().isoformat()}, f)
                
                return recent_filings
        except Exception as e:
            logger.error(f"Error fetching submissions for {ticker}: {e}")
        
        return None
    
    def get_10k_filings(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get 10-K annual report filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            limit: Maximum number of filings to return (default: 10)
            
        Returns:
            List of 10-K filing metadata
        """
        submissions = self.get_submissions(ticker, limit=limit * 3)  # Get more to filter
        if not submissions:
            return []
        
        # Filter for 10-K filings
        ten_k_filings = []
        for filing in submissions:
            if filing.get('form') == '10-K':
                filing_date = filing.get('reportDate', filing.get('filingDate', ''))
                
                # Apply date filters
                if start_date and filing_date < start_date:
                    continue
                if end_date and filing_date > end_date:
                    continue
                
                ten_k_filings.append(filing)
                
                if len(ten_k_filings) >= limit:
                    break
        
        logger.info(f"Found {len(ten_k_filings)} 10-K filings for {ticker}")
        return ten_k_filings
    
    def get_10q_filings(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get 10-Q quarterly report filings for a company.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            limit: Maximum number of filings to return (default: 20)
            
        Returns:
            List of 10-Q filing metadata
        """
        submissions = self.get_submissions(ticker, limit=limit * 3)  # Get more to filter
        if not submissions:
            return []
        
        # Filter for 10-Q filings
        ten_q_filings = []
        for filing in submissions:
            if filing.get('form') == '10-Q':
                filing_date = filing.get('reportDate', filing.get('filingDate', ''))
                
                # Apply date filters
                if start_date and filing_date < start_date:
                    continue
                if end_date and filing_date > end_date:
                    continue
                
                ten_q_filings.append(filing)
                
                if len(ten_q_filings) >= limit:
                    break
        
        logger.info(f"Found {len(ten_q_filings)} 10-Q filings for {ticker}")
        return ten_q_filings
    
    def get_filing_text(self, ticker: str, accession_number: str) -> Optional[str]:
        """
        Get full text of a SEC filing.
        
        Args:
            ticker: Stock ticker symbol
            accession_number: SEC accession number for the filing
            
        Returns:
            Full filing text or None if error
        """
        cache_key = f"filing_{ticker}_{accession_number.replace('-', '')}"
        
        # Check memory cache
        if self.cache_enabled and cache_key in self._memory_cache:
            if self._is_cache_valid(cache_key):
                return self._memory_cache[cache_key]
        
        # Check disk cache
        filing_file = self.data_dir / f"{ticker}_{accession_number.replace('-', '')}.txt"
        if self.cache_enabled and filing_file.exists():
            try:
                with open(filing_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text:
                        self._update_cache(cache_key, text)
                        return text
            except Exception as e:
                logger.warning(f"Error reading filing cache: {e}")
        
        # Fetch from SEC API
        try:
            cik = self.get_company_cik(ticker)
            if not cik:
                return None
            
            text = self.sec_client.get_filing_text(cik, accession_number)
            if text:
                self._update_cache(cache_key, text)
                
                # Save to disk cache
                if self.cache_enabled:
                    with open(filing_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                
                return text
        except Exception as e:
            logger.error(f"Error fetching filing text: {e}")
        
        return None
    
    def download_10k_text(
        self,
        ticker: str,
        filing_date: Optional[str] = None,
        fiscal_year_end: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download 10-K filing text for a specific period.
        
        Args:
            ticker: Stock ticker symbol
            filing_date: Specific filing date to retrieve (optional)
            fiscal_year_end: Fiscal year end date (optional)
            
        Returns:
            Dictionary with filing metadata and text, or None if error
        """
        # Get 10-K filings
        filings = self.get_10k_filings(ticker, limit=10)
        if not filings:
            logger.warning(f"No 10-K filings found for {ticker}")
            return None
        
        # Find matching filing
        target_filing = None
        if filing_date:
            for filing in filings:
                if filing.get('reportDate') == filing_date or filing.get('filingDate') == filing_date:
                    target_filing = filing
                    break
        elif fiscal_year_end:
            for filing in filings:
                if filing.get('reportDate', '').startswith(fiscal_year_end.split('-')[0]):
                    target_filing = filing
                    break
        
        # Use most recent if no specific date
        if not target_filing and filings:
            target_filing = filings[0]
        
        if not target_filing:
            logger.warning(f"No matching 10-K filing found for {ticker}")
            return None
        
        # Get filing text
        accession_number = target_filing.get('accessionNumber', '').replace('-', '')
        filing_text = self.get_filing_text(ticker, target_filing.get('accessionNumber', ''))
        
        if not filing_text:
            logger.warning(f"Could not retrieve filing text for {ticker}")
            return None
        
        return {
            'ticker': ticker,
            'form_type': '10-K',
            'accession_number': target_filing.get('accessionNumber'),
            'filing_date': target_filing.get('filingDate'),
            'report_date': target_filing.get('reportDate'),
            'fiscal_year_end': target_filing.get('reportDate', '')[:10],
            'cik': target_filing.get('cik'),
            'text': filing_text,
            'file_number': target_filing.get('fileNumber'),
            'film_number': target_filing.get('filmNumber')
        }
    
    def download_10q_text(
        self,
        ticker: str,
        filing_date: Optional[str] = None,
        quarter_end: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Download 10-Q filing text for a specific period.
        
        Args:
            ticker: Stock ticker symbol
            filing_date: Specific filing date to retrieve (optional)
            quarter_end: Quarter end date (optional)
            
        Returns:
            Dictionary with filing metadata and text, or None if error
        """
        # Get 10-Q filings
        filings = self.get_10q_filings(ticker, limit=20)
        if not filings:
            logger.warning(f"No 10-Q filings found for {ticker}")
            return None
        
        # Find matching filing
        target_filing = None
        if filing_date:
            for filing in filings:
                if filing.get('reportDate') == filing_date or filing.get('filingDate') == filing_date:
                    target_filing = filing
                    break
        elif quarter_end:
            for filing in filings:
                if filing.get('reportDate', '').startswith(quarter_end.replace('-', '')[:6]):
                    target_filing = filing
                    break
        
        # Use most recent if no specific date
        if not target_filing and filings:
            target_filing = filings[0]
        
        if not target_filing:
            logger.warning(f"No matching 10-Q filing found for {ticker}")
            return None
        
        # Get filing text
        accession_number = target_filing.get('accessionNumber', '').replace('-', '')
        filing_text = self.get_filing_text(ticker, target_filing.get('accessionNumber', ''))
        
        if not filing_text:
            logger.warning(f"Could not retrieve filing text for {ticker}")
            return None
        
        return {
            'ticker': ticker,
            'form_type': '10-Q',
            'accession_number': target_filing.get('accessionNumber'),
            'filing_date': target_filing.get('filingDate'),
            'report_date': target_filing.get('reportDate'),
            'fiscal_quarter_end': target_filing.get('reportDate', '')[:10],
            'cik': target_filing.get('cik'),
            'text': filing_text,
            'file_number': target_filing.get('fileNumber'),
            'film_number': target_filing.get('filmNumber')
        }
    
    def download_all_filings(
        self,
        tickers: List[str],
        start_date: str = "2024-01-01",
        end_date: str = "2025-09-30",
        include_10k: bool = True,
        include_10q: bool = True
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Download all SEC filings for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date for filings
            end_date: End date for filings
            include_10k: Whether to include 10-K filings
            include_10q: Whether to include 10-Q filings
            
        Returns:
            Nested dictionary: {ticker: {'10-K': [...], '10-Q': [...]}}
        """
        all_filings = {}
        
        for ticker in tickers:
            logger.info(f"Downloading SEC filings for {ticker}...")
            ticker_filings = {'10-K': [], '10-Q': []}
            
            # Download 10-K filings
            if include_10k:
                ten_k_filings = self.get_10k_filings(ticker, start_date, end_date, limit=5)
                for filing_meta in ten_k_filings:
                    filing_data = self.download_10k_text(
                        ticker,
                        filing_date=filing_meta.get('filingDate')
                    )
                    if filing_data:
                        ticker_filings['10-K'].append(filing_data)
                        logger.info(f"  Downloaded 10-K for {ticker} ({filing_meta.get('reportDate')})")
                    time.sleep(0.2)  # Rate limiting
            
            # Download 10-Q filings
            if include_10q:
                ten_q_filings = self.get_10q_filings(ticker, start_date, end_date, limit=10)
                for filing_meta in ten_q_filings:
                    filing_data = self.download_10q_text(
                        ticker,
                        filing_date=filing_meta.get('filingDate')
                    )
                    if filing_data:
                        ticker_filings['10-Q'].append(filing_data)
                        logger.info(f"  Downloaded 10-Q for {ticker} ({filing_meta.get('reportDate')})")
                    time.sleep(0.2)  # Rate limiting
            
            all_filings[ticker] = ticker_filings
            logger.info(f"Completed {ticker}: {len(ticker_filings['10-K'])} 10-K, {len(ticker_filings['10-Q'])} 10-Q")
            time.sleep(0.5)  # Rate limiting between tickers
        
        return all_filings
    
    def validate_filing_data(self, filing_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate SEC filing data for completeness and quality.
        
        Args:
            filing_data: Filing data dictionary
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check required fields
        required_fields = ['ticker', 'form_type', 'accession_number', 'text']
        for field in required_fields:
            if field not in filing_data:
                issues.append(f"Missing required field: {field}")
        
        # Check text length
        text = filing_data.get('text', '')
        if not text:
            issues.append("Empty filing text")
        elif len(text) < 1000:
            issues.append(f"Filing text too short: {len(text)} chars")
        elif len(text) > 500000:
            issues.append(f"Filing text very long: {len(text)} chars (may cause token issues)")
        
        # Check form type
        form_type = filing_data.get('form_type', '')
        if form_type not in ['10-K', '10-Q']:
            issues.append(f"Unexpected form type: {form_type}")
        
        # Check dates
        filing_date = filing_data.get('filing_date', '')
        if not filing_date:
            issues.append("Missing filing date")
        
        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(f"Filing validation failed for {filing_data.get('ticker', 'UNKNOWN')}: {issues}")
        
        return is_valid, issues
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached item is still valid based on TTL."""
        if cache_key not in self._cache_timestamps:
            return False
        
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age.total_seconds() < self.cache_ttl
    
    def _update_cache(self, cache_key: str, value: Any) -> None:
        """Update memory cache with new value and timestamp."""
        if self.cache_enabled:
            self._memory_cache[cache_key] = value
            self._cache_timestamps[cache_key] = datetime.now()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cache and API usage statistics."""
        return {
            'memory_cache_size': len(self._memory_cache),
            'memory_cache_keys': list(self._memory_cache.keys()),
            'data_directory': str(self.data_dir),
            'cache_enabled': self.cache_enabled,
            'cache_ttl_seconds': self.cache_ttl
        }
    
    def clear_cache(self) -> None:
        """Clear memory cache (disk cache preserved)."""
        self._memory_cache.clear()
        self._cache_timestamps.clear()
        logger.info("SEC loader memory cache cleared")


def create_sec_loader(
    data_dir: str = "data/raw/reports",
    cache_enabled: bool = True,
    cache_ttl: int = 86400,
    user_agent: str = "FinMemory Research Agent research@finmemory.example.com"
) -> SECFilingLoader:
    """
    Factory function to create configured SEC Filing Loader instance.
    
    Args:
        data_dir: Directory for storing cached filing data
        cache_enabled: Whether to enable caching
        cache_ttl: Cache TTL in seconds
        user_agent: User agent string for SEC API
        
    Returns:
        SECFilingLoader: Configured loader instance
    """
    return SECFilingLoader(
        data_dir=data_dir,
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
        user_agent=user_agent
    )


def download_sec_filings(
    tickers: List[str],
    start_date: str = "2024-01-01",
    end_date: str = "2025-09-30",
    data_dir: str = "data/raw/reports",
    include_10k: bool = True,
    include_10q: bool = True
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Convenience function to download SEC filings for multiple tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for filings (YYYY-MM-DD)
        end_date: End date for filings (YYYY-MM-DD)
        data_dir: Directory for storing cached data
        include_10k: Whether to include 10-K filings
        include_10q: Whether to include 10-Q filings
        
    Returns:
        Nested dictionary: {ticker: {'10-K': [...], '10-Q': [...]}}
    """
    loader = create_sec_loader(data_dir=data_dir)
    return loader.download_all_filings(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        include_10k=include_10k,
        include_10q=include_10q
    )
