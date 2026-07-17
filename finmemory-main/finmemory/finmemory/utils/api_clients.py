"""
API Clients for FinMemory Trading System

Provides wrapper classes for external API integrations:
- OpenAI API: GPT-4o access for LLM agents
- Finnhub API: Company and macroeconomic news
- SEC EDGAR API: 10-K and 10-Q filings

Implements rate limiting, error handling, and response caching to ensure
reliable API access within free tier limits.
"""

import os
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from pathlib import Path

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


class APIRateLimiter:
    """
    Rate limiter for API calls with exponential backoff.
    
    Tracks call counts and enforces rate limits to prevent API throttling.
    Implements exponential backoff for retry logic.
    """
    
    def __init__(self, calls_per_minute: int = 60, calls_per_month: int = 30000):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls allowed per minute
            calls_per_month: Maximum calls allowed per month
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_month = calls_per_month
        self.minute_calls: List[float] = []
        self.monthly_call_count = 0
        self.last_reset = time.time()
        
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Clean old calls (older than 1 minute)
        self.minute_calls = [t for t in self.minute_calls if now - t < 60]
        
        # Check minute limit
        if len(self.minute_calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.minute_calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.minute_calls = []  # Reset after sleep
        
        # Check monthly limit
        if self.monthly_call_count >= self.calls_per_month:
            raise RuntimeError(f"Monthly API call limit exceeded: {self.monthly_call_count}/{self.calls_per_month}")
        
        # Record this call
        self.minute_calls.append(now)
        self.monthly_call_count += 1
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        now = time.time()
        recent_calls = len([t for t in self.minute_calls if now - t < 60])
        
        return {
            "calls_last_minute": recent_calls,
            "calls_per_minute_limit": self.calls_per_minute,
            "monthly_calls": self.monthly_call_count,
            "monthly_limit": self.calls_per_month,
            "utilization_pct": (self.monthly_call_count / self.calls_per_month) * 100
        }
        
    def reset(self) -> None:
        """Reset all counters."""
        self.minute_calls = []
        self.monthly_call_count = 0
        self.last_reset = time.time()


class APICache:
    """
    Simple in-memory cache for API responses with TTL support.
    
    Caches API responses to avoid redundant calls and reduce costs.
    Uses MD5 hashing of request parameters for cache keys.
    """
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
            max_size: Maximum number of cache entries
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        
    def _generate_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and parameters."""
        key_string = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached response if available and not expired.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(endpoint, params)
        
        if key not in self._cache:
            return None
            
        # Check TTL
        if time.time() - self._access_times[key] > self.ttl_seconds:
            del self._cache[key]
            del self._access_times[key]
            return None
            
        self._access_times[key] = time.time()
        return self._cache[key]
        
    def set(self, endpoint: str, params: Dict[str, Any], response: Any) -> None:
        """
        Cache API response.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            response: API response to cache
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
            
        key = self._generate_key(endpoint, params)
        self._cache[key] = response
        self._access_times[key] = time.time()
        
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_times.clear()
        
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class OpenAIClient:
    """
    Client for OpenAI API (GPT-4o) with rate limiting and error handling.
    
    Provides methods for chat completions with retry logic and cost tracking.
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit_calls_per_minute: int = 500):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            rate_limit_calls_per_minute: Rate limit for API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
            
        self.base_url = "https://api.openai.com/v1"
        self.rate_limiter = APIRateLimiter(calls_per_minute=rate_limit_calls_per_minute)
        self.cache = APICache(ttl_seconds=300)  # 5 minute cache for API calls
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # GPT-4o pricing (approximate)
        self.prompt_price_per_1k = 0.005  # $5 per 1M tokens
        self.completion_price_per_1k = 0.015  # $15 per 1M tokens
        
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Call OpenAI chat completion API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (default: gpt-4o)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            response_format: Format specification (e.g., {"type": "json_object"})
            use_cache: Whether to use response caching
            
        Returns:
            API response dict with completion text and metadata
        """
        # Check cache
        if use_cache:
            params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            cached = self.cache.get("chat/completions", params)
            if cached:
                logger.debug("Using cached OpenAI response")
                return cached
                
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
            
        # Make API call with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract usage and cost
                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                
                self.total_tokens += prompt_tokens + completion_tokens
                self.total_cost += (prompt_tokens * self.prompt_price_per_1k / 1000)
                self.total_cost += (completion_tokens * self.completion_price_per_1k / 1000)
                
                # Cache response
                if use_cache:
                    self.cache.set("chat/completions", params, result)
                    
                logger.debug(f"OpenAI API call: {prompt_tokens} prompt, {completion_tokens} completion tokens")
                return result
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API call failed after {max_retries} attempts: {e}")
                    raise
                    
        raise RuntimeError("OpenAI API call failed")
        
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get token usage and cost summary."""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "prompt_price_per_1k": self.prompt_price_per_1k,
            "completion_price_per_1k": self.completion_price_per_1k
        }
        
    def reset_cost_tracking(self) -> None:
        """Reset cost tracking counters."""
        self.total_tokens = 0
        self.total_cost = 0.0


class FinnhubClient:
    """
    Client for Finnhub API with rate limiting and caching.
    
    Provides methods for company news, macro news, and stock data.
    Free tier: 60 calls/minute, 30,000 calls/month
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub client.
        
        Args:
            api_key: Finnhub API key (falls back to FINNHUB_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.warning("Finnhub API key not provided. Some features may not work.")
            self.api_key = "demo"  # Use demo key with limited access
            
        self.base_url = "https://finnhub.io/api/v1"
        self.rate_limiter = APIRateLimiter(calls_per_minute=60, calls_per_month=30000)
        self.cache = APICache(ttl_seconds=3600)  # 1 hour cache
        
    def get_company_news(
        self,
        symbol: str,
        from_date: str,
        to_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get company news for a specific symbol and date range.
        
        Args:
            symbol: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of news articles with headline, summary, datetime, etc.
        """
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self.api_key
        }
        
        # Check cache
        cached = self.cache.get("company-news", params)
        if cached:
            logger.debug(f"Using cached Finnhub news for {symbol}")
            return cached
            
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.base_url}/company-news",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache response
            self.cache.set("company-news", params, result)
            
            logger.info(f"Fetched {len(result)} news articles for {symbol}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub company news API error for {symbol}: {e}")
            return []
            
    def get_general_news(
        self,
        topic: str = "general",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get general/macroeconomic news.
        
        Args:
            topic: News topic (general, forex, crypto, merger)
            limit: Maximum number of articles to return
            
        Returns:
            List of news articles
        """
        params = {
            "topic": topic,
            "token": self.api_key
        }
        
        # Check cache
        cached = self.cache.get("general-news", params)
        if cached:
            logger.debug(f"Using cached Finnhub general news")
            return cached[:limit]
            
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.base_url}/news",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache response
            self.cache.set("general-news", params, result)
            
            logger.info(f"Fetched {len(result)} general news articles")
            return result[:limit]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub general news API error: {e}")
            return []
            
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current stock quote.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Quote dict with current price, change, etc.
        """
        params = {
            "symbol": symbol,
            "token": self.api_key
        }
        
        # Check cache (quotes expire quickly)
        cached = self.cache.get("quote", params)
        if cached:
            return cached
            
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.base_url}/quote",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache for 5 minutes only
            self.cache.set("quote", params, result)
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub quote API error for {symbol}: {e}")
            return None
            
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        rate_stats = self.rate_limiter.get_usage_stats()
        cache_stats = self.cache.get_stats()
        
        return {
            **rate_stats,
            "cache": cache_stats
        }


class SECClient:
    """
    Client for SEC EDGAR API for 10-K and 10-Q filings.
    
    No API key required - public access with rate limiting.
    Implements respectful crawling with delays between requests.
    """
    
    def __init__(self, user_agent: str = "FinMemory Research Agent"):
        """
        Initialize SEC EDGAR client.
        
        Args:
            user_agent: User agent string (required by SEC)
        """
        self.base_url = "https://data.sec.gov"
        self.search_url = "https://search.sec.gov"
        self.user_agent = user_agent
        self.rate_limiter = APIRateLimiter(calls_per_minute=10)  # Conservative rate limit
        self.cache = APICache(ttl_seconds=86400)  # 24 hour cache (filings don't change)
        
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate"
        }
        
    def get_company_facts(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Get company facts from SEC EDGAR.
        
        Args:
            cik: Company CIK number (10 digits, zero-padded)
            
        Returns:
            Company facts dict with financial data
        """
        # Ensure CIK is 10 digits
        cik = cik.zfill(10)
        
        params = {"cik": cik}
        
        # Check cache
        cached = self.cache.get("company-facts", params)
        if cached:
            logger.debug(f"Using cached SEC company facts for CIK {cik}")
            return cached
            
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.base_url}/api/xbrl/companyfacts/CIK{cik}.json",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache response
            self.cache.set("company-facts", params, result)
            
            logger.info(f"Fetched SEC company facts for CIK {cik}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC company facts API error for CIK {cik}: {e}")
            return None
            
    def get_submissions(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Get company filing submissions from SEC EDGAR.
        
        Args:
            cik: Company CIK number
            
        Returns:
            Submissions dict with filing metadata
        """
        cik = cik.zfill(10)
        params = {"cik": cik}
        
        # Check cache
        cached = self.cache.get("submissions", params)
        if cached:
            return cached
            
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            response = requests.get(
                f"{self.base_url}/submissions/CIK{cik}.json",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            self.cache.set("submissions", params, result)
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC submissions API error for CIK {cik}: {e}")
            return None
            
    def get_filing_text(self, cik: str, accession_number: str) -> Optional[str]:
        """
        Get full filing text by accession number.
        
        Args:
            accession_number: SEC accession number (e.g., "0001564590-23-000123")
            
        Returns:
            Full filing text (HTML or plain text)
        """
        # Format accession number for URL
        acc_no_dash = accession_number.replace("-", "")
        cik = str(int(cik))
        params = {"cik": cik, "accession_number": accession_number}
        
        # Check cache
        cached = self.cache.get("filing-text", params)
        if cached:
            return cached
            
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        try:
            url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{cik}/{acc_no_dash}/{acc_no_dash}.txt"
            )
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            text = response.text
            
            # Cache response
            self.cache.set("filing-text", params, text)
            
            return text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"SEC filing text API error for {accession_number}: {e}")
            return None
            
    def search_filings(
        self,
        ticker: str,
        form_type: str = "10-K",
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for filings by ticker and form type.
        
        Args:
            ticker: Stock ticker symbol
            form_type: Form type (10-K, 10-Q, 8-K, etc.)
            count: Maximum number of results
            
        Returns:
            List of filing metadata dicts
        """
        params = {
            "ticker": ticker,
            "form_type": form_type,
            "count": count
        }
        
        # Check cache
        cached = self.cache.get("search-filings", params)
        if cached:
            return cached
            
        # Get CIK from ticker first
        cik = self._get_cik_from_ticker(ticker)
        if not cik:
            logger.warning(f"Could not find CIK for ticker {ticker}")
            return []
            
        # Get submissions
        submissions = self.get_submissions(cik)
        if not submissions:
            return []
            
        # Filter by form type
        filings = []
        for filing in submissions.get("filings", {}).get("recent", {}):
            if len(filings) >= count:
                break
                
            form = filing.get("form", "")
            if form_type in form:
                filings.append({
                    "accession_number": filing.get("accessionNumber"),
                    "filing_date": filing.get("filingDate"),
                    "report_date": filing.get("reportDate"),
                    "form_type": form,
                    "file_number": filing.get("fileNumber"),
                    "film_number": filing.get("filmNumber")
                })
                
        # Cache results
        self.cache.set("search-filings", params, filings)
        
        logger.info(f"Found {len(filings)} {form_type} filings for {ticker}")
        return filings
        
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK number from ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK number or None if not found
        """
        # Use SEC company tickers file
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        
        try:
            response = requests.get(tickers_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            tickers_data = response.json()
            
            # Search for ticker
            for data in tickers_data.values():
                if data.get("ticker", "").upper() == ticker.upper():
                    return str(data["cik_str"]).zfill(10)
                    
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching SEC tickers: {e}")
            return None


# Convenience functions for creating clients

def create_openai_client(api_key: Optional[str] = None) -> OpenAIClient:
    """Create OpenAI client with optional API key."""
    return OpenAIClient(api_key=api_key)


def create_finnhub_client(api_key: Optional[str] = None) -> FinnhubClient:
    """Create Finnhub client with optional API key."""
    return FinnhubClient(api_key=api_key)


def create_sec_client(user_agent: str = "FinMemory Research Agent") -> SECClient:
    """Create SEC EDGAR client."""
    return SECClient(user_agent=user_agent)
