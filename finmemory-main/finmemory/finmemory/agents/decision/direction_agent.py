"""
Direction Decision Agent for FinMemory Trading System.

This module implements the Direction Agent that makes strategic buy/sell/hold decisions
based on working memory (all layers), price history, technical indicators, and market
sentiment from deterministic signal processing and the Market Context Agent.

Paper Reference: Section 4.2.1, Appendix A.2.1
"""

import logging
from typing import Any, Dict, List, Optional

from finmemory.agents.base_agent import BaseAgent
from finmemory.utils.prompt_manager import load_decision_prompt
from finmemory.utils.json_parser import validate_decision_output, validate_json_schema
from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


class DirectionAgent(BaseAgent):
    """
    Direction Decision Agent for making buy/sell/hold trading decisions.
    
    This agent analyzes market information from the hierarchical memory system,
    price history, and sentiment indicators to determine the optimal trading direction.
    
    Attributes:
        ticker (str): Target stock ticker symbol
        model_name (str): LLM model to use (default: gpt-4o)
        temperature (float): Sampling temperature (0.7 training, 0.3 evaluation)
        max_tokens (int): Maximum tokens in LLM response
        retry_attempts (int): Number of API retry attempts
        retry_delay (float): Delay between retries in seconds
    
    Input:
        - working_memory: Combined memory from shallow/intermediate/deep layers
        - price_history: Recent price data (OHLCV)
        - technical_indicators: MACD, RSI, moving averages, etc.
        - sentiment_indicators: Aggregated news sentiment scores
    
    Output:
        - investment_decision: "buy" | "sell" | "hold"
        - summary_reason: Brief explanation of decision rationale
        - memory_indices: List of memory item indices that influenced decision
        - reflection_analysis: Analysis of past similar decisions and outcomes
    
    Paper Reference: Section 4.2.1 (Dual Decision Architecture)
    """
    
    def __init__(
        self,
        ticker: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Direction Agent.
        
        Args:
            ticker: Target stock ticker symbol (e.g., 'TSLA', 'AAPL')
            model_name: OpenAI model name (default: 'gpt-4o')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            retry_attempts: Number of retry attempts on API failure
            retry_delay: Delay between retries in seconds
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay
        )
        self.ticker = ticker
        self.logger = logger
        self.logger.info(f"DirectionAgent initialized for {ticker}")
    
    def get_system_prompt(self) -> str:
        """
        Get system prompt for direction decision making.
        
        Returns:
            System prompt string defining the agent's role and output format
        """
        prompt_data = load_decision_prompt("direction_agent", ticker=self.ticker)
        return prompt_data.get("system_prompt", self._get_default_system_prompt())
    
    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt if YAML file not found.
        
        Returns:
            Default system prompt string
        """
        return f"""You are the Direction Decision Agent for the FinMemory trading system, 
specializing in {self.ticker} stock analysis.

YOUR ROLE:
- Analyze market information from multiple sources (news, SEC filings, macro events, price data)
- Make strategic buy/sell/hold decisions based on comprehensive analysis
- Consider short-term (1-day), mid-term (7-day), and long-term (30-day) trends
- Reference relevant memory items that support your decision
- Learn from past decisions through reflection analysis

OUTPUT REQUIREMENTS:
- Provide JSON output with exact schema specified
- investment_decision must be one of: "buy", "sell", "hold"
- summary_reason should be concise (2-3 sentences)
- memory_indices should reference specific memory item IDs
- reflection_analysis should note patterns from similar past situations

DECISION FRAMEWORK:
1. Evaluate current market sentiment from news and filings
2. Analyze price trends across multiple timescales
3. Consider technical indicators (momentum, volatility, support/resistance)
4. Review memory for similar historical patterns
5. Synthesize all signals into a clear directional decision

RISK AWARENESS:
- Be conservative during high volatility periods
- Consider position context (existing holdings)
- Avoid overconfidence in single signals
- Your decision will be passed to Quantity Agent for sizing"""
    
    def format_user_prompt(self, **kwargs) -> str:
        """
        Format user prompt with dynamic market data.
        
        Args:
            **kwargs: Dynamic variables including:
                - working_memory: Dict with shallow/intermediate/deep memory layers
                - price_history: List of recent price data points
                - technical_indicators: Dict with MACD, RSI, MA values
                - sentiment_indicators: Dict with sentiment scores
                - current_date: Current trading date
                - current_position: Current position size (optional)
                - current_holdings: Current number of shares held (optional)
        
        Returns:
            Formatted user prompt string
        """
        try:
            prompt_data = load_decision_prompt(
                "direction_agent",
                ticker=self.ticker,
                **kwargs
            )
            return prompt_data.get("user_template", self._get_default_user_prompt(kwargs))
        except Exception as e:
            self.logger.warning(f"Failed to load prompt template: {e}")
            return self._get_default_user_prompt(kwargs)
    
    def _get_default_user_prompt(self, kwargs: Dict[str, Any]) -> str:
        """
        Generate default user prompt if template loading fails.
        
        Args:
            kwargs: Dynamic variables for prompt
        
        Returns:
            Formatted user prompt string
        """
        working_memory = kwargs.get("working_memory", {})
        price_history = kwargs.get("price_history", [])
        technical_indicators = kwargs.get("technical_indicators", {})
        sentiment_indicators = kwargs.get("sentiment_indicators", {})
        current_date = kwargs.get("current_date", "Unknown")
        current_position = kwargs.get("current_position", 0)
        current_holdings = kwargs.get("current_holdings", 0)
        
        # Format memory summary
        memory_summary = self._format_memory_summary(working_memory)
        
        # Format price summary
        price_summary = self._format_price_summary(price_history)
        
        # Format technical indicators
        tech_summary = self._format_technical_summary(technical_indicators)
        
        # Format sentiment
        sentiment_summary = self._format_sentiment_summary(sentiment_indicators)
        
        return f"""CURRENT TRADING DECISION FOR {self.ticker}
Date: {current_date}

CURRENT POSITION:
- Position Size: {current_position}
- Shares Held: {current_holdings}

MARKET MEMORY (Hierarchical):
{memory_summary}

PRICE HISTORY (Recent):
{price_summary}

TECHNICAL INDICATORS:
{tech_summary}

SENTIMENT INDICATORS:
{sentiment_summary}

INSTRUCTION:
Based on all the above information, make a trading direction decision.
Provide your output in JSON format with the following structure:
{{
    "investment_decision": "buy" | "sell" | "hold",
    "summary_reason": "Brief explanation of your decision",
    "memory_indices": [list of memory item IDs that influenced this decision],
    "reflection_analysis": "Analysis of similar past decisions and their outcomes"
}}

Think carefully about:
1. Multi-timescale trends (1-day, 7-day, 30-day)
2. Risk level based on volatility and sentiment
3. Consistency with your memory of similar situations
4. Current position context"""
    
    def _format_memory_summary(self, working_memory: Dict[str, List[Dict]]) -> str:
        """Format working memory for prompt."""
        if not working_memory:
            return "No memory items available."
        
        lines = []
        for layer in ["shallow", "intermediate", "deep"]:
            items = working_memory.get(layer, working_memory.get(f"{layer}_items", []))
            if items:
                lines.append(f"{layer.capitalize()} Memory ({len(items)} items):")
                for i, item in enumerate(items[-5:]):  # Show last 5 items per layer
                    content = item.get("content", "")[:100]
                    importance = item.get("importance_score", "N/A")
                    lines.append(f"  [{i}] (importance: {importance}) {content}...")
        
        return "\n".join(lines) if lines else "No memory items available."
    
    def _format_price_summary(self, price_history: List[Dict]) -> str:
        """Format price history for prompt."""
        if not price_history:
            return "No price history available."
        
        # Show last 10 days
        recent = price_history[-10:] if len(price_history) > 10 else price_history
        
        lines = ["Date | Open | High | Low | Close | Volume"]
        lines.append("-" * 50)
        for day in recent:
            date = day.get("date", "N/A")
            open_p = day.get("open", 0)
            high = day.get("high", 0)
            low = day.get("low", 0)
            close = day.get("close", 0)
            volume = day.get("volume", 0)
            lines.append(f"{date} | {open_p:.2f} | {high:.2f} | {low:.2f} | {close:.2f} | {volume:,}")
        
        return "\n".join(lines)
    
    def _format_technical_summary(self, indicators: Dict) -> str:
        """Format technical indicators for prompt."""
        if not indicators:
            return "No technical indicators available."
        
        lines = []
        for key, value in indicators.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines) if lines else "No technical indicators available."
    
    def _format_sentiment_summary(self, sentiment: Dict) -> str:
        """Format sentiment indicators for prompt."""
        if not sentiment:
            return "No sentiment data available."
        
        lines = []
        for key, value in sentiment.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines) if lines else "No sentiment data available."
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate direction agent output against expected schema.
        
        Args:
            output: Parsed JSON output from LLM
        
        Returns:
            True if valid, False otherwise
        """
        is_valid, errors = validate_decision_output(output)
        
        if not is_valid:
            self.logger.warning(f"Direction output validation failed: {errors}")
            return False
        
        # Additional validation: decision must be one of buy/sell/hold
        decision = output.get("investment_decision", "").lower()
        if decision not in ["buy", "sell", "hold"]:
            self.logger.warning(f"Invalid investment_decision: {decision}")
            return False
        
        # Validate memory_indices is a list
        memory_indices = output.get("memory_indices", [])
        if not isinstance(memory_indices, list):
            self.logger.warning("memory_indices must be a list")
            return False
        
        self.logger.debug(f"Direction output validated successfully: {decision}")
        return True
    
    def decide_direction(
        self,
        working_memory: Dict[str, List[Dict]],
        price_history: List[Dict],
        technical_indicators: Dict[str, Any],
        sentiment_indicators: Dict[str, Any],
        current_date: str,
        current_position: int = 0,
        current_holdings: int = 0
    ) -> Dict[str, Any]:
        """
        Make a trading direction decision.
        
        Args:
            working_memory: Combined memory from all layers
            price_history: Recent OHLCV data
            technical_indicators: Technical analysis indicators
            sentiment_indicators: Sentiment scores from news/filings
            current_date: Current trading date
            current_position: Current position size (optional)
            current_holdings: Current shares held (optional)
        
        Returns:
            Dict with keys: investment_decision, summary_reason, memory_indices, reflection_analysis
        
        Raises:
            Exception: If LLM API call fails after retries
        """
        # Format user prompt with all market data
        user_prompt = self.format_user_prompt(
            working_memory=working_memory,
            price_history=price_history,
            technical_indicators=technical_indicators,
            sentiment_indicators=sentiment_indicators,
            current_date=current_date,
            current_position=current_position,
            current_holdings=current_holdings
        )
        
        # Get system prompt
        system_prompt = self.get_system_prompt()
        
        # Call LLM with retry logic
        self.logger.info(f"Calling LLM for direction decision on {current_date}")
        output = self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Validate output
        if not self.validate_output(output):
            self.logger.warning("Output validation failed, using fallback decision")
            return self._create_fallback_output(current_position)
        
        self.logger.info(f"Direction decision: {output.get('investment_decision')}")
        return output
    
    def _create_fallback_output(self, current_position: int) -> Dict[str, Any]:
        """
        Create fallback output when LLM fails.
        
        Args:
            current_position: Current position size
        
        Returns:
            Conservative fallback decision (typically 'hold')
        """
        self.logger.warning("Using fallback decision: hold")
        return {
            "investment_decision": "hold",
            "summary_reason": "Fallback decision due to analysis failure. Maintaining current position.",
            "memory_indices": [],
            "reflection_analysis": "No reflection available - fallback mode activated"
        }
    
    def decide_batch(
        self,
        decisions_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple direction decisions (for batch processing).
        
        Args:
            decisions_data: List of decision input dictionaries
        
        Returns:
            List of direction decision outputs
        """
        results = []
        for i, data in enumerate(decisions_data):
            try:
                result = self.decide_direction(**data)
                results.append(result)
                self.logger.info(f"Batch decision {i+1}/{len(decisions_data)} completed")
            except Exception as e:
                self.logger.error(f"Batch decision {i+1} failed: {e}")
                results.append(self._create_fallback_output(data.get("current_position", 0)))
        
        return results


def create_direction_agent(
    ticker: str,
    mode: str = "evaluation"
) -> DirectionAgent:
    """
    Factory function to create DirectionAgent with mode-appropriate settings.
    
    Args:
        ticker: Target stock ticker symbol
        mode: "training" (temperature=0.7) or "evaluation" (temperature=0.3)
    
    Returns:
        Configured DirectionAgent instance
    """
    temperature = 0.7 if mode == "training" else 0.3
    logger.info(f"Creating DirectionAgent for {ticker} in {mode} mode (temp={temperature})")
    
    return DirectionAgent(
        ticker=ticker,
        model_name="gpt-4o",
        temperature=temperature,
        max_tokens=1000,
        retry_attempts=3,
        retry_delay=1.0
    )
