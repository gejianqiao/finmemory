"""
Quantity & Risk Decision Agent for FinMemory Trading System.

This module implements the Quantity/Risk Agent that determines optimal position sizing
based on CVaR constraints, available capital, and the strategic direction from the
Direction Agent. It enforces risk discipline independently to prevent overconfidence
in strong signals from violating risk constraints.

Key Features:
- CVaR-constrained position sizing
- Cash availability validation
- Risk-aware order size determination
- Integration with hierarchical memory system
"""

import logging
from typing import Any, Dict, List, Optional

from finmemory.agents.base_agent import BaseAgent
from finmemory.utils.prompt_manager import load_decision_prompt
from finmemory.utils.json_parser import validate_quantity_output, validate_json_schema
from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


class QuantityRiskAgent(BaseAgent):
    """
    Quantity & Risk Decision Agent for position sizing with CVaR constraints.
    
    This agent receives the strategic direction (buy/sell/hold) from the Direction Agent
    and determines the optimal order size while respecting:
    1. CVaR-based maximum position constraints (maxcvar)
    2. Available cash constraints
    3. Current holdings and portfolio exposure
    
    The separation from direction decision ensures independent risk discipline.
    
    Attributes:
        ticker (str): Stock ticker symbol for this agent
        model_name (str): LLM model to use (default: "gpt-4o")
        temperature (float): Sampling temperature (0.7 training, 0.3 evaluation)
        max_tokens (int): Maximum tokens in LLM response
        api_key (Optional[str]): OpenAI API key
        retry_attempts (int): Number of retry attempts for API calls
        retry_delay (float): Delay between retries in seconds
    
    Example:
        >>> agent = QuantityRiskAgent(ticker="TSLA", temperature=0.3)
        >>> result = agent.decide_quantity(
        ...     direction_decision="buy",
        ...     maxcvar=150,
        ...     current_holdings=50,
        ...     account_value=100000,
        ...     current_price=250.0,
        ...     available_cash=50000
        ... )
        >>> print(result['order_size'])  # Integer between 0 and maxcvar
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
        Initialize Quantity & Risk Decision Agent.
        
        Args:
            ticker: Stock ticker symbol (e.g., "TSLA", "AAPL")
            model_name: OpenAI model name (default: "gpt-4o")
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens in LLM response
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            retry_attempts: Number of retry attempts for API failures
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
        logger.info(f"Initialized QuantityRiskAgent for {ticker} (temp={temperature})")
    
    def get_system_prompt(self) -> str:
        """
        Get system prompt for Quantity & Risk Agent.
        
        Returns:
            str: System prompt defining the agent's role and constraints
        """
        return """You are a Risk Management Agent responsible for determining optimal position sizes in a trading system.

YOUR ROLE:
- Receive strategic direction (buy/sell/hold) from the Direction Agent
- Determine the appropriate order size based on risk constraints
- Enforce CVaR-based position limits independently
- Prevent overconfidence from violating risk discipline

KEY CONSTRAINTS:
1. CVaR Constraint: Order size must NOT exceed maxcvar (CVaR-based maximum)
2. Cash Constraint: Order size must NOT exceed available_cash / current_price
3. Direction Alignment: Order size should align with strategic direction
   - If direction is "buy": order_size > 0 (increase position)
   - If direction is "sell": order_size > 0 (decrease position, but don't exceed current holdings)
   - If direction is "hold": order_size = 0 (no action)
4. No Short Selling: Position must remain >= 0 at all times
5. Integer Sizes: Order sizes must be whole numbers (round down for safety)

RISK MANAGEMENT PRINCIPLES:
- Be conservative during high volatility (use lower portion of maxcvar)
- Consider current exposure relative to account value
- Factor in recent performance and market conditions
- Prioritize capital preservation over aggressive positioning

OUTPUT REQUIREMENTS:
- Provide JSON output with exact schema specified
- order_size must be a non-negative integer
- Include clear reasoning for position sizing decision
- Reference relevant memory indices that influenced the decision
- Analyze reflections from past similar decisions"""
    
    def format_user_prompt(self, **kwargs) -> str:
        """
        Format user prompt with decision context and constraints.
        
        Args:
            **kwargs: Dynamic variables including:
                - ticker: Stock ticker symbol
                - direction_decision: Buy/sell/hold from Direction Agent
                - maxcvar: CVaR-based maximum position size
                - current_holdings: Current number of shares held
                - account_value: Total account value in dollars
                - current_price: Current stock price
                - available_cash: Available cash for trading
                - price_history: Recent price data
                - volatility_metrics: Volatility indicators
                - working_memory: Combined memory from all layers
                - reflection_notes: Past decision outcomes
        
        Returns:
            str: Formatted user prompt for LLM
        """
        # Extract variables with defaults
        ticker = kwargs.get('ticker', self.ticker)
        direction_decision = kwargs.get('direction_decision', 'hold')
        maxcvar = kwargs.get('maxcvar', 0)
        current_holdings = kwargs.get('current_holdings', 0)
        account_value = kwargs.get('account_value', 100000)
        current_price = kwargs.get('current_price', 100)
        available_cash = kwargs.get('available_cash', 50000)
        price_history = kwargs.get('price_history', [])
        volatility_metrics = kwargs.get('volatility_metrics', {})
        working_memory = kwargs.get('working_memory', [])
        reflection_notes = kwargs.get('reflection_notes', [])
        
        # Calculate maximum affordable shares
        max_affordable = int(available_cash / current_price) if current_price > 0 else 0
        
        # Calculate maximum sellable shares (can't sell more than held)
        max_sellable = current_holdings if direction_decision == 'sell' else 0
        
        # Determine effective maximum based on direction
        if direction_decision == 'buy':
            effective_max = min(maxcvar, max_affordable)
            constraint_note = f"Maximum buy size: min(CVaR limit={maxcvar}, Affordable={max_affordable}) = {effective_max}"
        elif direction_decision == 'sell':
            effective_max = min(maxcvar, max_sellable)
            constraint_note = f"Maximum sell size: min(CVaR limit={maxcvar}, Holdings={max_sellable}) = {effective_max}"
        else:  # hold
            effective_max = 0
            constraint_note = "Hold decision: order_size must be 0"
        
        # Format price history
        price_str = ""
        if price_history:
            recent_prices = price_history[-10:] if len(price_history) > 10 else price_history
            price_str = "\n".join([f"  {p}" for p in recent_prices])
        
        # Format working memory
        memory_str = ""
        if working_memory:
            memory_items = []
            for i, item in enumerate(working_memory[-10:]):  # Last 10 items
                memory_items.append(f"  [{i}] {item.get('content', 'N/A')[:100]}...")
            memory_str = "\n".join(memory_items)
        
        # Format reflection notes
        reflection_str = ""
        if reflection_notes:
            recent_reflections = reflection_notes[-5:]  # Last 5 reflections
            reflection_items = []
            for i, ref in enumerate(recent_reflections):
                reflection_items.append(f"  [{i}] {ref.get('note', 'N/A')[:150]}...")
            reflection_str = "\n".join(reflection_items)
        
        # Format volatility metrics
        volatility_str = ""
        if volatility_metrics:
            volatility_str = "\n".join([
                f"  {k}: {v}" for k, v in volatility_metrics.items()
            ])
        
        # Build user prompt
        user_prompt = f"""TICKER: {ticker}
DIRECTION DECISION: {direction_decision.upper()}

CURRENT PORTFOLIO STATE:
- Account Value: ${account_value:,.2f}
- Available Cash: ${available_cash:,.2f}
- Current Holdings: {current_holdings} shares
- Current Price: ${current_price:.2f}
- Holdings Value: ${current_holdings * current_price:,.2f}

RISK CONSTRAINTS:
- CVaR-Based Max Position (maxcvar): {maxcvar} shares
{constraint_note}

MARKET CONTEXT:
Recent Price History (last 10 days):
{price_str if price_str else "  No price history available"}

Volatility Metrics:
{volatility_str if volatility_str else "  No volatility data available"}

WORKING MEMORY (recent insights):
{memory_str if memory_str else "  No memory items available"}

REFLECTION NOTES (past decision outcomes):
{reflection_str if reflection_str else "  No reflection notes available"}

TASK:
Based on the direction decision ({direction_decision}) and all constraints above, determine the optimal order size.

REQUIREMENTS:
1. Order size must be a non-negative integer
2. Order size must NOT exceed {effective_max} (effective maximum)
3. If direction is "hold", order_size must be 0
4. Provide clear reasoning for your position sizing decision
5. Reference which memory items influenced your decision

OUTPUT FORMAT:
Return JSON with this exact schema:
{{
    "order_size": <integer 0-{effective_max}>,
    "summary_reason": "<brief explanation of position sizing decision>",
    "memory_indices": [<list of memory item indices used>],
    "reflection_analysis": "<analysis of how past outcomes influence this decision>"
}}"""
        
        return user_prompt
    
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate Quantity Agent output against expected schema.
        
        Args:
            output: Parsed JSON output from LLM
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Use shared validation from json_parser
        is_valid, errors = validate_quantity_output(output)
        
        if not is_valid:
            logger.warning(f"Quantity output validation failed: {errors}")
            return False
        
        # Additional validation: ensure order_size is non-negative integer
        order_size = output.get('order_size', -1)
        if not isinstance(order_size, int) or order_size < 0:
            logger.warning(f"Invalid order_size: {order_size} (must be non-negative integer)")
            return False
        
        # Validate memory_indices is a list
        memory_indices = output.get('memory_indices', [])
        if not isinstance(memory_indices, list):
            logger.warning(f"Invalid memory_indices: {memory_indices} (must be list)")
            return False
        
        logger.debug(f"Quantity output validated successfully: order_size={order_size}")
        return True
    
    def decide_quantity(
        self,
        direction_decision: str,
        maxcvar: int,
        current_holdings: int,
        account_value: float,
        current_price: float,
        available_cash: float,
        price_history: Optional[List[float]] = None,
        volatility_metrics: Optional[Dict[str, float]] = None,
        working_memory: Optional[List[Dict[str, Any]]] = None,
        reflection_notes: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make quantity/risk decision based on direction and constraints.
        
        Args:
            direction_decision: Strategic direction from Direction Agent (buy/sell/hold)
            maxcvar: CVaR-based maximum position size
            current_holdings: Current number of shares held
            account_value: Total account value in dollars
            current_price: Current stock price
            available_cash: Available cash for trading
            price_history: List of recent prices (optional)
            volatility_metrics: Dict of volatility indicators (optional)
            working_memory: List of memory items from hierarchical memory (optional)
            reflection_notes: List of reflection notes from past decisions (optional)
            **kwargs: Additional context variables
        
        Returns:
            Dict[str, Any]: Decision output with keys:
                - order_size: Integer position size (0 to maxcvar)
                - summary_reason: Explanation of decision
                - memory_indices: List of memory item indices used
                - reflection_analysis: Analysis of past outcomes
                - validation_passed: Boolean indicating if output passed validation
        
        Raises:
            ValueError: If direction_decision is not buy/sell/hold
        """
        # Validate direction decision
        valid_directions = ['buy', 'sell', 'hold']
        if direction_decision.lower() not in valid_directions:
            raise ValueError(f"Invalid direction_decision: {direction_decision}. Must be one of {valid_directions}")
        
        direction_decision = direction_decision.lower()
        
        # Prepare defaults for optional parameters
        price_history = price_history or []
        volatility_metrics = volatility_metrics or {}
        working_memory = working_memory or []
        reflection_notes = reflection_notes or []
        
        logger.info(
            f"Deciding quantity for {self.ticker}: direction={direction_decision}, "
            f"maxcvar={maxcvar}, holdings={current_holdings}, price=${current_price:.2f}"
        )
        
        try:
            # Format user prompt with all context
            user_prompt = self.format_user_prompt(
                direction_decision=direction_decision,
                maxcvar=maxcvar,
                current_holdings=current_holdings,
                account_value=account_value,
                current_price=current_price,
                available_cash=available_cash,
                price_history=price_history,
                volatility_metrics=volatility_metrics,
                working_memory=working_memory,
                reflection_notes=reflection_notes,
                **kwargs
            )
            
            # Get system prompt
            system_prompt = self.get_system_prompt()
            
            # Call LLM with retry logic
            result = self.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Validate output
            validation_passed = self.validate_output(result)
            result['validation_passed'] = validation_passed
            
            # Enforce hard constraints even if LLM violates them
            effective_max = maxcvar
            if direction_decision == 'buy':
                max_affordable = int(available_cash / current_price) if current_price > 0 else 0
                effective_max = min(maxcvar, max_affordable)
            elif direction_decision == 'sell':
                effective_max = min(maxcvar, current_holdings)
            else:  # hold
                effective_max = 0
            
            # Clamp order_size to valid range
            original_order_size = result.get('order_size', 0)
            result['order_size'] = max(0, min(effective_max, original_order_size))
            
            if original_order_size != result['order_size']:
                logger.warning(
                    f"Clamped order_size from {original_order_size} to {result['order_size']} "
                    f"(effective_max={effective_max})"
                )
                result['constraint_enforced'] = True
            else:
                result['constraint_enforced'] = False
            
            logger.info(
                f"Quantity decision: order_size={result['order_size']}, "
                f"reason={result.get('summary_reason', 'N/A')[:50]}..."
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in decide_quantity: {str(e)}")
            # Return fallback decision
            return self._create_fallback_output(
                direction_decision=direction_decision,
                maxcvar=maxcvar,
                current_holdings=current_holdings,
                available_cash=available_cash,
                current_price=current_price,
                error=str(e)
            )
    
    def decide_batch(
        self,
        decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process batch of quantity decisions.
        
        Args:
            decisions: List of decision contexts, each containing:
                - direction_decision
                - maxcvar
                - current_holdings
                - account_value
                - current_price
                - available_cash
                - (optional) price_history, volatility_metrics, working_memory, reflection_notes
        
        Returns:
            List[Dict[str, Any]]: List of quantity decisions
        """
        results = []
        for i, decision_ctx in enumerate(decisions):
            logger.debug(f"Processing batch decision {i+1}/{len(decisions)}")
            
            try:
                result = self.decide_quantity(**decision_ctx)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch decision {i+1} failed: {str(e)}")
                # Create minimal fallback
                results.append({
                    'order_size': 0,
                    'summary_reason': f"Error: {str(e)}",
                    'memory_indices': [],
                    'reflection_analysis': "Fallback due to error",
                    'validation_passed': False,
                    'constraint_enforced': False
                })
        
        logger.info(f"Completed batch of {len(results)} quantity decisions")
        return results
    
    def _create_fallback_output(
        self,
        direction_decision: str,
        maxcvar: int,
        current_holdings: int,
        available_cash: float,
        current_price: float,
        error: str
    ) -> Dict[str, Any]:
        """
        Create fallback output when LLM decision fails.
        
        Uses conservative heuristic:
        - Hold: order_size = 0
        - Buy: order_size = min(maxcvar, available_cash / price) * 0.5 (50% of max)
        - Sell: order_size = min(maxcvar, current_holdings) * 0.5 (50% of holdings)
        
        Args:
            direction_decision: Buy/sell/hold decision
            maxcvar: CVaR-based maximum
            current_holdings: Current shares held
            available_cash: Available cash
            current_price: Current price
            error: Error message that triggered fallback
        
        Returns:
            Dict[str, Any]: Fallback decision output
        """
        if direction_decision == 'hold':
            order_size = 0
            reason = "Hold decision - no action (fallback mode)"
        elif direction_decision == 'buy':
            max_affordable = int(available_cash / current_price) if current_price > 0 else 0
            effective_max = min(maxcvar, max_affordable)
            order_size = int(effective_max * 0.5)  # Conservative: 50% of max
            reason = f"Conservative buy at 50% of max (fallback mode, error: {error[:50]})"
        else:  # sell
            effective_max = min(maxcvar, current_holdings)
            order_size = int(effective_max * 0.5)  # Conservative: 50% of holdings
            reason = f"Conservative sell at 50% of holdings (fallback mode, error: {error[:50]})"
        
        logger.warning(f"Using fallback quantity decision: order_size={order_size}, reason={reason}")
        
        return {
            'order_size': order_size,
            'summary_reason': reason,
            'memory_indices': [],
            'reflection_analysis': "Fallback decision due to LLM failure - conservative heuristic applied",
            'validation_passed': True,  # Fallback is always valid
            'constraint_enforced': True,
            'fallback_used': True,
            'error': error
        }


def create_quantity_risk_agent(
    ticker: str,
    mode: str = "evaluation"
) -> QuantityRiskAgent:
    """
    Factory function to create QuantityRiskAgent with mode-appropriate settings.
    
    Args:
        ticker: Stock ticker symbol
        mode: Agent mode - "training" (temp=0.7) or "evaluation" (temp=0.3)
    
    Returns:
        QuantityRiskAgent: Configured agent instance
    
    Example:
        >>> agent = create_quantity_risk_agent("TSLA", mode="training")
        >>> assert agent.temperature == 0.7
        >>> agent = create_quantity_risk_agent("TSLA", mode="evaluation")
        >>> assert agent.temperature == 0.3
    """
    temperature = 0.7 if mode == "training" else 0.3
    logger.info(f"Creating QuantityRiskAgent for {ticker} in {mode} mode (temp={temperature})")
    
    return QuantityRiskAgent(
        ticker=ticker,
        model_name="gpt-4o",
        temperature=temperature,
        max_tokens=1000,
        retry_attempts=3,
        retry_delay=1.0
    )
