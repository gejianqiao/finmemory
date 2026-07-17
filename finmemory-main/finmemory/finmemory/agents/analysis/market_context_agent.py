"""Unified market-context agent for the simplified FinMemory pipeline."""

import json
from typing import Any, Dict, Optional

from finmemory.agents.base_agent import BaseAgent
from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


class MarketContextAgent(BaseAgent):
    """Summarize currently available market signals in one LLM call."""

    def __init__(
        self,
        ticker: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 700,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        self.ticker = ticker.upper()

    def get_system_prompt(self) -> str:
        return (
            "You are a concise financial market analyst. Summarize only the supplied "
            "information for the target stock. Do not predict an exact future price. "
            "Return valid JSON with market_summary, sentiment_score (-1 to 1), "
            "importance_score (1 to 10), key_catalysts (list), and key_risks (list)."
        )

    def format_user_prompt(self, **kwargs) -> str:
        payload = {
            "ticker": self.ticker,
            "date": kwargs.get("current_date"),
            "signal_snapshot": kwargs.get("signal_snapshot", {}),
            "technical_indicators": kwargs.get("technical_indicators", {}),
            "recent_memory": kwargs.get("recent_memory", []),
        }
        return "Analyze the current context:\n" + json.dumps(payload, default=str)[:30000]

    def validate_output(self, output: Dict[str, Any]) -> bool:
        required = {"market_summary", "sentiment_score", "importance_score", "key_catalysts", "key_risks"}
        if not required.issubset(output):
            return False
        try:
            sentiment = float(output["sentiment_score"])
            importance = int(output["importance_score"])
        except (TypeError, ValueError):
            return False
        return (
            -1 <= sentiment <= 1
            and 1 <= importance <= 10
            and isinstance(output["key_catalysts"], list)
            and isinstance(output["key_risks"], list)
        )

    def analyze_context(
        self,
        current_date: str,
        signal_snapshot: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        recent_memory: list,
    ) -> Dict[str, Any]:
        try:
            return self.call_llm(
                system_prompt=self.get_system_prompt(),
                user_prompt=self.format_user_prompt(
                    current_date=current_date,
                    signal_snapshot=signal_snapshot,
                    technical_indicators=technical_indicators,
                    recent_memory=recent_memory,
                ),
            )
        except Exception as exc:
            logger.warning("Market context LLM failed; using deterministic fallback: %s", exc)
            sentiment = max(-1.0, min(1.0, float(signal_snapshot.get("avg_sentiment", 0.0))))
            importance = max(1, min(10, int(round(float(signal_snapshot.get("max_importance", 1) or 1)))))
            return {
                "market_summary": "Deterministic fallback based on aggregated current signals.",
                "sentiment_score": sentiment,
                "importance_score": importance,
                "key_catalysts": [],
                "key_risks": ["Unified market-context LLM output unavailable"],
                "fallback": True,
            }


def create_market_context_agent(
    ticker: str,
    mode: str = "evaluation",
    api_key: Optional[str] = None,
) -> MarketContextAgent:
    temperature = 0.7 if mode == "training" else 0.3
    agent = MarketContextAgent(ticker=ticker, temperature=temperature, api_key=api_key)
    agent.set_mode(mode)
    return agent
