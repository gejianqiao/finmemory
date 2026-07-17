"""
FinMemory Utility Module

This module provides core utility functions and classes for the FinMemory trading agent system,
including logging, API client wrappers, JSON parsing, and prompt management.

Package Contents:
- logger: Centralized logging configuration with structured logging support
- api_clients: API wrapper classes for OpenAI, Finnhub, and SEC EDGAR APIs
- json_parser: Robust JSON parsing utilities for LLM outputs with error recovery
- prompt_manager: Prompt template loading and formatting for all agent types
"""

from finmemory.utils.logger import (
    get_logger,
    set_global_level,
    get_log_level_from_string,
    create_experiment_logger,
    log_trade_decision,
    log_memory_operation,
    log_agent_call,
    log_risk_metric,
    setup_default_logger,
)

from finmemory.utils.api_clients import (
    APIRateLimiter,
    APICache,
    OpenAIClient,
    FinnhubClient,
    SECClient,
    create_openai_client,
    create_finnhub_client,
    create_sec_client,
)

from finmemory.utils.json_parser import (
    JSONParseError,
    parse_json_output,
    validate_json_schema,
    safe_json_loads,
    validate_decision_output,
    validate_quantity_output,
)

from finmemory.utils.prompt_manager import (
    PromptManager,
    load_decision_prompt,
)

__version__ = "1.0.0"
__author__ = "FinMemory Research Team"
__all__ = [
    # Logger utilities
    "get_logger",
    "set_global_level",
    "get_log_level_from_string",
    "create_experiment_logger",
    "log_trade_decision",
    "log_memory_operation",
    "log_agent_call",
    "log_risk_metric",
    "setup_default_logger",
    # API clients
    "APIRateLimiter",
    "APICache",
    "OpenAIClient",
    "FinnhubClient",
    "SECClient",
    "create_openai_client",
    "create_finnhub_client",
    "create_sec_client",
    # JSON parsing
    "JSONParseError",
    "parse_json_output",
    "validate_json_schema",
    "safe_json_loads",
    "validate_decision_output",
    "validate_quantity_output",
    # Prompt management
    "PromptManager",
    "load_decision_prompt",
]
