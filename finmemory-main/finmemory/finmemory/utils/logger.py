"""
Logging utilities for the FinMemory trading agent system.

Provides centralized logging configuration with support for:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file output
- Structured logging with timestamps
- Log rotation for long-running experiments
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


# Global logger registry to avoid duplicate handlers
_logger_registry = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Get or create a logger with the specified name and configuration.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)
        log_dir: Directory for log files (if file_output=True)
        console_output: Enable console output (default: True)
        file_output: Enable file output (default: False)
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Trading decision made")
    """
    # Return existing logger if already configured
    if name in _logger_registry:
        return _logger_registry[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        _logger_registry[name] = logger
        return logger
    
    # Create formatter with detailed output
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (with rotation)
    if file_output and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{name.replace('.', '_')}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Register logger
    _logger_registry[name] = logger
    
    return logger


def set_global_level(level: int) -> None:
    """
    Set logging level for all registered loggers.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    
    Example:
        >>> set_global_level(logging.DEBUG)  # Enable debug logging
    """
    for logger in _logger_registry.values():
        for handler in logger.handlers:
            handler.setLevel(level)
        logger.setLevel(level)


def get_log_level_from_string(level_str: str) -> int:
    """
    Convert string log level to logging constant.
    
    Args:
        level_str: String representation of log level
    
    Returns:
        int: Logging level constant
    
    Raises:
        ValueError: If level_str is not a valid log level
    
    Example:
        >>> get_log_level_from_string("DEBUG")
        10
        >>> get_log_level_from_string("INFO")
        20
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.CRITICAL,
    }
    
    level_upper = level_str.upper().strip()
    if level_upper not in level_map:
        raise ValueError(
            f"Invalid log level: {level_str}. "
            f"Valid options: {list(level_map.keys())}"
        )
    
    return level_map[level_upper]


def create_experiment_logger(
    experiment_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger specifically for experiment runs.
    
    Creates both console and file output with experiment-specific naming.
    
    Args:
        experiment_name: Name of the experiment (used in log filename)
        log_dir: Directory for log files (default: "logs")
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured experiment logger
    
    Example:
        >>> logger = create_experiment_logger("main_experiment_tsla")
        >>> logger.info("Starting experiment run")
    """
    logger_name = f"finmemory.experiment.{experiment_name}"
    
    return get_logger(
        name=logger_name,
        level=level,
        log_dir=log_dir,
        console_output=True,
        file_output=True,
        max_bytes=50 * 1024 * 1024,  # 50 MB for experiments
        backup_count=10,
    )


def log_trade_decision(
    logger: logging.Logger,
    ticker: str,
    timestep: int,
    decision: str,
    quantity: int,
    price: float,
    reason: str,
) -> None:
    """
    Log a trading decision in a structured format.
    
    Args:
        logger: Logger instance
        ticker: Stock ticker symbol
        timestep: Current timestep in episode
        decision: Trading decision (buy/sell/hold)
        quantity: Number of shares
        price: Current price
        reason: Decision reasoning summary
    
    Example:
        >>> log_trade_decision(logger, "TSLA", 42, "buy", 10, 245.50, "Strong momentum signal")
    """
    logger.info(
        f"TRADE | {ticker} | Step {timestep:04d} | "
        f"{decision.upper():4s} | Qty: {quantity:4d} | "
        f"Price: ${price:8.2f} | Reason: {reason}"
    )


def log_memory_operation(
    logger: logging.Logger,
    operation: str,
    layer: str,
    importance_score: int,
    content_summary: str,
) -> None:
    """
    Log a memory system operation.
    
    Args:
        logger: Logger instance
        operation: Operation type (add/retrieve/reinforce/evict)
        layer: Memory layer (shallow/intermediate/deep/reflection)
        importance_score: Importance score (1-10)
        content_summary: Brief summary of memory content
    
    Example:
        >>> log_memory_operation(logger, "add", "deep", 9, "10-K filing shows 25% revenue growth")
    """
    logger.debug(
        f"MEMORY | {operation.upper():10s} | {layer:14s} | "
        f"Score: {importance_score:2d} | {content_summary}"
    )


def log_agent_call(
    logger: logging.Logger,
    agent_name: str,
    ticker: str,
    timestep: int,
    success: bool,
    latency_ms: float,
    token_usage: Optional[int] = None,
) -> None:
    """
    Log an LLM agent API call.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent (e.g., "DirectionAgent")
        ticker: Stock ticker symbol
        timestep: Current timestep
        success: Whether the call succeeded
        latency_ms: API call latency in milliseconds
        token_usage: Number of tokens used (optional)
    
    Example:
        >>> log_agent_call(logger, "DirectionAgent", "TSLA", 15, True, 1250.5, 450)
    """
    token_info = f" | Tokens: {token_usage}" if token_usage else ""
    status = "SUCCESS" if success else "FAILED"
    
    logger.info(
        f"AGENT | {agent_name:20s} | {ticker} | Step {timestep:04d} | "
        f"{status:7s} | Latency: {latency_ms:7.1f}ms{token_info}"
    )


def log_risk_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    ticker: str,
    timestep: int,
) -> None:
    """
    Log a risk metric calculation.
    
    Args:
        logger: Logger instance
        metric_name: Name of the metric (e.g., "CVaR_95", "Sharpe")
        value: Metric value
        ticker: Stock ticker symbol
        timestep: Current timestep
    
    Example:
        >>> log_risk_metric(logger, "CVaR_95", 0.0234, "TSLA", 50)
    """
    logger.debug(
        f"RISK | {metric_name:12s} | {ticker} | Step {timestep:04d} | "
        f"Value: {value:.6f}"
    )


# Convenience function for quick logging setup
def setup_default_logger(name: str = "finmemory") -> logging.Logger:
    """
    Set up a default logger with standard configuration.
    
    Args:
        name: Logger name (default: "finmemory")
    
    Returns:
        logging.Logger: Configured logger with INFO level and console output
    """
    return get_logger(
        name=name,
        level=logging.INFO,
        console_output=True,
        file_output=False,
    )
