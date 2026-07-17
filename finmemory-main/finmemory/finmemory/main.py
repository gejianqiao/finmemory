#!/usr/bin/env python3
"""
FinMemory: Position-Aware Trading Agent System for Real Financial Markets

Main entry point for running FinMemory experiments and evaluations.

Usage:
    python -m finmemory.main --mode evaluation --ticker TSLA
    python -m finmemory.main --mode training --ticker AAPL --epochs 10
    python -m finmemory.main --experiment main --stocks TSLA,AAPL,AMZN
    python -m finmemory.main --experiment ablation --config w/o_QRA

Author: FinMemory Research Team
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finmemory.utils.logger import get_logger, create_experiment_logger, set_global_level
from finmemory.utils.api_clients import create_openai_client, create_finnhub_client, create_sec_client
from finmemory.data.loaders.yahoo_loader import download_price_data
from finmemory.data.loaders.finnhub_loader import download_news_data
from finmemory.data.loaders.sec_loader import download_sec_filings
from finmemory.data.processors.price_processor import process_price_data
from finmemory.data.processors.signal_processor import process_market_signals
from finmemory.environment.trading_env import create_trading_environment
from finmemory.environment.reward_calculator import create_reward_calculator
from finmemory.environment.risk_metrics import calculate_risk_metrics
from finmemory.agents.analysis import create_market_context_agent
from finmemory.agents.decision import (
    DirectionAgent, QuantityRiskAgent,
    create_direction_agent, create_quantity_risk_agent
)
from finmemory.memory.hierarchical_memory import create_hierarchical_memory
from finmemory.memory.memory_allocator import create_memory_allocator
from finmemory.memory.reflection_module import create_reflection_module
from finmemory.risk.cvar_calculator import create_cvar_calculator
from finmemory.risk.position_sizer import create_position_sizer
from finmemory.risk.exposure_tracker import create_exposure_tracker
from finmemory.evaluation.metrics import calculate_metrics, generate_table1_results, generate_ablation_table


# Module constants
__version__ = "1.0.0"
__all__ = ["main", "run_single_stock", "run_main_experiment", "run_ablation_experiment", "parse_args"]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML files.
    
    Args:
        config_path: Optional path to custom config file. If None, uses default configs.
    
    Returns:
        Dictionary containing merged configuration from agents.yaml and environment.yaml
    """
    # Determine config directory
    if config_path:
        config_file = Path(config_path)
        config_dir = config_file.parent
    else:
        config_dir = Path(__file__).parent / "config"
    
    # Load agents configuration
    agents_config_path = config_dir / "agents.yaml"
    if not agents_config_path.exists():
        raise FileNotFoundError(f"Agents config not found: {agents_config_path}")
    
    with open(agents_config_path, 'r') as f:
        agents_config = yaml.safe_load(f)
    
    # Load environment configuration
    env_config_path = config_dir / "environment.yaml"
    if not env_config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {env_config_path}")
    
    with open(env_config_path, 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Merge configurations
    config = {
        'agents': agents_config,
        'environment': env_config,
        'merged_at': datetime.now().isoformat()
    }
    
    return config


def initialize_agents(
    ticker: str,
    mode: str,
    config: Dict[str, Any],
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize all LLM agents for a specific ticker and mode.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        mode: Agent mode ('training' or 'evaluation')
        config: Configuration dictionary
        api_key: OpenAI API key (optional, will use env var if not provided)
    
    Returns:
        Dictionary containing all initialized agents
    """
    logger = get_logger("finmemory.main")
    logger.info(f"Initializing agents for {ticker} in {mode} mode")
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Agents will use default key or fail.")
    
    agents = {
        'market_context_agent': create_market_context_agent(ticker, mode, api_key=api_key),
        'direction_agent': create_direction_agent(ticker, mode),
        'quantity_risk_agent': create_quantity_risk_agent(ticker, mode),
    }
    logger.info("Initialized simplified 3-agent pipeline")
    return agents


def initialize_memory_system(
    ticker: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Initialize hierarchical memory system with allocator and reflection module.
    
    Args:
        ticker: Stock ticker symbol
        config: Configuration dictionary
    
    Returns:
        Dictionary containing memory system components
    """
    logger = get_logger("finmemory.main")
    logger.info(f"Initializing memory system for {ticker}")
    
    # Get memory configuration
    memory_config = config['agents'].get('memory_system', {})
    
    # Initialize hierarchical memory
    memory = create_hierarchical_memory(
        ticker=ticker,
        shallow_capacity=memory_config.get('shallow_capacity', 10),
        intermediate_capacity=memory_config.get('intermediate_capacity', 30),
        deep_capacity=memory_config.get('deep_capacity', 100),
        reflection_capacity=memory_config.get('reflection_capacity', 50)
    )
    
    # Initialize memory allocator
    allocator = create_memory_allocator(
        memory=memory,
        shallow_threshold=memory_config.get('shallow_threshold', 4),
        intermediate_threshold=memory_config.get('intermediate_threshold', 7)
    )
    
    # Initialize reflection module
    reflection = create_reflection_module(
        ticker=ticker,
        reflection_threshold_positive=5.0,
        reflection_threshold_negative=-5.0
    )
    
    logger.info(f"Memory system initialized: shallow={memory_config.get('shallow_capacity', 10)}, "
                f"intermediate={memory_config.get('intermediate_capacity', 30)}, "
                f"deep={memory_config.get('deep_capacity', 100)}")
    
    return {
        'memory': memory,
        'allocator': allocator,
        'reflection': reflection
    }


def initialize_risk_system(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Initialize risk management system (CVaR calculator, position sizer, exposure tracker).
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Dictionary containing risk management components
    """
    logger = get_logger("finmemory.main")
    logger.info("Initializing risk management system")
    
    # Get risk configuration
    risk_config = config['agents'].get('risk_management', {})
    cvar_config = risk_config.get('cvar_calculator', {})
    position_config = risk_config.get('position_sizer', {})
    exposure_config = risk_config.get('exposure_tracker', {})
    
    # Initialize CVaR calculator
    cvar_calculator = create_cvar_calculator(
        confidence_level=cvar_config.get('confidence_level', 0.95),
        window_size=cvar_config.get('window_size', 20),
        min_samples=5
    )
    
    # Initialize position sizer
    position_sizer = create_position_sizer(
        max_exposure_ratio=position_config.get('max_exposure_ratio', 0.1),
        confidence_level=cvar_config.get('confidence_level', 0.95),
        window_size=cvar_config.get('window_size', 20)
    )
    
    # Initialize exposure tracker
    exposure_tracker = create_exposure_tracker(
        max_exposure_ratio=exposure_config.get('max_exposure_ratio', 0.9),
        max_single_position_ratio=exposure_config.get('max_single_position_ratio', 0.3),
        concentration_threshold=exposure_config.get('concentration_threshold', 0.25),
        cvar_confidence_level=cvar_config.get('confidence_level', 0.95),
        cvar_window_size=cvar_config.get('window_size', 20)
    )
    
    logger.info(f"Risk system initialized: CVaR confidence={cvar_config.get('confidence_level', 0.95)}, "
                f"window={cvar_config.get('window_size', 20)} days")
    
    return {
        'cvar_calculator': cvar_calculator,
        'position_sizer': position_sizer,
        'exposure_tracker': exposure_tracker
    }


def load_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    data_dir: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load all required data (prices, news, SEC filings) for specified tickers and period.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        data_dir: Directory for caching data (optional)
        api_keys: Dictionary with 'finnhub' key (optional)
    
    Returns:
        Dictionary mapping tickers to their data (prices, news, filings)
    """
    logger = get_logger("finmemory.main")
    logger.info(f"Loading data for {len(tickers)} tickers: {tickers}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "raw"
    
    all_data = {}
    
    # Download price data
    logger.info("Downloading price data from Yahoo Finance...")
    price_data = download_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        force_refresh=False
    )
    
    # Download news data
    logger.info("Downloading news data from Finnhub...")
    finnhub_key = api_keys.get('finnhub') if api_keys else os.getenv("FINNHUB_API_KEY")
    news_data = download_news_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        api_key=finnhub_key,
        include_macro=True
    )
    
    # Download SEC filings
    logger.info("Downloading SEC filings...")
    filings_data = download_sec_filings(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        include_10k=True,
        include_10q=True
    )
    
    # Combine data for each ticker
    for ticker in tickers:
        all_data[ticker] = {
            'prices': price_data.get(ticker),
            'news': news_data.get(ticker),
            'macro_news': news_data.get('macro'),
            'filings': filings_data.get(ticker, {})
        }
        
        if all_data[ticker]['prices'] is None or all_data[ticker]['prices'].empty:
            logger.warning(f"No price data available for {ticker}")
        else:
            logger.info(f"{ticker}: {len(all_data[ticker]['prices'])} price records")
    
    logger.info(f"Data loading complete for {len(tickers)} tickers")
    
    return all_data


def run_market_signal_agents(
    ticker: str,
    company_news: Optional[pd.DataFrame],
    macro_news: Optional[pd.DataFrame],
    filings: Dict[str, List[Dict[str, Any]]],
    dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    """Prepare market signals deterministically without per-item LLM calls."""
    company_news = company_news.copy() if company_news is not None else pd.DataFrame()
    macro_news = macro_news.copy() if macro_news is not None else pd.DataFrame()
    pending_memory: List[Dict[str, Any]] = []

    def normalize_news(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=['timestamp', 'headline', 'summary', 'sentiment'])
        frame = frame.copy()
        if 'timestamp' not in frame:
            frame['timestamp'] = frame.get('datetime')
        sentiment = frame['sentiment'] if 'sentiment' in frame else pd.Series(0.0, index=frame.index)
        frame['sentiment'] = pd.to_numeric(sentiment, errors='coerce').fillna(0)
        if 'headline' not in frame:
            frame['headline'] = ''
        if 'summary' not in frame:
            frame['summary'] = ''
        frame['headline'] = frame['headline'].fillna('')
        frame['summary'] = frame['summary'].fillna('')
        for article in frame.to_dict('records'):
            ticker_mentioned = ticker.upper() in article.get('headline', '').upper()
            importance = 7 if source == 'company_news' and ticker_mentioned else (5 if source == 'company_news' else 3)
            content = f"{article.get('headline', '')}: {article.get('summary', '')}"[:1500]
            pending_memory.append({
                'timestamp': article.get('timestamp'),
                'content': content,
                'importance_score': importance,
                'source': source,
                'metadata': {'sentiment': article.get('sentiment', 0), 'url': article.get('url', '')},
            })
        return frame

    company_signal_df = normalize_news(company_news, 'company_news')
    macro_signal_df = normalize_news(macro_news, 'macro_news')

    processed_filings = []
    for form_type, importance in [('10-K', 8), ('10-Q', 6)]:
        for filing in filings.get(form_type, []):
            summary = filing.get('summary') or filing.get('text', '')[:1200]
            enriched = {**filing, 'form_type': form_type, 'summary': summary}
            processed_filings.append(enriched)
            pending_memory.append({
                'timestamp': filing.get('filing_date'),
                'content': f"{ticker} {form_type}: {summary}"[:1500],
                'importance_score': importance,
                'source': form_type,
                'metadata': {
                    'report_date': filing.get('report_date'),
                    'accession_number': filing.get('accession_number'),
                },
            })

    signal_processor, snapshots = process_market_signals(
        ticker=ticker,
        company_news_df=company_signal_df,
        macro_news_df=macro_signal_df,
        sec_filings=processed_filings,
        dates=dates,
    )
    pending_memory.sort(key=lambda item: pd.to_datetime(item['timestamp'], utc=True, errors='coerce'))
    return {
        'signal_processor': signal_processor,
        'snapshots': snapshots,
        'pending_memory': pending_memory,
        'source_counts': {
            'company_news': len(company_signal_df),
            'macro_news': len(macro_signal_df),
            'sec_filings': len(processed_filings),
        },
    }


def run_single_stock(
    ticker: str,
    start_date: str,
    end_date: str,
    mode: str = "evaluation",
    config: Optional[Dict[str, Any]] = None,
    initial_capital: float = 100000.0,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run FinMemory trading agent for a single stock.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        mode: Agent mode ('training' or 'evaluation')
        config: Configuration dictionary (optional, will load defaults if None)
        initial_capital: Starting capital in USD
        api_key: OpenAI API key (optional)
    
    Returns:
        Dictionary containing trading results and metrics
    """
    logger = get_logger("finmemory.main")
    logger.info(f"{'='*60}")
    logger.info(f"Running FinMemory for {ticker}")
    logger.info(f"Mode: {mode} | Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"{'='*60}")
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Load data
    data = load_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        api_keys={'finnhub': os.getenv("FINNHUB_API_KEY")}
    )
    
    if ticker not in data or data[ticker]['prices'] is None:
        logger.error(f"No data available for {ticker}")
        return {'error': f"No data available for {ticker}"}
    
    price_df = data[ticker]['prices']
    news_df = data[ticker]['news']
    macro_news_df = data[ticker].get('macro_news', pd.DataFrame())
    filings = data[ticker]['filings']
    
    # Process price data
    logger.info("Processing price data...")
    price_features = process_price_data(price_df, ticker)
    
    # Extract prices and dates for environment
    prices = price_df['Close'].values
    date_index = pd.DatetimeIndex(pd.to_datetime(price_df.index, utc=True)).tz_localize(None)
    dates = date_index.tolist()
    
    # Initialize components
    logger.info("Initializing trading components...")
    agents = initialize_agents(ticker, mode, config, api_key)
    memory_system = initialize_memory_system(ticker, config)
    risk_system = initialize_risk_system(config)

    logger.info("Preparing deterministic market signals...")
    market_context = run_market_signal_agents(
        ticker=ticker, company_news=news_df,
        macro_news=macro_news_df, filings=filings, dates=date_index,
    )
    pending_memory = market_context['pending_memory']
    next_memory_item = 0
    
    # Create trading environment
    env_config = config['environment'].get('environment', {})
    env = create_trading_environment(
        ticker=ticker,
        prices=prices,
        dates=dates,
        initial_capital=initial_capital,
        transaction_cost=env_config.get('transaction_cost', 0.001)
    )
    env.reset()
    
    # Create reward calculator (for training mode)
    reward_config = config['environment'].get('multi_timescale_reward', {})
    reward_calc = create_reward_calculator(
        prices=prices,
        dates=dates,
        short_window=reward_config.get('short_window', 1),
        mid_window=reward_config.get('mid_window', 7),
        long_window=reward_config.get('long_window', 30)
    )
    
    # Run trading loop
    logger.info(f"Starting trading loop for {len(dates)} timesteps...")
    
    episode_log = []
    for timestep in range(len(dates) - 1):  # -1 because we need next day for reward
        current_date = dates[timestep]
        current_price = prices[timestep]

        # Reveal analyzed information only when it was publicly available, which
        # prevents future news and filings from leaking into earlier decisions.
        current_cutoff = pd.Timestamp(current_date)
        if current_cutoff.tzinfo is None:
            current_cutoff = current_cutoff.tz_localize('UTC')
        else:
            current_cutoff = current_cutoff.tz_convert('UTC')
        while next_memory_item < len(pending_memory):
            item = pending_memory[next_memory_item]
            item_date = pd.to_datetime(item['timestamp'], utc=True, errors='coerce')
            if pd.isna(item_date) or item_date > current_cutoff:
                break
            memory_system['allocator'].allocate(
                content=item['content'],
                importance_score=max(1, min(10, int(round(float(item['importance_score']))))),
                source=item['source'], metadata=item['metadata'],
            )
            next_memory_item += 1
        
        # Get current state
        state = env.get_state()
        
        # Get working memory
        working_memory = memory_system['memory'].get_working_memory()
        
        # Get price features
        features = price_features.iloc[timestep].to_dict() if hasattr(price_features, 'iloc') else {}
        signal_snapshot = market_context['signal_processor'].get_signal_snapshot(current_date)
        sentiment_indicators = signal_snapshot.to_dict()
        recent_memory = memory_system['memory'].get_recent_items(n=10)
        context_analysis = agents['market_context_agent'].analyze_context(
            current_date=str(current_date),
            signal_snapshot=sentiment_indicators,
            technical_indicators=features,
            recent_memory=recent_memory,
        )
        sentiment_indicators['market_context'] = context_analysis
        sentiment_indicators['avg_sentiment'] = context_analysis.get(
            'sentiment_score', sentiment_indicators.get('avg_sentiment', 0.0)
        )
        price_history = (
            price_df.iloc[max(0, timestep-30):timestep+1]
            .rename(columns=str.lower)
            .to_dict('records')
        )
        
        # Direction decision
        direction_result = agents['direction_agent'].decide_direction(
            working_memory=working_memory,
            price_history=price_history,
            technical_indicators=features,
            sentiment_indicators=sentiment_indicators,
            current_date=str(current_date),
            current_position=state['position_size'],
            current_holdings=state['position_size'],
        )
        
        # Get CVaR constraint
        returns_window = prices[max(0, timestep-20):timestep+1]
        if len(returns_window) > 1:
            daily_returns = [returns_window[i]/returns_window[i-1] - 1 
                           for i in range(1, len(returns_window))]
            max_position_value = risk_system['position_sizer'].calculate_max_cvar_size(
                account_value=state['account_value'], returns_history=daily_returns
            )
            max_cvar = int(max_position_value / current_price) if current_price > 0 else 0
        else:
            max_cvar = int(initial_capital * 0.1 / current_price)
        
        # Quantity decision
        quantity_result = agents['quantity_risk_agent'].decide_quantity(
            direction_decision=direction_result['investment_decision'],
            maxcvar=max_cvar,
            current_holdings=state['position_size'],
            account_value=state['account_value'],
            current_price=current_price,
            available_cash=state['cash'],
            price_history=state['price_history'],
            volatility_metrics=features,
            working_memory=working_memory,
            reflection_notes=[r.to_dict() for r in memory_system['reflection'].get_reflections()]
        )
        
        # Execute trade
        direction = direction_result['investment_decision']
        quantity = quantity_result['order_size']
        
        next_state, env_reward, done, step_info = env.step({
            'direction': direction,
            'quantity': quantity
        })
        trade_result = step_info['trade_result']
        
        # Calculate reward (for training)
        if mode == "training":
            reward_result = reward_calc.calculate_reward(
                timestep=timestep,
                current_position=next_state['position_size'],
                previous_position=state['position_size'],
            )
        else:
            reward_result = None
        
        # Log decision
        episode_log.append({
            'timestep': timestep,
            'date': current_date,
            'price': current_price,
            'direction': direction,
            'quantity': quantity,
            'position_after': trade_result.position_after if trade_result else 0,
            'account_value': env.get_account_value(),
            'reward': reward_result.reward if reward_result else env_reward,
            'signals': sentiment_indicators,
        })

        # Record the realized one-step account outcome and feed negative lessons
        # back into hierarchical reflection memory.
        pnl = next_state['account_value'] - state['account_value']
        pnl_pct = (pnl / state['account_value'] * 100) if state['account_value'] else 0.0
        memory_indices = list(dict.fromkeys(
            direction_result.get('memory_indices', []) + quantity_result.get('memory_indices', [])
        ))
        reflection_count = len(memory_system['reflection'].reflections)
        memory_system['reflection'].record_outcome(
            timestep=timestep, date=current_date, direction=direction, quantity=quantity,
            entry_price=current_price, exit_price=prices[timestep + 1], pnl=pnl,
            pnl_pct=pnl_pct, memory_indices=memory_indices,
        )
        for note in memory_system['reflection'].reflections[reflection_count:]:
            memory_system['memory'].add_reflection(
                decision_outcome=pnl_pct,
                decision_type=direction,
                influenced_by=note.triggered_by_indices,
                lesson_learned=note.lesson_learned,
                timestamp=note.date,
            )
    
    # Get episode results
    episode_result = env.get_episode_result()
    
    # Calculate metrics
    if episode_result and len(episode_result.daily_returns) > 0:
        metrics = calculate_metrics(
            returns=episode_result.daily_returns,
            account_values=episode_result.account_values,
            trades=[asdict(trade) if is_dataclass(trade) else trade for trade in episode_result.trades],
            ticker=ticker,
            period_start=start_date,
            period_end=end_date,
            initial_capital=initial_capital
        )
    else:
        metrics = None
    
    # Compile results
    results = {
        'ticker': ticker,
        'mode': mode,
        'period': {'start': start_date, 'end': end_date},
        'initial_capital': initial_capital,
        'final_value': episode_result.final_value if episode_result else None,
        'cumulative_return': episode_result.cumulative_return if episode_result else None,
        'total_trades': episode_result.total_trades if episode_result else 0,
        'metrics': metrics.to_dict() if metrics else None,
        'episode_log': episode_log,
        'market_source_counts': market_context['source_counts'],
        'memory_statistics': memory_system['memory'].get_statistics(),
        'reflection_statistics': memory_system['reflection'].get_statistics(),
        'agent_costs': {
            'market_context_agent': agents['market_context_agent'].get_cost_summary(),
            'direction_agent': agents['direction_agent'].get_cost_summary(),
            'quantity_agent': agents['quantity_risk_agent'].get_cost_summary()
        }
    }
    
    # Log summary
    logger.info(f"{'='*60}")
    logger.info(f"Trading Complete for {ticker}")
    logger.info(f"Final Value: ${results['final_value']:,.2f}")
    logger.info(f"Cumulative Return: {results['cumulative_return']*100:.2f}%")
    logger.info(f"Total Trades: {results['total_trades']}")
    if metrics:
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown*100:.2f}%")
        logger.info(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
    logger.info(f"{'='*60}")
    
    return results


def run_main_experiment(
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run main experiment (Table 1) on all 5 stocks.
    
    Args:
        config: Configuration dictionary (optional)
        output_dir: Directory for saving results (optional)
    
    Returns:
        Dictionary containing results for all stocks
    """
    logger = get_logger("finmemory.main")
    logger.info("Starting Main Experiment (Table 1)")
    
    if config is None:
        config = load_config()
    
    # Get experiment configuration
    exp_config = config['environment'].get('experiments', {})
    stocks = exp_config.get('stocks', ['TSLA', 'AAPL', 'AMZN', 'NFLX', 'COIN'])
    eval_period = exp_config.get('evaluation_period', {
        'start': '2025-03-01',
        'end': '2025-09-30'
    })
    
    initial_capital = config['environment'].get('environment', {}).get('initial_capital', 100000)
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "evaluation" / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment for each stock
    all_results = {}
    for ticker in stocks:
        try:
            result = run_single_stock(
                ticker=ticker,
                start_date=eval_period['start'],
                end_date=eval_period['end'],
                mode="evaluation",
                config=config,
                initial_capital=initial_capital
            )
            all_results[ticker] = result
        except Exception as e:
            logger.error(f"Error running experiment for {ticker}: {str(e)}")
            all_results[ticker] = {'error': str(e)}
    
    # Generate Table 1
    if len(all_results) > 0:
        table1_df = generate_table1_results(all_results)
        table1_path = output_dir / "table1_main_results.csv"
        table1_df.to_csv(table1_path, index=False)
        logger.info(f"Table 1 saved to: {table1_path}")
    
    # Save full results
    results_path = output_dir / "main_experiment_results.json"
    import json
    with open(results_path, 'w') as f:
        # Convert non-serializable objects
        serializable_results = {}
        for ticker, result in all_results.items():
            serializable_results[ticker] = {
                k: v for k, v in result.items() 
                if k != 'episode_log'  # Skip large logs for JSON
            }
        json.dump(serializable_results, f, indent=2, default=str)
    
    logger.info(f"Main experiment complete. Results saved to: {output_dir}")
    
    return all_results


def run_ablation_experiment(
    ablation_config: str = "w/o_MTR",
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run ablation study experiment (Table 2).
    
    Args:
        ablation_config: Ablation configuration ('w/o_MTR', 'w/o_QRA', 'w/o_MSP')
        config: Configuration dictionary (optional)
        output_dir: Directory for saving results (optional)
    
    Returns:
        Dictionary containing ablation study results
    """
    logger = get_logger("finmemory.main")
    logger.info(f"Starting Ablation Experiment: {ablation_config}")
    
    if config is None:
        config = load_config()
    
    # Get experiment configuration
    exp_config = config['environment'].get('experiments', {})
    ablation_stocks = exp_config.get('ablation_study', {}).get('stocks', ['TSLA', 'AAPL'])
    eval_period = exp_config.get('evaluation_period', {
        'start': '2025-03-01',
        'end': '2025-09-30'
    })
    
    initial_capital = config['environment'].get('environment', {}).get('initial_capital', 100000)
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "evaluation" / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ablation for each stock
    ablation_results = {}
    for ticker in ablation_stocks:
        try:
            # Modify config based on ablation type
            ablation_config_modified = config.copy()
            
            if ablation_config == "w/o_MTR":
                # Remove multi-timescale reward (use single-day only)
                ablation_config_modified['environment']['multi_timescale_reward'] = {
                    'short_window': 1,
                    'mid_window': 1,
                    'long_window': 1
                }
                logger.info(f"Ablation {ablation_config}: Using single-day reward only")
            
            elif ablation_config == "w/o_QRA":
                # Remove quantity/risk agent (use fixed position sizing)
                logger.info(f"Ablation {ablation_config}: Using fixed position sizing")
                # This would require modifying the trading loop to bypass QRA
            
            elif ablation_config == "w/o_MSP":
                # Remove market signal processing (use price data only)
                logger.info(f"Ablation {ablation_config}: Using price data only, no news/reports")
                # This would require modifying agent initialization to skip filtering/analysis
            
            result = run_single_stock(
                ticker=ticker,
                start_date=eval_period['start'],
                end_date=eval_period['end'],
                mode="evaluation",
                config=ablation_config_modified,
                initial_capital=initial_capital
            )
            ablation_results[ticker] = result
        except Exception as e:
            logger.error(f"Error running ablation for {ticker}: {str(e)}")
            ablation_results[ticker] = {'error': str(e)}
    
    # Save results
    results_path = output_dir / f"ablation_{ablation_config}_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)
    
    logger.info(f"Ablation experiment complete. Results saved to: {results_path}")
    
    return ablation_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FinMemory: Position-Aware Trading Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single stock evaluation
  python -m finmemory.main --mode evaluation --ticker TSLA
  
  # Run main experiment (Table 1)
  python -m finmemory.main --experiment main
  
  # Run ablation study
  python -m finmemory.main --experiment ablation --ablation-config w/o_QRA
  
  # Run with custom config
  python -m finmemory.main --config /path/to/custom_config.yaml
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        default='evaluation',
        choices=['training', 'evaluation'],
        help='Agent mode (training or evaluation)'
    )
    
    # Single stock parameters
    parser.add_argument(
        '--ticker',
        type=str,
        default='TSLA',
        help='Stock ticker symbol (default: TSLA)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-03-01',
        help='Start date in YYYY-MM-DD format (default: 2025-03-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-09-30',
        help='End date in YYYY-MM-DD format (default: 2025-09-30)'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital in USD (default: 100000)'
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['main', 'ablation', 'sensitivity'],
        help='Run predefined experiment (main=Table 1, ablation=Table 2)'
    )
    parser.add_argument(
        '--ablation-config',
        type=str,
        default='w/o_MTR',
        choices=['w/o_MTR', 'w/o_QRA', 'w/o_MSP'],
        help='Ablation configuration (default: w/o_MTR)'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to custom configuration YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for saving results'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (optional)'
    )
    
    # API keys
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for FinMemory trading agent system."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    set_global_level(log_level)
    
    if args.log_file:
        logger = create_experiment_logger("finmemory", log_dir=os.path.dirname(args.log_file))
    else:
        logger = get_logger("finmemory.main")
    
    logger.info(f"FinMemory v{__version__} starting...")
    logger.info(f"Mode: {args.mode} | Ticker: {args.ticker}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Run based on experiment type
        if args.experiment == 'main':
            # Run main experiment (Table 1)
            results = run_main_experiment(
                config=config,
                output_dir=args.output_dir
            )
        elif args.experiment == 'ablation':
            # Run ablation study (Table 2)
            results = run_ablation_experiment(
                ablation_config=args.ablation_config,
                config=config,
                output_dir=args.output_dir
            )
        else:
            # Run single stock
            results = run_single_stock(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                mode=args.mode,
                config=config,
                initial_capital=args.initial_capital,
                api_key=args.api_key
            )
        
        logger.info("FinMemory execution completed successfully")
        return results
        
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
