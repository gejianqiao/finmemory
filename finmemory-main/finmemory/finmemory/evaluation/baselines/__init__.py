"""
Baseline Implementations for FinMemory Evaluation

This module provides 10 baseline trading strategies for comparison with FinMemory:

LLM-Based Agents (4):
- FinGPT: General financial LLM agent
- FINMEM: Memory-augmented trading agent
- FinAgent: Multi-modal financial agent
- FINCON: Context-aware financial agent

Deep Reinforcement Learning (3):
- A2C: Advantage Actor-Critic
- PPO: Proximal Policy Optimization
- DQN: Deep Q-Network

Rule-Based Strategies (3):
- MACD: Moving Average Convergence Divergence
- RSI: Relative Strength Index
- Random: Random buy/sell/hold baseline

Usage:
    from finmemory.evaluation.baselines import (
        FinGPTAgent, FINMEMAgent, FinAgent, FINCONAgent,
        A2CAgent, PPOAgent, DQNAgent,
        MACDAgent, RSIAgent, RandomAgent,
        get_all_baselines, run_baseline_comparison
    )
"""

from finmemory.evaluation.baselines.fingpt_agent import FinGPTAgent, create_fingpt_agent
from finmemory.evaluation.baselines.finmem_agent import FINMEMAgent, create_finmem_agent
from finmemory.evaluation.baselines.finagent import FinAgent, create_finagent
from finmemory.evaluation.baselines.fincon_agent import FINCONAgent, create_fincon_agent
from finmemory.evaluation.baselines.a2c_agent import A2CAgent, create_a2c_agent
from finmemory.evaluation.baselines.ppo_agent import PPOAgent, create_ppo_agent
from finmemory.evaluation.baselines.dqn_agent import DQNAgent, create_dqn_agent
from finmemory.evaluation.baselines.macd_agent import MACDAgent, create_macd_agent
from finmemory.evaluation.baselines.rsi_agent import RSIAgent, create_rsi_agent
from finmemory.evaluation.baselines.random_agent import RandomAgent, create_random_agent

# Convenience function to get all baseline agents
def get_all_baselines(ticker: str, mode: str = "evaluation", **kwargs):
    """
    Get all 10 baseline agents for comparison.
    
    Args:
        ticker: Stock ticker symbol
        mode: "training" or "evaluation"
        **kwargs: Additional arguments passed to all agents
    
    Returns:
        dict: Dictionary mapping baseline names to agent instances
    """
    return {
        "FinGPT": create_fingpt_agent(ticker, mode, **kwargs),
        "FINMEM": create_finmem_agent(ticker, mode, **kwargs),
        "FinAgent": create_finagent(ticker, mode, **kwargs),
        "FINCON": create_fincon_agent(ticker, mode, **kwargs),
        "A2C": create_a2c_agent(ticker, mode, **kwargs),
        "PPO": create_ppo_agent(ticker, mode, **kwargs),
        "DQN": create_dqn_agent(ticker, mode, **kwargs),
        "MACD": create_macd_agent(ticker, mode, **kwargs),
        "RSI": create_rsi_agent(ticker, mode, **kwargs),
        "Random": create_random_agent(ticker, mode, **kwargs),
    }


def run_baseline_comparison(baselines_dict, prices, dates, initial_capital=100000):
    """
    Run all baseline agents and compare their performance.
    
    Args:
        baselines_dict: Dictionary of baseline agents
        prices: Array of prices
        dates: Array of dates
        initial_capital: Starting capital
    
    Returns:
        dict: Performance metrics for each baseline
    """
    from finmemory.environment.trading_env import create_trading_environment
    from finmemory.evaluation.metrics import calculate_metrics
    
    results = {}
    
    for name, agent in baselines_dict.items():
        # Create trading environment
        env = create_trading_environment(
            ticker=agent.ticker,
            prices=prices,
            dates=dates,
            initial_capital=initial_capital
        )
        
        # Reset environment
        env.reset()
        
        # Run episode
        trades = []
        while not env.is_done():
            state = env.get_state()
            decision = agent.decide(state)
            
            # Execute action
            result = env.step(decision)
            if result and result.get('executed', False):
                trades.append(result)
        
        # Calculate metrics
        episode_result = env.get_episode_result()
        metrics = calculate_metrics(
            returns=episode_result.daily_returns,
            account_values=episode_result.account_values,
            trades=trades,
            ticker=agent.ticker,
            period_start=dates[0],
            period_end=dates[-1],
            initial_capital=initial_capital
        )
        
        results[name] = metrics
    
    return results


__version__ = "1.0.0"
__all__ = [
    # LLM-Based Agents
    "FinGPTAgent",
    "create_fingpt_agent",
    "FINMEMAgent",
    "create_finmem_agent",
    "FinAgent",
    "create_finagent",
    "FINCONAgent",
    "create_fincon_agent",
    # DRL Agents
    "A2CAgent",
    "create_a2c_agent",
    "PPOAgent",
    "create_ppo_agent",
    "DQNAgent",
    "create_dqn_agent",
    # Rule-Based Agents
    "MACDAgent",
    "create_macd_agent",
    "RSIAgent",
    "create_rsi_agent",
    "RandomAgent",
    "create_random_agent",
    # Utility functions
    "get_all_baselines",
    "run_baseline_comparison",
]
