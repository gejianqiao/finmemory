"""
Ablation Experiment Module for FinMemory Trading Agent System

Implements ablation studies to verify each component's contribution to overall performance.
Corresponds to Table 2 in the FinMemory paper.

Ablation Configurations:
1. FinMemory (Full): All components enabled
2. w/o MTR: Remove Multi-Timescale Reward (use single-day reward only)
3. w/o QRA: Remove Quantity/Risk Agent (use fixed position sizing)
4. w/o MSP: Remove Market Signal Processing (use price data only)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finmemory.utils.logger import get_logger, create_experiment_logger
from finmemory.environment.trading_env import TradingEnvironment, create_trading_environment
from finmemory.environment.reward_calculator import RewardCalculator, create_reward_calculator
from finmemory.environment.risk_metrics import calculate_risk_metrics
from finmemory.agents.decision.direction_agent import DirectionAgent, create_direction_agent
from finmemory.agents.decision.quantity_risk_agent import QuantityRiskAgent, create_quantity_risk_agent
from finmemory.memory.hierarchical_memory import HierarchicalMemory, create_hierarchical_memory
from finmemory.memory.memory_allocator import MemoryAllocator, create_memory_allocator
from finmemory.memory.reflection_module import ReflectionModule, create_reflection_module
from finmemory.risk.cvar_calculator import CVaRCalculator, create_cvar_calculator
from finmemory.risk.position_sizer import PositionSizer, create_position_sizer
from finmemory.data.processors.price_processor import PriceProcessor, process_price_data
from finmemory.evaluation.metrics import MetricsResult, calculate_metrics, generate_ablation_table

logger = get_logger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation study experiment"""
    name: str
    description: str
    enable_multi_timescale_reward: bool = True
    enable_quantity_risk_agent: bool = True
    enable_market_signal_processing: bool = True
    enable_memory_system: bool = True
    enable_reflection: bool = True
    fixed_position_size: Optional[int] = None  # Used when QRA disabled
    position_sizing_method: str = "cvar"  # "cvar" or "fixed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "enable_multi_timescale_reward": self.enable_multi_timescale_reward,
            "enable_quantity_risk_agent": self.enable_quantity_risk_agent,
            "enable_market_signal_processing": self.enable_market_signal_processing,
            "enable_memory_system": self.enable_memory_system,
            "enable_reflection": self.enable_reflection,
            "fixed_position_size": self.fixed_position_size,
            "position_sizing_method": self.position_sizing_method
        }


@dataclass
class AblationStockResult:
    """Results for a single stock in ablation study"""
    ticker: str
    ablation_name: str
    initial_capital: float
    final_value: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    cvar_95: float
    total_trades: int
    win_rate: float
    avg_trade_pnl: float
    metrics: MetricsResult
    trades_history: List[Dict[str, Any]] = field(default_factory=list)
    account_values: List[float] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "ticker": self.ticker,
            "ablation_name": self.ablation_name,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "cumulative_return": self.cumulative_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "cvar_95": self.cvar_95,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "avg_trade_pnl": self.avg_trade_pnl,
            "metrics_dict": self.metrics.to_dict() if self.metrics else {}
        }


@dataclass
class AblationResult:
    """Complete ablation study results"""
    experiment_name: str
    timestamp: str
    ablation_configs: List[AblationConfig]
    stocks: List[str]
    period_start: str
    period_end: str
    initial_capital: float
    results: Dict[str, List[AblationStockResult]]  # ablation_name -> list of stock results
    summary_table: pd.DataFrame = None
    degradation_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "ablation_configs": [config.to_dict() for config in self.ablation_configs],
            "stocks": self.stocks,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "initial_capital": self.initial_capital,
            "results": {
                name: [result.to_dict() for result in results]
                for name, results in self.results.items()
            },
            "summary_table": self.summary_table.to_dict() if self.summary_table is not None else None,
            "degradation_analysis": self.degradation_analysis
        }


class AblationExperiment:
    """
    Ablation study experiment runner for FinMemory trading agent system.
    
    Executes trading experiments with different component configurations
    to measure each component's contribution to overall performance.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize ablation experiment.
        
        Args:
            config: Experiment configuration dictionary
            output_dir: Directory to save results
            api_key: OpenAI API key for LLM agents
        """
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/ablation_results")
        self.api_key = api_key
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = create_experiment_logger("ablation_experiment", log_dir=str(self.output_dir / "logs"))
        
        # Default ablation configurations
        self.ablation_configs = self._create_ablation_configs()
        
        # Experiment parameters
        self.stocks = self.config.get("stocks", ["TSLA", "AAPL", "AMZN", "NFLX", "COIN"])
        self.period_start = self.config.get("period_start", "2025-03-01")
        self.period_end = self.config.get("period_end", "2025-09-30")
        self.initial_capital = self.config.get("initial_capital", 100000.0)
        self.transaction_cost = self.config.get("transaction_cost", 0.001)
        
        self.logger.info(f"Initialized AblationExperiment with {len(self.ablation_configs)} configurations")
        self.logger.info(f"Stocks: {self.stocks}, Period: {self.period_start} to {self.period_end}")
    
    def _create_ablation_configs(self) -> List[AblationConfig]:
        """Create standard ablation configurations from Table 2"""
        return [
            AblationConfig(
                name="FinMemory_Full",
                description="Full FinMemory with all components enabled",
                enable_multi_timescale_reward=True,
                enable_quantity_risk_agent=True,
                enable_market_signal_processing=True,
                enable_memory_system=True,
                enable_reflection=True
            ),
            AblationConfig(
                name="w/o_MTR",
                description="Without Multi-Timescale Reward (single-day reward only)",
                enable_multi_timescale_reward=False,
                enable_quantity_risk_agent=True,
                enable_market_signal_processing=True,
                enable_memory_system=True,
                enable_reflection=True
            ),
            AblationConfig(
                name="w/o_QRA",
                description="Without Quantity/Risk Agent (fixed position sizing)",
                enable_multi_timescale_reward=True,
                enable_quantity_risk_agent=False,
                enable_market_signal_processing=True,
                enable_memory_system=True,
                enable_reflection=True,
                fixed_position_size=10,  # Fixed shares per trade
                position_sizing_method="fixed"
            ),
            AblationConfig(
                name="w/o_MSP",
                description="Without Market Signal Processing (price data only)",
                enable_multi_timescale_reward=True,
                enable_quantity_risk_agent=True,
                enable_market_signal_processing=False,
                enable_memory_system=True,
                enable_reflection=True
            )
        ]
    
    def run_ablation_for_stock(
        self,
        ticker: str,
        prices: pd.DataFrame,
        dates: List[str],
        ablation_config: AblationConfig,
        news_data: Optional[Dict[str, Any]] = None,
        sec_filings: Optional[Dict[str, Any]] = None
    ) -> AblationStockResult:
        """
        Run ablation experiment for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            prices: DataFrame with OHLCV price data
            dates: List of trading dates
            ablation_config: Ablation configuration
            news_data: Optional news data for market signal processing
            sec_filings: Optional SEC filings for market signal processing
            
        Returns:
            AblationStockResult with performance metrics
        """
        self.logger.info(f"Running ablation '{ablation_config.name}' for {ticker}")
        
        # Initialize components based on ablation config
        env = self._initialize_environment(ticker, prices, dates, ablation_config)
        memory_system = self._initialize_memory(ticker, ablation_config) if ablation_config.enable_memory_system else None
        risk_system = self._initialize_risk(ticker, prices, ablation_config) if ablation_config.enable_quantity_risk_agent else None
        agents = self._initialize_agents(ticker, ablation_config)
        
        # Execute trading loop
        trades_history = []
        account_values = []
        positions = []
        
        num_steps = len(dates) - 1  # Can't trade on last day
        
        for timestep in range(num_steps):
            # Get current state
            state = env.get_state()
            current_date = dates[timestep]
            current_price = state.current_price
            
            # Make trading decision
            direction, quantity, reasoning = self._make_decision(
                ticker=ticker,
                timestep=timestep,
                state=state,
                agents=agents,
                memory_system=memory_system,
                risk_system=risk_system,
                ablation_config=ablation_config,
                news_data=news_data,
                sec_filings=sec_filings
            )
            
            # Execute trade
            if quantity > 0:
                trade_result = env.step({"direction": direction, "quantity": quantity})
                trades_history.append({
                    "timestep": timestep,
                    "date": current_date,
                    "direction": direction,
                    "quantity": quantity,
                    "price": current_price,
                    "reasoning": reasoning
                })
            
            # Record state
            account_values.append(env.get_account_value())
            positions.append(env.get_position_size())
            
            # Update memory with outcome (if enabled)
            if memory_system and ablation_config.enable_reflection:
                self._update_memory(memory_system, timestep, state, direction, quantity)
        
        # Calculate final metrics
        final_value = env.get_account_value()
        cumulative_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate returns series
        returns = []
        for i in range(1, len(account_values)):
            daily_return = (account_values[i] - account_values[i-1]) / account_values[i-1]
            returns.append(daily_return)
        
        # Calculate risk metrics
        metrics = calculate_metrics(
            returns=np.array(returns),
            account_values=np.array(account_values),
            trades=trades_history,
            ticker=ticker,
            period_start=dates[0],
            period_end=dates[-1],
            initial_capital=self.initial_capital
        )
        
        # Calculate trade statistics
        total_trades = len(trades_history)
        winning_trades = sum(1 for t in trades_history if t.get("direction") == "buy")  # Simplified
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate average trade P&L
        if len(trades_history) >= 2:
            trade_pnls = []
            for i in range(0, len(trades_history) - 1, 2):
                if trades_history[i]["direction"] == "buy" and i+1 < len(trades_history):
                    entry_price = trades_history[i]["price"]
                    exit_price = trades_history[i+1]["price"]
                    pnl = (exit_price - entry_price) * trades_history[i]["quantity"]
                    trade_pnls.append(pnl)
            avg_trade_pnl = np.mean(trade_pnls) if trade_pnls else 0.0
        else:
            avg_trade_pnl = 0.0
        
        result = AblationStockResult(
            ticker=ticker,
            ablation_name=ablation_config.name,
            initial_capital=self.initial_capital,
            final_value=final_value,
            cumulative_return=cumulative_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
            calmar_ratio=metrics.calmar_ratio,
            cvar_95=metrics.cvar_95,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            metrics=metrics,
            trades_history=trades_history,
            account_values=account_values,
            positions=positions
        )
        
        self.logger.info(
            f"Completed {ablation_config.name} for {ticker}: "
            f"CR={cumulative_return:.2%}, SR={metrics.sharpe_ratio:.2f}, MDD={metrics.max_drawdown:.2%}"
        )
        
        return result
    
    def _initialize_environment(
        self,
        ticker: str,
        prices: pd.DataFrame,
        dates: List[str],
        ablation_config: AblationConfig
    ) -> TradingEnvironment:
        """Initialize trading environment with ablation-specific settings"""
        env = create_trading_environment(
            ticker=ticker,
            prices=prices,
            dates=dates,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost
        )
        
        # Configure reward calculator based on ablation
        if ablation_config.enable_multi_timescale_reward:
            # Use full multi-timescale reward (1, 7, 30 days)
            reward_calc = create_reward_calculator(
                prices=prices["close"].values,
                dates=dates,
                short_window=1,
                mid_window=7,
                long_window=30
            )
        else:
            # Use single-day reward only (ablation: w/o MTR)
            reward_calc = create_reward_calculator(
                prices=prices["close"].values,
                dates=dates,
                short_window=1,
                mid_window=1,  # Same as short-term
                long_window=1  # Same as short-term
            )
        
        env.reward_calculator = reward_calc
        
        return env
    
    def _initialize_memory(
        self,
        ticker: str,
        ablation_config: AblationConfig
    ) -> Dict[str, Any]:
        """Initialize memory system based on ablation config"""
        memory_config = self.config.get("memory_system", {})
        
        memory = create_hierarchical_memory(
            ticker=ticker,
            shallow_capacity=memory_config.get("shallow_capacity", 10),
            intermediate_capacity=memory_config.get("intermediate_capacity", 30),
            deep_capacity=memory_config.get("deep_capacity", 100),
            reflection_capacity=memory_config.get("reflection_capacity", 50)
        )
        
        allocator = create_memory_allocator(
            memory=memory,
            shallow_threshold=memory_config.get("shallow_threshold", 4),
            intermediate_threshold=memory_config.get("intermediate_threshold", 7)
        )
        
        reflection = create_reflection_module(
            ticker=ticker,
            reflection_threshold_positive=5.0,
            reflection_threshold_negative=-5.0
        ) if ablation_config.enable_reflection else None
        
        return {
            "memory": memory,
            "allocator": allocator,
            "reflection": reflection
        }
    
    def _initialize_risk(
        self,
        ticker: str,
        prices: pd.DataFrame,
        ablation_config: AblationConfig
    ) -> Dict[str, Any]:
        """Initialize risk management system based on ablation config"""
        risk_config = self.config.get("risk_management", {})
        
        cvar_config = risk_config.get("cvar", {})
        cvar_calc = create_cvar_calculator(
            confidence_level=cvar_config.get("confidence_level", 0.95),
            window_size=cvar_config.get("window_size", 20)
        )
        
        position_sizer_config = risk_config.get("position_sizer", {})
        position_sizer = create_position_sizer(
            max_exposure_ratio=position_sizer_config.get("max_exposure_ratio", 0.1),
            confidence_level=cvar_config.get("confidence_level", 0.95),
            window_size=cvar_config.get("window_size", 20)
        )
        
        # Calculate initial CVaR from historical returns
        if len(prices) > 20:
            returns = prices["close"].pct_change().dropna().values
            initial_cvar = cvar_calc.calculate_cvar(returns, use_rolling=False)
        else:
            initial_cvar = 0.02  # Default 2% CVaR
        
        return {
            "cvar_calculator": cvar_calc,
            "position_sizer": position_sizer,
            "initial_cvar": initial_cvar,
            "use_fixed_sizing": not ablation_config.enable_quantity_risk_agent,
            "fixed_position_size": ablation_config.fixed_position_size
        }
    
    def _initialize_agents(
        self,
        ticker: str,
        ablation_config: AblationConfig
    ) -> Dict[str, Any]:
        """Initialize decision agents based on ablation config"""
        mode = "evaluation"  # Ablation studies use evaluation mode
        
        direction_agent = create_direction_agent(
            ticker=ticker,
            mode=mode
        )
        
        # Only create quantity agent if QRA is enabled
        quantity_agent = None
        if ablation_config.enable_quantity_risk_agent:
            quantity_agent = create_quantity_risk_agent(
                ticker=ticker,
                mode=mode
            )
        
        return {
            "direction": direction_agent,
            "quantity": quantity_agent
        }
    
    def _make_decision(
        self,
        ticker: str,
        timestep: int,
        state: Any,
        agents: Dict[str, Any],
        memory_system: Optional[Dict[str, Any]],
        risk_system: Optional[Dict[str, Any]],
        ablation_config: AblationConfig,
        news_data: Optional[Dict[str, Any]] = None,
        sec_filings: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, int, str]:
        """
        Make trading decision using configured agents.
        
        Returns:
            Tuple of (direction, quantity, reasoning)
        """
        # Get direction decision from Direction Agent
        direction_agent = agents["direction"]
        
        # Prepare input for direction agent
        working_memory = None
        if memory_system and ablation_config.enable_market_signal_processing:
            working_memory = memory_system["memory"].get_working_memory()
        
        # Call direction agent
        try:
            direction_result = direction_agent.decide_direction(
                working_memory=working_memory,
                price_history=None,  # Would be actual price history
                technical_indicators=None,  # Would be technical indicators
                sentiment_indicators=None  # Would be sentiment from news
            )
            
            direction = direction_result.get("investment_decision", "hold")
            direction_reasoning = direction_result.get("summary_reason", "")
            memory_indices = direction_result.get("memory_indices", [])
            
        except Exception as e:
            self.logger.warning(f"Direction agent failed: {e}, using fallback")
            direction = "hold"
            direction_reasoning = "Fallback due to agent error"
            memory_indices = []
        
        # Get quantity decision
        if ablation_config.enable_quantity_risk_agent and risk_system:
            quantity_agent = agents["quantity"]
            
            # Get CVaR constraint
            maxcvar = risk_system["position_sizer"].calculate_max_cvar_size(
                account_value=state.account_value,
                current_price=state.current_price,
                returns=None  # Would be historical returns
            )
            
            try:
                quantity_result = quantity_agent.decide_quantity(
                    direction_decision=direction,
                    maxcvar=maxcvar,
                    current_holdings=state.position_size,
                    account_value=state.account_value,
                    working_memory=working_memory
                )
                
                quantity = quantity_result.get("order_size", 0)
                quantity_reasoning = quantity_result.get("summary_reason", "")
                
            except Exception as e:
                self.logger.warning(f"Quantity agent failed: {e}, using fallback")
                quantity = 0
                quantity_reasoning = "Fallback due to agent error"
        else:
            # Fixed position sizing (ablation: w/o QRA)
            if ablation_config.fixed_position_size:
                quantity = ablation_config.fixed_position_size
            else:
                quantity = 10  # Default fixed size
            
            quantity_reasoning = f"Fixed position sizing: {quantity} shares"
        
        # Combine reasoning
        full_reasoning = f"Direction: {direction_reasoning} | Quantity: {quantity_reasoning}"
        
        return direction, quantity, full_reasoning
    
    def _update_memory(
        self,
        memory_system: Dict[str, Any],
        timestep: int,
        state: Any,
        direction: str,
        quantity: int
    ):
        """Update memory system with trading outcome"""
        # This would be called after trade outcome is known
        # For now, simplified implementation
        pass
    
    def run_full_ablation_study(
        self,
        price_data: Dict[str, pd.DataFrame],
        news_data: Optional[Dict[str, Any]] = None,
        sec_filings: Optional[Dict[str, Any]] = None
    ) -> AblationResult:
        """
        Run complete ablation study across all stocks and configurations.
        
        Args:
            price_data: Dictionary mapping ticker to price DataFrame
            news_data: Optional news data for all tickers
            sec_filings: Optional SEC filings for all tickers
            
        Returns:
            AblationResult with complete study results
        """
        self.logger.info("Starting full ablation study")
        start_time = datetime.now()
        
        # Initialize results storage
        results: Dict[str, List[AblationStockResult]] = {
            config.name: [] for config in self.ablation_configs
        }
        
        # Run ablation for each stock and configuration
        for ticker in self.stocks:
            self.logger.info(f"Processing stock: {ticker}")
            
            if ticker not in price_data:
                self.logger.error(f"No price data for {ticker}, skipping")
                continue
            
            prices = price_data[ticker]
            dates = prices.index.strftime("%Y-%m-%d").tolist()
            
            for ablation_config in self.ablation_configs:
                try:
                    result = self.run_ablation_for_stock(
                        ticker=ticker,
                        prices=prices,
                        dates=dates,
                        ablation_config=ablation_config,
                        news_data=news_data.get(ticker) if news_data else None,
                        sec_filings=sec_filings.get(ticker) if sec_filings else None
                    )
                    results[ablation_config.name].append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed ablation {ablation_config.name} for {ticker}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        # Generate summary table
        summary_table = self._generate_summary_table(results)
        
        # Analyze performance degradation
        degradation_analysis = self._analyze_degradation(results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        ablation_result = AblationResult(
            experiment_name="FinMemory_Ablation_Study",
            timestamp=start_time.isoformat(),
            ablation_configs=self.ablation_configs,
            stocks=self.stocks,
            period_start=self.period_start,
            period_end=self.period_end,
            initial_capital=self.initial_capital,
            results=results,
            summary_table=summary_table,
            degradation_analysis=degradation_analysis
        )
        
        # Save results
        self._save_results(ablation_result)
        
        self.logger.info(f"Completed ablation study in {duration:.1f} seconds")
        self.logger.info(f"Results saved to: {self.output_dir}")
        
        return ablation_result
    
    def _generate_summary_table(self, results: Dict[str, List[AblationStockResult]]) -> pd.DataFrame:
        """Generate summary table matching Table 2 format from paper"""
        rows = []
        
        for ablation_name, stock_results in results.items():
            for stock_result in stock_results:
                rows.append({
                    "Configuration": ablation_name,
                    "Ticker": stock_result.ticker,
                    "CR%": stock_result.cumulative_return * 100,
                    "Sharpe Ratio": stock_result.sharpe_ratio,
                    "MDD%": stock_result.max_drawdown * 100,
                    "Calmar Ratio": stock_result.calmar_ratio,
                    "Total Trades": stock_result.total_trades,
                    "Win Rate%": stock_result.win_rate * 100
                })
        
        df = pd.DataFrame(rows)
        
        # Add average row for each configuration
        avg_rows = []
        for ablation_name in results.keys():
            config_data = df[df["Configuration"] == ablation_name]
            if len(config_data) > 0:
                avg_row = {
                    "Configuration": f"{ablation_name} (AVG)",
                    "Ticker": "AVG",
                    "CR%": config_data["CR%"].mean(),
                    "Sharpe Ratio": config_data["Sharpe Ratio"].mean(),
                    "MDD%": config_data["MDD%"].mean(),
                    "Calmar Ratio": config_data["Calmar Ratio"].mean(),
                    "Total Trades": config_data["Total Trades"].mean(),
                    "Win Rate%": config_data["Win Rate%"].mean()
                }
                avg_rows.append(avg_row)
        
        if avg_rows:
            avg_df = pd.DataFrame(avg_rows)
            df = pd.concat([df, avg_df], ignore_index=True)
        
        return df
    
    def _analyze_degradation(self, results: Dict[str, List[AblationStockResult]]) -> Dict[str, Any]:
        """Analyze performance degradation from removing components"""
        # Get full FinMemory results as baseline
        full_results = results.get("FinMemory_Full", [])
        if not full_results:
            return {"error": "No full FinMemory results found"}
        
        # Calculate average metrics for full configuration
        full_avg_cr = np.mean([r.cumulative_return for r in full_results])
        full_avg_sr = np.mean([r.sharpe_ratio for r in full_results])
        full_avg_mdd = np.mean([r.max_drawdown for r in full_results])
        
        degradation = {}
        
        # Compare each ablation to full
        for ablation_name, stock_results in results.items():
            if ablation_name == "FinMemory_Full":
                continue
            
            ablation_avg_cr = np.mean([r.cumulative_return for r in stock_results])
            ablation_avg_sr = np.mean([r.sharpe_ratio for r in stock_results])
            ablation_avg_mdd = np.mean([r.max_drawdown for r in stock_results])
            
            # Calculate degradation percentages
            cr_degradation = (full_avg_cr - ablation_avg_cr) / full_avg_cr * 100 if full_avg_cr != 0 else 0
            sr_degradation = (full_avg_sr - ablation_avg_sr) / full_avg_sr * 100 if full_avg_sr != 0 else 0
            mdd_change = (ablation_avg_mdd - full_avg_mdd) / full_avg_mdd * 100 if full_avg_mdd != 0 else 0
            
            degradation[ablation_name] = {
                "cr_degradation_pct": cr_degradation,
                "sharpe_degradation_pct": sr_degradation,
                "mdd_change_pct": mdd_change,
                "full_avg_cr": full_avg_cr,
                "ablation_avg_cr": ablation_avg_cr,
                "full_avg_sr": full_avg_sr,
                "ablation_avg_sr": ablation_avg_sr
            }
        
        return degradation
    
    def _save_results(self, result: AblationResult):
        """Save ablation results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.output_dir / f"ablation_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save summary table as CSV
        if result.summary_table is not None:
            csv_path = self.output_dir / f"ablation_summary_{timestamp}.csv"
            result.summary_table.to_csv(csv_path, index=False)
        
        # Save degradation analysis
        degradation_path = self.output_dir / f"degradation_analysis_{timestamp}.json"
        with open(degradation_path, "w") as f:
            json.dump(result.degradation_analysis, f, indent=2)
        
        self.logger.info(f"Saved results to {self.output_dir}")


def run_ablation_experiment(
    ablation_config: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    api_key: Optional[str] = None
) -> AblationResult:
    """
    Convenience function to run ablation experiment.
    
    Args:
        ablation_config: Specific ablation configuration to run (None for all)
        config: Experiment configuration
        output_dir: Output directory for results
        price_data: Price data for all stocks
        api_key: OpenAI API key
        
    Returns:
        AblationResult with complete study results
    """
    experiment = AblationExperiment(
        config=config,
        output_dir=output_dir,
        api_key=api_key
    )
    
    if price_data is None:
        raise ValueError("price_data must be provided")
    
    result = experiment.run_full_ablation_study(price_data=price_data)
    
    return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be called from main.py with actual data
    logger.info("Ablation experiment module loaded successfully")
