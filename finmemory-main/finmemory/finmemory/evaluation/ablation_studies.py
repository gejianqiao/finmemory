"""
Ablation Studies Module for FinMemory Trading Agent System.

This module provides comprehensive ablation study functionality to evaluate
the contribution of each component in the FinMemory system. It implements the
ablation experiments described in Table 2 of the paper, testing the impact
of removing key components: Multi-Timescale Reward (MTR), Quantity/Risk Agent
(QRA), and Market Signal Processing (MSP).

Paper Reference: Section 5 (Experiments), Table 2 (Ablation Study)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import numpy as np
import pandas as pd

from finmemory.utils.logger import get_logger
from finmemory.evaluation.metrics import MetricsResult, calculate_metrics, generate_ablation_table
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


logger = get_logger(__name__)


@dataclass
class AblationConfig:
    """
    Configuration for ablation study experiments.
    
    Attributes:
        name: Unique identifier for ablation configuration (e.g., "w/o_MTR")
        description: Human-readable description of what's ablated
        enable_multi_timescale_reward: Whether to use multi-timescale rewards (MTR)
        enable_quantity_risk_agent: Whether to use CVaR-constrained quantity agent (QRA)
        enable_market_signal_processing: Whether to use news/SEC filing analysis (MSP)
        fixed_position_size: Fixed position size when QRA is disabled (default: 10 shares)
        use_single_day_reward: Use only 1-day reward when MTR disabled
        price_only_mode: Use only price data when MSP disabled
    """
    name: str
    description: str
    enable_multi_timescale_reward: bool = True
    enable_quantity_risk_agent: bool = True
    enable_market_signal_processing: bool = True
    fixed_position_size: int = 10
    use_single_day_reward: bool = False
    price_only_mode: bool = False
    
    @classmethod
    def full_finmemory(cls) -> 'AblationConfig':
        """Create configuration for full FinMemory (all components enabled)."""
        return cls(
            name="Full_FinMemory",
            description="Full FinMemory system with all components enabled",
            enable_multi_timescale_reward=True,
            enable_quantity_risk_agent=True,
            enable_market_signal_processing=True
        )
    
    @classmethod
    def without_mtr(cls) -> 'AblationConfig':
        """Create configuration without Multi-Timescale Reward (MTR)."""
        return cls(
            name="w/o_MTR",
            description="FinMemory without Multi-Timescale Reward (single-day reward only)",
            enable_multi_timescale_reward=False,
            enable_quantity_risk_agent=True,
            enable_market_signal_processing=True,
            use_single_day_reward=True
        )
    
    @classmethod
    def without_qra(cls) -> 'AblationConfig':
        """Create configuration without Quantity/Risk Agent (QRA)."""
        return cls(
            name="w/o_QRA",
            description="FinMemory without Quantity/Risk Agent (fixed position sizing)",
            enable_multi_timescale_reward=True,
            enable_quantity_risk_agent=False,
            enable_market_signal_processing=True,
            fixed_position_size=10
        )
    
    @classmethod
    def without_msp(cls) -> 'AblationConfig':
        """Create configuration without Market Signal Processing (MSP)."""
        return cls(
            name="w/o_MSP",
            description="FinMemory without Market Signal Processing (price data only)",
            enable_multi_timescale_reward=True,
            enable_quantity_risk_agent=True,
            enable_market_signal_processing=False,
            price_only_mode=True
        )
    
    @classmethod
    def get_all_configs(cls) -> List['AblationConfig']:
        """Get all standard ablation configurations for Table 2."""
        return [
            cls.full_finmemory(),
            cls.without_mtr(),
            cls.without_qra(),
            cls.without_msp()
        ]


@dataclass
class AblationStockResult:
    """
    Results for ablation study on a single stock.
    
    Attributes:
        ticker: Stock ticker symbol
        ablation_name: Name of ablation configuration
        ablation_description: Description of ablation configuration
        cumulative_return: Cumulative return percentage
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown percentage
        calmar_ratio: Calmar ratio
        cvar_95: 95% CVaR
        total_trades: Total number of trades executed
        win_rate: Win rate percentage
        metrics: Full metrics result object
    """
    ticker: str
    ablation_name: str
    ablation_description: str
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    cvar_95: float
    total_trades: int
    win_rate: float
    metrics: MetricsResult
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ticker': self.ticker,
            'ablation_name': self.ablation_name,
            'ablation_description': self.ablation_description,
            'cumulative_return': self.cumulative_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'cvar_95': self.cvar_95,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'metrics_dict': self.metrics.to_dict() if self.metrics else None
        }


@dataclass
class AblationResult:
    """
    Complete ablation study results.
    
    Attributes:
        experiment_name: Name of the ablation experiment
        experiment_date: Date experiment was run
        ablation_configs: List of ablation configurations tested
        stock_results: List of per-stock ablation results
        summary_table: Pandas DataFrame with summary statistics
        degradation_analysis: Analysis of performance degradation per component
    """
    experiment_name: str
    experiment_date: str
    ablation_configs: List[AblationConfig]
    stock_results: List[AblationStockResult]
    summary_table: pd.DataFrame
    degradation_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'experiment_date': self.experiment_date,
            'ablation_configs': [
                {
                    'name': cfg.name,
                    'description': cfg.description,
                    'enable_mtr': cfg.enable_multi_timescale_reward,
                    'enable_qra': cfg.enable_quantity_risk_agent,
                    'enable_msp': cfg.enable_market_signal_processing
                }
                for cfg in self.ablation_configs
            ],
            'stock_results': [result.to_dict() for result in self.stock_results],
            'summary_table': self.summary_table.to_dict() if self.summary_table is not None else None,
            'degradation_analysis': self.degradation_analysis
        }
    
    def save(self, output_dir: str) -> None:
        """Save results to JSON and CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_path / f"ablation_results_{self.experiment_date}.json"
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Ablation results saved to {json_path}")
        
        # Save summary table CSV
        if self.summary_table is not None:
            csv_path = output_path / f"ablation_summary_{self.experiment_date}.csv"
            self.summary_table.to_csv(csv_path)
            logger.info(f"Ablation summary saved to {csv_path}")


class AblationStudyRunner:
    """
    Runner for ablation study experiments.
    
    This class executes trading simulations with different component
    configurations to measure the contribution of each FinMemory component.
    
    Usage:
        runner = AblationStudyRunner(config, output_dir, api_key)
        result = runner.run_full_study(price_data, tickers)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "evaluation/results/ablation",
        api_key: Optional[str] = None
    ):
        """
        Initialize ablation study runner.
        
        Args:
            config: Configuration dictionary from environment.yaml/agents.yaml
            output_dir: Directory to save results
            api_key: OpenAI API key for LLM agents
        """
        self.config = config
        self.output_dir = output_dir
        self.api_key = api_key
        self.logger = get_logger(__name__)
        
        # Default experiment parameters
        self.initial_capital = config.get('trading_environment', {}).get('initial_capital', 100000)
        self.transaction_cost = config.get('trading_environment', {}).get('transaction_cost', 0.001)
        
    def run_ablation_for_stock(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        ablation_config: AblationConfig
    ) -> AblationStockResult:
        """
        Run ablation experiment for a single stock and configuration.
        
        Args:
            ticker: Stock ticker symbol
            price_data: DataFrame with OHLCV price data
            ablation_config: Ablation configuration to test
            
        Returns:
            AblationStockResult with performance metrics
        """
        self.logger.info(
            f"Running ablation '{ablation_config.name}' for {ticker}: {ablation_config.description}"
        )
        
        # Initialize components based on ablation configuration
        env = self._initialize_environment(ticker, price_data)
        reward_calc = self._initialize_reward_calculator(price_data, ablation_config)
        memory_system = self._initialize_memory(ticker)
        risk_system = self._initialize_risk(ablation_config)
        agents = self._initialize_agents(ticker, ablation_config)
        
        # Run trading simulation
        trades = []
        account_values = [self.initial_capital]
        positions = []
        
        num_steps = len(price_data) - 1
        
        for timestep in range(num_steps):
            # Get current state
            state = env.get_state()
            current_price = state['current_price']
            
            # Get direction decision from Direction Agent
            direction_result = self._get_direction_decision(
                agents['direction_agent'],
                state,
                ticker,
                timestep,
                ablation_config
            )
            direction = direction_result.get('investment_decision', 'hold')
            
            # Get quantity decision
            if ablation_config.enable_quantity_risk_agent:
                quantity_result = self._get_quantity_decision(
                    agents['quantity_risk_agent'],
                    direction,
                    state,
                    risk_system['cvar_calculator'],
                    risk_system['position_sizer'],
                    timestep
                )
                quantity = quantity_result.get('order_size', 0)
            else:
                # Use fixed position size when QRA disabled
                quantity = self._get_fixed_quantity(
                    direction,
                    ablation_config.fixed_position_size,
                    state
                )
            
            # Execute trade
            trade_result = env.step({
                'direction': direction,
                'quantity': quantity
            })
            
            if trade_result['success']:
                trades.append(trade_result)
            
            # Calculate reward (for training mode)
            if ablation_config.enable_multi_timescale_reward:
                reward_result = reward_calc.calculate_reward(
                    position=state['position_size'],
                    timestep=timestep,
                    action_taken=(quantity > 0)
                )
            else:
                # Single-day reward only
                reward_result = reward_calc.calculate_reward(
                    position=state['position_size'],
                    timestep=timestep,
                    action_taken=(quantity > 0),
                    use_single_day=True
                )
            
            # Update memory with reflection if needed
            self._update_memory(
                memory_system['memory'],
                memory_system['reflection'],
                timestep,
                trade_result,
                direction,
                quantity
            )
            
            # Track account value
            account_values.append(env.get_account_value())
            positions.append(state['position_size'])
        
        # Calculate final metrics
        returns = env.calculate_return()
        metrics_result = calculate_metrics(
            returns=returns,
            account_values=account_values,
            trades=trades,
            ticker=ticker,
            period_start=str(price_data.index[0].date()) if hasattr(price_data.index[0], 'date') else str(price_data.index[0]),
            period_end=str(price_data.index[-1].date()) if hasattr(price_data.index[-1], 'date') else str(price_data.index[-1]),
            initial_capital=self.initial_capital
        )
        
        # Create result object
        result = AblationStockResult(
            ticker=ticker,
            ablation_name=ablation_config.name,
            ablation_description=ablation_config.description,
            cumulative_return=metrics_result.cumulative_return,
            sharpe_ratio=metrics_result.sharpe_ratio,
            max_drawdown=metrics_result.max_drawdown,
            calmar_ratio=metrics_result.calmar_ratio,
            cvar_95=metrics_result.cvar_95,
            total_trades=metrics_result.total_trades,
            win_rate=metrics_result.win_rate,
            metrics=metrics_result
        )
        
        self.logger.info(
            f"Ablation '{ablation_config.name}' for {ticker}: "
            f"CR={result.cumulative_return:.2f}%, SR={result.sharpe_ratio:.2f}, "
            f"MDD={result.max_drawdown:.2f}%"
        )
        
        return result
    
    def run_full_study(
        self,
        price_data_dict: Dict[str, pd.DataFrame],
        tickers: Optional[List[str]] = None
    ) -> AblationResult:
        """
        Run complete ablation study across all stocks and configurations.
        
        Args:
            price_data_dict: Dictionary mapping ticker to price DataFrame
            tickers: List of tickers to test (default: all in price_data_dict)
            
        Returns:
            AblationResult with complete study results
        """
        if tickers is None:
            tickers = list(price_data_dict.keys())
        
        ablation_configs = AblationConfig.get_all_configs()
        stock_results = []
        
        self.logger.info(f"Starting ablation study on {len(tickers)} stocks")
        self.logger.info(f"Testing {len(ablation_configs)} configurations: {[cfg.name for cfg in ablation_configs]}")
        
        for ticker in tickers:
            if ticker not in price_data_dict:
                self.logger.warning(f"Price data not found for {ticker}, skipping")
                continue
            
            price_data = price_data_dict[ticker]
            
            for config in ablation_configs:
                try:
                    result = self.run_ablation_for_stock(ticker, price_data, config)
                    stock_results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Error running ablation '{config.name}' for {ticker}: {str(e)}",
                        exc_info=True
                    )
                    # Create fallback result with NaN metrics
                    stock_results.append(AblationStockResult(
                        ticker=ticker,
                        ablation_name=config.name,
                        ablation_description=config.description,
                        cumulative_return=np.nan,
                        sharpe_ratio=np.nan,
                        max_drawdown=np.nan,
                        calmar_ratio=np.nan,
                        cvar_95=np.nan,
                        total_trades=0,
                        win_rate=np.nan,
                        metrics=None
                    ))
        
        # Generate summary table
        summary_table = self._generate_summary_table(stock_results, ablation_configs, tickers)
        
        # Analyze degradation
        degradation_analysis = self._analyze_degradation(stock_results, ablation_configs)
        
        # Create result object
        experiment_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = AblationResult(
            experiment_name="FinMemory_Ablation_Study",
            experiment_date=experiment_date,
            ablation_configs=ablation_configs,
            stock_results=stock_results,
            summary_table=summary_table,
            degradation_analysis=degradation_analysis
        )
        
        # Save results
        result.save(self.output_dir)
        
        self.logger.info(f"Ablation study completed. Results saved to {self.output_dir}")
        
        return result
    
    def _initialize_environment(
        self,
        ticker: str,
        price_data: pd.DataFrame
    ) -> TradingEnvironment:
        """Initialize trading environment."""
        prices = price_data['Close'].values
        dates = price_data.index
        
        env = create_trading_environment(
            ticker=ticker,
            prices=prices,
            dates=dates,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost
        )
        
        return env
    
    def _initialize_reward_calculator(
        self,
        price_data: pd.DataFrame,
        ablation_config: AblationConfig
    ) -> RewardCalculator:
        """Initialize reward calculator based on ablation configuration."""
        prices = price_data['Close'].values
        dates = price_data.index
        
        if ablation_config.enable_multi_timescale_reward:
            # Full multi-timescale reward
            reward_calc = create_reward_calculator(
                prices=prices,
                dates=dates,
                short_window=1,
                mid_window=7,
                long_window=30
            )
        else:
            # Single-day reward only
            reward_calc = create_reward_calculator(
                prices=prices,
                dates=dates,
                short_window=1,
                mid_window=1,
                long_window=1
            )
        
        return reward_calc
    
    def _initialize_memory(self, ticker: str) -> Dict[str, Any]:
        """Initialize memory system."""
        memory = create_hierarchical_memory(
            ticker=ticker,
            shallow_capacity=10,
            intermediate_capacity=30,
            deep_capacity=100,
            reflection_capacity=50
        )
        
        allocator = create_memory_allocator(
            memory=memory,
            shallow_threshold=4,
            intermediate_threshold=7
        )
        
        reflection = create_reflection_module(
            ticker=ticker,
            reflection_threshold_positive=5.0,
            reflection_threshold_negative=-5.0
        )
        
        return {
            'memory': memory,
            'allocator': allocator,
            'reflection': reflection
        }
    
    def _initialize_risk(self, ablation_config: AblationConfig) -> Dict[str, Any]:
        """Initialize risk management system."""
        cvar_calc = create_cvar_calculator(
            confidence_level=0.95,
            window_size=20,
            min_samples=5
        )
        
        position_sizer = create_position_sizer(
            max_exposure_ratio=0.1,
            confidence_level=0.95,
            window_size=20
        )
        
        return {
            'cvar_calculator': cvar_calc,
            'position_sizer': position_sizer
        }
    
    def _initialize_agents(
        self,
        ticker: str,
        ablation_config: AblationConfig
    ) -> Dict[str, Any]:
        """Initialize decision agents based on ablation configuration."""
        mode = "evaluation"
        
        direction_agent = create_direction_agent(
            ticker=ticker,
            mode=mode
        )
        
        if ablation_config.enable_quantity_risk_agent:
            quantity_agent = create_quantity_risk_agent(
                ticker=ticker,
                mode=mode
            )
        else:
            quantity_agent = None
        
        return {
            'direction_agent': direction_agent,
            'quantity_risk_agent': quantity_agent
        }
    
    def _get_direction_decision(
        self,
        direction_agent: DirectionAgent,
        state: Dict[str, Any],
        ticker: str,
        timestep: int,
        ablation_config: AblationConfig
    ) -> Dict[str, Any]:
        """Get direction decision from Direction Agent."""
        # In full MSP mode, would include news/SEC analysis
        # In price-only mode, only use price features
        
        try:
            result = direction_agent.decide_direction(
                working_memory=[],  # Would be populated in full implementation
                price_history=state.get('price_history', []),
                technical_indicators=state.get('technical_indicators', {}),
                sentiment_indicators={} if ablation_config.price_only_mode else state.get('sentiment', {})
            )
            return result
        except Exception as e:
            self.logger.warning(f"Direction agent failed, using fallback: {e}")
            return {'investment_decision': 'hold', 'summary_reason': 'API fallback'}
    
    def _get_quantity_decision(
        self,
        quantity_agent: QuantityRiskAgent,
        direction: str,
        state: Dict[str, Any],
        cvar_calc: CVaRCalculator,
        position_sizer: PositionSizer,
        timestep: int
    ) -> Dict[str, Any]:
        """Get quantity decision from Quantity/Risk Agent."""
        try:
            # Calculate CVaR constraint
            returns = state.get('recent_returns', [])
            if len(returns) > 0:
                cvar_result = cvar_calc.calculate_cvar(returns)
                maxcvar = int(self.initial_capital * 0.1 / max(cvar_result.cvar, 0.01))
            else:
                maxcvar = 100  # Default max when no return history
            
            result = quantity_agent.decide_quantity(
                direction_decision=direction,
                maxcvar=maxcvar,
                current_holdings=state.get('position_size', 0),
                account_value=state.get('account_value', self.initial_capital),
                current_price=state.get('current_price', 100),
                working_memory=[]
            )
            return result
        except Exception as e:
            self.logger.warning(f"Quantity agent failed, using fallback: {e}")
            return {'order_size': 0, 'summary_reason': 'API fallback'}
    
    def _get_fixed_quantity(
        self,
        direction: str,
        fixed_size: int,
        state: Dict[str, Any]
    ) -> int:
        """Get fixed quantity when QRA is disabled."""
        if direction == 'hold':
            return 0
        elif direction == 'buy':
            # Respect cash constraint
            max_affordable = int(state.get('cash', 0) / max(state.get('current_price', 1), 1))
            return min(fixed_size, max_affordable)
        else:  # sell
            return min(fixed_size, state.get('position_size', 0))
    
    def _update_memory(
        self,
        memory: HierarchicalMemory,
        reflection: ReflectionModule,
        timestep: int,
        trade_result: Dict[str, Any],
        direction: str,
        quantity: int
    ) -> None:
        """Update memory with reflection if trade was significant."""
        # Simplified memory update - full implementation would track P&L
        if quantity > 0 and trade_result.get('success', False):
            # Record outcome for potential reflection
            pass  # Full implementation would calculate P&L and record outcome
    
    def _generate_summary_table(
        self,
        stock_results: List[AblationStockResult],
        ablation_configs: List[AblationConfig],
        tickers: List[str]
    ) -> pd.DataFrame:
        """Generate summary table matching Table 2 format from paper."""
        # Create pivot table: rows=ablation configs, columns=stocks, values=CR%
        data = []
        
        for config in ablation_configs:
            row = {'Ablation_Config': config.name, 'Description': config.description}
            
            for ticker in tickers:
                # Find matching result
                matching_results = [
                    r for r in stock_results
                    if r.ticker == ticker and r.ablation_name == config.name
                ]
                
                if matching_results:
                    result = matching_results[0]
                    row[f'{ticker}_CR'] = result.cumulative_return
                    row[f'{ticker}_SR'] = result.sharpe_ratio
                    row[f'{ticker}_MDD'] = result.max_drawdown
                else:
                    row[f'{ticker}_CR'] = np.nan
                    row[f'{ticker}_SR'] = np.nan
                    row[f'{ticker}_MDD'] = np.nan
            
            # Calculate average CR across stocks
            cr_cols = [f'{ticker}_CR' for ticker in tickers]
            row['Avg_CR'] = np.nanmean([row[col] for col in cr_cols])
            
            # Calculate degradation from full FinMemory
            if config.name != 'Full_FinMemory':
                full_results = [
                    r for r in stock_results
                    if r.ablation_name == 'Full_FinMemory'
                ]
                if full_results:
                    avg_full_cr = np.nanmean([r.cumulative_return for r in full_results])
                    row['Degradation_%'] = ((row['Avg_CR'] - avg_full_cr) / avg_full_cr) * 100
                else:
                    row['Degradation_%'] = np.nan
            else:
                row['Degradation_%'] = 0.0
            
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def _analyze_degradation(
        self,
        stock_results: List[AblationStockResult],
        ablation_configs: List[AblationConfig]
    ) -> Dict[str, Any]:
        """Analyze performance degradation for each ablated component."""
        analysis = {
            'component_impact': {},
            'rankings': {},
            'statistical_significance': {}
        }
        
        # Get full FinMemory baseline
        full_results = [r for r in stock_results if r.ablation_name == 'Full_FinMemory']
        if not full_results:
            return analysis
        
        full_avg_cr = np.nanmean([r.cumulative_return for r in full_results])
        full_avg_sr = np.nanmean([r.sharpe_ratio for r in full_results])
        full_avg_mdd = np.nanmean([r.max_drawdown for r in full_results])
        
        # Analyze each ablation
        for config in ablation_configs:
            if config.name == 'Full_FinMemory':
                continue
            
            ablation_results = [r for r in stock_results if r.ablation_name == config.name]
            if not ablation_results:
                continue
            
            avg_cr = np.nanmean([r.cumulative_return for r in ablation_results])
            avg_sr = np.nanmean([r.sharpe_ratio for r in ablation_results])
            avg_mdd = np.nanmean([r.max_drawdown for r in ablation_results])
            
            # Calculate degradation percentages
            cr_degradation = ((avg_cr - full_avg_cr) / full_avg_cr) * 100
            sr_degradation = ((avg_sr - full_avg_sr) / full_avg_sr) * 100 if full_avg_sr != 0 else 0
            mdd_change = avg_mdd - full_avg_mdd  # Absolute change in MDD
            
            # Identify which component was ablated
            if not config.enable_multi_timescale_reward:
                component = 'MTR'
            elif not config.enable_quantity_risk_agent:
                component = 'QRA'
            elif not config.enable_market_signal_processing:
                component = 'MSP'
            else:
                component = 'Unknown'
            
            analysis['component_impact'][component] = {
                'cr_degradation_pct': cr_degradation,
                'sr_degradation_pct': sr_degradation,
                'mdd_change_pct': mdd_change,
                'ablation_name': config.name
            }
        
        # Rank components by impact (largest degradation = most important)
        rankings = sorted(
            analysis['component_impact'].items(),
            key=lambda x: x[1]['cr_degradation_pct'],
            reverse=True  # Most negative (largest drop) first
        )
        analysis['rankings']['by_cr_degradation'] = [r[0] for r in rankings]
        analysis['rankings']['most_critical_component'] = rankings[0][0] if rankings else None
        
        return analysis


def run_ablation_studies(
    price_data_dict: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "evaluation/results/ablation",
    api_key: Optional[str] = None,
    tickers: Optional[List[str]] = None
) -> AblationResult:
    """
    Convenience function to run complete ablation studies.
    
    Args:
        price_data_dict: Dictionary mapping ticker to price DataFrame
        config: Configuration dictionary (optional, uses defaults if None)
        output_dir: Directory to save results
        api_key: OpenAI API key
        tickers: List of tickers to test (default: all in price_data_dict)
        
    Returns:
        AblationResult with complete study results
        
    Example:
        >>> price_data = {'TSLA': tsla_df, 'AAPL': aapl_df}
        >>> result = run_ablation_studies(price_data, tickers=['TSLA', 'AAPL'])
        >>> print(result.summary_table)
    """
    if config is None:
        # Default configuration
        config = {
            'trading_environment': {
                'initial_capital': 100000,
                'transaction_cost': 0.001
            }
        }
    
    runner = AblationStudyRunner(
        config=config,
        output_dir=output_dir,
        api_key=api_key
    )
    
    result = runner.run_full_study(price_data_dict, tickers)
    
    return result


def compare_ablation_results(
    result1: AblationResult,
    result2: AblationResult
) -> pd.DataFrame:
    """
    Compare results from two ablation studies.
    
    Args:
        result1: First ablation study result
        result2: Second ablation study result
        
    Returns:
        DataFrame comparing key metrics between studies
    """
    comparison_data = []
    
    for ablation_name in set(
        [r.ablation_name for r in result1.stock_results] +
        [r.ablation_name for r in result2.stock_results]
    ):
        results1 = [r for r in result1.stock_results if r.ablation_name == ablation_name]
        results2 = [r for r in result2.stock_results if r.ablation_name == ablation_name]
        
        if results1 and results2:
            avg_cr1 = np.nanmean([r.cumulative_return for r in results1])
            avg_cr2 = np.nanmean([r.cumulative_return for r in results2])
            
            comparison_data.append({
                'Ablation': ablation_name,
                'Study1_Avg_CR': avg_cr1,
                'Study2_Avg_CR': avg_cr2,
                'Difference': avg_cr2 - avg_cr1,
                'Difference_Pct': ((avg_cr2 - avg_cr1) / avg_cr1 * 100) if avg_cr1 != 0 else 0
            })
    
    return pd.DataFrame(comparison_data)


__all__ = [
    'AblationConfig',
    'AblationStockResult',
    'AblationResult',
    'AblationStudyRunner',
    'run_ablation_studies',
    'compare_ablation_results'
]
