# FinMemory: A Position-Aware Trading Agent System for Real Financial Markets

A position-aware LLM trading agent system that explicitly models and manages continuous positions through dual-agent decision architecture (Direction Agent + Quantity/Risk Agent) and multi-timescale reward signals (1-day, 7-day, 30-day).

## 📋 Table of Contents

- [Overview](#overview)
- [Current Status](#current-status)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

FinMemory is a sophisticated trading agent system designed for real financial markets. Unlike traditional trading agents that treat each decision independently, FinMemory explicitly models and manages **continuous positions** over time, enabling more realistic and effective trading strategies.

## ✅ Current Status

The repository now has an executable end-to-end pipeline:

```text
Yahoo Finance / Finnhub / SEC EDGAR
→ deterministic preprocessing and signal aggregation
→ Market Context Agent
→ hierarchical memory
→ Direction Agent
→ Quantity/Risk Agent
→ trading environment
→ metrics and reflection updates
```

The data loaders, simplified three-agent loop, risk components, memory, and reflection integration are covered by automated tests. Run them with:

```bash
python -m pytest -q
```

Real Yahoo Finance prices and SEC metadata have been verified. A fully live LLM run still requires valid OpenAI and Finnhub API keys. The simplified architecture makes three LLM calls per trading day instead of processing every article with separate filtering and analysis agents.

This project is an evaluation/backtesting framework. It does not submit orders to a brokerage account.

### What You Need to Run It

You do not need to download datasets manually. A normal run automatically downloads and caches:

- Daily OHLCV prices from Yahoo Finance
- Company and macro news from Finnhub
- 10-K and 10-Q filings from SEC EDGAR

You only need:

1. Python 3.10+ and the packages in `requirements.txt`
2. A valid `OPENAI_API_KEY`
3. A valid `FINNHUB_API_KEY` for news
4. Internet access during the first download and uncached runs

Yahoo Finance and SEC EDGAR do not require API keys. All capital, positions, orders, transaction costs, and P&L remain simulated.

### Core Contributions

1. **Position-Aware Trading Environment**: Simulates real trading with continuous position management across multi-step episodes using position-aware log returns (Eq. 2)

2. **Dual-Agent Decision Architecture**: Separates strategic direction decisions (buy/sell/hold) from tactical position sizing (quantity/risk) for better risk management

3. **Multi-Timescale Reward Design**: Provides training signals across 1-day, 7-day, and 30-day horizons to align short-term actions with long-term outcomes

4. **CVaR-Based Position Sizing**: Constrains position sizes based on tail risk (95% CVaR, 20-day rolling window) to prevent catastrophic losses

5. **Hierarchical Memory System**: Stores market information at different depths (shallow/intermediate/deep) based on importance for strategic reasoning

## ✨ Key Features

### Simplified Agent Pipeline (3 Agents)
- **Market Context Agent**: Summarizes currently available news, filings, technical indicators, and memory
- **Direction Agent**: Selects buy, sell, or hold
- **Quantity/Risk Agent**: Sizes the simulated order under cash and CVaR constraints
- Relevance handling, signal aggregation, memory allocation, and hard risk limits are deterministic code

### Risk Management
- 95% CVaR calculation with 20-day rolling window
- Dynamic position sizing based on tail risk
- Real-time exposure tracking with alerts
- Maximum drawdown monitoring

### Evaluation Metrics
- Cumulative Return (CR%)
- Sharpe Ratio (SR)
- Maximum Drawdown (MDD%)
- Calmar Ratio
- CVaR (95%)
- Win Rate, Profit Factor, and more

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher (Python 3.11 recommended)
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/finmemory.git
cd finmemory
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n finmemory python=3.10
conda activate finmemory
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Export the keys in the shell before running FinMemory:

```bash
# OpenAI API (required for LLM agents)
export OPENAI_API_KEY="sk-your-openai-key-here"

# Finnhub API (required for news data)
export FINNHUB_API_KEY="your-finnhub-key-here"
```

If you prefer a local `.env` file, load it into the shell explicitly before running:

```bash
set -a
source .env
set +a
```

Do not commit `.env` or API keys to version control.

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Finnhub: https://finnhub.io/register (free tier sufficient)

### Step 5: Verify Installation

```bash
python -c "import finmemory; print(f'FinMemory version: {finmemory.__version__}')"
python -m finmemory.main --help
python -m pytest -q
```

## 🏃 Quick Start

The command below downloads the required historical data automatically. No separate data-download command is required.

### Run Single Stock Evaluation

```bash
# Recommended first real-key smoke test (short range to control API usage)
python -m finmemory.main \
  --mode evaluation \
  --ticker AAPL \
  --start-date 2025-03-03 \
  --end-date 2025-03-11

# Longer evaluation after the smoke test succeeds
python -m finmemory.main \
  --mode evaluation \
  --ticker TSLA \
  --start-date 2025-03-01 \
  --end-date 2025-09-30 \
  --initial-capital 100000
```

The end date follows Yahoo Finance's historical-data convention and is effectively exclusive for price downloads.

### Run Main Experiment (Table 1)

```bash
# Run on all 5 stocks (TSLA, AAPL, AMZN, NFLX, COIN)
python -m finmemory.main --experiment main --output-dir results/main

# This will take several hours due to API calls
```

### Run Ablation Study (Table 2)

```bash
# Run one ablation configuration
python -m finmemory.main --experiment ablation --ablation-config w/o_MTR --output-dir results/ablation

# Other supported configurations
python -m finmemory.main --experiment ablation --ablation-config w/o_QRA
python -m finmemory.main --experiment ablation --ablation-config w/o_MSP
```

### Python API Usage

```python
from finmemory.main import run_single_stock
from finmemory.utils.logger import setup_default_logger

# Set up logging
logger = setup_default_logger("finmemory_demo")

# Run evaluation on a single stock
results = run_single_stock(
    ticker="TSLA",
    start_date="2025-03-01",
    end_date="2025-09-30",
    mode="evaluation",
    initial_capital=100000
)

# Print results
print(f"Cumulative Return: {results['metrics']['cumulative_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
print(f"Calmar Ratio: {results['metrics']['calmar_ratio']:.2f}")
print(f"Market Source Counts: {results['market_source_counts']}")
print(f"Memory Statistics: {results['memory_statistics']}")
print(f"Reflection Statistics: {results['reflection_statistics']}")
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FinMemory System                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Data       │    │Market Context│    │Decision Agents│     │
│  │   Loaders    │───▶│   (1 LLM)    │───▶│   (2 LLM)    │     │
│  │              │    │              │    │Direction/Qty │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                    │              │
│         ▼                   ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Price      │    │   Memory     │    │   Reward     │      │
│  │   Processor  │    │   System     │    │  Calculator  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                    │              │
│         ▼                   ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Signal     │    │  Reflection  │    │   Risk       │      │
│  │   Processor  │    │   Module     │    │  Metrics     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Data Layer (`finmemory/data/`)
- **Loaders**: Yahoo Finance (prices), Finnhub (news), SEC EDGAR (filings)
- **Processors**: Price features, signal aggregation, sentiment normalization

#### 2. Agent Layer (`finmemory/agents/`)
- **Market Context Agent** (1): Produces one concise daily interpretation of available signals
- **Decision Agents** (2): Direction (buy/sell/hold) + Quantity (position sizing)

#### 3. Memory Layer (`finmemory/memory/`)
- **Hierarchical Memory**: Shallow (10), Intermediate (30), Deep (100) capacity
- **Memory Allocator**: Routes items by importance score (1-10)
- **Reflection Module**: Updates memory based on trading outcomes

#### 4. Environment Layer (`finmemory/environment/`)
- **Trading Environment**: Position-aware state tracking
- **Position Manager**: Continuous position lifecycle management
- **Reward Calculator**: Multi-timescale reward signals (1/7/30 days)
- **Risk Metrics**: CR, SR, MDD, Calmar, CVaR calculations

#### 5. Risk Layer (`finmemory/risk/`)
- **CVaR Calculator**: 95% CVaR with 20-day rolling window
- **Position Sizer**: CVaR-constrained position sizing
- **Exposure Tracker**: Real-time risk monitoring with alerts

#### 6. Evaluation Layer (`finmemory/evaluation/`)
- **Metrics**: Comprehensive performance evaluation
- **Baselines**: 10 baseline implementations (LLM, DRL, rule-based)
- **Ablation Studies**: Component contribution analysis

## ⚙️ Configuration

### Configuration Files

FinMemory uses YAML configuration files located in `finmemory/config/`:

- `agents.yaml`: Agent settings, model parameters, hyperparameters
- `environment.yaml`: Environment parameters, risk settings, experiment config
- `prompts/`: Optional decision-agent prompt templates; active agents also provide built-in fallback prompts

### Key Configuration Parameters

#### Agent Configuration (`config/agents.yaml`)

```yaml
global:
  model_name: "gpt-4o"
  training_temperature: 0.7    # Exploration during training
  evaluation_temperature: 0.3  # Consistency during evaluation

active_agents:
  market_context_agent:
    purpose: "Summarize current signals, technical indicators, and memory"
  direction_agent:
    purpose: "Select buy, sell, or hold"
  quantity_risk_agent:
    purpose: "Size orders under hard risk constraints"
    enforce_cvar_constraint: true
    enforce_cash_constraint: true

deterministic_preprocessing:
  company_news_importance: 5
  ticker_mention_importance: 7
  macro_news_importance: 3
  ten_k_importance: 8
  ten_q_importance: 6

memory_system:
  shallow_capacity: 10
  intermediate_capacity: 30
  deep_capacity: 100
  reflection_capacity: 50

risk_management:
  cvar_calculator:
    confidence_level: 0.95     # 95% confidence
    window_size: 20            # 20-day rolling window
```

#### Environment Configuration (`config/environment.yaml`)

```yaml
environment:
  initial_capital: 100000      # $100,000 starting capital
  transaction_cost: 0.001      # 0.1% per trade
  allow_short_selling: false   # No short selling

multi_timescale_reward:
  short_window: 1              # 1-day trend
  mid_window: 7                # 7-day trend
  long_window: 30              # 30-day trend
  combine_method: "sum"

experiments:
  stocks: ["TSLA", "AAPL", "AMZN", "NFLX", "COIN"]
  training_period:
    start: "2024-01-01"
    end: "2025-02-28"
  evaluation_period:
    start: "2025-03-01"
    end: "2025-09-30"
```

### Modifying Configuration

You can modify configuration files directly or pass overrides via command line:

```bash
# Use custom configuration file
python -m finmemory.main --config custom_config.yaml

# Override specific parameters
python -m finmemory.main --ticker TSLA --initial-capital 50000
```

## 📖 Usage

### Trading Loop

Before the trading loop, FinMemory downloads and processes market information:

1. Download OHLCV prices from Yahoo Finance
2. Download company and macro news from Finnhub
3. Download 10-K and 10-Q filings from SEC EDGAR
4. Normalize sentiment and assign deterministic importance scores
5. Build dated signal snapshots and pending memory items

At each trading timestep, FinMemory then:

1. **Reveal Available Information**: Add only news and filings published by the current date to memory
2. **Get State**: Retrieve current price, position, cash, account value, technical features, and memory
3. **Aggregate Signals**: Build sentiment, intensity, and source-count indicators without future-data leakage
4. **Market Context**: One agent summarizes the current signals, technical indicators, and recent memory
5. **Direction Decision**: Direction Agent decides buy/sell/hold
6. **CVaR Constraint**: Calculate the maximum position exposure from recent tail risk
7. **Quantity Decision**: Quantity/Risk Agent sizes the order using cash, holdings, volatility, memory, and reflections
8. **Execute Trade**: Update position, cash, holdings, and account value
9. **Calculate Reward**: Calculate the multi-timescale reward in training mode
10. **Reflect**: Record the outcome and feed generated lessons back into hierarchical memory

### Result Structure

`run_single_stock()` returns:

| Field | Description |
|---|---|
| `metrics` | Return, Sharpe, drawdown, Calmar, CVaR, and trading statistics |
| `episode_log` | Per-timestep decisions, quantities, prices, account values, rewards, and signal snapshots |
| `market_source_counts` | Number of company-news, macro-news, and SEC items prepared by deterministic code |
| `memory_statistics` | Hierarchical memory usage and allocation statistics |
| `reflection_statistics` | Recorded outcomes and generated reflection statistics |
| `agent_costs` | Market Context, Direction, and Quantity/Risk Agent token/cost summaries |

### API Usage Notes

- `OPENAI_API_KEY` is required for the three active LLM agents.
- `FINNHUB_API_KEY` is required for company and macro news. Without it, news loading returns empty frames and the run continues with prices and SEC data.
- SEC EDGAR and Yahoo Finance do not require API keys.
- The active pipeline makes up to three LLM calls per trading day; long periods can still generate substantial API usage.
- Downloaded data is cached locally and reused when possible.

### Generated Data and Cache

Downloaded files are created under the configured data directory during execution. They can be deleted safely when you want to reclaim disk space; the next run downloads them again. Deleting the cache does not delete source code or experiment configuration.

### Memory Allocation

Information is allocated to memory layers based on importance scores:

| Importance Score | Memory Layer | Capacity | Example Content |
|-----------------|--------------|----------|-----------------|
| 1-4 | Shallow | 10 items | Minor news, routine updates |
| 5-7 | Intermediate | 30 items | Earnings reports, moderate events |
| 8-10 | Deep | 100 items | Major mergers, bankruptcies, strategic shifts |
