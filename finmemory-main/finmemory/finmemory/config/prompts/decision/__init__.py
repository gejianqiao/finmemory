"""
Decision Agent Prompt Templates Package

This package contains YAML prompt templates for the dual-agent decision architecture:
- direction_agent.yaml: Strategic buy/sell/hold decision prompts
- quantity_agent.yaml: Position sizing with CVaR constraint prompts

These templates are loaded by prompt_manager.py and used by:
- finmemory.agents.decision.direction_agent (DirectionAgent)
- finmemory.agents.decision.quantity_risk_agent (QuantityRiskAgent)

Prompt Structure:
Each YAML file contains:
- system_prompt: Defines agent role, decision framework, and output format
- user_template: Dynamic template with variables for market state, memory, and constraints

Output Format:
Both agents produce JSON-formatted decisions with:
- Primary decision (investment_decision or order_size)
- summary_reason: Concise explanation of reasoning
- memory_indices: List of memory item indices referenced
- reflection_analysis: Analysis based on past experiences

Version: 1.0.0
"""

__version__ = "1.0.0"
__all__ = []

# Note: Actual prompt templates are YAML files loaded dynamically:
# - direction_agent.yaml: Direction decision prompts (buy/sell/hold)
# - quantity_agent.yaml: Quantity/risk decision prompts (position sizing)
#
# These are loaded via prompt_manager.load_decision_prompt()
