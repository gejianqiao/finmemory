"""
Prompt Manager Module for FinMemory Trading System.

This module provides utilities for loading, formatting, and managing
prompt templates from YAML configuration files. It supports dynamic
variable substitution and prompt validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Manages prompt templates for all agent types in the FinMemory system.
    
    Loads YAML prompt configurations, formats them with dynamic variables,
    and validates prompt structure. Supports hierarchical prompt organization
    by agent type (filtering, analysis, decision, memory).
    
    Attributes:
        prompts_dir (Path): Base directory for prompt YAML files
        prompts_cache (Dict[str, Dict]): Cache of loaded prompt templates
        default_model (str): Default LLM model for prompts
    """
    
    def __init__(self, prompts_dir: Optional[str] = None):
        """
        Initialize PromptManager with prompts directory.
        
        Args:
            prompts_dir: Path to prompts directory. Defaults to 
                        finmemory/config/prompts if not specified
        """
        if prompts_dir is None:
            # Default to config/prompts relative to finmemory package
            base_dir = Path(__file__).parent.parent
            self.prompts_dir = base_dir / "config" / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self.prompts_cache: Dict[str, Dict[str, Any]] = {}
        self.default_model = "gpt-4o"
        
        logger.info(f"PromptManager initialized with prompts_dir: {self.prompts_dir}")
    
    def load_prompt(self, agent_type: str, prompt_name: str) -> Dict[str, Any]:
        """
        Load a prompt template from YAML file.
        
        Args:
            agent_type: Type of agent (filtering, analysis, decision, memory)
            prompt_name: Name of the prompt file (without .yaml extension)
        
        Returns:
            Dict containing system_prompt and user_template keys
        
        Raises:
            FileNotFoundError: If prompt file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        cache_key = f"{agent_type}/{prompt_name}"
        
        # Check cache first
        if cache_key in self.prompts_cache:
            logger.debug(f"Loading prompt from cache: {cache_key}")
            return self.prompts_cache[cache_key]
        
        # Construct file path
        prompt_file = self.prompts_dir / agent_type / f"{prompt_name}.yaml"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        logger.info(f"Loading prompt from: {prompt_file}")
        
        # Load YAML file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
        
        # Validate structure
        if not isinstance(prompt_config, dict):
            raise ValueError(f"Invalid prompt structure in {prompt_file}: expected dict")
        
        if 'system_prompt' not in prompt_config:
            raise ValueError(f"Missing 'system_prompt' in {prompt_file}")
        
        if 'user_template' not in prompt_config:
            raise ValueError(f"Missing 'user_template' in {prompt_file}")
        
        # Cache the loaded prompt
        self.prompts_cache[cache_key] = prompt_config
        
        return prompt_config
    
    def format_prompt(
        self,
        agent_type: str,
        prompt_name: str,
        **variables
    ) -> Dict[str, str]:
        """
        Load and format a prompt template with dynamic variables.
        
        Args:
            agent_type: Type of agent (filtering, analysis, decision, memory)
            prompt_name: Name of the prompt file (without .yaml extension)
            **variables: Keyword arguments for template substitution
        
        Returns:
            Dict with formatted 'system_prompt' and 'user_prompt' strings
        
        Example:
            >>> pm = PromptManager()
            >>> formatted = pm.format_prompt(
            ...     'filtering', 'company_news_filter',
            ...     ticker='TSLA',
            ...     headline='Tesla announces new factory',
            ...     summary='Tesla plans to build...',
            ...     timestamp='2024-03-15'
            ... )
            >>> print(formatted['user_prompt'])
        """
        # Load prompt template
        prompt_config = self.load_prompt(agent_type, prompt_name)
        
        system_prompt = prompt_config['system_prompt']
        user_template = prompt_config['user_template']
        
        # Format user template with variables
        try:
            user_prompt = user_template.format(**variables)
        except KeyError as e:
            missing_var = str(e).strip("'")
            available_vars = list(variables.keys())
            raise KeyError(
                f"Missing variable '{missing_var}' for prompt {agent_type}/{prompt_name}. "
                f"Available variables: {available_vars}"
            )
        
        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt
        }
    
    def get_system_prompt(self, agent_type: str, prompt_name: str) -> str:
        """
        Get only the system prompt for an agent.
        
        Args:
            agent_type: Type of agent (filtering, analysis, decision, memory)
            prompt_name: Name of the prompt file (without .yaml extension)
        
        Returns:
            System prompt string
        """
        prompt_config = self.load_prompt(agent_type, prompt_name)
        return prompt_config['system_prompt']
    
    def format_user_prompt(
        self,
        agent_type: str,
        prompt_name: str,
        **variables
    ) -> str:
        """
        Format only the user prompt template with variables.
        
        Args:
            agent_type: Type of agent (filtering, analysis, decision, memory)
            prompt_name: Name of the prompt file (without .yaml extension)
            **variables: Keyword arguments for template substitution
        
        Returns:
            Formatted user prompt string
        """
        prompt_config = self.load_prompt(agent_type, prompt_name)
        user_template = prompt_config['user_template']
        
        try:
            return user_template.format(**variables)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise KeyError(
                f"Missing variable '{missing_var}' for prompt {agent_type}/{prompt_name}"
            )
    
    def list_available_prompts(self, agent_type: Optional[str] = None) -> Dict[str, list]:
        """
        List all available prompt templates.
        
        Args:
            agent_type: Optional filter by agent type
        
        Returns:
            Dict mapping agent types to lists of available prompt names
        """
        available = {}
        
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory does not exist: {self.prompts_dir}")
            return available
        
        # Get all subdirectories (agent types)
        agent_types = [
            d.name for d in self.prompts_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ]
        
        if agent_type:
            agent_types = [agent_type] if agent_type in agent_types else []
        
        for atype in agent_types:
            type_dir = self.prompts_dir / atype
            prompts = [
                f.stem for f in type_dir.glob('*.yaml')
                if not f.name.startswith('_')
            ]
            available[atype] = sorted(prompts)
        
        return available
    
    def validate_prompt(self, agent_type: str, prompt_name: str) -> tuple[bool, list]:
        """
        Validate a prompt template structure.
        
        Args:
            agent_type: Type of agent (filtering, analysis, decision, memory)
            prompt_name: Name of the prompt file (without .yaml extension)
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        try:
            prompt_config = self.load_prompt(agent_type, prompt_name)
            
            # Check required keys
            if 'system_prompt' not in prompt_config:
                errors.append("Missing 'system_prompt' key")
            elif not isinstance(prompt_config['system_prompt'], str):
                errors.append("'system_prompt' must be a string")
            elif len(prompt_config['system_prompt'].strip()) == 0:
                errors.append("'system_prompt' is empty")
            
            if 'user_template' not in prompt_config:
                errors.append("Missing 'user_template' key")
            elif not isinstance(prompt_config['user_template'], str):
                errors.append("'user_template' must be a string")
            elif len(prompt_config['user_template'].strip()) == 0:
                errors.append("'user_template' is empty")
            
            # Check for common template variables (optional validation)
            user_template = prompt_config.get('user_template', '')
            # Look for {variable} patterns
            import re
            variables = re.findall(r'\{(\w+)\}', user_template)
            if not variables:
                logger.warning(f"Prompt {agent_type}/{prompt_name} has no template variables")
        
        except Exception as e:
            errors.append(str(e))
        
        return (len(errors) == 0, errors)
    
    def clear_cache(self) -> None:
        """Clear the prompts cache to force reload from files."""
        self.prompts_cache.clear()
        logger.debug("Prompt cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics (num_prompts, memory_estimate)
        """
        total_size = sum(
            len(str(v)) for v in self.prompts_cache.values()
        )
        return {
            'num_prompts': len(self.prompts_cache),
            'estimated_memory_bytes': total_size
        }


def load_decision_prompt(prompt_name: str, **variables) -> Dict[str, str]:
    """
    Load and format a decision agent prompt.
    
    Args:
        prompt_name: Name of the decision prompt (e.g., 'direction_agent')
        **variables: Variables for template formatting
    
    Returns:
        Dict with 'system_prompt' and 'user_prompt'
    """
    pm = PromptManager()
    return pm.format_prompt('decision', prompt_name, **variables)
