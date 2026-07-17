"""
Base LLM Agent Class for FinMemory Trading System.

This module provides the foundational agent class with OpenAI API integration,
JSON parsing, error handling, and retry logic. All active analysis and decision
agents inherit from this base class.

Author: FinMemory Team
Date: 2024
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI

from finmemory.utils.json_parser import parse_json_output
from finmemory.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all LLM-based agents in the FinMemory system.
    
    Provides common functionality for:
    - OpenAI API integration with GPT-4o
    - JSON output parsing with validation
    - Retry logic with exponential backoff
    - Token counting and cost tracking
    - Error handling for API failures
    
    Attributes:
        model_name (str): The LLM model to use (default: "gpt-4o")
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens in response
        api_key (str): OpenAI API key
        client (OpenAI): OpenAI client instance
        retry_attempts (int): Number of retry attempts on failure
        retry_delay (float): Initial delay between retries (seconds)
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the base agent with API configuration.
        
        Args:
            model_name: Name of the OpenAI model to use
            temperature: Sampling temperature for generation (lower = more deterministic)
            max_tokens: Maximum number of tokens in the response
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            retry_attempts: Number of times to retry on API failure
            retry_delay: Initial delay between retries in seconds
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Token tracking for cost estimation
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        
        # Cost per 1K tokens (approximate, as of 2024)
        self.cost_per_1k_prompt = 0.005  # $5 per 1M tokens
        self.cost_per_1k_completion = 0.015  # $15 per 1M tokens
        
        logger.info(f"BaseAgent initialized with model={model_name}, temperature={temperature}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        
        Must be implemented by subclasses to define the agent's role and behavior.
        
        Returns:
            str: System prompt text
        """
        pass
    
    @abstractmethod
    def format_user_prompt(self, **kwargs) -> str:
        """
        Format the user prompt with dynamic variables.
        
        Must be implemented by subclasses to construct the user message.
        
        Args:
            **kwargs: Dynamic variables to insert into the prompt
            
        Returns:
            str: Formatted user prompt
        """
        pass
    
    @abstractmethod
    def validate_output(self, output: Dict[str, Any]) -> bool:
        """
        Validate the parsed JSON output from the LLM.
        
        Must be implemented by subclasses to ensure output conforms to expected schema.
        
        Args:
            output: Parsed JSON output from LLM
            
        Returns:
            bool: True if output is valid, False otherwise
        """
        pass
    
    def call_llm(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the LLM API with retry logic and error handling.
        
        Args:
            system_prompt: System prompt (uses get_system_prompt() if None)
            user_prompt: User prompt (uses format_user_prompt() if None)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional arguments for format_user_prompt()
            
        Returns:
            Dict[str, Any]: Parsed JSON output from LLM
            
        Raises:
            ValueError: If output cannot be parsed or validated
            openai.APIError: If API call fails after all retries
        """
        # Use defaults if not provided
        if system_prompt is None:
            system_prompt = self.get_system_prompt()
        if user_prompt is None:
            user_prompt = self.format_user_prompt(**kwargs)
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Prepare messages for API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"LLM API call attempt {attempt + 1}/{self.retry_attempts}")
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},  # Force JSON output
                )
                
                # Extract response
                content = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                
                # Update token tracking
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self._update_cost(prompt_tokens, completion_tokens)
                
                logger.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}")
                
                # Parse JSON output
                parsed_output = parse_json_output(content)
                
                # Validate output
                if not self.validate_output(parsed_output):
                    logger.warning(f"Output validation failed: {parsed_output}")
                    raise ValueError("Output validation failed")
                
                logger.info(f"LLM call successful, tokens: {prompt_tokens + completion_tokens}")
                return parsed_output
                
            except openai.APIError as e:
                last_error = e
                logger.warning(f"API error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"API call failed after {self.retry_attempts} attempts")
                    raise
                    
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON parsing error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"JSON parsing failed after {self.retry_attempts} attempts")
                    raise ValueError(f"Failed to parse JSON output: {content[:200]}...")
                    
            except ValueError as e:
                last_error = e
                logger.warning(f"Validation error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Output validation failed after {self.retry_attempts} attempts")
                    raise
        
        # Should not reach here, but just in case
        raise last_error
    
    def _update_cost(self, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Update cost tracking based on token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
        """
        prompt_cost = (prompt_tokens / 1000) * self.cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000) * self.cost_per_1k_completion
        self.total_cost += prompt_cost + completion_cost
    
    def get_cost_summary(self) -> Dict[str, float]:
        """
        Get summary of API usage and costs.
        
        Returns:
            Dict with total_prompt_tokens, total_completion_tokens, total_cost
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost,
        }
    
    def reset_cost_tracking(self) -> None:
        """Reset token and cost tracking counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        logger.info("Cost tracking reset")
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update the temperature setting.
        
        Args:
            temperature: New temperature value (0.0-1.0)
        """
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature
        logger.info(f"Temperature updated to {temperature}")
    
    def set_mode(self, mode: str) -> None:
        """
        Set agent mode (training or evaluation).
        
        Args:
            mode: Either "training" (temperature=0.7) or "evaluation" (temperature=0.3)
        """
        if mode == "training":
            self.set_temperature(0.7)
            logger.info("Agent set to training mode (temperature=0.7)")
        elif mode == "evaluation":
            self.set_temperature(0.3)
            logger.info("Agent set to evaluation mode (temperature=0.3)")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'training' or 'evaluation'.")
