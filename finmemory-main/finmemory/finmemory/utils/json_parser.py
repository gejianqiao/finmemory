"""
JSON Parser Utility for FinMemory Trading System.

This module provides robust JSON parsing functionality for LLM outputs,
handling common issues like malformed JSON, extra whitespace, and code blocks.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple


class JSONParseError(Exception):
    """Custom exception for JSON parsing errors."""
    
    def __init__(self, message: str, raw_output: str = "", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.raw_output = raw_output
        self.details = details or {}
        super().__init__(self.message)


def parse_json_output(
    raw_output: str,
    expected_keys: Optional[list] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Parse JSON output from LLM response with robust error handling.
    
    This function handles common LLM output issues:
    - Markdown code blocks (```json ... ```)
    - Extra whitespace and newlines
    - Trailing commas
    - Single quotes instead of double quotes
    - Missing closing braces
    
    Args:
        raw_output: Raw string output from LLM
        expected_keys: Optional list of expected keys to validate against
        strict: If True, raise error on any deviation from valid JSON
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        JSONParseError: If parsing fails after all recovery attempts
    """
    if not raw_output or not isinstance(raw_output, str):
        raise JSONParseError(
            message="Empty or invalid raw output",
            raw_output=str(raw_output)
        )
    
    # Clean the output
    cleaned = _clean_json_string(raw_output)
    
    # Try parsing with standard json.loads
    try:
        parsed = json.loads(cleaned)
        
        # Validate it's a dictionary
        if not isinstance(parsed, dict):
            raise JSONParseError(
                message=f"Expected dictionary, got {type(parsed).__name__}",
                raw_output=raw_output,
                details={"parsed_type": type(parsed).__name__}
            )
        
        # Validate expected keys if provided
        if expected_keys and strict:
            missing_keys = set(expected_keys) - set(parsed.keys())
            if missing_keys:
                raise JSONParseError(
                    message=f"Missing expected keys: {missing_keys}",
                    raw_output=raw_output,
                    details={"missing_keys": list(missing_keys), "found_keys": list(parsed.keys())}
                )
        
        return parsed
        
    except json.JSONDecodeError as e:
        # Try recovery strategies
        return _recover_json_parse(raw_output, expected_keys, strict)


def _clean_json_string(raw: str) -> str:
    """
    Clean JSON string by removing common formatting issues.
    
    Args:
        raw: Raw string potentially containing JSON
        
    Returns:
        Cleaned string ready for JSON parsing
    """
    cleaned = raw.strip()
    
    # Remove markdown code blocks
    cleaned = _remove_code_blocks(cleaned)
    
    # Remove leading/trailing non-JSON content
    cleaned = _extract_json_content(cleaned)
    
    # Fix common JSON issues
    cleaned = _fix_json_issues(cleaned)
    
    return cleaned


def _remove_code_blocks(text: str) -> str:
    """Remove markdown code block formatting."""
    # Pattern for ```json ... ``` or ``` ... ```
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    return text


def _extract_json_content(text: str) -> str:
    """Extract JSON content from text that may have surrounding content."""
    # Find first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    
    # Try with arrays
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1]
    
    return text


def _fix_json_issues(text: str) -> str:
    """Fix common JSON formatting issues."""
    # Replace single quotes with double quotes (careful with apostrophes in strings)
    # This is a simple heuristic and may not work for all cases
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')
    
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    # Fix unquoted keys (simple cases)
    # Pattern: { key: "value" } -> { "key": "value" }
    text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', text)
    
    # Remove control characters that might break JSON
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def _recover_json_parse(
    raw: str,
    expected_keys: Optional[list] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Attempt to recover JSON parsing through various strategies.
    
    Args:
        raw: Raw LLM output
        expected_keys: Expected keys for validation
        strict: Strict validation mode
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        JSONParseError: If all recovery attempts fail
    """
    recovery_strategies = [
        _try_fix_quotes,
        _try_fix_missing_braces,
        _try_fix_trailing_commas,
        _try_partial_parse,
    ]
    
    cleaned = _clean_json_string(raw)
    
    for strategy in recovery_strategies:
        try:
            fixed = strategy(cleaned)
            if fixed != cleaned:
                parsed = json.loads(fixed)
                if isinstance(parsed, dict):
                    return parsed
        except (json.JSONDecodeError, Exception):
            continue
    
    # If all strategies fail, raise error with helpful information
    raise JSONParseError(
        message="Failed to parse JSON after all recovery attempts",
        raw_output=raw,
        details={
            "cleaned_output": cleaned[:500],  # Truncate for readability
            "expected_keys": expected_keys,
            "strict_mode": strict
        }
    )


def _try_fix_quotes(text: str) -> str:
    """Try to fix quote issues in JSON."""
    # Replace smart quotes with regular quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Try converting single quotes to double quotes for keys and string values
    # This is risky but sometimes necessary
    if "'" in text:
        # Simple approach: replace all single quotes with double quotes
        # This won't work for strings containing apostrophes, but it's a last resort
        text = text.replace("'", '"')
    
    return text


def _try_fix_missing_braces(text: str) -> str:
    """Try to fix missing opening or closing braces."""
    open_braces = text.count('{')
    close_braces = text.count('}')
    
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    elif close_braces > open_braces:
        text = '{' * (close_braces - open_braces) + text
    
    return text


def _try_fix_trailing_commas(text: str) -> str:
    """Try to fix trailing commas."""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text


def _try_partial_parse(text: str) -> str:
    """Try to parse partial JSON by finding valid JSON substring."""
    # Try progressively shorter substrings from the start
    for i in range(len(text), 0, -1):
        try:
            substring = text[:i]
            # Ensure we end at a logical point
            if substring.rstrip().endswith('}') or substring.rstrip().endswith(']'):
                json.loads(substring)
                return substring
        except json.JSONDecodeError:
            continue
    
    return text


def validate_json_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    raise_on_error: bool = True
) -> Tuple[bool, list]:
    """
    Validate JSON data against a simple schema.
    
    Schema format:
    {
        "required_keys": ["key1", "key2"],
        "optional_keys": ["key3"],
        "key_types": {"key1": str, "key2": int}
    }
    
    Args:
        data: Dictionary to validate
        schema: Schema definition
        raise_on_error: If True, raise JSONParseError on validation failure
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required keys
    required_keys = schema.get("required_keys", [])
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Check key types
    key_types = schema.get("key_types", {})
    for key, expected_type in key_types.items():
        if key in data:
            if not isinstance(data[key], expected_type):
                errors.append(
                    f"Key '{key}' has wrong type: expected {expected_type.__name__}, "
                    f"got {type(data[key]).__name__}"
                )
    
    # Check value ranges for numeric fields
    key_ranges = schema.get("key_ranges", {})
    for key, (min_val, max_val) in key_ranges.items():
        if key in data:
            value = data[key]
            if isinstance(value, (int, float)):
                if value < min_val or value > max_val:
                    errors.append(
                        f"Key '{key}' value {value} out of range [{min_val}, {max_val}]"
                    )
    
    is_valid = len(errors) == 0
    
    if not is_valid and raise_on_error:
        raise JSONParseError(
            message="JSON schema validation failed",
            details={"errors": errors, "data": data, "schema": schema}
        )
    
    return is_valid, errors


def safe_json_loads(
    text: str,
    default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Safely parse JSON with a default fallback value.
    
    Args:
        text: String to parse
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        result = parse_json_output(text)
        return result
    except JSONParseError:
        return default if default is not None else {}


# Convenience functions for common validation patterns

def validate_decision_output(output: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate decision agent output format.
    
    Expected format:
    {
        "investment_decision": "buy" | "sell" | "hold",
        "summary_reason": str,
        "memory_indices": list[int],
        "reflection_analysis": str (optional)
    }
    """
    schema = {
        "required_keys": ["investment_decision", "summary_reason"],
        "key_types": {
            "investment_decision": str,
            "summary_reason": str,
            "memory_indices": list
        },
        "key_values": {
            "investment_decision": ["buy", "sell", "hold"]
        }
    }
    
    errors = []
    
    # Check required keys
    for key in schema["required_keys"]:
        if key not in output:
            errors.append(f"Missing required key: {key}")
    
    # Check types
    for key, expected_type in schema["key_types"].items():
        if key in output and not isinstance(output[key], expected_type):
            errors.append(f"Key '{key}' has wrong type")
    
    # Check enum values
    for key, allowed_values in schema.get("key_values", {}).items():
        if key in output and output[key] not in allowed_values:
            errors.append(f"Key '{key}' must be one of {allowed_values}")
    
    return len(errors) == 0, errors


def validate_quantity_output(output: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate quantity agent output format.
    
    Expected format:
    {
        "order_size": int (>= 0),
        "summary_reason": str,
        "memory_indices": list[int] (optional),
        "reflection_analysis": str (optional)
    }
    """
    errors = []
    
    if "order_size" not in output:
        errors.append("Missing required key: order_size")
    elif not isinstance(output["order_size"], int) or output["order_size"] < 0:
        errors.append("order_size must be a non-negative integer")
    
    if "summary_reason" not in output:
        errors.append("Missing required key: summary_reason")
    elif not isinstance(output["summary_reason"], str):
        errors.append("summary_reason must be a string")
    
    return len(errors) == 0, errors


def validate_filter_output(output: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate filtering agent output format.
    
    Expected format:
    {
        "is_relevant": bool,
        "relevance_score": int (1-10),
        "reason": str
    }
    """
    errors = []
    
    required_keys = ["is_relevant", "relevance_score", "reason"]
    for key in required_keys:
        if key not in output:
            errors.append(f"Missing required key: {key}")
    
    if "is_relevant" in output and not isinstance(output["is_relevant"], bool):
        errors.append("is_relevant must be a boolean")
    
    if "relevance_score" in output:
        score = output["relevance_score"]
        if not isinstance(score, (int, float)) or score < 1 or score > 10:
            errors.append("relevance_score must be an integer between 1 and 10")
    
    if "reason" in output and not isinstance(output["reason"], str):
        errors.append("reason must be a string")
    
    return len(errors) == 0, errors
