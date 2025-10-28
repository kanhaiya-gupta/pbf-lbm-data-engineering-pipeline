"""
Robust JSON parsing utilities for handling large, malformed JSON strings.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def find_safe_truncation_point(json_string: str, max_length: int) -> Tuple[str, bool]:
    """
    Find a safe truncation point in a JSON string that maintains valid JSON structure.

    Args:
        json_string: The JSON string to truncate
        max_length: Maximum length for truncation (e.g., 5000 for 5KB)

    Returns:
        Tuple of (truncated string, success flag)
    """
    if len(json_string) <= max_length:
        return json_string, True

    # Validate starting character
    if not json_string or json_string[0] not in '{[':
        logger.warning("JSON string does not start with valid object or array")
        return '{"raw_data": "' + json_string[:1000].replace('"', '\\"') + '"}', False

    # Initialize state for parsing
    truncated = json_string[:max_length]
    in_string = False
    escape = False
    brace_depth = 0  # Tracks { }
    bracket_depth = 0  # Tracks [ ]
    last_safe_point = 0  # Last position where truncation would be safe
    last_comma = -1  # Last comma outside a string

    for i, char in enumerate(truncated):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue

        # Track nesting levels
        if char == '{':
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and bracket_depth == 0:
                last_safe_point = i + 1
        elif char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
            if brace_depth == 0 and bracket_depth == 0:
                last_safe_point = i + 1
        elif char == ',' and brace_depth <= 1 and bracket_depth <= 1:
            last_comma = i

        # Early exit if we go negative (malformed JSON)
        if brace_depth < 0 or bracket_depth < 0:
            logger.warning(f"Malformed JSON detected at position {i}")
            break

    # If we're at a complete object/array boundary, use it
    if last_safe_point > 0:
        return json_string[:last_safe_point], True

    # If we're at top-level and have a comma, try to truncate at the last comma
    if last_comma > 0 and brace_depth <= 1 and bracket_depth <= 1:
        suffix = '}' if json_string[0] == '{' else ']' if json_string[0] == '[' else ''
        return json_string[:last_comma] + suffix, True

    # Fallback: no safe truncation point found
    logger.warning(f"No safe truncation point found within {max_length} chars")
    return '{"raw_data": "' + json_string[:1000].replace('"', '\\"') + '"}', False

def safe_json_loads(json_string: str, field_name: str = "unknown", max_size: int = 5000) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON strings with aggressive truncation for large/malformed data.

    Args:
        json_string: The JSON string to parse
        field_name: Name of the field being parsed (for logging)
        max_size: Maximum size for truncation (default 5KB)

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not json_string or not isinstance(json_string, str) or json_string.isspace():
        logger.warning(f"Invalid or empty JSON string in {field_name}")
        return None

    try:
        # Bypass parsing for extremely large JSON strings
        if len(json_string) > 50000:  # 50KB limit
            logger.warning(f"Massive JSON string in {field_name} ({len(json_string)} chars), bypassing parsing")
            return {"raw_data": json_string[:1000], "size": len(json_string), "bypassed": True}

        # Truncate large JSON strings
        if len(json_string) > max_size:
            logger.warning(f"Large JSON string in {field_name} ({len(json_string)} chars), truncating to {max_size} chars")
            truncated, success = find_safe_truncation_point(json_string, max_size)
            if not success:
                # If truncation failed, return a safe fallback
                return {"raw_data": json_string[:1000], "truncated": True, "size": len(json_string)}
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                return {"raw_data": json_string[:1000], "truncated": True, "size": len(json_string)}

        # Parse smaller JSON strings directly
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            # For smaller strings that still fail, return a safe fallback
            return {"raw_data": json_string[:500], "parse_error": True}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse {field_name} JSON at {e.pos}: {e}")
        return {"raw_data": json_string[:500] if len(json_string) > 500 else json_string}
    except Exception as e:
        logger.warning(f"Unexpected error parsing {field_name}: {e}")
        return None

def safe_json_loads_with_fallback(json_string: Union[str, Dict[str, Any]], field_name: str = "unknown", max_size: int = 5000, fallback: Any = None) -> Any:
    """
    Safely parse JSON strings with a fallback value, handling dictionaries and non-JSON strings.

    Args:
        json_string: The JSON string or dictionary to parse
        field_name: Name of the field being parsed (for logging)
        max_size: Maximum size for truncation (default 5KB)
        fallback: Value to return if parsing fails

    Returns:
        Parsed data or fallback value
    """
    if isinstance(json_string, dict):
        return json_string
    if not isinstance(json_string, str) or json_string.isspace():
        logger.warning(f"Non-string or empty input in {field_name}")
        return fallback

    # Handle range strings like "0-1000" for specific fields
    if '-' in json_string and len(json_string) < 100 and field_name == "range_data":
        try:
            parts = json_string.split('-')
            if len(parts) == 2:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                if min_val <= max_val:
                    return {"min": min_val, "max": max_val}
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse range string in {field_name}: {json_string}")

    result = safe_json_loads(json_string, field_name, max_size)
    return result if result is not None else fallback