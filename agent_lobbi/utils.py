"""
Agent Lobbi Utilities
====================
Shared utilities, logging, and helper functions.
"""

import logging
import sys
from typing import Optional

# Create package logger
logger = logging.getLogger("agent_lobby")

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """
    Set up logging for Agent Lobbi
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in log format
    
    Returns:
        Configured logger instance
    """
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def validate_agent_id(agent_id: str) -> bool:
    """
    Validate agent ID format
    
    Args:
        agent_id: Agent identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not agent_id or not isinstance(agent_id, str):
        return False
    
    # Agent ID should be alphanumeric with underscores, 3-64 chars
    if not agent_id.replace("_", "").replace("-", "").isalnum():
        return False
    
    if len(agent_id) < 3 or len(agent_id) > 64:
        return False
    
    return True

def validate_capabilities(capabilities: list) -> bool:
    """
    Validate capabilities list
    
    Args:
        capabilities: List of capability strings
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(capabilities, list):
        return False
    
    if len(capabilities) == 0:
        return False
    
    for cap in capabilities:
        if not isinstance(cap, str) or len(cap) < 1:
            return False
    
    return True

__all__ = [
    "logger",
    "setup_logging",
    "validate_agent_id",
    "validate_capabilities"
] 