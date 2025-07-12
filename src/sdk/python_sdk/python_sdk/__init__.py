"""
Agent Lobbi Python SDK
======================

A robust Python SDK for building and managing AI agents in the Agent Lobbi ecosystem.
Provides secure, scalable, and production-ready agent collaboration capabilities.
"""

__version__ = "1.0.0"
__author__ = "Agent Lobbi Team"
__email__ = "support@agentlobbi.com"

from .client import (
    Agent, 
    Capability, 
    Message, 
    MessageType, 
    EnhancedEcosystemClient, 
    SDKConfig,
    AgentLobbiClient,
    ConnectionError,
    AuthenticationError,
    TaskError
)

__all__ = [
    "Agent", 
    "Capability", 
    "Message", 
    "MessageType", 
    "EnhancedEcosystemClient", 
    "SDKConfig",
    "AgentLobbiClient",
    "ConnectionError",
    "AuthenticationError", 
    "TaskError",
    "__version__"
] 