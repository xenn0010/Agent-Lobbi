"""
Agent Lobbi - Robust Multi-Agent Collaboration Platform
======================================================

A production-ready platform for AI agent collaboration with built-in security,
consensus mechanisms, recovery systems, and real-time monitoring.

Key Features:
- Multi-layer security with consensus and reputation systems
- Automatic connection recovery and state management
- Real-time WebSocket communication with HTTP fallbacks  
- Comprehensive metrics tracking and monitoring
- MCP (Model Context Protocol) compliance
- Game theory-based security incentives

Basic Usage:
    from agent_lobby import AgentLobbiSDK
    
    sdk = AgentLobbiSDK(
        lobby_host="localhost",
        lobby_port=8098,
        enable_security=True
    )
    
    await sdk.register_agent(
        agent_id="my_agent",
        agent_type="Assistant",
        capabilities=["analysis", "writing"]
    )

Enterprise Usage:
    from agent_lobby import AgentLobbiSDK, create_secure_agent
    
    # Full security and monitoring
    sdk = await create_secure_agent(
        agent_id="enterprise_agent",
        agent_type="DataAnalyst", 
        capabilities=["data_analysis", "reporting"],
        lobby_host="production.agentlobby.com"
    )

Advanced Features:
- Task delegation between agents
- Consensus-based decision making
- Automatic error recovery
- Real-time collaboration monitoring
- Cross-agent data sharing with security
"""

__version__ = "1.0.1"
__author__ = "Agent Lobbi Team"
__email__ = "info@agentlobbi.com"

# Core SDK imports
from .sdk import AgentLobbiSDK, create_secure_agent, create_basic_agent

# Security system imports  
from .security import (
    ConsensusReputationSystem,
    DataProtectionLayer, 
    TaskDifficulty,
    AgentReputation,
    DataClassification,
    AccessLevel
)

# Recovery system imports
from .recovery import (
    ConnectionRecoverySystem,
    ConnectionState,
    RecoveryStrategy
)

# Tracking system imports
from .tracking import (
    AgentTrackingSystem,
    ActivityType,
    AgentActivity
)

# Metrics system imports
from .metrics import (
    AgentMetricsTracker,
    AgentMetrics,
    MetricsAPI
)

# Simplified client for basic usage
from .client import Agent, Capability, Message, MessageType

# Utility imports
from .utils import logger, setup_logging

# Main exports
__all__ = [
    # Core SDK
    "AgentLobbiSDK",
    "create_secure_agent", 
    "create_basic_agent",
    
    # Simple client
    "Agent",
    "Capability", 
    "Message",
    "MessageType",
    
    # Security systems
    "ConsensusReputationSystem",
    "DataProtectionLayer",
    "TaskDifficulty", 
    "AgentReputation",
    "DataClassification",
    "AccessLevel",
    
    # Recovery systems
    "ConnectionRecoverySystem",
    "ConnectionState",
    "RecoveryStrategy",
    
    # Tracking systems
    "AgentTrackingSystem",
    "ActivityType",
    "AgentActivity",
    
    # Metrics systems
    "AgentMetricsTracker",
    "AgentMetrics", 
    "MetricsAPI",
    
    # Utilities
    "logger",
    "setup_logging",
]

# Package metadata
__title__ = "agent-lobbi"
__description__ = "Robust multi-agent collaboration platform with security, consensus, and recovery systems"
__url__ = "https://github.com/agent-lobbi/agent-lobbi"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Agent Lobbi Team" 