"""
Agent Lobbi - Production Ready Multi-Agent Collaboration Platform
Honest engineering with practical security features
"""

# Core SDK exports
from .sdk import (
    AgentLobbySDK,
    create_secure_agent,
    create_basic_agent,
    TaskDifficulty,
    DataClassification,
    AccessLevel,
    RecoveryStrategy,
    ActivityType
)

# Core components
from .core.lobby import Lobby
from .core.message import Message, MessageType, MessagePriority

# Security systems
from .security.consensus_system import ConsensusReputationSystem
from .security.data_protection_layer import DataProtectionLayer
from .recovery.connection_recovery import ConnectionRecoverySystem
from .tracking.agent_tracking_system import AgentTrackingSystem

__version__ = "1.0.0"
__author__ = "Agent Lobbi Team"
__description__ = "The best engineered protocol-based multi-agent collaboration platform"

# Main public API
__all__ = [
    # SDK - Primary interface
    'AgentLobbySDK',
    'create_secure_agent',
    'create_basic_agent',
    
    # Core components
    'Lobby',
    'Message',
    'MessageType',
    'MessagePriority',
    
    # Security systems
    'ConsensusReputationSystem',
    'DataProtectionLayer',
    'ConnectionRecoverySystem',
    'AgentTrackingSystem',
    
    # Enums and constants
    'TaskDifficulty',
    'DataClassification',
    'AccessLevel', 
    'RecoveryStrategy',
    'ActivityType'
] 