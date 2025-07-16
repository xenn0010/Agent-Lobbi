"""
Agent Lobbi SDK Package
Complete integration of all security and collaboration features
"""

from .agent_lobbi_sdk import (
    AgentLobbySDK,
    create_secure_agent,
    create_basic_agent
)

# Import security system enums and classes
try:
    from ..security.consensus_system import TaskDifficulty, AgentReputation
    from ..security.data_protection_layer import DataClassification, AccessLevel
    from ..recovery.connection_recovery import RecoveryStrategy, ConnectionState
    from ..tracking.agent_tracking_system import ActivityType
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from security.consensus_system import TaskDifficulty, AgentReputation
    from security.data_protection_layer import DataClassification, AccessLevel
    from recovery.connection_recovery import RecoveryStrategy, ConnectionState
    from tracking.agent_tracking_system import ActivityType

# Import monitoring SDK if available
try:
    from .monitoring_sdk import monitoring_sdk, MonitoringConfig
except ImportError:
    monitoring_sdk = None
    MonitoringConfig = None

__version__ = "1.0.1"
__author__ = "Agent Lobbi Team"
__description__ = "Secure multi-agent collaboration platform with honest engineering"

# Main exports
__all__ = [
    # Core SDK
    'AgentLobbySDK',
    'create_secure_agent',
    'create_basic_agent',
    
    # Security enums
    'TaskDifficulty',
    'DataClassification', 
    'AccessLevel',
    'RecoveryStrategy',
    'ConnectionState',
    'ActivityType',
    'AgentReputation',
    
    # Monitoring (if available)
    'monitoring_sdk',
    'MonitoringConfig'
] 