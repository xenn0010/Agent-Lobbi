"""
Connection Recovery System for Agent Lobbi
==========================================
Automatic connection recovery and resilience for agent communications.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import time
import asyncio

class ConnectionState(Enum):
    """Connection state enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

@dataclass
class ConnectionInfo:
    """Information about a connection"""
    connection_id: str
    agent_id: str
    connection_type: str
    endpoint: str
    state: ConnectionState
    last_heartbeat: float
    failure_count: int = 0
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not hasattr(self, 'last_heartbeat') or self.last_heartbeat is None:
            self.last_heartbeat = time.time()

class ConnectionRecoverySystem:
    """
    Automatic connection recovery and resilience system.
    
    This is a placeholder implementation for PyPI packaging.
    In production, this would implement:
    - Automatic reconnection with exponential backoff
    - Circuit breaker patterns
    - Health monitoring and alerting
    - Load balancing across endpoints
    - Graceful degradation strategies
    """
    
    def __init__(self, db_path: str = "connection_recovery.db"):
        self.db_path = db_path
        self.connections: Dict[str, ConnectionInfo] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.heartbeat_interval = 30  # seconds
        self.max_retry_attempts = 5
        
    async def register_connection(self, 
                                agent_id: str, 
                                connection_type: str,
                                endpoint: str,
                                metadata: Dict[str, Any] = None) -> str:
        """Register a new connection for monitoring"""
        connection_id = f"{agent_id}_{connection_type}_{int(time.time())}"
        
        connection = ConnectionInfo(
            connection_id=connection_id,
            agent_id=agent_id,
            connection_type=connection_type,
            endpoint=endpoint,
            state=ConnectionState.CONNECTED,
            last_heartbeat=time.time(),
            metadata=metadata or {}
        )
        
        self.connections[connection_id] = connection
        self.recovery_strategies[connection_id] = RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        return connection_id
    
    async def update_heartbeat(self, connection_id: str) -> bool:
        """Update heartbeat for a connection"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.last_heartbeat = time.time()
        
        if connection.state == ConnectionState.DISCONNECTED:
            connection.state = ConnectionState.CONNECTED
            connection.failure_count = 0
        
        return True
    
    async def report_connection_failure(self, connection_id: str, error: str = "") -> bool:
        """Report a connection failure"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.state = ConnectionState.FAILED
        connection.failure_count += 1
        connection.metadata['last_error'] = error
        connection.metadata['last_failure'] = time.time()
        
        # Trigger recovery if within retry limits
        if connection.recovery_attempts < self.max_retry_attempts:
            await self._initiate_recovery(connection_id)
        
        return True
    
    async def _initiate_recovery(self, connection_id: str):
        """Initiate connection recovery"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        connection.state = ConnectionState.RECONNECTING
        connection.recovery_attempts += 1
        
        strategy = self.recovery_strategies.get(connection_id, RecoveryStrategy.EXPONENTIAL_BACKOFF)
        
        # Calculate delay based on strategy
        if strategy == RecoveryStrategy.IMMEDIATE:
            delay = 0
        elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(2 ** connection.recovery_attempts, 60)  # Cap at 60 seconds
        elif strategy == RecoveryStrategy.SCHEDULED:
            delay = 30  # Fixed 30 second delay
        else:
            return  # Manual recovery required
        
        # Schedule recovery attempt
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Simulate recovery attempt
        # In production, this would actually try to reconnect
        success = connection.failure_count < 3  # Simple simulation
        
        if success:
            connection.state = ConnectionState.CONNECTED
            connection.failure_count = 0
            connection.last_heartbeat = time.time()
        else:
            connection.state = ConnectionState.FAILED
    
    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific connection"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        
        return {
            "connection_id": connection_id,
            "agent_id": connection.agent_id,
            "state": connection.state.value,
            "last_heartbeat": connection.last_heartbeat,
            "failure_count": connection.failure_count,
            "recovery_attempts": connection.recovery_attempts,
            "uptime_seconds": time.time() - connection.last_heartbeat if connection.state == ConnectionState.CONNECTED else 0
        }
    
    def get_agent_connections(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all connections for an agent"""
        agent_connections = [
            conn for conn in self.connections.values() 
            if conn.agent_id == agent_id
        ]
        
        return [
            {
                "connection_id": conn.connection_id,
                "connection_type": conn.connection_type,
                "state": conn.state.value,
                "failure_count": conn.failure_count,
                "last_heartbeat": conn.last_heartbeat
            }
            for conn in agent_connections
        ]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        total_connections = len(self.connections)
        
        if total_connections == 0:
            return {
                "total_connections": 0,
                "healthy_connections": 0,
                "failed_connections": 0,
                "health_score": 100,
                "status": "no_connections"
            }
        
        healthy = sum(1 for c in self.connections.values() if c.state == ConnectionState.CONNECTED)
        failed = sum(1 for c in self.connections.values() if c.state == ConnectionState.FAILED)
        reconnecting = sum(1 for c in self.connections.values() if c.state == ConnectionState.RECONNECTING)
        
        health_score = (healthy / total_connections) * 100
        
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "total_connections": total_connections,
            "healthy_connections": healthy,
            "failed_connections": failed,
            "reconnecting_connections": reconnecting,
            "health_score": round(health_score, 1),
            "status": status,
            "recovery_system_active": True
        }
    
    async def cleanup_stale_connections(self, max_age_hours: int = 24):
        """Clean up old/stale connections"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        stale_connections = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.last_heartbeat < cutoff_time
        ]
        
        for conn_id in stale_connections:
            del self.connections[conn_id]
            if conn_id in self.recovery_strategies:
                del self.recovery_strategies[conn_id]
        
        return len(stale_connections)

# Global instance for easy access
recovery_system = ConnectionRecoverySystem()

def get_recovery_system() -> ConnectionRecoverySystem:
    """Get the global recovery system instance"""
    return recovery_system 