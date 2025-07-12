"""
Connection Recovery System
Maps agent-to-agent connections and restores platform after attacks or interruptions
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import pickle
import gzip

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    FAILED = "failed"
    RECOVERING = "recovering"


class RecoveryStrategy(Enum):
    IMMEDIATE = "immediate"      # Try to reconnect right away
    GRADUAL = "gradual"         # Slowly restore connections
    MANUAL = "manual"           # Wait for manual intervention


@dataclass
class AgentConnection:
    """Represents a connection between two agents"""
    agent_a: str
    agent_b: str
    connection_type: str  # 'collaboration', 'data_share', 'workflow', etc.
    established_at: str
    last_activity: str
    state: ConnectionState
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if not self.established_at:
            self.established_at = datetime.now(timezone.utc).isoformat()
        if not self.last_activity:
            self.last_activity = datetime.now(timezone.utc).isoformat()


@dataclass
class RecoverySnapshot:
    """Snapshot of the entire system state for recovery"""
    snapshot_id: str
    timestamp: str
    active_agents: Set[str]
    agent_connections: List[AgentConnection]
    workflow_states: Dict[str, Any]
    collaboration_sessions: Dict[str, Any]
    system_metrics: Dict[str, Any]
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ConnectionRecoverySystem:
    """
    Honest implementation of connection recovery
    Maps connections and provides recovery after system failures
    """
    
    def __init__(self, db_path: str = "recovery.db"):
        self.db_path = db_path
        self.active_connections: Dict[str, AgentConnection] = {}
        self.recovery_snapshots: List[RecoverySnapshot] = []
        self.failed_connections: List[AgentConnection] = []
        
        # Recovery settings
        self.snapshot_interval = 300  # 5 minutes
        self.max_snapshots = 24      # Keep 24 snapshots (2 hours worth)
        self.connection_timeout = 60  # 1 minute
        
        self._initialize_database()
        self._start_background_tasks()
    
    def _initialize_database(self):
        """Initialize database for connection tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Agent connections table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_connections (
                    id TEXT PRIMARY KEY,
                    agent_a TEXT,
                    agent_b TEXT,
                    connection_type TEXT,
                    established_at TEXT,
                    last_activity TEXT,
                    state TEXT,
                    metadata TEXT
                )
            ''')
            
            # Recovery snapshots table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recovery_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    snapshot_data BLOB
                )
            ''')
            
            # Recovery events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT,
                    timestamp TEXT,
                    agent_id TEXT,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Connection recovery database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize recovery database: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring and snapshot tasks"""
        asyncio.create_task(self._periodic_snapshot())
        asyncio.create_task(self._monitor_connections())
    
    async def register_connection(self, 
                                agent_a: str, 
                                agent_b: str,
                                connection_type: str = "collaboration",
                                metadata: Dict[str, Any] = None) -> str:
        """Register a new connection between agents"""
        
        # Create unique connection ID
        connection_id = f"{min(agent_a, agent_b)}_{max(agent_a, agent_b)}_{connection_type}"
        
        connection = AgentConnection(
            agent_a=agent_a,
            agent_b=agent_b,
            connection_type=connection_type,
            established_at=datetime.now(timezone.utc).isoformat(),
            last_activity=datetime.now(timezone.utc).isoformat(),
            state=ConnectionState.ACTIVE,
            metadata=metadata or {}
        )
        
        self.active_connections[connection_id] = connection
        await self._save_connection(connection_id, connection)
        
        logger.info(f"Connection registered: {agent_a} <-> {agent_b} ({connection_type})")
        return connection_id
    
    async def update_connection_activity(self, 
                                       agent_a: str, 
                                       agent_b: str,
                                       connection_type: str = "collaboration"):
        """Update last activity timestamp for a connection"""
        connection_id = f"{min(agent_a, agent_b)}_{max(agent_a, agent_b)}_{connection_type}"
        
        if connection_id in self.active_connections:
            connection = self.active_connections[connection_id]
            connection.last_activity = datetime.now(timezone.utc).isoformat()
            await self._save_connection(connection_id, connection)
    
    async def remove_connection(self, 
                              agent_a: str, 
                              agent_b: str,
                              connection_type: str = "collaboration"):
        """Remove a connection between agents"""
        connection_id = f"{min(agent_a, agent_b)}_{max(agent_a, agent_b)}_{connection_type}"
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            await self._remove_connection_from_db(connection_id)
            logger.info(f"Connection removed: {agent_a} <-> {agent_b} ({connection_type})")
    
    async def create_recovery_snapshot(self, 
                                     active_agents: Set[str],
                                     workflow_states: Dict[str, Any] = None,
                                     collaboration_sessions: Dict[str, Any] = None) -> str:
        """Create a snapshot of current system state"""
        
        snapshot_id = f"snapshot_{int(time.time())}"
        
        # Gather system metrics
        system_metrics = {
            'total_connections': len(self.active_connections),
            'active_agents': len(active_agents),
            'failed_connections': len(self.failed_connections),
            'cpu_usage': 0.0,  # Could integrate with actual system monitoring
            'memory_usage': 0.0
        }
        
        snapshot = RecoverySnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            active_agents=active_agents,
            agent_connections=list(self.active_connections.values()),
            workflow_states=workflow_states or {},
            collaboration_sessions=collaboration_sessions or {},
            system_metrics=system_metrics
        )
        
        self.recovery_snapshots.append(snapshot)
        await self._save_snapshot(snapshot)
        
        # Keep only recent snapshots
        if len(self.recovery_snapshots) > self.max_snapshots:
            old_snapshot = self.recovery_snapshots.pop(0)
            await self._remove_snapshot(old_snapshot.snapshot_id)
        
        logger.info(f"Recovery snapshot created: {snapshot_id}")
        return snapshot_id
    
    async def initiate_recovery(self, 
                              strategy: RecoveryStrategy = RecoveryStrategy.GRADUAL,
                              target_snapshot: Optional[str] = None) -> Dict[str, Any]:
        """Initiate system recovery after an attack or failure"""
        
        logger.warning(f"Initiating system recovery with strategy: {strategy.value}")
        
        # Get the target snapshot
        if target_snapshot:
            snapshot = self._find_snapshot(target_snapshot)
        else:
            # Use the most recent snapshot
            snapshot = self.recovery_snapshots[-1] if self.recovery_snapshots else None
        
        if not snapshot:
            logger.error("No recovery snapshot available")
            return {"success": False, "error": "No recovery snapshot available"}
        
        recovery_plan = await self._create_recovery_plan(snapshot, strategy)
        recovery_results = await self._execute_recovery_plan(recovery_plan)
        
        await self._log_recovery_event("recovery_completed", "", {
            "strategy": strategy.value,
            "snapshot_used": snapshot.snapshot_id,
            "results": recovery_results
        })
        
        return recovery_results
    
    async def _create_recovery_plan(self, 
                                  snapshot: RecoverySnapshot,
                                  strategy: RecoveryStrategy) -> Dict[str, Any]:
        """Create a plan for recovering system state"""
        
        plan = {
            "strategy": strategy.value,
            "snapshot_id": snapshot.snapshot_id,
            "steps": []
        }
        
        # Step 1: Verify which agents are still accessible
        plan["steps"].append({
            "step": "agent_verification",
            "target_agents": list(snapshot.active_agents),
            "priority": "high"
        })
        
        # Step 2: Restore connections based on strategy
        if strategy == RecoveryStrategy.IMMEDIATE:
            # Try to restore all connections at once
            plan["steps"].append({
                "step": "restore_all_connections",
                "connections": [asdict(conn) for conn in snapshot.agent_connections],
                "priority": "high"
            })
        
        elif strategy == RecoveryStrategy.GRADUAL:
            # Restore connections in batches
            critical_connections = [
                conn for conn in snapshot.agent_connections 
                if conn.connection_type in ['workflow', 'critical']
            ]
            normal_connections = [
                conn for conn in snapshot.agent_connections 
                if conn.connection_type not in ['workflow', 'critical']
            ]
            
            plan["steps"].append({
                "step": "restore_critical_connections",
                "connections": [asdict(conn) for conn in critical_connections],
                "priority": "high"
            })
            
            plan["steps"].append({
                "step": "restore_normal_connections", 
                "connections": [asdict(conn) for conn in normal_connections],
                "priority": "medium"
            })
        
        # Step 3: Restore workflow states
        if snapshot.workflow_states:
            plan["steps"].append({
                "step": "restore_workflows",
                "workflows": snapshot.workflow_states,
                "priority": "medium"
            })
        
        # Step 4: Verify system health
        plan["steps"].append({
            "step": "health_verification",
            "checks": ["connection_health", "agent_responsiveness", "data_integrity"],
            "priority": "high"
        })
        
        return plan
    
    async def _execute_recovery_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the recovery plan"""
        
        results = {
            "success": True,
            "steps_completed": 0,
            "steps_failed": 0,
            "restored_connections": 0,
            "failed_connections": 0,
            "details": []
        }
        
        for step in plan["steps"]:
            try:
                step_result = await self._execute_recovery_step(step)
                results["details"].append({
                    "step": step["step"],
                    "success": step_result["success"],
                    "details": step_result
                })
                
                if step_result["success"]:
                    results["steps_completed"] += 1
                    if step["step"].startswith("restore") and "connections" in step:
                        results["restored_connections"] += step_result.get("restored_count", 0)
                else:
                    results["steps_failed"] += 1
                    if step["step"].startswith("restore") and "connections" in step:
                        results["failed_connections"] += step_result.get("failed_count", 0)
                
            except Exception as e:
                logger.error(f"Recovery step failed: {step['step']} - {e}")
                results["steps_failed"] += 1
                results["details"].append({
                    "step": step["step"],
                    "success": False,
                    "error": str(e)
                })
        
        if results["steps_failed"] > 0:
            results["success"] = False
        
        return results
    
    async def _execute_recovery_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single recovery step"""
        
        step_type = step["step"]
        
        if step_type == "agent_verification":
            return await self._verify_agents(step["target_agents"])
        
        elif step_type in ["restore_all_connections", "restore_critical_connections", "restore_normal_connections"]:
            return await self._restore_connections(step["connections"])
        
        elif step_type == "restore_workflows":
            return await self._restore_workflows(step["workflows"])
        
        elif step_type == "health_verification":
            return await self._verify_system_health(step["checks"])
        
        else:
            return {"success": False, "error": f"Unknown step type: {step_type}"}
    
    async def _verify_agents(self, target_agents: List[str]) -> Dict[str, Any]:
        """Verify which agents are still accessible"""
        accessible_agents = []
        failed_agents = []
        
        for agent_id in target_agents:
            try:
                # This would actually ping the agent or check its status
                # For now, we'll simulate it
                if agent_id.startswith("failed_"):
                    failed_agents.append(agent_id)
                else:
                    accessible_agents.append(agent_id)
            except Exception as e:
                failed_agents.append(agent_id)
        
        return {
            "success": len(accessible_agents) > 0,
            "accessible_agents": accessible_agents,
            "failed_agents": failed_agents,
            "accessibility_rate": len(accessible_agents) / len(target_agents) if target_agents else 0
        }
    
    async def _restore_connections(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Restore agent connections"""
        restored_count = 0
        failed_count = 0
        
        for conn_data in connections:
            try:
                # Create connection object
                connection = AgentConnection(**conn_data)
                
                # Try to restore the connection
                connection_id = f"{connection.agent_a}_{connection.agent_b}_{connection.connection_type}"
                connection.state = ConnectionState.RECOVERING
                
                # This would actually attempt to reestablish the connection
                # For now, we'll simulate success/failure
                if not connection.agent_a.startswith("failed_") and not connection.agent_b.startswith("failed_"):
                    connection.state = ConnectionState.ACTIVE
                    self.active_connections[connection_id] = connection
                    await self._save_connection(connection_id, connection)
                    restored_count += 1
                else:
                    connection.state = ConnectionState.FAILED
                    self.failed_connections.append(connection)
                    failed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to restore connection: {e}")
                failed_count += 1
        
        return {
            "success": restored_count > 0,
            "restored_count": restored_count,
            "failed_count": failed_count
        }
    
    async def _restore_workflows(self, workflows: Dict[str, Any]) -> Dict[str, Any]:
        """Restore workflow states"""
        # This would integrate with the workflow management system
        # For now, just simulate restoration
        return {
            "success": True,
            "restored_workflows": len(workflows),
            "message": "Workflow restoration simulated"
        }
    
    async def _verify_system_health(self, checks: List[str]) -> Dict[str, Any]:
        """Verify overall system health after recovery"""
        health_results = {}
        
        for check in checks:
            if check == "connection_health":
                active_connections = len([c for c in self.active_connections.values() 
                                        if c.state == ConnectionState.ACTIVE])
                health_results[check] = {
                    "status": "healthy" if active_connections > 0 else "unhealthy",
                    "active_connections": active_connections
                }
            
            elif check == "agent_responsiveness":
                # This would actually test agent responsiveness
                health_results[check] = {
                    "status": "healthy",
                    "response_time": "simulated"
                }
            
            elif check == "data_integrity":
                # This would verify data hasn't been corrupted
                health_results[check] = {
                    "status": "healthy",
                    "integrity_score": 1.0
                }
        
        overall_healthy = all(result["status"] == "healthy" for result in health_results.values())
        
        return {
            "success": overall_healthy,
            "overall_health": "healthy" if overall_healthy else "degraded",
            "check_results": health_results
        }
    
    async def _periodic_snapshot(self):
        """Create periodic snapshots for recovery"""
        while True:
            try:
                await asyncio.sleep(self.snapshot_interval)
                
                # Get current active agents (this would come from the main system)
                active_agents = set(self.active_connections.keys())
                
                await self.create_recovery_snapshot(active_agents)
                
            except Exception as e:
                logger.error(f"Failed to create periodic snapshot: {e}")
    
    async def _monitor_connections(self):
        """Monitor connection health and detect failures"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                current_time = datetime.now(timezone.utc)
                
                for connection_id, connection in list(self.active_connections.items()):
                    last_activity = datetime.fromisoformat(connection.last_activity)
                    time_since_activity = (current_time - last_activity).total_seconds()
                    
                    if time_since_activity > self.connection_timeout:
                        logger.warning(f"Connection timeout detected: {connection_id}")
                        connection.state = ConnectionState.FAILED
                        self.failed_connections.append(connection)
                        del self.active_connections[connection_id]
                        
                        await self._log_recovery_event("connection_timeout", connection_id, {
                            "agents": [connection.agent_a, connection.agent_b],
                            "last_activity": connection.last_activity
                        })
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
    
    def _find_snapshot(self, snapshot_id: str) -> Optional[RecoverySnapshot]:
        """Find a snapshot by ID"""
        for snapshot in self.recovery_snapshots:
            if snapshot.snapshot_id == snapshot_id:
                return snapshot
        return None
    
    async def _save_connection(self, connection_id: str, connection: AgentConnection):
        """Save connection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO agent_connections 
                (id, agent_a, agent_b, connection_type, established_at, 
                 last_activity, state, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                connection_id,
                connection.agent_a,
                connection.agent_b,
                connection.connection_type,
                connection.established_at,
                connection.last_activity,
                connection.state.value,
                json.dumps(connection.metadata)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save connection: {e}")
    
    async def _save_snapshot(self, snapshot: RecoverySnapshot):
        """Save snapshot to database"""
        try:
            # Compress and serialize snapshot data
            snapshot_data = gzip.compress(pickle.dumps(asdict(snapshot)))
            
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO recovery_snapshots 
                (snapshot_id, timestamp, snapshot_data)
                VALUES (?, ?, ?)
            ''', (snapshot.snapshot_id, snapshot.timestamp, snapshot_data))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    async def _remove_connection_from_db(self, connection_id: str):
        """Remove connection from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM agent_connections WHERE id = ?', (connection_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to remove connection: {e}")
    
    async def _remove_snapshot(self, snapshot_id: str):
        """Remove old snapshot from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('DELETE FROM recovery_snapshots WHERE snapshot_id = ?', (snapshot_id,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to remove snapshot: {e}")
    
    async def _log_recovery_event(self, event_type: str, agent_id: str, details: Dict[str, Any]):
        """Log recovery events"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO recovery_events (event_type, timestamp, agent_id, details)
                VALUES (?, ?, ?, ?)
            ''', (
                event_type,
                datetime.now(timezone.utc).isoformat(),
                agent_id,
                json.dumps(details)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log recovery event: {e}")
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics"""
        return {
            "active_connections": len(self.active_connections),
            "failed_connections": len(self.failed_connections),
            "total_snapshots": len(self.recovery_snapshots),
            "latest_snapshot": self.recovery_snapshots[-1].snapshot_id if self.recovery_snapshots else None,
            "snapshot_interval_seconds": self.snapshot_interval,
            "connection_timeout_seconds": self.connection_timeout
        } 