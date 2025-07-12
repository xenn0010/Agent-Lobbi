#!/usr/bin/env python3
"""
Agent Metrics Tracker
=====================
Comprehensive tracking system for monitoring agent activities, collaborations, and performance metrics.
"""

import time
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentActivity:
    """Represents a single agent activity record"""
    agent_id: str
    activity_type: str  # 'task_received', 'task_completed', 'task_failed', 'collaboration_started', 'collaboration_ended'
    task_id: Optional[str] = None
    collaboration_id: Optional[str] = None
    timestamp: float = None
    duration: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentMetrics:
    """Aggregated metrics for an agent"""
    agent_id: str
    agent_name: str
    agent_type: str
    total_requests: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = 0.0
    avg_task_duration: float = 0.0
    collaboration_count: int = 0
    last_activity: Optional[float] = None
    status: str = 'offline'
    
    def calculate_success_rate(self):
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            self.success_rate = 0.0
        else:
            self.success_rate = round((self.completed_tasks / self.total_requests) * 100, 1)

class AgentMetricsTracker:
    """Comprehensive agent metrics tracking system"""
    
    def __init__(self, db_path: str = "agent_metrics.db"):
        self.db_path = db_path
        self.agent_registry: Dict[str, AgentMetrics] = {}
        self.active_tasks: Dict[str, Dict] = {}  # task_id -> {agent_id, start_time, task_info}
        self.active_collaborations: Dict[str, Dict] = {}  # collab_id -> {agents, start_time, metadata}
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for persistent metrics storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Agent activities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    activity_type TEXT NOT NULL,
                    task_id TEXT,
                    collaboration_id TEXT,
                    timestamp REAL NOT NULL,
                    duration REAL,
                    success BOOLEAN NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Agent metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    agent_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    total_requests INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    failed_tasks INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    avg_task_duration REAL DEFAULT 0.0,
                    collaboration_count INTEGER DEFAULT 0,
                    last_activity REAL,
                    status TEXT DEFAULT 'offline',
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Collaboration sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collaboration_sessions (
                    collaboration_id TEXT PRIMARY KEY,
                    agents TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    status TEXT DEFAULT 'active',
                    result TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def register_agent(self, agent_id: str, agent_name: str, agent_type: str):
        """Register a new agent in the tracking system"""
        try:
            metrics = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                agent_type=agent_type,
                status='online'
            )
            
            self.agent_registry[agent_id] = metrics
            self._save_agent_metrics(metrics)
            
            # Log agent registration
            activity = AgentActivity(
                agent_id=agent_id,
                activity_type='agent_registered',
                metadata={'name': agent_name, 'type': agent_type}
            )
            self._log_activity(activity)
            
            logger.info(f"Agent registered: {agent_name} ({agent_id})")
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
    
    def track_task_start(self, agent_id: str, task_id: str, task_info: Dict[str, Any]):
        """Track when an agent starts a task"""
        try:
            # Update agent metrics
            if agent_id in self.agent_registry:
                self.agent_registry[agent_id].total_requests += 1
                self.agent_registry[agent_id].last_activity = time.time()
                self.agent_registry[agent_id].status = 'online'
            
            # Track active task
            self.active_tasks[task_id] = {
                'agent_id': agent_id,
                'start_time': time.time(),
                'task_info': task_info
            }
            
            # Log activity
            activity = AgentActivity(
                agent_id=agent_id,
                activity_type='task_received',
                task_id=task_id,
                metadata=task_info
            )
            self._log_activity(activity)
            
            logger.info(f"Task started: {task_id} by {agent_id}")
            
        except Exception as e:
            logger.error(f"Task start tracking failed: {e}")
    
    def track_task_completion(self, agent_id: str, task_id: str, success: bool = True, result: Any = None):
        """Track when an agent completes a task"""
        try:
            duration = None
            
            # Calculate duration if task was tracked
            if task_id in self.active_tasks:
                start_time = self.active_tasks[task_id]['start_time']
                duration = time.time() - start_time
                del self.active_tasks[task_id]
            
            # Update agent metrics
            if agent_id in self.agent_registry:
                metrics = self.agent_registry[agent_id]
                if success:
                    metrics.completed_tasks += 1
                else:
                    metrics.failed_tasks += 1
                
                metrics.last_activity = time.time()
                metrics.calculate_success_rate()
                
                # Update average task duration
                if duration and success:
                    if metrics.avg_task_duration == 0:
                        metrics.avg_task_duration = duration
                    else:
                        metrics.avg_task_duration = (metrics.avg_task_duration + duration) / 2
            
            # Log activity
            activity = AgentActivity(
                agent_id=agent_id,
                activity_type='task_completed' if success else 'task_failed',
                task_id=task_id,
                duration=duration,
                success=success,
                metadata={'result': str(result) if result else None}
            )
            self._log_activity(activity)
            
            logger.info(f"Task {'completed' if success else 'failed'}: {task_id} by {agent_id}")
            
        except Exception as e:
            logger.error(f"Task completion tracking failed: {e}")
    
    def track_collaboration_start(self, collaboration_id: str, agent_ids: List[str], metadata: Dict[str, Any] = None):
        """Track when a collaboration session starts"""
        try:
            self.active_collaborations[collaboration_id] = {
                'agents': agent_ids,
                'start_time': time.time(),
                'metadata': metadata or {}
            }
            
            # Update collaboration count for each agent
            for agent_id in agent_ids:
                if agent_id in self.agent_registry:
                    self.agent_registry[agent_id].collaboration_count += 1
                    self.agent_registry[agent_id].last_activity = time.time()
                
                # Log activity for each agent
                activity = AgentActivity(
                    agent_id=agent_id,
                    activity_type='collaboration_started',
                    collaboration_id=collaboration_id,
                    metadata=metadata
                )
                self._log_activity(activity)
            
            # Save collaboration session
            self._save_collaboration_session(collaboration_id, agent_ids, metadata)
            
            logger.info(f"Collaboration started: {collaboration_id} with agents {agent_ids}")
            
        except Exception as e:
            logger.error(f"Collaboration start tracking failed: {e}")
    
    def track_collaboration_end(self, collaboration_id: str, result: str = 'completed'):
        """Track when a collaboration session ends"""
        try:
            if collaboration_id in self.active_collaborations:
                collab_data = self.active_collaborations[collaboration_id]
                duration = time.time() - collab_data['start_time']
                
                # Log activity for each agent
                for agent_id in collab_data['agents']:
                    activity = AgentActivity(
                        agent_id=agent_id,
                        activity_type='collaboration_ended',
                        collaboration_id=collaboration_id,
                        duration=duration,
                        success=result == 'completed',
                        metadata={'result': result}
                    )
                    self._log_activity(activity)
                
                # Update collaboration session
                self._update_collaboration_session(collaboration_id, result)
                
                del self.active_collaborations[collaboration_id]
                logger.info(f"Collaboration ended: {collaboration_id} - {result}")
                
        except Exception as e:
            logger.error(f"Collaboration end tracking failed: {e}")
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status (online/offline)"""
        try:
            if agent_id in self.agent_registry:
                self.agent_registry[agent_id].status = status
                self.agent_registry[agent_id].last_activity = time.time()
                self._save_agent_metrics(self.agent_registry[agent_id])
                
        except Exception as e:
            logger.error(f"Agent status update failed: {e}")
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent"""
        return self.agent_registry.get(agent_id)
    
    def get_all_agent_metrics(self) -> List[AgentMetrics]:
        """Get metrics for all registered agents"""
        return list(self.agent_registry.values())
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get aggregated system metrics"""
        try:
            total_agents = len(self.agent_registry)
            active_agents = len([a for a in self.agent_registry.values() if a.status == 'online'])
            
            total_requests = sum(a.total_requests for a in self.agent_registry.values())
            completed_tasks = sum(a.completed_tasks for a in self.agent_registry.values())
            failed_tasks = sum(a.failed_tasks for a in self.agent_registry.values())
            
            success_rate = 0.0
            if total_requests > 0:
                success_rate = round((completed_tasks / total_requests) * 100, 1)
            
            total_collaborations = len(self._get_collaboration_history())
            active_collaborations = len(self.active_collaborations)
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'total_requests': total_requests,
                'completed_collaborations': completed_tasks,
                'failed_collaborations': failed_tasks,
                'success_rate': success_rate,
                'total_collaborations': total_collaborations,
                'active_collaborations': active_collaborations,
                'status': 'operational' if active_agents > 0 else 'standby'
            }
            
        except Exception as e:
            logger.error(f"System metrics calculation failed: {e}")
            return {}
    
    def _log_activity(self, activity: AgentActivity):
        """Log activity to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO agent_activities 
                (agent_id, activity_type, task_id, collaboration_id, timestamp, duration, success, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                activity.agent_id,
                activity.activity_type,
                activity.task_id,
                activity.collaboration_id,
                activity.timestamp,
                activity.duration,
                activity.success,
                json.dumps(activity.metadata) if activity.metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Activity logging failed: {e}")
    
    def _save_agent_metrics(self, metrics: AgentMetrics):
        """Save agent metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agent_metrics 
                (agent_id, agent_name, agent_type, total_requests, completed_tasks, failed_tasks,
                 success_rate, avg_task_duration, collaboration_count, last_activity, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.agent_id,
                metrics.agent_name,
                metrics.agent_type,
                metrics.total_requests,
                metrics.completed_tasks,
                metrics.failed_tasks,
                metrics.success_rate,
                metrics.avg_task_duration,
                metrics.collaboration_count,
                metrics.last_activity,
                metrics.status,
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Agent metrics save failed: {e}")
    
    def _save_collaboration_session(self, collaboration_id: str, agent_ids: List[str], metadata: Dict[str, Any]):
        """Save collaboration session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO collaboration_sessions 
                (collaboration_id, agents, start_time, status, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                collaboration_id,
                json.dumps(agent_ids),
                time.time(),
                'active',
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Collaboration session save failed: {e}")
    
    def _update_collaboration_session(self, collaboration_id: str, result: str):
        """Update collaboration session result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE collaboration_sessions 
                SET end_time = ?, status = ?, result = ?
                WHERE collaboration_id = ?
            ''', (time.time(), 'completed', result, collaboration_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Collaboration session update failed: {e}")
    
    def _get_collaboration_history(self) -> List[Dict]:
        """Get collaboration history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM collaboration_sessions')
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip([col[0] for col in cursor.description], row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Collaboration history fetch failed: {e}")
            return []
    
    def get_activity_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get activity history, optionally filtered by agent"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if agent_id:
                cursor.execute('''
                    SELECT * FROM agent_activities 
                    WHERE agent_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (agent_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM agent_activities 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(zip([col[0] for col in cursor.description], row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Activity history fetch failed: {e}")
            return []

# Global metrics tracker instance
metrics_tracker = AgentMetricsTracker()

def get_metrics_tracker() -> AgentMetricsTracker:
    """Get the global metrics tracker instance"""
    return metrics_tracker 