"""
Agent Tracking System
Allows API key holders to track their agents' activity and interactions
Honest implementation - straightforward monitoring without overselling
"""

import asyncio
import json
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    REGISTERED = "registered"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    COLLABORATION_JOINED = "collaboration_joined"
    DATA_ACCESS = "data_access"
    ERROR_OCCURRED = "error_occurred"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    DELEGATED = "delegated"
    BROWSED = "browsed"
    ACCEPTED = "accepted"
    ERROR = "error"
    TASK_BROWSED = "task_browsed"
    TASK_DELEGATED = "task_delegated"


@dataclass
class AgentActivity:
    """Record of agent activity"""
    activity_id: str
    agent_id: str
    api_key_hash: str  # Hashed API key for privacy
    activity_type: ActivityType
    timestamp: str
    details: Dict[str, Any]
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        if not self.activity_id:
            self.activity_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AgentMetrics:
    """Aggregated metrics for an agent"""
    agent_id: str
    api_key_hash: str
    total_activities: int = 0
    successful_activities: int = 0
    failed_activities: int = 0
    total_runtime_seconds: float = 0.0
    collaborations_count: int = 0
    data_accesses_count: int = 0
    last_active: str = ""
    performance_score: float = 1.0
    
    def __post_init__(self):
        if not self.last_active:
            self.last_active = datetime.now(timezone.utc).isoformat()


class AgentTrackingSystem:
    """
    Simple, honest agent tracking system
    Tracks what agents do without overhyped analytics
    """
    
    def __init__(self, db_path: str = "agent_tracking.db"):
        self.db_path = db_path
        self.api_keys: Dict[str, str] = {}  # api_key -> hashed_key
        self.agent_activities: List[AgentActivity] = []
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Tracking settings
        self.max_activities_per_agent = 1000
        self.metrics_update_interval = 60  # 1 minute
        
        self._initialize_database()
        self._start_background_tasks()
    
    def _initialize_database(self):
        """Initialize database for tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # API keys table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key_hash TEXT PRIMARY KEY,
                    created_at TEXT,
                    last_used TEXT,
                    is_active BOOLEAN
                )
            ''')
            
            # Agent activities table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_activities (
                    activity_id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    api_key_hash TEXT,
                    activity_type TEXT,
                    timestamp TEXT,
                    details TEXT,
                    duration_seconds REAL,
                    success BOOLEAN,
                    error_message TEXT
                )
            ''')
            
            # Agent metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_metrics (
                    agent_id TEXT PRIMARY KEY,
                    api_key_hash TEXT,
                    total_activities INTEGER,
                    successful_activities INTEGER,
                    failed_activities INTEGER,
                    total_runtime_seconds REAL,
                    collaborations_count INTEGER,
                    data_accesses_count INTEGER,
                    last_active TEXT,
                    performance_score REAL,
                    updated_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Agent tracking database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracking database: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks for metrics updates"""
        asyncio.create_task(self._periodic_metrics_update())
    
    def generate_api_key(self) -> str:
        """Generate a new API key for tracking"""
        api_key = f"ak_{uuid.uuid4().hex}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        self.api_keys[api_key] = api_key_hash
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO api_keys (api_key_hash, created_at, last_used, is_active)
                VALUES (?, ?, ?, ?)
            ''', (
                api_key_hash,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
                True
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save API key: {e}")
        
        logger.info(f"New API key generated: {api_key[:10]}...")
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key"""
        if api_key in self.api_keys:
            # Update last used timestamp
            api_key_hash = self.api_keys[api_key]
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    UPDATE api_keys 
                    SET last_used = ? 
                    WHERE api_key_hash = ?
                ''', (datetime.now(timezone.utc).isoformat(), api_key_hash))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to update API key usage: {e}")
            
            return True
        return False
    
    async def track_agent_activity(self,
                                 agent_id: str,
                                 api_key: str,
                                 activity_type: ActivityType,
                                 details: Dict[str, Any] = None,
                                 duration_seconds: Optional[float] = None,
                                 success: bool = True,
                                 error_message: str = "") -> bool:
        """Track an agent activity"""
        
        if not self.validate_api_key(api_key):
            logger.warning(f"Invalid API key used for tracking: {api_key[:10]}...")
            return False
        
        api_key_hash = self.api_keys[api_key]
        
        activity = AgentActivity(
            activity_id=str(uuid.uuid4()),
            agent_id=agent_id,
            api_key_hash=api_key_hash,
            activity_type=activity_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            details=details or {},
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message
        )
        
        self.agent_activities.append(activity)
        await self._save_activity(activity)
        
        # Update metrics
        await self._update_agent_metrics(agent_id, api_key_hash, activity)
        
        logger.info(f"Activity tracked: {agent_id} - {activity_type.value}")
        return True
    
    async def start_agent_session(self,
                                agent_id: str,
                                api_key: str,
                                session_details: Dict[str, Any] = None) -> Optional[str]:
        """Start tracking an agent session"""
        
        if not self.validate_api_key(api_key):
            return None
        
        session_id = str(uuid.uuid4())
        session = {
            'session_id': session_id,
            'agent_id': agent_id,
            'api_key_hash': self.api_keys[api_key],
            'started_at': datetime.now(timezone.utc).isoformat(),
            'last_activity': datetime.now(timezone.utc).isoformat(),
            'details': session_details or {},
            'activity_count': 0
        }
        
        self.active_sessions[session_id] = session
        
        # Track session start
        await self.track_agent_activity(
            agent_id=agent_id,
            api_key=api_key,
            activity_type=ActivityType.REGISTERED,
            details={'session_id': session_id, 'session_details': session_details}
        )
        
        logger.info(f"Agent session started: {agent_id} (session: {session_id})")
        return session_id
    
    async def end_agent_session(self, session_id: str) -> bool:
        """End an agent session"""
        
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Calculate session duration
        started_at = datetime.fromisoformat(session['started_at'])
        ended_at = datetime.now(timezone.utc)
        duration = (ended_at - started_at).total_seconds()
        
        # Find API key from hash
        api_key = None
        for key, hash_val in self.api_keys.items():
            if hash_val == session['api_key_hash']:
                api_key = key
                break
        
        if api_key:
            await self.track_agent_activity(
                agent_id=session['agent_id'],
                api_key=api_key,
                activity_type=ActivityType.CONNECTION_LOST,
                details={
                    'session_id': session_id,
                    'session_duration': duration,
                    'activities_in_session': session['activity_count']
                },
                duration_seconds=duration
            )
        
        del self.active_sessions[session_id]
        logger.info(f"Agent session ended: {session['agent_id']} (duration: {duration:.1f}s)")
        return True
    
    async def _update_agent_metrics(self, 
                                  agent_id: str, 
                                  api_key_hash: str, 
                                  activity: AgentActivity):
        """Update aggregated metrics for an agent"""
        
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                api_key_hash=api_key_hash
            )
        
        metrics = self.agent_metrics[agent_id]
        metrics.total_activities += 1
        metrics.last_active = activity.timestamp
        
        if activity.success:
            metrics.successful_activities += 1
        else:
            metrics.failed_activities += 1
        
        if activity.duration_seconds:
            metrics.total_runtime_seconds += activity.duration_seconds
        
        # Count specific activity types
        if activity.activity_type == ActivityType.COLLABORATION_JOINED:
            metrics.collaborations_count += 1
        elif activity.activity_type == ActivityType.DATA_ACCESS:
            metrics.data_accesses_count += 1
        
        # Calculate performance score (simple success rate)
        if metrics.total_activities > 0:
            metrics.performance_score = metrics.successful_activities / metrics.total_activities
        
        await self._save_metrics(metrics)
    
    def get_agent_activities(self, 
                           api_key: str,
                           agent_id: Optional[str] = None,
                           activity_type: Optional[ActivityType] = None,
                           limit: int = 100,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None) -> List[AgentActivity]:
        """Get activities for agents owned by API key holder"""
        
        if not self.validate_api_key(api_key):
            return []
        
        api_key_hash = self.api_keys[api_key]
        
        # Filter activities
        filtered_activities = []
        for activity in self.agent_activities:
            # Check API key ownership
            if activity.api_key_hash != api_key_hash:
                continue
            
            # Check agent ID filter
            if agent_id and activity.agent_id != agent_id:
                continue
            
            # Check activity type filter
            if activity_type and activity.activity_type != activity_type:
                continue
            
            # Check time range
            if start_time:
                if activity.timestamp < start_time:
                    continue
            
            if end_time:
                if activity.timestamp > end_time:
                    continue
            
            filtered_activities.append(activity)
        
        # Sort by timestamp (newest first) and limit
        filtered_activities.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_activities[:limit]
    
    def get_agent_metrics(self, api_key: str, agent_id: Optional[str] = None) -> List[AgentMetrics]:
        """Get metrics for agents owned by API key holder"""
        
        if not self.validate_api_key(api_key):
            return []
        
        api_key_hash = self.api_keys[api_key]
        
        # Filter metrics
        filtered_metrics = []
        for metrics in self.agent_metrics.values():
            if metrics.api_key_hash != api_key_hash:
                continue
            
            if agent_id and metrics.agent_id != agent_id:
                continue
            
            filtered_metrics.append(metrics)
        
        return filtered_metrics
    
    def get_active_sessions(self, api_key: str) -> List[Dict[str, Any]]:
        """Get active sessions for API key holder"""
        
        if not self.validate_api_key(api_key):
            return []
        
        api_key_hash = self.api_keys[api_key]
        
        filtered_sessions = []
        for session in self.active_sessions.values():
            if session['api_key_hash'] == api_key_hash:
                filtered_sessions.append(session)
        
        return filtered_sessions
    
    def get_tracking_summary(self, api_key: str) -> Dict[str, Any]:
        """Get overall tracking summary for API key holder"""
        
        if not self.validate_api_key(api_key):
            return {}
        
        api_key_hash = self.api_keys[api_key]
        
        # Count agents
        agent_ids = set()
        total_activities = 0
        successful_activities = 0
        
        for activity in self.agent_activities:
            if activity.api_key_hash == api_key_hash:
                agent_ids.add(activity.agent_id)
                total_activities += 1
                if activity.success:
                    successful_activities += 1
        
        # Active sessions count
        active_sessions = len([s for s in self.active_sessions.values() 
                             if s['api_key_hash'] == api_key_hash])
        
        # Calculate success rate
        success_rate = (successful_activities / total_activities * 100) if total_activities > 0 else 0
        
        return {
            'total_agents': len(agent_ids),
            'active_sessions': active_sessions,
            'total_activities': total_activities,
            'successful_activities': successful_activities,
            'success_rate': round(success_rate, 2),
            'agent_ids': list(agent_ids),
            'tracking_period': {
                'start': self.agent_activities[0].timestamp if self.agent_activities else None,
                'end': self.agent_activities[-1].timestamp if self.agent_activities else None
            }
        }
    
    async def _periodic_metrics_update(self):
        """Periodically update and save metrics"""
        while True:
            try:
                await asyncio.sleep(self.metrics_update_interval)
                
                # Update active session activity counts
                for session_id, session in self.active_sessions.items():
                    # Count recent activities for this session
                    api_key_hash = session['api_key_hash']
                    agent_id = session['agent_id']
                    
                    recent_activities = [
                        a for a in self.agent_activities[-50:]  # Check last 50 activities
                        if a.api_key_hash == api_key_hash and a.agent_id == agent_id
                    ]
                    
                    session['activity_count'] = len(recent_activities)
                    session['last_activity'] = datetime.now(timezone.utc).isoformat()
                
                # Clean up old activities if needed
                if len(self.agent_activities) > self.max_activities_per_agent * 10:
                    # Keep only recent activities
                    self.agent_activities = self.agent_activities[-self.max_activities_per_agent * 5:]
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
    
    async def _save_activity(self, activity: AgentActivity):
        """Save activity to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO agent_activities 
                (activity_id, agent_id, api_key_hash, activity_type, timestamp, 
                 details, duration_seconds, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                activity.activity_id,
                activity.agent_id,
                activity.api_key_hash,
                activity.activity_type.value,
                activity.timestamp,
                json.dumps(activity.details),
                activity.duration_seconds,
                activity.success,
                activity.error_message
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save activity: {e}")
    
    async def _save_metrics(self, metrics: AgentMetrics):
        """Save metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO agent_metrics 
                (agent_id, api_key_hash, total_activities, successful_activities, 
                 failed_activities, total_runtime_seconds, collaborations_count, 
                 data_accesses_count, last_active, performance_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.agent_id,
                metrics.api_key_hash,
                metrics.total_activities,
                metrics.successful_activities,
                metrics.failed_activities,
                metrics.total_runtime_seconds,
                metrics.collaborations_count,
                metrics.data_accesses_count,
                metrics.last_active,
                metrics.performance_score,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            api_key_hash = self.api_keys[api_key]
            
            # Mark as inactive in database
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    UPDATE api_keys 
                    SET is_active = ? 
                    WHERE api_key_hash = ?
                ''', (False, api_key_hash))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to revoke API key: {e}")
                return False
            
            # Remove from memory
            del self.api_keys[api_key]
            logger.info(f"API key revoked: {api_key[:10]}...")
            return True
        
        return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall tracking system statistics"""
        return {
            'total_api_keys': len(self.api_keys),
            'total_activities': len(self.agent_activities),
            'active_sessions': len(self.active_sessions),
            'tracked_agents': len(self.agent_metrics),
            'activities_per_minute': self.metrics_update_interval,
            'max_activities_per_agent': self.max_activities_per_agent
        } 