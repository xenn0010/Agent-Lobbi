"""
Agent Tracking System for Agent Lobbi
=====================================
Comprehensive tracking and analytics for agent behavior and performance.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import time
import secrets
import json

class ActivityType(Enum):
    """Types of agent activities to track"""
    REGISTERED = "registered"
    TASK_RECEIVED = "task_received"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    COLLABORATION_STARTED = "collaboration_started"
    COLLABORATION_ENDED = "collaboration_ended"
    STATUS_CHANGED = "status_changed"
    ERROR_OCCURRED = "error_occurred"

@dataclass
class AgentActivity:
    """Represents a tracked agent activity"""
    activity_id: str
    agent_id: str
    api_key: str
    activity_type: ActivityType
    timestamp: float
    session_id: str
    metadata: Dict[str, Any] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class AgentSession:
    """Represents an agent session"""
    session_id: str
    agent_id: str
    api_key: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"
    activities_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not hasattr(self, 'start_time') or self.start_time is None:
            self.start_time = time.time()

class AgentTrackingSystem:
    """
    Comprehensive agent tracking and analytics system.
    
    This is a placeholder implementation for PyPI packaging.
    In production, this would implement:
    - Real-time activity streaming
    - Advanced analytics and ML insights
    - Anomaly detection
    - Performance optimization recommendations
    - Privacy-preserving analytics
    """
    
    def __init__(self, db_path: str = "agent_tracking.db"):
        self.db_path = db_path
        self.activities: List[AgentActivity] = []
        self.sessions: Dict[str, AgentSession] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> agent_id
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        
    def generate_api_key(self) -> str:
        """Generate a secure API key for tracking"""
        return f"al_{secrets.token_hex(16)}"
    
    async def start_agent_session(self, 
                                agent_id: str, 
                                api_key: str,
                                metadata: Dict[str, Any] = None) -> str:
        """Start a new tracking session for an agent"""
        session_id = f"session_{agent_id}_{int(time.time())}_{secrets.token_hex(4)}"
        
        session = AgentSession(
            session_id=session_id,
            agent_id=agent_id,
            api_key=api_key,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        self.api_keys[api_key] = agent_id
        
        # Initialize agent stats if not exists
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {
                "total_sessions": 0,
                "total_activities": 0,
                "first_seen": time.time(),
                "last_seen": time.time()
            }
        
        self.agent_stats[agent_id]["total_sessions"] += 1
        self.agent_stats[agent_id]["last_seen"] = time.time()
        
        return session_id
    
    async def track_agent_activity(self, 
                                 agent_id: str,
                                 api_key: str,
                                 activity_type: ActivityType,
                                 metadata: Dict[str, Any] = None,
                                 duration_seconds: Optional[float] = None) -> str:
        """Track a specific agent activity"""
        
        # Verify API key
        if api_key not in self.api_keys or self.api_keys[api_key] != agent_id:
            raise ValueError("Invalid API key for agent")
        
        activity_id = f"activity_{int(time.time())}_{secrets.token_hex(4)}"
        
        # Find active session
        active_session = None
        for session in self.sessions.values():
            if session.agent_id == agent_id and session.status == "active":
                active_session = session
                break
        
        if not active_session:
            # Create a temporary session
            session_id = await self.start_agent_session(agent_id, api_key)
            active_session = self.sessions[session_id]
        
        activity = AgentActivity(
            activity_id=activity_id,
            agent_id=agent_id,
            api_key=api_key,
            activity_type=activity_type,
            timestamp=time.time(),
            session_id=active_session.session_id,
            metadata=metadata or {},
            duration_seconds=duration_seconds
        )
        
        self.activities.append(activity)
        active_session.activities_count += 1
        
        # Update agent stats
        if agent_id in self.agent_stats:
            self.agent_stats[agent_id]["total_activities"] += 1
            self.agent_stats[agent_id]["last_seen"] = time.time()
        
        return activity_id
    
    async def end_agent_session(self, session_id: str):
        """End an agent session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.end_time = time.time()
            session.status = "ended"
    
    def get_agent_activities(self, 
                           agent_id: str, 
                           limit: int = 100,
                           activity_type: Optional[ActivityType] = None) -> List[Dict[str, Any]]:
        """Get activities for a specific agent"""
        agent_activities = [
            activity for activity in self.activities 
            if activity.agent_id == agent_id
        ]
        
        if activity_type:
            agent_activities = [
                activity for activity in agent_activities
                if activity.activity_type == activity_type
            ]
        
        # Sort by timestamp (most recent first)
        agent_activities.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        agent_activities = agent_activities[:limit]
        
        return [
            {
                "activity_id": activity.activity_id,
                "activity_type": activity.activity_type.value,
                "timestamp": activity.timestamp,
                "session_id": activity.session_id,
                "metadata": activity.metadata,
                "duration_seconds": activity.duration_seconds
            }
            for activity in agent_activities
        ]
    
    def get_agent_sessions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for an agent"""
        agent_sessions = [
            session for session in self.sessions.values()
            if session.agent_id == agent_id
        ]
        
        return [
            {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "status": session.status,
                "activities_count": session.activities_count,
                "duration_seconds": (session.end_time or time.time()) - session.start_time
            }
            for session in agent_sessions
        ]
    
    def get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for an agent"""
        if agent_id not in self.agent_stats:
            return {
                "agent_id": agent_id,
                "error": "Agent not found in tracking system"
            }
        
        stats = self.agent_stats[agent_id]
        
        # Calculate activity patterns
        agent_activities = [a for a in self.activities if a.agent_id == agent_id]
        
        activity_types_count = {}
        for activity in agent_activities:
            activity_type = activity.activity_type.value
            activity_types_count[activity_type] = activity_types_count.get(activity_type, 0) + 1
        
        # Calculate average session duration
        agent_sessions = [s for s in self.sessions.values() if s.agent_id == agent_id]
        completed_sessions = [s for s in agent_sessions if s.end_time is not None]
        
        avg_session_duration = 0
        if completed_sessions:
            total_duration = sum(s.end_time - s.start_time for s in completed_sessions)
            avg_session_duration = total_duration / len(completed_sessions)
        
        return {
            "agent_id": agent_id,
            "total_sessions": stats["total_sessions"],
            "total_activities": stats["total_activities"],
            "first_seen": stats["first_seen"],
            "last_seen": stats["last_seen"],
            "activity_breakdown": activity_types_count,
            "average_session_duration_seconds": round(avg_session_duration, 2),
            "tracking_status": "active"
        }
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics"""
        total_agents = len(self.agent_stats)
        total_sessions = len(self.sessions)
        total_activities = len(self.activities)
        
        # Active sessions
        active_sessions = sum(1 for s in self.sessions.values() if s.status == "active")
        
        # Activity breakdown
        activity_breakdown = {}
        for activity in self.activities:
            activity_type = activity.activity_type.value
            activity_breakdown[activity_type] = activity_breakdown.get(activity_type, 0) + 1
        
        # Calculate system health
        if total_agents > 0:
            recent_threshold = time.time() - 3600  # 1 hour
            active_agents = sum(
                1 for stats in self.agent_stats.values()
                if stats["last_seen"] > recent_threshold
            )
            activity_rate = active_agents / total_agents
        else:
            activity_rate = 0
        
        return {
            "total_agents_tracked": total_agents,
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_activities": total_activities,
            "activity_breakdown": activity_breakdown,
            "system_activity_rate": round(activity_rate * 100, 1),
            "tracking_system_status": "operational"
        }
    
    def cleanup_old_data(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old tracking data"""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        # Remove old activities
        old_activities = [a for a in self.activities if a.timestamp < cutoff_time]
        self.activities = [a for a in self.activities if a.timestamp >= cutoff_time]
        
        # Remove old sessions
        old_sessions = [s for s in self.sessions.values() if s.start_time < cutoff_time]
        for session in old_sessions:
            del self.sessions[session.session_id]
        
        return {
            "activities_removed": len(old_activities),
            "sessions_removed": len(old_sessions)
        }

# Global instance for easy access
tracking_system = AgentTrackingSystem()

def get_tracking_system() -> AgentTrackingSystem:
    """Get the global tracking system instance"""
    return tracking_system 