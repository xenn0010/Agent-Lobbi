"""
Analytics and reporting module for the agent ecosystem.
Provides insights into agent behavior, message patterns, performance metrics, and system usage.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from collections import defaultdict, Counter

# Data analysis imports
import pandas as pd
import numpy as np
from scipy import stats

# Our imports
from .database import DatabaseManager, DatabaseConfig, DatabaseType
from .monitoring import get_monitoring_system


class AnalyticsError(Exception):
    """Raised when analytics operations fail"""
    pass


class TimeRange(Enum):
    """Time range options for analytics"""
    HOUR = "1h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1m"
    QUARTER = "3m"
    YEAR = "1y"


@dataclass
class AgentAnalytics:
    """Analytics data for a specific agent"""
    agent_id: str
    agent_type: str
    total_messages_sent: int
    total_messages_received: int
    total_interactions: int
    avg_response_time: float
    success_rate: float
    active_time: float  # in hours
    capabilities_used: Dict[str, int]
    error_count: int
    last_active: datetime


@dataclass
class SystemAnalytics:
    """System-wide analytics data"""
    total_agents: int
    active_agents: int
    total_messages: int
    total_interactions: int
    avg_system_response_time: float
    peak_concurrent_agents: int
    message_throughput: float  # messages per minute
    error_rate: float
    uptime_percentage: float


@dataclass
class MessagePatternAnalysis:
    """Analysis of message patterns"""
    most_common_message_types: List[Tuple[str, int]]
    peak_hours: List[int]
    conversation_length_stats: Dict[str, float]
    response_time_percentiles: Dict[str, float]
    message_size_stats: Dict[str, float]


@dataclass
class InteractionAnalysis:
    """Analysis of agent interactions"""
    most_active_pairs: List[Tuple[str, str, int]]
    interaction_types: Dict[str, int]
    success_rates_by_type: Dict[str, float]
    avg_duration_by_type: Dict[str, float]
    network_centrality: Dict[str, float]


class AnalyticsEngine:
    """Main analytics engine for the agent ecosystem"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self.logger = logging.getLogger(__name__)
        self.monitoring_system = get_monitoring_system()
    
    async def get_agent_analytics(self, 
                                 agent_id: str, 
                                 time_range: TimeRange = TimeRange.DAY) -> AgentAnalytics:
        """Get comprehensive analytics for a specific agent"""
        try:
            start_time = self._get_start_time(time_range)
            
            # Get agent info
            agent = await self.db_manager.get_agent(agent_id)
            if not agent:
                raise AnalyticsError(f"Agent {agent_id} not found")
            
            # Get message statistics
            messages_sent = await self._count_messages_by_sender(agent_id, start_time)
            messages_received = await self._count_messages_by_receiver(agent_id, start_time)
            
            # Get interaction statistics
            interactions = await self._count_interactions_by_agent(agent_id, start_time)
            
            # Calculate response time
            avg_response_time = await self._calculate_avg_response_time(agent_id, start_time)
            
            # Calculate success rate
            success_rate = await self._calculate_success_rate(agent_id, start_time)
            
            # Calculate active time
            active_time = await self._calculate_active_time(agent_id, start_time)
            
            # Get capabilities usage
            capabilities_used = await self._get_capabilities_usage(agent_id, start_time)
            
            # Get error count
            error_count = await self._count_errors(agent_id, start_time)
            
            return AgentAnalytics(
                agent_id=agent_id,
                agent_type=agent.agent_type,
                total_messages_sent=messages_sent,
                total_messages_received=messages_received,
                total_interactions=interactions,
                avg_response_time=avg_response_time,
                success_rate=success_rate,
                active_time=active_time,
                capabilities_used=capabilities_used,
                error_count=error_count,
                last_active=agent.last_seen
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get agent analytics for {agent_id}: {e}")
            raise AnalyticsError(f"Failed to get agent analytics: {e}")
    
    async def get_system_analytics(self, time_range: TimeRange = TimeRange.DAY) -> SystemAnalytics:
        """Get system-wide analytics"""
        try:
            start_time = self._get_start_time(time_range)
            
            # Get agent counts
            total_agents = await self._count_total_agents()
            active_agents = await self._count_active_agents(start_time)
            
            # Get message and interaction counts
            total_messages = await self._count_total_messages(start_time)
            total_interactions = await self._count_total_interactions(start_time)
            
            # Calculate system response time
            avg_response_time = await self._calculate_system_avg_response_time(start_time)
            
            # Get peak concurrent agents
            peak_concurrent = await self._get_peak_concurrent_agents(start_time)
            
            # Calculate message throughput
            throughput = await self._calculate_message_throughput(start_time)
            
            # Calculate error rate
            error_rate = await self._calculate_system_error_rate(start_time)
            
            # Calculate uptime percentage
            uptime = await self._calculate_uptime_percentage(start_time)
            
            return SystemAnalytics(
                total_agents=total_agents,
                active_agents=active_agents,
                total_messages=total_messages,
                total_interactions=total_interactions,
                avg_system_response_time=avg_response_time,
                peak_concurrent_agents=peak_concurrent,
                message_throughput=throughput,
                error_rate=error_rate,
                uptime_percentage=uptime
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system analytics: {e}")
            raise AnalyticsError(f"Failed to get system analytics: {e}")
    
    async def analyze_message_patterns(self, time_range: TimeRange = TimeRange.DAY) -> MessagePatternAnalysis:
        """Analyze message patterns and communication behavior"""
        try:
            start_time = self._get_start_time(time_range)
            
            # Get message type distribution
            message_types = await self._get_message_type_distribution(start_time)
            
            # Analyze peak hours
            peak_hours = await self._analyze_peak_hours(start_time)
            
            # Analyze conversation lengths
            conversation_stats = await self._analyze_conversation_lengths(start_time)
            
            # Analyze response times
            response_time_percentiles = await self._analyze_response_time_percentiles(start_time)
            
            # Analyze message sizes
            message_size_stats = await self._analyze_message_sizes(start_time)
            
            return MessagePatternAnalysis(
                most_common_message_types=message_types,
                peak_hours=peak_hours,
                conversation_length_stats=conversation_stats,
                response_time_percentiles=response_time_percentiles,
                message_size_stats=message_size_stats
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze message patterns: {e}")
            raise AnalyticsError(f"Failed to analyze message patterns: {e}")
    
    async def analyze_interactions(self, time_range: TimeRange = TimeRange.DAY) -> InteractionAnalysis:
        """Analyze agent interactions and collaboration patterns"""
        try:
            start_time = self._get_start_time(time_range)
            
            # Get most active agent pairs
            active_pairs = await self._get_most_active_pairs(start_time)
            
            # Get interaction type distribution
            interaction_types = await self._get_interaction_type_distribution(start_time)
            
            # Calculate success rates by interaction type
            success_rates = await self._calculate_success_rates_by_type(start_time)
            
            # Calculate average duration by interaction type
            avg_durations = await self._calculate_avg_durations_by_type(start_time)
            
            # Calculate network centrality
            centrality = await self._calculate_network_centrality(start_time)
            
            return InteractionAnalysis(
                most_active_pairs=active_pairs,
                interaction_types=interaction_types,
                success_rates_by_type=success_rates,
                avg_duration_by_type=avg_durations,
                network_centrality=centrality
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze interactions: {e}")
            raise AnalyticsError(f"Failed to analyze interactions: {e}")
    
    async def generate_performance_report(self, 
                                        agent_id: Optional[str] = None,
                                        time_range: TimeRange = TimeRange.DAY) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        try:
            report = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'time_range': time_range.value,
                'start_time': self._get_start_time(time_range).isoformat()
            }
            
            if agent_id:
                # Agent-specific report
                agent_analytics = await self.get_agent_analytics(agent_id, time_range)
                report['agent_analytics'] = agent_analytics.__dict__
                
                # Add agent-specific insights
                report['insights'] = await self._generate_agent_insights(agent_analytics)
            else:
                # System-wide report
                system_analytics = await self.get_system_analytics(time_range)
                message_patterns = await self.analyze_message_patterns(time_range)
                interaction_analysis = await self.analyze_interactions(time_range)
                
                report['system_analytics'] = system_analytics.__dict__
                report['message_patterns'] = message_patterns.__dict__
                report['interaction_analysis'] = interaction_analysis.__dict__
                
                # Add system insights
                report['insights'] = await self._generate_system_insights(
                    system_analytics, message_patterns, interaction_analysis
                )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            raise AnalyticsError(f"Failed to generate performance report: {e}")
    
    async def detect_anomalies(self, time_range: TimeRange = TimeRange.DAY) -> List[Dict[str, Any]]:
        """Detect anomalies in system behavior"""
        try:
            anomalies = []
            start_time = self._get_start_time(time_range)
            
            # Check for unusual message volume
            message_anomalies = await self._detect_message_volume_anomalies(start_time)
            anomalies.extend(message_anomalies)
            
            # Check for unusual response times
            response_time_anomalies = await self._detect_response_time_anomalies(start_time)
            anomalies.extend(response_time_anomalies)
            
            # Check for unusual error rates
            error_rate_anomalies = await self._detect_error_rate_anomalies(start_time)
            anomalies.extend(error_rate_anomalies)
            
            # Check for inactive agents
            inactive_agent_anomalies = await self._detect_inactive_agents(start_time)
            anomalies.extend(inactive_agent_anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect anomalies: {e}")
            raise AnalyticsError(f"Failed to detect anomalies: {e}")
    
    def _get_start_time(self, time_range: TimeRange) -> datetime:
        """Get start time based on time range"""
        now = datetime.now(timezone.utc)
        
        if time_range == TimeRange.HOUR:
            return now - timedelta(hours=1)
        elif time_range == TimeRange.DAY:
            return now - timedelta(days=1)
        elif time_range == TimeRange.WEEK:
            return now - timedelta(weeks=1)
        elif time_range == TimeRange.MONTH:
            return now - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return now - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=1)
    
    # Database query methods (implementation depends on database type)
    async def _count_messages_by_sender(self, agent_id: str, start_time: datetime) -> int:
        """Count messages sent by agent since start_time"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.messages.count_documents({
                'sender_id': agent_id,
                'timestamp': {'$gte': start_time}
            })
        else:
            # PostgreSQL implementation
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute(
                    "SELECT COUNT(*) FROM messages WHERE sender_id = :agent_id AND timestamp >= :start_time",
                    {'agent_id': agent_id, 'start_time': start_time}
                )
                return result.scalar() or 0
    
    async def _count_messages_by_receiver(self, agent_id: str, start_time: datetime) -> int:
        """Count messages received by agent since start_time"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.messages.count_documents({
                'receiver_id': agent_id,
                'timestamp': {'$gte': start_time}
            })
        else:
            # PostgreSQL implementation
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute(
                    "SELECT COUNT(*) FROM messages WHERE receiver_id = :agent_id AND timestamp >= :start_time",
                    {'agent_id': agent_id, 'start_time': start_time}
                )
                return result.scalar() or 0
    
    async def _count_interactions_by_agent(self, agent_id: str, start_time: datetime) -> int:
        """Count interactions involving agent since start_time"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.interactions.count_documents({
                '$or': [
                    {'initiator_id': agent_id},
                    {'target_id': agent_id}
                ],
                'started_at': {'$gte': start_time}
            })
        else:
            # PostgreSQL implementation
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute(
                    """SELECT COUNT(*) FROM interactions 
                       WHERE (initiator_id = :agent_id OR target_id = :agent_id) 
                       AND started_at >= :start_time""",
                    {'agent_id': agent_id, 'start_time': start_time}
                )
                return result.scalar() or 0
    
    async def _calculate_avg_response_time(self, agent_id: str, start_time: datetime) -> float:
        """Calculate average response time for agent"""
        # This would require analyzing request-response pairs
        # Implementation depends on how response times are tracked
        return 0.0  # Placeholder
    
    async def _calculate_success_rate(self, agent_id: str, start_time: datetime) -> float:
        """Calculate success rate for agent interactions"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            total = await self.db_manager._mongo_db.interactions.count_documents({
                'initiator_id': agent_id,
                'started_at': {'$gte': start_time}
            })
            successful = await self.db_manager._mongo_db.interactions.count_documents({
                'initiator_id': agent_id,
                'started_at': {'$gte': start_time},
                'status': 'completed'
            })
        else:
            # PostgreSQL implementation
            async with self.db_manager._pg_session_factory() as session:
                total_result = await session.execute(
                    "SELECT COUNT(*) FROM interactions WHERE initiator_id = :agent_id AND started_at >= :start_time",
                    {'agent_id': agent_id, 'start_time': start_time}
                )
                total = total_result.scalar() or 0
                
                success_result = await session.execute(
                    """SELECT COUNT(*) FROM interactions 
                       WHERE initiator_id = :agent_id AND started_at >= :start_time AND status = 'completed'""",
                    {'agent_id': agent_id, 'start_time': start_time}
                )
                successful = success_result.scalar() or 0
        
        return (successful / total * 100) if total > 0 else 0.0
    
    async def _calculate_active_time(self, agent_id: str, start_time: datetime) -> float:
        """Calculate active time for agent in hours"""
        # This would require analyzing agent activity patterns
        # Implementation depends on how activity is tracked
        return 0.0  # Placeholder
    
    async def _get_capabilities_usage(self, agent_id: str, start_time: datetime) -> Dict[str, int]:
        """Get capabilities usage statistics for agent"""
        # This would require analyzing capability usage in messages/interactions
        return {}  # Placeholder
    
    async def _count_errors(self, agent_id: str, start_time: datetime) -> int:
        """Count errors for agent since start_time"""
        # This would require analyzing error logs or failed interactions
        return 0  # Placeholder
    
    async def _count_total_agents(self) -> int:
        """Count total number of agents"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.agents.count_documents({})
        else:
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute("SELECT COUNT(*) FROM agents")
                return result.scalar() or 0
    
    async def _count_active_agents(self, start_time: datetime) -> int:
        """Count active agents since start_time"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.agents.count_documents({
                'last_seen': {'$gte': start_time}
            })
        else:
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute(
                    "SELECT COUNT(*) FROM agents WHERE last_seen >= :start_time",
                    {'start_time': start_time}
                )
                return result.scalar() or 0
    
    async def _count_total_messages(self, start_time: datetime) -> int:
        """Count total messages since start_time"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.messages.count_documents({
                'timestamp': {'$gte': start_time}
            })
        else:
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute(
                    "SELECT COUNT(*) FROM messages WHERE timestamp >= :start_time",
                    {'start_time': start_time}
                )
                return result.scalar() or 0
    
    async def _count_total_interactions(self, start_time: datetime) -> int:
        """Count total interactions since start_time"""
        if self.db_manager.config.db_type == DatabaseType.MONGODB:
            return await self.db_manager._mongo_db.interactions.count_documents({
                'started_at': {'$gte': start_time}
            })
        else:
            async with self.db_manager._pg_session_factory() as session:
                result = await session.execute(
                    "SELECT COUNT(*) FROM interactions WHERE started_at >= :start_time",
                    {'start_time': start_time}
                )
                return result.scalar() or 0
    
    # Additional helper methods would be implemented here...
    # (Placeholder implementations for brevity)
    
    async def _calculate_system_avg_response_time(self, start_time: datetime) -> float:
        return 0.0
    
    async def _get_peak_concurrent_agents(self, start_time: datetime) -> int:
        return 0
    
    async def _calculate_message_throughput(self, start_time: datetime) -> float:
        return 0.0
    
    async def _calculate_system_error_rate(self, start_time: datetime) -> float:
        return 0.0
    
    async def _calculate_uptime_percentage(self, start_time: datetime) -> float:
        return 100.0
    
    async def _get_message_type_distribution(self, start_time: datetime) -> List[Tuple[str, int]]:
        return []
    
    async def _analyze_peak_hours(self, start_time: datetime) -> List[int]:
        return []
    
    async def _analyze_conversation_lengths(self, start_time: datetime) -> Dict[str, float]:
        return {}
    
    async def _analyze_response_time_percentiles(self, start_time: datetime) -> Dict[str, float]:
        return {}
    
    async def _analyze_message_sizes(self, start_time: datetime) -> Dict[str, float]:
        return {}
    
    async def _get_most_active_pairs(self, start_time: datetime) -> List[Tuple[str, str, int]]:
        return []
    
    async def _get_interaction_type_distribution(self, start_time: datetime) -> Dict[str, int]:
        return {}
    
    async def _calculate_success_rates_by_type(self, start_time: datetime) -> Dict[str, float]:
        return {}
    
    async def _calculate_avg_durations_by_type(self, start_time: datetime) -> Dict[str, float]:
        return {}
    
    async def _calculate_network_centrality(self, start_time: datetime) -> Dict[str, float]:
        return {}
    
    async def _generate_agent_insights(self, analytics: AgentAnalytics) -> List[str]:
        """Generate insights for agent performance"""
        insights = []
        
        if analytics.success_rate < 80:
            insights.append(f"Low success rate ({analytics.success_rate:.1f}%) - consider investigating failures")
        
        if analytics.avg_response_time > 5.0:
            insights.append(f"High response time ({analytics.avg_response_time:.2f}s) - performance optimization needed")
        
        if analytics.error_count > 10:
            insights.append(f"High error count ({analytics.error_count}) - debugging required")
        
        return insights
    
    async def _generate_system_insights(self, 
                                      system: SystemAnalytics,
                                      patterns: MessagePatternAnalysis,
                                      interactions: InteractionAnalysis) -> List[str]:
        """Generate insights for system performance"""
        insights = []
        
        if system.error_rate > 5.0:
            insights.append(f"High system error rate ({system.error_rate:.1f}%) - investigate system issues")
        
        if system.uptime_percentage < 99.0:
            insights.append(f"Low uptime ({system.uptime_percentage:.1f}%) - improve system reliability")
        
        if system.message_throughput < 10.0:
            insights.append(f"Low message throughput ({system.message_throughput:.1f} msg/min) - check system capacity")
        
        return insights
    
    async def _detect_message_volume_anomalies(self, start_time: datetime) -> List[Dict[str, Any]]:
        return []
    
    async def _detect_response_time_anomalies(self, start_time: datetime) -> List[Dict[str, Any]]:
        return []
    
    async def _detect_error_rate_anomalies(self, start_time: datetime) -> List[Dict[str, Any]]:
        return []
    
    async def _detect_inactive_agents(self, start_time: datetime) -> List[Dict[str, Any]]:
        return []


def create_analytics_engine(database_manager: DatabaseManager) -> AnalyticsEngine:
    """Factory function to create analytics engine"""
    return AnalyticsEngine(database_manager) 