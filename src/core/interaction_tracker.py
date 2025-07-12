"""
Advanced Interaction Tracking System for Agent Lobbi
===================================================
Comprehensive tracking of all agent interactions, task executions, and system analytics
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import logging

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of interactions to track"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    AGENT_REGISTRATION = "agent_registration"
    AGENT_DISCONNECTION = "agent_disconnection"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_COMPLETED = "workflow_completed"
    COLLABORATION_START = "collaboration_start"
    COLLABORATION_END = "collaboration_end"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"

class ExecutionStage(Enum):
    """Task execution stages"""
    RECEIVED = "received"
    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class InteractionRecord:
    """Single interaction record"""
    id: str
    interaction_type: InteractionType
    source_agent_id: str
    target_agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    task_id: Optional[str] = None
    interaction_data: Dict[str, Any] = None
    success: bool = True
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.interaction_data is None:
            self.interaction_data = {}

@dataclass
class TaskExecutionRecord:
    """Task execution tracking record"""
    id: str
    task_id: str
    workflow_id: str
    agent_id: str
    execution_stage: ExecutionStage
    stage_data: Dict[str, Any] = None
    execution_time_ms: Optional[int] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    error_details: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.stage_data is None:
            self.stage_data = {}

@dataclass
class SystemMetric:
    """System performance metric"""
    id: str
    metric_name: str
    metric_value: float
    metric_metadata: Dict[str, Any] = None
    aggregation_period: str = "real_time"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metric_metadata is None:
            self.metric_metadata = {}

class InteractionTracker:
    """Advanced interaction tracking system"""
    
    def __init__(self, database_manager=None, enable_real_time_analytics=True):
        self.database_manager = database_manager
        self.enable_real_time_analytics = enable_real_time_analytics
        
        # In-memory buffers for high-performance logging
        self.interaction_buffer: List[InteractionRecord] = []
        self.execution_buffer: List[TaskExecutionRecord] = []
        self.metrics_buffer: List[SystemMetric] = []
        
        # Real-time analytics
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.system_health: Dict[str, Any] = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "failed_interactions": 0,
            "average_response_time_ms": 0.0,
            "active_agents": 0,
            "active_workflows": 0
        }
        
        # Buffer management
        self.buffer_size = 1000
        self.flush_interval = 30  # seconds
        self.last_flush = time.time()
        
        # Start background tasks
        self._background_tasks = []
        if self.enable_real_time_analytics:
            self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background analytics and buffer management tasks"""
        try:
            loop = asyncio.get_event_loop()
            self._background_tasks.append(
                loop.create_task(self._periodic_flush())
            )
            self._background_tasks.append(
                loop.create_task(self._real_time_analytics())
            )
        except RuntimeError:
            # No event loop, will start manually when needed
            pass
    
    async def _periodic_flush(self):
        """Periodically flush buffers to database"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_buffers()
            except Exception as e:
                logger.error("Error in periodic flush", error=str(e))
    
    async def _real_time_analytics(self):
        """Update real-time analytics"""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                await self._update_system_health()
            except Exception as e:
                logger.error("Error in real-time analytics", error=str(e))
    
    async def track_interaction(self, 
                              interaction_type: InteractionType,
                              source_agent_id: str,
                              target_agent_id: Optional[str] = None,
                              workflow_id: Optional[str] = None,
                              task_id: Optional[str] = None,
                              interaction_data: Optional[Dict[str, Any]] = None,
                              success: bool = True,
                              error_message: Optional[str] = None,
                              response_time_ms: Optional[int] = None):
        """Track a single interaction"""
        
        record = InteractionRecord(
            id=str(uuid.uuid4()),
            interaction_type=interaction_type,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            workflow_id=workflow_id,
            task_id=task_id,
            interaction_data=interaction_data or {},
            success=success,
            error_message=error_message,
            response_time_ms=response_time_ms
        )
        
        # Add to buffer
        self.interaction_buffer.append(record)
        
        # Update real-time metrics
        self.system_health["total_interactions"] += 1
        if success:
            self.system_health["successful_interactions"] += 1
        else:
            self.system_health["failed_interactions"] += 1
        
        # Update agent performance
        if source_agent_id not in self.agent_performance:
            self.agent_performance[source_agent_id] = {
                "total_interactions": 0,
                "successful_interactions": 0,
                "failed_interactions": 0,
                "average_response_time_ms": 0.0,
                "last_seen": datetime.now(timezone.utc)
            }
        
        agent_stats = self.agent_performance[source_agent_id]
        agent_stats["total_interactions"] += 1
        agent_stats["last_seen"] = datetime.now(timezone.utc)
        
        if success:
            agent_stats["successful_interactions"] += 1
        else:
            agent_stats["failed_interactions"] += 1
            
        if response_time_ms:
            # Update running average
            current_avg = agent_stats["average_response_time_ms"]
            total = agent_stats["total_interactions"]
            agent_stats["average_response_time_ms"] = (
                (current_avg * (total - 1) + response_time_ms) / total
            )
        
        # Auto-flush if buffer is full
        if len(self.interaction_buffer) >= self.buffer_size:
            await self.flush_buffers()
        
        logger.debug("Interaction tracked", 
                    type=interaction_type.value,
                    source=source_agent_id,
                    target=target_agent_id,
                    success=success)
    
    async def track_task_execution(self,
                                 task_id: str,
                                 workflow_id: str,
                                 agent_id: str,
                                 execution_stage: ExecutionStage,
                                 stage_data: Optional[Dict[str, Any]] = None,
                                 execution_time_ms: Optional[int] = None,
                                 memory_usage_mb: Optional[float] = None,
                                 cpu_usage_percent: Optional[float] = None,
                                 error_details: Optional[str] = None):
        """Track task execution stage"""
        
        record = TaskExecutionRecord(
            id=str(uuid.uuid4()),
            task_id=task_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            execution_stage=execution_stage,
            stage_data=stage_data or {},
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            error_details=error_details
        )
        
        # Add to buffer
        self.execution_buffer.append(record)
        
        # Update active tasks tracking
        if execution_stage == ExecutionStage.STARTED:
            self.active_tasks[task_id] = {
                "workflow_id": workflow_id,
                "agent_id": agent_id,
                "started_at": datetime.now(timezone.utc),
                "last_update": datetime.now(timezone.utc)
            }
        elif execution_stage in [ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.TIMEOUT]:
            self.active_tasks.pop(task_id, None)
        else:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["last_update"] = datetime.now(timezone.utc)
        
        # Auto-flush if buffer is full
        if len(self.execution_buffer) >= self.buffer_size:
            await self.flush_buffers()
        
        logger.debug("Task execution tracked",
                    task_id=task_id,
                    agent_id=agent_id,
                    stage=execution_stage.value,
                    execution_time=execution_time_ms)
    
    async def track_system_metric(self,
                                metric_name: str,
                                metric_value: float,
                                metric_metadata: Optional[Dict[str, Any]] = None,
                                aggregation_period: str = "real_time"):
        """Track system performance metric"""
        
        record = SystemMetric(
            id=str(uuid.uuid4()),
            metric_name=metric_name,
            metric_value=metric_value,
            metric_metadata=metric_metadata or {},
            aggregation_period=aggregation_period
        )
        
        # Add to buffer
        self.metrics_buffer.append(record)
        
        # Auto-flush if buffer is full
        if len(self.metrics_buffer) >= self.buffer_size:
            await self.flush_buffers()
        
        logger.debug("System metric tracked",
                    metric=metric_name,
                    value=metric_value,
                    period=aggregation_period)
    
    async def flush_buffers(self):
        """Flush all buffers to database"""
        if not self.database_manager:
            logger.warning("No database manager configured, cannot flush buffers")
            return
        
        try:
            # Flush interactions
            if self.interaction_buffer:
                await self._flush_interactions()
            
            # Flush task executions
            if self.execution_buffer:
                await self._flush_task_executions()
            
            # Flush metrics
            if self.metrics_buffer:
                await self._flush_metrics()
            
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error("Error flushing buffers", error=str(e))
    
    async def _flush_interactions(self):
        """Flush interaction buffer to database"""
        interactions_to_flush = self.interaction_buffer.copy()
        self.interaction_buffer.clear()
        
        for interaction in interactions_to_flush:
            try:
                # Convert to database format
                data = {
                    "id": interaction.id,
                    "interaction_type": interaction.interaction_type.value,
                    "source_agent_id": interaction.source_agent_id,
                    "target_agent_id": interaction.target_agent_id,
                    "workflow_id": interaction.workflow_id,
                    "task_id": interaction.task_id,
                    "interaction_data": interaction.interaction_data,
                    "success": interaction.success,
                    "error_message": interaction.error_message,
                    "response_time_ms": interaction.response_time_ms,
                    "timestamp": interaction.timestamp,
                    "created_at": interaction.timestamp
                }
                
                # Save to database (implement based on your database schema)
                if hasattr(self.database_manager, 'save_interaction'):
                    await self.database_manager.save_interaction(data)
                
            except Exception as e:
                logger.error("Error saving interaction", 
                           interaction_id=interaction.id, 
                           error=str(e))
    
    async def _flush_task_executions(self):
        """Flush task execution buffer to database"""
        executions_to_flush = self.execution_buffer.copy()
        self.execution_buffer.clear()
        
        for execution in executions_to_flush:
            try:
                # Convert to database format
                data = {
                    "id": execution.id,
                    "task_id": execution.task_id,
                    "workflow_id": execution.workflow_id,
                    "agent_id": execution.agent_id,
                    "execution_stage": execution.execution_stage.value,
                    "stage_data": execution.stage_data,
                    "execution_time_ms": execution.execution_time_ms,
                    "memory_usage_mb": execution.memory_usage_mb,
                    "cpu_usage_percent": execution.cpu_usage_percent,
                    "error_details": execution.error_details,
                    "timestamp": execution.timestamp
                }
                
                # Save to database
                if hasattr(self.database_manager, 'save_task_execution'):
                    await self.database_manager.save_task_execution(data)
                
            except Exception as e:
                logger.error("Error saving task execution",
                           execution_id=execution.id,
                           error=str(e))
    
    async def _flush_metrics(self):
        """Flush metrics buffer to database"""
        metrics_to_flush = self.metrics_buffer.copy()
        self.metrics_buffer.clear()
        
        for metric in metrics_to_flush:
            try:
                # Convert to database format
                data = {
                    "id": metric.id,
                    "metric_name": metric.metric_name,
                    "metric_value": metric.metric_value,
                    "metric_metadata": metric.metric_metadata,
                    "aggregation_period": metric.aggregation_period,
                    "timestamp": metric.timestamp
                }
                
                # Save to database
                if hasattr(self.database_manager, 'save_system_metric'):
                    await self.database_manager.save_system_metric(data)
                
            except Exception as e:
                logger.error("Error saving system metric",
                           metric_id=metric.id,
                           error=str(e))
    
    async def _update_system_health(self):
        """Update system health metrics"""
        try:
            # Update active counts
            self.system_health["active_agents"] = len(self.agent_performance)
            self.system_health["active_workflows"] = len(set(
                task_info["workflow_id"] for task_info in self.active_tasks.values()
            ))
            
            # Calculate average response time
            total_response_time = 0
            total_with_response_time = 0
            
            for agent_stats in self.agent_performance.values():
                if agent_stats["average_response_time_ms"] > 0:
                    total_response_time += agent_stats["average_response_time_ms"]
                    total_with_response_time += 1
            
            if total_with_response_time > 0:
                self.system_health["average_response_time_ms"] = (
                    total_response_time / total_with_response_time
                )
            
            # Track system health as metric
            await self.track_system_metric(
                "system_health_total_interactions",
                self.system_health["total_interactions"]
            )
            await self.track_system_metric(
                "system_health_success_rate",
                (self.system_health["successful_interactions"] / 
                 max(1, self.system_health["total_interactions"]) * 100)
            )
            await self.track_system_metric(
                "system_health_active_agents",
                self.system_health["active_agents"]
            )
            
        except Exception as e:
            logger.error("Error updating system health", error=str(e))
    
    def get_real_time_analytics(self) -> Dict[str, Any]:
        """Get current real-time analytics"""
        return {
            "system_health": self.system_health.copy(),
            "agent_performance": self.agent_performance.copy(),
            "active_tasks": len(self.active_tasks),
            "active_task_details": self.active_tasks.copy(),
            "buffer_status": {
                "interactions": len(self.interaction_buffer),
                "executions": len(self.execution_buffer),
                "metrics": len(self.metrics_buffer),
                "last_flush": self.last_flush
            }
        }
    
    def get_agent_analytics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get analytics for specific agent"""
        if agent_id not in self.agent_performance:
            return None
        
        stats = self.agent_performance[agent_id].copy()
        
        # Add current tasks
        current_tasks = [
            task_id for task_id, task_info in self.active_tasks.items()
            if task_info["agent_id"] == agent_id
        ]
        stats["current_tasks"] = current_tasks
        stats["current_task_count"] = len(current_tasks)
        
        # Calculate success rate
        if stats["total_interactions"] > 0:
            stats["success_rate"] = (
                stats["successful_interactions"] / stats["total_interactions"] * 100
            )
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    async def close(self):
        """Clean shutdown"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Final flush
        await self.flush_buffers()
        
        logger.info("Interaction tracker closed")

# Global instance
_global_tracker: Optional[InteractionTracker] = None

def get_interaction_tracker() -> Optional[InteractionTracker]:
    """Get global interaction tracker instance"""
    return _global_tracker

def initialize_interaction_tracker(database_manager=None, enable_real_time_analytics=True) -> InteractionTracker:
    """Initialize global interaction tracker"""
    global _global_tracker
    _global_tracker = InteractionTracker(database_manager, enable_real_time_analytics)
    return _global_tracker

# Convenience functions for easy tracking
async def track_task_assignment(agent_id: str, task_id: str, workflow_id: str, **kwargs):
    """Track task assignment"""
    if _global_tracker:
        await _global_tracker.track_interaction(
            InteractionType.TASK_ASSIGNMENT,
            source_agent_id="system",
            target_agent_id=agent_id,
            task_id=task_id,
            workflow_id=workflow_id,
            **kwargs
        )

async def track_task_completion(agent_id: str, task_id: str, workflow_id: str, execution_time_ms: int = None, **kwargs):
    """Track task completion"""
    if _global_tracker:
        await _global_tracker.track_interaction(
            InteractionType.TASK_COMPLETION,
            source_agent_id=agent_id,
            task_id=task_id,
            workflow_id=workflow_id,
            response_time_ms=execution_time_ms,
            **kwargs
        )
        
        await _global_tracker.track_task_execution(
            task_id=task_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            execution_stage=ExecutionStage.COMPLETED,
            execution_time_ms=execution_time_ms
        )

async def track_task_failure(agent_id: str, task_id: str, workflow_id: str, error_message: str, **kwargs):
    """Track task failure"""
    if _global_tracker:
        await _global_tracker.track_interaction(
            InteractionType.TASK_FAILURE,
            source_agent_id=agent_id,
            task_id=task_id,
            workflow_id=workflow_id,
            success=False,
            error_message=error_message,
            **kwargs
        )
        
        await _global_tracker.track_task_execution(
            task_id=task_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            execution_stage=ExecutionStage.FAILED,
            error_details=error_message
        ) 