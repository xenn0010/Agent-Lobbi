"""
Agent Lobby Enhanced Metrics System
Advanced metrics, analytics, and monitoring for A2A+ platform
"""

import time
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import logging
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    PERFORMANCE = "performance"
    USER_EXPERIENCE = "user_experience"
    COLLABORATION = "collaboration"
    BUSINESS = "business"
    SECURITY = "security"
    RELIABILITY = "reliability"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricPoint:
    """Individual metric measurement"""
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    
@dataclass
class UserExperienceMetrics:
    """User interaction and experience metrics"""
    session_duration: float = 0.0
    interaction_frequency: float = 0.0
    satisfaction_score: float = 0.0
    task_completion_rate: float = 0.0
    user_retention_rate: float = 0.0
    bounce_rate: float = 0.0
    
@dataclass
class CollaborationMetrics:
    """Agent collaboration effectiveness"""
    cross_agent_calls: int = 0
    collaboration_success_rate: float = 0.0
    network_efficiency: float = 0.0
    collective_intelligence_score: float = 0.0
    task_delegation_success: float = 0.0
    
@dataclass
class BusinessMetrics:
    """Business intelligence metrics"""
    cost_per_interaction: float = 0.0
    revenue_per_user: float = 0.0
    roi: float = 0.0
    conversion_rate: float = 0.0
    customer_lifetime_value: float = 0.0

class MetricsCollector:
    """High-performance metrics collection engine"""
    
    def __init__(self, buffer_size: int = 10000):
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.real_time_metrics = defaultdict(deque)
        self.aggregated_metrics = defaultdict(dict)
        self.subscribers = []
        self.collection_thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def start_collection(self):
        """Start real-time metrics collection"""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("📊 Metrics collection started")
        
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("📊 Metrics collection stopped")
        
    def record_metric(self, metric_name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric point"""
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics_buffer.append(metric_point)
            self.real_time_metrics[metric_name].append(metric_point)
            
    def _collection_loop(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                self._process_metrics()
                time.sleep(1)  # Collect every second
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                
    def _process_metrics(self):
        """Process and aggregate metrics"""
        with self.lock:
            # Aggregate metrics for the last minute
            now = datetime.now()
            for metric_name, points in self.real_time_metrics.items():
                # Remove old points (older than 1 minute)
                while points and (now - points[0].timestamp).seconds > 60:
                    points.popleft()
                    
                # Calculate aggregations
                if points:
                    values = [p.value for p in points]
                    self.aggregated_metrics[metric_name] = {
                        'avg': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values),
                        'last': values[-1] if values else 0
                    }
                    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics snapshot"""
        return dict(self.aggregated_metrics)
        
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to metrics updates"""
        self.subscribers.append(callback)

class A2AMetricsTracker:
    """A2A-specific metrics tracking"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
    def track_task_start(self, task_id: str, agent_id: str, task_type: str):
        """Track A2A task initiation"""
        self.active_tasks[task_id] = {
            'start_time': time.time(),
            'agent_id': agent_id,
            'task_type': task_type,
            'messages_sent': 0,
            'messages_received': 0
        }
        
        self.collector.record_metric(
            'a2a_tasks_started',
            1,
            tags={'agent_id': agent_id, 'task_type': task_type}
        )
        
    def track_task_completion(self, task_id: str, status: str, 
                            result_size: int = 0):
        """Track A2A task completion"""
        if task_id not in self.active_tasks:
            return
            
        task_info = self.active_tasks.pop(task_id)
        completion_time = time.time()
        duration = completion_time - task_info['start_time']
        
        # Record completion metrics
        self.collector.record_metric(
            'a2a_task_duration',
            duration * 1000,  # Convert to milliseconds
            tags={
                'agent_id': task_info['agent_id'],
                'task_type': task_info['task_type'],
                'status': status
            }
        )
        
        self.collector.record_metric(
            'a2a_task_success_rate',
            1 if status == 'completed' else 0,
            tags={'agent_id': task_info['agent_id']}
        )
        
        # Store completed task info
        self.completed_tasks.append({
            'task_id': task_id,
            'duration': duration,
            'status': status,
            'result_size': result_size,
            'messages_sent': task_info['messages_sent'],
            'messages_received': task_info['messages_received']
        })
        
    def track_message_exchange(self, task_id: str, direction: str, 
                             message_size: int):
        """Track A2A message exchange"""
        if task_id in self.active_tasks:
            if direction == 'sent':
                self.active_tasks[task_id]['messages_sent'] += 1
            else:
                self.active_tasks[task_id]['messages_received'] += 1
                
        self.collector.record_metric(
            f'a2a_message_{direction}',
            1,
            metadata={'size': message_size}
        )

class UserExperienceTracker:
    """Advanced user experience tracking"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.user_sessions = {}
        self.user_interactions = defaultdict(list)
        
    def track_user_session_start(self, user_id: str, session_id: str):
        """Track user session start"""
        self.user_sessions[session_id] = {
            'user_id': user_id,
            'start_time': time.time(),
            'interactions': 0,
            'tasks_completed': 0,
            'errors': 0
        }
        
        self.collector.record_metric(
            'user_sessions_started',
            1,
            tags={'user_id': user_id}
        )
        
    def track_user_interaction(self, session_id: str, interaction_type: str,
                             response_time: float):
        """Track user interaction"""
        if session_id in self.user_sessions:
            self.user_sessions[session_id]['interactions'] += 1
            
        self.collector.record_metric(
            'user_interaction_response_time',
            response_time,
            tags={'type': interaction_type}
        )
        
        self.collector.record_metric(
            'user_interactions_total',
            1,
            tags={'type': interaction_type}
        )
        
    def calculate_satisfaction_score(self, session_id: str,
                                   task_success_rate: float,
                                   avg_response_time: float) -> float:
        """Calculate user satisfaction score"""
        # Advanced satisfaction scoring algorithm
        time_score = max(0, 1 - (avg_response_time / 5000))  # 5s baseline
        success_score = task_success_rate
        
        satisfaction = (time_score * 0.4) + (success_score * 0.6)
        
        self.collector.record_metric(
            'user_satisfaction_score',
            satisfaction,
            tags={'session_id': session_id}
        )
        
        return satisfaction

class BusinessIntelligenceTracker:
    """Advanced business intelligence tracking"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.cost_tracking = defaultdict(float)
        self.revenue_tracking = defaultdict(float)
        
    def track_cost_per_interaction(self, interaction_type: str, cost: float):
        """Track cost per interaction"""
        self.cost_tracking[interaction_type] += cost
        
        self.collector.record_metric(
            'cost_per_interaction',
            cost,
            tags={'type': interaction_type}
        )
        
    def track_revenue_generation(self, user_id: str, revenue: float):
        """Track revenue generation"""
        self.revenue_tracking[user_id] += revenue
        
        self.collector.record_metric(
            'revenue_generated',
            revenue,
            tags={'user_id': user_id}
        )
        
    def calculate_roi(self, time_period: str = '24h') -> float:
        """Calculate ROI for specified time period"""
        total_revenue = sum(self.revenue_tracking.values())
        total_cost = sum(self.cost_tracking.values())
        
        roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
        
        self.collector.record_metric(
            'roi',
            roi,
            tags={'period': time_period}
        )
        
        return roi

class AlertManager:
    """Intelligent alerting system"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.alert_rules = []
        self.alert_history = deque(maxlen=1000)
        
    def add_alert_rule(self, metric_name: str, threshold: float,
                      level: AlertLevel, condition: str = 'gt'):
        """Add alert rule"""
        self.alert_rules.append({
            'metric': metric_name,
            'threshold': threshold,
            'level': level,
            'condition': condition
        })
        
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check for alert conditions"""
        alerts = []
        
        for rule in self.alert_rules:
            metric_data = metrics.get(rule['metric'])
            if not metric_data:
                continue
                
            current_value = metric_data.get('last', 0)
            
            should_alert = False
            if rule['condition'] == 'gt' and current_value > rule['threshold']:
                should_alert = True
            elif rule['condition'] == 'lt' and current_value < rule['threshold']:
                should_alert = True
                
            if should_alert:
                alert = {
                    'timestamp': datetime.now(),
                    'metric': rule['metric'],
                    'value': current_value,
                    'threshold': rule['threshold'],
                    'level': rule['level'],
                    'message': f"{rule['metric']} is {current_value}, threshold: {rule['threshold']}"
                }
                alerts.append(alert)
                self.alert_history.append(alert)
                
        return alerts

class EnhancedMetricsSystem:
    """Complete enhanced metrics system for Agent Lobby"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.a2a_tracker = A2AMetricsTracker(self.collector)
        self.ux_tracker = UserExperienceTracker(self.collector)
        self.bi_tracker = BusinessIntelligenceTracker(self.collector)
        self.alert_manager = AlertManager(self.collector)
        self.running = False
        
        # Set up default alerts
        self._setup_default_alerts()
        
    def _setup_default_alerts(self):
        """Set up default alerting rules"""
        # Performance alerts
        self.alert_manager.add_alert_rule(
            'a2a_task_duration', 5000, AlertLevel.WARNING, 'gt'
        )
        self.alert_manager.add_alert_rule(
            'a2a_task_success_rate', 0.9, AlertLevel.ERROR, 'lt'
        )
        
        # User experience alerts
        self.alert_manager.add_alert_rule(
            'user_satisfaction_score', 0.7, AlertLevel.WARNING, 'lt'
        )
        
        # Business alerts
        self.alert_manager.add_alert_rule(
            'cost_per_interaction', 1.0, AlertLevel.INFO, 'gt'
        )
        
    def start(self):
        """Start the enhanced metrics system"""
        self.collector.start_collection()
        self.running = True
        logger.info("🚀 Enhanced Metrics System started")
        
    def stop(self):
        """Stop the enhanced metrics system"""
        self.collector.stop_collection()
        self.running = False
        logger.info("🛑 Enhanced Metrics System stopped")
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        metrics = self.collector.get_real_time_metrics()
        alerts = self.alert_manager.check_alerts(metrics)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'alerts': alerts,
            'system_health': self._calculate_system_health(metrics),
            'performance_summary': self._get_performance_summary(metrics),
            'user_experience_summary': self._get_ux_summary(metrics),
            'business_summary': self._get_business_summary(metrics)
        }
        
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall system health"""
        # Advanced health calculation logic
        health_score = 100
        
        # Check critical metrics
        task_success_rate = metrics.get('a2a_task_success_rate', {}).get('avg', 1.0)
        if task_success_rate < 0.9:
            health_score -= 20
            
        avg_response_time = metrics.get('a2a_task_duration', {}).get('avg', 0)
        if avg_response_time > 3000:  # 3 seconds
            health_score -= 15
            
        if health_score >= 90:
            return "Excellent"
        elif health_score >= 70:
            return "Good"
        elif health_score >= 50:
            return "Fair"
        else:
            return "Poor"
            
    def _get_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'avg_response_time': metrics.get('a2a_task_duration', {}).get('avg', 0),
            'success_rate': metrics.get('a2a_task_success_rate', {}).get('avg', 0),
            'throughput': metrics.get('a2a_tasks_started', {}).get('count', 0),
            'active_tasks': len(self.a2a_tracker.active_tasks)
        }
        
    def _get_ux_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get user experience summary"""
        return {
            'satisfaction_score': metrics.get('user_satisfaction_score', {}).get('avg', 0),
            'avg_session_duration': metrics.get('user_session_duration', {}).get('avg', 0),
            'interaction_frequency': metrics.get('user_interactions_total', {}).get('count', 0)
        }
        
    def _get_business_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get business intelligence summary"""
        return {
            'cost_per_interaction': metrics.get('cost_per_interaction', {}).get('avg', 0),
            'revenue_generated': metrics.get('revenue_generated', {}).get('last', 0),
            'roi': metrics.get('roi', {}).get('last', 0)
        }

# Export main class
__all__ = ['EnhancedMetricsSystem', 'MetricsCollector', 'A2AMetricsTracker'] 