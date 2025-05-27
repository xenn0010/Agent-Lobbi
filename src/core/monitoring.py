"""
Monitoring and observability system for the agent ecosystem.
Provides metrics collection, structured logging, health checks, and performance tracking.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import defaultdict, deque
import statistics

# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry, REGISTRY
from prometheus_client.core import CollectorRegistry

# Structured logging
import structlog

# OpenTelemetry (optional, for distributed tracing)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    INFO = "info"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    name: str
    check_function: Callable
    timeout: float = 5.0
    critical: bool = True
    last_check: Optional[datetime] = None
    last_status: Optional[HealthStatus] = None
    last_error: Optional[str] = None


@dataclass
class MetricDefinition:
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


class AgentMetrics:
    """Metrics collection for individual agents"""
    
    def __init__(self, agent_id: str, registry: Optional[CollectorRegistry] = None):
        self.agent_id = agent_id
        self.registry = registry or REGISTRY
        
        # Define metrics
        self.messages_sent = Counter(
            'agent_messages_sent_total',
            'Total messages sent by agent',
            ['agent_id', 'message_type', 'receiver_id'],
            registry=self.registry
        )
        
        self.messages_received = Counter(
            'agent_messages_received_total',
            'Total messages received by agent',
            ['agent_id', 'message_type', 'sender_id'],
            registry=self.registry
        )
        
        self.message_processing_time = Histogram(
            'agent_message_processing_seconds',
            'Time spent processing messages',
            ['agent_id', 'message_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.interactions_initiated = Counter(
            'agent_interactions_initiated_total',
            'Total interactions initiated by agent',
            ['agent_id', 'interaction_type', 'target_id'],
            registry=self.registry
        )
        
        self.interactions_completed = Counter(
            'agent_interactions_completed_total',
            'Total interactions completed by agent',
            ['agent_id', 'interaction_type', 'status'],
            registry=self.registry
        )
        
        self.interaction_duration = Histogram(
            'agent_interaction_duration_seconds',
            'Duration of agent interactions',
            ['agent_id', 'interaction_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        self.connection_status = Gauge(
            'agent_connection_status',
            'Agent connection status (1=connected, 0=disconnected)',
            ['agent_id'],
            registry=self.registry
        )
        
        self.last_activity = Gauge(
            'agent_last_activity_timestamp',
            'Timestamp of agent last activity',
            ['agent_id'],
            registry=self.registry
        )
        
        self.capability_usage = Counter(
            'agent_capability_usage_total',
            'Usage count of agent capabilities',
            ['agent_id', 'capability_name'],
            registry=self.registry
        )
        
        self.error_count = Counter(
            'agent_errors_total',
            'Total errors encountered by agent',
            ['agent_id', 'error_type'],
            registry=self.registry
        )
    
    def record_message_sent(self, message_type: str, receiver_id: str):
        """Record a message sent by the agent"""
        self.messages_sent.labels(
            agent_id=self.agent_id,
            message_type=message_type,
            receiver_id=receiver_id
        ).inc()
        self.last_activity.labels(agent_id=self.agent_id).set_to_current_time()
    
    def record_message_received(self, message_type: str, sender_id: str):
        """Record a message received by the agent"""
        self.messages_received.labels(
            agent_id=self.agent_id,
            message_type=message_type,
            sender_id=sender_id
        ).inc()
        self.last_activity.labels(agent_id=self.agent_id).set_to_current_time()
    
    def record_message_processing_time(self, message_type: str, duration: float):
        """Record time spent processing a message"""
        self.message_processing_time.labels(
            agent_id=self.agent_id,
            message_type=message_type
        ).observe(duration)
    
    def record_interaction_initiated(self, interaction_type: str, target_id: str):
        """Record an interaction initiated by the agent"""
        self.interactions_initiated.labels(
            agent_id=self.agent_id,
            interaction_type=interaction_type,
            target_id=target_id
        ).inc()
        self.last_activity.labels(agent_id=self.agent_id).set_to_current_time()
    
    def record_interaction_completed(self, interaction_type: str, status: str, duration: float):
        """Record a completed interaction"""
        self.interactions_completed.labels(
            agent_id=self.agent_id,
            interaction_type=interaction_type,
            status=status
        ).inc()
        
        self.interaction_duration.labels(
            agent_id=self.agent_id,
            interaction_type=interaction_type
        ).observe(duration)
    
    def record_capability_usage(self, capability_name: str):
        """Record usage of an agent capability"""
        self.capability_usage.labels(
            agent_id=self.agent_id,
            capability_name=capability_name
        ).inc()
    
    def record_error(self, error_type: str):
        """Record an error encountered by the agent"""
        self.error_count.labels(
            agent_id=self.agent_id,
            error_type=error_type
        ).inc()
    
    def set_connection_status(self, connected: bool):
        """Set the agent's connection status"""
        self.connection_status.labels(agent_id=self.agent_id).set(1 if connected else 0)


class LobbyMetrics:
    """Metrics collection for the lobby/central system"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        
        # System metrics
        self.active_agents = Gauge(
            'lobby_active_agents',
            'Number of currently active agents',
            registry=self.registry
        )
        
        self.total_messages = Counter(
            'lobby_messages_total',
            'Total messages processed by lobby',
            ['message_type'],
            registry=self.registry
        )
        
        self.message_routing_time = Histogram(
            'lobby_message_routing_seconds',
            'Time spent routing messages',
            ['message_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.websocket_connections = Gauge(
            'lobby_websocket_connections',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        self.registration_requests = Counter(
            'lobby_registration_requests_total',
            'Total agent registration requests',
            ['status'],
            registry=self.registry
        )
        
        self.conversation_count = Gauge(
            'lobby_active_conversations',
            'Number of active conversations',
            registry=self.registry
        )
        
        self.system_errors = Counter(
            'lobby_system_errors_total',
            'Total system errors',
            ['error_type'],
            registry=self.registry
        )
    
    def record_agent_registered(self):
        """Record a successful agent registration"""
        self.registration_requests.labels(status='success').inc()
    
    def record_agent_registration_failed(self):
        """Record a failed agent registration"""
        self.registration_requests.labels(status='failed').inc()
    
    def record_message_processed(self, message_type: str, routing_time: float):
        """Record a processed message"""
        self.total_messages.labels(message_type=message_type).inc()
        self.message_routing_time.labels(message_type=message_type).observe(routing_time)
    
    def set_active_agents(self, count: int):
        """Set the number of active agents"""
        self.active_agents.set(count)
    
    def set_websocket_connections(self, count: int):
        """Set the number of WebSocket connections"""
        self.websocket_connections.set(count)
    
    def set_active_conversations(self, count: int):
        """Set the number of active conversations"""
        self.conversation_count.set(count)
    
    def record_system_error(self, error_type: str):
        """Record a system error"""
        self.system_errors.labels(error_type=error_type).inc()


class PerformanceTracker:
    """Track performance metrics and detect anomalies"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        with self.lock:
            self.metrics[metric_name].append((timestamp, value))
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        with self.lock:
            values = [value for _, value in self.metrics[metric_name]]
            
            if not values:
                return {}
            
            return {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
            }
    
    def detect_anomalies(self, metric_name: str, threshold_std_devs: float = 2.0) -> List[tuple]:
        """Detect anomalies in a metric (values beyond threshold standard deviations)"""
        with self.lock:
            data = list(self.metrics[metric_name])
            
            if len(data) < 10:  # Need sufficient data
                return []
            
            values = [value for _, value in data]
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values)
            
            anomalies = []
            for timestamp, value in data:
                if abs(value - mean) > threshold_std_devs * std_dev:
                    anomalies.append((timestamp, value, abs(value - mean) / std_dev))
            
            return anomalies


class HealthMonitor:
    """Monitor system health with configurable checks"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.HEALTHY
        self.last_check_time: Optional[datetime] = None
        self.check_interval = 30  # seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def add_health_check(self, check: HealthCheck):
        """Add a health check"""
        self.checks[check.name] = check
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        self.checks.pop(name, None)
    
    async def run_check(self, check: HealthCheck) -> tuple[HealthStatus, Optional[str]]:
        """Run a single health check"""
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout
            )
            
            if result is True:
                return HealthStatus.HEALTHY, None
            elif isinstance(result, tuple):
                status, message = result
                return status, message
            else:
                return HealthStatus.UNHEALTHY, str(result)
                
        except asyncio.TimeoutError:
            return HealthStatus.UNHEALTHY, f"Health check timed out after {check.timeout}s"
        except Exception as e:
            return HealthStatus.UNHEALTHY, str(e)
    
    async def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all health checks"""
        results = {}
        critical_failures = 0
        total_failures = 0
        
        for name, check in self.checks.items():
            status, error = await self.run_check(check)
            
            check.last_check = datetime.now(timezone.utc)
            check.last_status = status
            check.last_error = error
            
            results[name] = {
                'status': status.value,
                'critical': check.critical,
                'last_check': check.last_check.isoformat(),
                'error': error
            }
            
            if status != HealthStatus.HEALTHY:
                total_failures += 1
                if check.critical:
                    critical_failures += 1
        
        # Determine overall status
        if critical_failures > 0:
            self.overall_status = HealthStatus.UNHEALTHY
        elif total_failures > 0:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
        
        self.last_check_time = datetime.now(timezone.utc)
        
        return results
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self._running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'overall_status': self.overall_status.value,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'checks': {
                name: {
                    'status': check.last_status.value if check.last_status else 'unknown',
                    'last_check': check.last_check.isoformat() if check.last_check else None,
                    'error': check.last_error,
                    'critical': check.critical
                }
                for name, check in self.checks.items()
            }
        }


class MonitoringSystem:
    """Central monitoring system coordinating all monitoring components"""
    
    def __init__(self, 
                 metrics_port: int = 8000,
                 enable_telemetry: bool = False,
                 jaeger_endpoint: Optional[str] = None):
        
        # Create separate registry for cleaner metrics
        self.registry = CollectorRegistry()
        
        # Initialize components
        self.lobby_metrics = LobbyMetrics(self.registry)
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()
        
        # Configuration
        self.metrics_port = metrics_port
        self.enable_telemetry = enable_telemetry and TELEMETRY_AVAILABLE
        
        # Setup structured logging
        self.setup_logging()
        
        # Setup telemetry if enabled
        if self.enable_telemetry:
            self.setup_telemetry(jaeger_endpoint)
        
        # Start metrics server
        self.start_metrics_server()
        
        # Setup default health checks
        self.setup_default_health_checks()
    
    def setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(__name__)
    
    def setup_telemetry(self, jaeger_endpoint: Optional[str]):
        """Setup distributed tracing with OpenTelemetry"""
        if not TELEMETRY_AVAILABLE:
            return
        
        trace.set_tracer_provider(TracerProvider())
        
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
                collector_endpoint=jaeger_endpoint,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(__name__)
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.metrics_port, registry=self.registry)
            self.logger.info(f"Metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    def setup_default_health_checks(self):
        """Setup default health checks"""
        # System memory check
        async def check_memory():
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return HealthStatus.UNHEALTHY, f"Memory usage too high: {memory.percent}%"
            elif memory.percent > 80:
                return HealthStatus.DEGRADED, f"Memory usage high: {memory.percent}%"
            return True
        
        # System CPU check
        async def check_cpu():
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return HealthStatus.UNHEALTHY, f"CPU usage too high: {cpu_percent}%"
            elif cpu_percent > 80:
                return HealthStatus.DEGRADED, f"CPU usage high: {cpu_percent}%"
            return True
        
        # Disk space check
        async def check_disk():
            import psutil
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            if percent_used > 95:
                return HealthStatus.UNHEALTHY, f"Disk usage too high: {percent_used:.1f}%"
            elif percent_used > 85:
                return HealthStatus.DEGRADED, f"Disk usage high: {percent_used:.1f}%"
            return True
        
        self.health_monitor.add_health_check(HealthCheck("memory", check_memory, critical=True))
        self.health_monitor.add_health_check(HealthCheck("cpu", check_cpu, critical=False))
        self.health_monitor.add_health_check(HealthCheck("disk", check_disk, critical=True))
    
    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get or create metrics for an agent"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id, self.registry)
        return self.agent_metrics[agent_id]
    
    def remove_agent_metrics(self, agent_id: str):
        """Remove metrics for an agent"""
        self.agent_metrics.pop(agent_id, None)
    
    async def start(self):
        """Start the monitoring system"""
        await self.health_monitor.start_monitoring()
        self.logger.info("Monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system"""
        await self.health_monitor.stop_monitoring()
        self.logger.info("Monitoring system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'health': self.health_monitor.get_status(),
            'active_agents': len(self.agent_metrics),
            'metrics_endpoint': f"http://localhost:{self.metrics_port}/metrics"
        }


# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None


def get_monitoring_system() -> Optional[MonitoringSystem]:
    """Get the global monitoring system instance"""
    return _monitoring_system


def initialize_monitoring(
    metrics_port: int = 8000,
    enable_telemetry: bool = False,
    jaeger_endpoint: Optional[str] = None
) -> MonitoringSystem:
    """Initialize the global monitoring system"""
    global _monitoring_system
    
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem(
            metrics_port=metrics_port,
            enable_telemetry=enable_telemetry,
            jaeger_endpoint=jaeger_endpoint
        )
    
    return _monitoring_system


# Decorator for automatic performance tracking
def track_performance(metric_name: str):
    """Decorator to automatically track function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if _monitoring_system:
                    _monitoring_system.performance_tracker.record_metric(metric_name, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if _monitoring_system:
                    _monitoring_system.performance_tracker.record_metric(f"{metric_name}_error", duration)
                
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if _monitoring_system:
                    _monitoring_system.performance_tracker.record_metric(metric_name, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                if _monitoring_system:
                    _monitoring_system.performance_tracker.record_metric(f"{metric_name}_error", duration)
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator 