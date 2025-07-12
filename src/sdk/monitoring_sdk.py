"""
Production Monitoring SDK for Agent Lobbi
Includes metrics, health checks, error recovery, and observability
"""
import asyncio
import time
import json
import os
import sys
import psutil
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import deque, defaultdict
import threading
from contextlib import asynccontextmanager

# Error recovery and resilience
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Simple built-in circuit breaker implementation
class SimpleCircuitBreaker:
    """Simple circuit breaker implementation"""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                # Check if we should try half-open
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.recovery_timeout):
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                # Success - reset circuit breaker
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e
        
        return wrapper

logger = structlog.get_logger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    interval: int = 30  # seconds
    timeout: int = 5    # seconds
    critical: bool = False
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0

class MonitoringConfig:
    """Configuration for monitoring SDK"""
    def __init__(self):
        self.agent_id: str = os.getenv("AGENT_ID", "unknown")
        self.api_key: str = os.getenv("AGENT_API_KEY", "")
        self.lobby_url: str = os.getenv("LOBBY_URL", "http://localhost:8080")
        self.metrics_interval: int = int(os.getenv("METRICS_INTERVAL", "30"))
        self.health_check_interval: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "15"))
        self.enable_system_metrics: bool = os.getenv("ENABLE_SYSTEM_METRICS", "true").lower() == "true"
        self.enable_auto_recovery: bool = os.getenv("ENABLE_AUTO_RECOVERY", "true").lower() == "true"
        self.circuit_breaker_threshold: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"))
        self.circuit_breaker_timeout: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_backoff: int = int(os.getenv("RETRY_BACKOFF", "2"))

class ErrorRecoveryManager:
    """Handles error recovery and circuit breaking"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.circuit_breakers: Dict[str, SimpleCircuitBreaker] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_errors: Dict[str, datetime] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def register_circuit_breaker(self, name: str, failure_threshold: int = None, timeout: int = None):
        """Register a circuit breaker for a specific operation"""
        failure_threshold = failure_threshold or self.config.circuit_breaker_threshold
        timeout = timeout or self.config.circuit_breaker_timeout
        
        self.circuit_breakers[name] = SimpleCircuitBreaker(failure_threshold, timeout)
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for specific error types"""
        self.recovery_strategies[error_type] = strategy
    
    async def execute_with_recovery(self, operation_name: str, operation: Callable, *args, **kwargs):
        """Execute operation with circuit breaker and recovery"""
        if operation_name not in self.circuit_breakers:
            self.register_circuit_breaker(operation_name)
        
        circuit_breaker = self.circuit_breakers[operation_name]
        
        try:
            result = await circuit_breaker(operation)(*args, **kwargs)
            return result
        except Exception as e:
            self.error_counts[operation_name] += 1
            self.last_errors[operation_name] = datetime.now(timezone.utc)
            
            # Try recovery strategy
            if self.config.enable_auto_recovery:
                await self._attempt_recovery(operation_name, e)
            
            logger.error("Operation failed", operation=operation_name, error=str(e))
            raise

    async def _attempt_recovery(self, operation_name: str, error: Exception):
        """Attempt to recover from an error"""
        error_type = type(error).__name__
        if error_type in self.recovery_strategies:
            try:
                await self.recovery_strategies[error_type]()
                logger.info("Recovery strategy executed", operation=operation_name, error_type=error_type)
            except Exception as recovery_error:
                logger.error("Recovery strategy failed", operation=operation_name, error=str(recovery_error))

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        with self._lock:
            self.counters[name] += value
            metric = Metric(name, self.counters[name], MetricType.COUNTER, labels or {})
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        with self._lock:
            self.gauges[name] = value
            metric = Metric(name, value, MetricType.GAUGE, labels or {})
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        with self._lock:
            self.histograms[name].append(value)
            metric = Metric(name, value, MetricType.HISTOGRAM, labels or {})
            self.metrics[name].append(metric)
    
    @asynccontextmanager
    async def timer(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            with self._lock:
                self.timers[name].append(duration)
                metric = Metric(name, duration, MetricType.TIMER, labels or {})
                self.metrics[name].append(metric)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            recent_metrics = list(self.metrics[name])[-50:]  # Last 50 data points
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            return {
                "name": name,
                "count": len(values),
                "current": values[-1] if values else 0,
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "last_updated": recent_metrics[-1].timestamp.isoformat()
            }
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        if not self.config.enable_system_metrics:
            return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.set_gauge("system.cpu.usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.set_gauge("system.memory.usage_percent", memory.percent)
            self.set_gauge("system.memory.available_mb", memory.available / 1024 / 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.set_gauge("system.disk.usage_percent", disk_percent)
            
            # Network I/O
            network = psutil.net_io_counters()
            self.increment_counter("system.network.bytes_sent", network.bytes_sent)
            self.increment_counter("system.network.bytes_recv", network.bytes_recv)
            
            # Process info
            process = psutil.Process()
            self.set_gauge("system.process.memory_mb", process.memory_info().rss / 1024 / 1024)
            self.set_gauge("system.process.cpu_percent", process.cpu_percent())
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))

class HealthChecker:
    """Manages health checks for the agent"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.status_history: deque = deque(maxlen=100)
    
    def register_health_check(self, check: HealthCheck):
        """Register a new health check"""
        self.health_checks[check.name] = check
        logger.info("Health check registered", name=check.name, interval=check.interval)
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        critical_failures = 0
        total_checks = len(self.health_checks)
        
        for name, check in self.health_checks.items():
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    asyncio.create_task(self._run_single_check(check)),
                    timeout=check.timeout
                )
                
                check.last_result = result
                check.last_check = datetime.now(timezone.utc)
                
                if result:
                    check.consecutive_failures = 0
                else:
                    check.consecutive_failures += 1
                    if check.critical:
                        critical_failures += 1
                
                results[name] = {
                    "status": "pass" if result else "fail",
                    "critical": check.critical,
                    "consecutive_failures": check.consecutive_failures,
                    "last_check": check.last_check.isoformat()
                }
                
            except asyncio.TimeoutError:
                check.consecutive_failures += 1
                if check.critical:
                    critical_failures += 1
                
                results[name] = {
                    "status": "timeout",
                    "critical": check.critical,
                    "consecutive_failures": check.consecutive_failures,
                    "error": "Health check timed out"
                }
                
            except Exception as e:
                check.consecutive_failures += 1
                if check.critical:
                    critical_failures += 1
                
                results[name] = {
                    "status": "error",
                    "critical": check.critical,
                    "consecutive_failures": check.consecutive_failures,
                    "error": str(e)
                }
        
        # Determine overall health status
        if critical_failures > 0:
            self.overall_status = HealthStatus.UNHEALTHY
        elif total_checks > 0 and sum(1 for r in results.values() if r["status"] == "pass") / total_checks < 0.8:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
        
        # Record status in history
        self.status_history.append({
            "status": self.overall_status.value,
            "timestamp": datetime.now(timezone.utc),
            "critical_failures": critical_failures,
            "total_checks": total_checks
        })
        
        return {
            "overall_status": self.overall_status.value,
            "checks": results,
            "summary": {
                "total_checks": total_checks,
                "passing_checks": sum(1 for r in results.values() if r["status"] == "pass"),
                "failing_checks": sum(1 for r in results.values() if r["status"] in ["fail", "error", "timeout"]),
                "critical_failures": critical_failures
            }
        }
    
    async def _run_single_check(self, check: HealthCheck) -> bool:
        """Run a single health check"""
        if asyncio.iscoroutinefunction(check.check_function):
            return await check.check_function()
        else:
            return check.check_function()

class ProductionMonitoringSDK:
    """Production-ready monitoring SDK with comprehensive observability"""
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector(self.config)
        self.health_checker = HealthChecker(self.config)
        self.error_recovery = ErrorRecoveryManager(self.config)
        
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default recovery strategies
        self._setup_default_recovery_strategies()
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        def memory_check():
            """Check if memory usage is under 90%"""
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 90
            except:
                return True  # Assume healthy if can't check
        
        def disk_check():
            """Check if disk usage is under 95%"""
            try:
                disk = psutil.disk_usage('/')
                return (disk.used / disk.total) * 100 < 95
            except:
                return True
        
        self.health_checker.register_health_check(
            HealthCheck("memory", memory_check, interval=30, critical=True)
        )
        self.health_checker.register_health_check(
            HealthCheck("disk", disk_check, interval=60, critical=True)
        )
    
    def _setup_default_recovery_strategies(self):
        """Setup default error recovery strategies"""
        
        async def connection_recovery():
            """Recovery strategy for connection errors"""
            logger.info("Attempting connection recovery")
            await asyncio.sleep(5)  # Brief pause before retry
        
        async def memory_recovery():
            """Recovery strategy for memory errors"""
            logger.info("Attempting memory cleanup")
            import gc
            gc.collect()
        
        self.error_recovery.register_recovery_strategy("ConnectionError", connection_recovery)
        self.error_recovery.register_recovery_strategy("MemoryError", memory_recovery)
    
    async def start(self):
        """Start the monitoring SDK"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start metrics collection loop
        metrics_task = asyncio.create_task(self._metrics_loop())
        self.monitoring_tasks.append(metrics_task)
        
        # Start health check loop
        health_task = asyncio.create_task(self._health_check_loop())
        self.monitoring_tasks.append(health_task)
        
        logger.info("Monitoring SDK started", agent_id=self.config.agent_id)
    
    async def stop(self):
        """Stop the monitoring SDK"""
        self.is_running = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Monitoring SDK stopped")
    
    async def _metrics_loop(self):
        """Main metrics collection loop"""
        while self.is_running:
            try:
                # Collect system metrics
                self.metrics_collector.collect_system_metrics()
                
                # Report metrics to lobby (if configured)
                if self.config.api_key:
                    await self._report_metrics()
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.is_running:
            try:
                health_results = await self.health_checker.run_health_checks()
                
                # Report health status to lobby (if configured)
                if self.config.api_key:
                    await self._report_health(health_results)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error", error=str(e))
                await asyncio.sleep(5)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _report_metrics(self):
        """Report metrics to the lobby"""
        # This would integrate with the lobby's metrics endpoint
        # For now, just log the metrics
        summaries = {}
        for metric_name in self.metrics_collector.metrics.keys():
            summaries[metric_name] = self.metrics_collector.get_metric_summary(metric_name)
        
        logger.debug("Metrics reported", agent_id=self.config.agent_id, metrics_count=len(summaries))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def _report_health(self, health_results: Dict[str, Any]):
        """Report health status to the lobby"""
        # This would integrate with the lobby's health endpoint
        logger.debug("Health status reported", 
                    agent_id=self.config.agent_id, 
                    status=health_results["overall_status"])
    
    # Public API methods
    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        self.metrics_collector.increment_counter(name, value, labels)
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        self.metrics_collector.set_gauge(name, value, labels)
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        self.metrics_collector.record_histogram(name, value, labels)
    
    def timer(self, name: str, labels: Dict[str, str] = None):
        """Get a timer context manager"""
        return self.metrics_collector.timer(name, labels)
    
    def add_health_check(self, name: str, check_function: Callable, interval: int = 30, critical: bool = False):
        """Add a custom health check"""
        health_check = HealthCheck(name, check_function, interval, critical=critical)
        self.health_checker.register_health_check(health_check)
    
    async def execute_with_recovery(self, operation_name: str, operation: Callable, *args, **kwargs):
        """Execute operation with error recovery"""
        return await self.error_recovery.execute_with_recovery(operation_name, operation, *args, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "agent_id": self.config.agent_id,
            "monitoring_active": self.is_running,
            "health_status": self.health_checker.overall_status.value,
            "metrics_collected": len(self.metrics_collector.metrics),
            "health_checks": len(self.health_checker.health_checks),
            "circuit_breakers": len(self.error_recovery.circuit_breakers),
            "uptime": "unknown"  # Would track actual uptime
        }

# Global monitoring instance
monitoring_sdk = ProductionMonitoringSDK() 