#!/usr/bin/env python3
"""
Prometheus Monitoring System for Agent Lobbi Security
Enterprise-grade metrics collection and monitoring for security events
"""

import time
from typing import Dict, Any, List, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Info, start_http_server, 
    CollectorRegistry, generate_latest
)
import structlog
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict

logger = structlog.get_logger(__name__)

@dataclass
class PrometheusConfig:
    """Prometheus monitoring configuration"""
    enabled: bool = True
    port: int = 8090
    host: str = "0.0.0.0"
    metrics_path: str = "/metrics"
    collect_interval: int = 10  # seconds

class SecurityMetrics:
    """Comprehensive security metrics collection"""
    
    def __init__(self, config: PrometheusConfig = None, registry: CollectorRegistry = None):
        self.config = config or PrometheusConfig()
        self.registry = registry or CollectorRegistry()
        # Authentication metrics
        self.auth_attempts_total = Counter(
            'security_auth_attempts_total',
            'Total authentication attempts',
            ['method', 'result', 'agent_id'],
            registry=self.registry
        )
        
        self.auth_failures_total = Counter(
            'security_auth_failures_total',
            'Total authentication failures',
            ['method', 'reason', 'ip_address'],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'security_active_sessions',
            'Number of active authenticated sessions',
            ['agent_id'],
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits_total = Counter(
            'security_rate_limit_hits_total',
            'Total rate limit hits',
            ['limiter_type', 'agent_id'],
            registry=self.registry
        )
        
        self.rate_limit_wait_time = Histogram(
            'security_rate_limit_wait_seconds',
            'Rate limit wait time in seconds',
            ['limiter_type'],
            registry=self.registry
        )
        
        # Security events metrics
        self.security_events_total = Counter(
            'security_events_total',
            'Total security events',
            ['event_type', 'risk_level'],
            registry=self.registry
        )
        
        self.blocked_requests_total = Counter(
            'security_blocked_requests_total',
            'Total blocked requests',
            ['reason', 'ip_address'],
            registry=self.registry
        )
        
        # Input validation metrics
        self.validation_checks_total = Counter(
            'security_validation_checks_total',
            'Total input validation checks',
            ['check_type', 'result'],
            registry=self.registry
        )
        
        self.validation_time = Histogram(
            'security_validation_duration_seconds',
            'Input validation processing time',
            ['check_type'],
            registry=self.registry
        )
        
        # Encryption metrics
        self.encryption_operations_total = Counter(
            'security_encryption_operations_total',
            'Total encryption/decryption operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.encryption_time = Histogram(
            'security_encryption_duration_seconds',
            'Encryption/decryption processing time',
            ['operation'],
            registry=self.registry
        )
        
        # System health metrics
        self.security_system_status = Gauge(
            'security_system_status',
            'Security system component status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        self.blocked_ips = Gauge(
            'security_blocked_ips_total',
            'Total number of blocked IP addresses',
            registry=self.registry
        )
        
        self.api_keys_active = Gauge(
            'security_api_keys_active',
            'Number of active API keys',
            registry=self.registry
        )
        
        # Performance metrics
        self.request_processing_time = Histogram(
            'security_request_duration_seconds',
            'Security request processing time',
            ['operation'],
            registry=self.registry
        )
        
        self.concurrent_requests = Gauge(
            'security_concurrent_requests',
            'Number of concurrent security requests being processed',
            registry=self.registry
        )
        
        # Agent-specific metrics
        self.agent_connections = Gauge(
            'security_agent_connections',
            'Number of connected agents',
            ['agent_type'],
            registry=self.registry
        )
        
        self.agent_last_activity = Gauge(
            'security_agent_last_activity_timestamp',
            'Timestamp of last activity for each agent',
            ['agent_id'],
            registry=self.registry
        )
        
        # System info
        self.security_info = Info(
            'security_system_info',
            'Security system information',
            registry=self.registry
        )
    
        self._http_server = None
        self._running = False
        
    def start_http_server(self):
        """Start Prometheus HTTP metrics server"""
        if not self.config.enabled:
            logger.info("Prometheus monitoring disabled")
            return
        
        try:
            self._http_server = start_http_server(
                self.config.port, 
                addr=self.config.host,
                registry=self.registry
            )
            self._running = True
            logger.info("Prometheus metrics server started", 
                       port=self.config.port, host=self.config.host)
        except Exception as e:
            logger.error("Failed to start Prometheus metrics server", error=str(e))
    
    def stop_http_server(self):
        """Stop Prometheus HTTP metrics server"""
        if self._http_server:
            self._http_server.shutdown()
            self._running = False
            logger.info("Prometheus metrics server stopped")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    # Authentication metrics methods
    def record_auth_attempt(self, method: str, result: str, agent_id: str = "unknown"):
        """Record authentication attempt"""
        self.auth_attempts_total.labels(method=method, result=result, agent_id=agent_id).inc()
    
    def record_auth_failure(self, method: str, reason: str, ip_address: str):
        """Record authentication failure"""
        self.auth_failures_total.labels(method=method, reason=reason, ip_address=ip_address).inc()
    
    def set_active_sessions(self, agent_id: str, count: int):
        """Set number of active sessions for agent"""
        self.active_sessions.labels(agent_id=agent_id).set(count)
    
    # Rate limiting metrics methods
    def record_rate_limit_hit(self, limiter_type: str, agent_id: str = "global"):
        """Record rate limit hit"""
        self.rate_limit_hits_total.labels(limiter_type=limiter_type, agent_id=agent_id).inc()
    
    def record_rate_limit_wait(self, limiter_type: str, wait_time: float):
        """Record rate limit wait time"""
        self.rate_limit_wait_time.labels(limiter_type=limiter_type).observe(wait_time)
    
    # Security events metrics methods
    def record_security_event(self, event_type: str, risk_level: str):
        """Record security event"""
        self.security_events_total.labels(event_type=event_type, risk_level=risk_level).inc()
    
    def record_blocked_request(self, reason: str, ip_address: str):
        """Record blocked request"""
        self.blocked_requests_total.labels(reason=reason, ip_address=ip_address).inc()
    
    # Input validation metrics methods
    def record_validation_check(self, check_type: str, result: str, duration: float = None):
        """Record input validation check"""
        self.validation_checks_total.labels(check_type=check_type, result=result).inc()
        if duration is not None:
            self.validation_time.labels(check_type=check_type).observe(duration)
    
    # Encryption metrics methods
    def record_encryption_operation(self, operation: str, result: str, duration: float = None):
        """Record encryption/decryption operation"""
        self.encryption_operations_total.labels(operation=operation, result=result).inc()
        if duration is not None:
            self.encryption_time.labels(operation=operation).observe(duration)
    
    # System health metrics methods
    def set_system_status(self, component: str, healthy: bool):
        """Set system component status"""
        self.security_system_status.labels(component=component).set(1 if healthy else 0)
    
    def set_blocked_ips_count(self, count: int):
        """Set total blocked IPs"""
        self.blocked_ips.set(count)
    
    def set_active_api_keys_count(self, count: int):
        """Set active API keys count"""
        self.api_keys_active.set(count)
    
    # Performance metrics methods
    def record_request_duration(self, operation: str, duration: float):
        """Record request processing duration"""
        self.request_processing_time.labels(operation=operation).observe(duration)
    
    def set_concurrent_requests(self, count: int):
        """Set concurrent requests count"""
        self.concurrent_requests.set(count)
    
    # Agent metrics methods
    def set_agent_connections(self, agent_type: str, count: int):
        """Set agent connections count"""
        self.agent_connections.labels(agent_type=agent_type).set(count)
    
    def record_agent_activity(self, agent_id: str):
        """Record agent activity timestamp"""
        self.agent_last_activity.labels(agent_id=agent_id).set(time.time())
    
    def set_system_info(self, info_dict: Dict[str, str]):
        """Set system information"""
        self.security_info.info(info_dict)

class MetricsCollector:
    """Automated metrics collection and reporting"""
    
    def __init__(self, security_manager, metrics: SecurityMetrics):
        self.security_manager = security_manager
        self.metrics = metrics
        self._collection_thread = None
        self._running = False
        
    def start_collection(self):
        """Start automated metrics collection"""
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop automated metrics collection"""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join()
        logger.info("Metrics collection stopped")
    
    def _collect_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(self.metrics.config.collect_interval)
            except Exception as e:
                logger.error("Error in metrics collection", error=str(e))
                time.sleep(5)  # Short delay on error
    
    def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            # Get security status
            status = self.security_manager.get_security_status()
            
            # Update system health metrics
            self.metrics.set_system_status("rate_limiter", status['health']['rate_limiter_active'])
            self.metrics.set_system_status("audit_logger", status['health']['audit_logger_active'])
            self.metrics.set_system_status("encryption", status['health'].get('encryption_active', False))
            
            # Update stats
            self.metrics.set_blocked_ips_count(status['stats']['blocked_ips'])
            self.metrics.set_active_api_keys_count(status['stats']['active_api_keys'])
            
            # System info
            info = {
                'version': '1.0.0',
                'auth_required': str(status['config']['auth_required']),
                'rate_limiting_enabled': str(status['config']['rate_limiting_enabled']),
                'encryption_enabled': str(status['config'].get('encryption_enabled', False)),
                'tls_enabled': str(status['config'].get('tls_enabled', False))
            }
            self.metrics.set_system_info(info)
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))

# Global metrics instance
_global_metrics = None

def get_metrics() -> SecurityMetrics:
    """Get global metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SecurityMetrics()
    return _global_metrics

def start_monitoring(security_manager, config: PrometheusConfig = None) -> tuple[SecurityMetrics, MetricsCollector]:
    """Start comprehensive monitoring system"""
    config = config or PrometheusConfig()
    metrics = SecurityMetrics(config)
    collector = MetricsCollector(security_manager, metrics)
    
    # Start HTTP server
    metrics.start_http_server()
    
    # Start automated collection
    collector.start_collection()
    
    logger.info("Security monitoring system started")
    return metrics, collector

def stop_monitoring(metrics: SecurityMetrics, collector: MetricsCollector):
    """Stop monitoring system"""
    collector.stop_collection()
    metrics.stop_http_server()
    logger.info("Security monitoring system stopped") 