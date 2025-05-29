"""
Production Load Balancer for Agent Lobby
Implements multiple load balancing strategies with health checks
"""
import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import structlog
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections" 
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    PERFORMANCE_BASED = "performance_based"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"

@dataclass
class AgentNode:
    """Represents an agent node in the load balancer"""
    agent_id: str
    capabilities: List[str]
    current_load: int = 0
    max_load: int = 100
    weight: float = 1.0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_times: deque = field(default_factory=lambda: deque(maxlen=10))
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    last_request_time: Optional[datetime] = None
    node_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 5    # seconds
    max_retries: int = 3
    circuit_breaker_threshold: int = 5  # failures before circuit opens
    circuit_breaker_timeout: int = 60   # seconds to wait before retry
    enable_health_checks: bool = True
    response_time_weight: float = 0.3
    success_rate_weight: float = 0.4
    load_weight: float = 0.3

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreaker:
    """Circuit breaker for agent nodes"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    threshold: int = 5
    timeout: int = 60  # seconds

class ProductionLoadBalancer:
    """Production-ready load balancer with advanced features"""
    
    def __init__(self, config: LoadBalancerConfig = None):
        self.config = config or LoadBalancerConfig()
        self.agents: Dict[str, AgentNode] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.strategy_counters: Dict[str, int] = defaultdict(int)
        self.health_check_task: Optional[asyncio.Task] = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_per_second": 0.0
        }
        self.request_history: deque = deque(maxlen=1000)
        
    async def start(self):
        """Start the load balancer"""
        if self.config.enable_health_checks:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Load balancer started", strategy=self.config.strategy.value)
    
    async def stop(self):
        """Stop the load balancer"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Load balancer stopped")
    
    def register_agent(self, agent_id: str, capabilities: List[str], max_load: int = 100, weight: float = 1.0, metadata: Dict[str, Any] = None):
        """Register a new agent with the load balancer"""
        self.agents[agent_id] = AgentNode(
            agent_id=agent_id,
            capabilities=capabilities,
            max_load=max_load,
            weight=weight,
            node_metadata=metadata or {}
        )
        self.circuit_breakers[agent_id] = CircuitBreaker(threshold=self.config.circuit_breaker_threshold)
        logger.info("Agent registered", agent_id=agent_id, capabilities=capabilities)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the load balancer"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.circuit_breakers[agent_id]
            logger.info("Agent unregistered", agent_id=agent_id)
    
    def update_agent_load(self, agent_id: str, current_load: int):
        """Update current load for an agent"""
        if agent_id in self.agents:
            self.agents[agent_id].current_load = current_load
    
    def get_agent_for_capability(self, capability: str, exclude_agents: List[str] = None) -> Optional[str]:
        """Get the best agent for a specific capability"""
        exclude_agents = exclude_agents or []
        
        # Filter agents by capability and health
        available_agents = [
            agent for agent in self.agents.values()
            if (capability in agent.capabilities and 
                agent.agent_id not in exclude_agents and
                self._is_agent_available(agent))
        ]
        
        if not available_agents:
            logger.warning("No available agents for capability", capability=capability)
            return None
        
        # Apply load balancing strategy
        selected_agent = self._apply_strategy(available_agents)
        if selected_agent:
            self._record_request(selected_agent.agent_id)
            logger.debug("Agent selected", agent_id=selected_agent.agent_id, capability=capability, strategy=self.config.strategy.value)
        
        return selected_agent.agent_id if selected_agent else None
    
    def _is_agent_available(self, agent: AgentNode) -> bool:
        """Check if an agent is available for requests"""
        # Check circuit breaker
        circuit = self.circuit_breakers.get(agent.agent_id)
        if circuit and circuit.state == CircuitState.OPEN:
            # Check if circuit should move to half-open
            if (circuit.last_failure_time and 
                datetime.now(timezone.utc) - circuit.last_failure_time > timedelta(seconds=circuit.timeout)):
                circuit.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker half-open", agent_id=agent.agent_id)
            else:
                return False
        
        # Check health status
        if agent.health_status == HealthStatus.UNHEALTHY:
            return False
        
        # Check load capacity
        load_percentage = (agent.current_load / agent.max_load) * 100
        if load_percentage >= 95:  # 95% capacity threshold
            return False
        
        return True
    
    def _apply_strategy(self, available_agents: List[AgentNode]) -> Optional[AgentNode]:
        """Apply the configured load balancing strategy"""
        if not available_agents:
            return None
        
        strategy = self.config.strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_strategy(available_agents)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_strategy(available_agents)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_strategy(available_agents)
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_strategy(available_agents)
        elif strategy == LoadBalancingStrategy.RANDOM:
            return self._random_strategy(available_agents)
        elif strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_strategy(available_agents)
        else:
            return available_agents[0]  # Fallback
    
    def _round_robin_strategy(self, agents: List[AgentNode]) -> AgentNode:
        """Simple round-robin selection"""
        capability = agents[0].capabilities[0] if agents[0].capabilities else "default"
        index = self.strategy_counters[f"rr_{capability}"] % len(agents)
        self.strategy_counters[f"rr_{capability}"] += 1
        return agents[index]
    
    def _least_connections_strategy(self, agents: List[AgentNode]) -> AgentNode:
        """Select agent with least current load"""
        return min(agents, key=lambda a: a.current_load)
    
    def _weighted_round_robin_strategy(self, agents: List[AgentNode]) -> AgentNode:
        """Weighted round-robin based on agent weights"""
        total_weight = sum(agent.weight for agent in agents)
        if total_weight == 0:
            return agents[0]
        
        # Simple weighted selection
        weights = [agent.weight / total_weight for agent in agents]
        return random.choices(agents, weights=weights)[0]
    
    def _least_response_time_strategy(self, agents: List[AgentNode]) -> AgentNode:
        """Select agent with lowest average response time"""
        def avg_response_time(agent):
            if not agent.response_times:
                return 0  # New agents get priority
            return sum(agent.response_times) / len(agent.response_times)
        
        return min(agents, key=avg_response_time)
    
    def _random_strategy(self, agents: List[AgentNode]) -> AgentNode:
        """Random selection"""
        return random.choice(agents)
    
    def _performance_based_strategy(self, agents: List[AgentNode]) -> AgentNode:
        """Advanced strategy considering multiple performance metrics"""
        def performance_score(agent):
            # Normalize metrics (lower is better for response time and load)
            load_score = 1.0 - (agent.current_load / agent.max_load)
            success_score = agent.success_rate
            
            # Response time score (lower response time = higher score)
            if agent.response_times:
                avg_response_time = sum(agent.response_times) / len(agent.response_times)
                response_score = 1.0 / (1.0 + avg_response_time)  # Inverse relationship
            else:
                response_score = 1.0  # New agents get high score
            
            # Weighted combination
            total_score = (
                load_score * self.config.load_weight +
                success_score * self.config.success_rate_weight +
                response_score * self.config.response_time_weight
            )
            
            return total_score
        
        return max(agents, key=performance_score)
    
    def _record_request(self, agent_id: str):
        """Record a request being sent to an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.total_requests += 1
            agent.last_request_time = datetime.now(timezone.utc)
            agent.current_load += 1
        
        # Update global metrics
        self.metrics["total_requests"] += 1
        self.request_history.append(time.time())
    
    def record_response(self, agent_id: str, response_time: float, success: bool):
        """Record the response from an agent"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        circuit = self.circuit_breakers[agent_id]
        
        # Update response time
        agent.response_times.append(response_time)
        
        # Update load (request completed)
        agent.current_load = max(0, agent.current_load - 1)
        
        if success:
            # Reset circuit breaker on success
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                logger.info("Circuit breaker closed", agent_id=agent_id)
            
            self.metrics["successful_requests"] += 1
        else:
            # Handle failure
            agent.failed_requests += 1
            circuit.failure_count += 1
            circuit.last_failure_time = datetime.now(timezone.utc)
            
            # Open circuit breaker if threshold reached
            if circuit.failure_count >= circuit.threshold and circuit.state == CircuitState.CLOSED:
                circuit.state = CircuitState.OPEN
                logger.warning("Circuit breaker opened", agent_id=agent_id, failures=circuit.failure_count)
            
            self.metrics["failed_requests"] += 1
        
        # Update success rate
        if agent.total_requests > 0:
            agent.success_rate = (agent.total_requests - agent.failed_requests) / agent.total_requests
    
    async def _health_check_loop(self):
        """Continuous health check loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _perform_health_checks(self):
        """Perform health checks on all agents"""
        if not self.agents:
            return
        
        health_check_tasks = []
        for agent_id in list(self.agents.keys()):
            task = asyncio.create_task(self._check_agent_health(agent_id))
            health_check_tasks.append(task)
        
        # Wait for all health checks to complete
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_agent_health(self, agent_id: str):
        """Check health of a specific agent"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        try:
            # Simple health check - check if agent responded recently
            if agent.last_request_time:
                time_since_last_request = datetime.now(timezone.utc) - agent.last_request_time
                
                if time_since_last_request > timedelta(minutes=5):
                    # Agent hasn't been used recently, consider it unknown
                    if agent.health_status != HealthStatus.UNKNOWN:
                        agent.health_status = HealthStatus.UNKNOWN
                        logger.info("Agent health unknown - no recent activity", agent_id=agent_id)
                elif agent.success_rate >= 0.9:
                    # High success rate = healthy
                    if agent.health_status != HealthStatus.HEALTHY:
                        agent.health_status = HealthStatus.HEALTHY
                        logger.info("Agent health good", agent_id=agent_id, success_rate=agent.success_rate)
                elif agent.success_rate >= 0.7:
                    # Moderate success rate = degraded
                    if agent.health_status != HealthStatus.DEGRADED:
                        agent.health_status = HealthStatus.DEGRADED
                        logger.warning("Agent health degraded", agent_id=agent_id, success_rate=agent.success_rate)
                else:
                    # Low success rate = unhealthy
                    if agent.health_status != HealthStatus.UNHEALTHY:
                        agent.health_status = HealthStatus.UNHEALTHY
                        logger.error("Agent unhealthy", agent_id=agent_id, success_rate=agent.success_rate)
            else:
                # No requests yet, assume healthy
                agent.health_status = HealthStatus.HEALTHY
            
            agent.last_health_check = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error("Health check failed", agent_id=agent_id, error=str(e))
            agent.health_status = HealthStatus.UNHEALTHY
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        now = time.time()
        recent_requests = [req_time for req_time in self.request_history if now - req_time <= 60]
        
        agent_stats = []
        for agent in self.agents.values():
            circuit = self.circuit_breakers.get(agent.agent_id)
            avg_response_time = sum(agent.response_times) / len(agent.response_times) if agent.response_times else 0
            
            agent_stats.append({
                "agent_id": agent.agent_id,
                "capabilities": agent.capabilities,
                "current_load": agent.current_load,
                "max_load": agent.max_load,
                "load_percentage": (agent.current_load / agent.max_load) * 100,
                "health_status": agent.health_status.value,
                "success_rate": agent.success_rate,
                "total_requests": agent.total_requests,
                "failed_requests": agent.failed_requests,
                "average_response_time": avg_response_time,
                "circuit_breaker_state": circuit.state.value if circuit else "unknown",
                "weight": agent.weight
            })
        
        return {
            "strategy": self.config.strategy.value,
            "total_agents": len(self.agents),
            "healthy_agents": len([a for a in self.agents.values() if a.health_status == HealthStatus.HEALTHY]),
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": self.metrics["successful_requests"] / max(1, self.metrics["total_requests"]),
            "requests_per_minute": len(recent_requests),
            "agents": agent_stats
        }

# Global load balancer instance
load_balancer = ProductionLoadBalancer() 