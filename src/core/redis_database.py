"""
Redis Database Layer for Agent Lobbi
High-performance replacement for SQLite agent operations
Maintains same interface while providing sub-10ms response times
"""
import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import uuid
import logging

# Redis is temporarily disabled due to aioredis 2.0.1 TimeoutError conflict
# To enable Redis, either:
# 1. Set environment variable ENABLE_REDIS=true, or
# 2. Upgrade aioredis to a version that fixes the TimeoutError conflict
import os
REDIS_AVAILABLE = False
aioredis = None

# Only attempt Redis import if explicitly enabled
if os.environ.get("ENABLE_REDIS", "false").lower() == "true":
    try:
        import aioredis
        REDIS_AVAILABLE = True
        import logging
        logging.info("Redis enabled via ENABLE_REDIS environment variable")
    except ImportError:
        import logging
        logging.warning("ENABLE_REDIS is set but aioredis is not installed. Redis functionality will be disabled.")
                REDIS_AVAILABLE = False
    except Exception as e:
        import logging
        logging.warning(f"ENABLE_REDIS is set but Redis import failed: {e}. Redis functionality will be disabled.")
        REDIS_AVAILABLE = False
else:
    # Redis is disabled by default due to compatibility issues
    import logging
    logging.debug("Redis is disabled by default. Set ENABLE_REDIS=true to enable (may cause TimeoutError conflicts).")

try:
    import structlog
    logger = structlog.get_logger(__name__)
    HAS_STRUCTLOG = True
except ImportError:
    logger = logging.getLogger(__name__)
    HAS_STRUCTLOG = False

class RedisAgentDatabase:
    """High-performance Redis-based agent database"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis = None
        self.initialized = False
        
        # Redis key patterns
        self.AGENT_KEY = "agents:{agent_id}"
        self.AGENTS_SET = "agents:active"
        self.WORKFLOW_KEY = "workflows:{workflow_id}"
        self.WORKFLOWS_SET = "workflows:all"
        self.MESSAGE_KEY = "messages:{message_id}"
        self.METRICS_KEY = "metrics:{agent_id}"
        
    async def initialize(self):
        """Initialize Redis connection with aioredis 2.x compatibility"""
        if not REDIS_AVAILABLE:
            raise Exception("Redis not available due to import issues")
            
        try:
            # Create Redis connection with aioredis 2.x compatible parameters
            try:
                # For aioredis 2.x, use updated parameter names
                self.redis = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True,
                    health_check_interval=60,
                    max_connections=20
                )
            except TypeError as param_error:
                # Fallback for different aioredis versions with different parameter names
                print(f"Trying alternative Redis connection parameters: {param_error}")
                self.redis = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=10
                )
            
            # Test connection with timeout handling
            try:
                await asyncio.wait_for(self.redis.ping(), timeout=5.0)
                self.initialized = True
                logger.info("Redis database initialized successfully", url=self.redis_url)
            except asyncio.TimeoutError:
                logger.error("Redis connection timed out during initialization")
                raise Exception("Redis connection timeout")
            except ConnectionRefusedError:
                logger.error("Redis server is not running or not accessible")
                raise Exception("Redis server not accessible - please start Redis server")
            except OSError as os_error:
                # Handle connection refused and other OS-level errors
                if "refused" in str(os_error).lower():
                    logger.error("Redis server is not running or not accessible")
                    raise Exception("Redis server not accessible - please start Redis server")
                else:
                    logger.error("Redis connection failed", error=str(os_error))
                    raise Exception(f"Redis connection failed: {os_error}")
            except Exception as ping_error:
                logger.error("Redis ping failed", error=str(ping_error))
                raise Exception(f"Redis ping failed: {ping_error}")
            
        except Exception as e:
            logger.error("Failed to initialize Redis database", error=str(e))
            # Clean up on failure
            if self.redis:
                try:
                    await self.redis.close()
                except:
                    pass
                self.redis = None
            raise
    
    async def close(self):
        """Close Redis connections"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connections closed")
    
    # Agent Operations (High Performance)
    async def save_agent(self, agent_data: Dict[str, Any]) -> bool:
        """Save agent data to Redis (sub-5ms operation)"""
        try:
            agent_id = agent_data.get("id") or agent_data.get("agent_id")
            if not agent_id:
                logger.error("Agent ID required for save operation")
                return False
            
            # Prepare agent data for Redis
            redis_data = {
                "id": agent_id,
                "agent_type": agent_data.get("agent_type", "unknown"),
                "capabilities": json.dumps(agent_data.get("capabilities", [])),
                "status": agent_data.get("status", "online"),
                "reputation": str(agent_data.get("reputation", 100.0)),
                "last_seen": agent_data.get("last_seen", datetime.now(timezone.utc)).isoformat(),
                "metadata": json.dumps(agent_data.get("metadata", {})),
                "created_at": agent_data.get("created_at", datetime.now(timezone.utc)).isoformat(),
                "name": agent_data.get("name", agent_id)
            }
            
            # Use pipeline for atomic operations
            async with self.redis.pipeline() as pipe:
                # Store agent hash
                await pipe.hset(self.AGENT_KEY.format(agent_id=agent_id), mapping=redis_data)
                # Add to active agents set
                await pipe.sadd(self.AGENTS_SET, agent_id)
                # Set TTL for active status (optional - 24 hours)
                await pipe.expire(self.AGENT_KEY.format(agent_id=agent_id), 86400)
                await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error("Failed to save agent to Redis", agent_id=agent_id, error=str(e))
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID from Redis (sub-2ms operation)"""
        try:
            agent_data = await self.redis.hgetall(self.AGENT_KEY.format(agent_id=agent_id))
            
            if not agent_data:
                return None
            
            # Convert back to expected format
            return {
                "id": agent_data["id"],
                "agent_type": agent_data["agent_type"],
                "capabilities": json.loads(agent_data.get("capabilities", "[]")),
                "status": agent_data["status"],
                "reputation": float(agent_data.get("reputation", 100.0)),
                "last_seen": datetime.fromisoformat(agent_data["last_seen"]),
                "metadata": json.loads(agent_data.get("metadata", "{}")),
                "created_at": datetime.fromisoformat(agent_data["created_at"]),
                "name": agent_data.get("name", agent_id)
            }
            
        except Exception as e:
            logger.error("Failed to get agent from Redis", agent_id=agent_id, error=str(e))
            return None
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all active agents (sub-10ms operation)"""
        try:
            # Get all active agent IDs
            agent_ids = await self.redis.smembers(self.AGENTS_SET)
            
            if not agent_ids:
                return []
            
            # Get all agent data in parallel using pipeline
            async with self.redis.pipeline() as pipe:
                for agent_id in agent_ids:
                    await pipe.hgetall(self.AGENT_KEY.format(agent_id=agent_id))
                
                results = await pipe.execute()
            
            agents = []
            for agent_data in results:
                if agent_data:  # Skip empty results
                    agents.append({
                        "id": agent_data["id"],
                        "agent_type": agent_data["agent_type"],
                        "capabilities": json.loads(agent_data.get("capabilities", "[]")),
                        "status": agent_data["status"],
                        "reputation": float(agent_data.get("reputation", 100.0)),
                        "last_seen": datetime.fromisoformat(agent_data["last_seen"]),
                        "metadata": json.loads(agent_data.get("metadata", "{}")),
                        "created_at": datetime.fromisoformat(agent_data["created_at"]),
                        "name": agent_data.get("name", agent_data["id"])
                    })
            
            return agents
            
        except Exception as e:
            logger.error("Failed to get all agents from Redis", error=str(e))
            return []
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from Redis"""
        try:
            async with self.redis.pipeline() as pipe:
                await pipe.delete(self.AGENT_KEY.format(agent_id=agent_id))
                await pipe.srem(self.AGENTS_SET, agent_id)
                await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error("Failed to remove agent from Redis", agent_id=agent_id, error=str(e))
            return False
    
    # Workflow Operations
    async def save_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Save workflow to Redis"""
        try:
            workflow_id = workflow_data.get("id")
            if not workflow_id:
                workflow_id = str(uuid.uuid4())
                workflow_data["id"] = workflow_id
            
            # Prepare workflow data
            redis_data = {
                "id": workflow_id,
                "name": workflow_data.get("name", ""),
                "description": workflow_data.get("description", ""),
                "created_by": workflow_data.get("created_by", ""),
                "status": workflow_data.get("status", "created"),
                "tasks": json.dumps(workflow_data.get("tasks", {})),
                "shared_state": json.dumps(workflow_data.get("shared_state", {})),
                "participants": json.dumps(workflow_data.get("participants", [])),
                "result": json.dumps(workflow_data.get("result", {})),
                "error": workflow_data.get("error", ""),
                "created_at": workflow_data.get("created_at", datetime.now(timezone.utc)).isoformat(),
                "started_at": workflow_data.get("started_at", "").isoformat() if workflow_data.get("started_at") else "",
                "completed_at": workflow_data.get("completed_at", "").isoformat() if workflow_data.get("completed_at") else ""
            }
            
            async with self.redis.pipeline() as pipe:
                await pipe.hset(self.WORKFLOW_KEY.format(workflow_id=workflow_id), mapping=redis_data)
                await pipe.sadd(self.WORKFLOWS_SET, workflow_id)
                await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error("Failed to save workflow to Redis", workflow_id=workflow_id, error=str(e))
            return False
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID from Redis"""
        try:
            workflow_data = await self.redis.hgetall(self.WORKFLOW_KEY.format(workflow_id=workflow_id))
            
            if not workflow_data:
                return None
            
            return {
                "id": workflow_data["id"],
                "name": workflow_data["name"],
                "description": workflow_data["description"],
                "created_by": workflow_data["created_by"],
                "status": workflow_data["status"],
                "tasks": json.loads(workflow_data.get("tasks", "{}")),
                "shared_state": json.loads(workflow_data.get("shared_state", "{}")),
                "participants": json.loads(workflow_data.get("participants", "[]")),
                "result": json.loads(workflow_data.get("result", "{}")),
                "error": workflow_data.get("error", ""),
                "created_at": datetime.fromisoformat(workflow_data["created_at"]) if workflow_data["created_at"] else None,
                "started_at": datetime.fromisoformat(workflow_data["started_at"]) if workflow_data["started_at"] else None,
                "completed_at": datetime.fromisoformat(workflow_data["completed_at"]) if workflow_data["completed_at"] else None
            }
            
        except Exception as e:
            logger.error("Failed to get workflow from Redis", workflow_id=workflow_id, error=str(e))
            return None
    
    # Message Operations (Lightweight)
    async def save_message(self, message_data: Dict[str, Any]) -> bool:
        """Save message to Redis with TTL"""
        try:
            message_id = message_data.get("id") or str(uuid.uuid4())
            
            redis_data = {
                "id": message_id,
                "sender_id": message_data.get("sender_id", ""),
                "receiver_id": message_data.get("receiver_id", ""),
                "message_type": message_data.get("message_type", ""),
                "payload": json.dumps(message_data.get("payload", {})),
                "conversation_id": message_data.get("conversation_id", ""),
                "status": message_data.get("status", "pending"),
                "priority": str(message_data.get("priority", 2)),
                "created_at": message_data.get("created_at", datetime.now(timezone.utc)).isoformat()
            }
            
            # Store with 24 hour TTL (messages are transient)
            await self.redis.hset(self.MESSAGE_KEY.format(message_id=message_id), mapping=redis_data)
            await self.redis.expire(self.MESSAGE_KEY.format(message_id=message_id), 86400)
            
            return True
            
        except Exception as e:
            logger.error("Failed to save message to Redis", error=str(e))
            return False
    
    # Metrics Operations (High Performance)
    async def save_agent_metric(self, agent_id: str, metric_type: str, value: float, metadata: Dict[str, Any] = None) -> bool:
        """Save agent metric to Redis"""
        try:
            metric_data = {
                "agent_id": agent_id,
                "metric_type": metric_type,
                "value": str(value),
                "metadata": json.dumps(metadata or {}),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Use list to store metrics (FIFO with max length)
            metric_key = f"{self.METRICS_KEY.format(agent_id=agent_id)}:{metric_type}"
            
            async with self.redis.pipeline() as pipe:
                await pipe.lpush(metric_key, json.dumps(metric_data))
                await pipe.ltrim(metric_key, 0, 999)  # Keep last 1000 metrics
                await pipe.expire(metric_key, 604800)  # 7 days TTL
                await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error("Failed to save agent metric to Redis", agent_id=agent_id, error=str(e))
            return False
    
    async def get_agent_metrics(self, agent_id: str, metric_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent metrics from Redis"""
        try:
            if metric_type:
                metric_key = f"{self.METRICS_KEY.format(agent_id=agent_id)}:{metric_type}"
                raw_metrics = await self.redis.lrange(metric_key, 0, limit - 1)
            else:
                # Get all metric types for agent
                pattern = f"{self.METRICS_KEY.format(agent_id=agent_id)}:*"
                keys = await self.redis.keys(pattern)
                
                raw_metrics = []
                for key in keys:
                    metrics = await self.redis.lrange(key, 0, limit // len(keys) if keys else limit)
                    raw_metrics.extend(metrics)
            
            # Parse and return metrics
            parsed_metrics = []
            for raw_metric in raw_metrics:
                try:
                    metric = json.loads(raw_metric)
                    metric["timestamp"] = datetime.fromisoformat(metric["timestamp"])
                    metric["value"] = float(metric["value"])
                    metric["metadata"] = json.loads(metric["metadata"])
                    parsed_metrics.append(metric)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse metric", error=str(e))
            
            return parsed_metrics[:limit]
            
        except Exception as e:
            logger.error("Failed to get agent metrics from Redis", agent_id=agent_id, error=str(e))
            return []

# Note: RedisAgentDatabase instances are now created as needed by DatabaseManager
# No singleton instance required 