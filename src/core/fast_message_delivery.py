#!/usr/bin/env python3
"""
FAST MESSAGE DELIVERY SYSTEM
===========================
High-performance message delivery system for optimized agent communication
Addresses latency and speed issues in collaboration matching and task delivery
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger()

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class MessageType(Enum):
    TASK_DELEGATION = "task_delegation"
    TASK_RESULT = "task_result"
    AGENT_STATUS = "agent_status"
    COLLABORATION_REQUEST = "collaboration_request"
    PERFORMANCE_METRIC = "performance_metric"
    SYSTEM_STATUS = "system_status"
    TASK_ASSIGNMENT = "task_assignment"

@dataclass
class FastMessage:
    """Optimized message structure for fast delivery"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }

class FastMessageQueue:
    """High-performance priority queue for message delivery"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            MessagePriority.URGENT: asyncio.Queue(maxsize=1000),
            MessagePriority.HIGH: asyncio.Queue(maxsize=2000),
            MessagePriority.NORMAL: asyncio.Queue(maxsize=5000),
            MessagePriority.LOW: asyncio.Queue(maxsize=2000)
        }
        self.message_count = 0
        self.delivery_metrics = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "average_delivery_time": 0.0
        }
    
    async def enqueue(self, message: FastMessage) -> bool:
        """Add message to priority queue"""
        try:
            if self.message_count >= self.max_size:
                logger.warning("Message queue full, dropping low priority messages")
                await self._drop_low_priority_messages()
            
            queue = self.queues[message.priority]
            await queue.put(message)
            self.message_count += 1
            self.delivery_metrics["messages_sent"] += 1
            
            logger.debug(f"Enqueued message {message.message_id} with priority {message.priority.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue message: {e}")
            return False
    
    async def dequeue(self) -> Optional[FastMessage]:
        """Get next message by priority"""
        # Check queues in priority order
        for priority in [MessagePriority.URGENT, MessagePriority.HIGH, 
                        MessagePriority.NORMAL, MessagePriority.LOW]:
            queue = self.queues[priority]
            
            if not queue.empty():
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                    self.message_count -= 1
                    return message
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error dequeuing message: {e}")
        
        return None
    
    async def _drop_low_priority_messages(self):
        """Drop low priority messages when queue is full"""
        dropped = 0
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL]:
            queue = self.queues[priority]
            while not queue.empty() and dropped < 100:
                try:
                    await asyncio.wait_for(queue.get(), timeout=0.01)
                    self.message_count -= 1
                    dropped += 1
                except asyncio.TimeoutError:
                    break
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} low priority messages")

class FastCollaborationMatcher:
    """Optimized collaboration matching for fast agent selection"""
    
    def __init__(self):
        self.agent_capabilities = {}  # agent_id -> capabilities
        self.agent_performance = {}   # agent_id -> performance metrics
        self.agent_availability = {}  # agent_id -> availability status
        self.matching_cache = {}      # capability_hash -> agents (TTL cache)
        self.cache_ttl = 30  # seconds
    
    def register_agent(self, agent_id: str, capabilities: List[str], 
                      performance_score: float = 1.0):
        """Register agent with fast matching optimization"""
        self.agent_capabilities[agent_id] = set(capabilities)
        self.agent_performance[agent_id] = {
            "score": performance_score,
            "task_count": 0,
            "average_completion_time": 0.0,
            "last_updated": datetime.now(timezone.utc)
        }
        self.agent_availability[agent_id] = "available"
        
        # Invalidate relevant cache entries
        self._invalidate_cache_for_capabilities(capabilities)
        
        logger.info(f"Fast matcher: Registered agent {agent_id} with capabilities: {capabilities}")
    
    def update_agent_performance(self, agent_id: str, completion_time: float, 
                               success: bool = True):
        """Update agent performance metrics for better matching"""
        if agent_id in self.agent_performance:
            perf = self.agent_performance[agent_id]
            
            # Update completion time average
            old_avg = perf["average_completion_time"]
            old_count = perf["task_count"]
            
            perf["task_count"] += 1
            perf["average_completion_time"] = (
                (old_avg * old_count + completion_time) / perf["task_count"]
            )
            
            # Update performance score based on success/failure
            if success:
                perf["score"] = min(2.0, perf["score"] * 1.1)  # Boost successful agents
            else:
                perf["score"] = max(0.1, perf["score"] * 0.9)  # Penalize failures
            
            perf["last_updated"] = datetime.now(timezone.utc)
    
    def find_best_agents(self, required_capabilities: List[str], 
                        max_agents: int = 1) -> List[str]:
        """Fast agent matching with caching and performance optimization"""
        matching_start = time.time()
        
        candidates = []
        required_caps = set(required_capabilities)
        
        for agent_id, agent_caps in self.agent_capabilities.items():
            # Skip unavailable agents
            if self.agent_availability.get(agent_id) != "available":
                continue
            
            # Check capability overlap
            if required_caps.intersection(agent_caps):
                perf = self.agent_performance.get(agent_id, {"score": 1.0})
                
                # Calculate matching score
                capability_match = len(required_caps.intersection(agent_caps)) / len(required_caps)
                performance_score = perf["score"]
                
                total_score = capability_match * performance_score
                candidates.append((agent_id, total_score))
        
        # Sort by score and select best agents
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agents = [agent_id for agent_id, score in candidates[:max_agents]]
        
        matching_time = time.time() - matching_start
        logger.info(f"Fast matching completed in {matching_time*1000:.2f}ms: {len(candidates)} candidates, selected {len(best_agents)}")
        
        return best_agents
    
    def _invalidate_cache_for_capabilities(self, capabilities: List[str]):
        """Invalidate cache entries that might be affected by new agent"""
        self.matching_cache.clear()  # Simple invalidation

class FastDeliverySystem:
    """Main fast delivery system coordinating all components"""
    
    def __init__(self):
        self.message_queue = FastMessageQueue()
        self.collaboration_matcher = FastCollaborationMatcher()
        self.delivery_handlers = {}  # message_type -> handler_function
        self.active_deliveries = {}  # message_id -> delivery_info
        self.running = False
        
        # Performance tracking
        self.performance_metrics = {
            "total_messages": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "average_latency_ms": 0.0,
            "peak_latency_ms": 0.0,
            "throughput_per_second": 0.0
        }
        
        # Setup default handlers
        self._setup_default_handlers()
    
    def register_delivery_handler(self, message_type: MessageType, 
                                handler: Callable[[FastMessage], Any]):
        """Register handler for specific message type"""
        self.delivery_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type.value}")
    
    async def send_message(self, sender_id: str, receiver_id: str, 
                          message_type: MessageType, payload: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Send message with fast delivery"""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        
        message = FastMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            created_at=datetime.now(timezone.utc)
        )
        
        delivery_start = time.time()
        success = await self.message_queue.enqueue(message)
        
        if success:
            self.active_deliveries[message_id] = {
                "message": message,
                "started_at": delivery_start,
                "status": "queued"
            }
            
            logger.debug(f"Message {message_id} queued for delivery")
            return message_id
        else:
            logger.error(f"Failed to queue message {message_id}")
            return ""
    
    async def fast_task_delegation(self, task_title: str, task_description: str,
                                 required_capabilities: List[str], 
                                 requester_id: str, max_agents: int = 1) -> Dict[str, Any]:
        """Optimized task delegation with fast matching"""
        delegation_start = time.time()
        
        # Fast agent matching
        matching_start = time.time()
        best_agents = self.collaboration_matcher.find_best_agents(
            required_capabilities, max_agents
        )
        matching_time = time.time() - matching_start
        
        if not best_agents:
            return {
                "success": False,
                "error": "No suitable agents available",
                "matching_time_ms": matching_time * 1000
            }
        
        # Delegate to best agent (simplified for now)
        selected_agent = best_agents[0]
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Send task message
        message_id = await self.send_message(
            sender_id="fast_delegation_system",
            receiver_id=selected_agent,
            message_type=MessageType.TASK_DELEGATION,
            payload={
                "task_id": task_id,
                "task_title": task_title,
                "task_description": task_description,
                "required_capabilities": required_capabilities,
                "requester_id": requester_id,
                "delegation_time": datetime.now(timezone.utc).isoformat()
            },
            priority=MessagePriority.HIGH
        )
        
        delegation_time = time.time() - delegation_start
        
        return {
            "success": True,
            "task_id": task_id,
            "assigned_agent": selected_agent,
            "message_id": message_id,
            "delegation_time_ms": delegation_time * 1000,
            "matching_time_ms": matching_time * 1000,
            "available_agents": len(best_agents)
        }
    
    async def start_delivery_loop(self):
        """Start the message delivery processing loop"""
        self.running = True
        logger.info("Fast delivery system started")
        
        while self.running:
            try:
                # Process messages from queue
                message = await self.message_queue.dequeue()
                
                if message:
                    await self._process_message(message)
                else:
                    # No messages, brief pause
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in delivery loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_message(self, message: FastMessage):
        """Process individual message delivery"""
        delivery_start = time.time()
        
        try:
            # Update delivery status
            if message.message_id in self.active_deliveries:
                self.active_deliveries[message.message_id]["status"] = "processing"
            
            # Get handler for message type
            handler = self.delivery_handlers.get(message.message_type)
            
            if handler:
                result = await handler(message)
                
                # Update metrics
                delivery_time = time.time() - delivery_start
                self._update_performance_metrics(delivery_time, True)
                
                # Mark as delivered
                if message.message_id in self.active_deliveries:
                    self.active_deliveries[message.message_id]["status"] = "delivered"
                    self.active_deliveries[message.message_id]["delivery_time"] = delivery_time
                
                logger.debug(f"Message {message.message_id} delivered in {delivery_time*1000:.2f}ms")
                
            else:
                logger.warning(f"No handler for message type {message.message_type.value}")
                self._update_performance_metrics(0, False)
                
        except Exception as e:
            logger.error(f"Failed to process message {message.message_id}: {e}")
            self._update_performance_metrics(0, False)
            
            # Handle retry logic
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                await self.message_queue.enqueue(message)
    
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        
        async def handle_task_delegation(message: FastMessage):
            """Handle task delegation messages"""
            payload = message.payload
            logger.info(f"Processing task delegation: {payload.get('task_title', 'Unknown')}")
            
            # Simulate task processing (replace with actual task execution)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            return {"status": "delegated", "agent": message.receiver_id}
        
        async def handle_task_result(message: FastMessage):
            """Handle task result messages"""
            payload = message.payload
            logger.info(f"Processing task result from {message.sender_id}")
            
            # Update agent performance
            completion_time = payload.get('execution_time', 1.0)
            success = payload.get('success', True)
            
            self.collaboration_matcher.update_agent_performance(
                message.sender_id, completion_time, success
            )
            
            return {"status": "result_processed"}
        
        # Register handlers
        self.register_delivery_handler(MessageType.TASK_DELEGATION, handle_task_delegation)
        self.register_delivery_handler(MessageType.TASK_RESULT, handle_task_result)
    
    def _update_performance_metrics(self, delivery_time: float, success: bool):
        """Update performance metrics"""
        self.performance_metrics["total_messages"] += 1
        
        if success:
            self.performance_metrics["successful_deliveries"] += 1
            
            # Update latency metrics
            latency_ms = delivery_time * 1000
            current_avg = self.performance_metrics["average_latency_ms"]
            total_successful = self.performance_metrics["successful_deliveries"]
            
            self.performance_metrics["average_latency_ms"] = (
                (current_avg * (total_successful - 1) + latency_ms) / total_successful
            )
            
            if latency_ms > self.performance_metrics["peak_latency_ms"]:
                self.performance_metrics["peak_latency_ms"] = latency_ms
        else:
            self.performance_metrics["failed_deliveries"] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total = self.performance_metrics["total_messages"]
        success_rate = (
            self.performance_metrics["successful_deliveries"] / total * 100 
            if total > 0 else 0
        )
        
        return {
            **self.performance_metrics,
            "success_rate": success_rate,
            "queue_size": self.message_queue.message_count,
            "active_deliveries": len(self.active_deliveries)
        }
    
    async def stop(self):
        """Stop the delivery system"""
        self.running = False
        logger.info("Fast delivery system stopped")

# Global fast delivery system instance
fast_delivery = None

def get_fast_delivery_system():
    """Get or create global fast delivery system"""
    global fast_delivery
    if fast_delivery is None:
        fast_delivery = FastCollaborationMatcher()
    return fast_delivery 