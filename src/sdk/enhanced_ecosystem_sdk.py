"""
Enhanced Agent Ecosystem SDK with comprehensive features.
Integrates database persistence, monitoring, security, configuration management, and advanced capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto, IntEnum
from typing import Any, Dict, Optional, List, Set, Callable, Awaitable, Union
import uuid
import json
from json import JSONDecodeError
import httpx
import websockets
import sys
from pathlib import Path

# Import our enhanced core modules
from ..core.database import DatabaseManager, AgentRegistration, MessageRecord, InteractionRecord, create_database_manager
from ..core.monitoring import AgentMetrics, get_monitoring_system, track_performance
from ..core.security import SecurityManager, Permission, Role, create_security_manager
from ..core.config import get_config, AgentEcosystemConfig, initialize_config

# Import base message types (keeping compatibility)
from ..core.message import Message, MessageType, MessagePriority, MessageValidationError, AgentCapability


class SDKError(Exception):
    """Base exception for SDK errors"""
    pass


class ConnectionError(SDKError):
    """Raised when connection to lobby fails"""
    pass


class AuthenticationError(SDKError):
    """Raised when authentication fails"""
    pass


class ConfigurationError(SDKError):
    """Raised when configuration is invalid"""
    pass


@dataclass
class SDKConfig:
    """SDK-specific configuration"""
    # Connection settings
    lobby_http_url: str = "http://localhost:8080"
    lobby_ws_url: str = "ws://localhost:8081"
    
    # Timeouts
    http_timeout: float = 30.0
    websocket_timeout: float = 10.0
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0
    
    # Message handling
    message_queue_size: int = 1000
    message_timeout: float = 30.0
    max_message_size: int = 1024 * 1024  # 1MB
    
    # Persistence
    enable_persistence: bool = True
    database_type: str = "mongodb"  # mongodb or postgresql
    database_host: str = "localhost"
    database_port: Optional[int] = None
    database_username: Optional[str] = None
    database_password: Optional[str] = None
    database_name: str = "agent_ecosystem"
    
    # Security
    enable_security: bool = True
    auth_token: Optional[str] = None
    jwt_secret_key: str = "your-secret-key-change-in-production"
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_enabled: bool = True
    performance_tracking: bool = True
    
    # Features
    auto_reconnect: bool = True
    heartbeat_interval: float = 60.0
    capability_caching: bool = True
    message_encryption: bool = False


class AgentState(Enum):
    """Agent connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    REGISTERED = "registered"
    ERROR = "error"


class InteractionManager:
    """Enhanced interaction manager with persistence and monitoring"""
    
    def __init__(self, sdk_client: 'EnhancedEcosystemClient'):
        self.sdk_client = sdk_client
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.active_interactions: Dict[str, InteractionRecord] = {}
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def start_interaction(self, 
                               target_id: str, 
                               interaction_type: str,
                               metadata: Dict[str, Any] = None) -> str:
        """Start a new interaction and return interaction ID"""
        interaction_id = str(uuid.uuid4())
        
        interaction = InteractionRecord(
            interaction_id=interaction_id,
            initiator_id=self.sdk_client.agent_id,
            target_id=target_id,
            interaction_type=interaction_type,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.active_interactions[interaction_id] = interaction
        
        # Store in database if persistence is enabled
        if self.sdk_client.database_manager:
            await self.sdk_client.database_manager.create_interaction(interaction)
        
        # Record metrics
        if self.sdk_client.metrics:
            self.sdk_client.metrics.record_interaction_initiated(interaction_type, target_id)
        
        return interaction_id
    
    async def complete_interaction(self, 
                                  interaction_id: str, 
                                  result: Dict[str, Any],
                                  status: str = "completed"):
        """Complete an interaction"""
        async with self._lock:
            interaction = self.active_interactions.get(interaction_id)
            if not interaction:
                self.logger.warning(f"Interaction {interaction_id} not found")
                return
            
            # Calculate duration
            duration = (datetime.now(timezone.utc) - interaction.started_at).total_seconds()
            
            # Update interaction
            interaction.completed_at = datetime.now(timezone.utc)
            interaction.status = status
            interaction.result = result
            
            # Remove from active interactions
            del self.active_interactions[interaction_id]
        
        # Update database
        if self.sdk_client.database_manager:
            await self.sdk_client.database_manager.complete_interaction(interaction_id, result)
        
        # Record metrics
        if self.sdk_client.metrics:
            self.sdk_client.metrics.record_interaction_completed(
                interaction.interaction_type, status, duration
            )
    
    async def register_request(self, conversation_id: str) -> asyncio.Future:
        """Register a pending request"""
        async with self._lock:
            if conversation_id in self.pending_requests:
                raise ValueError(f"Duplicate conversation_id: {conversation_id}")
            
            future = asyncio.Future()
            self.pending_requests[conversation_id] = future
            return future
    
    async def resolve_request(self, response_message: Message):
        """Resolve a pending request"""
        if not response_message.conversation_id:
            self.logger.warning(f"Response without conversation_id: {response_message.message_id}")
            return
        
        async with self._lock:
            future = self.pending_requests.pop(response_message.conversation_id, None)
            if future and not future.done():
                future.set_result(response_message)
    
    async def cancel_request(self, conversation_id: str, reason: str = "Cancelled"):
        """Cancel a pending request"""
        async with self._lock:
            future = self.pending_requests.pop(conversation_id, None)
            if future and not future.done():
                future.cancel()


class MessageQueue:
    """Enhanced message queue with persistence and priority handling"""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.PriorityQueue(maxsize=maxsize)
        self.failed_messages: List[Message] = []
        self._lock = asyncio.Lock()
    
    async def put(self, message: Message):
        """Add message to queue with priority"""
        priority = message.priority.value
        await self.queue.put((priority, time.time(), message))
    
    async def get(self) -> Message:
        """Get next message from queue"""
        _, _, message = await self.queue.get()
        return message
    
    async def add_failed_message(self, message: Message):
        """Add a failed message for retry"""
        async with self._lock:
            self.failed_messages.append(message)
    
    async def get_failed_messages(self) -> List[Message]:
        """Get and clear failed messages"""
        async with self._lock:
            messages = self.failed_messages.copy()
            self.failed_messages.clear()
            return messages
    
    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()


class EnhancedEcosystemClient:
    """Enhanced ecosystem client with comprehensive features"""
    
    def __init__(self,
                 agent_id: str,
                 agent_type: str,
                 capabilities: List[AgentCapability],
                 agent_message_handler: Callable[[Message], Awaitable[Optional[Message]]],
                 config: Optional[SDKConfig] = None,
                 ecosystem_config: Optional[AgentEcosystemConfig] = None):
        
        # Basic agent information
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.agent_message_handler = agent_message_handler
        
        # Configuration
        self.config = config or SDKConfig()
        self.ecosystem_config = ecosystem_config or get_config()
        
        # State management
        self.state = AgentState.DISCONNECTED
        self.last_heartbeat = None
        self.connection_attempts = 0
        
        # Communication
        self.http_client: Optional[httpx.AsyncClient] = None
        self.ws_connection = None
        self.message_queue = MessageQueue(self.config.message_queue_size)
        self.outbound_queue = MessageQueue(self.config.message_queue_size)
        
        # Enhanced features
        self.database_manager: Optional[DatabaseManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.metrics: Optional[AgentMetrics] = None
        self.interaction_manager = InteractionManager(self)
        
        # Tasks and event loop
        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize enhanced components based on configuration"""
        
        # Initialize database if persistence is enabled
        if self.config.enable_persistence:
            try:
                self.database_manager = create_database_manager(
                    db_type=self.config.database_type,
                    host=self.config.database_host,
                    port=self.config.database_port,
                    username=self.config.database_username,
                    password=self.config.database_password,
                    database=self.config.database_name
                )
                self.logger.info("Database manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {e}")
                if self.ecosystem_config.is_production():
                    raise ConfigurationError(f"Database initialization failed: {e}")
        
        # Initialize security if enabled
        if self.config.enable_security:
            try:
                self.security_manager = create_security_manager(
                    jwt_secret_key=self.config.jwt_secret_key
                )
                self.logger.info("Security manager initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize security: {e}")
                if self.ecosystem_config.is_production():
                    raise ConfigurationError(f"Security initialization failed: {e}")
        
        # Initialize monitoring if enabled
        if self.config.enable_monitoring:
            try:
                monitoring_system = get_monitoring_system()
                if monitoring_system:
                    self.metrics = monitoring_system.get_agent_metrics(self.agent_id)
                    self.logger.info("Monitoring initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize monitoring: {e}")
    
    @track_performance("sdk_start")
    async def start(self, auth_token: Optional[str] = None) -> bool:
        """Start the enhanced SDK client"""
        try:
            self.logger.info(f"Starting enhanced SDK client for agent {self.agent_id}")
            
            # Set auth token
            if auth_token:
                self.config.auth_token = auth_token
            
            # Connect to database
            if self.database_manager:
                await self.database_manager.connect()
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=self.config.http_timeout)
            
            # Set state
            self.state = AgentState.CONNECTING
            self.running = True
            
            # Register with lobby
            if not await self._register_with_lobby():
                raise ConnectionError("Failed to register with lobby")
            
            # Connect WebSocket
            if not await self._connect_websocket():
                raise ConnectionError("Failed to connect WebSocket")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update state and metrics
            self.state = AgentState.REGISTERED
            if self.metrics:
                self.metrics.set_connection_status(True)
            
            self.logger.info(f"Enhanced SDK client started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start SDK client: {e}")
            self.state = AgentState.ERROR
            if self.metrics:
                self.metrics.record_error("start_failed")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the enhanced SDK client"""
        try:
            self.logger.info("Stopping enhanced SDK client")
            self.running = False
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Close WebSocket connection
            if self.ws_connection:
                await self.ws_connection.close()
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
            
            # Disconnect from database
            if self.database_manager:
                await self.database_manager.disconnect()
            
            # Update metrics
            if self.metrics:
                self.metrics.set_connection_status(False)
            
            self.state = AgentState.DISCONNECTED
            self.logger.info("Enhanced SDK client stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping SDK client: {e}")
    
    async def _register_with_lobby(self) -> bool:
        """Register agent with lobby"""
        try:
            # Prepare registration data
            registration_data = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "capabilities": [cap.to_dict() for cap in self.capabilities]
            }
            
            # Add auth token if available
            headers = {}
            if self.config.auth_token:
                headers["Authorization"] = f"Bearer {self.config.auth_token}"
            
            # Send registration request
            response = await self.http_client.post(
                f"{self.config.lobby_http_url}/register",
                json=registration_data,
                headers=headers
            )
            
            if response.status_code == 200:
                self.logger.info("Successfully registered with lobby")
                
                # Store registration in database
                if self.database_manager:
                    agent_registration = AgentRegistration(
                        agent_id=self.agent_id,
                        agent_type=self.agent_type,
                        capabilities=[cap.to_dict() for cap in self.capabilities]
                    )
                    await self.database_manager.register_agent(agent_registration)
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_capability_usage("registration")
                
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            if self.metrics:
                self.metrics.record_error("registration_failed")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect to lobby WebSocket"""
        try:
            # Build WebSocket URL with auth token
            ws_url = f"{self.config.lobby_ws_url}/ws/{self.agent_id}"
            if self.config.auth_token:
                ws_url += f"?token={self.config.auth_token}"
            
            # Connect to WebSocket
            self.ws_connection = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.logger.info("WebSocket connected")
            self.state = AgentState.CONNECTED
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            if self.metrics:
                self.metrics.record_error("websocket_connection_failed")
            return False
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        # WebSocket listener
        task = asyncio.create_task(self._ws_listener())
        self.tasks.add(task)
        
        # WebSocket sender
        task = asyncio.create_task(self._ws_sender())
        self.tasks.add(task)
        
        # Message processor
        task = asyncio.create_task(self._message_processor())
        self.tasks.add(task)
        
        # Heartbeat sender
        if self.config.heartbeat_interval > 0:
            task = asyncio.create_task(self._heartbeat_sender())
            self.tasks.add(task)
        
        # Auto-reconnect handler
        if self.config.auto_reconnect:
            task = asyncio.create_task(self._reconnect_handler())
            self.tasks.add(task)
        
        # Failed message retry handler
        task = asyncio.create_task(self._retry_failed_messages())
        self.tasks.add(task)
    
    async def _ws_listener(self):
        """Listen for WebSocket messages"""
        try:
            while self.running and self.ws_connection:
                try:
                    raw_message = await asyncio.wait_for(
                        self.ws_connection.recv(),
                        timeout=self.config.websocket_timeout
                    )
                    
                    # Parse message
                    message_data = json.loads(raw_message)
                    message = Message.from_dict(message_data)
                    
                    # Record metrics
                    if self.metrics:
                        self.metrics.record_message_received(
                            message.message_type.name,
                            message.sender_id
                        )
                    
                    # Store message in database
                    if self.database_manager:
                        message_record = MessageRecord(
                            message_id=message.message_id,
                            sender_id=message.sender_id,
                            receiver_id=message.receiver_id,
                            message_type=message.message_type.name,
                            payload=message.payload,
                            conversation_id=message.conversation_id,
                            timestamp=message.timestamp
                        )
                        await self.database_manager.store_message(message_record)
                    
                    # Add to processing queue
                    await self.message_queue.put(message)
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    self.logger.error(f"Error in WebSocket listener: {e}")
                    if self.metrics:
                        self.metrics.record_error("websocket_listener_error")
                    
        except Exception as e:
            self.logger.error(f"WebSocket listener failed: {e}")
    
    async def _ws_sender(self):
        """Send WebSocket messages"""
        try:
            while self.running and self.ws_connection:
                try:
                    # Get message from outbound queue
                    message = await asyncio.wait_for(
                        self.outbound_queue.get(),
                        timeout=1.0
                    )
                    
                    # Send message
                    await self.ws_connection.send(json.dumps(message.to_dict()))
                    
                    # Record metrics
                    if self.metrics:
                        self.metrics.record_message_sent(
                            message.message_type.name,
                            message.receiver_id
                        )
                    
                    # Store message in database
                    if self.database_manager:
                        message_record = MessageRecord(
                            message_id=message.message_id,
                            sender_id=message.sender_id,
                            receiver_id=message.receiver_id,
                            message_type=message.message_type.name,
                            payload=message.payload,
                            conversation_id=message.conversation_id,
                            timestamp=message.timestamp
                        )
                        await self.database_manager.store_message(message_record)
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    self.logger.error(f"Error sending message: {e}")
                    if self.metrics:
                        self.metrics.record_error("message_send_error")
                    
                    # Add message to failed queue for retry
                    await self.message_queue.add_failed_message(message)
                    
        except Exception as e:
            self.logger.error(f"WebSocket sender failed: {e}")
    
    @track_performance("message_processing")
    async def _message_processor(self):
        """Process incoming messages"""
        try:
            while self.running:
                try:
                    # Get message from queue
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                    
                    start_time = time.time()
                    
                    # Handle special message types
                    if message.message_type == MessageType.RESPONSE:
                        await self.interaction_manager.resolve_request(message)
                    elif message.message_type == MessageType.ACK:
                        # Handle acknowledgment
                        self.logger.debug(f"Received ACK for message: {message.payload.get('ack_for')}")
                    else:
                        # Process with user handler
                        try:
                            response = await self.agent_message_handler(message)
                            if response:
                                await self.send_message(response)
                        except Exception as e:
                            self.logger.error(f"Error in message handler: {e}")
                            if self.metrics:
                                self.metrics.record_error("message_handler_error")
                    
                    # Record processing time
                    processing_time = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_message_processing_time(
                            message.message_type.name,
                            processing_time
                        )
                    
                    # Update last activity
                    if self.database_manager:
                        await self.database_manager.update_agent_last_seen(self.agent_id)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    if self.metrics:
                        self.metrics.record_error("message_processing_error")
                    
        except Exception as e:
            self.logger.error(f"Message processor failed: {e}")
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat messages"""
        try:
            while self.running:
                try:
                    await asyncio.sleep(self.config.heartbeat_interval)
                    
                    if self.state == AgentState.REGISTERED:
                        heartbeat_message = Message(
                            sender_id=self.agent_id,
                            receiver_id="lobby",
                            message_type=MessageType.HEALTH_CHECK,
                            payload={"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
                        )
                        
                        await self.send_message(heartbeat_message)
                        self.last_heartbeat = datetime.now(timezone.utc)
                        
                except Exception as e:
                    self.logger.error(f"Error sending heartbeat: {e}")
                    if self.metrics:
                        self.metrics.record_error("heartbeat_error")
                    
        except Exception as e:
            self.logger.error(f"Heartbeat sender failed: {e}")
    
    async def _reconnect_handler(self):
        """Handle automatic reconnection"""
        try:
            while self.running:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self.state in [AgentState.DISCONNECTED, AgentState.ERROR]:
                    if self.connection_attempts < self.config.connection_retry_attempts:
                        self.logger.info(f"Attempting reconnection ({self.connection_attempts + 1}/{self.config.connection_retry_attempts})")
                        
                        self.connection_attempts += 1
                        
                        # Try to reconnect
                        if await self._connect_websocket():
                            self.connection_attempts = 0
                            self.state = AgentState.CONNECTED
                            self.logger.info("Reconnection successful")
                        else:
                            await asyncio.sleep(self.config.connection_retry_delay)
                    else:
                        self.logger.error("Max reconnection attempts reached")
                        break
                        
        except Exception as e:
            self.logger.error(f"Reconnect handler failed: {e}")
    
    async def _retry_failed_messages(self):
        """Retry failed messages"""
        try:
            while self.running:
                await asyncio.sleep(30)  # Retry every 30 seconds
                
                failed_messages = await self.message_queue.get_failed_messages()
                for message in failed_messages:
                    try:
                        await self.send_message(message)
                        self.logger.info(f"Retried failed message: {message.message_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to retry message {message.message_id}: {e}")
                        # Add back to failed queue
                        await self.message_queue.add_failed_message(message)
                        
        except Exception as e:
            self.logger.error(f"Failed message retry handler failed: {e}")
    
    async def send_message(self, message: Message):
        """Send a message through the SDK"""
        try:
            # Set sender ID if not set
            if not message.sender_id:
                message.sender_id = self.agent_id
            
            # Validate message
            message.validate()
            
            # Add to outbound queue
            await self.outbound_queue.put(message)
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            if self.metrics:
                self.metrics.record_error("send_message_error")
            raise
    
    @track_performance("request_response")
    async def request(self,
                     receiver_id: str,
                     payload: Dict[str, Any],
                     timeout: float = 15.0,
                     message_type: MessageType = MessageType.REQUEST) -> Message:
        """Send a request and wait for response"""
        try:
            conversation_id = str(uuid.uuid4())
            
            # Create request message
            request_message = Message(
                sender_id=self.agent_id,
                receiver_id=receiver_id,
                message_type=message_type,
                payload=payload,
                conversation_id=conversation_id,
                requires_ack=True,
                ack_timeout=timeout
            )
            
            # Register for response
            response_future = await self.interaction_manager.register_request(conversation_id)
            
            # Start interaction tracking
            interaction_id = await self.interaction_manager.start_interaction(
                target_id=receiver_id,
                interaction_type="request_response",
                metadata={"request_type": message_type.name}
            )
            
            # Send request
            await self.send_message(request_message)
            
            try:
                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout)
                
                # Complete interaction
                await self.interaction_manager.complete_interaction(
                    interaction_id,
                    {"response_received": True, "response_id": response.message_id}
                )
                
                return response
                
            except asyncio.TimeoutError:
                # Complete interaction with timeout
                await self.interaction_manager.complete_interaction(
                    interaction_id,
                    {"response_received": False, "error": "timeout"},
                    status="timeout"
                )
                raise
                
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            if self.metrics:
                self.metrics.record_error("request_error")
            raise
    
    async def discover_services(self, 
                               capability_name: Optional[str] = None,
                               agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover available services"""
        try:
            payload = {}
            if capability_name:
                payload["capability_name"] = capability_name
            if agent_type:
                payload["agent_type"] = agent_type
            
            response = await self.request(
                receiver_id="lobby",
                payload=payload,
                message_type=MessageType.DISCOVER_SERVICES
            )
            
            return response.payload.get("services", [])
            
        except Exception as e:
            self.logger.error(f"Service discovery failed: {e}")
            if self.metrics:
                self.metrics.record_error("service_discovery_error")
            return []
    
    async def advertise_capabilities(self):
        """Advertise agent capabilities to lobby"""
        try:
            payload = {
                "capabilities": [cap.to_dict() for cap in self.capabilities]
            }
            
            message = Message(
                sender_id=self.agent_id,
                receiver_id="lobby",
                message_type=MessageType.ADVERTISE_CAPABILITIES,
                payload=payload
            )
            
            await self.send_message(message)
            
            if self.metrics:
                self.metrics.record_capability_usage("advertise_capabilities")
                
        except Exception as e:
            self.logger.error(f"Failed to advertise capabilities: {e}")
            if self.metrics:
                self.metrics.record_error("advertise_capabilities_error")
    
    async def report_task_outcome(self,
                                 provider_agent_id: str,
                                 capability_name: str,
                                 status: str,
                                 details: str,
                                 original_conversation_id: Optional[str] = None):
        """Report task outcome to lobby"""
        try:
            payload = {
                "provider_agent_id": provider_agent_id,
                "capability_name": capability_name,
                "status": status,
                "details": details,
                "reporter_agent_id": self.agent_id
            }
            
            message = Message(
                sender_id=self.agent_id,
                receiver_id="lobby",
                message_type=MessageType.TASK_OUTCOME_REPORT,
                payload=payload,
                conversation_id=original_conversation_id
            )
            
            await self.send_message(message)
            
            if self.metrics:
                self.metrics.record_capability_usage("task_outcome_report")
                
        except Exception as e:
            self.logger.error(f"Failed to report task outcome: {e}")
            if self.metrics:
                self.metrics.record_error("task_outcome_report_error")
    
    async def get_agent_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get agent statistics"""
        try:
            if self.database_manager:
                return await self.database_manager.get_agent_stats(self.agent_id, hours)
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get agent stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            health_data = {
                "agent_id": self.agent_id,
                "state": self.state.value,
                "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                "message_queue_size": self.message_queue.qsize(),
                "outbound_queue_size": self.outbound_queue.qsize(),
                "connection_attempts": self.connection_attempts,
                "running": self.running
            }
            
            # Add database health if available
            if self.database_manager:
                db_health = await self.database_manager.health_check()
                health_data["database"] = db_health
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}


# Factory function for easy client creation
def create_enhanced_client(
    agent_id: str,
    agent_type: str,
    capabilities: List[AgentCapability],
    agent_message_handler: Callable[[Message], Awaitable[Optional[Message]]],
    config_file: Optional[str] = None,
    **config_overrides
) -> EnhancedEcosystemClient:
    """Factory function to create an enhanced ecosystem client"""
    
    # Initialize configuration if needed
    if config_file:
        initialize_config(config_dir=str(Path(config_file).parent))
    
    # Get ecosystem configuration
    ecosystem_config = get_config()
    
    # Create SDK configuration
    sdk_config = SDKConfig(**config_overrides)
    
    # Create and return client
    return EnhancedEcosystemClient(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities,
        agent_message_handler=agent_message_handler,
        config=sdk_config,
        ecosystem_config=ecosystem_config
    ) 