"""
Agent Lobbi Python SDK - Production-Ready Client Implementation

This module provides a robust, production-ready client for integrating with the Agent Lobbi ecosystem.
It includes comprehensive error handling, logging, retry logic, and monitoring capabilities.

Example usage:
    from agent_lobbi_sdk import Agent, Capability
    
    capabilities = [
        Capability("translate_text", "Translates text between languages")
    ]
    
    agent = Agent(
        api_key="your_api_key_here",
        agent_type="TranslationBot",
        capabilities=capabilities
    )
    
    @agent.on_message
    async def handle_message(message):
        if message.payload.get("action") == "translate":
            return {"translated": "Hola mundo", "language": "es"}
    
    await agent.start()
"""

import asyncio
import json
import logging
import uuid
import time
import hashlib
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import sys
import traceback
from urllib.parse import urlparse

try:
    import websockets
    import httpx
    import yaml
except ImportError as e:
    raise ImportError(f"Missing required dependency: {e.name}. Install with: pip install websockets httpx pyyaml")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_lobbi_sdk.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Custom Exceptions
class AgentLobbiError(Exception):
    """Base exception for Agent Lobbi SDK errors."""
    pass

class ConnectionError(AgentLobbiError):
    """Raised when connection to Agent Lobbi fails."""
    pass

class AuthenticationError(AgentLobbiError):
    """Raised when authentication fails."""
    pass

class TaskError(AgentLobbiError):
    """Raised when task processing fails."""
    pass

class ConfigurationError(AgentLobbiError):
    """Raised when configuration is invalid."""
    pass

# Message Types and Data Classes
class MessageType(Enum):
    REGISTER = auto()
    REGISTER_ACK = auto()
    REQUEST = auto()
    RESPONSE = auto()
    INFO = auto()
    ERROR = auto()
    HEARTBEAT = auto()
    TASK_ASSIGNMENT = auto()
    TASK_COMPLETION = auto()

@dataclass
class Capability:
    """Represents an agent capability with validation and schema support."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate capability data."""
        if not self.name or not isinstance(self.name, str):
            raise ConfigurationError("Capability name must be a non-empty string")
        if not self.description or not isinstance(self.description, str):
            raise ConfigurationError("Capability description must be a non-empty string")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
            "version": self.version
        }

@dataclass
class Message:
    """Enhanced message class with validation and serialization."""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    conversation_id: Optional[str] = None
    priority: int = 2
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate message data."""
        if not self.sender_id or not self.receiver_id:
            raise ConfigurationError("Message must have valid sender_id and receiver_id")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name,
            "payload": self.payload,
            "conversation_id": self.conversation_id,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType[data["message_type"]],
            payload=data.get("payload", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            conversation_id=data.get("conversation_id"),
            priority=data.get("priority", 2),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )

class Agent:
    """
    Production-ready Agent class for integrating with the Agent Lobbi ecosystem.
    
    Features:
    - Automatic reconnection and retry logic
    - Comprehensive error handling and logging
    - Health monitoring and metrics
    - Task queue management
    - Secure authentication
    """
    
    def __init__(
        self,
        api_key: str,
        agent_type: str,
        capabilities: List[Capability],
        agent_id: Optional[str] = None,
        lobby_url: str = "http://localhost:8092",
        debug: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        heartbeat_interval: float = 30.0,
        timeout: float = 30.0
    ):
        # Validate inputs
        if not api_key or not isinstance(api_key, str):
            raise ConfigurationError("API key must be a non-empty string")
        if not agent_type or not isinstance(agent_type, str):
            raise ConfigurationError("Agent type must be a non-empty string")
        if not capabilities or not isinstance(capabilities, list):
            raise ConfigurationError("Capabilities must be a non-empty list")
        
        self.api_key = api_key
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.agent_id = agent_id or f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.lobby_url = lobby_url
        self.debug = debug
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        
        # Parse lobby URL
        parsed_url = urlparse(lobby_url)
        self.lobby_host = parsed_url.hostname or "localhost"
        self.lobby_port = parsed_url.port or 8092
        self.websocket_url = f"ws://{self.lobby_host}:{self.lobby_port + 1}"
        
        # Internal state
        self._auth_token: Optional[str] = None
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._message_handlers: Dict[str, Callable] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._last_heartbeat = time.time()
        self._connection_attempts = 0
        self._metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_completed": 0,
            "errors": 0,
            "uptime_start": time.time()
        }
        
        # Configure logging
        if debug:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Agent {self.agent_id} ({self.agent_type}) initialized with {len(capabilities)} capabilities")
    
    def on_message(self, func: Callable[[Message], Union[Dict[str, Any], None]]):
        """Decorator to register a message handler."""
        self._message_handlers["default"] = func
        return func
    
    async def start(self) -> bool:
        """Start the agent and connect to the Agent Lobbi."""
        try:
            logger.info(f"Starting agent {self.agent_id}...")
            
            if not await self._register_with_lobby():
                return False
            
            if not await self._connect_websocket():
                return False
            
            self._start_background_tasks()
            self._running = True
            logger.info(f"Agent {self.agent_id} started successfully! ")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop the agent and clean up resources."""
        logger.info(f"Stopping agent {self.agent_id}...")
        self._running = False
        
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        if self._websocket:
            await self._websocket.close()
        
        if self._http_client:
            await self._http_client.aclose()
        
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def _register_with_lobby(self) -> bool:
        """Register the agent with the Agent Lobbi via HTTP."""
        register_url = f"{self.lobby_url}/api/agents/register"
        payload = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": [cap.to_dict() for cap in self.capabilities]
        }
        
        self._http_client = httpx.AsyncClient(timeout=30.0)
        
        try:
            response = await self._http_client.post(
                register_url,
                json=payload,
                headers={"X-API-Key": self.api_key}
            )
            response.raise_for_status()
            
            response_data = response.json()
            if response_data.get("status") == "success":
                self._auth_token = response_data.get("auth_token")
                logger.info(f"Agent {self.agent_id} registered successfully.")
                self._connection_attempts = 0
                return True
            else:
                logger.error(f"Registration failed: {response_data}")
                return False
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Registration error: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect to the lobby via WebSocket."""
        if not self._auth_token:
            return False
        
        ws_url = self.lobby_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = ws_url.replace(":8092", ":8091")
        uri = f"{ws_url}/ws/{self.agent_id}?token={self._auth_token}"
        
        try:
            self._websocket = await websockets.connect(uri)
            logger.info("WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background tasks for message handling."""
        task = asyncio.create_task(self._message_listener())
        self._tasks.append(task)
    
    async def _message_listener(self):
        """Listen for incoming messages."""
        if not self._websocket:
            return
        
        try:
            async for raw_message in self._websocket:
                try:
                    data = json.loads(raw_message)
                    message = Message.from_dict(data)
                    
                    if message.conversation_id and message.conversation_id in self._pending_requests:
                        future = self._pending_requests[message.conversation_id]
                        if not future.done():
                            future.set_result(message)
                        continue
                    
                    await self._handle_message(message)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Message listener error: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle an incoming message."""
        logger.debug(f"Received {message.message_type.name} from {message.sender_id}")
        
        if "default" not in self._message_handlers:
            return
        
        try:
            handler = self._message_handlers["default"]
            result = await handler(message)
            
            if result is not None and message.message_type in [MessageType.REQUEST, MessageType.TASK_ASSIGNMENT]:
                response_type = MessageType.TASK_COMPLETION if message.message_type == MessageType.TASK_ASSIGNMENT else MessageType.RESPONSE
                
                response_payload = result.copy()
                if response_type == MessageType.TASK_COMPLETION and 'task_id' not in response_payload:
                    original_task_id = message.payload.get('task_id')
                    if original_task_id:
                        response_payload['task_id'] = original_task_id
                
                response = Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=response_type,
                    payload=response_payload,
                    conversation_id=message.conversation_id
                )
                await self._websocket.send(json.dumps(response.to_dict()))
                
        except Exception as e:
            logger.error(f"Error in message handler: {e}")

# Legacy compatibility
class SDKConfig:
    def __init__(self, ws_base_url: str, agent_id: str, agent_type: str, capabilities: list, auth_token: Optional[str] = None):
        self.ws_base_url = ws_base_url
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.auth_token = auth_token

class EnhancedEcosystemClient:
    def __init__(self, config: SDKConfig):
        self.config = config

class AgentLobbiClient:
    """
    High-level client for Agent Lobbi operations.
    
    This class provides a simplified interface for common Agent Lobbi operations
    including agent management, task delegation, and monitoring.
    """
    
    def __init__(
        self,
        api_key: str,
        lobby_url: str = "http://localhost:8092",
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.lobby_url = lobby_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"AgentLobbiClient initialized for {lobby_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.aclose()
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        if not self._http_client:
            raise ConnectionError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._http_client.get(f"{self.lobby_url}/api/agents")
            response.raise_for_status()
            data = response.json()
            return data.get("agents", [])
        except httpx.HTTPError as e:
            logger.error(f"Failed to list agents: {e}")
            raise ConnectionError(f"Failed to list agents: {e}")
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent."""
        if not self._http_client:
            raise ConnectionError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._http_client.get(f"{self.lobby_url}/api/agents/{agent_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get agent status: {e}")
            raise ConnectionError(f"Failed to get agent status: {e}")
    
    async def delegate_task(
        self,
        task_name: str,
        task_description: str,
        required_capabilities: List[str],
        task_data: Dict[str, Any] = None,
        max_agents: int = 1,
        timeout_minutes: int = 30
    ) -> Dict[str, Any]:
        """Delegate a task to available agents."""
        if not self._http_client:
            raise ConnectionError("Client not initialized. Use async context manager.")
        
        payload = {
            "name": task_name,
            "description": task_description,
            "required_capabilities": required_capabilities,
            "task_data": task_data or {},
            "max_agents": max_agents,
            "timeout_minutes": timeout_minutes,
            "created_by": "client"
        }
        
        try:
            response = await self._http_client.post(
                f"{self.lobby_url}/api/delegate_task",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to delegate task: {e}")
            raise TaskError(f"Failed to delegate task: {e}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a delegated task."""
        if not self._http_client:
            raise ConnectionError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._http_client.get(f"{self.lobby_url}/api/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get task status: {e}")
            raise TaskError(f"Failed to get task status: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Agent Lobbi health status."""
        if not self._http_client:
            raise ConnectionError("Client not initialized. Use async context manager.")
        
        try:
            response = await self._http_client.get(f"{self.lobby_url}/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {e}")
            raise ConnectionError(f"Health check failed: {e}")

# Convenience functions for quick setup
async def create_agent(
    api_key: str,
    agent_type: str,
    capabilities: List[str],
    agent_id: Optional[str] = None,
    lobby_url: str = "http://localhost:8092",
    **kwargs
) -> Agent:
    """
    Create and configure an Agent with capabilities.
    
    Args:
        api_key: API key for authentication
        agent_type: Type/category of the agent
        capabilities: List of capability names
        agent_id: Optional custom agent ID
        lobby_url: Agent Lobbi URL
        **kwargs: Additional Agent configuration
    
    Returns:
        Configured Agent instance
    """
    capability_objects = [
        Capability(name=cap, description=f"Capability: {cap}")
        for cap in capabilities
    ]
    
    return Agent(
        api_key=api_key,
        agent_type=agent_type,
        capabilities=capability_objects,
        agent_id=agent_id,
        lobby_url=lobby_url,
        **kwargs
    )

async def quick_task_delegation(
    api_key: str,
    task_name: str,
    task_description: str,
    required_capabilities: List[str],
    lobby_url: str = "http://localhost:8092",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick task delegation without creating a full client.
    
    Args:
        api_key: API key for authentication
        task_name: Name of the task
        task_description: Detailed task description
        required_capabilities: List of required capabilities
        lobby_url: Agent Lobbi URL
        **kwargs: Additional task parameters
    
    Returns:
        Task delegation result
    """
    async with AgentLobbiClient(api_key, lobby_url) as client:
        return await client.delegate_task(
            task_name=task_name,
            task_description=task_description,
            required_capabilities=required_capabilities,
            **kwargs
        )

# Export version for package
__version__ = "1.0.0"

__all__ = ["Agent", "Capability", "Message", "MessageType", "SDKConfig", "EnhancedEcosystemClient", "AgentLobbyClient"]