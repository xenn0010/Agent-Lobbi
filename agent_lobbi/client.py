"""
Agent Lobbi Simple Client
=========================
Simplified client for basic agent integration without advanced features.
Perfect for getting started quickly.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    import websockets
    import aiohttp
except ImportError as e:
    raise ImportError(f"Missing required dependency: {e.name}. Install with: pip install agent-lobby[basic]")

logger = logging.getLogger(__name__)

class MessageType(Enum):
    REGISTER = auto()
    REGISTER_ACK = auto()
    REQUEST = auto()
    RESPONSE = auto()
    INFO = auto()
    ERROR = auto()

@dataclass
class Capability:
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        }

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    conversation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name,
            "payload": self.payload,
            "conversation_id": self.conversation_id,
            "auth_token": None,
            "priority": 2,
            "requires_ack": False,
            "ack_timeout": None,
            "broadcast_scope": None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType[data["message_type"]],
            payload=data.get("payload", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            conversation_id=data.get("conversation_id")
        )

class Agent:
    """Simple Agent class for basic Agent Lobbi integration."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[Union[Capability, str]],
        lobby_host: str = "localhost",
        lobby_port: int = 8098,
        debug: bool = False
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.lobby_host = lobby_host
        self.lobby_port = lobby_port
        self.debug = debug
        
        # Convert string capabilities to Capability objects
        self.capabilities = []
        for cap in capabilities:
            if isinstance(cap, str):
                self.capabilities.append(Capability(cap, f"Capability: {cap}"))
            else:
                self.capabilities.append(cap)
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._http_client: Optional[aiohttp.ClientSession] = None
        self._message_handlers: Dict[str, Callable] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Simple Agent {self.agent_id} ({self.agent_type}) initialized")
    
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
            logger.info(f"Agent {self.agent_id} started successfully!")
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
        """Register with the lobby via HTTP."""
        self._http_client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30.0))
        
        registration_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "goal": "",
            "specialization": "",
            "capabilities": [cap.name for cap in self.capabilities]
        }
        
        try:
            async with self._http_client.post(
                f"http://{self.lobby_host}:{self.lobby_port}/api/agents/register",
                json=registration_data
            ) as response:
                if response.status == 200:
                    logger.info(f"Agent {self.agent_id} registered successfully")
                    return True
                else:
                    logger.error(f"Registration failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect to WebSocket for real-time communication."""
        try:
            ws_url = f"ws://{self.lobby_host}:8081/api/ws/{self.agent_id}"
            self._websocket = await websockets.connect(ws_url)
            logger.info(f"WebSocket connected for agent {self.agent_id}")
            return True
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background tasks for message handling."""
        if self._websocket:
            self._tasks.append(asyncio.create_task(self._message_listener()))
    
    async def _message_listener(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg = Message.from_dict(data)
                        await self._handle_message(msg)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Message listener error: {e}")
    
    async def _handle_message(self, message: Message):
        """Handle incoming messages."""
        try:
            if "default" in self._message_handlers:
                handler = self._message_handlers["default"]
                result = await handler(message) if asyncio.iscoroutinefunction(handler) else handler(message)
                
                if result:
                    # Send response back
                    response = Message(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=MessageType.RESPONSE,
                        payload=result,
                        conversation_id=message.conversation_id
                    )
                    
                    if self._websocket:
                        await self._websocket.send(json.dumps(response.to_dict()))
                        
        except Exception as e:
            logger.error(f"Error handling message: {e}")

__all__ = ["Agent", "Capability", "Message", "MessageType"] 