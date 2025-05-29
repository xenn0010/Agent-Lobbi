"""
Agent Lobby Python SDK - Simple Integration for AI Agents

Example usage:
    from agent_lobby import Agent, Capability
    
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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto

try:
    import websockets
    import httpx
except ImportError as e:
    raise ImportError(f"Missing required dependency: {e.name}. Install with: pip install websockets httpx")

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
    """Main Agent class for integrating with the Agent Lobby ecosystem."""
    
    def __init__(
        self,
        api_key: str,
        agent_type: str,
        capabilities: List[Capability],
        agent_id: Optional[str] = None,
        lobby_url: str = "http://localhost:8092",
        debug: bool = False
    ):
        self.api_key = api_key
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.agent_id = agent_id or f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.lobby_url = lobby_url
        self.debug = debug
        
        self._auth_token: Optional[str] = None
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._message_handlers: Dict[str, Callable] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"Agent {self.agent_id} ({self.agent_type}) initialized")
    
    def on_message(self, func: Callable[[Message], Union[Dict[str, Any], None]]):
        """Decorator to register a message handler."""
        self._message_handlers["default"] = func
        return func
    
    async def start(self) -> bool:
        """Start the agent and connect to the Agent Lobby."""
        try:
            logger.info(f"Starting agent {self.agent_id}...")
            
            if not await self._register_with_lobby():
                return False
            
            if not await self._connect_websocket():
                return False
            
            self._start_background_tasks()
            self._running = True
            logger.info(f"Agent {self.agent_id} started successfully! 🚀")
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
        self._http_client = httpx.AsyncClient(timeout=30.0)
        
        registration_data = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender_id": self.agent_id,
            "receiver_id": "lobby",
            "message_type": "REGISTER",
            "payload": {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "capabilities": [cap.to_dict() for cap in self.capabilities]
            },
            "conversation_id": None,
            "auth_token": None,
            "priority": 2,
            "requires_ack": False,
            "ack_timeout": None,
            "broadcast_scope": None
        }
        
        try:
            response = await self._http_client.post(
                f"{self.lobby_url}/api/register",
                json=registration_data,
                headers={"X-API-Key": self.api_key}
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("payload", {}).get("status") == "success":
                self._auth_token = data["payload"]["auth_token"]
                logger.info("Successfully registered with lobby")
                return True
            else:
                logger.error(f"Registration failed: {data}")
                return False
                
        except Exception as e:
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
            
            if result is not None and message.message_type == MessageType.REQUEST:
                response = Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    payload=result,
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

__all__ = ["Agent", "Capability", "Message", "MessageType", "SDKConfig", "EnhancedEcosystemClient"]