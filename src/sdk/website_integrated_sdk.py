"""
Agent Lobbi SDK with Website Integration
Enhanced SDK that tracks activity and integrates with the Agent Lobbi website dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, IntEnum
from typing import Any, Dict, Optional, List, Set, Callable, Awaitable
import uuid
import json
import asyncio
import httpx
import websockets
import sys
import logging

# Copy message types from core
class MessageType(Enum):
    REGISTER = auto()
    REGISTER_ACK = auto()
    REQUEST = auto()
    RESPONSE = auto()
    ACTION = auto()
    INFO = auto()
    ERROR = auto()
    DISCOVER_SERVICES = auto()
    SERVICES_AVAILABLE = auto()
    ADVERTISE_CAPABILITIES = auto()
    BROADCAST = auto()
    ACK = auto()
    HEALTH_CHECK = auto()
    CAPABILITY_UPDATE = auto()
    TASK_OUTCOME_REPORT = auto()
    TASK_ASSIGNMENT = auto()  # For task assignment to agents

class MessagePriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BULK = 4

@dataclass
class Capability:
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    handler: Optional[Callable] = None

class WebsiteIntegratedClient:
    """Enhanced Agent Lobbi client with website integration for tracking and analytics."""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str = "Agent",
        api_key: str = None,
        lobby_url: str = "http://localhost:8092",
        websocket_url: str = "ws://localhost:8091",
        website_url: str = "http://localhost:5000",
        auto_track: bool = True,
        capabilities: List[Capability] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.api_key = api_key
        self.lobby_url = lobby_url
        self.websocket_url = websocket_url
        self.website_url = website_url
        self.auto_track = auto_track
        
        # Core functionality
        self.capabilities = capabilities or []
        self.is_running = False
        self.websocket = None
        self.http_client = None
        self.auth_token = None
        
        # Tracking and analytics
        self.session_start = None
        self.request_count = 0
        self.error_count = 0
        self.collaboration_count = 0
        
        # Event handlers
        self.message_handlers = {}
        self.capability_handlers = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"agent_lobby_sdk.{agent_id}")
        
    async def start(self) -> bool:
        """Start the agent and connect to Agent Lobbi."""
        try:
            self.session_start = datetime.now(timezone.utc)
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Register with lobby
            if not await self._register_with_lobby():
                self.logger.error("Failed to register with lobby")
                return False
            
            # Connect WebSocket
            if not await self._connect_websocket():
                self.logger.error("Failed to connect WebSocket")
                return False
            
            self.is_running = True
            
            # Track startup
            if self.auto_track:
                await self._track_activity("agent_started", {
                    "startup_time": self.session_start.isoformat(),
                    "capabilities_count": len(self.capabilities)
                })
            
            self.logger.info(f" Agent {self.agent_id} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start agent: {e}")
            return False
    
    async def stop(self):
        """Stop the agent and cleanup connections."""
        self.is_running = False
        
        # Track shutdown
        if self.auto_track and self.session_start:
            uptime = datetime.now(timezone.utc) - self.session_start
            await self._track_activity("agent_stopped", {
                "uptime_seconds": uptime.total_seconds(),
                "requests_handled": self.request_count,
                "errors_encountered": self.error_count,
                "collaborations": self.collaboration_count
            })
        
        # Close connections
        if self.websocket:
            await self.websocket.close()
        
        if self.http_client:
            await self.http_client.aclose()
        
        self.logger.info(f"👋 Agent {self.agent_id} stopped")
    
    async def _register_with_lobby(self) -> bool:
        """Register the agent with the lobby."""
        try:
            registration_data = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "input_schema": cap.input_schema or {},
                        "output_schema": cap.output_schema or {}
                    }
                    for cap in self.capabilities
                ]
            }
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = await self.http_client.post(
                f"{self.lobby_url}/api/register",
                json=registration_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                self.auth_token = result.get("auth_token")
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect to the lobby WebSocket."""
        try:
            ws_url = f"{self.websocket_url}/ws/{self.agent_id}"
            if self.auth_token:
                ws_url += f"?token={self.auth_token}"
            
            self.websocket = await websockets.connect(ws_url)
            
            # Start message listener
            asyncio.create_task(self._message_listener())
            
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            return False
    
    async def _message_listener(self):
        """Listen for incoming messages from the lobby."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages."""
        message_type = data.get("message_type")
        
        # Track incoming requests
        if message_type == "REQUEST":
            self.request_count += 1
            if self.auto_track:
                await self._track_activity("request_received", {
                    "capability": data.get("payload", {}).get("capability"),
                    "sender": data.get("sender_id")
                })
        
        # Handle capability requests
        if message_type == "REQUEST":
            capability_name = data.get("payload", {}).get("capability")
            if capability_name in self.capability_handlers:
                try:
                    result = await self.capability_handlers[capability_name](
                        data.get("payload", {}).get("data", {})
                    )
                    
                    # Send response
                    await self._send_response(data, result)
                    
                    # Track successful execution
                    if self.auto_track:
                        await self._track_activity("capability_executed", {
                            "capability": capability_name,
                            "success": True
                        })
                        
                except Exception as e:
                    self.error_count += 1
                    self.logger.error(f"Error executing capability {capability_name}: {e}")
                    
                    # Send error response
                    await self._send_error_response(data, str(e))
                    
                    # Track error
                    if self.auto_track:
                        await self._track_activity("capability_error", {
                            "capability": capability_name,
                            "error": str(e)
                        }, status="error")
    
    async def _send_response(self, original_message: Dict[str, Any], result: Any):
        """Send a response message."""
        response = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender_id": self.agent_id,
            "receiver_id": original_message.get("sender_id"),
            "message_type": "RESPONSE",
            "conversation_id": original_message.get("conversation_id"),
            "payload": {"result": result}
        }
        
        await self.websocket.send(json.dumps(response))
    
    async def _send_error_response(self, original_message: Dict[str, Any], error: str):
        """Send an error response message."""
        response = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender_id": self.agent_id,
            "receiver_id": original_message.get("sender_id"),
            "message_type": "ERROR",
            "conversation_id": original_message.get("conversation_id"),
            "payload": {"error": error}
        }
        
        await self.websocket.send(json.dumps(response))
    
    async def _track_activity(self, action: str, details: Dict[str, Any] = None, status: str = "success"):
        """Track activity with the website dashboard."""
        if not self.auto_track or not self.api_key:
            return
        
        try:
            tracking_data = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "action": action,
                "details": details or {},
                "status": status
            }
            
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.website_url}/api/track-agent",
                    json=tracking_data,
                    headers=headers,
                    timeout=5.0
                )
                
                if response.status_code != 200:
                    self.logger.debug(f"Activity tracking failed: {response.status_code}")
                    
        except Exception as e:
            self.logger.debug(f"Activity tracking error: {e}")
    
    def add_capability(self, name: str, handler: Callable, description: str = "", 
                      input_schema: Dict = None, output_schema: Dict = None):
        """Add a capability to the agent."""
        capability = Capability(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            handler=handler
        )
        
        self.capabilities.append(capability)
        self.capability_handlers[name] = handler
        
        # Track capability addition
        if self.auto_track and self.is_running:
            asyncio.create_task(self._track_activity("capability_added", {
                "capability_name": name,
                "description": description
            }))
    
    def capability(self, name: str, description: str = "", 
                  input_schema: Dict = None, output_schema: Dict = None):
        """Decorator to register a capability."""
        def decorator(func: Callable):
            self.add_capability(name, func, description, input_schema, output_schema)
            return func
        return decorator
    
    async def request_capability(self, target_agent: str, capability: str, data: Dict[str, Any]) -> Any:
        """Request a capability from another agent."""
        if not self.is_running:
            raise RuntimeError("Agent is not running")
        
        message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender_id": self.agent_id,
            "receiver_id": target_agent,
            "message_type": "REQUEST",
            "conversation_id": str(uuid.uuid4()),
            "payload": {
                "capability": capability,
                "data": data
            }
        }
        
        # Send request
        await self.websocket.send(json.dumps(message))
        
        # Track collaboration
        self.collaboration_count += 1
        if self.auto_track:
            await self._track_activity("collaboration_initiated", {
                "target_agent": target_agent,
                "capability": capability
            })
        
        # Note: In a full implementation, you'd wait for and return the response
        # For now, just track the request
        return {"status": "request_sent"}
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover available agents and their capabilities."""
        if not self.is_running:
            raise RuntimeError("Agent is not running")
        
        message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sender_id": self.agent_id,
            "receiver_id": "lobby",
            "message_type": "DISCOVER_SERVICES"
        }
        
        await self.websocket.send(json.dumps(message))
        
        if self.auto_track:
            await self._track_activity("discovery_request", {})
        
        # Note: In a full implementation, you'd wait for and return the response
        return []
    
    async def run_forever(self):
        """Keep the agent running indefinitely."""
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        finally:
            await self.stop()

# Convenience function for creating agents
def create_agent(agent_id: str, agent_type: str = "Agent", api_key: str = None, **kwargs) -> WebsiteIntegratedClient:
    """Create a new Agent Lobbi client with website integration."""
    return WebsiteIntegratedClient(
        agent_id=agent_id,
        agent_type=agent_type,
        api_key=api_key,
        **kwargs
    )

# Export the main class
EcosystemClient = WebsiteIntegratedClient 