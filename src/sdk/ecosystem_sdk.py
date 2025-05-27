from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, IntEnum
from typing import Any, Dict, Optional, List, Set, Callable, Awaitable
import uuid
import json
from json import JSONDecodeError
import asyncio # For InteractionManager and async operations
import httpx # For HTTP communication
import websockets # For WebSocket communication
import sys # For sys.exc_info() in stop() method


# Copied from src/core/message.py - MUST BE KEPT IN SYNC
class MessagePriority(IntEnum):
    """Priority levels for message routing."""
    CRITICAL = 0    # System-critical messages (e.g., shutdown, security alerts)
    HIGH = 1        # Time-sensitive operations
    NORMAL = 2      # Standard operations
    LOW = 3         # Background tasks
    BULK = 4        # Batch operations, logging

# Copied from src/core/message.py - MUST BE KEPT IN SYNC
class MessageType(Enum):
    REGISTER = auto()
    REGISTER_ACK = auto()
    REQUEST = auto()
    RESPONSE = auto()
    ACTION = auto() # Agent wants to perform an action in the world
    INFO = auto() # Agent shares information
    ERROR = auto()
    DISCOVER_SERVICES = auto()    # Agent asks lobby to find other agents/services
    SERVICES_AVAILABLE = auto()   # Lobby responds with available services/agents
    ADVERTISE_CAPABILITIES = auto() # Agent proactively tells lobby its capabilities
    BROADCAST = auto()        # New: For messages to all agents
    ACK = auto()             # New: For message acknowledgment
    HEALTH_CHECK = auto()     # New: For agent health monitoring
    CAPABILITY_UPDATE = auto() # New: For dynamic capability updates
    TASK_OUTCOME_REPORT = auto() # New: For agents to report task success/failure to Lobby for reputation
    # Potential new types for SDK/Lobby interaction if needed for pub/sub, etc.
    # SUBSCRIBE_TOPIC = auto()
    # UNSUBSCRIBE_TOPIC = auto()
    # PUBLISH_TOPIC = auto() # Message from SDK to Lobby to publish
    # PUBLISH_EVENT = auto() # Message from Lobby to SDK with a subscribed event


# Copied from src/core/message.py - MUST BE KEPT IN SYNC
class MessageValidationError(Exception):
    """Raised when message validation fails."""
    pass

# Copied from src/core/message.py - MUST BE KEPT IN SYNC
@dataclass
class Message:
    sender_id: str
    receiver_id: str  # Can be "lobby", "broadcast", or specific agent_id
    message_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: Optional[str] = None 
    auth_token: Optional[str] = None 
    priority: MessagePriority = field(default=MessagePriority.NORMAL)
    requires_ack: bool = field(default=False)
    ack_timeout: Optional[float] = None
    broadcast_scope: Optional[Set[str]] = None
    
    def __post_init__(self):
        """Validate message after initialization."""
        # SDK-side validation might be less strict or different from Lobby's internal
        # For now, keeping it simple. Full validation can be added if needed.
        pass 
        # self.validate() # Original validation might be too much for SDK construction side before sending

    # Keeping validate() for reference, but may not be called in __post_init__ for SDK
    def validate(self) -> None:
        """Validate message contents (can be used by SDK before sending if desired)."""
        if not self.sender_id or not isinstance(self.sender_id, str):
            raise MessageValidationError("Invalid sender_id")
        
        if not self.receiver_id or not isinstance(self.receiver_id, str):
            raise MessageValidationError("Invalid receiver_id")
            
        if self.receiver_id == "broadcast" and not self.broadcast_scope:
            raise MessageValidationError("Broadcast message requires broadcast_scope")
            
        if self.requires_ack and (self.ack_timeout is None or self.ack_timeout <= 0):
            raise MessageValidationError("Acknowledgment timeout must be a positive value when requires_ack is True")
            
        try:
            json.dumps(self.payload) # Ensure payload is JSON serializable
        except (TypeError, JSONDecodeError) as e:
            raise MessageValidationError(f"Payload must be JSON serializable: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for JSON serialization."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name, # Send name for readability
            "payload": self.payload,
            "conversation_id": self.conversation_id,
            "auth_token": self.auth_token,
            "priority": self.priority.value,
            "requires_ack": self.requires_ack,
            "ack_timeout": self.ack_timeout,
            "broadcast_scope": list(self.broadcast_scope) if self.broadcast_scope else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary format (e.g., after JSON deserialization)."""
        broadcast_scope = set(data["broadcast_scope"]) if data.get("broadcast_scope") else None
        
        # Handle potential missing optional fields gracefully
        priority_val = data.get("priority")
        priority = MessagePriority(priority_val) if priority_val is not None else MessagePriority.NORMAL

        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType[data["message_type"]], # Assumes message_type name is sent
            payload=data.get("payload", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            conversation_id=data.get("conversation_id"),
            auth_token=data.get("auth_token"),
            priority=priority,
            requires_ack=data.get("requires_ack", False),
            ack_timeout=data.get("ack_timeout"),
            broadcast_scope=broadcast_scope
        )

    def create_ack(self, ack_sender_id: str) -> "Message":
        """
        Create an acknowledgment message for this message.
        ack_sender_id is the ID of the entity sending this ACK (usually the receiver of original msg).
        """
        if not self.requires_ack:
            # Optionally, log a warning or allow it, as ACK can be sent even if not strictly required
            print(f"Warning: Creating ACK for message {self.message_id} that did not explicitly require_ack.")
            # raise MessageValidationError("Cannot create acknowledgment for message that doesn't require it")
            
        return Message(
            sender_id=ack_sender_id, # The one sending the ACK
            receiver_id=self.sender_id, # ACK goes back to the original sender
            message_type=MessageType.ACK,
            payload={"ack_for": self.message_id, "status": "received"},
            conversation_id=self.conversation_id, # Preserve conversation_id
            # auth_token: self.auth_token, # Auth token might not be needed for ACK from SDK perspective or should be SDK's own
            priority=MessagePriority.HIGH # Acknowledgments are typically high priority
        )

# For Agent Capabilities in SDK registration
@dataclass
class AgentCapabilitySDK:
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)  # JSON schema for expected input
    output_schema: Dict[str, Any] = field(default_factory=dict) # JSON schema for expected output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }

# Type alias for the callback function the agent developer provides
AgentMessageHandler = Callable[[Message], Awaitable[Optional[Message]]]


class InteractionManager:
    """Handles state for ongoing conversations (e.g., request-reply)."""
    def __init__(self, sdk_client: 'EcosystemClient'): # Forward reference EcosystemClient
        self.sdk_client = sdk_client
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock() # To protect access to pending_requests

    async def register_request(self, conversation_id: str) -> asyncio.Future:
        """Registers a new pending request and returns a Future for its response."""
        async with self._lock:
            if conversation_id in self.pending_requests:
                # This shouldn't happen if conversation_ids are unique UUIDs
                raise ValueError(f"Duplicate conversation_id detected: {conversation_id}")
            future = asyncio.Future()
            self.pending_requests[conversation_id] = future
            return future

    async def resolve_request(self, response_message: Message):
        """Resolves a pending request with the given response message."""
        if not response_message.conversation_id:
            # Log this, as it's unexpected for a response in a managed interaction
            print(f"SDK - InteractionManager: Received response without conversation_id: {response_message.message_id}")
            return

        async with self._lock:
            future = self.pending_requests.get(response_message.conversation_id)
            if future and not future.done():
                future.set_result(response_message)
                # Optionally, remove from dict: del self.pending_requests[response_message.conversation_id]
                # Keeping it might be useful for debugging or re-transmissions, but can grow memory.
                # For simplicity, let's remove it after resolution.
                del self.pending_requests[response_message.conversation_id]
            elif future and future.done():
                # Response arrived after timeout or already processed
                print(f"SDK - InteractionManager: Received late/duplicate response for {response_message.conversation_id}")
            # If future is None, it means this wasn't a request initiated by our InteractionManager 
            # or it was already cleaned up after a timeout elsewhere.

    async def cancel_request(self, conversation_id: str, reason: str = "Cancelled by SDK"):
        """Cancels a pending request, e.g., due to timeout."""
        async with self._lock:
            future = self.pending_requests.get(conversation_id)
            if future and not future.done():
                future.set_exception(TimeoutError(f"Request {conversation_id} {reason}"))
                del self.pending_requests[conversation_id]


class EcosystemClient:
    """SDK Client for agents to connect and interact with the Agent Lobby Ecosystem."""
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapabilitySDK], # Use AgentCapabilitySDK
        lobby_http_url: str, # e.g., "http://localhost:8000"
        lobby_ws_url: str,   # e.g., "ws://localhost:8000"
        agent_message_handler: AgentMessageHandler,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        http_timeout: float = 30.0 # Default timeout for HTTP requests
    ):
        if not agent_id or not agent_type:
            raise ValueError("agent_id and agent_type must be provided")
        if not lobby_http_url.startswith(("http://", "https://")):
            raise ValueError("lobby_http_url must be a valid HTTP/S URL")
        if not lobby_ws_url.startswith(("ws://", "wss://")):
            raise ValueError("lobby_ws_url must be a valid WS/S URL")
            
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.lobby_http_base_url = lobby_http_url.rstrip('/') # Ensure no trailing slash
        self.lobby_ws_base_url = lobby_ws_url.rstrip('/') # Ensure no trailing slash
        self._agent_message_handler = agent_message_handler
        self.loop = loop or asyncio.get_event_loop()
        self.http_timeout = http_timeout

        self._api_key: Optional[str] = None # For initial registration
        self._auth_token: Optional[str] = None # Session token from Lobby
        self._lobby_id: Optional[str] = None # The ID of the lobby we're connected to
        self._is_registered: bool = False
        self._is_running: bool = False

        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_listener_task: Optional[asyncio.Task] = None
        self._ws_sender_task: Optional[asyncio.Task] = None
        self._outgoing_message_queue: asyncio.Queue[Message] = asyncio.Queue()
        
        self.interactions = InteractionManager(self)
        self._stop_event = asyncio.Event() # Used to signal shutdown to async tasks

        print(f"SDK ({self.agent_id}): Initialized. HTTP: {self.lobby_http_base_url}, WS: {self.lobby_ws_base_url}")

    # --- Core Lifecycle & Connection Methods (Placeholders) ---
    async def register_with_lobby(self) -> bool:
        """Registers the agent with the Lobby using its API key."""
        if self._is_registered:
            print(f"SDK ({self.agent_id}): Already registered.")
            return True
        if not self._api_key:
            # This should ideally be caught in start() before calling register_with_lobby
            print(f"SDK ({self.agent_id}): API key not set. Cannot register.")
            raise ValueError("API key must be set before attempting registration.")
        
        if not self._http_client:
            # This should also be set in start() before this call
            print(f"SDK ({self.agent_id}): HTTP client not initialized.")
            raise RuntimeError("HTTP client must be initialized before registration.")

        print(f"SDK ({self.agent_id}): Attempting registration with Lobby at {self.lobby_http_base_url}...")
        
        registration_payload_for_message = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": [cap.to_dict() for cap in self.capabilities]
        }
        register_msg_obj = Message(
            sender_id=self.agent_id,
            receiver_id="lobby", # Standardized receiver ID for lobby interactions
            message_type=MessageType.REGISTER,
            payload=registration_payload_for_message,
            # auth_token is not sent for initial registration; API key is used in header
        )

        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json"
        }
        
        # Assuming the Lobby has an endpoint like "/api/register" or a generic "/api/message"
        # For now, let's assume a specific registration endpoint for clarity: /api/register
        # If using a generic /api/message, the Lobby would inspect MessageType.
        registration_url = f"{self.lobby_http_base_url}/api/register" 

        try:
            print(f"SDK ({self.agent_id}): Sending REGISTER message to {registration_url}")
            response = await self._http_client.post(
                registration_url,
                json=register_msg_obj.to_dict(), # Send the Message object as JSON
                headers=headers
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            response_data = response.json()
            ack_msg = Message.from_dict(response_data) # Expecting a Message back

            if ack_msg.message_type == MessageType.REGISTER_ACK and ack_msg.payload.get("status") == "success":
                self._auth_token = ack_msg.payload.get("auth_token")
                self._lobby_id = ack_msg.sender_id # Lobby's ID is the sender of REGISTER_ACK
                
                if not self._auth_token:
                    print(f"SDK ({self.agent_id}): Registration ACK received, but no auth_token provided by Lobby.")
                    return False
                
                self._is_registered = True
                print(f"SDK ({self.agent_id}): Successfully registered with Lobby '{self._lobby_id}'. Auth Token: {self._auth_token[:10]}...")
                return True
            else:
                error_detail = ack_msg.payload.get("detail", "Unknown registration error from Lobby")
                status_code = ack_msg.payload.get("status_code", "N/A")
                print(f"SDK ({self.agent_id}): Registration failed. Lobby response status: {status_code}, Detail: {error_detail}")
                if ack_msg.message_type != MessageType.REGISTER_ACK:
                    print(f"SDK ({self.agent_id}): Expected REGISTER_ACK, but got {ack_msg.message_type.name}")
                return False

        except httpx.HTTPStatusError as e:
            # More detailed error for HTTP issues
            error_body = "No additional error detail in response body."
            try:
                error_body = e.response.json() # Try to get JSON error details from Lobby
            except json.JSONDecodeError:
                error_body = e.response.text # Fallback to raw text if not JSON
            print(f"SDK ({self.agent_id}): HTTP error during registration: {e.response.status_code} - {e.request.url}. Response: {error_body}")
            return False
        except httpx.RequestError as e:
            # For network errors, timeouts not covered by HTTPStatusError, etc.
            print(f"SDK ({self.agent_id}): Network error during registration to {e.request.url}: {e}")
            return False
        except MessageValidationError as e:
            # Should not happen if Message.from_dict is robust, but good to catch
            print(f"SDK ({self.agent_id}): Error validating REGISTER_ACK message from Lobby: {e}")
            return False
        except Exception as e:
            print(f"SDK ({self.agent_id}): Unexpected error during registration: {e}")
            return False

    async def _connect_websocket(self) -> bool:
        """Establishes a WebSocket connection to the Lobby for real-time messages."""
        if not self._is_registered or not self._auth_token:
            print(f"SDK ({self.agent_id}): Cannot connect WebSocket, agent not registered or no auth token.")
            return False
        
        # Check if connection already exists and try a simple ping to see if it's alive
        # This is a basic check; proper state management of ClientConnection might use its .state attribute
        if self._ws_connection:
            try:
                await asyncio.wait_for(self._ws_connection.ping(), timeout=5.0)
                print(f"SDK ({self.agent_id}): WebSocket already connected and responsive to ping.")
                return True
            except Exception as e:
                print(f"SDK ({self.agent_id}): Existing WebSocket connection is not responsive: {e}. Attempting to reconnect.")
                # Ensure old connection is cleaned up if possible before reconnecting
                try:
                    await self._ws_connection.close()
                except Exception:
                    pass # Ignore errors during cleanup of potentially broken connection
                self._ws_connection = None

        ws_uri = f"{self.lobby_ws_base_url}/ws/{self.agent_id}?token={self._auth_token}"
        print(f"SDK ({self.agent_id}): Connecting to WebSocket: {ws_uri}")

        try:
            self._ws_connection = await websockets.connect(ws_uri, ping_interval=20, ping_timeout=20, open_timeout=self.http_timeout)
            print(f"SDK ({self.agent_id}): WebSocket connect() returned type: {type(self._ws_connection)}")
            print(f"SDK ({self.agent_id}): WebSocket connection established. Remote: {self._ws_connection.remote_address}")
            
            # Start listener and sender tasks
            if self._ws_listener_task and not self._ws_listener_task.done():
                self._ws_listener_task.cancel()
            if self._ws_sender_task and not self._ws_sender_task.done():
                self._ws_sender_task.cancel()
                
            self._ws_listener_task = self.loop.create_task(self._ws_listener(), name=f"sdk_{self.agent_id}_ws_listener")
            self._ws_sender_task = self.loop.create_task(self._ws_sender(), name=f"sdk_{self.agent_id}_ws_sender")
            return True
        except asyncio.TimeoutError:
            print(f"SDK ({self.agent_id}): Timeout connecting to WebSocket at {ws_uri}.")
        except websockets.exceptions.InvalidURI:
            print(f"SDK ({self.agent_id}): Invalid WebSocket URI: {ws_uri}.")
        except websockets.exceptions.WebSocketException as e: # Covers ConnectionRefusedError, InvalidHandshake etc.
            print(f"SDK ({self.agent_id}): WebSocket connection failed to {ws_uri}: {type(e).__name__} - {e}")
        except Exception as e:
            print(f"SDK ({self.agent_id}): Unexpected error connecting WebSocket: {type(e).__name__} - {e}")
        
        self._ws_connection = None # Ensure connection is None if failed
        return False

    async def _ws_listener(self):
        """Listens for incoming messages on the WebSocket and handles them."""
        if not self._ws_connection:
            print(f"SDK ({self.agent_id}): WS Listener: No WebSocket connection. Exiting listener.")
            return
        
        print(f"SDK ({self.agent_id}): WebSocket listener started for {self._ws_connection.remote_address}.")
        try:
            # The ClientConnection object itself is an async iterator for messages
            async for raw_message in self._ws_connection:
                if self._stop_event.is_set(): break
                # print(f"SDK ({self.agent_id}): Received raw WS message: {raw_message[:200]}") # Log snippet
                try:
                    msg_data = json.loads(raw_message)
                    incoming_msg = Message.from_dict(msg_data)
                    
                    print(f"SDK ({self.agent_id}): Received Message {incoming_msg.message_id} (Type: {incoming_msg.message_type.name}) from {incoming_msg.sender_id}")

                    handled_by_interaction_manager = False
                    # 1. Check if it's a response to a pending request
                    if incoming_msg.conversation_id and \
                       (incoming_msg.message_type == MessageType.RESPONSE or incoming_msg.message_type == MessageType.ERROR):
                        # InteractionManager expects a Message object
                        await self.interactions.resolve_request(incoming_msg)
                        # We assume if resolve_request was called, it might have handled it.
                        # A more robust way would be for resolve_request to return a bool.
                        # For now, if conversation_id was present, assume it was for interaction manager.
                        # This needs refinement: resolve_request should only be called if it was a PENDING request.
                        # A quick check:
                        if incoming_msg.conversation_id in self.interactions.pending_requests:
                             handled_by_interaction_manager = True # Tentatively true

                    # TODO: 2. Check if it's a message for a subscription (e.g., Lobby sends MessageType.PUBLISH_EVENT)

                    if not handled_by_interaction_manager:
                        if self._agent_message_handler:
                            try:
                                response_to_send = await self._agent_message_handler(incoming_msg)
                                if response_to_send:
                                    # Ensure the agent_message_handler doesn't need to set sender_id or auth_token
                                    # as send_message will take care of it.
                                    await self.send_message(response_to_send)
                            except Exception as e:
                                print(f"SDK ({self.agent_id}): Error in agent_message_handler: {e}")
                                # Optionally send an ERROR message back to original sender or Lobby if appropriate
                        else:
                            print(f"SDK ({self.agent_id}): No agent_message_handler configured for message type {incoming_msg.message_type.name}")

                except json.JSONDecodeError:
                    print(f"SDK ({self.agent_id}): Error decoding JSON from WS: {raw_message}")
                except MessageValidationError as e:
                    print(f"SDK ({self.agent_id}): Invalid message structure received via WS: {e}")
                except Exception as e:
                    print(f"SDK ({self.agent_id}): Error processing message from WS: {type(e).__name__} - {e}")
        
        except websockets.exceptions.ConnectionClosedOK:
            print(f"SDK ({self.agent_id}): WebSocket connection closed gracefully by peer.")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"SDK ({self.agent_id}): WebSocket connection closed with error: Code {e.code}, Reason: '{e.reason}'")
        except asyncio.CancelledError:
            print(f"SDK ({self.agent_id}): WebSocket listener task was cancelled.")
        except Exception as e:
            print(f"SDK ({self.agent_id}): Unexpected error in WebSocket listener: {type(e).__name__} - {e}")
        finally:
            print(f"SDK ({self.agent_id}): WebSocket listener stopped.")
            if not self._stop_event.is_set():
                print(f"SDK ({self.agent_id}): WS listener stopped unexpectedly. Signaling stop to SDK.")
                # If the listener stops unexpectedly (e.g. connection drops and not handled by reconnect logic yet)
                # we might need to trigger a broader SDK stop or reconnect attempt.
                # For now, just log. Reconnect logic would go into _connect_websocket or a wrapper.
                # self.loop.create_task(self.stop()) # Or a reconnect attempt
                pass

    async def _ws_sender(self):
        """Sends messages from the _outgoing_message_queue via WebSocket."""
        if not self._ws_connection:
            print(f"SDK ({self.agent_id}): WS Sender: No WebSocket connection. Exiting sender.")
            return

        print(f"SDK ({self.agent_id}): WebSocket sender started for {self._ws_connection.remote_address}. Type: {type(self._ws_connection)}")
        try:
            while not self._stop_event.is_set() or not self._outgoing_message_queue.empty():
                try:
                    message_to_send = await asyncio.wait_for(self._outgoing_message_queue.get(), timeout=0.5)
                    try:
                        # print(f"SDK ({self.agent_id}): Sending message {message_to_send.message_id} via WS.")
                        await self._ws_connection.send(json.dumps(message_to_send.to_dict()))
                        self._outgoing_message_queue.task_done()
                    except websockets.exceptions.ConnectionClosed:
                        print(f"SDK ({self.agent_id}): WS Sender: Connection closed while trying to send {message_to_send.message_id}. Re-queuing.")
                        await self._outgoing_message_queue.put(message_to_send) # Re-queue
                        # Signal for reconnection or wait for it - this part needs robust handling in a real SDK
                        await asyncio.sleep(1) 
                        # Connection is closed, break the send loop
                        print(f"SDK ({self.agent_id}): WS Sender: Connection closed after send failure. Breaking send loop.")
                        break # Exit loop if connection is definitively closed
                    except Exception as e:
                        print(f"SDK ({self.agent_id}): Error sending message {message_to_send.message_id} via WS: {e}. Message discarded.")
                        self._outgoing_message_queue.task_done()
                except asyncio.TimeoutError:
                    if self._stop_event.is_set() and self._outgoing_message_queue.empty():
                        break 
                    continue 
        except asyncio.CancelledError:
            print(f"SDK ({self.agent_id}): WebSocket sender task was cancelled.")
        except Exception as e:
            print(f"SDK ({self.agent_id}): Unexpected error in WebSocket sender: {type(e).__name__} - {e}")
        finally:
            print(f"SDK ({self.agent_id}): WebSocket sender stopped.")

    # --- Public API Methods for Agent Developers ---
    async def start(self, api_key: str) -> bool:
        """Starts the SDK client: registers with Lobby and connects WebSocket."""
        if self._is_running:
            print(f"SDK ({self.agent_id}): Already running.")
            return True

        self._api_key = api_key
        self._stop_event.clear()
        self._http_client = httpx.AsyncClient(timeout=self.http_timeout)

        if not await self.register_with_lobby():
            print(f"SDK ({self.agent_id}): Failed to register with Lobby. Cannot start.")
            await self._http_client.aclose() # Clean up httpx client
            self._http_client = None
            return False
        
        if not await self._connect_websocket():
            print(f"SDK ({self.agent_id}): Failed to connect WebSocket. Starting in degraded mode or failing? For now, failing.")
            # Optionally, could allow starting without WebSocket if only HTTP interactions are needed, 
            # but typical operation relies on WS for receiving messages.
            await self.stop() # Call stop to clean up what might have started (like http_client)
            return False

        self._is_running = True
        print(f"SDK ({self.agent_id}): Successfully started and connected.")
        # TODO: Optionally send an initial ADVERTISE_CAPABILITIES message here
        return True

    async def stop(self):
        """Stops the SDK client, closing connections and cleaning up tasks."""
        if not self._is_running and not self._stop_event.is_set():
            pass

        print(f"SDK ({self.agent_id}): Stopping...")
        self._stop_event.set()

        if self._ws_listener_task and not self._ws_listener_task.done():
            self._ws_listener_task.cancel()
            try: await self._ws_listener_task
            except asyncio.CancelledError: print(f"SDK ({self.agent_id}): WebSocket listener task cancelled.")
        
        if self._ws_sender_task and not self._ws_sender_task.done():
            if not self._outgoing_message_queue.empty():
                 print(f"SDK ({self.agent_id}): Waiting for outgoing queue to empty...")
                 try: await asyncio.wait_for(self._outgoing_message_queue.join(), timeout=5.0)
                 except asyncio.TimeoutError: print(f"SDK ({self.agent_id}): Timeout waiting for outgoing queue.")
            self._ws_sender_task.cancel()
            try: await self._ws_sender_task
            except asyncio.CancelledError: print(f"SDK ({self.agent_id}): WebSocket sender task cancelled.")
        
        if self._ws_connection:
            print(f"SDK ({self.agent_id}): Closing WebSocket connection ({type(self._ws_connection)})...")
            try:
                await self._ws_connection.close()
                print(f"SDK ({self.agent_id}): WebSocket connection closed.")
            except Exception as e:
                print(f"SDK ({self.agent_id}): Error closing WebSocket: {e}")
            self._ws_connection = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            print(f"SDK ({self.agent_id}): HTTP client closed.")

        self._is_registered = False
        self._is_running = False
        self._auth_token = None # Clear session token
        self._api_key = None # Clear API key
        print(f"SDK ({self.agent_id}): Stopped.")

    async def send_message(self, message: Message):
        """Queues a message to be sent to the Lobby or another agent via the Lobby."""
        if not self._is_running and not self._auth_token: # Check if we can even send (registered)
            print(f"SDK ({self.agent_id}): Cannot send message, SDK not running or not registered.")
            # Consider raising an error or returning a status
            return

        message.sender_id = self.agent_id # Ensure sender_id is this agent
        if not message.auth_token and self._auth_token:
            message.auth_token = self._auth_token # Add current session token if not set
        
        # Validate message before queuing (optional, can be strict)
        try:
            message.validate() # Uses the validate method in Message class
        except MessageValidationError as e:
            print(f"SDK ({self.agent_id}): Invalid message, not sending. Error: {e}. Message: {message.to_dict()}")
            return # Or raise error

        await self._outgoing_message_queue.put(message)
        print(f"SDK ({self.agent_id}): Queued message {message.message_id} (Type: {message.message_type.name}) to {message.receiver_id}")

    async def request(self, 
                      receiver_id: str, 
                      payload: Dict[str, Any], 
                      timeout: float = 15.0, 
                      request_message_type: MessageType = MessageType.REQUEST, # Allow customizing message type
                      expected_response_type: MessageType = MessageType.RESPONSE
                      ) -> Message:
        """Sends a request and waits for a direct response with a matching conversation_id."""
        if not self._is_running or not self._is_registered:
            raise RuntimeError(f"SDK ({self.agent_id}): Not running or not registered. Cannot send request.")

        conv_id = str(uuid.uuid4())
        request_msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=request_message_type,
            payload=payload,
            conversation_id=conv_id,
            auth_token=self._auth_token, # SDK ensures this is set
            requires_ack=True, # Good practice for requests to ensure delivery to Lobby at least
            ack_timeout=5.0
        )

        response_future = await self.interactions.register_request(conv_id)
        await self.send_message(request_msg) # This will queue the message
        
        print(f"SDK ({self.agent_id}): Sent request {conv_id} to {receiver_id}, awaiting response...")
        try:
            response_message = await asyncio.wait_for(response_future, timeout)
            # Basic check if the response is of the expected type
            if response_message.message_type != expected_response_type:
                # This could be an ERROR message from the lobby or agent
                print(f"SDK ({self.agent_id}): Received response for {conv_id}, but type {response_message.message_type.name} was not expected {expected_response_type.name}. Payload: {response_message.payload}")
                # Decide whether to raise an error or return the message as is.
                # For now, return as is, caller can inspect.
            return response_message
        except asyncio.TimeoutError:
            print(f"SDK ({self.agent_id}): Request {conv_id} to {receiver_id} timed out after {timeout}s.")
            await self.interactions.cancel_request(conv_id, reason=f"timed out after {timeout}s")
            raise TimeoutError(f"Request {conv_id} to {receiver_id} timed out.")
        except Exception as e:
            print(f"SDK ({self.agent_id}): Error awaiting response for {conv_id}: {e}")
            await self.interactions.cancel_request(conv_id, reason=f"error: {e}")
            raise

    # --- Placeholder for higher-level service methods ---
    async def discover_services(self, capability_name: Optional[str] = None, agent_type: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        print(f"SDK ({self.agent_id}): discover_services not fully implemented yet.")
        # Example of how it would use self.request:
        # discovery_payload = {}
        # if capability_name: discovery_payload["capability_name"] = capability_name
        # if agent_type: discovery_payload["agent_type"] = agent_type
        # try:
        #     response_msg = await self.request(
        #         receiver_id="lobby",
        #         payload=discovery_payload,
        #         request_message_type=MessageType.DISCOVER_SERVICES,
        #         expected_response_type=MessageType.SERVICES_AVAILABLE
        #     )
        #     return response_msg.payload.get("services", [])
        # except TimeoutError:
        #     print(f"SDK ({self.agent_id}): Timeout during service discovery.")
        #     return None
        # except Exception as e:
        #     print(f"SDK ({self.agent_id}): Error during service discovery: {e}")
        #     return None
        return None

    async def advertise_capabilities(self):
        # payload = {"capabilities": [cap.to_dict() for cap in self.capabilities]}
        # msg = Message(receiver_id="lobby", message_type=MessageType.ADVERTISE_CAPABILITIES, payload=payload)
        # await self.send_message(msg)
        print(f"SDK ({self.agent_id}): advertise_capabilities not fully implemented yet.")

    async def report_task_outcome(self, provider_agent_id: str, capability_name: str, status: str, details: str, original_conversation_id: Optional[str]):
        # payload = { ... }
        # msg = Message(receiver_id="lobby", message_type=MessageType.TASK_OUTCOME_REPORT, payload=payload, conversation_id=original_conversation_id)
        # await self.send_message(msg)
        print(f"SDK ({self.agent_id}): report_task_outcome not fully implemented yet.")

# --- End Placeholder for EcosystemClient and InteractionManager ---

if __name__ == "__main__":
    # Example SDK-side message creation
    sdk_msg = Message(
        sender_id="my_sdk_agent_001",
        receiver_id="lobby",
        message_type=MessageType.REGISTER,
        payload={"agent_type": "example_bot", "capabilities": ["echo"]},
        priority=MessagePriority.HIGH
    )
    print("SDK Message to_dict():")
    print(json.dumps(sdk_msg.to_dict(), indent=2))

    dict_data = sdk_msg.to_dict()
    recreated_sdk_msg = Message.from_dict(dict_data)
    print("\nRecreated SDK Message from_dict():")
    assert recreated_sdk_msg.sender_id == sdk_msg.sender_id
    assert recreated_sdk_msg.message_type == sdk_msg.message_type
    print(json.dumps(recreated_sdk_msg.to_dict(), indent=2))

    # Example ACK
    original_msg_requiring_ack = Message(
        sender_id="requester_agent",
        receiver_id="my_sdk_agent_001",
        message_type=MessageType.REQUEST,
        payload={"data": "please process"},
        requires_ack=True,
        ack_timeout=5.0
    )
    ack_by_sdk_agent = original_msg_requiring_ack.create_ack(ack_sender_id="my_sdk_agent_001")
    print("\nACK created by SDK agent:")
    print(json.dumps(ack_by_sdk_agent.to_dict(), indent=2)) 