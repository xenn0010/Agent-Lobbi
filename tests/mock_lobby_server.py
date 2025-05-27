import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum, auto, IntEnum
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass, field

from aiohttp import web
import websockets

# --- Copied from src/sdk/ecosystem_sdk.py (ensure these are in sync or imported if structure allows) ---
class MessagePriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BULK = 4

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

class MessageValidationError(Exception):
    pass

@dataclass
class Message:
    sender_id: str
    receiver_id: str
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name,
            "payload": self.payload,
            "conversation_id": self.conversation_id,
            "auth_token": self.auth_token,
            "priority": self.priority.value,
            "requires_ack": self.requires_ack,
            "ack_timeout": self.ack_timeout,
            "broadcast_scope": list(self.broadcast_scope) if self.broadcast_scope else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        broadcast_scope = set(data["broadcast_scope"]) if data.get("broadcast_scope") else None
        priority_val = data.get("priority")
        priority = MessagePriority(priority_val) if priority_val is not None else MessagePriority.NORMAL
        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType[data["message_type"]],
            payload=data.get("payload", {}),
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            conversation_id=data.get("conversation_id"),
            auth_token=data.get("auth_token"),
            priority=priority,
            requires_ack=data.get("requires_ack", False),
            ack_timeout=data.get("ack_timeout"),
            broadcast_scope=broadcast_scope,
        )
# --- End of copied definitions ---

MOCK_LOBBY_ID = "mock_lobby_001"
VALID_API_KEY = "test_api_key"
GENERATED_AUTH_TOKEN = "dummy_auth_token_for_sdk_testing"

# Store active WebSocket connections for agents
# Key: agent_id, Value: WebSocket connection
connected_agents_ws: Dict[str, Any] = {}

# Store the current path for websocket handler (workaround for newer websockets versions)
current_ws_path: str = ""


async def handle_register(request: web.Request):
    """Handles agent registration requests."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        print("MOCK LOBBY: Registration attempt missing X-API-Key.")
        return web.json_response(
            Message(
                sender_id=MOCK_LOBBY_ID,
                receiver_id="unknown_agent", # We don't know sender_id yet
                message_type=MessageType.REGISTER_ACK,
                payload={"status": "error", "detail": "Missing X-API-Key header", "status_code": 401}
            ).to_dict(),
            status=401
        )

    if api_key != VALID_API_KEY:
        print(f"MOCK LOBBY: Registration attempt with invalid API Key: {api_key}")
        return web.json_response(
            Message(
                sender_id=MOCK_LOBBY_ID,
                receiver_id="unknown_agent",
                message_type=MessageType.REGISTER_ACK,
                payload={"status": "error", "detail": "Invalid API Key", "status_code": 403}
            ).to_dict(),
            status=403
        )

    try:
        data = await request.json()
        print(f"MOCK LOBBY: Received registration data: {data}")
        reg_msg = Message.from_dict(data)
        print(f"MOCK LOBBY: Received registration from Agent ID: {reg_msg.sender_id}, Type: {reg_msg.payload.get('agent_type')}")

        ack_payload = {
            "status": "success",
            "auth_token": GENERATED_AUTH_TOKEN,
            "lobby_id": MOCK_LOBBY_ID,
            "detail": "Registration successful."
        }
        ack_msg = Message(
            sender_id=MOCK_LOBBY_ID,
            receiver_id=reg_msg.sender_id,
            message_type=MessageType.REGISTER_ACK,
            payload=ack_payload,
            conversation_id=reg_msg.conversation_id # Echo back conversation_id if present
        )
        print(f"MOCK LOBBY: Sending REGISTER_ACK to {reg_msg.sender_id}")
        return web.json_response(ack_msg.to_dict(), status=200)

    except json.JSONDecodeError:
        print("MOCK LOBBY: Registration failed - Invalid JSON.")
        return web.json_response({"error": "Invalid JSON format"}, status=400)
    except MessageValidationError as e:
        print(f"MOCK LOBBY: Registration failed - Message validation error: {e}")
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        print(f"MOCK LOBBY: Registration failed - Unexpected error: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({"error": "Internal server error"}, status=500)


async def websocket_handler(websocket):
    """Handles WebSocket connections from agents."""
    global current_ws_path
    path = current_ws_path
    print(f"MOCK LOBBY: WebSocket connection established for path: {path}")
    # Path is expected to be /ws/{agent_id}?token={auth_token}
    # For simplicity, we'll parse it crudely. A real server would use a router or regex.
    try:
        parts = path.strip("/").split("?")
        print(f"MOCK LOBBY: Path parts: {parts}")
        agent_id_path = parts[0].split("/")
        print(f"MOCK LOBBY: Agent ID path: {agent_id_path}")
        if len(agent_id_path) < 2 or agent_id_path[0] != "ws":
            raise ValueError(f"Invalid WebSocket path format. Expected /ws/{{agent_id}}, got: {parts[0]}")
        agent_id = agent_id_path[1]

        token = None
        if len(parts) > 1:
            params = dict(p.split("=") for p in parts[1].split("&"))
            token = params.get("token")
        print(f"MOCK LOBBY: Parsed agent_id: {agent_id}, token: {token[:10] if token else None}...")

        if token != GENERATED_AUTH_TOKEN:
            print(f"MOCK LOBBY: WS connection from {agent_id} denied. Invalid or missing token: {token}")
            await websocket.close(code=1008, reason="Invalid auth token")
            return

        print(f"MOCK LOBBY: Agent {agent_id} connected via WebSocket.")
        connected_agents_ws[agent_id] = websocket
        
        try:
            async for raw_message in websocket:
                print(f"MOCK LOBBY (WS): Received from {agent_id}: {raw_message[:200]}...") # Log snippet
                try:
                    msg_data = json.loads(raw_message)
                    incoming_msg = Message.from_dict(msg_data)

                    # Simple echo for INFO messages
                    if incoming_msg.message_type == MessageType.INFO:
                        print(f"MOCK LOBBY (WS): Echoing INFO message {incoming_msg.message_id} back to {agent_id}")
                        await websocket.send(json.dumps(incoming_msg.to_dict()))
                    
                    # Handle specific REQUEST messages for testing
                    elif incoming_msg.message_type == MessageType.REQUEST:
                        if incoming_msg.payload.get("action") == "get_time":
                            response_payload = {"time": datetime.now(timezone.utc).isoformat(), "status": "success"}
                            response_msg = Message(
                                sender_id=MOCK_LOBBY_ID,
                                receiver_id=agent_id,
                                message_type=MessageType.RESPONSE,
                                payload=response_payload,
                                conversation_id=incoming_msg.conversation_id
                            )
                            print(f"MOCK LOBBY (WS): Sending time RESPONSE to {agent_id} for conv_id {incoming_msg.conversation_id}")
                            await websocket.send(json.dumps(response_msg.to_dict()))
                        else:
                            error_payload = {"error": "unknown_request_action", "detail": "Mock lobby doesn't understand this request."}
                            error_msg = Message(
                                sender_id=MOCK_LOBBY_ID,
                                receiver_id=agent_id,
                                message_type=MessageType.ERROR,
                                payload=error_payload,
                                conversation_id=incoming_msg.conversation_id
                            )
                            print(f"MOCK LOBBY (WS): Sending ERROR (unknown action) to {agent_id} for conv_id {incoming_msg.conversation_id}")
                            await websocket.send(json.dumps(error_msg.to_dict()))
                    
                    # You can add more handlers here for other message types if needed for testing

                except json.JSONDecodeError:
                    print(f"MOCK LOBBY (WS) from {agent_id}: Error decoding JSON: {raw_message}")
                except MessageValidationError as e:
                    print(f"MOCK LOBBY (WS) from {agent_id}: Invalid message: {e}")
                except Exception as e:
                    print(f"MOCK LOBBY (WS) from {agent_id}: Error processing message: {type(e).__name__} - {e}")

        except websockets.exceptions.ConnectionClosedOK:
            print(f"MOCK LOBBY: Agent {agent_id} WebSocket connection closed gracefully.")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"MOCK LOBBY: Agent {agent_id} WebSocket connection closed with error: Code {e.code}, Reason: '{e.reason}'")
        finally:
            if agent_id in connected_agents_ws:
                del connected_agents_ws[agent_id]
            print(f"MOCK LOBBY: Agent {agent_id} disconnected.")
            
    except ValueError as e:
        print(f"MOCK LOBBY: Error processing WebSocket path '{path}': {e}")
        await websocket.close(code=1003, reason="Invalid path format")
    except Exception as e:
        print(f"MOCK LOBBY: Unexpected error in websocket_handler for path '{path}': {type(e).__name__} - {e}")
        # Ensure websocket is closed if an unexpected error occurs before the main try-except block for messages
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass  # Ignore errors during cleanup


async def main():
    # --- HTTP Server (aiohttp) ---
    http_app = web.Application()
    http_app.router.add_post("/api/register", handle_register)
    
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, "localhost", 8080) # Using 8080 for mock HTTP
    await http_site.start()
    print("MOCK LOBBY: HTTP server started on http://localhost:8080")

    # --- WebSocket Server (websockets) ---
    # Use process_request to handle path routing
    async def process_request(path, request_headers):
        global current_ws_path
        current_ws_path = path
        print(f"MOCK LOBBY: WebSocket request to path: {path}")
        return None  # Continue with normal WebSocket handling
    
    ws_server = await websockets.serve(
        websocket_handler, 
        "localhost", 
        8081,
        process_request=process_request
    )
    print("MOCK LOBBY: WebSocket server started on ws://localhost:8081")

    try:
        await asyncio.Event().wait() # Keep servers running indefinitely
    except KeyboardInterrupt:
        print("MOCK LOBBY: Shutting down...")
    finally:
        await http_runner.cleanup()
        ws_server.close()
        await ws_server.wait_closed()
        print("MOCK LOBBY: Servers stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("MOCK LOBBY: Main loop interrupted.") 