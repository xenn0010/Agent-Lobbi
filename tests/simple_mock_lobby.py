#!/usr/bin/env python3
"""
Simple Mock Lobby Server for testing the SDK
"""
import asyncio
import json
import uuid
from datetime import datetime, timezone
from aiohttp import web
import websockets

# Simple message structure for testing
def create_message(sender_id, receiver_id, message_type, payload=None):
    return {
        "message_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "message_type": message_type,
        "payload": payload or {},
        "conversation_id": None,
        "auth_token": None,
        "priority": 2,
        "requires_ack": False,
        "ack_timeout": None,
        "broadcast_scope": None
    }

# Constants
MOCK_LOBBY_ID = "mock_lobby_001"
VALID_API_KEY = "test_api_key"
GENERATED_AUTH_TOKEN = "dummy_auth_token_for_sdk_testing"

# Store active WebSocket connections
connected_agents = {}

async def handle_register(request):
    """Handle HTTP registration requests."""
    print("SIMPLE MOCK: Received registration request")
    
    # Check API key
    api_key = request.headers.get("X-API-Key")
    if api_key != VALID_API_KEY:
        print(f"SIMPLE MOCK: Invalid API key: {api_key}")
        return web.json_response({"error": "Invalid API key"}, status=403)
    
    try:
        data = await request.json()
        print(f"SIMPLE MOCK: Registration data: {data}")
        
        agent_id = data.get("sender_id", "unknown")
        print(f"SIMPLE MOCK: Registering agent: {agent_id}")
        
        # Send success response
        response = create_message(
            sender_id=MOCK_LOBBY_ID,
            receiver_id=agent_id,
            message_type="REGISTER_ACK",
            payload={
                "status": "success",
                "auth_token": GENERATED_AUTH_TOKEN,
                "lobby_id": MOCK_LOBBY_ID
            }
        )
        
        print(f"SIMPLE MOCK: Sending registration ACK to {agent_id}")
        return web.json_response(response, status=200)
        
    except Exception as e:
        print(f"SIMPLE MOCK: Registration error: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def websocket_handler(websocket):
    """Handle WebSocket connections."""
    # In newer websockets versions, we need to get the path differently
    path = websocket.request.path if hasattr(websocket, 'request') else "/unknown"
    print(f"SIMPLE MOCK: WebSocket connection to path: {path}")
    
    try:
        # Parse path to get agent_id and token
        # Expected: /ws/{agent_id}?token={token}
        parts = path.strip("/").split("?")
        if not parts[0].startswith("ws/"):
            print(f"SIMPLE MOCK: Invalid path format: {path}")
            await websocket.close(code=1003, reason="Invalid path")
            return
            
        agent_id = parts[0].split("/")[1]
        
        # Check token
        token = None
        if len(parts) > 1:
            params = dict(p.split("=") for p in parts[1].split("&"))
            token = params.get("token")
            
        if token != GENERATED_AUTH_TOKEN:
            print(f"SIMPLE MOCK: Invalid token for {agent_id}: {token}")
            await websocket.close(code=1008, reason="Invalid token")
            return
            
        print(f"SIMPLE MOCK: Agent {agent_id} connected via WebSocket")
        connected_agents[agent_id] = websocket
        
        # Handle messages
        async for raw_message in websocket:
            print(f"SIMPLE MOCK: Received from {agent_id}: {raw_message[:100]}...")
            
            try:
                msg_data = json.loads(raw_message)
                msg_type = msg_data.get("message_type")
                
                if msg_type == "INFO":
                    # Echo INFO messages back
                    print(f"SIMPLE MOCK: Echoing INFO message back to {agent_id}")
                    await websocket.send(raw_message)
                    
                elif msg_type == "REQUEST":
                    # Handle REQUEST messages
                    action = msg_data.get("payload", {}).get("action")
                    conv_id = msg_data.get("conversation_id")
                    
                    if action == "get_time":
                        response = create_message(
                            sender_id=MOCK_LOBBY_ID,
                            receiver_id=agent_id,
                            message_type="RESPONSE",
                            payload={"time": datetime.now(timezone.utc).isoformat(), "status": "success"}
                        )
                        response["conversation_id"] = conv_id
                        print(f"SIMPLE MOCK: Sending time response to {agent_id}")
                        await websocket.send(json.dumps(response))
                    else:
                        # Unknown action
                        error_response = create_message(
                            sender_id=MOCK_LOBBY_ID,
                            receiver_id=agent_id,
                            message_type="ERROR",
                            payload={"error": "unknown_action", "detail": f"Unknown action: {action}"}
                        )
                        error_response["conversation_id"] = conv_id
                        print(f"SIMPLE MOCK: Sending error response to {agent_id}")
                        await websocket.send(json.dumps(error_response))
                        
            except json.JSONDecodeError:
                print(f"SIMPLE MOCK: Invalid JSON from {agent_id}: {raw_message}")
            except Exception as e:
                print(f"SIMPLE MOCK: Error processing message from {agent_id}: {e}")
                
    except websockets.exceptions.ConnectionClosedOK:
        print(f"SIMPLE MOCK: {agent_id} disconnected gracefully")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"SIMPLE MOCK: {agent_id} disconnected with error: {e.code} - {e.reason}")
    except Exception as e:
        print(f"SIMPLE MOCK: Unexpected error in websocket handler: {e}")
    finally:
        if agent_id in connected_agents:
            del connected_agents[agent_id]
        print(f"SIMPLE MOCK: Cleaned up connection for {agent_id}")

async def main():
    print("SIMPLE MOCK: Starting servers...")
    
    # HTTP Server
    app = web.Application()
    app.router.add_post("/api/register", handle_register)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()
    print("SIMPLE MOCK: HTTP server started on http://localhost:8080")
    
    # WebSocket Server
    ws_server = await websockets.serve(websocket_handler, "localhost", 8081)
    print("SIMPLE MOCK: WebSocket server started on ws://localhost:8081")
    
    print("SIMPLE MOCK: Servers ready!")
    
    try:
        await asyncio.Event().wait()  # Run forever
    except KeyboardInterrupt:
        print("SIMPLE MOCK: Shutting down...")
    finally:
        await runner.cleanup()
        ws_server.close()
        await ws_server.wait_closed()
        print("SIMPLE MOCK: Servers stopped.")

if __name__ == "__main__":
    asyncio.run(main()) 