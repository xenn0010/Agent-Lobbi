import asyncio
import websockets
import json
import sys
import os

# Add src directory to Python path to import SDK components
current_dir = os.path.dirname(os.path.abspath(__file__))
sdk_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, sdk_path)

from sdk.ecosystem_sdk import Message, MessageType

async def test_websocket():
    uri = "ws://localhost:8081/ws/test_agent?token=dummy_auth_token_for_sdk_testing"
    print(f"Connecting to: {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected! Type: {type(websocket)}")
            print(f"Remote address: {websocket.remote_address}")
            
            # Create a proper Message object like the SDK does
            test_msg = Message(
                sender_id="test_agent",
                receiver_id="lobby",
                message_type=MessageType.INFO,
                payload={"test": "hello from simple test"}
            )
            
            print("Sending test message...")
            print(f"Message dict: {test_msg.to_dict()}")
            await websocket.send(json.dumps(test_msg.to_dict()))
            print("Message sent!")
            
            # Wait for response
            print("Waiting for response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"Received: {response}")
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket closed: Code {e.code}, Reason: '{e.reason}'")
    except Exception as e:
        print(f"Error: {type(e).__name__} - {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket()) 