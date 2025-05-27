import asyncio
import sys
import os
from typing import Optional

# Add src directory to Python path to import SDK components
# This is a common way to handle imports for tests when modules are in a sibling directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sdk_path = os.path.join(current_dir, '..', 'src') # Assuming tests is sibling to src
sys.path.insert(0, sdk_path)

# Now we can import from the SDK
from sdk.ecosystem_sdk import EcosystemClient, Message, MessageType, AgentCapabilitySDK

# Configuration for the mock lobby
MOCK_LOBBY_HTTP_URL = "http://localhost:8080"
MOCK_LOBBY_WS_URL = "ws://localhost:8081"
TEST_API_KEY = "test_api_key"
TEST_AGENT_ID = "sdk_tester_001"
TEST_AGENT_TYPE = "TestAgent"

async def my_agent_message_handler(message: Message) -> Optional[Message]:
    """Handles messages received by the test agent."""
    print(f"TEST AGENT ({TEST_AGENT_ID}): Received message: Type={message.message_type.name}, From={message.sender_id}, ConvID={message.conversation_id}")
    print(f"TEST AGENT ({TEST_AGENT_ID}): Payload: {message.payload}")
    
    # Example: if we got an INFO message (perhaps an echo from the mock lobby)
    if message.message_type == MessageType.INFO:
        print(f"TEST AGENT ({TEST_AGENT_ID}): Confirmed INFO message {message.message_id} received.")
    
    # This handler doesn't need to send a reply for this basic test,
    # as request-response is handled by the sdk_client.request() future.
    # If it were to send a direct reply, it would be: 
    # return Message(receiver_id=message.sender_id, message_type=MessageType.INFO, payload={"status": "got your message"})
    return None

async def run_sdk_test():
    print(f"TEST AGENT ({TEST_AGENT_ID}): Initializing SDK client...")
    capabilities = [
        AgentCapabilitySDK(name="echo_test", description="Can echo messages for testing."),
        AgentCapabilitySDK(name="time_requester", description="Can request time from lobby.")
    ]
    
    sdk_client = EcosystemClient(
        agent_id=TEST_AGENT_ID,
        agent_type=TEST_AGENT_TYPE,
        capabilities=capabilities,
        lobby_http_url=MOCK_LOBBY_HTTP_URL,
        lobby_ws_url=MOCK_LOBBY_WS_URL,
        agent_message_handler=my_agent_message_handler,
        loop=asyncio.get_event_loop()
    )

    print(f"TEST AGENT ({TEST_AGENT_ID}): Starting SDK client and registering with API key: {TEST_API_KEY[:5]}...")
    try:
        if not await sdk_client.start(api_key=TEST_API_KEY):
            print(f"TEST AGENT ({TEST_AGENT_ID}): SDK client failed to start. Exiting.")
            return

        print(f"TEST AGENT ({TEST_AGENT_ID}): SDK client started successfully.")

        # 1. Send a simple INFO message (fire and forget, but mock lobby should echo it)
        print(f"TEST AGENT ({TEST_AGENT_ID}): Sending INFO message to lobby...")
        info_payload = {"data": "Hello from SDK Test Agent!", "agent_version": "0.1-test"}
        info_message = Message(
            sender_id=TEST_AGENT_ID,
            receiver_id="lobby", # Mock lobby will handle this
            message_type=MessageType.INFO,
            payload=info_payload
        )
        await sdk_client.send_message(info_message)
        print(f"TEST AGENT ({TEST_AGENT_ID}): INFO message {info_message.message_id} sent.")

        # Give a moment for the INFO message to be processed and echoed
        await asyncio.sleep(1)

        # 2. Make a request that expects a response (get_time)
        print(f"TEST AGENT ({TEST_AGENT_ID}): Sending REQUEST for current time to lobby...")
        try:
            request_payload = {"action": "get_time", "details": "Need current server time"}
            response_message = await sdk_client.request(
                receiver_id="lobby", # Mock lobby will handle this request
                payload=request_payload,
                timeout=10.0 # Generous timeout for testing
            )
            print(f"TEST AGENT ({TEST_AGENT_ID}): Received RESPONSE for get_time request.")
            print(f"TEST AGENT ({TEST_AGENT_ID}): Response Payload: {response_message.payload}")
            if response_message.payload.get("status") == "success" and "time" in response_message.payload:
                print(f"TEST AGENT ({TEST_AGENT_ID}): Successfully got time: {response_message.payload['time']}")
            else:
                print(f"TEST AGENT ({TEST_AGENT_ID}): Time request did not succeed as expected. Response: {response_message.payload}")
        
        except TimeoutError:
            print(f"TEST AGENT ({TEST_AGENT_ID}): Request for get_time timed out.")
        except Exception as e:
            print(f"TEST AGENT ({TEST_AGENT_ID}): Error during get_time request: {type(e).__name__} - {e}")

        # 3. Test a request that the mock lobby doesn't know how to handle (to get an ERROR response)
        print(f"TEST AGENT ({TEST_AGENT_ID}): Sending REQUEST for unknown_action to lobby...")
        try:
            unknown_request_payload = {"action": "unknown_action"}
            error_response = await sdk_client.request(
                receiver_id="lobby",
                payload=unknown_request_payload,
                timeout=5.0,
                expected_response_type=MessageType.ERROR # We expect an ERROR message
            )
            print(f"TEST AGENT ({TEST_AGENT_ID}): Received ERROR response for unknown_action as expected.")
            print(f"TEST AGENT ({TEST_AGENT_ID}): Error Payload: {error_response.payload}")
        except TimeoutError:
            print(f"TEST AGENT ({TEST_AGENT_ID}): Request for unknown_action timed out.")
        except Exception as e:
            print(f"TEST AGENT ({TEST_AGENT_ID}): Error during unknown_action request: {type(e).__name__} - {e}")

    except Exception as e:
        print(f"TEST AGENT ({TEST_AGENT_ID}): An unexpected error occurred during the test: {type(e).__name__} - {e}")
    finally:
        print(f"TEST AGENT ({TEST_AGENT_ID}): Stopping SDK client...")
        await sdk_client.stop()
        print(f"TEST AGENT ({TEST_AGENT_ID}): SDK client stopped. Test finished.")

if __name__ == "__main__":
    # Ensure the mock server is running before starting the client test
    print("---------------------------------------------------------------------")
    print("MAKE SURE mock_lobby_server.py IS RUNNING ON localhost:8080 (HTTP) and localhost:8081 (WS)")
    print("Run: python tests/mock_lobby_server.py")
    print("---------------------------------------------------------------------")
    input("Press Enter to start the test agent once the mock server is running...")
    
    asyncio.run(run_sdk_test()) 