import asyncio
import logging
import uuid
import websockets
from typing import Dict, Any

# Use the correct SDK client and config
from src.sdk.python_sdk.python_sdk.client import EnhancedEcosystemClient, SDKConfig

# Configure basic logging for the agent
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - GREETER_AGENT - %(message)s')

greeting_payload_content = {}

async def my_message_handler(message: Dict[str, Any], agent_id: str):
    global greeting_payload_content
    logging.info(f"Greeter Agent ({agent_id}) received message: {message}")
    
    if message.get("message_type") == "INFO" and \
       message.get("payload", {}).get("content") == greeting_payload_content.get("content") and \
       message.get("sender_id") == agent_id: # Echoed message will retain original sender_id
        logging.info(f"SUCCESS: Received echo of our greeting! Original payload: {greeting_payload_content}")

async def run_greeter_agent():
    global greeting_payload_content
    agent_id_suffix = str(uuid.uuid4())[:8]
    logging.info(f"Starting Greeter Agent (example suffix: {agent_id_suffix}). Final agent_id will be determined by SDKConfig.")

    client = None # Initialize client to None for robust finally block
    sdk_config_instance = None # For logging in ConnectionRefusedError

    try:
        sdk_config_instance = SDKConfig.load_from_yaml(agent_id=f"greeter_agent_{agent_id_suffix}")
        logging.info(f"SDKConfig loaded for agent_id: {sdk_config_instance.agent_id}, ws_base_url: {sdk_config_instance.ws_base_url}, auth_token: {'present' if sdk_config_instance.auth_token else 'missing'}")

    except FileNotFoundError:
        logging.error("Configuration file (config/config.yaml) not found. Please ensure it exists.")
        return
    except Exception as e:
        logging.error(f"Failed to load SDK configuration: {e}", exc_info=True)
        return

    client = EnhancedEcosystemClient(config=sdk_config_instance)
    client.set_message_handler(lambda msg: my_message_handler(msg, client.config.agent_id))

    try:
        logging.info(f"Agent {client.config.agent_id} attempting to connect to the lobby at {client.config.ws_base_url}...")
        await client.connect()

        if client.is_connected:
            logging.info(f"Agent {client.config.agent_id} successfully connected to the lobby.")
            logging.info(f"Agent {client.config.agent_id} attempting to register...")
            registered = await client.register()

            if registered or client.is_registered: # client.is_registered is set true on send by current SDK
                logging.info(f"Agent {client.config.agent_id} sent registration request.")
                greeting_payload_content = {"content": f"Hello from {client.config.agent_id}!"}
                message_to_send = {
                    "sender_id": client.config.agent_id, # Mock lobby echoes this, so include for check
                    "type": "INFO", # simple_mock_lobby is set to echo INFO type messages
                    "payload": greeting_payload_content
                }
                logging.info(f"Agent {client.config.agent_id} sending greeting message: {message_to_send['payload']}")
                await client.send_message(message_to_send)
                logging.info(f"Agent {client.config.agent_id} will wait for 10 seconds for messages...")
                await asyncio.sleep(10)
            else:
                logging.error(f"Agent {client.config.agent_id} failed to register.")
        else:
            logging.error(f"Agent {client.config.agent_id} failed to connect to the lobby.")

    except ValueError as ve: 
        logging.error(f"ValueError for agent {client.config.agent_id if client else 'UNKNOWN'}: {ve}", exc_info=True)
    except websockets.exceptions.InvalidURI as e:
        logging.error(f"Invalid WebSocket URI for agent {client.config.agent_id if client else 'UNKNOWN'}: {e}", exc_info=True)
    except ConnectionRefusedError as e:
        ws_url_for_log = sdk_config_instance.ws_base_url if sdk_config_instance else "N/A"
        agent_id_for_log = client.config.agent_id if client else (sdk_config_instance.agent_id if sdk_config_instance else 'UNKNOWN')
        logging.error(f"Connection refused for agent {agent_id_for_log}. Is the mock lobby running at {ws_url_for_log}? Error: {e}", exc_info=True)
    except Exception as e:
        agent_id_for_log = client.config.agent_id if client else (sdk_config_instance.agent_id if sdk_config_instance else 'UNKNOWN')
        logging.error(f"An error occurred during agent {agent_id_for_log} operation: {e}", exc_info=True)
    finally:
        if client and client.is_connected:
            logging.info(f"Shutting down agent {client.config.agent_id}.")
            await client.close()
        else:
            agent_id_for_log = client.config.agent_id if client and hasattr(client, 'config') else (sdk_config_instance.agent_id if sdk_config_instance else 'UNKNOWN')
            logging.info(f"Shutting down agent {agent_id_for_log} (was not connected or client not fully initialized).")

if __name__ == "__main__":
    asyncio.run(run_greeter_agent()) 