import asyncio
import logging
import uuid

from src.sdk.enhanced_ecosystem_sdk import EnhancedEcosystemClient, SDKConfig, Message
from src.core.config import ConfigManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - TIME_ORACLE_AGENT - %(message)s')

async def run_time_oracle_agent():
    agent_id = f"time_oracle_{str(uuid.uuid4())[:8]}"
    logging.info(f"Starting Time Oracle Agent with ID: {agent_id}")

    try:
        config_manager = ConfigManager(config_path='config/config.yaml')
        await config_manager.load_config()
        app_config = config_manager.get_config()
        
        sdk_cfg = SDKConfig(
            agent_id=agent_id,
            agent_type="TimeOracleAgent",
            api_key=app_config.sdk_settings.api_key if app_config.sdk_settings else "test_api_key",
            lobby_http_url=app_config.lobby.http_url if app_config.lobby else "http://localhost:8080",
            lobby_ws_url=app_config.lobby.ws_url if app_config.lobby else "ws://localhost:8081",
            capabilities=[{"name": "request_time", "description": "Can request the current time."}]
        )
    except Exception as e:
        logging.error(f"Failed to load configuration for SDK: {e}")
        logging.info("Attempting to use default SDKConfig values for testing with mock_lobby.")
        sdk_cfg = SDKConfig(
            agent_id=agent_id,
            agent_type="TimeOracleAgent",
            api_key="test_api_key",
            lobby_http_url="http://localhost:8088",
            lobby_ws_url="ws://localhost:8081",
            capabilities=[{"name": "request_time", "description": "Can request the current time."}]
        )

    client = EnhancedEcosystemClient(sdk_config=sdk_cfg)
    conversation_id_for_time_request = str(uuid.uuid4())

    try:
        logging.info("Attempting to register and connect to the lobby...")
        await client.start()

        if client.is_connected():
            logging.info("Successfully connected to the lobby.")

            time_request_message = client.create_message(
                receiver_id=client.lobby_id if client.lobby_id else "mock_lobby_001",
                message_type="REQUEST",
                payload={"action": "get_time"},
                conversation_id=conversation_id_for_time_request
            )
            
            logging.info(f"Sending time request message: {time_request_message.payload}")
            await client.send_message(time_request_message)

            async def on_time_response(message: Message):
                if message.message_type == "RESPONSE" and message.conversation_id == conversation_id_for_time_request:
                    logging.info(f"SUCCESS: Received time response: {message.payload}")
                elif message.message_type == "ERROR" and message.conversation_id == conversation_id_for_time_request:
                    logging.error(f"Received ERROR from lobby for time request: {message.payload}")
                else:
                    logging.debug(f"Received other message: {message.payload}")
            
            # Register handlers for RESPONSE and ERROR to catch the reply to our request
            client.register_message_handler("RESPONSE", on_time_response)
            client.register_message_handler("ERROR", on_time_response) 

            await asyncio.sleep(5) # Keep alive to receive response

        else:
            logging.error("Failed to connect to the lobby.")

    except Exception as e:
        logging.error(f"An error occurred during agent operation: {e}", exc_info=True)
    finally:
        logging.info("Shutting down Time Oracle Agent.")
        await client.stop()

if __name__ == "__main__":
    asyncio.run(run_time_oracle_agent()) 