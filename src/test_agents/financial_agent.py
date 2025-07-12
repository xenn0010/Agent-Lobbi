
import asyncio
import json
import logging
import uuid
import sys
import os
import websockets

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.sdk.agent_lobby_sdk import AgentLobbySDK, Message, MessageType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FinancialAgent")

class FinancialAgent:
    """A mock agent that performs financial analysis."""
    
    def __init__(self, agent_id, lobby_host="localhost", http_port=8080, ws_port=8081):
        self.agent_id = agent_id
        self.sdk = AgentLobbySDK(
            lobby_host=lobby_host,
            http_port=http_port,
            ws_port=ws_port,
            agent_id=self.agent_id,
            enable_security=False
        )
        self.sdk.task_handler = self.handle_task

    async def handle_task(self, message: Message):
        """Handles financial analysis tasks."""
        task_payload = message.payload
        task_name = task_payload.get("task_title", "Unknown Task")
        logger.info(f"Received task: {task_name}")

        # Simulate financial analysis
        await asyncio.sleep(2) 
        
        analysis_result = {
            "ticker": "META",
            "period": "Q4 2024",
            "revenue": 35.8,
            "net_income": 12.3,
            "eps": 4.13,
            "summary": "Strong performance driven by ad revenue growth and operational efficiency.",
            "outlook": "Positive, with continued investment in AI and Metaverse."
        }
        
        # Create a response message
        response_payload = {
            "status": "completed",
            "result": analysis_result,
            "task_id": task_payload.get("task_id")
        }
        
        response_message = Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id, # Respond to the original sender (the lobby)
            message_type=MessageType.RESPONSE,
            payload=response_payload,
            conversation_id=message.conversation_id
        )
        
        await self.sdk._send_message_via_websocket(response_message)
        logger.info(f"Sent analysis result for task: {task_name}")

    async def start(self):
        """Register and start the agent."""
        await self.sdk.register_agent(
            agent_id=self.agent_id,
            agent_type="FinancialAgent",
            capabilities=["financial_analysis", "data_analysis"]
        )
        logger.info("Financial Agent is running and listening for tasks.")
        # Keep the agent running
        await asyncio.Event().wait()

async def main():
    # Use environment variables for ports, with defaults
    http_port = int(os.getenv("LOBBY_HTTP_PORT", 8080))
    ws_port = int(os.getenv("LOBBY_WS_PORT", 8081))
    
    agent = FinancialAgent(
        agent_id="financial_agent_001",
        http_port=http_port,
        ws_port=ws_port
    )
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Financial Agent shutting down.")
    finally:
        if agent.sdk:
            await agent.sdk.stop()

if __name__ == "__main__":
    asyncio.run(main())
