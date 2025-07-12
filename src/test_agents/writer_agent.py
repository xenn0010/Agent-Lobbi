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
logger = logging.getLogger("WriterAgent")

class WriterAgent:
    """A mock agent that writes content based on input."""
    
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
        """Handles content creation tasks."""
        task_payload = message.payload
        task_name = task_payload.get("task_title", "Unknown Task")
        logger.info(f"Received task: {task_name}")

        # Get the analysis from the previous step
        analysis_data = task_payload.get('input_data', {}).get('Execute Capability: financial_analysis', {}).get('result', {})
        
        if not analysis_data:
            logger.error("Did not receive financial analysis data!")
            blog_post = "Error: Could not generate blog post due to missing financial analysis."
        else:
            # Simulate writing a blog post
            await asyncio.sleep(1)
            blog_post = (
                f"# META's Strong Q4 Performance\n\n"
                f"META has reported a robust financial performance for the fourth quarter of 2024. "
                f"The company's revenue reached ${analysis_data.get('revenue', 'N/A')}B, with a net income of ${analysis_data.get('net_income', 'N/A')}B. "
                f"This was driven by strong advertising growth and reflects a positive outlook for the company, which continues to invest heavily in AI and the Metaverse."
            )

        # Create a response message
        response_payload = {
            "status": "completed",
            "result": {"blog_post": blog_post},
            "task_id": task_payload.get("task_id")
        }
        
        response_message = Message(
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            payload=response_payload,
            conversation_id=message.conversation_id
        )
        
        await self.sdk._send_message_via_websocket(response_message)
        logger.info(f"Sent blog post for task: {task_name}")

    async def start(self):
        """Register and start the agent."""
        await self.sdk.register_agent(
            agent_id=self.agent_id,
            agent_type="WriterAgent",
            capabilities=["content_creation", "text_summarization"]
        )
        logger.info("Writer Agent is running and listening for tasks.")
        # Keep the agent running
        await asyncio.Event().wait()

async def main():
    # Use environment variables for ports, with defaults
    http_port = int(os.getenv("LOBBY_HTTP_PORT", 8080))
    ws_port = int(os.getenv("LOBBY_WS_PORT", 8081))
    
    agent = WriterAgent(
        agent_id="writer_agent_001",
        http_port=http_port,
        ws_port=ws_port
    )
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Writer Agent shutting down.")
    finally:
        if agent.sdk:
            await agent.sdk.stop()

if __name__ == "__main__":
    asyncio.run(main())