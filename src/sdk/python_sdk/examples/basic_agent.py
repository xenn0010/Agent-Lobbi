#!/usr/bin/env python3
"""
Basic Agent Example - Agent Lobbi SDK

This example demonstrates how to create a simple agent that can:
1. Register with the Agent Lobbi
2. Handle incoming messages/tasks
3. Respond to requests

Usage:
    python basic_agent.py
"""

import asyncio
import logging
from python_sdk import Agent, Capability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the basic agent example."""
    
    # Define agent capabilities
    capabilities = [
        Capability(
            name="echo",
            description="Echoes back the input message",
            input_schema={"message": "string"},
            output_schema={"echoed": "string"},
            tags=["utility", "testing"]
        ),
        Capability(
            name="uppercase",
            description="Converts text to uppercase",
            input_schema={"text": "string"},
            output_schema={"result": "string"},
            tags=["text", "transformation"]
        ),
        Capability(
            name="count_words",
            description="Counts words in text",
            input_schema={"text": "string"},
            output_schema={"word_count": "integer"},
            tags=["text", "analysis"]
        )
    ]
    
    # Create agent instance
    agent = Agent(
        api_key="demo_api_key_12345",
        agent_type="BasicUtilityAgent",
        capabilities=capabilities,
        agent_id="basic_agent_001",
        lobby_url="http://localhost:8092",
        debug=True
    )
    
    # Define message handler
    @agent.on_message
    async def handle_message(message):
        """Handle incoming messages and tasks."""
        logger.info(f"Received message: {message.message_type.name}")
        logger.info(f"Payload: {message.payload}")
        
        try:
            action = message.payload.get("action")
            
            if action == "echo":
                # Echo back the message
                input_message = message.payload.get("message", "No message provided")
                return {
                    "success": True,
                    "result": {"echoed": input_message},
                    "message": "Message echoed successfully"
                }
            
            elif action == "uppercase":
                # Convert text to uppercase
                text = message.payload.get("text", "")
                return {
                    "success": True,
                    "result": {"result": text.upper()},
                    "message": "Text converted to uppercase"
                }
            
            elif action == "count_words":
                # Count words in text
                text = message.payload.get("text", "")
                word_count = len(text.split())
                return {
                    "success": True,
                    "result": {"word_count": word_count},
                    "message": f"Counted {word_count} words"
                }
            
            else:
                # Unknown action
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": ["echo", "uppercase", "count_words"]
                }
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Internal error occurred"
            }
    
    # Start the agent
    logger.info("Starting Basic Agent...")
    try:
        success = await agent.start()
        if success:
            logger.info("OK Agent started successfully!")
            logger.info("Agent is now listening for tasks...")
            
            # Keep the agent running
            while True:
                await asyncio.sleep(1)
                
        else:
            logger.error("ERROR Failed to start agent")
            return
            
    except KeyboardInterrupt:
        logger.info("STOP Shutting down agent...")
    except Exception as e:
        logger.error(f"ERROR Agent error: {e}")
    finally:
        await agent.stop()
        logger.info("Agent stopped.")

if __name__ == "__main__":
    asyncio.run(main()) 