#!/usr/bin/env python3
"""
Task Delegation Example - Agent Lobbi SDK

This example demonstrates how to use the AgentLobbiClient to:
1. Check Agent Lobbi health
2. List available agents
3. Delegate tasks to agents
4. Monitor task progress

Usage:
    python task_delegation.py
"""

import asyncio
import logging
from python_sdk import AgentLobbiClient, quick_task_delegation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main function to demonstrate task delegation."""
    
    # Initialize client
    api_key = "demo_api_key_12345"
    lobby_url = "http://localhost:8092"
    
    logger.info("START Starting Task Delegation Example")
    
    try:
        # Use AgentLobbiClient as context manager
        async with AgentLobbiClient(api_key, lobby_url) as client:
            
            # 1. Check Agent Lobbi health
            logger.info("1️⃣ Checking Agent Lobbi health...")
            try:
                health = await client.health_check()
                logger.info(f"OK Agent Lobbi is healthy: {health}")
            except Exception as e:
                logger.error(f"ERROR Health check failed: {e}")
                return
            
            # 2. List available agents
            logger.info("2️⃣ Listing available agents...")
            try:
                agents = await client.list_agents()
                logger.info(f"TASK Found {len(agents)} agents:")
                for agent in agents:
                    logger.info(f"  - {agent.get('agent_id', 'Unknown')} ({agent.get('agent_type', 'Unknown')})")
                    if 'capabilities' in agent:
                        caps = agent['capabilities']
                        logger.info(f"    Capabilities: {[cap.get('name', 'Unknown') for cap in caps]}")
            except Exception as e:
                logger.error(f"ERROR Failed to list agents: {e}")
                agents = []
            
            # 3. Delegate various tasks
            if agents:
                logger.info("3️⃣ Delegating tasks to agents...")
                
                # Task 1: Echo task
                try:
                    logger.info("SEND Delegating echo task...")
                    echo_result = await client.delegate_task(
                        task_name="Echo Test",
                        task_description="Test the echo capability",
                        required_capabilities=["echo"],
                        task_data={"action": "echo", "message": "Hello from task delegation!"},
                        max_agents=1,
                        timeout_minutes=5
                    )
                    logger.info(f"OK Echo task delegated: {echo_result}")
                except Exception as e:
                    logger.error(f"ERROR Echo task delegation failed: {e}")
                
                # Task 2: Text processing task
                try:
                    logger.info("SEND Delegating text processing task...")
                    text_result = await client.delegate_task(
                        task_name="Text Processing",
                        task_description="Process text with multiple operations",
                        required_capabilities=["uppercase", "count_words"],
                        task_data={
                            "action": "uppercase",
                            "text": "this is a sample text for processing"
                        },
                        max_agents=1,
                        timeout_minutes=10
                    )
                    logger.info(f"OK Text processing task delegated: {text_result}")
                except Exception as e:
                    logger.error(f"ERROR Text processing task delegation failed: {e}")
                
                # Task 3: Complex multi-step task
                try:
                    logger.info("SEND Delegating complex multi-step task...")
                    complex_result = await client.delegate_task(
                        task_name="Complex Analysis",
                        task_description="Perform complex text analysis with multiple steps",
                        required_capabilities=["echo", "uppercase", "count_words"],
                        task_data={
                            "steps": [
                                {"action": "echo", "message": "Starting analysis..."},
                                {"action": "count_words", "text": "The quick brown fox jumps over the lazy dog"},
                                {"action": "uppercase", "text": "convert this to uppercase"}
                            ]
                        },
                        max_agents=2,
                        timeout_minutes=15
                    )
                    logger.info(f"OK Complex task delegated: {complex_result}")
                except Exception as e:
                    logger.error(f"ERROR Complex task delegation failed: {e}")
                
            else:
                logger.warning("WARNING No agents available for task delegation")
            
            # 4. Demonstrate quick task delegation
            logger.info("4️⃣ Demonstrating quick task delegation...")
            try:
                quick_result = await quick_task_delegation(
                    api_key=api_key,
                    task_name="Quick Echo",
                    task_description="Quick echo test using convenience function",
                    required_capabilities=["echo"],
                    lobby_url=lobby_url,
                    task_data={"action": "echo", "message": "Quick delegation test!"},
                    max_agents=1
                )
                logger.info(f"OK Quick task delegation result: {quick_result}")
            except Exception as e:
                logger.error(f"ERROR Quick task delegation failed: {e}")
    
    except Exception as e:
        logger.error(f"ERROR Client error: {e}")
    
    logger.info("🏁 Task Delegation Example completed")

async def monitor_task_example():
    """Example of monitoring task progress."""
    
    api_key = "demo_api_key_12345"
    lobby_url = "http://localhost:8092"
    
    logger.info("INFO Starting Task Monitoring Example")
    
    try:
        async with AgentLobbyClient(api_key, lobby_url) as client:
            
            # Delegate a task
            result = await client.delegate_task(
                task_name="Monitored Task",
                task_description="Task that will be monitored",
                required_capabilities=["echo"],
                task_data={"action": "echo", "message": "Monitor this task!"},
                timeout_minutes=5
            )
            
            if "data" in result and "delegation_id" in result["data"]:
                task_id = result["data"]["delegation_id"]
                logger.info(f"TASK Monitoring task: {task_id}")
                
                # Monitor task progress
                for i in range(10):  # Check 10 times
                    try:
                        status = await client.get_task_status(task_id)
                        logger.info(f"INFO Task status ({i+1}/10): {status}")
                        
                        if status.get("status") in ["completed", "failed"]:
                            logger.info("OK Task finished!")
                            break
                        
                        await asyncio.sleep(2)  # Wait 2 seconds between checks
                        
                    except Exception as e:
                        logger.error(f"ERROR Failed to get task status: {e}")
                        break
            else:
                logger.error("ERROR No task ID returned from delegation")
                
    except Exception as e:
        logger.error(f"ERROR Monitoring error: {e}")

if __name__ == "__main__":
    # Run the main example
    asyncio.run(main())
    
    # Optionally run the monitoring example
    # asyncio.run(monitor_task_example()) 