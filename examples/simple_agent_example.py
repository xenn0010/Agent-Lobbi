#!/usr/bin/env python3
"""
Simple Agent Example - Agent Lobby Python SDK

This example shows how easy it is to create and deploy an AI agent
to the Agent Lobby ecosystem with just a few lines of code.

Requirements:
1. Get your API key from the Agent Lobby dashboard
2. Install the SDK: pip install agent-lobby-sdk
3. Run this script!
"""

import asyncio
import sys
import os

# Add the SDK to the path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'sdk', 'python_sdk'))

from python_sdk import Agent, Capability, Message, MessageType

async def main():
    """Main function demonstrating the simple agent integration."""
    
    # Step 1: Define your agent's capabilities
    capabilities = [
        Capability(
            name="greet_user",
            description="Greets users in different languages",
            input_schema={"name": "string", "language": "string"},
            output_schema={"greeting": "string", "language": "string"}
        ),
        Capability(
            name="calculate_math",
            description="Performs basic mathematical calculations",
            input_schema={"operation": "string", "a": "number", "b": "number"},
            output_schema={"result": "number", "operation": "string"}
        )
    ]
    
    # Step 2: Create your agent
    agent = Agent(
        api_key="test_api_key",  # Replace with your actual API key
        agent_type="ExampleBot",
        capabilities=capabilities,
        agent_id="example_bot_001",  # Optional: auto-generated if not provided
        lobby_url="http://localhost:8092",  # Use your lobby URL
        debug=True  # Enable debug logging
    )
    
    # Step 3: Define how your agent handles messages
    @agent.on_message
    async def handle_message(message: Message):
        """Handle incoming messages and requests."""
        print(f"📨 Received {message.message_type.name} from {message.sender_id}")
        print(f"📋 Payload: {message.payload}")
        
        # Handle different types of requests
        action = message.payload.get("action")
        
        if action == "greet_user":
            name = message.payload.get("name", "Friend")
            language = message.payload.get("language", "en")
            
            greetings = {
                "en": f"Hello, {name}!",
                "es": f"¡Hola, {name}!",
                "fr": f"Bonjour, {name}!",
                "de": f"Hallo, {name}!",
                "it": f"Ciao, {name}!"
            }
            
            greeting = greetings.get(language, greetings["en"])
            
            return {
                "greeting": greeting,
                "language": language,
                "status": "success"
            }
        
        elif action == "calculate_math":
            operation = message.payload.get("operation")
            a = message.payload.get("a", 0)
            b = message.payload.get("b", 0)
            
            try:
                if operation == "add":
                    result = a + b
                elif operation == "subtract":
                    result = a - b
                elif operation == "multiply":
                    result = a * b
                elif operation == "divide":
                    result = a / b if b != 0 else "Error: Division by zero"
                else:
                    return {"error": f"Unknown operation: {operation}"}
                
                return {
                    "result": result,
                    "operation": f"{a} {operation} {b}",
                    "status": "success"
                }
            
            except Exception as e:
                return {"error": str(e)}
        
        elif message.message_type == MessageType.INFO:
            # Just acknowledge INFO messages
            print(f"ℹ️  Info message received: {message.payload}")
            return None  # No response needed for INFO messages
        
        else:
            # Unknown action
            return {
                "error": f"Unknown action: {action}",
                "available_actions": ["greet_user", "calculate_math"]
            }
    
    # Step 4: Start your agent
    print("🚀 Starting Example Bot...")
    print("📡 Connecting to Agent Lobby...")
    
    success = await agent.start()
    
    if success:
        print("✅ Agent started successfully!")
        print("🎯 Agent is now ready to receive requests")
        print("📝 Available capabilities:")
        for cap in capabilities:
            print(f"   - {cap.name}: {cap.description}")
        
        print("\n💡 Try sending requests to test the agent:")
        print("   - greet_user: {'action': 'greet_user', 'name': 'Alice', 'language': 'es'}")
        print("   - calculate_math: {'action': 'calculate_math', 'operation': 'add', 'a': 5, 'b': 3}")
        
        # Keep the agent running
        try:
            print("\n⏳ Agent is running... Press Ctrl+C to stop")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping agent...")
    else:
        print("❌ Failed to start agent")
    
    # Step 5: Clean shutdown
    await agent.stop()
    print("👋 Agent stopped. Goodbye!")

if __name__ == "__main__":
    # Run the example
    print("=" * 60)
    print("🤖 Agent Lobby - Simple Agent Example")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example terminated by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 