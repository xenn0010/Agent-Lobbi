#!/usr/bin/env python3
"""
Real-World GPT Agent - Connects to Agent Lobby for text analysis
"""
import asyncio
import sys
import os
from typing import Optional

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from sdk.ecosystem_sdk import EcosystemClient, Message, MessageType, AgentCapabilitySDK

# Mock GPT functionality (replace with real OpenAI API)
class MockGPTProcessor:
    def __init__(self):
        self.model = "gpt-4"
    
    def analyze_text(self, text: str) -> dict:
        """Mock text analysis - replace with real OpenAI API call"""
        # Simple analysis for demo
        word_count = len(text.split())
        sentiment = "positive" if any(word in text.lower() for word in ["good", "great", "amazing", "excellent"]) else "neutral"
        
        return {
            "word_count": word_count,
            "sentiment": sentiment,
            "summary": f"Analyzed {word_count} words with {sentiment} sentiment",
            "key_topics": ["technology", "innovation", "AI"] if "AI" in text else ["general"],
            "analysis_model": self.model
        }
    
    def generate_insights(self, data: dict) -> dict:
        """Generate insights from data"""
        return {
            "insights": [
                "Data shows positive sentiment trends",
                "Word count indicates detailed content",
                "Topics suggest technology focus"
            ],
            "recommendations": [
                "Continue positive messaging",
                "Expand on technical details",
                "Include more AI-related content"
            ]
        }

class GPTAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.gpt = MockGPTProcessor()
        self.processed_count = 0
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages from the lobby"""
        print(f"🤖 {self.agent_id}: Received message type {message.message_type.name}")
        print(f"📝 Payload: {message.payload}")
        
        # Handle capability requests
        if message.message_type == MessageType.REQUEST:
            capability = message.payload.get("capability_name")
            input_data = message.payload.get("input_data", {})
            
            if capability == "analyze_text":
                text = input_data.get("text", "")
                result = self.gpt.analyze_text(text)
                self.processed_count += 1
                
                print(f"✅ {self.agent_id}: Analyzed text (#{self.processed_count})")
                
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "status": "success",
                        "result": result,
                        "agent_id": self.agent_id,
                        "task_id": message.payload.get("task_id")
                    },
                    conversation_id=message.conversation_id
                )
            
            elif capability == "generate_insights":
                data = input_data.get("data", {})
                result = self.gpt.generate_insights(data)
                self.processed_count += 1
                
                print(f"💡 {self.agent_id}: Generated insights (#{self.processed_count})")
                
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "status": "success",
                        "result": result,
                        "agent_id": self.agent_id,
                        "task_id": message.payload.get("task_id")
                    },
                    conversation_id=message.conversation_id
                )
        
        elif message.message_type == MessageType.INFO:
            print(f"ℹ️ {self.agent_id}: Received info - {message.payload}")
        
        return None

async def run_gpt_agent():
    """Run the GPT agent"""
    agent_id = "gpt_analyst_001"
    agent = GPTAgent(agent_id)
    
    print(f"🚀 Starting {agent_id}...")
    
    # Define capabilities
    capabilities = [
        AgentCapabilitySDK(
            name="analyze_text",
            description="Analyze text for sentiment, topics, and insights using GPT-4",
            input_schema={"text": "string"},
            output_schema={"word_count": "number", "sentiment": "string", "summary": "string"}
        ),
        AgentCapabilitySDK(
            name="generate_insights", 
            description="Generate strategic insights from data analysis",
            input_schema={"data": "object"},
            output_schema={"insights": "array", "recommendations": "array"}
        )
    ]
    
    # Create SDK client
    sdk_client = EcosystemClient(
        agent_id=agent_id,
        agent_type="LLMAnalyst",
        capabilities=capabilities,
        lobby_http_url="http://localhost:8092",
        lobby_ws_url="ws://localhost:8091",
        agent_message_handler=agent.handle_message
    )
    
    try:
        # Start the agent
        success = await sdk_client.start("test_api_key")
        if success:
            print(f"✅ {agent_id} connected successfully!")
            print(f"🎯 Ready to analyze text and generate insights")
            
            # Keep running and processing messages
            while True:
                await asyncio.sleep(1)
                if sdk_client._should_stop:
                    break
        else:
            print(f"❌ {agent_id} failed to connect")
            
    except KeyboardInterrupt:
        print(f"\n🛑 {agent_id} shutting down...")
    finally:
        await sdk_client.stop()
        print(f"👋 {agent_id} disconnected")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 REAL WORLD GPT AGENT")
    print("=" * 60)
    print("This agent connects to your running mock lobby and provides:")
    print("• Text analysis capabilities")
    print("• Insight generation")
    print("• Real-time collaboration with other agents")
    print("=" * 60)
    
    asyncio.run(run_gpt_agent()) 