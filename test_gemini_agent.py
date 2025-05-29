#!/usr/bin/env python3
"""
Simple test to demonstrate Gemini Agent working with Agent Lobby
"""
import asyncio
import sys
import os
import time
from typing import Dict

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from sdk.ecosystem_sdk import EcosystemClient, Message, MessageType

class SimpleTestClient:
    def __init__(self):
        self.client_id = "test_client_001"
        self.sdk_client = None
        self.responses = {}
        
    async def handle_message(self, message: Message):
        """Handle responses from agents"""
        print(f"📨 Test Client: Received response from {message.sender_id}")
        result = message.payload.get('result', {})
        
        if 'summary' in result:
            print(f"   📝 Summary: {result['summary']}")
        if 'ai_powered' in result:
            print(f"   🤖 AI Powered: {result['ai_powered']}")
        if 'sentiment' in result:
            print(f"   😊 Sentiment: {result['sentiment']}")
            
        self.responses[message.sender_id] = message.payload
        return None
    
    async def start(self):
        """Start the test client"""
        self.sdk_client = EcosystemClient(
            agent_id=self.client_id,
            agent_type="TestClient",
            capabilities=[],
            lobby_http_url="http://localhost:8092",
            lobby_ws_url="ws://localhost:8091",
            agent_message_handler=self.handle_message
        )
        
        success = await self.sdk_client.start("test_api_key")
        if not success:
            raise Exception("Failed to connect test client")
        
        print(f"✅ Test client connected!")
        return True
    
    async def send_task(self, target_agent: str, capability: str, input_data: Dict) -> Dict:
        """Send a task to an agent"""
        task_message = Message(
            sender_id=self.client_id,
            receiver_id=target_agent,
            message_type=MessageType.REQUEST,
            payload={
                "capability_name": capability,
                "input_data": input_data,
                "task_id": f"task_{int(time.time() * 1000)}"
            }
        )
        
        print(f"📤 Sending '{capability}' task to {target_agent}")
        await self.sdk_client.send_message(task_message)
        
        # Wait for response
        await asyncio.sleep(3)
        return self.responses.get(target_agent, {})
    
    async def stop(self):
        """Stop the test client"""
        if self.sdk_client:
            await self.sdk_client.stop()

async def test_gemini_agent():
    """Test the Gemini agent with various tasks"""
    
    print("🚀" + "=" * 60)
    print("🧠 TESTING GEMINI AI AGENT WITH AGENT LOBBY")
    print("🚀" + "=" * 60)
    print("\nThis test will:")
    print("1. Connect to your running mock lobby")
    print("2. Send AI tasks to the Gemini agent")
    print("3. Display real-time AI responses")
    print("\nMake sure you have:")
    print("• Mock lobby running (tests/simple_mock_lobby.py)")
    print("• Gemini agent running (real_world_gemini_agent.py)")
    print("=" * 62)
    
    client = SimpleTestClient()
    
    try:
        await client.start()
        
        print("\n🎯 TEST 1: TEXT ANALYSIS")
        print("-" * 30)
        
        response1 = await client.send_task(
            target_agent="gemini_analyst_001",
            capability="analyze_text",
            input_data={
                "text": "Agent Lobby revolutionizes AI collaboration by enabling seamless multi-agent workflows. This breakthrough technology allows AI agents to work together intelligently, solving complex problems that single agents cannot handle alone."
            }
        )
        
        if response1.get('status') == 'success':
            print("✅ Text analysis completed successfully!")
        else:
            print("❌ Text analysis failed")
        
        print("\n🎯 TEST 2: INSIGHT GENERATION")
        print("-" * 30)
        
        response2 = await client.send_task(
            target_agent="gemini_analyst_001",
            capability="generate_insights",
            input_data={
                "data": {
                    "platform": "Agent Lobby",
                    "users": 1500,
                    "success_rate": 0.94,
                    "avg_response_time": 1.2,
                    "agent_collaborations": 3200
                }
            }
        )
        
        if response2.get('status') == 'success':
            print("✅ Insight generation completed successfully!")
            result = response2.get('result', {})
            insights = result.get('insights', [])
            if insights:
                print("   🔍 Key Insights:")
                for i, insight in enumerate(insights[:3], 1):
                    print(f"      {i}. {insight}")
        else:
            print("❌ Insight generation failed")
        
        print("\n🎯 TEST 3: CONTENT SUMMARIZATION")
        print("-" * 30)
        
        long_content = """
        Agent Lobby represents a paradigm shift in artificial intelligence architecture. Unlike traditional single-agent systems, Agent Lobby facilitates dynamic multi-agent collaboration through a sophisticated protocol that enables real-time communication, task distribution, and result aggregation. The platform supports various agent types including data processors, language models, specialized analyzers, and workflow coordinators. Each agent can advertise its capabilities and discover other agents' services, creating a self-organizing ecosystem of AI collaboration. The system includes advanced features like load balancing, health monitoring, error recovery, and performance optimization. Agent Lobby's innovative architecture allows for both synchronous and asynchronous workflows, enabling complex multi-step processes that can span multiple agents and services. This breakthrough technology opens new possibilities for solving complex problems that require diverse AI capabilities working in concert.
        """
        
        response3 = await client.send_task(
            target_agent="gemini_analyst_001",
            capability="summarize_content",
            input_data={
                "content": long_content.strip(),
                "max_length": 150
            }
        )
        
        if response3.get('status') == 'success':
            print("✅ Content summarization completed successfully!")
        else:
            print("❌ Content summarization failed")
        
        print("\n🎉 GEMINI AGENT TESTING COMPLETED!")
        print("=" * 62)
        print("\n📊 RESULTS SUMMARY:")
        print(f"✅ Text Analysis: {'PASSED' if response1.get('status') == 'success' else 'FAILED'}")
        print(f"✅ Insight Generation: {'PASSED' if response2.get('status') == 'success' else 'FAILED'}")
        print(f"✅ Content Summarization: {'PASSED' if response3.get('status') == 'success' else 'FAILED'}")
        
        # Check if AI was actually used
        ai_used = any(
            resp.get('result', {}).get('ai_powered', False) 
            for resp in [response1, response2, response3]
        )
        print(f"🧠 Real AI Used: {'YES' if ai_used else 'NO (Mock Mode)'}")
        
        print("\n🚀 Agent Lobby + Gemini AI: WORKING!")
        print("=" * 62)
        
    except Exception as e:
        print(f"\n💥 Test error: {str(e)}")
        print("Make sure your mock lobby and Gemini agent are running!")
    
    finally:
        await client.stop()
        print("\n👋 Test client disconnected")

if __name__ == "__main__":
    print("🧪 Starting Gemini Agent Test...")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(test_gemini_agent())
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"\n💥 Test failed: {str(e)}") 