#!/usr/bin/env python3
"""
Real-World Gemini Agent - Connects to Agent Lobby for AI-powered text analysis
"""
import asyncio
import sys
import os
import json
from typing import Optional

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from sdk.ecosystem_sdk import EcosystemClient, Message, MessageType, AgentCapabilitySDK

# Gemini AI functionality
class GeminiProcessor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = "gemini-1.5-flash"
        
        if not self.api_key:
            print("⚠️ No Gemini API key found. Set GEMINI_API_KEY environment variable or pass api_key parameter")
            print("   Falling back to mock responses for demo purposes")
            self.use_mock = True
        else:
            self.use_mock = False
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                print(f"✅ Gemini AI initialized with model: {self.model}")
            except ImportError:
                print("📦 Installing google-generativeai...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai"])
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
                print(f"✅ Gemini AI installed and initialized with model: {self.model}")
    
    async def analyze_text(self, text: str) -> dict:
        """Analyze text using Gemini AI or mock response"""
        if self.use_mock:
            return self._mock_analyze_text(text)
        
        try:
            prompt = f"""
            Analyze the following text and provide a JSON response with:
            1. word_count: number of words
            2. sentiment: positive, negative, or neutral
            3. summary: brief summary of the content
            4. key_topics: array of main topics/themes
            5. complexity_score: 1-10 scale of text complexity
            6. language_quality: assessment of writing quality
            
            Text to analyze: "{text}"
            
            Respond only with valid JSON.
            """
            
            response = self.client.generate_content(prompt)
            
            # Parse the JSON response
            try:
                result = json.loads(response.text)
                result["analysis_model"] = self.model
                result["ai_powered"] = True
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, create structured response
                return {
                    "word_count": len(text.split()),
                    "sentiment": "neutral",
                    "summary": response.text[:200] + "..." if len(response.text) > 200 else response.text,
                    "key_topics": ["general"],
                    "complexity_score": 5,
                    "language_quality": "good",
                    "analysis_model": self.model,
                    "ai_powered": True,
                    "raw_response": response.text
                }
                
        except Exception as e:
            print(f"🚨 Gemini API error: {str(e)}")
            return self._mock_analyze_text(text)
    
    async def generate_insights(self, data: dict) -> dict:
        """Generate strategic insights using Gemini AI"""
        if self.use_mock:
            return self._mock_generate_insights(data)
        
        try:
            prompt = f"""
            Based on the following data analysis, generate strategic insights and recommendations.
            Provide a JSON response with:
            1. insights: array of key insights (3-5 items)
            2. recommendations: array of actionable recommendations (3-5 items)
            3. risk_factors: potential risks or concerns
            4. opportunities: potential opportunities identified
            5. confidence_score: 1-10 confidence in the analysis
            
            Data: {json.dumps(data, indent=2)}
            
            Respond only with valid JSON.
            """
            
            response = self.client.generate_content(prompt)
            
            try:
                result = json.loads(response.text)
                result["analysis_model"] = self.model
                result["ai_powered"] = True
                return result
            except json.JSONDecodeError:
                return {
                    "insights": [response.text[:100] + "..."],
                    "recommendations": ["Based on analysis, consider strategic planning"],
                    "risk_factors": ["Data quality should be monitored"],
                    "opportunities": ["Potential for optimization exists"],
                    "confidence_score": 7,
                    "analysis_model": self.model,
                    "ai_powered": True,
                    "raw_response": response.text
                }
                
        except Exception as e:
            print(f"🚨 Gemini API error: {str(e)}")
            return self._mock_generate_insights(data)
    
    async def summarize_content(self, content: str, max_length: int = 200) -> dict:
        """Summarize content using Gemini AI"""
        if self.use_mock:
            return {"summary": content[:max_length] + "...", "ai_powered": False}
        
        try:
            prompt = f"""
            Summarize the following content in {max_length} characters or less.
            Provide a JSON response with:
            1. summary: concise summary
            2. key_points: array of main points (3-5 items)
            3. original_length: character count of original
            4. compression_ratio: how much was compressed
            
            Content: "{content}"
            
            Respond only with valid JSON.
            """
            
            response = self.client.generate_content(prompt)
            
            try:
                result = json.loads(response.text)
                result["analysis_model"] = self.model
                result["ai_powered"] = True
                return result
            except json.JSONDecodeError:
                summary = response.text[:max_length]
                return {
                    "summary": summary,
                    "key_points": [summary[:50] + "..."],
                    "original_length": len(content),
                    "compression_ratio": len(summary) / len(content),
                    "analysis_model": self.model,
                    "ai_powered": True
                }
                
        except Exception as e:
            print(f"🚨 Gemini API error: {str(e)}")
            return {"summary": content[:max_length] + "...", "ai_powered": False}
    
    def _mock_analyze_text(self, text: str) -> dict:
        """Mock text analysis for demo when API is not available"""
        word_count = len(text.split())
        sentiment = "positive" if any(word in text.lower() for word in ["good", "great", "amazing", "excellent"]) else "neutral"
        
        return {
            "word_count": word_count,
            "sentiment": sentiment,
            "summary": f"Mock analysis: {word_count} words with {sentiment} sentiment",
            "key_topics": ["technology", "AI"] if "AI" in text else ["general"],
            "complexity_score": min(word_count // 10, 10),
            "language_quality": "good",
            "analysis_model": "mock",
            "ai_powered": False
        }
    
    def _mock_generate_insights(self, data: dict) -> dict:
        """Mock insight generation for demo"""
        return {
            "insights": [
                "Mock insight: Data shows positive trends",
                "Mock insight: Content complexity is moderate",
                "Mock insight: Technology focus is evident"
            ],
            "recommendations": [
                "Continue positive messaging strategy",
                "Expand technical detail where appropriate",
                "Monitor sentiment trends over time"
            ],
            "risk_factors": ["Data sample size may be limited"],
            "opportunities": ["Potential for content optimization"],
            "confidence_score": 6,
            "analysis_model": "mock",
            "ai_powered": False
        }

class GeminiAgent:
    def __init__(self, agent_id: str, gemini_api_key: str = None):
        self.agent_id = agent_id
        self.gemini = GeminiProcessor(gemini_api_key)
        self.processed_count = 0
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages from the lobby"""
        print(f"🤖 {self.agent_id}: Received message type {message.message_type.name}")
        print(f"📝 Payload: {message.payload}")
        
        # Handle capability requests
        if message.message_type == MessageType.REQUEST:
            capability = message.payload.get("capability_name")
            input_data = message.payload.get("input_data", {})
            
            try:
                if capability == "analyze_text":
                    text = input_data.get("text", "")
                    result = await self.gemini.analyze_text(text)
                    self.processed_count += 1
                    
                    print(f"✅ {self.agent_id}: Analyzed text (#{self.processed_count}) - AI: {result.get('ai_powered', False)}")
                    
                elif capability == "generate_insights":
                    data = input_data.get("data", {})
                    result = await self.gemini.generate_insights(data)
                    self.processed_count += 1
                    
                    print(f"💡 {self.agent_id}: Generated insights (#{self.processed_count}) - AI: {result.get('ai_powered', False)}")
                    
                elif capability == "summarize_content":
                    content = input_data.get("content", "")
                    max_length = input_data.get("max_length", 200)
                    result = await self.gemini.summarize_content(content, max_length)
                    self.processed_count += 1
                    
                    print(f"📄 {self.agent_id}: Summarized content (#{self.processed_count}) - AI: {result.get('ai_powered', False)}")
                    
                else:
                    print(f"❌ {self.agent_id}: Unknown capability '{capability}'")
                    return Message(
                        sender_id=self.agent_id,
                        receiver_id=message.sender_id,
                        message_type=MessageType.ERROR,
                        payload={"error": f"Unknown capability: {capability}"},
                        conversation_id=message.conversation_id
                    )
                
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
                
            except Exception as e:
                print(f"💥 {self.agent_id}: Error processing {capability}: {str(e)}")
                return Message(
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={
                        "error": str(e),
                        "capability": capability,
                        "agent_id": self.agent_id
                    },
                    conversation_id=message.conversation_id
                )
        
        elif message.message_type == MessageType.INFO:
            print(f"ℹ️ {self.agent_id}: Received info - {message.payload}")
        
        return None

async def run_gemini_agent():
    """Run the Gemini-powered agent"""
    agent_id = "gemini_analyst_001"
    
    # Get Gemini API key from environment or prompt user
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("\n🔑 Gemini API Key Setup:")
        print("Set your Gemini API key as an environment variable:")
        print("   export GEMINI_API_KEY='your_api_key_here'")
        print("Or get one at: https://aistudio.google.com/app/apikey")
        print("\n⚠️ Running in MOCK mode for demo...")
    
    agent = GeminiAgent(agent_id, gemini_api_key)
    
    print(f"🚀 Starting {agent_id}...")
    
    # Define capabilities
    capabilities = [
        AgentCapabilitySDK(
            name="analyze_text",
            description="AI-powered text analysis using Gemini for sentiment, topics, and complexity",
            input_schema={"text": "string"},
            output_schema={"word_count": "number", "sentiment": "string", "summary": "string", "ai_powered": "boolean"}
        ),
        AgentCapabilitySDK(
            name="generate_insights", 
            description="AI-powered strategic insights and recommendations using Gemini",
            input_schema={"data": "object"},
            output_schema={"insights": "array", "recommendations": "array", "confidence_score": "number"}
        ),
        AgentCapabilitySDK(
            name="summarize_content",
            description="AI-powered content summarization using Gemini",
            input_schema={"content": "string", "max_length": "number"},
            output_schema={"summary": "string", "key_points": "array", "compression_ratio": "number"}
        )
    ]
    
    # Create SDK client with correct API key
    sdk_client = EcosystemClient(
        agent_id=agent_id,
        agent_type="GeminiAIAnalyst",
        capabilities=capabilities,
        lobby_http_url="http://localhost:8092",
        lobby_ws_url="ws://localhost:8091",
        agent_message_handler=agent.handle_message
    )
    
    try:
        # Start the agent with the correct API key for the mock lobby
        success = await sdk_client.start("test_api_key")  # This is the key the mock lobby expects
        if success:
            print(f"✅ {agent_id} connected successfully!")
            print(f"🧠 Ready to provide AI-powered analysis with Gemini")
            print(f"🔄 AI Mode: {'REAL' if not agent.gemini.use_mock else 'MOCK'}")
            
            # Keep running and processing messages
            while True:
                await asyncio.sleep(1)
        else:
            print(f"❌ {agent_id} failed to connect")
            
    except KeyboardInterrupt:
        print(f"\n🛑 {agent_id} shutting down...")
    finally:
        await sdk_client.stop()
        print(f"👋 {agent_id} disconnected")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 REAL WORLD GEMINI AI AGENT")
    print("=" * 60)
    print("This agent connects to your running mock lobby and provides:")
    print("• AI-powered text analysis using Google Gemini")
    print("• Strategic insights and recommendations")
    print("• Content summarization")
    print("• Real-time collaboration with other agents")
    print("=" * 60)
    
    asyncio.run(run_gemini_agent()) 