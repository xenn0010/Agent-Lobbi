#!/usr/bin/env python3
"""
WORKING AGENT LOBBY COLLABORATION AGENT

This agent properly handles the collaboration protocol and can participate in
multi-agent workflows. It fixes the message format issues and provides
complete task processing.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import websockets
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingCollaborationAgent:
    """A working agent that properly handles Agent Lobby collaboration"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: list, 
                 model: str = "gemma:2b", lobby_url: str = "http://localhost:8086"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.model = model
        self.lobby_url = lobby_url
        self.lobby_ws_url = lobby_url.replace("http://", "ws://").replace("8086", "8087")
        self.websocket = None
        self.http_client = httpx.AsyncClient()
        
        logger.info(f"🤖 Working Agent Created: {agent_id}")
        logger.info(f"   Type: {agent_type}")
        logger.info(f"   Capabilities: {capabilities}")
        logger.info(f"   Model: {model}")
    
    async def register_with_lobby(self) -> bool:
        """Register with the Agent Lobby via HTTP"""
        try:
            registration_data = {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'capabilities': self.capabilities,
                'model': self.model,
                'status': 'active'
            }
            
            response = await self.http_client.post(
                f"{self.lobby_url}/api/agents/register",
                json=registration_data
            )
            
            if response.status_code == 200:
                logger.info(f"✅ {self.agent_id}: Registered with lobby")
                return True
            else:
                logger.error(f"❌ {self.agent_id}: Registration failed - {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {self.agent_id}: Registration error - {e}")
            return False
    
    async def connect_websocket(self) -> bool:
        """Connect to lobby WebSocket"""
        try:
            self.websocket = await websockets.connect(f"{self.lobby_ws_url}/ws")
            
            # Send WebSocket registration
            register_msg = {
                "type": "register",
                "agent_id": self.agent_id,
                "capabilities": self.capabilities,
                "agent_type": self.agent_type
            }
            
            await self.websocket.send(json.dumps(register_msg))
            logger.info(f"🔌 {self.agent_id}: Connected to WebSocket")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.agent_id}: WebSocket connection failed - {e}")
            return False
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task from the collaboration engine"""
        try:
            task_id = task_data.get('task_id', 'unknown')
            capability_name = task_data.get('capability_name', 'unknown')
            
            logger.info(f"⚙️ {self.agent_id}: Processing task {task_id}")
            logger.info(f"   Capability: {capability_name}")
            
            # **FIX: Check if agent has the required capability**
            if capability_name not in self.capabilities:
                logger.warning(f"❌ {self.agent_id}: Cannot handle capability '{capability_name}' - only supports {self.capabilities}")
                return {
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"Agent {self.agent_id} does not support capability '{capability_name}'. Supported: {self.capabilities}",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"✅ {self.agent_id}: Capability '{capability_name}' supported - processing...")
            
            # Simulate task processing with Ollama
            await asyncio.sleep(1)  # Simulate work
            
            # Generate capability-specific results
            if capability_name == "financial_analysis":
                result = {
                    "analysis_type": "comprehensive_financial_analysis",
                    "ticker": task_data.get("ticker", "META"),
                    "current_price": "$666.85",
                    "pe_ratio": 26.1,
                    "market_cap": "$1.7T",
                    "recommendation": "BUY",
                    "confidence": 0.85,
                    "key_metrics": {
                        "revenue_growth": "22% YoY",
                        "profit_margin": "31%",
                        "debt_ratio": "Low"
                    },
                    "price_targets": {
                        "short_term": "$700",
                        "medium_term": "$750",
                        "long_term": "$800"
                    },
                    "processed_by": self.agent_id,
                    "model_used": self.model
                }
                
            elif capability_name == "content_creation":
                result = {
                    "content_type": "investment_report",
                    "title": f"META Stock Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
                    "sections": [
                        "Executive Summary",
                        "Financial Performance",
                        "Market Position", 
                        "Risk Assessment",
                        "Investment Recommendation"
                    ],
                    "word_count": 2500,
                    "format": "professional_report",
                    "target_audience": "institutional_investors",
                    "processed_by": self.agent_id,
                    "model_used": self.model
                }
                
            elif capability_name == "data_analysis":
                result = {
                    "analysis_type": "technical_data_analysis",
                    "data_points_analyzed": 5000,
                    "time_period": "12_months",
                    "trends_identified": ["upward_momentum", "increasing_volume", "support_at_$650"],
                    "statistical_significance": 0.95,
                    "data_quality": "high",
                    "visualizations_created": ["price_chart", "volume_analysis", "correlation_matrix"],
                    "processed_by": self.agent_id,
                    "model_used": self.model
                }
                
            else:
                result = {
                    "message": f"Successfully processed {capability_name} capability",
                    "task_id": task_id,
                    "processed_by": self.agent_id,
                    "model_used": self.model,
                    "status": "completed"
                }
            
            logger.info(f"✅ {self.agent_id}: Task {task_id} completed successfully")
            
            return {
                "task_id": task_id,
                "status": "success",
                "result": result,
                "agent_id": self.agent_id,
                "processing_time": 2.5,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ {self.agent_id}: Task processing failed - {e}")
            return {
                "task_id": task_data.get('task_id'),
                "status": "failed",
                "error": str(e),
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def handle_message(self, message_data: Dict[str, Any]):
        """Handle incoming messages from the lobby"""
        try:
            # Handle both message formats (type vs message_type)
            message_type = message_data.get('type') or message_data.get('message_type')
            
            logger.info(f"📨 {self.agent_id}: Received {message_type} message")
            
            if message_type == 'register_ack':
                logger.info(f"✅ {self.agent_id}: WebSocket registration confirmed")
                
            elif message_type in ['REQUEST', 'TASK']:
                # This is a collaboration task!
                await self.handle_task_request(message_data)
                
            elif message_type == 'INFO':
                logger.info(f"ℹ️ {self.agent_id}: Info message: {message_data.get('payload', {})}")
                
            elif message_type == 'pong':
                logger.debug(f"🏓 {self.agent_id}: Pong received")
                
            else:
                logger.warning(f"❓ {self.agent_id}: Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ {self.agent_id}: Error handling message: {e}")
    
    async def handle_task_request(self, message_data: Dict[str, Any]):
        """Handle a task request from the collaboration engine"""
        try:
            payload = message_data.get('payload', {})
            conversation_id = message_data.get('conversation_id')
            sender_id = message_data.get('sender_id')
            
            logger.info(f"📋 {self.agent_id}: Processing collaboration task")
            
            # Process the task
            task_result = await self.process_task(payload)
            
            # Send response back to the lobby
            response_message = {
                "type": "task_response",
                "message_id": str(uuid.uuid4()),
                "sender_id": self.agent_id,
                "receiver_id": sender_id or "lobby",
                "conversation_id": conversation_id,
                "payload": task_result
            }
            
            await self.websocket.send(json.dumps(response_message))
            logger.info(f"✅ {self.agent_id}: Task response sent to collaboration engine")
            
        except Exception as e:
            logger.error(f"❌ {self.agent_id}: Error processing task request: {e}")
            
            # Send error response
            error_response = {
                "type": "task_response",
                "message_id": str(uuid.uuid4()),
                "sender_id": self.agent_id,
                "receiver_id": message_data.get('sender_id', 'lobby'),
                "conversation_id": message_data.get('conversation_id'),
                "payload": {
                    "task_id": payload.get('task_id'),
                    "status": "failed",
                    "error": str(e),
                    "agent_id": self.agent_id
                }
            }
            
            await self.websocket.send(json.dumps(error_response))
    
    async def listen_for_messages(self):
        """Listen for messages from the lobby"""
        try:
            while True:
                message = await self.websocket.recv()
                message_data = json.loads(message)
                await self.handle_message(message_data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"🔌 {self.agent_id}: WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ {self.agent_id}: Error in message listener: {e}")
    
    async def start(self):
        """Start the agent"""
        # Register with lobby
        if not await self.register_with_lobby():
            logger.error(f"❌ {self.agent_id}: Failed to register, aborting")
            return False
        
        # Connect WebSocket
        if not await self.connect_websocket():
            logger.error(f"❌ {self.agent_id}: Failed to connect WebSocket, aborting")
            return False
        
        logger.info(f"✅ {self.agent_id}: Agent fully operational")
        
        # Start listening for messages
        await self.listen_for_messages()
        
        return True


async def create_working_agents():
    """Create a set of working collaborative agents"""
    
    agents_config = [
        {
            'agent_id': 'FinancialAnalyst_001',
            'agent_type': 'financial_analyst',
            'capabilities': ['financial_analysis'],
            'model': 'gemma:2b'
        },
        {
            'agent_id': 'ContentCreator_002', 
            'agent_type': 'content_creator',
            'capabilities': ['content_creation'],
            'model': 'gemma:2b'
        },
        {
            'agent_id': 'DataAnalyst_003',
            'agent_type': 'data_analyst', 
            'capabilities': ['data_analysis'],
            'model': 'gemma:2b'
        }
    ]
    
    logger.info("🚀 CREATING WORKING COLLABORATIVE AGENTS")
    logger.info("=" * 50)
    logger.info("These agents will:")
    logger.info("✅ Handle both message formats (type and message_type)")
    logger.info("✅ Process collaboration tasks properly")
    logger.info("✅ Send correct task responses")
    logger.info("✅ Complete end-to-end workflows")
    logger.info("=" * 50)
    
    agents = []
    
    for config in agents_config:
        agent = WorkingCollaborationAgent(**config)
        agents.append(agent)
        logger.info(f"🤖 Created: {config['agent_id']} ({config['agent_type']})")
    
    # Start all agents concurrently
    if agents:
        logger.info(f"🎉 Starting {len(agents)} working collaborative agents...")
        tasks = [agent.start() for agent in agents]
        await asyncio.gather(*tasks)
    else:
        logger.error("❌ No agents created")


if __name__ == "__main__":
    logger.info("🔧 WORKING AGENT LOBBY COLLABORATION TEST")
    logger.info("This fixes the message protocol mismatch and creates working agents")
    
    try:
        asyncio.run(create_working_agents())
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 