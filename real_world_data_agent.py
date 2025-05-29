#!/usr/bin/env python3
"""
Real-World Data Processing Agent - Connects to Agent Lobby for data operations
"""
import asyncio
import sys
import os
import json
import csv
from typing import Optional, Dict, List, Any
from datetime import datetime

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from sdk.ecosystem_sdk import EcosystemClient, Message, MessageType, AgentCapabilitySDK

class DataProcessor:
    def __init__(self):
        self.processed_datasets = 0
        
    def process_json_data(self, data: Dict) -> Dict:
        """Process JSON data and extract insights"""
        result = {
            "processed_at": datetime.now().isoformat(),
            "data_type": "json",
            "record_count": 1 if isinstance(data, dict) else len(data) if isinstance(data, list) else 0,
            "fields": list(data.keys()) if isinstance(data, dict) else [],
            "summary": "JSON data processed successfully"
        }
        
        # Extract numeric fields for basic statistics
        if isinstance(data, dict):
            numeric_fields = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            if numeric_fields:
                result["statistics"] = {
                    "numeric_fields": len(numeric_fields),
                    "total_sum": sum(numeric_fields.values()),
                    "average": sum(numeric_fields.values()) / len(numeric_fields)
                }
        
        return result
    
    def aggregate_data(self, data_list: List[Dict]) -> Dict:
        """Aggregate multiple data records"""
        if not data_list:
            return {"error": "No data provided"}
        
        total_records = len(data_list)
        all_fields = set()
        numeric_totals = {}
        
        for record in data_list:
            if isinstance(record, dict):
                all_fields.update(record.keys())
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        numeric_totals[key] = numeric_totals.get(key, 0) + value
        
        return {
            "aggregation_summary": {
                "total_records": total_records,
                "unique_fields": len(all_fields),
                "field_names": list(all_fields),
                "numeric_aggregations": numeric_totals,
                "processed_at": datetime.now().isoformat()
            }
        }
    
    def generate_report(self, processed_data: Dict) -> Dict:
        """Generate a comprehensive data report"""
        return {
            "report": {
                "title": "Data Processing Report",
                "generated_at": datetime.now().isoformat(),
                "data_summary": processed_data,
                "recommendations": [
                    "Data quality appears good",
                    "Consider adding validation for numeric fields",
                    "Monitor data volume trends over time"
                ],
                "next_steps": [
                    "Schedule regular data processing",
                    "Set up automated quality checks",
                    "Create data visualization dashboards"
                ]
            }
        }

class DataAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.processor = DataProcessor()
        self.task_count = 0
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages from the lobby"""
        print(f"📊 {self.agent_id}: Received message type {message.message_type.name}")
        
        if message.message_type == MessageType.REQUEST:
            capability = message.payload.get("capability_name")
            input_data = message.payload.get("input_data", {})
            
            try:
                if capability == "process_data":
                    data = input_data.get("data", {})
                    result = self.processor.process_json_data(data)
                    self.task_count += 1
                    
                    print(f"✅ {self.agent_id}: Processed data (Task #{self.task_count})")
                    
                elif capability == "aggregate_data":
                    data_list = input_data.get("data_list", [])
                    result = self.processor.aggregate_data(data_list)
                    self.task_count += 1
                    
                    print(f"📈 {self.agent_id}: Aggregated {len(data_list)} records (Task #{self.task_count})")
                    
                elif capability == "generate_report":
                    processed_data = input_data.get("processed_data", {})
                    result = self.processor.generate_report(processed_data)
                    self.task_count += 1
                    
                    print(f"📋 {self.agent_id}: Generated report (Task #{self.task_count})")
                    
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
                        "task_id": message.payload.get("task_id"),
                        "capability_used": capability
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
            print(f"ℹ️ {self.agent_id}: Info received - {message.payload}")
        
        return None

async def run_data_agent():
    """Run the data processing agent"""
    agent_id = "data_processor_001"
    agent = DataAgent(agent_id)
    
    print(f"🚀 Starting {agent_id}...")
    
    # Define capabilities
    capabilities = [
        AgentCapabilitySDK(
            name="process_data",
            description="Process JSON data and extract insights and statistics",
            input_schema={"data": "object"},
            output_schema={"processed_at": "string", "record_count": "number", "summary": "string"}
        ),
        AgentCapabilitySDK(
            name="aggregate_data",
            description="Aggregate multiple data records into summary statistics",
            input_schema={"data_list": "array"},
            output_schema={"aggregation_summary": "object"}
        ),
        AgentCapabilitySDK(
            name="generate_report",
            description="Generate comprehensive reports from processed data",
            input_schema={"processed_data": "object"},
            output_schema={"report": "object"}
        )
    ]
    
    # Create SDK client
    sdk_client = EcosystemClient(
        agent_id=agent_id,
        agent_type="DataProcessor",
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
            print(f"📊 Ready to process data, aggregate records, and generate reports")
            
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
    print("📊 REAL WORLD DATA PROCESSING AGENT")
    print("=" * 60)
    print("This agent connects to your running mock lobby and provides:")
    print("• JSON data processing and analysis")
    print("• Data aggregation and statistics")
    print("• Report generation")
    print("• Real-time collaboration with other agents")
    print("=" * 60)
    
    asyncio.run(run_data_agent()) 