#!/usr/bin/env python3
"""
Real-World Coordinator Agent - Orchestrates multi-agent workflows
"""
import asyncio
import sys
import os
import uuid
from typing import Optional, Dict, List, Any
from datetime import datetime

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from sdk.ecosystem_sdk import EcosystemClient, Message, MessageType, AgentCapabilitySDK

class WorkflowCoordinator:
    def __init__(self):
        self.active_workflows = {}
        self.agent_registry = {}
        self.task_results = {}
    
    def create_workflow(self, workflow_id: str, tasks: List[Dict]) -> Dict:
        """Create a new multi-agent workflow"""
        workflow = {
            "id": workflow_id,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "tasks": tasks,
            "completed_tasks": [],
            "pending_tasks": tasks.copy(),
            "failed_tasks": [],
            "results": {}
        }
        
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    def assign_task_to_agent(self, task: Dict, target_agent: str) -> Dict:
        """Assign a specific task to a target agent"""
        task_id = str(uuid.uuid4())
        
        assignment = {
            "task_id": task_id,
            "target_agent": target_agent,
            "capability": task["capability"],
            "input_data": task["input_data"],
            "assigned_at": datetime.now().isoformat(),
            "status": "assigned"
        }
        
        return assignment
    
    def process_task_result(self, workflow_id: str, task_id: str, result: Dict) -> Dict:
        """Process the result of a completed task"""
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.active_workflows[workflow_id]
        workflow["results"][task_id] = result
        
        # Update workflow status
        completed_count = len(workflow["completed_tasks"])
        total_count = len(workflow["tasks"])
        
        if completed_count == total_count:
            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now().isoformat()
        
        return {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "workflow_status": workflow["status"],
            "progress": f"{completed_count}/{total_count}"
        }

class CoordinatorAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.coordinator = WorkflowCoordinator()
        self.orchestrated_workflows = 0
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming messages from the lobby"""
        print(f"🎯 {self.agent_id}: Received message type {message.message_type.name}")
        
        if message.message_type == MessageType.REQUEST:
            capability = message.payload.get("capability_name")
            input_data = message.payload.get("input_data", {})
            
            try:
                if capability == "create_workflow":
                    workflow_id = input_data.get("workflow_id", str(uuid.uuid4()))
                    tasks = input_data.get("tasks", [])
                    result = self.coordinator.create_workflow(workflow_id, tasks)
                    self.orchestrated_workflows += 1
                    
                    print(f"🚀 {self.agent_id}: Created workflow '{workflow_id}' with {len(tasks)} tasks")
                    
                elif capability == "assign_task":
                    task = input_data.get("task", {})
                    target_agent = input_data.get("target_agent", "")
                    result = self.coordinator.assign_task_to_agent(task, target_agent)
                    
                    print(f"📋 {self.agent_id}: Assigned task to '{target_agent}'")
                    
                elif capability == "orchestrate_analysis":
                    # Example: Orchestrate a text analysis workflow
                    text_data = input_data.get("text", "")
                    additional_data = input_data.get("data", {})
                    
                    # Create a multi-step workflow
                    workflow_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    workflow_tasks = [
                        {
                            "step": 1,
                            "capability": "analyze_text",
                            "target_agent": "gpt_analyst_001",
                            "input_data": {"text": text_data},
                            "description": "Analyze text content"
                        },
                        {
                            "step": 2,
                            "capability": "process_data",
                            "target_agent": "data_processor_001",
                            "input_data": {"data": additional_data},
                            "description": "Process associated data"
                        }
                    ]
                    
                    result = {
                        "workflow_created": self.coordinator.create_workflow(workflow_id, workflow_tasks),
                        "orchestration_plan": workflow_tasks,
                        "estimated_completion": "2-5 minutes",
                        "coordinator": self.agent_id
                    }
                    
                    print(f"🎭 {self.agent_id}: Orchestrated analysis workflow '{workflow_id}'")
                    
                elif capability == "get_workflow_status":
                    workflow_id = input_data.get("workflow_id", "")
                    workflow = self.coordinator.active_workflows.get(workflow_id)
                    
                    if workflow:
                        result = {
                            "workflow_id": workflow_id,
                            "status": workflow["status"],
                            "progress": f"{len(workflow['completed_tasks'])}/{len(workflow['tasks'])}",
                            "results_available": len(workflow["results"]),
                            "created_at": workflow["created_at"]
                        }
                        print(f"📊 {self.agent_id}: Retrieved status for workflow '{workflow_id}'")
                    else:
                        result = {"error": f"Workflow '{workflow_id}' not found"}
                        print(f"❌ {self.agent_id}: Workflow '{workflow_id}' not found")
                
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

async def run_coordinator_agent():
    """Run the coordinator agent"""
    agent_id = "workflow_coordinator_001"
    agent = CoordinatorAgent(agent_id)
    
    print(f"🚀 Starting {agent_id}...")
    
    # Define capabilities
    capabilities = [
        AgentCapabilitySDK(
            name="create_workflow",
            description="Create and manage multi-agent workflows",
            input_schema={"workflow_id": "string", "tasks": "array"},
            output_schema={"id": "string", "status": "string", "tasks": "array"}
        ),
        AgentCapabilitySDK(
            name="assign_task",
            description="Assign specific tasks to target agents",
            input_schema={"task": "object", "target_agent": "string"},
            output_schema={"task_id": "string", "target_agent": "string", "status": "string"}
        ),
        AgentCapabilitySDK(
            name="orchestrate_analysis",
            description="Orchestrate complex analysis workflows across multiple agents",
            input_schema={"text": "string", "data": "object"},
            output_schema={"workflow_created": "object", "orchestration_plan": "array"}
        ),
        AgentCapabilitySDK(
            name="get_workflow_status",
            description="Get the current status of a workflow",
            input_schema={"workflow_id": "string"},
            output_schema={"workflow_id": "string", "status": "string", "progress": "string"}
        )
    ]
    
    # Create SDK client
    sdk_client = EcosystemClient(
        agent_id=agent_id,
        agent_type="WorkflowCoordinator",
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
            print(f"🎯 Ready to orchestrate multi-agent workflows")
            print(f"🔄 Available capabilities: create_workflow, assign_task, orchestrate_analysis")
            
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
    print("🎯 REAL WORLD WORKFLOW COORDINATOR AGENT")
    print("=" * 60)
    print("This agent connects to your running mock lobby and provides:")
    print("• Multi-agent workflow creation and management")
    print("• Task assignment and coordination")
    print("• Analysis orchestration across multiple agents")
    print("• Workflow status tracking and monitoring")
    print("=" * 60)
    
    asyncio.run(run_coordinator_agent()) 