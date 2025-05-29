#!/usr/bin/env python3
"""
Simple test of the collaboration engine functionality
"""
import asyncio
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from core.collaboration_engine import CollaborationEngine, Task, Workflow, WorkflowStatus


class MockLobby:
    """Mock lobby for testing"""
    def __init__(self):
        self.lobby_id = "test_lobby_001"
        self.agents = {"agent1": None, "agent2": None, "agent3": None}
        self.agent_capabilities = {
            "agent1": {"analyze_text": {"name": "analyze_text"}},
            "agent2": {"process_data": {"name": "process_data"}}, 
            "agent3": {"summarize_content": {"name": "summarize_content"}}
        }
        self.messages_sent = []
    
    async def route_message(self, message):
        self.messages_sent.append(message)
        print(f"MOCK LOBBY: Routed message {message.message_type.name} to {message.receiver_id}")
        print(f"  Payload: {message.payload}")


async def test_collaboration_engine():
    """Test the collaboration engine"""
    print("=== Testing Collaboration Engine ===")
    
    # Create mock lobby and engine
    lobby = MockLobby()
    engine = CollaborationEngine(lobby)
    
    # Test 1: Create a workflow
    print("\n1. Creating workflow...")
    task_definitions = [
        {
            "name": "Analyze Text",
            "capability": "analyze_text",
            "input": {"text": "Hello world", "type": "sentiment"}
        },
        {
            "name": "Process Data", 
            "capability": "process_data",
            "input": {"data": [1, 2, 3], "operations": ["sort"]}
        },
        {
            "name": "Create Summary",
            "capability": "summarize_content", 
            "input": {"content": "Analysis complete", "max_length": 50},
            "dependencies": []  # Could depend on previous tasks
        }
    ]
    
    workflow_id = await engine.create_workflow(
        name="Test Workflow",
        description="Testing multi-agent collaboration",
        created_by="test_user",
        task_definitions=task_definitions
    )
    
    print(f"Created workflow: {workflow_id}")
    
    # Test 2: Start workflow
    print("\n2. Starting workflow...")
    success = await engine.start_workflow(workflow_id)
    print(f"Workflow started: {success}")
    
    # Test 3: Check workflow status
    print("\n3. Checking workflow status...")
    status = engine.get_workflow_status(workflow_id)
    print(f"Workflow status: {status}")
    
    # Test 4: Simulate task completion
    print("\n4. Simulating task completions...")
    workflow = engine.workflows[workflow_id]
    
    # Complete all tasks
    for task_id, task in workflow.tasks.items():
        if task.status.value in ["assigned", "in_progress"]:
            # Create mock response message
            from core.message import Message, MessageType
            
            mock_response = Message(
                sender_id=task.assigned_agent,
                receiver_id=lobby.lobby_id,
                message_type=MessageType.RESPONSE,
                payload={
                    "task_id": task_id,
                    "status": "success",
                    "result": {"output": f"Task {task.name} completed successfully"}
                },
                conversation_id=workflow_id
            )
            
            await engine.handle_task_result(mock_response)
            print(f"  Completed task: {task.name}")
    
    # Test 5: Final status check
    print("\n5. Final workflow status...")
    final_status = engine.get_workflow_status(workflow_id)
    print(f"Final status: {final_status}")
    
    # Test 6: System stats
    print("\n6. System statistics...")
    stats = engine.get_system_stats()
    print(f"System stats: {stats}")
    
    print("\n=== Messages sent by lobby ===")
    for i, msg in enumerate(lobby.messages_sent):
        print(f"{i+1}. {msg.message_type.name} -> {msg.receiver_id}")
        print(f"   {msg.payload}")
    
    print("\n=== Collaboration Engine Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_collaboration_engine()) 