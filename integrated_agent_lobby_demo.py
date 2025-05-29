#!/usr/bin/env python3
"""
Integrated Agent Lobby Demo - Complete System Demonstration
Shows the full power of the Agent Lobby with multi-agent collaboration
"""
import asyncio
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from core.lobby import Lobby
from core.message import Message, MessageType, MessagePriority
from examples.workflow_demo_agent import (
    TextAnalyzerAgent, DataProcessorAgent, CodeGeneratorAgent, 
    TranslatorAgent, MultiSkillAgent
)


async def integrated_demo():
    """Complete demonstration of the Agent Lobby ecosystem"""
    print("🚀 === INTEGRATED AGENT LOBBY DEMONSTRATION ===")
    print("Showcasing the most advanced multi-agent collaboration system")
    
    # Create the lobby with full capabilities
    lobby = Lobby()
    await lobby.start()
    
    print("✅ Agent Lobby started with:")
    print("   • Advanced collaboration engine")
    print("   • Learning capabilities")
    print("   • Reputation system")
    print("   • Priority message queues")
    print("   • World state management")
    print("   • Multi-agent workflows")
    
    # Create a diverse set of agents
    agents = [
        TextAnalyzerAgent("sentiment_specialist"),
        TextAnalyzerAgent("nlp_expert"), 
        DataProcessorAgent("data_scientist"),
        CodeGeneratorAgent("software_engineer"),
        TranslatorAgent("linguist"),
        MultiSkillAgent("generalist_ai")
    ]
    
    # Register all agents with the lobby
    for agent in agents:
        await lobby.register_agent(agent)
    
    # Start all agents
    agent_tasks = []
    for agent in agents:
        task = asyncio.create_task(agent.run())
        agent_tasks.append(task)
    
    # Wait for agents to be ready
    await asyncio.sleep(2)
    
    print(f"📋 Registered {len(agents)} intelligent agents")
    print("   Each agent has specialized capabilities and can collaborate")
    
    # Demo 1: Create and execute a complex workflow via lobby
    print("\n🔄 === DEMO 1: COMPLEX WORKFLOW ORCHESTRATION ===")
    
    # Create workflow request message
    workflow_request = Message(
        sender_id="demo_orchestrator",
        receiver_id=lobby.lobby_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "create_workflow",
            "workflow_name": "AI News Analysis Pipeline",
            "workflow_description": "Comprehensive analysis of AI news articles",
            "tasks": [
                {
                    "name": "Sentiment Analysis",
                    "capability": "analyze_text",
                    "input": {
                        "text": "OpenAI releases GPT-5 with groundbreaking capabilities that exceed all expectations. Industry experts are amazed by the unprecedented performance improvements.",
                        "analysis_type": "sentiment"
                    }
                },
                {
                    "name": "Content Summarization", 
                    "capability": "summarize_content",
                    "input": {
                        "content": "The latest AI breakthrough represents a significant milestone in artificial intelligence development. Researchers have achieved remarkable results that could revolutionize multiple industries.",
                        "max_length": 60
                    }
                },
                {
                    "name": "Data Processing",
                    "capability": "process_data", 
                    "input": {
                        "data": [95, 87, 92, 88, 94, 90, 96],
                        "operations": ["sort", "deduplicate"]
                    }
                },
                {
                    "name": "Code Generation",
                    "capability": "generate_code",
                    "input": {
                        "requirements": "Create a Python function to analyze sentiment scores",
                        "language": "python"
                    }
                }
            ]
        },
        priority=MessagePriority.HIGH
    )
    
    # Send workflow creation request
    await lobby.route_message(workflow_request)
    await asyncio.sleep(0.5)
    
    # Demo 2: Real-time collaboration session
    print("\n🤝 === DEMO 2: REAL-TIME AGENT COLLABORATION ===")
    
    collab_request = Message(
        sender_id="demo_orchestrator",
        receiver_id=lobby.lobby_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "create_collaboration",
            "participant_ids": ["sentiment_specialist", "nlp_expert", "generalist_ai"],
            "purpose": "Real-time sentiment analysis collaboration and knowledge sharing"
        },
        priority=MessagePriority.HIGH
    )
    
    await lobby.route_message(collab_request)
    await asyncio.sleep(0.5)
    
    # Demo 3: System performance and analytics
    print("\n📊 === DEMO 3: SYSTEM ANALYTICS ===")
    
    # Get collaboration engine statistics
    stats = lobby.collaboration_engine.get_system_stats()
    print(f"🎯 System Performance:")
    print(f"   • Active Workflows: {stats['active_workflows']}")
    print(f"   • Completed Workflows: {stats['completed_workflows']}")
    print(f"   • Active Collaborations: {stats['active_collaborations']}")
    print(f"   • Total Agents: {stats['total_agents']}")
    print(f"   • System Success Rate: {stats['avg_system_success_rate']:.1%}")
    
    # Show agent workloads
    print(f"\n🤖 Agent Workload Distribution:")
    for agent in agents:
        workload = lobby.collaboration_engine.get_agent_workload(agent.agent_id)
        reputation = lobby.agent_reputation.get(agent.agent_id, lobby.default_reputation)
        print(f"   • {agent.agent_id}: {workload['active_tasks']} tasks, Rep: {reputation:.1f}")
    
    # Demo 4: Advanced features showcase
    print("\n🌟 === DEMO 4: ADVANCED FEATURES ===")
    
    print("✨ Agent Lobby Advanced Features:")
    print("   🔹 Dynamic workflow orchestration with dependency management")
    print("   🔹 Intelligent agent selection based on workload and performance")
    print("   🔹 Real-time collaboration sessions with message broadcasting")
    print("   🔹 Reputation-based service discovery and ranking")
    print("   🔹 Priority-based message queues for optimal routing")
    print("   🔹 Learning sessions for collaborative AI development")
    print("   🔹 Test environments for model validation")
    print("   🔹 Comprehensive logging and analytics")
    
    # Demo 5: Show message flow and system health
    print("\n📈 === DEMO 5: MESSAGE FLOW ANALYSIS ===")
    
    # Let the system process for a bit longer
    await asyncio.sleep(3)
    
    # Get final statistics
    final_stats = lobby.collaboration_engine.get_system_stats()
    print(f"📋 Final System State:")
    print(f"   • Total Workflows Completed: {final_stats['completed_workflows']}")
    print(f"   • Active Collaborations: {final_stats['active_collaborations']}")
    print(f"   • Average Success Rate: {final_stats['avg_system_success_rate']:.1%}")
    
    # Show agent performance evolution
    print(f"\n🏆 Agent Performance Rankings:")
    agent_performance = []
    for agent_id, perf in lobby.collaboration_engine.agent_performance.items():
        agent_performance.append({
            "agent": agent_id,
            "tasks": perf.get("total_tasks", 0),
            "success_rate": perf.get("avg_success_rate", 0),
            "reputation": lobby.agent_reputation.get(agent_id, lobby.default_reputation)
        })
    
    # Sort by success rate
    agent_performance.sort(key=lambda x: x["success_rate"], reverse=True)
    
    for i, perf in enumerate(agent_performance, 1):
        print(f"   {i}. {perf['agent']}: {perf['tasks']} tasks, {perf['success_rate']:.1%} success, Rep: {perf['reputation']:.1f}")
    
    print("\n🎉 === DEMONSTRATION COMPLETE ===")
    print("✅ Successfully demonstrated:")
    print("   • Multi-agent workflow orchestration")
    print("   • Dynamic collaboration sessions")
    print("   • Performance-based agent selection")
    print("   • Real-time system analytics")
    print("   • Reputation and learning systems")
    print("   • Scalable message routing")
    
    print("\n🌟 The Agent Lobby is now the most advanced multi-agent")
    print("    collaboration platform with sophisticated orchestration,")
    print("    learning capabilities, and performance optimization!")
    
    # Cleanup
    await lobby.stop()
    lobby.close_log_file()
    
    for task in agent_tasks:
        task.cancel()
    
    print("\n🏁 Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(integrated_demo()) 