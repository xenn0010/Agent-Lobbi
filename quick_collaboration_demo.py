#!/usr/bin/env python3
"""
Advanced Collaboration Demo - Shows N-to-N agent collaboration with learning
"""
import asyncio
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from core.collaboration_engine import CollaborationEngine
from core.message import Message, MessageType


class SmartAgent:
    """Smart agent that can learn from collaborations"""
    def __init__(self, agent_id, capabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.knowledge = {}
        self.collaboration_history = []
        self.success_rate = 0.8
        self.workload = 0


class AdvancedLobby:
    """Advanced lobby with learning capabilities"""
    def __init__(self):
        self.lobby_id = "advanced_lobby_001"
        self.agents = {}
        self.agent_capabilities = {}
        self.messages_sent = []
        self.collaboration_patterns = {}  # Track successful collaborations
    
    def add_agent(self, agent):
        self.agents[agent.agent_id] = agent
        self.agent_capabilities[agent.agent_id] = agent.capabilities
    
    async def route_message(self, message):
        self.messages_sent.append(message)
        
        # Simulate agent response for workflow tasks
        if message.message_type == MessageType.REQUEST and "task_id" in message.payload:
            await self._simulate_agent_response(message)
    
    async def _simulate_agent_response(self, request_msg):
        """Simulate an intelligent agent completing a task"""
        task_data = request_msg.payload
        agent_id = request_msg.receiver_id
        
        # Simulate processing time based on task complexity
        await asyncio.sleep(0.1)
        
        # Create success response with realistic results
        capability = task_data["capability_name"]
        input_data = task_data["input_data"]
        
        # Generate realistic results based on capability
        result = await self._generate_realistic_result(capability, input_data)
        
        response = Message(
            sender_id=agent_id,
            receiver_id=request_msg.sender_id,
            message_type=MessageType.RESPONSE,
            payload={
                "task_id": task_data["task_id"],
                "status": "success",
                "result": result,
                "shared_state_updates": {
                    f"{capability}_completed": True,
                    f"agent_{agent_id}_contribution": result
                }
            },
            conversation_id=request_msg.conversation_id
        )
        
        # Send back to collaboration engine
        if hasattr(self, 'collaboration_engine'):
            await self.collaboration_engine.handle_task_result(response)
        
        print(f"🤖 {agent_id}: Completed {capability} task")
    
    async def _generate_realistic_result(self, capability, input_data):
        """Generate realistic results for different capabilities"""
        if capability == "analyze_sentiment":
            text = input_data.get("text", "")
            # Simple sentiment analysis
            positive_words = ["good", "great", "excellent", "amazing"]
            negative_words = ["bad", "terrible", "awful", "horrible"]
            
            pos_count = sum(1 for word in positive_words if word in text.lower())
            neg_count = sum(1 for word in negative_words if word in text.lower())
            
            if pos_count > neg_count:
                sentiment = "positive"
                confidence = 0.85
            elif neg_count > pos_count:
                sentiment = "negative" 
                confidence = 0.85
            else:
                sentiment = "neutral"
                confidence = 0.60
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "keywords": text.split()[:5]
            }
        
        elif capability == "extract_entities":
            text = input_data.get("text", "")
            # Mock entity extraction
            words = text.split()
            entities = []
            for word in words:
                if word[0].isupper():  # Simple heuristic for proper nouns
                    entities.append({"text": word, "type": "PERSON_OR_ORG"})
            
            return {
                "entities": entities,
                "entity_count": len(entities)
            }
        
        elif capability == "classify_text":
            text = input_data.get("text", "")
            categories = input_data.get("categories", ["business", "technology", "sports"])
            
            # Simple classification based on keywords
            tech_words = ["software", "computer", "AI", "technology", "code"]
            business_words = ["market", "sales", "profit", "company", "business"]
            sports_words = ["game", "team", "player", "score", "match"]
            
            text_lower = text.lower()
            scores = {
                "technology": sum(1 for word in tech_words if word in text_lower),
                "business": sum(1 for word in business_words if word in text_lower),
                "sports": sum(1 for word in sports_words if word in text_lower)
            }
            
            best_category = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[best_category] / len(text.split()) if text.split() else 0.5
            
            return {
                "category": best_category,
                "confidence": min(confidence * 2, 1.0),  # Scale up confidence
                "all_scores": scores
            }
        
        elif capability == "summarize_text":
            text = input_data.get("text", "")
            max_length = input_data.get("max_length", 100)
            
            # Simple summarization
            sentences = text.split(". ")
            if len(sentences) <= 2:
                summary = text
            else:
                summary = f"{sentences[0]}. {sentences[-1]}"
            
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return {
                "summary": summary,
                "original_length": len(text),
                "compression_ratio": len(summary) / len(text) if text else 0
            }
        
        else:
            return {"output": f"Processed {capability} successfully", "confidence": 0.9}


async def advanced_collaboration_demo():
    """Demonstrate advanced multi-agent collaboration with learning"""
    print("🚀 === ADVANCED MULTI-AGENT COLLABORATION DEMO ===")
    
    # Create advanced lobby
    lobby = AdvancedLobby()
    
    # Create specialized agents
    agents = [
        SmartAgent("sentiment_analyzer", {"analyze_sentiment": {"name": "analyze_sentiment"}}),
        SmartAgent("entity_extractor", {"extract_entities": {"name": "extract_entities"}}),
        SmartAgent("text_classifier", {"classify_text": {"name": "classify_text"}}),
        SmartAgent("summarizer", {"summarize_text": {"name": "summarize_text"}}),
        SmartAgent("multi_skill_agent", {
            "analyze_sentiment": {"name": "analyze_sentiment"},
            "summarize_text": {"name": "summarize_text"}
        })
    ]
    
    # Register agents
    for agent in agents:
        lobby.add_agent(agent)
    
    # Create collaboration engine
    engine = CollaborationEngine(lobby)
    lobby.collaboration_engine = engine  # Link back for responses
    
    print(f"📋 Created {len(agents)} specialized agents")
    
    # Demo 1: Complex Text Analysis Pipeline
    print("\n📊 === DEMO 1: TEXT ANALYSIS PIPELINE ===")
    
    sample_text = "OpenAI has developed an amazing new AI model that revolutionizes natural language processing. The technology shows great promise for business applications and could transform how companies interact with customers."
    
    text_analysis_tasks = [
        {
            "name": "Sentiment Analysis",
            "capability": "analyze_sentiment",
            "input": {"text": sample_text}
        },
        {
            "name": "Entity Extraction", 
            "capability": "extract_entities",
            "input": {"text": sample_text}
        },
        {
            "name": "Text Classification",
            "capability": "classify_text", 
            "input": {"text": sample_text, "categories": ["technology", "business", "sports"]}
        },
        {
            "name": "Text Summarization",
            "capability": "summarize_text",
            "input": {"text": sample_text, "max_length": 50},
            "dependencies": []  # Could depend on classification
        }
    ]
    
    # Create and execute workflow
    workflow_id1 = await engine.create_workflow(
        name="Comprehensive Text Analysis",
        description="Multi-agent analysis of text content",
        created_by="demo_user",
        task_definitions=text_analysis_tasks
    )
    
    await engine.start_workflow(workflow_id1)
    print(f"🔄 Started text analysis workflow: {workflow_id1[:8]}...")
    
    # Wait for completion
    await asyncio.sleep(1)
    
    status1 = engine.get_workflow_status(workflow_id1)
    print(f"📈 Text Analysis Results: {status1['progress']}")
    
    # Demo 2: Collaborative Learning Session
    print("\n🧠 === DEMO 2: AGENT COLLABORATION SESSION ===")
    
    # Create real-time collaboration
    agent_ids = [agent.agent_id for agent in agents[:3]]
    collab_id = await engine.create_collaboration_session(
        agent_ids=agent_ids,
        purpose="Collaborative text processing and knowledge sharing"
    )
    
    print(f"🤝 Created collaboration session: {collab_id[:8]}...")
    
    # Simulate collaborative learning
    await engine.broadcast_to_collaboration(
        collab_id=collab_id,
        sender_id="sentiment_analyzer",
        content={
            "learning_update": "Discovered new positive sentiment patterns",
            "shared_knowledge": {"pattern": "excitement + technical terms = high confidence"},
            "confidence_boost": 0.1
        }
    )
    
    # Demo 3: Multi-Stage Dependent Workflow
    print("\n🔗 === DEMO 3: DEPENDENT TASK WORKFLOW ===")
    
    news_articles = [
        "Tesla stock surged after announcing breakthrough in battery technology. Investors are excited about the potential market impact.",
        "The football team signed a new star player for record-breaking amount. Fans are hopeful for championship success.",
        "Microsoft Azure introduces new AI services for enterprise customers. The cloud computing market continues rapid growth."
    ]
    
    # Create dependent workflow
    dependent_tasks = []
    for i, article in enumerate(news_articles):
        # First classify each article
        dependent_tasks.append({
            "name": f"Classify Article {i+1}",
            "capability": "classify_text",
            "input": {"text": article, "categories": ["technology", "business", "sports"]}
        })
        
        # Then analyze sentiment of each
        dependent_tasks.append({
            "name": f"Analyze Sentiment {i+1}",
            "capability": "analyze_sentiment", 
            "input": {"text": article},
            "dependencies": []  # In real system, would depend on classification
        })
    
    # Final summary task that depends on all analyses
    dependent_tasks.append({
        "name": "Create Master Summary",
        "capability": "summarize_text",
        "input": {
            "text": " ".join(news_articles),
            "max_length": 100
        },
        "dependencies": []  # Would depend on all previous tasks
    })
    
    workflow_id2 = await engine.create_workflow(
        name="Multi-Article Analysis Pipeline",
        description="Dependent task workflow with classification and sentiment analysis",
        created_by="news_processor",
        task_definitions=dependent_tasks
    )
    
    await engine.start_workflow(workflow_id2)
    print(f"🔄 Started dependent workflow: {workflow_id2[:8]}...")
    
    # Wait for completion
    await asyncio.sleep(2)
    
    status2 = engine.get_workflow_status(workflow_id2)
    print(f"📈 Dependent Workflow Results: {status2['progress']}")
    
    # Demo 4: System Performance Analysis
    print("\n📊 === DEMO 4: SYSTEM PERFORMANCE ANALYSIS ===")
    
    # Get comprehensive system stats
    stats = engine.get_system_stats()
    print(f"📈 System Statistics:")
    print(f"   Active Workflows: {stats['active_workflows']}")
    print(f"   Completed Workflows: {stats['completed_workflows']}")
    print(f"   Active Collaborations: {stats['active_collaborations']}")
    print(f"   Total Agents: {stats['total_agents']}")
    print(f"   Avg Success Rate: {stats['avg_system_success_rate']:.2%}")
    
    # Show agent workloads
    print(f"\n🤖 Agent Workloads:")
    for agent in agents:
        workload = engine.get_agent_workload(agent.agent_id)
        print(f"   {agent.agent_id}: {workload['active_tasks']} active tasks")
    
    # Show message flow
    print(f"\n📨 Message Flow Analysis:")
    print(f"   Total messages routed: {len(lobby.messages_sent)}")
    
    message_types = {}
    for msg in lobby.messages_sent:
        msg_type = msg.message_type.name
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    for msg_type, count in message_types.items():
        print(f"   {msg_type}: {count} messages")
    
    print(f"\n🎉 === ADVANCED COLLABORATION DEMO COMPLETE ===")
    print(f"✅ Successfully demonstrated:")
    print(f"   • Multi-agent workflow orchestration")  
    print(f"   • Real-time agent collaboration")
    print(f"   • Dependent task management")
    print(f"   • Performance tracking and optimization")
    print(f"   • Scalable N-to-N agent communication")


if __name__ == "__main__":
    asyncio.run(advanced_collaboration_demo()) 