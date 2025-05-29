#!/usr/bin/env python3
"""
Workflow Demo Agent - Shows multi-agent collaboration capabilities
"""
import asyncio
import sys
import os
import json
from typing import Dict, Any, List, Optional, Set

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

from core.agent import Agent
from core.message import Message, MessageType, MessagePriority, Capability
from core.lobby import Lobby


class WorkflowCapableAgent(Agent):
    """Agent that can participate in multi-agent workflows"""
    
    def __init__(self, agent_id: str, specialized_skills: List[str]):
        super().__init__(agent_id)
        self.specialized_skills = specialized_skills
        self.active_tasks: Dict[str, Dict] = {}  # task_id -> task_info
        self.knowledge_base: Dict[str, Any] = {}
        self.collaboration_sessions: Set[str] = set()
        
    def get_capabilities(self) -> List[Capability]:
        """Define what this agent can do"""
        capabilities = []
        
        for skill in self.specialized_skills:
            if skill == "text_analysis":
                capabilities.append({
                    "name": "analyze_text",
                    "description": "Analyze text for sentiment, keywords, and structure",
                    "input_schema": {"text": "string", "analysis_type": "string"},
                    "output_schema": {"analysis_result": "object"},
                    "authorized_requester_ids": None  # Public capability
                })
            elif skill == "data_processing":
                capabilities.append({
                    "name": "process_data",
                    "description": "Clean and transform data sets",
                    "input_schema": {"data": "array", "operations": "array"},
                    "output_schema": {"processed_data": "array", "metadata": "object"},
                    "authorized_requester_ids": None
                })
            elif skill == "code_generation":
                capabilities.append({
                    "name": "generate_code",
                    "description": "Generate code based on specifications",
                    "input_schema": {"requirements": "string", "language": "string"},
                    "output_schema": {"code": "string", "tests": "array"},
                    "authorized_requester_ids": None
                })
            elif skill == "summarization":
                capabilities.append({
                    "name": "summarize_content",
                    "description": "Create summaries of long content",
                    "input_schema": {"content": "string", "max_length": "integer"},
                    "output_schema": {"summary": "string", "key_points": "array"},
                    "authorized_requester_ids": None
                })
            elif skill == "translation":
                capabilities.append({
                    "name": "translate_text",
                    "description": "Translate text between languages",
                    "input_schema": {"text": "string", "source_lang": "string", "target_lang": "string"},
                    "output_schema": {"translated_text": "string", "confidence": "number"},
                    "authorized_requester_ids": None
                })
        
        return capabilities
    
    async def process_incoming_message(self, msg: Message):
        """Process incoming messages"""
        print(f"{self.agent_id}: Received {msg.message_type.name} from {msg.sender_id}")
        
        if msg.message_type == MessageType.REQUEST:
            await self._handle_request(msg)
        elif msg.message_type == MessageType.INFO:
            await self._handle_info_message(msg)
        elif msg.message_type == MessageType.RESPONSE:
            await self._handle_workflow_response(msg)
        elif msg.message_type == MessageType.ERROR:
            print(f"{self.agent_id}: Error received: {msg.payload.get('error')}")
    
    async def _handle_request(self, msg: Message):
        """Handle task requests from workflow engine or other agents"""
        payload = msg.payload
        
        # Check if this is a workflow task
        if "task_id" in payload and "workflow_id" in payload:
            await self._handle_workflow_task(msg)
        else:
            # Handle direct capability invocation
            capability_name = payload.get("capability_name")
            if capability_name:
                await self._execute_capability(msg, capability_name, payload.get("input_data", {}))
    
    async def _handle_workflow_task(self, msg: Message):
        """Handle a task assigned by the workflow engine"""
        payload = msg.payload
        task_id = payload["task_id"]
        workflow_id = payload["workflow_id"]
        capability_name = payload["capability_name"]
        input_data = payload["input_data"]
        shared_state = payload.get("shared_state", {})
        
        print(f"{self.agent_id}: Executing workflow task '{payload.get('task_name')}' (ID: {task_id})")
        
        # Store task info
        self.active_tasks[task_id] = {
            "workflow_id": workflow_id,
            "capability_name": capability_name,
            "input_data": input_data,
            "shared_state": shared_state,
            "started_at": asyncio.get_event_loop().time()
        }
        
        try:
            # Execute the capability
            result = await self._execute_capability_internal(capability_name, input_data, shared_state)
            
            # Send success response
            response_payload = {
                "task_id": task_id,
                "status": "success",
                "result": result,
                "shared_state_updates": result.get("shared_state_updates", {})
            }
            
            await self.send_message(
                receiver_id=msg.sender_id,
                msg_type=MessageType.RESPONSE,
                payload=response_payload,
                conversation_id=workflow_id
            )
            
            print(f"{self.agent_id}: Completed task {task_id} successfully")
            
        except Exception as e:
            # Send error response
            response_payload = {
                "task_id": task_id,
                "status": "error",
                "error": str(e)
            }
            
            await self.send_message(
                receiver_id=msg.sender_id,
                msg_type=MessageType.RESPONSE,
                payload=response_payload,
                conversation_id=workflow_id
            )
            
            print(f"{self.agent_id}: Task {task_id} failed: {e}")
        
        finally:
            # Clean up task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_capability_internal(self, capability_name: str, input_data: Dict, shared_state: Dict) -> Dict[str, Any]:
        """Execute a capability and return results"""
        
        if capability_name == "analyze_text":
            text = input_data.get("text", "")
            analysis_type = input_data.get("analysis_type", "sentiment")
            
            # Simulate text analysis
            if analysis_type == "sentiment":
                # Simple sentiment analysis simulation
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "worse", "worst"]
                
                text_lower = text.lower()
                positive_count = sum(word in text_lower for word in positive_words)
                negative_count = sum(word in text_lower for word in negative_words)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                    confidence = 0.8
                elif negative_count > positive_count:
                    sentiment = "negative"
                    confidence = 0.8
                else:
                    sentiment = "neutral"
                    confidence = 0.6
                
                return {
                    "analysis_result": {
                        "sentiment": sentiment,
                        "confidence": confidence,
                        "word_count": len(text.split()),
                        "positive_indicators": positive_count,
                        "negative_indicators": negative_count
                    }
                }
            
            elif analysis_type == "keywords":
                # Simple keyword extraction
                words = text.lower().split()
                word_freq = {}
                for word in words:
                    word = word.strip(".,!?\"'")
                    if len(word) > 3:  # Only consider words longer than 3 chars
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top 5 keywords
                keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                
                return {
                    "analysis_result": {
                        "keywords": [{"word": word, "frequency": freq} for word, freq in keywords],
                        "total_words": len(words),
                        "unique_words": len(word_freq)
                    }
                }
        
        elif capability_name == "process_data":
            data = input_data.get("data", [])
            operations = input_data.get("operations", [])
            
            processed_data = data.copy()
            metadata = {"operations_applied": []}
            
            for operation in operations:
                if operation == "remove_nulls":
                    processed_data = [item for item in processed_data if item is not None]
                    metadata["operations_applied"].append("remove_nulls")
                elif operation == "sort":
                    try:
                        processed_data = sorted(processed_data)
                        metadata["operations_applied"].append("sort")
                    except:
                        pass  # Skip if not sortable
                elif operation == "deduplicate":
                    processed_data = list(set(processed_data))
                    metadata["operations_applied"].append("deduplicate")
            
            return {
                "processed_data": processed_data,
                "metadata": {
                    **metadata,
                    "original_length": len(data),
                    "processed_length": len(processed_data)
                }
            }
        
        elif capability_name == "generate_code":
            requirements = input_data.get("requirements", "")
            language = input_data.get("language", "python")
            
            # Simple code generation simulation
            if "hello world" in requirements.lower():
                if language.lower() == "python":
                    code = 'print("Hello, World!")'
                    tests = ['assert "Hello, World!" in captured_output']
                elif language.lower() == "javascript":
                    code = 'console.log("Hello, World!");'
                    tests = ['expect(output).toContain("Hello, World!")']
                else:
                    code = f'// Hello World in {language}\n// Implementation needed'
                    tests = ['// Test cases needed']
            else:
                code = f'// Generated {language} code for: {requirements}\n// Implementation based on requirements'
                tests = [f'// Test case for {requirements}']
            
            return {
                "code": code,
                "tests": tests,
                "language": language,
                "confidence": 0.7
            }
        
        elif capability_name == "summarize_content":
            content = input_data.get("content", "")
            max_length = input_data.get("max_length", 100)
            
            # Simple summarization
            sentences = content.split(". ")
            if len(sentences) <= 2:
                summary = content
                key_points = sentences
            else:
                # Take first and last sentences
                summary = f"{sentences[0]}. {sentences[-1]}"
                key_points = sentences[:3]  # First 3 sentences as key points
            
            # Truncate if too long
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return {
                "summary": summary,
                "key_points": key_points,
                "original_length": len(content),
                "compression_ratio": len(summary) / len(content) if content else 0
            }
        
        elif capability_name == "translate_text":
            text = input_data.get("text", "")
            source_lang = input_data.get("source_lang", "en")
            target_lang = input_data.get("target_lang", "es")
            
            # Simulated translation (very basic)
            translations = {
                ("en", "es"): {"hello": "hola", "world": "mundo", "good": "bueno"},
                ("es", "en"): {"hola": "hello", "mundo": "world", "bueno": "good"},
                ("en", "fr"): {"hello": "bonjour", "world": "monde", "good": "bon"},
            }
            
            trans_dict = translations.get((source_lang, target_lang), {})
            translated_text = text.lower()
            
            for source_word, target_word in trans_dict.items():
                translated_text = translated_text.replace(source_word, target_word)
            
            confidence = 0.6  # Simulated confidence
            
            return {
                "translated_text": translated_text,
                "confidence": confidence,
                "source_language": source_lang,
                "target_language": target_lang
            }
        
        else:
            raise ValueError(f"Unknown capability: {capability_name}")
    
    async def _execute_capability(self, msg: Message, capability_name: str, input_data: Dict):
        """Execute capability and send response for direct invocation"""
        try:
            result = await self._execute_capability_internal(capability_name, input_data, {})
            
            response_payload = {
                "status": "success",
                "capability_name": capability_name,
                "result": result
            }
            
            await self.send_message(
                receiver_id=msg.sender_id,
                msg_type=MessageType.RESPONSE,
                payload=response_payload,
                conversation_id=msg.conversation_id
            )
            
        except Exception as e:
            error_payload = {
                "status": "error",
                "capability_name": capability_name,
                "error": str(e)
            }
            
            await self.send_message(
                receiver_id=msg.sender_id,
                msg_type=MessageType.ERROR,
                payload=error_payload,
                conversation_id=msg.conversation_id
            )
    
    async def _handle_info_message(self, msg: Message):
        """Handle informational messages"""
        payload = msg.payload
        
        if payload.get("event_type") == "collaboration_started":
            collab_data = payload.get("data", {})
            collab_id = collab_data.get("collaboration_id")
            if collab_id:
                self.collaboration_sessions.add(collab_id)
                print(f"{self.agent_id}: Joined collaboration session {collab_id}")
        
        elif payload.get("collaboration_broadcast"):
            collab_id = payload.get("collaboration_id")
            content = payload.get("content", {})
            print(f"{self.agent_id}: Collaboration broadcast in {collab_id}: {content}")
    
    async def _handle_workflow_response(self, msg: Message):
        """Handle responses from workflow-related requests"""
        payload = msg.payload
        print(f"{self.agent_id}: Workflow response: {payload}")
    
    async def run(self):
        """Main agent loop"""
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} with skills {self.specialized_skills} is running and ready for workflows!")
        
        # Wait for auth token
        while not self.auth_token:
            await asyncio.sleep(0.1)
        
        print(f"{self.agent_id}: Authenticated and ready for collaboration!")
        
        # Main message processing loop
        while True:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.1)


# Specialized agent types
class TextAnalyzerAgent(WorkflowCapableAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["text_analysis", "summarization"])


class DataProcessorAgent(WorkflowCapableAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["data_processing"])


class CodeGeneratorAgent(WorkflowCapableAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["code_generation"])


class TranslatorAgent(WorkflowCapableAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["translation"])


class MultiSkillAgent(WorkflowCapableAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id, ["text_analysis", "data_processing", "summarization", "translation"])


async def main():
    """Demo of multi-agent workflow system"""
    print("=== Agent Lobby Workflow Demo ===")
    
    # Create lobby
    lobby = Lobby()
    await lobby.start()
    
    # Create specialized agents
    text_agent = TextAnalyzerAgent("text_analyzer_001")
    data_agent = DataProcessorAgent("data_processor_001")
    code_agent = CodeGeneratorAgent("code_generator_001")
    translator_agent = TranslatorAgent("translator_001")
    
    # Register agents with lobby
    await lobby.register_agent(text_agent)
    await lobby.register_agent(data_agent)
    await lobby.register_agent(code_agent)
    await lobby.register_agent(translator_agent)
    
    # Start agents
    agent_tasks = [
        asyncio.create_task(text_agent.run()),
        asyncio.create_task(data_agent.run()),
        asyncio.create_task(code_agent.run()),
        asyncio.create_task(translator_agent.run())
    ]
    
    # Wait for agents to be ready
    await asyncio.sleep(2)
    
    print("\n=== Creating Multi-Agent Workflow ===")
    
    # Create a workflow that uses multiple agents
    task_definitions = [
        {
            "name": "Analyze Customer Feedback",
            "capability": "analyze_text",
            "input": {
                "text": "The product is amazing and works great! However, the documentation could be better.",
                "analysis_type": "sentiment"
            }
        },
        {
            "name": "Process Customer Data",
            "capability": "process_data",
            "input": {
                "data": [1, 2, None, 3, 2, 4, None, 5],
                "operations": ["remove_nulls", "deduplicate", "sort"]
            },
            "dependencies": []  # Can run in parallel with sentiment analysis
        },
        {
            "name": "Generate Summary Report",
            "capability": "summarize_content",
            "input": {
                "content": "Customer feedback analysis shows positive sentiment. Data processing removed nulls and duplicates from dataset. Overall customer satisfaction is high with room for documentation improvement.",
                "max_length": 50
            },
            "dependencies": []  # This would normally depend on previous tasks, but simplified for demo
        }
    ]
    
    # Create workflow via lobby message
    workflow_message = Message(
        sender_id="demo_controller",
        receiver_id=lobby.lobby_id,
        message_type=MessageType.REQUEST,
        payload={
            "action": "create_workflow",
            "workflow_name": "Customer Analysis Pipeline",
            "workflow_description": "Analyze customer feedback and process related data",
            "tasks": task_definitions
        }
    )
    
    # Send workflow creation request (simulating external controller)
    # In a real system, this would come from a client or admin interface
    await lobby.route_message(workflow_message)
    
    # Let the demo run for a while
    print("Workflow demo running... (will run for 30 seconds)")
    await asyncio.sleep(30)
    
    # Cleanup
    await lobby.stop()
    for task in agent_tasks:
        task.cancel()
    
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main()) 