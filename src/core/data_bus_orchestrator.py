#!/usr/bin/env python3
"""
DATA BUS + TRAFFIC LIGHT ORCHESTRATION SYSTEM
============================================
A masterpiece of multi-agent collaboration engineering.

This system solves the Agent Lobby's core problems:
1. Standardized JSON communication (Data Bus)
2. Stage-based orchestration (Traffic Lights) 
3. Proper multi-agent distribution
4. Dynamic goal achievement
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import copy

logger = logging.getLogger(__name__)

# ============================================
# STANDARDIZED JSON MESSAGE PROTOCOL
# ============================================

@dataclass
class StandardizedMessage:
    """Standardized JSON message for the data bus"""
    message_id: str
    workflow_id: str
    current_stage: str
    message_type: str  # "stage_request", "stage_response", "workflow_init", "workflow_complete"
    timestamp: str
    data_payload: Dict[str, Any]
    next_stages: List[str] = field(default_factory=list)
    completed_stages: List[str] = field(default_factory=list)
    destination_goal: str = ""
    route_history: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.__dict__, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StandardizedMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(**data)
    
    def clone_for_next_stage(self, next_stage: str) -> 'StandardizedMessage':
        """Create a new message for the next stage"""
        new_msg = copy.deepcopy(self)
        new_msg.message_id = str(uuid.uuid4())
        new_msg.route_history.append(self.current_stage)
        new_msg.completed_stages.append(self.current_stage)
        new_msg.current_stage = next_stage
        new_msg.timestamp = datetime.now(timezone.utc).isoformat()
        return new_msg

# ============================================
# TRAFFIC LIGHT SYSTEM
# ============================================

class TrafficLightState(Enum):
    RED = "red"       # Stage blocked/busy
    YELLOW = "yellow" # Stage preparing
    GREEN = "green"   # Stage ready to process

@dataclass
class StageDefinition:
    """Definition of an orchestration stage"""
    stage_name: str
    required_capability: str
    agent_pool: List[str]  # Available agents for this stage
    max_concurrent: int = 1
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)  # Previous stages required
    outputs_to: List[str] = field(default_factory=list)   # Next possible stages

class TrafficLight:
    """Traffic light controller for an orchestration stage"""
    
    def __init__(self, stage_definition: StageDefinition, orchestrator_ref):
        self.stage_def = stage_definition
        self.orchestrator = orchestrator_ref
        self.state = TrafficLightState.GREEN
        self.message_queue: List[StandardizedMessage] = []
        self.processing_queue: Dict[str, StandardizedMessage] = {}  # message_id -> message
        self.assigned_agents: Dict[str, str] = {}  # message_id -> agent_id
        
        logger.info(f"🚦 Traffic Light created for stage '{stage_definition.stage_name}'")
        logger.info(f"   Capability: {stage_definition.required_capability}")
        logger.info(f"   Agent Pool: {stage_definition.agent_pool}")
    
    async def receive_message(self, message: StandardizedMessage) -> bool:
        """Receive a message at this traffic light"""
        logger.info(f"🚦 {self.stage_def.stage_name}: Message {message.message_id} arrived")
        
        # Check dependencies
        if not self._dependencies_met(message):
            logger.warning(f"🚦 {self.stage_def.stage_name}: Dependencies not met for {message.message_id}")
            return False
        
        # Add to queue
        self.message_queue.append(message)
        logger.info(f"🚦 {self.stage_def.stage_name}: Queued message {message.message_id} (queue size: {len(self.message_queue)})")
        
        # Try to process immediately
        await self._process_queue()
        return True
    
    def _dependencies_met(self, message: StandardizedMessage) -> bool:
        """Check if stage dependencies are satisfied"""
        for dep_stage in self.stage_def.dependencies:
            if dep_stage not in message.completed_stages:
                return False
        return True
    
    async def _process_queue(self):
        """Process queued messages if capacity allows"""
        while (len(self.processing_queue) < self.stage_def.max_concurrent and 
               self.message_queue and 
               self.state == TrafficLightState.GREEN):
            
            message = self.message_queue.pop(0)
            
            # Find available agent
            available_agent = await self._select_agent()
            if not available_agent:
                # No agents available, put back in queue
                self.message_queue.insert(0, message)
                self.state = TrafficLightState.YELLOW
                logger.warning(f"🚦 {self.stage_def.stage_name}: No agents available, switching to YELLOW")
                break
            
            # Assign to agent
            self.processing_queue[message.message_id] = message
            self.assigned_agents[message.message_id] = available_agent
            
            logger.info(f"🚦 {self.stage_def.stage_name}: Processing {message.message_id} with agent {available_agent}")
            
            # Send to agent
            await self._send_to_agent(message, available_agent)
            
            # Update traffic light state
            if len(self.processing_queue) >= self.stage_def.max_concurrent:
                self.state = TrafficLightState.RED
                logger.info(f"🚦 {self.stage_def.stage_name}: At capacity, switching to RED")
    
    async def _select_agent(self) -> Optional[str]:
        """Select the best available agent for this stage"""
        # Check which agents are available (not currently processing)
        busy_agents = set(self.assigned_agents.values())
        available_agents = [agent for agent in self.stage_def.agent_pool if agent not in busy_agents]
        
        if not available_agents:
            return None
        
        # Simple selection - could be enhanced with load balancing
        return available_agents[0]
    
    async def _send_to_agent(self, message: StandardizedMessage, agent_id: str):
        """Send stage request to agent"""
        # Create agent task payload
        task_payload = {
            "task_id": message.message_id,
            "workflow_id": message.workflow_id,
            "task_name": f"{self.stage_def.stage_name} Task",
            "capability_name": self.stage_def.required_capability,
            "input_data": message.data_payload,
            "stage_context": {
                "stage_name": self.stage_def.stage_name,
                "completed_stages": message.completed_stages,
                "destination_goal": message.destination_goal
            },
            "timeout_seconds": self.stage_def.timeout_seconds
        }
        
        # Send through orchestrator to lobby
        await self.orchestrator.send_task_to_agent(agent_id, task_payload, message.workflow_id)
    
    async def handle_agent_response(self, message_id: str, response: Dict[str, Any]) -> bool:
        """Handle response from agent"""
        if message_id not in self.processing_queue:
            logger.error(f"🚦 {self.stage_def.stage_name}: Unknown message {message_id}")
            return False
        
        original_message = self.processing_queue.pop(message_id)
        agent_id = self.assigned_agents.pop(message_id)
        
        logger.info(f"🚦 {self.stage_def.stage_name}: Received response for {message_id} from {agent_id}")
        
        if response.get("status") == "success":
            # Update message with stage results
            original_message.data_payload.update({
                f"{self.stage_def.stage_name}_result": response.get("result", {}),
                f"{self.stage_def.stage_name}_agent": agent_id,
                f"{self.stage_def.stage_name}_timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Route to next stages
            await self._route_to_next_stages(original_message)
        else:
            logger.error(f"🚦 {self.stage_def.stage_name}: Task failed for {message_id}: {response.get('error')}")
            # Could implement retry logic here
        
        # Update traffic light state
        if self.state == TrafficLightState.RED and len(self.processing_queue) < self.stage_def.max_concurrent:
            self.state = TrafficLightState.GREEN
            logger.info(f"🚦 {self.stage_def.stage_name}: Capacity available, switching to GREEN")
            await self._process_queue()
        
        return True
    
    async def _route_to_next_stages(self, message: StandardizedMessage):
        """Route message to next stages"""
        next_stages = self.stage_def.outputs_to
        
        if not next_stages:
            # This is a terminal stage - workflow complete
            logger.info(f"🚦 {self.stage_def.stage_name}: Terminal stage reached for {message.message_id}")
            await self.orchestrator.handle_workflow_completion(message)
            return
        
        # Route to all next stages
        for next_stage in next_stages:
            next_message = message.clone_for_next_stage(next_stage)
            logger.info(f"🚦 {self.stage_def.stage_name}: Routing {next_message.message_id} to {next_stage}")
            await self.orchestrator.route_message_to_stage(next_message)

# ============================================
# JSON DATA BUS
# ============================================

class JsonDataBus:
    """Central JSON communication hub"""
    
    def __init__(self):
        self.message_history: Dict[str, StandardizedMessage] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
    async def publish_message(self, message: StandardizedMessage) -> bool:
        """Publish a message to the data bus"""
        self.message_history[message.message_id] = message
        
        logger.info(f"🚌 Data Bus: Published message {message.message_id}")
        logger.info(f"   Stage: {message.current_stage}")
        logger.info(f"   Workflow: {message.workflow_id}")
        
        return True
    
    def get_message(self, message_id: str) -> Optional[StandardizedMessage]:
        """Retrieve a message from the bus"""
        return self.message_history.get(message_id)
    
    def get_workflow_messages(self, workflow_id: str) -> List[StandardizedMessage]:
        """Get all messages for a workflow"""
        return [msg for msg in self.message_history.values() if msg.workflow_id == workflow_id]

# ============================================
# WORKFLOW ORCHESTRATOR (TRAFFIC CONTROL SYSTEM)
# ============================================

class WorkflowOrchestrator:
    """Advanced traffic control system for multi-agent workflows"""
    
    def __init__(self, lobby_ref):
        self.lobby = lobby_ref
        self.data_bus = JsonDataBus()
        self.traffic_lights: Dict[str, TrafficLight] = {}
        self.stage_definitions: Dict[str, StageDefinition] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🎛️ Workflow Orchestrator initialized")
    
    def register_stage(self, stage_def: StageDefinition):
        """Register a new orchestration stage"""
        self.stage_definitions[stage_def.stage_name] = stage_def
        self.traffic_lights[stage_def.stage_name] = TrafficLight(stage_def, self)
        
        logger.info(f"🎛️ Registered stage: {stage_def.stage_name}")
    
    async def start_workflow(self, workflow_name: str, goal: str, initial_data: Dict[str, Any], 
                           entry_stage: str, requester_id: str) -> str:
        """Start a new orchestrated workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Create initial message
        initial_message = StandardizedMessage(
            message_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            current_stage=entry_stage,
            message_type="workflow_init",
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_payload=initial_data,
            destination_goal=goal
        )
        
        # Track workflow
        self.active_workflows[workflow_id] = {
            "name": workflow_name,
            "goal": goal,
            "requester_id": requester_id,
            "started_at": datetime.now(timezone.utc),
            "status": "running",
            "stages_completed": [],
            "current_stages": [entry_stage]
        }
        
        logger.info(f"🎛️ Starting workflow: {workflow_name}")
        logger.info(f"   ID: {workflow_id}")
        logger.info(f"   Goal: {goal}")
        logger.info(f"   Entry Stage: {entry_stage}")
        
        # Publish to data bus
        await self.data_bus.publish_message(initial_message)
        
        # Route to entry stage
        await self.route_message_to_stage(initial_message)
        
        return workflow_id
    
    async def route_message_to_stage(self, message: StandardizedMessage):
        """Route a message to the appropriate traffic light"""
        stage_name = message.current_stage
        
        if stage_name not in self.traffic_lights:
            logger.error(f"🎛️ Unknown stage: {stage_name}")
            return False
        
        traffic_light = self.traffic_lights[stage_name]
        return await traffic_light.receive_message(message)
    
    async def handle_agent_response(self, task_id: str, response: Dict[str, Any], workflow_id: str):
        """Handle response from an agent"""
        # Find which traffic light is handling this task
        for traffic_light in self.traffic_lights.values():
            if task_id in traffic_light.processing_queue:
                return await traffic_light.handle_agent_response(task_id, response)
        
        logger.error(f"🎛️ No traffic light found for task {task_id}")
        return False
    
    async def handle_workflow_completion(self, final_message: StandardizedMessage):
        """Handle workflow completion"""
        workflow_id = final_message.workflow_id
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now(timezone.utc)
            workflow["final_result"] = final_message.data_payload
            
            logger.info(f"🎛️ Workflow completed: {workflow['name']}")
            logger.info(f"   ID: {workflow_id}")
            logger.info(f"   Goal achieved: {final_message.destination_goal}")
            
            # Notify requester
            await self._notify_workflow_completion(workflow, final_message)
    
    async def _notify_workflow_completion(self, workflow: Dict[str, Any], final_message: StandardizedMessage):
        """Notify workflow requester of completion"""
        # This would integrate with the lobby's message system
        logger.info(f"🎛️ Notifying requester: {workflow['requester_id']}")
    
    async def send_task_to_agent(self, agent_id: str, task_payload: Dict[str, Any], workflow_id: str):
        """Send task to agent through lobby"""
        # This integrates with the existing lobby message system
        from .message import Message, MessageType, MessagePriority
        
        message = Message(
            sender_id=self.lobby.lobby_id,
            receiver_id=agent_id,
            message_type=MessageType.REQUEST,
            payload=task_payload,
            conversation_id=workflow_id,
            priority=MessagePriority.HIGH
        )
        
        await self.lobby._process_single_message(message)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate progress from traffic lights
        total_stages = len(self.stage_definitions)
        completed_stages = len(workflow.get("stages_completed", []))
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["name"],
            "goal": workflow["goal"],
            "status": workflow["status"],
            "progress": {
                "completed_stages": completed_stages,
                "total_stages": total_stages,
                "current_stages": workflow.get("current_stages", [])
            },
            "started_at": workflow["started_at"].isoformat(),
            "messages_processed": len(self.data_bus.get_workflow_messages(workflow_id))
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        traffic_status = {}
        for stage_name, traffic_light in self.traffic_lights.items():
            traffic_status[stage_name] = {
                "state": traffic_light.state.value,
                "queue_size": len(traffic_light.message_queue),
                "processing": len(traffic_light.processing_queue),
                "max_concurrent": traffic_light.stage_def.max_concurrent
            }
        
        return {
            "active_workflows": len([w for w in self.active_workflows.values() if w["status"] == "running"]),
            "total_workflows": len(self.active_workflows),
            "traffic_lights": traffic_status,
            "data_bus_messages": len(self.data_bus.message_history)
        }

# ============================================
# FACTORY FOR COMMON WORKFLOW PATTERNS
# ============================================

class WorkflowFactory:
    """Factory for creating common workflow patterns"""
    
    @staticmethod
    def create_meta_analysis_workflow(orchestrator: WorkflowOrchestrator, available_agents: Dict[str, List[str]]):
        """Create the META analysis workflow pattern"""
        
        # Define stages
        stages = [
            StageDefinition(
                stage_name="financial_analysis",
                required_capability="financial_analysis",
                agent_pool=available_agents.get("financial_analysis", []),
                max_concurrent=1,
                outputs_to=["data_analysis", "content_creation"]
            ),
            StageDefinition(
                stage_name="data_analysis", 
                required_capability="data_analysis",
                agent_pool=available_agents.get("data_analysis", []),
                max_concurrent=1,
                dependencies=["financial_analysis"],
                outputs_to=["content_creation"]
            ),
            StageDefinition(
                stage_name="content_creation",
                required_capability="content_creation", 
                agent_pool=available_agents.get("content_creation", []),
                max_concurrent=1,
                dependencies=["financial_analysis"],
                outputs_to=[]  # Terminal stage
            )
        ]
        
        # Register all stages
        for stage in stages:
            orchestrator.register_stage(stage)
        
        logger.info("🏭 META analysis workflow pattern created")
        return stages 