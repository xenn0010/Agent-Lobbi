#!/usr/bin/env python3
"""
FAST DATABUS - High Performance Agent Collaboration System
=========================================================
Streamlined JSON-based communication for fast agent collaboration.
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DataBusMessage:
    """Fast DataBus message for agent collaboration"""
    message_id: str
    collaboration_id: str
    task_id: str
    sender_agent: str
    target_agents: List[str]
    message_type: str  # "task_request", "task_response", "collaboration_complete"
    timestamp: str
    data: Dict[str, Any]
    task_state: str = "pending"
    capabilities_needed: List[str] = field(default_factory=list)
    completed_by: List[str] = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DataBusMessage':
        data = json.loads(json_str)
        return cls(**data)

@dataclass
class CollaborationSession:
    """Track a multi-agent collaboration session"""
    collaboration_id: str
    original_requester: str
    task_description: str
    required_capabilities: List[str]
    participating_agents: Set[str]
    completed_capabilities: Set[str]
    task_data: Dict[str, Any]
    status: str  # "active", "completed", "failed"
    created_at: datetime
    last_activity: datetime
    final_result: Optional[Dict[str, Any]] = None
    
    def is_complete(self) -> bool:
        return set(self.required_capabilities).issubset(self.completed_capabilities)

class FastDataBus:
    """High-performance JSON DataBus for agent collaboration"""
    
    def __init__(self, lobby_ref):
        self.lobby = lobby_ref
        self.message_queues: Dict[str, List[DataBusMessage]] = {}
        self.collaboration_sessions: Dict[str, CollaborationSession] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        logger.info("FastDataBus initialized")
    
    async def start_collaboration(self, requester_id: str, task_description: str, 
                                required_capabilities: List[str], initial_data: Dict[str, Any]) -> str:
        """Start a new collaboration session"""
        collaboration_id = f"collab_{uuid.uuid4().hex[:12]}"
        
        session = CollaborationSession(
            collaboration_id=collaboration_id,
            original_requester=requester_id,
            task_description=task_description,
            required_capabilities=required_capabilities,
            participating_agents=set(),
            completed_capabilities=set(),
            task_data=initial_data,
            status="active",
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc)
        )
        
        self.collaboration_sessions[collaboration_id] = session
        
        # Find agents with required capabilities
        suitable_agents = await self._find_agents_by_capabilities(required_capabilities)
        
        if not suitable_agents:
            session.status = "failed"
            logger.warning(f"No agents found for collaboration {collaboration_id}")
            return collaboration_id
        
        # Send tasks to capable agents
        for capability in required_capabilities:
            agents_for_capability = suitable_agents.get(capability, [])
            if agents_for_capability:
                target_agent = agents_for_capability[0]
                session.participating_agents.add(target_agent)
                
                message = DataBusMessage(
                    message_id=str(uuid.uuid4()),
                    collaboration_id=collaboration_id,
                    task_id=f"task_{uuid.uuid4().hex[:8]}",
                    sender_agent=requester_id,
                    target_agents=[target_agent],
                    message_type="task_request",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    data=initial_data.copy(),
                    capabilities_needed=[capability]
                )
                
                await self._deliver_message_to_agent(target_agent, message)
        
        logger.info(f"Started collaboration {collaboration_id} with {len(session.participating_agents)} agents")
        return collaboration_id
    
    async def agent_complete_task(self, agent_id: str, task_id: str, result: Dict[str, Any], 
                                next_agents: List[str] = None) -> bool:
        """Agent reports task completion and optionally hands off to next agents"""
        
        # Find the collaboration session
        collaboration_id = None
        for session in self.collaboration_sessions.values():
            if agent_id in session.participating_agents:
                collaboration_id = session.collaboration_id
                break
        
        if not collaboration_id:
            logger.warning(f"No collaboration found for agent {agent_id} task {task_id}")
            return False
        
        session = self.collaboration_sessions[collaboration_id]
        
        # Update session data with result
        session.task_data.update(result)
        session.last_activity = datetime.now(timezone.utc)
        
        # Mark capabilities as completed (simple approach - mark all required capabilities)
        for capability in session.required_capabilities:
            session.completed_capabilities.add(capability)
        
        # If there are next agents, create handoff
        if next_agents:
            for next_agent in next_agents:
                session.participating_agents.add(next_agent)
                
                handoff_message = DataBusMessage(
                    message_id=str(uuid.uuid4()),
                    collaboration_id=collaboration_id,
                    task_id=f"task_{uuid.uuid4().hex[:8]}",
                    sender_agent=agent_id,
                    target_agents=[next_agent],
                    message_type="task_request",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    data=session.task_data.copy()
                )
                
                await self._deliver_message_to_agent(next_agent, handoff_message)
            logger.info(f"Task {task_id} handed off from {agent_id} to {next_agents}")
        
        # Check if collaboration is complete
        if session.is_complete():
            session.status = "completed"
            session.final_result = session.task_data
            
            # Notify original requester
            completion_message = DataBusMessage(
                message_id=str(uuid.uuid4()),
                collaboration_id=collaboration_id,
                task_id=task_id,
                sender_agent="databus",
                target_agents=[session.original_requester],
                message_type="collaboration_complete",
                timestamp=datetime.now(timezone.utc).isoformat(),
                data=session.final_result
            )
            
            await self._deliver_message_to_agent(session.original_requester, completion_message)
            logger.info(f"Collaboration {collaboration_id} completed successfully")
        
        return True
    
    async def _deliver_message_to_agent(self, agent_id: str, message: DataBusMessage):
        """Deliver message to specific agent"""
        # Add to agent's message queue
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = []
        
        self.message_queues[agent_id].append(message)
        
        # Try WebSocket delivery first (fastest)
        if agent_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[agent_id]
                await websocket.send(message.to_json())
                logger.debug(f"Message delivered via WebSocket to {agent_id}")
                return
            except Exception as e:
                logger.warning(f"WebSocket delivery failed for {agent_id}: {e}")
        
        # Fallback to HTTP task assignment via lobby
        try:
            task_payload = {
                "task_id": message.task_id,
                "collaboration_id": message.collaboration_id,
                "task_data": message.data,
                "capabilities_needed": message.capabilities_needed,
                "sender": message.sender_agent,
                "databus_message": True
            }
            
            if hasattr(self.lobby, 'assign_task_to_agent'):
                await self.lobby.assign_task_to_agent(agent_id, task_payload)
                logger.debug(f"Message delivered via HTTP to {agent_id}")
                
        except Exception as e:
            logger.error(f"Failed to deliver message to {agent_id}: {e}")
    
    async def _find_agents_by_capabilities(self, required_capabilities: List[str]) -> Dict[str, List[str]]:
        """Find agents that have the required capabilities"""
        suitable_agents = {}
        
        for agent_id, agent_info in self.lobby.agents.items():
            if isinstance(agent_info, dict):
                agent_capabilities = agent_info.get('capabilities', [])
                
                for capability in required_capabilities:
                    if capability in agent_capabilities:
                        if capability not in suitable_agents:
                            suitable_agents[capability] = []
                        suitable_agents[capability].append(agent_id)
        
        return suitable_agents
    
    def register_agent_websocket(self, agent_id: str, websocket):
        """Register agent's WebSocket connection for fast messaging"""
        self.websocket_connections[agent_id] = websocket
        logger.info(f"Agent {agent_id} registered WebSocket connection")
    
    def unregister_agent_websocket(self, agent_id: str):
        """Unregister agent's WebSocket connection"""
        if agent_id in self.websocket_connections:
            del self.websocket_connections[agent_id]
            logger.info(f"Agent {agent_id} WebSocket connection removed")
    
    def get_collaboration_status(self, collaboration_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a collaboration session"""
        session = self.collaboration_sessions.get(collaboration_id)
        if not session:
            return None
        
        return {
            "collaboration_id": session.collaboration_id,
            "status": session.status,
            "original_requester": session.original_requester,
            "task_description": session.task_description,
            "required_capabilities": session.required_capabilities,
            "participating_agents": list(session.participating_agents),
            "completed_capabilities": list(session.completed_capabilities),
            "progress": len(session.completed_capabilities) / len(session.required_capabilities) if session.required_capabilities else 0,
            "final_result": session.final_result
        }
    
    def get_agent_tasks(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get pending tasks for an agent"""
        if agent_id not in self.message_queues:
            return []
        
        return [
            {
                "message_id": msg.message_id,
                "task_id": msg.task_id,
                "collaboration_id": msg.collaboration_id,
                "sender": msg.sender_agent,
                "data": msg.data,
                "capabilities_needed": msg.capabilities_needed,
                "timestamp": msg.timestamp
            }
            for msg in self.message_queues[agent_id]
            if msg.task_state == "pending"
        ] 