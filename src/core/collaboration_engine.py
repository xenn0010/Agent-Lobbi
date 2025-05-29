"""
Multi-Agent Collaboration Engine for Agent Lobby
Handles complex workflows, state management, and agent coordination
"""
import asyncio
import uuid
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import json

from .message import Message, MessageType, MessagePriority


class WorkflowStatus(Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Individual task within a workflow"""
    task_id: str
    workflow_id: str
    name: str
    required_capability: str
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # Other task_ids this depends on
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: Optional[int] = 300  # 5 minute default timeout


@dataclass
class Workflow:
    """Multi-agent workflow definition and state"""
    workflow_id: str
    name: str
    description: str
    created_by: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.CREATED
    shared_state: Dict[str, Any] = field(default_factory=dict)  # Shared between agents
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    participants: Set[str] = field(default_factory=set)  # Agent IDs participating
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgentCollaboration:
    """Tracks collaboration between specific agents"""
    collab_id: str
    agent_ids: Set[str]
    purpose: str
    shared_context: Dict[str, Any] = field(default_factory=dict)
    message_history: List[str] = field(default_factory=list)  # Message IDs
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success_count: int = 0
    failure_count: int = 0


class CollaborationEngine:
    """Engine for managing multi-agent collaborations and workflows"""
    
    def __init__(self, lobby_ref):
        self.lobby = lobby_ref
        self.workflows: Dict[str, Workflow] = {}
        self.active_collaborations: Dict[str, AgentCollaboration] = {}
        self.agent_workloads: Dict[str, Set[str]] = {}  # agent_id -> set of task_ids
        self.capability_agents: Dict[str, List[str]] = {}  # capability -> [agent_ids]
        self._workflow_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.collaboration_success_rates: Dict[tuple, float] = {}  # (agent1, agent2) -> success_rate
        self.agent_performance: Dict[str, Dict] = {}  # agent_id -> performance metrics
        
    async def create_workflow(self, name: str, description: str, created_by: str, 
                            task_definitions: List[Dict[str, Any]]) -> str:
        """Create a new multi-agent workflow"""
        workflow_id = str(uuid.uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            created_by=created_by
        )
        
        # Create tasks
        for task_def in task_definitions:
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                workflow_id=workflow_id,
                name=task_def["name"],
                required_capability=task_def["capability"],
                input_data=task_def.get("input", {}),
                dependencies=task_def.get("dependencies", []),
                timeout_seconds=task_def.get("timeout", 300)
            )
            workflow.tasks[task_id] = task
            
        self.workflows[workflow_id] = workflow
        
        print(f"COLLAB ENGINE: Created workflow '{name}' with {len(task_definitions)} tasks")
        return workflow_id
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start executing a workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(timezone.utc)
        
        print(f"COLLAB ENGINE: Starting workflow '{workflow.name}'")
        
        # Start by finding ready tasks (no dependencies)
        await self._process_workflow_tasks(workflow_id)
        return True
    
    async def _process_workflow_tasks(self, workflow_id: str):
        """Process all ready tasks in a workflow"""
        workflow = self.workflows[workflow_id]
        
        ready_tasks = []
        for task in workflow.tasks.values():
            if task.status == TaskStatus.PENDING and self._are_dependencies_met(task, workflow):
                ready_tasks.append(task)
        
        print(f"COLLAB ENGINE: Found {len(ready_tasks)} ready tasks in workflow '{workflow.name}'")
        
        # Assign and start ready tasks
        for task in ready_tasks:
            await self._assign_and_start_task(task, workflow)
    
    def _are_dependencies_met(self, task: Task, workflow: Workflow) -> bool:
        """Check if all task dependencies are completed"""
        for dep_task_id in task.dependencies:
            if dep_task_id in workflow.tasks:
                dep_task = workflow.tasks[dep_task_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        return True
    
    async def _assign_and_start_task(self, task: Task, workflow: Workflow):
        """Find suitable agent and assign task"""
        # Find agents with required capability
        suitable_agents = await self._find_agents_for_capability(task.required_capability)
        
        if not suitable_agents:
            print(f"COLLAB ENGINE: No agents found for capability '{task.required_capability}'")
            task.status = TaskStatus.FAILED
            task.error = f"No agents available for capability: {task.required_capability}"
            return
        
        # Select best agent (least loaded, highest success rate)
        best_agent = self._select_best_agent(suitable_agents, task)
        task.assigned_agent = best_agent
        task.status = TaskStatus.ASSIGNED
        task.started_at = datetime.now(timezone.utc)
        
        # Add to agent workload
        if best_agent not in self.agent_workloads:
            self.agent_workloads[best_agent] = set()
        self.agent_workloads[best_agent].add(task.task_id)
        
        # Add agent to workflow participants
        workflow.participants.add(best_agent)
        
        print(f"COLLAB ENGINE: Assigned task '{task.name}' to agent '{best_agent}'")
        
        # Send task to agent
        await self._send_task_to_agent(task, workflow, best_agent)
    
    async def _find_agents_for_capability(self, capability: str) -> List[str]:
        """Find all agents that have a specific capability"""
        suitable_agents = []
        
        # Check lobby's agent capabilities
        for agent_id, capabilities in self.lobby.agent_capabilities.items():
            if capability in capabilities:
                # Check if agent is online
                if agent_id in self.lobby.agents:
                    suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _select_best_agent(self, candidates: List[str], task: Task) -> str:
        """Select the best agent for a task based on workload and performance"""
        if not candidates:
            return None
        
        best_agent = None
        best_score = -1
        
        for agent_id in candidates:
            # Calculate score based on:
            # 1. Current workload (lower is better)
            # 2. Historical performance (higher is better)
            # 3. Success rate with similar tasks
            
            workload = len(self.agent_workloads.get(agent_id, set()))
            performance = self.agent_performance.get(agent_id, {}).get("avg_success_rate", 0.5)
            
            # Simple scoring: performance - workload_penalty
            workload_penalty = workload * 0.1
            score = performance - workload_penalty
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    async def _send_task_to_agent(self, task: Task, workflow: Workflow, agent_id: str):
        """Send a task to an agent for execution"""
        task_payload = {
            "task_id": task.task_id,
            "workflow_id": task.workflow_id,
            "task_name": task.name,
            "capability_name": task.required_capability,
            "input_data": task.input_data,
            "shared_state": workflow.shared_state,
            "timeout_seconds": task.timeout_seconds
        }
        
        message = Message(
            sender_id=self.lobby.lobby_id,
            receiver_id=agent_id,
            message_type=MessageType.REQUEST,
            payload=task_payload,
            conversation_id=workflow.workflow_id,
            priority=MessagePriority.HIGH
        )
        
        task.status = TaskStatus.IN_PROGRESS
        await self.lobby.route_message(message)
    
    async def handle_task_result(self, message: Message):
        """Handle task completion from an agent"""
        payload = message.payload
        task_id = payload.get("task_id")
        workflow_id = message.conversation_id
        
        if not workflow_id or workflow_id not in self.workflows:
            print(f"COLLAB ENGINE: Unknown workflow_id: {workflow_id}")
            return
        
        workflow = self.workflows[workflow_id]
        if task_id not in workflow.tasks:
            print(f"COLLAB ENGINE: Unknown task_id: {task_id}")
            return
        
        task = workflow.tasks[task_id]
        agent_id = message.sender_id
        
        # Update task status
        if payload.get("status") == "success":
            task.status = TaskStatus.COMPLETED
            task.result = payload.get("result", {})
            task.completed_at = datetime.now(timezone.utc)
            
            # Update shared state if provided
            shared_updates = payload.get("shared_state_updates", {})
            workflow.shared_state.update(shared_updates)
            
            print(f"COLLAB ENGINE: Task '{task.name}' completed by '{agent_id}'")
            
            # Update agent performance
            self._update_agent_performance(agent_id, True, task)
            
        else:
            task.status = TaskStatus.FAILED
            task.error = payload.get("error", "Unknown error")
            task.completed_at = datetime.now(timezone.utc)
            
            print(f"COLLAB ENGINE: Task '{task.name}' failed: {task.error}")
            
            # Update agent performance
            self._update_agent_performance(agent_id, False, task)
        
        # Remove from agent workload
        if agent_id in self.agent_workloads:
            self.agent_workloads[agent_id].discard(task_id)
        
        # Check if workflow is complete or can continue
        await self._check_workflow_completion(workflow_id)
    
    def _update_agent_performance(self, agent_id: str, success: bool, task: Task):
        """Update agent performance metrics"""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "avg_success_rate": 0.5,
                "avg_completion_time": 0,
                "capabilities_used": set()
            }
        
        perf = self.agent_performance[agent_id]
        perf["total_tasks"] += 1
        
        if success:
            perf["successful_tasks"] += 1
        
        perf["avg_success_rate"] = perf["successful_tasks"] / perf["total_tasks"]
        perf["capabilities_used"].add(task.required_capability)
        
        # Calculate completion time
        if task.started_at and task.completed_at:
            completion_time = (task.completed_at - task.started_at).total_seconds()
            if perf["avg_completion_time"] == 0:
                perf["avg_completion_time"] = completion_time
            else:
                # Running average
                perf["avg_completion_time"] = (perf["avg_completion_time"] + completion_time) / 2
    
    async def _check_workflow_completion(self, workflow_id: str):
        """Check if workflow is complete and handle accordingly"""
        workflow = self.workflows[workflow_id]
        
        # Count task statuses
        completed = sum(1 for task in workflow.tasks.values() if task.status == TaskStatus.COMPLETED)
        failed = sum(1 for task in workflow.tasks.values() if task.status == TaskStatus.FAILED)
        total = len(workflow.tasks)
        
        if completed + failed == total:
            # Workflow is done
            if failed == 0:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.result = {"message": "All tasks completed successfully"}
                print(f"COLLAB ENGINE: Workflow '{workflow.name}' COMPLETED successfully!")
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = f"{failed} out of {total} tasks failed"
                print(f"COLLAB ENGINE: Workflow '{workflow.name}' FAILED with {failed} failed tasks")
            
            workflow.completed_at = datetime.now(timezone.utc)
            
            # Notify workflow creator
            await self._notify_workflow_completion(workflow)
            
        else:
            # Process any newly ready tasks
            await self._process_workflow_tasks(workflow_id)
    
    async def _notify_workflow_completion(self, workflow: Workflow):
        """Notify the workflow creator of completion"""
        result_payload = {
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "status": workflow.status.value,
            "completed_tasks": len([t for t in workflow.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in workflow.tasks.values() if t.status == TaskStatus.FAILED]),
            "total_tasks": len(workflow.tasks),
            "result": workflow.result,
            "error": workflow.error,
            "participants": list(workflow.participants),
            "duration_seconds": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.started_at else 0
        }
        
        message = Message(
            sender_id=self.lobby.lobby_id,
            receiver_id=workflow.created_by,
            message_type=MessageType.RESPONSE,
            payload=result_payload,
            conversation_id=workflow.workflow_id,
            priority=MessagePriority.HIGH
        )
        
        await self.lobby.route_message(message)
    
    # Real-time Collaboration Methods
    
    async def create_collaboration_session(self, agent_ids: List[str], purpose: str) -> str:
        """Create a real-time collaboration session between agents"""
        collab_id = str(uuid.uuid4())
        collaboration = AgentCollaboration(
            collab_id=collab_id,
            agent_ids=set(agent_ids),
            purpose=purpose
        )
        
        self.active_collaborations[collab_id] = collaboration
        
        # Notify all agents about the collaboration
        for agent_id in agent_ids:
            if agent_id in self.lobby.agents:
                await self._notify_collaboration_start(agent_id, collaboration)
        
        print(f"COLLAB ENGINE: Created collaboration session {collab_id} with {len(agent_ids)} agents")
        return collab_id
    
    async def _notify_collaboration_start(self, agent_id: str, collaboration: AgentCollaboration):
        """Notify an agent about starting a collaboration"""
        payload = {
            "collaboration_id": collaboration.collab_id,
            "purpose": collaboration.purpose,
            "participants": list(collaboration.agent_ids),
            "shared_context": collaboration.shared_context
        }
        
        message = Message(
            sender_id=self.lobby.lobby_id,
            receiver_id=agent_id,
            message_type=MessageType.INFO,
            payload={
                "event_type": "collaboration_started",
                "data": payload
            },
            conversation_id=collaboration.collab_id
        )
        
        await self.lobby.route_message(message)
    
    async def broadcast_to_collaboration(self, collab_id: str, sender_id: str, content: Dict[str, Any]):
        """Broadcast a message to all participants in a collaboration"""
        if collab_id not in self.active_collaborations:
            return False
        
        collaboration = self.active_collaborations[collab_id]
        collaboration.last_activity = datetime.now(timezone.utc)
        
        # Send to all participants except sender
        for agent_id in collaboration.agent_ids:
            if agent_id != sender_id and agent_id in self.lobby.agents:
                message = Message(
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    message_type=MessageType.INFO,
                    payload={
                        "collaboration_broadcast": True,
                        "collaboration_id": collab_id,
                        "content": content
                    },
                    conversation_id=collab_id
                )
                
                await self.lobby.route_message(message)
                collaboration.message_history.append(message.message_id)
        
        return True
    
    # Utility Methods
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": {
                "completed": len([t for t in workflow.tasks.values() if t.status == TaskStatus.COMPLETED]),
                "in_progress": len([t for t in workflow.tasks.values() if t.status == TaskStatus.IN_PROGRESS]),
                "pending": len([t for t in workflow.tasks.values() if t.status == TaskStatus.PENDING]),
                "failed": len([t for t in workflow.tasks.values() if t.status == TaskStatus.FAILED]),
                "total": len(workflow.tasks)
            },
            "participants": list(workflow.participants),
            "shared_state": workflow.shared_state,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None
        }
    
    def get_agent_workload(self, agent_id: str) -> Dict[str, Any]:
        """Get current workload for an agent"""
        workload = self.agent_workloads.get(agent_id, set())
        performance = self.agent_performance.get(agent_id, {})
        
        return {
            "agent_id": agent_id,
            "active_tasks": len(workload),
            "task_ids": list(workload),
            "performance": performance,
            "collaborations": [
                collab_id for collab_id, collab in self.active_collaborations.items()
                if agent_id in collab.agent_ids
            ]
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall collaboration system statistics"""
        return {
            "active_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.RUNNING]),
            "completed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]),
            "active_collaborations": len(self.active_collaborations),
            "total_agents": len(self.agent_performance),
            "avg_system_success_rate": sum(
                perf.get("avg_success_rate", 0) for perf in self.agent_performance.values()
            ) / len(self.agent_performance) if self.agent_performance else 0
        } 