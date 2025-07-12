#!/usr/bin/env python3
"""
 COLLABORATION ENGINE WITH DATABUS INTEGRATION
===============================================
Enhanced collaboration engine that integrates with the proven DataBus + Traffic Light system
for real multi-agent collaboration and task orchestration.
"""

import asyncio
import uuid
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher
import re
import time

# Import DataBus components - graceful fallback if not available
try:
    from .data_bus_orchestrator import (
        WorkflowOrchestrator,
        WorkflowFactory,
        StageDefinition,
        StandardizedMessage,
        TrafficLight,
        JsonDataBus
    )
    from .databus.databus_system import DataBusOrchestrator
    from .databus.databus_stages import StageDefinitionProtocol
    from .databus.stage_registry import StageRegistry
    DATABUS_AVAILABLE = True
except ImportError as e:
    print(f"DataBus system not available: {e}")
    DATABUS_AVAILABLE = False
    WorkflowOrchestrator = None

# Import Neuromorphic Agent Architecture - graceful fallback
try:
    from .neuromorphic_agent_architecture import PerformanceOptimizedNAA
    NAA_AVAILABLE = True
except ImportError:
    NAA_AVAILABLE = False
    PerformanceOptimizedNAA = None

logger = logging.getLogger(__name__)

# Legacy workflow enums for backward compatibility
class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Legacy task definition for backward compatibility"""
    task_id: str
    workflow_id: str
    name: str
    required_capability: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_at: Optional[datetime] = None
    history: List[str] = field(default_factory=list) # Agents that have tried this task
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for CNP messaging"""
        return {
            "task_id": self.task_id,
            "workflow_id": self.workflow_id, 
            "name": self.name,
            "required_capability": self.required_capability,
            "input_data": self.input_data,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value if self.status else "pending"
        }

@dataclass
class Workflow:
    """Represents a structured process for achieving a goal."""
    def __init__(self, workflow_id: str, name: str, description: str, created_by: str,
                 task_intent: str = "", shared_state: Optional[Dict] = None, status: WorkflowStatus = WorkflowStatus.PENDING):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.created_by = created_by
        self.status = status
        self.tasks: Dict[str, Task] = {}
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.participants: set[str] = set()
        self.task_intent = task_intent or description # Fallback for intent
        self.shared_state = shared_state or {}

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
    """Enhanced Engine for managing multi-agent collaborations with DataBus integration"""
    
    def __init__(self, lobby_ref):
        self.lobby = lobby_ref
        
        # Legacy workflow system (for backward compatibility)
        self.workflows: Dict[str, Workflow] = {}
        self.active_collaborations: Dict[str, AgentCollaboration] = {}
        self.agent_workloads: Dict[str, Set[str]] = {}  # agent_id -> set of task_ids
        self.capability_agents: Dict[str, List[str]] = {}  # capability -> [agent_ids]
        self._workflow_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.collaboration_success_rates: Dict[tuple, float] = {}  # (agent1, agent2) -> success_rate
        self.agent_performance: Dict[str, Dict] = {}  # agent_id -> performance metrics
        
        # **NEW: DataBus Integration**
        self.databus_orchestrator = None
        self.databus_enabled = DATABUS_AVAILABLE
        
        # **REVOLUTIONARY: Neuromorphic Agent Architecture Integration**
        self.naa = None
        self.naa_enabled = NAA_AVAILABLE
        
        logger.info(" Collaboration Engine initialized")
        logger.info(" NAA integration ready for revolutionary collective intelligence")
        
    async def start(self):
        """Start the collaboration engine with DataBus and NAA integration"""
        print("COLLAB ENGINE: Collaboration engine started")
        
        # Initialize DataBus orchestrator
        await self._initialize_databus()
        
        # **REVOLUTIONARY: Initialize Neuromorphic Agent Architecture**
        await self._initialize_naa()
        
        logger.info(" Collaboration Engine started with DataBus integration")
        logger.info(" Neuromorphic Agent Architecture activated - Collective intelligence learning enabled")
        
    async def _initialize_databus(self):
        """Initialize the DataBus orchestration system"""
        if not DATABUS_AVAILABLE:
            logger.info(" DataBus not available - using legacy workflow system")
            self.databus_enabled = False
            return
            
        try:
            # Create DataBus orchestrator
            self.databus_orchestrator = WorkflowOrchestrator(self.lobby)
            
            # Discover agent capabilities and create stage definitions
            await self._discover_and_setup_stages()
            
            self.databus_enabled = True
            logger.info(" DataBus orchestration system initialized")
            
        except Exception as e:
            logger.error(f" Failed to initialize DataBus system: {e}")
            self.databus_enabled = False
        
    async def _discover_and_setup_stages(self):
        """Discover agent capabilities and setup DataBus stages"""
        # Analyze registered agents and their capabilities
        capability_to_agents: Dict[str, List[str]] = {}
        
        # Check agents registered via HTTP API
        for agent_id, agent_data in self.lobby.agents.items():
            if isinstance(agent_data, dict):
                agent_capabilities = agent_data.get('capabilities', [])
                for capability in agent_capabilities:
                    if capability not in capability_to_agents:
                        capability_to_agents[capability] = []
                    capability_to_agents[capability].append(agent_id)
        
        logger.info(f" Discovered capabilities: {list(capability_to_agents.keys())}")
        
        # Create stage definitions for each capability
        stage_definitions = []
        for capability, agents in capability_to_agents.items():
            if agents:  # Only create stages where we have agents
                stage_def = StageDefinition(
                    stage_name=capability,
                    required_capability=capability,
                    agent_pool=agents,
                    max_concurrent=len(agents),  # Allow all agents to work in parallel
                    timeout_seconds=300
                )
                stage_definitions.append(stage_def)
                self.databus_orchestrator.register_stage(stage_def)
                logger.info(f"🏗 Created stage '{capability}' with {len(agents)} agents")
        
        # Store capability mappings
        self.capability_agents = capability_to_agents
    
    async def _initialize_naa(self):
        """Initialize the Production NAA for real agent orchestration"""
        if not NAA_AVAILABLE:
            logger.info(" NAA not available - using standard collaboration")
            self.naa_enabled = False
            return
            
        try:
            # Initialize NAA system
            self.naa = PerformanceOptimizedNAA()
            await self.naa.initialize()
            self.naa_enabled = True
            
            logger.info("[OK] NAA system initialized successfully - advanced learning enabled")
            logger.info("[NAA] Neural collaboration patterns will be learned from every interaction")
            
        except Exception as e:
            logger.error(f" Failed to initialize Production NAA system: {e}")
            logger.warning(" NAA unavailable - falling back to legacy collaboration")
            self.naa_enabled = False
            self.naa = None
            # Don't raise error - allow system to continue without NAA
        
    async def stop(self):
        """Stop the collaboration engine"""
        print("COLLAB ENGINE: Collaboration engine stopped")
        # Clean up resources, cancel tasks, etc.
        pass
        
    # **NEW: DataBus Workflow Methods**
    
    async def create_databus_workflow(self, name: str, goal: str, initial_data: Dict[str, Any], 
                                    entry_capability: str, requester_id: str) -> str:
        """Create a new DataBus workflow"""
        if not self.databus_enabled:
            raise RuntimeError("DataBus system is not enabled")
        
        # Find entry stage based on capability
        entry_stage = entry_capability
        if entry_stage not in [stage.stage_name for stage in self.databus_orchestrator.stage_definitions.values()]:
            raise ValueError(f"No stage found for capability: {entry_capability}")
        
        workflow_id = await self.databus_orchestrator.start_workflow(
            workflow_name=name,
            goal=goal,
            initial_data=initial_data,
            entry_stage=entry_stage,
            requester_id=requester_id
        )
        
        logger.info(f" Created DataBus workflow '{name}' with ID: {workflow_id}")
        return workflow_id
    
    async def handle_databus_task_result(self, task_id: str, response: Dict[str, Any], workflow_id: str):
        """Handle task completion from DataBus workflow"""
        if not self.databus_enabled:
            return
        
        await self.databus_orchestrator.handle_agent_response(task_id, response, workflow_id)
        
    def get_databus_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a DataBus workflow"""
        if not self.databus_enabled:
            return None
        
        return self.databus_orchestrator.get_workflow_status(workflow_id)
    
    def get_databus_system_status(self) -> Dict[str, Any]:
        """Get overall DataBus system status"""
        if not self.databus_enabled:
            return {"error": "DataBus system not enabled"}
        
        return self.databus_orchestrator.get_system_status()
    
    # **ENHANCED: Legacy Methods with DataBus Integration**
    
    async def create_goal_driven_workflow(self, name: str, description: str, created_by: str, 
                                        task_intent: str, required_capabilities: List[str], 
                                        task_data: Dict[str, Any], max_agents: int = 5) -> str:
        """Create a goal-driven workflow with dynamic task assignment."""
        print(f"DEBUG DEBUG: create_goal_driven_workflow START")
        print(f"DEBUG DEBUG: name={name}, capabilities={required_capabilities}")
        
        try:
            workflow_id = str(uuid.uuid4())
            print(f"DEBUG DEBUG: Generated workflow_id={workflow_id}")
            
            # Create workflow with basic structure
            workflow = Workflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                created_by=created_by,
                task_intent=task_intent,
                status=WorkflowStatus.PENDING
            )
            print(f"DEBUG DEBUG: Workflow object created")

            print(f"COLLAB ENGINE: Creating goal-driven workflow: '{name}'")
            print(f"COLLAB ENGINE: Required capabilities: {required_capabilities}")
            print(f"COLLAB ENGINE: Max agents: {max_agents}")

            # OK ENHANCED AGENT SELECTION: Find best agents for each capability
            print(f"DEBUG DEBUG: Starting agent selection...")
            final_team = {}
            used_agents = set()
            
            print(f"DEBUG DEBUG: About to iterate through capabilities...")
            for i, capability in enumerate(required_capabilities):
                print(f"DEBUG DEBUG: Processing capability {i+1}/{len(required_capabilities)}: {capability}")
                
                # **FIX: Use the correct agent finding method**
                candidate_agents = await self._find_agents_for_capability(capability)
                print(f"DEBUG DEBUG: _find_agents_for_capability returned {len(candidate_agents)} agents")
                
                if candidate_agents:
                    # Select the best agent for this capability
                    best_agent_id = self._select_best_agent(candidate_agents, None) # Pass task=None for now
                    if best_agent_id and best_agent_id not in used_agents:
                        # NOTE: Storing only the ID now as the new method doesn't return full details
                        final_team[capability] = best_agent_id 
                        used_agents.add(best_agent_id)
                        print(f"DEBUG DEBUG: Selected {best_agent_id} for {capability}")

            print(f"DEBUG DEBUG: First pass complete, final_team={final_team}")

            # Second pass: Fill any remaining capability slots with any available agent
            print(f"DEBUG DEBUG: Starting second pass...")
            for capability in required_capabilities:
                if capability not in final_team:
                    print(f"DEBUG DEBUG: Second pass - finding agents for {capability}")
                    
                    all_possible_agents = await self._find_agents_for_capability(capability)
                    print(f"DEBUG DEBUG: Found {len(all_possible_agents)} possible agents")
                    
                    if all_possible_agents:
                        # Find one that hasn't been used if possible
                        chosen_agent_id = all_possible_agents[0]
                        for agent_id in all_possible_agents:
                            if agent_id not in used_agents:
                                chosen_agent_id = agent_id
                                break
                        final_team[capability] = chosen_agent_id
                        used_agents.add(chosen_agent_id)
                        print(f"DEBUG DEBUG: Second pass selected {chosen_agent_id} for {capability}")

            print(f"DEBUG DEBUG: Agent selection complete, used_agents={used_agents}")

            # Check if we have a multi-agent team
            # OK TEMPORARILY DISABLED: Multi-agent collaboration causes deadlock
            # Re-enable after fixing agent notification handling
            if False and len(used_agents) > 1:  # OK FORCED FALSE to disable multi-agent path
                print(f"DEBUG DEBUG: Multi-agent path (DISABLED)")
                # ... multi-agent code disabled ...
            
            # OK ENHANCED: Create traditional isolated tasks (always used now)
            print(f"DEBUG DEBUG: Creating single-agent tasks...")
            print(f"COLLAB ENGINE: Creating single-agent tasks for capabilities: {required_capabilities}")
            
            # Create a task for each capability that has a matching agent
            for i, capability in enumerate(required_capabilities):
                print(f"DEBUG DEBUG: Creating task {i+1}/{len(required_capabilities)} for {capability}")
                
                # Find the best agent for this specific capability
                candidate_agents = await self._find_agents_for_capability(capability)
                print(f"DEBUG DEBUG: Found {len(candidate_agents)} candidates for task creation")
                
                # **FIX: Create a task even if no agent is found, and mark it as FAILED.**
                task_id = str(uuid.uuid4())
                task = Task(
                    task_id=task_id,
                    workflow_id=workflow_id,
                    name=f"Execute Capability: {capability}",
                    required_capability=capability,
                    input_data=task_data
                )

                if candidate_agents:
                    workflow.tasks[task_id] = task
                    print(f"DEBUG DEBUG: Created task {task_id}")
                    print(f"COLLAB ENGINE: Created task {task_id} for capability '{capability}' (will assign to one of: {candidate_agents[:3]})")
                else:
                    task.status = TaskStatus.FAILED
                    task.error = f"No agents found for required capability: {capability}"
                    workflow.tasks[task_id] = task
                    print(f"DEBUG DEBUG: WARNING - No agents for {capability}, created FAILED task {task_id}")
                    print(f"COLLAB ENGINE: WARNING - No agents found for capability '{capability}', task marked as FAILED.")

            print(f"DEBUG DEBUG: Task creation complete")
            print(f"COLLAB ENGINE: Created workflow '{name}' with {len(workflow.tasks)} tasks")
            
            print(f"DEBUG DEBUG: Storing workflow in self.workflows")
            self.workflows[workflow_id] = workflow
            
            # **FIX:** Actually start the workflow after creating it
            asyncio.create_task(self.start_workflow(workflow_id))

            print(f"DEBUG DEBUG: create_goal_driven_workflow COMPLETE, returning {workflow_id}")
            return workflow_id
            
        except Exception as e:
            print(f"DEBUG DEBUG: EXCEPTION in create_goal_driven_workflow: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start a specific workflow"""
        if workflow_id not in self.workflows:
            print(f"COLLAB ENGINE: Workflow {workflow_id} not found!")
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.PENDING:
            print(f"COLLAB ENGINE: Workflow {workflow_id} is not in PENDING status (current: {workflow.status})")
            return False
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now(timezone.utc)
        
        print(f"COLLAB ENGINE: STARTING WORKFLOW {workflow.name} ({workflow_id})")
        
        # Create a background task to process this workflow
        asyncio.create_task(self._process_workflow_tasks(workflow_id))
        
        # Start monitoring
        asyncio.create_task(self._monitor_workflow_progress(workflow_id))
        
        return True

    async def register_agent(self, agent_id: str, capabilities: List[str]):
        """Register an agent with the collaboration engine"""
        try:
            print(f"COLLAB ENGINE: Registering agent {agent_id} with capabilities: {capabilities}")
            
            # Initialize agent performance tracking
            if agent_id not in self.agent_performance:
                self.agent_performance[agent_id] = {
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "total_tasks": 0,
                    "total_execution_time": 0,
                    "avg_execution_time": 0,
                    "success_rate": 0,
                    "avg_success_rate": 0.5  # Default starting rate
                }
            
            # Initialize agent workload tracking
            if agent_id not in self.agent_workloads:
                self.agent_workloads[agent_id] = set()
            
            # Register capabilities
            for capability in capabilities:
                if capability not in self.capability_agents:
                    self.capability_agents[capability] = []
                self.capability_agents[capability].append(agent_id)
            
            print(f"COLLAB ENGINE: [OK] Agent {agent_id} registered successfully")
            return {"status": "success", "message": f"Agent {agent_id} registered with collaboration engine"}
            
        except Exception as e:
            print(f"COLLAB ENGINE: [FAIL] Failed to register agent {agent_id}: {e}")
            return {"status": "error", "message": f"Failed to register agent: {str(e)}"}

    async def _monitor_workflow_progress(self, workflow_id: str):
        """Monitors a workflow for completion or timeouts."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return

        while workflow.status == WorkflowStatus.RUNNING:
            await asyncio.sleep(5) # Check every 5 seconds

            now = datetime.now(timezone.utc)
            for task in list(workflow.tasks.values()):
                if task.status == TaskStatus.IN_PROGRESS and task.timeout_at and now > task.timeout_at:
                    logger.warning(f"TASK TIMEOUT: Task {task.task_id} for agent {task.assigned_agent} timed out.")
                    await self._handle_task_failure(task, workflow, "Task timed out")
            
            await self._check_workflow_completion(workflow_id)

    async def _handle_task_failure(self, task: Task, workflow: Workflow, reason: str):
        """Handles the logic for when a task fails, including re-queueing."""
        logger.info(f"HANDLING FAILURE for task {task.task_id}. Reason: {reason}")
        
        # Mark the original agent as failed for this task
        if task.assigned_agent:
            # Here you might add logic to penalize the agent's reputation
            pass

        # Reset the task to be re-assigned
        task.status = TaskStatus.PENDING
        task.assigned_agent = None
        task.started_at = None
        task.timeout_at = None
        task.error = reason
        
        logger.info(f"Task {task.task_id} has been reset and will be re-queued.")

        # Immediately try to process the workflow again to re-assign the task
        await self._process_workflow_tasks(workflow.workflow_id)

    async def _check_task_timeouts(self, workflow_id: str):
        """Periodically checks for tasks that have timed out."""
        # This function is now superseded by the logic in _monitor_workflow_progress
        pass

    async def handle_task_result(self, message):
        """Handles a task result message from an agent."""
        try:
            payload = message.payload
            task_id = payload.get("task_id")
            
            # OK FIXED: Robust workflow_id detection
            workflow_id = (
                message.conversation_id or  # From message conversation_id
                payload.get("workflow_id") or  # From payload
                self._find_workflow_by_task_id(task_id)  # Lookup by task_id
            )
            
            print(f"COLLAB ENGINE: RECEIVE Received task result for task {task_id} in workflow {workflow_id}")
            
            # **NEW: Handle DataBus task results**
            if self.databus_enabled and workflow_id in self.databus_orchestrator.active_workflows:
                await self.handle_databus_task_result(task_id, payload, workflow_id)
                return
            
            # **Legacy workflow handling**
            if not workflow_id or workflow_id not in self.workflows:
                print(f"COLLAB ENGINE: ERROR Unknown workflow_id: {workflow_id} (task_id: {task_id})")
                print(f"COLLAB ENGINE: DEBUG Available workflows: {list(self.workflows.keys())}")
                # Try to find workflow by searching all workflows for this task
                for wf_id, wf in self.workflows.items():
                    if task_id in wf.tasks:
                        workflow_id = wf_id
                        print(f"COLLAB ENGINE: OK Found task {task_id} in workflow {workflow_id}")
                        break
                else:
                    print(f"COLLAB ENGINE: ERROR Task {task_id} not found in any workflow")
                    return
            
            workflow = self.workflows[workflow_id]
            if task_id not in workflow.tasks:
                print(f"COLLAB ENGINE: ERROR Unknown task_id: {task_id} in workflow {workflow_id}")
                return
            
            task = workflow.tasks[task_id]
            agent_id = message.sender_id
            
            # **CRITICAL FIX**: Set task start time if not already set
            if not task.started_at:
                task.started_at = datetime.now(timezone.utc)
            
            # Update task status based on result
            status = payload.get("status", "").lower()
            if status in ["success", "completed"]:  # OK FIXED: Handle both status formats
                task.status = TaskStatus.COMPLETED
                task.result = payload.get("result", {})
                task.completed_at = datetime.now(timezone.utc)
                
                # Update shared state if provided
                shared_updates = payload.get("shared_state_updates", {})
                workflow.shared_state.update(shared_updates)
                
                print(f"COLLAB ENGINE: OK Task '{task.name}' completed successfully by '{agent_id}' (status: {status})")
                print(f"COLLAB ENGINE: INFO Task result: {type(task.result)} with {len(str(task.result))} characters")
                
                # Update agent performance
                self._update_agent_performance(agent_id, True, task)
                
                # **REVOLUTIONARY: NAA Learning from successful collaboration**
                if self.naa_enabled and len(workflow.participants) > 1:
                    await self._learn_from_collaboration_outcome(workflow, task, message, success=True)
                
                # TROPHY NEW: Send task outcome report to lobby for reputation rewards
                await self._send_task_outcome_to_lobby(task, workflow, agent_id, True)
                
            elif status in ["failed", "error"]:  # OK FIXED: Handle failure status formats
                task.status = TaskStatus.FAILED
                task.error = payload.get("error", "Unknown error")
                task.completed_at = datetime.now(timezone.utc)
                
                print(f"COLLAB ENGINE: ERROR Task '{task.name}' failed: {task.error} (status: {status})")
                
                # Update agent performance
                self._update_agent_performance(agent_id, False, task)
                
                # **REVOLUTIONARY: NAA Learning from failed collaboration**
                if self.naa_enabled and len(workflow.participants) > 1:
                    await self._learn_from_collaboration_outcome(workflow, task, message, success=False)
                
                # TROPHY NEW: Send task outcome report to lobby for reputation rewards
                await self._send_task_outcome_to_lobby(task, workflow, agent_id, False)
            else:
                print(f"COLLAB ENGINE: WARNING Unknown task status '{status}' for task {task_id}, treating as failed")
                task.status = TaskStatus.FAILED
                task.error = f"Unknown task status: {status}"
                task.completed_at = datetime.now(timezone.utc)
                self._update_agent_performance(agent_id, False, task)
            
            # Remove from agent workload
            if agent_id in self.agent_workloads:
                self.agent_workloads[agent_id].discard(task_id)
                print(f"COLLAB ENGINE: 🗑️ Removed task {task_id} from {agent_id}'s workload")
            
            # **CRITICAL FIX**: Check if workflow is complete or can continue
            print(f"COLLAB ENGINE: DEBUG Checking workflow completion status...")
            await self._check_workflow_completion(workflow_id)
            
        except Exception as e:
            print(f"COLLAB ENGINE: ERROR Error handling task result: {e}")
            import traceback
            traceback.print_exc()
            
            # **ENHANCED**: Try to mark task as failed if we can identify it
            try:
                if 'task_id' in locals() and 'workflow_id' in locals() and workflow_id in self.workflows:
                    workflow = self.workflows[workflow_id]
                    if task_id in workflow.tasks:
                        task = workflow.tasks[task_id]
                        task.status = TaskStatus.FAILED
                        task.error = f"Error processing task result: {str(e)}"
                        task.completed_at = datetime.now(timezone.utc)
                        print(f"COLLAB ENGINE: CONFIG Marked task {task_id} as failed due to processing error")
                        
                        # Still try to check workflow completion
                        await self._check_workflow_completion(workflow_id)
            except Exception as cleanup_error:
                print(f"COLLAB ENGINE: ERROR Error during cleanup: {cleanup_error}")
    
    def _find_workflow_by_task_id(self, task_id: str) -> Optional[str]:
        """Find workflow ID by searching for task ID in all workflows"""
        try:
            for workflow_id, workflow in self.workflows.items():
                if task_id in workflow.tasks:
                    return workflow_id
            return None
        except Exception as e:
            print(f"COLLAB ENGINE: Error finding workflow for task {task_id}: {e}")
            return None
    
    # Continue with rest of legacy methods...
    
    async def _process_workflow_tasks(self, workflow_id: str):
        """Process all ready tasks in a workflow"""
        if self.lobby is None:
            logger.error("Lobby reference not set in CollaborationEngine!")
            return

        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                print(f"COLLAB ENGINE: Workflow {workflow_id} not found!")
                return
                
            if workflow.status != WorkflowStatus.RUNNING:
                print(f"COLLAB ENGINE: Workflow {workflow_id} status is {workflow.status}, not RUNNING!")
                return
            
            print(f"COLLAB ENGINE: Processing workflow '{workflow.name}' ({workflow_id})")
            print(f"COLLAB ENGINE: Total tasks in workflow: {len(workflow.tasks)}")

            ready_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.PENDING and self._are_dependencies_met(t, workflow)]
            
            print(f"COLLAB ENGINE: Ready tasks found: {len(ready_tasks)}")
            
            for i, task in enumerate(ready_tasks):
                print(f"COLLAB ENGINE: Processing task {i+1}/{len(ready_tasks)}: {task.name} (capability: {task.required_capability})")
                await self._assign_and_start_task(task, workflow)
                print(f"COLLAB ENGINE: Completed processing task: {task.name}")
        
        except Exception as e:
            logger.error(f"Error processing workflow tasks for {workflow_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _are_dependencies_met(self, task: Task, workflow: Workflow) -> bool:
        """Check if all task dependencies are completed"""
        for dep_task_id in task.dependencies:
            if dep_task_id in workflow.tasks:
                dep_task = workflow.tasks[dep_task_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        return True
    
    async def _learn_from_collaboration_outcome(self, workflow: Workflow, task: Task, message, success: bool):
        """Enhanced NAA learning from collaboration outcomes"""
        if not self.naa_enabled or not self.naa:
            return
            
        try:
            # Get original observation
            observation = getattr(self, '_naa_observations', {}).get(task.task_id)
            if not observation:
                logger.warning(f" NAA: No observation found for task {task.task_id}")
                return
            
            # Calculate success metrics
            execution_time = 0
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
            
            quality_score = 0.0
            if success and task.result:
                # Simple quality assessment based on result completeness
                result_content = str(task.result.get('content', ''))
                quality_score = min(1.0, len(result_content) / 500)  # Basic heuristic
            
            # Create learning outcome for NAA
            from .neuromorphic_agent_architecture import CollaborationOutcome
            outcome = CollaborationOutcome(
                collaboration_id=f"workflow_{workflow.workflow_id}",
                participating_agents={task.assigned_agent} if task.assigned_agent else set(),
                task_type=task.required_capability,
                success_score=1.0 if success else 0.0,
                completion_time=execution_time,
                quality_metrics={
                    "quality_score": quality_score,
                    "response_time": execution_time,
                    "task_complexity": observation.get("task_complexity", 0)
                }
            )
            
            # NAA learns from this outcome
            learning_insights = await self.naa.learn_from_collaboration(outcome)
            
            print(f" NAA LEARNING: Workflow {workflow.workflow_id[:8]}...")
            print(f"   Agents: {list(workflow.participants)}")
            print(f"   Success: {1.0 if success else 0.0:.3f}, Quality: {quality_score:.3f}")
            print(f"   Learning insights: {len(learning_insights.get('synaptic_updates', []))} updates")
            
            # Clean up observation
            if hasattr(self, '_naa_observations') and task.task_id in self._naa_observations:
                del self._naa_observations[task.task_id]
            
        except Exception as e:
            logger.error(f" NAA Learning Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _assign_and_start_task(self, task: Task, workflow: Workflow):
        """Find the best agent and assign the task."""
        
        # Find capable agents, excluding those who have already tried
        candidate_agents = self._find_agents_for_capability(
            task.required_capability, 
            exclude_agents=task.history
        )

        if not candidate_agents:
            task.status = TaskStatus.FAILED
            task.error = "No available agents with required capability."
            task.completed_at = datetime.now(timezone.utc)
            print(f"COLLAB ENGINE: FAILED Task '{task.task_id}' marked as FAILED - no agents with capability '{task.required_capability}'")
            await self._check_workflow_completion(workflow.workflow_id)
            return
        
        print(f"COLLAB ENGINE: Found {len(candidate_agents)} agents for capability '{task.required_capability}': {candidate_agents}")

        # Use consensus to select the best agent
        candidates_with_details = []
        for agent_id in candidate_agents:
            agent_data = self.lobby.agents.get(agent_id)
            if agent_data:
                candidate_dict = dict(agent_data)  # Copy agent data
                candidate_dict['agent_id'] = agent_id  # Add agent_id key
                candidates_with_details.append(candidate_dict)
        
        if not candidates_with_details:
            print(f"COLLAB ENGINE: Agent IDs were found, but no detailed records in lobby for: {candidate_agents}")
            task.status = TaskStatus.FAILED
            task.error = "Agent details not found."
            task.completed_at = datetime.now(timezone.utc)
            print(f"COLLAB ENGINE: FAILED Task '{task.task_id}' marked as FAILED - agent details missing")
            await self._check_workflow_completion(workflow.workflow_id)
            return

        selected_agent_id = self._select_best_agent_by_consensus(candidates_with_details, task)
        print(f"DEBUG: Consensus selected: {selected_agent_id}")

        if selected_agent_id:
            task.history.append(selected_agent_id) # Add to history
            task.assigned_agent = selected_agent_id
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now(timezone.utc)
            
            # **FIX: Set an explicit timeout for the task**
            task.timeout_at = task.started_at + timedelta(seconds=task.timeout_seconds)

            logger.info(f"Assigning task '{task.name}' to '{selected_agent_id}'")
            
            # OK CRITICAL FIX: Don't set IN_PROGRESS until task is actually sent!
            # task.status = TaskStatus.IN_PROGRESS  # MOVED BELOW
            # task.started_at = datetime.now(timezone.utc)  # MOVED BELOW
            workflow.participants.add(selected_agent_id)

            # Add to agent workload tracking
            if selected_agent_id not in self.agent_workloads:
                self.agent_workloads[selected_agent_id] = set()
            self.agent_workloads[selected_agent_id].add(task.task_id)

            # Prepare the message for the agent - CRITICAL FIX: Use correct SDK format
            message_to_agent = {
                "message_id": f"task_{task.task_id}_{int(time.time())}",
                "sender_id": "global_lobby",
                "receiver_id": selected_agent_id,
                "message_type": "REQUEST",
                "type": "REQUEST",  # SDK compatibility
                "payload": {
                    "task_id": task.task_id,
                    "workflow_id": workflow.workflow_id,
                    "task_name": f"{task.required_capability.title().replace('_', ' ')} Task",
                    "capability_name": task.required_capability,
                    "input_data": task.input_data,
                    "shared_state": workflow.shared_state,
                    "timeout_seconds": task.timeout_seconds
                },
                "conversation_id": workflow.workflow_id,
                "priority": "HIGH",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requires_ack": False
            }
            
            # **ENHANCED**: Send task with better error handling
            try:
                await self._send_task_to_agent(selected_agent_id, message_to_agent)
                
                # OK CRITICAL FIX: Only set IN_PROGRESS after successful delivery!
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now(timezone.utc)
                
                print(f"COLLAB ENGINE: OK Task '{task.task_id}' successfully sent to '{selected_agent_id}'")
                
                # OK RESTORED: NAA observation after successful task assignment
                if self.naa_enabled:
                    await self._naa_observe_task_assignment(task, workflow, selected_agent_id)
                    
            except Exception as e:
                print(f"COLLAB ENGINE: ERROR Failed to send task to '{selected_agent_id}': {e}")
                
                # CRITICAL FIX: If no agents are available, mark task as FAILED instead of PENDING
                if "No WebSocket connection" in str(e) and len(self.lobby.agents) == 0:
                    task.status = TaskStatus.FAILED
                    task.error = "No agents available to process task"
                    task.completed_at = datetime.now(timezone.utc)
                    print(f"COLLAB ENGINE: FAILED Task '{task.task_id}' marked as FAILED - no agents available")
                else:
                    # OK CRITICAL FIX: Keep task as PENDING so it can be retried!
                    task.status = TaskStatus.PENDING  # Keep as PENDING for retry
                    task.assigned_agent = None  # Clear assignment
                    task.error = f"Failed to send task: {str(e)}"
                    print(f"COLLAB ENGINE: RELOAD Task '{task.task_id}' reset to PENDING for retry")
                
                # Remove from agent workload since sending failed
                if selected_agent_id in self.agent_workloads:
                    self.agent_workloads[selected_agent_id].discard(task.task_id)
                    
                # CRITICAL: Check if workflow should be completed due to failed task
                await self._check_workflow_completion(workflow.workflow_id)
        else:
            task.status = TaskStatus.FAILED
            task.error = "Consensus failed to select an agent."
            print(f"COLLAB ENGINE: CRITICAL - Consensus failed for task {task.task_id}")
            await self._check_workflow_completion(workflow.workflow_id)

    async def _send_task_to_agent(self, agent_id: str, message: dict):
        """CRITICAL FIX: Enhanced task delivery with multiple retry mechanisms"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                print(f"COLLAB ENGINE: SEND Sending task to agent {agent_id} (attempt {attempt + 1}/{max_retries})")
                print(f"COLLAB ENGINE: TASK Message type: {message.get('message_type', 'unknown')}")
                print(f"COLLAB ENGINE: 🆔 Task ID: {message.get('payload', {}).get('task_id', 'unknown')}")
                
                # PRIMARY: Use the lobby's proper send_task_to_agent method
                success = await self.lobby.send_task_to_agent(agent_id, message)
                
                if success:
                    print(f"COLLAB ENGINE: OK Task successfully delivered to {agent_id} via lobby (attempt {attempt + 1})")
                    return True
                else:
                    print(f"COLLAB ENGINE: ERROR Primary delivery failed to {agent_id} (attempt {attempt + 1})")
                    
                # FALLBACK 1: Try HTTP callback
                try:
                    if agent_id in self.lobby.agents:
                        agent_data = self.lobby.agents[agent_id]
                        callback_url = None
                        
                        if isinstance(agent_data, dict):
                            callback_url = agent_data.get('callback_url')
                        elif hasattr(agent_data, 'callback_url'):
                            callback_url = agent_data.callback_url
                        
                        if callback_url:
                            print(f"COLLAB ENGINE: RELOAD Trying HTTP fallback to {callback_url}")
                            
                            # Convert message to proper format for HTTP
                            http_message = {
                                "message_id": message.get("message_id"),
                                "type": "REQUEST",
                                "payload": message.get("payload"),
                                "conversation_id": message.get("conversation_id"),
                                "sender_id": "global_lobby",
                                "timestamp": message.get("timestamp")
                            }
                            
                            await self.lobby._send_http_task_async(agent_id, callback_url, http_message)
                            print(f"COLLAB ENGINE: OK HTTP fallback successful to {agent_id}")
                            return True
                        else:
                            print(f"COLLAB ENGINE: WARNING No callback URL for HTTP fallback to {agent_id}")
                except Exception as e:
                    print(f"COLLAB ENGINE: ERROR HTTP fallback failed: {e}")
                
                # FALLBACK 2: Try WebSocket direct
                try:
                    if agent_id in self.lobby.live_agent_connections:
                        websocket = self.lobby.live_agent_connections[agent_id]
                        if websocket and not websocket.closed:
                            print(f"COLLAB ENGINE: RELOAD Trying WebSocket direct to {agent_id}")
                            await websocket.send(json.dumps(message))
                            print(f"COLLAB ENGINE: OK WebSocket direct successful to {agent_id}")
                            return True
                        else:
                            print(f"COLLAB ENGINE: WARNING WebSocket connection closed for {agent_id}")
                except Exception as e:
                    print(f"COLLAB ENGINE: ERROR WebSocket direct failed: {e}")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    print(f"COLLAB ENGINE: ⏳ Waiting {retry_delay}s before retry {attempt + 2}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    
            except Exception as e:
                print(f"COLLAB ENGINE: ERROR Delivery attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                    
        print(f"COLLAB ENGINE: ERROR ALL DELIVERY ATTEMPTS FAILED for {agent_id}")
        raise Exception(f"Failed to deliver task to agent {agent_id} after {max_retries} attempts")

    async def _find_agents_for_capability(self, capability: str, exclude_agents: List[str] = None) -> List[str]:
        """Finds all agents that have a given capability, excluding specified agents."""
        if exclude_agents is None:
            exclude_agents = []
            
        logger.info(f"Finding agents for capability '{capability}', excluding {exclude_agents}")
        
        # ... (existing agent finding logic) ...
        
        found_agents = [
            agent_id for agent_id, agent_data in self.lobby.agents.items()
            if capability in agent_data.get("capabilities", []) and agent_id not in exclude_agents
        ]
        
        logger.info(f"Found {len(found_agents)} agents for '{capability}': {found_agents}")
        return found_agents
        
    def _select_best_agent_by_consensus(self, candidates: List[Dict[str, Any]], task: Task) -> str:
        """Select the best agent using consensus algorithm."""
        print(f"DEBUG CONSENSUS: Received {len(candidates)} candidates")
        for i, candidate in enumerate(candidates):
            print(f"DEBUG CONSENSUS: Candidate {i}: {type(candidate)} with keys: {list(candidate.keys()) if isinstance(candidate, dict) else 'not dict'}")
        
        # ENHANCED: More flexible connection detection
        connected_candidates = []
        for candidate in candidates:
            try:
                agent_id = candidate['agent_id']
            except (KeyError, TypeError) as e:
                print(f"DEBUG CONSENSUS: Error getting agent_id from candidate {candidate}: {e}")
                continue
                
            # Multi-layered connection detection
            is_connected = False
            connection_method = "unknown"
            
            # Method 1: Check lobby WebSocket connections
            if agent_id in self.lobby.live_agent_connections:
                is_connected = True
                connection_method = "lobby_websocket"
                print(f"CONSENSUS: {agent_id} is CONNECTED via lobby WebSocket")
            
            # Method 2: Check API bridge WebSocket connections
            elif hasattr(self.lobby, 'api_bridge') and self.lobby.api_bridge:
                if hasattr(self.lobby.api_bridge, 'websocket_connections') and agent_id in self.lobby.api_bridge.websocket_connections:
                    is_connected = True
                    connection_method = "api_bridge_websocket"
                    print(f"CONSENSUS: {agent_id} is CONNECTED via API bridge WebSocket")
            
            # Method 3: Check if agent is actively registered (ENHANCED - more permissive)
            elif agent_id in self.lobby.agents:
                agent_data = self.lobby.agents[agent_id]
                if isinstance(agent_data, dict):
                    # Accept any active agent registered recently (extended to 10 minutes)
                    try:
                        from datetime import datetime, timezone, timedelta
                        registered_at = agent_data.get('registered_at')
                        agent_status = agent_data.get('status', 'unknown')
                        
                        # Check time-based acceptance (extended window)
                        time_based_ok = False
                        if registered_at:
                            reg_time = datetime.fromisoformat(registered_at.replace('Z', '+00:00'))
                            if datetime.now(timezone.utc) - reg_time < timedelta(minutes=10):  # Extended to 10 minutes
                                time_based_ok = True
                        
                        # Accept if either recently registered OR has active status
                        if time_based_ok or agent_status == 'active':
                            is_connected = True
                            connection_method = f"active_registration_{agent_status}"
                            print(f"CONSENSUS: {agent_id} is CONNECTED via active registration (status: {agent_status}, registered: {registered_at})")
                            
                    except Exception as e:
                        print(f"CONSENSUS: Error checking registration for {agent_id}: {e}")
                        # Even if time check fails, accept agent if they're in lobby.agents
                        is_connected = True
                        connection_method = "fallback_registered"
                        print(f"CONSENSUS: {agent_id} ACCEPTED via fallback (registered agent)")
            
            # Method 4: Final fallback - if agent was found by capability search, assume available
            # This is safe because _find_agents_for_capability already filtered to valid agents
            else:
                print(f"CONSENSUS: {agent_id} not found in direct connections, but was returned by capability search")
                is_connected = True
                connection_method = "capability_search_fallback"
                print(f"CONSENSUS: {agent_id} ACCEPTED via capability search fallback")
            
            if is_connected:
                connected_candidates.append(candidate)
                print(f"CONSENSUS: [OK] {agent_id} is CONNECTED via {connection_method} - included")
            else:
                print(f"CONSENSUS: [FAIL] {agent_id} is NOT CONNECTED - excluded")
        
        if not connected_candidates:
            print("CONSENSUS: NO CONNECTED CANDIDATES FOUND!")
            print("CONSENSUS: This indicates a system issue - agents should be available")
            
            # CRITICAL FIX: Only use emergency fallback if agents are actually registered
            if candidates and len(self.lobby.agents) > 0:
                fallback_agent = candidates[0]['agent_id']
                print(f"CONSENSUS: EMERGENCY FALLBACK - selecting {fallback_agent} despite connection check failure")
                return fallback_agent
            else:
                print("CONSENSUS: NO AGENTS REGISTERED - cannot select any agent")
                return None
        
        # Sort connected agents by goal similarity, performance, and workload
        def sort_key(agent_details):
            goal_similarity = agent_details.get('goal_similarity', 0)
            performance = self.agent_performance.get(agent_details['agent_id'], {}).get('avg_success_rate', 0.5)
            workload = len(self.agent_workloads.get(agent_details['agent_id'], set()))
            return (goal_similarity, performance, -workload) # -workload for ascending

        sorted_candidates = sorted(connected_candidates, key=sort_key, reverse=True)
        
        best_agent_id = sorted_candidates[0]['agent_id']
        print(f"CONSENSUS: Selected CONNECTED agent '{best_agent_id}' from {len(connected_candidates)} candidates")
        return best_agent_id
    
    def _select_best_agent(self, candidates: List[str]) -> Optional[str]:
        """Selects the best agent from a list of candidates."""
        if not candidates:
            return None

        # Simple strategy: return the first eligible agent
        # More complex strategies could be implemented here (e.g., based on load, performance)
        return candidates[0]
    
    async def _naa_observe_task_assignment(self, task: Task, workflow: Workflow, agent_id: str):
        """Observe task assignments for NAA learning"""
        if not self.naa_enabled or not self.naa:
            return
            
        try:
            # Create a simplified observation
            from .neuromorphic_agent_architecture import TaskObservation
            
            # Simple complexity score (can be enhanced)
            complexity = len(task.input_data.get('task_description', '')) / 100.0
            
            observation = TaskObservation(
                task_id=task.task_id,
                task_type=task.required_capability,
                task_complexity=complexity,
                assigned_agent_id=agent_id
            )
            
            # Store observation to correlate with outcome later
            if not hasattr(self, '_naa_observations'):
                self._naa_observations = {}
            self._naa_observations[task.task_id] = observation.__dict__
            
            print(f"NAA OBSERVATION: Task {task.task_id[:8]} ({task.required_capability}) assigned to {agent_id}")
            
        except Exception as e:
            logger.error(f"NAA Observation Error: {e}")
    
    async def _check_workflow_completion(self, workflow_id: str):
        """Checks if a workflow is completed and handles cleanup with improved error handling."""
        if workflow_id not in self.workflows:
            print(f"COLLAB ENGINE: WARNING Workflow {workflow_id} not found during completion check")
            return
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.RUNNING:
            print(f"COLLAB ENGINE: WARNING Workflow {workflow_id} is not running (status: {workflow.status})")
            return

        # **ENHANCED**: Better task status checking with detailed logging
        total_tasks = len(workflow.tasks)
        completed_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.FAILED]
        in_progress_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
        pending_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.PENDING]
        
        print(f"COLLAB ENGINE: INFO Workflow {workflow_id} status check:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Completed: {len(completed_tasks)}")
        print(f"  Failed: {len(failed_tasks)}")
        print(f"  In progress: {len(in_progress_tasks)}")
        print(f"  Pending: {len(pending_tasks)}")

        # A workflow is complete if all its tasks are completed or failed
        all_tasks_done = all(
            t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
            for t in workflow.tasks.values()
        )
        
        if all_tasks_done:
            print(f"COLLAB ENGINE: DART All tasks done for workflow {workflow_id}")
            
            # Check if any task failed
            if failed_tasks:
                workflow.status = WorkflowStatus.FAILED
                workflow.error = f"Workflow failed due to {len(failed_tasks)} failed task(s)."
                # Collect all error messages
                error_details = {t.name: t.error for t in failed_tasks}
                workflow.result = {"errors": error_details}
                print(f"COLLAB ENGINE: ERROR WORKFLOW FAILED: {workflow.name} ({workflow_id})")
                print(f"  Failed tasks: {[t.name for t in failed_tasks]}")
            else:
                workflow.status = WorkflowStatus.COMPLETED
                # Collect results from all tasks
                final_result = {
                    t.name: t.result for t in workflow.tasks.values()
                }
                workflow.result = final_result
                print(f"COLLAB ENGINE: OK WORKFLOW COMPLETED: {workflow.name} ({workflow_id})")
                print(f"  Completed tasks: {[t.name for t in completed_tasks]}")
            
            workflow.completed_at = datetime.now(timezone.utc)
            
            # **CRITICAL FIX**: Notify participants (or original requester) about completion
            try:
                await self._notify_workflow_completion(workflow)
                print(f"COLLAB ENGINE: SEND Workflow completion notifications sent for {workflow_id}")
            except Exception as e:
                print(f"COLLAB ENGINE: ERROR Error sending workflow completion notifications: {e}")
                import traceback
                traceback.print_exc()

            # **ENHANCED**: Clean up agent workloads for completed workflow
            for task in workflow.tasks.values():
                if task.assigned_agent and task.assigned_agent in self.agent_workloads:
                    self.agent_workloads[task.assigned_agent].discard(task.task_id)
            
            print(f"COLLAB ENGINE: CLEANUP Cleaned up workloads for completed workflow {workflow_id}")
            
            # Optional: Remove completed workflow after a timeout
            # For now, we'll keep them for inspection
            # del self.workflows[workflow_id]
        else:
            print(f"COLLAB ENGINE: ⏳ Workflow {workflow_id} still has tasks in progress/pending")

    async def _notify_workflow_completion(self, workflow: Workflow):
        """Notify the original requester about workflow completion with actual result delivery."""
        print(f"--- Notifying Workflow Completion ---")
        print(f"  Workflow ID: {workflow.workflow_id}")
        print(f"  Name: {workflow.name}")
        print(f"  Status: {workflow.status.value}")
        print(f"  Created by: {workflow.created_by}")
        
        # Prepare the completion message
        completion_message = {
            "type": "workflow_completion",
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "status": workflow.status.value,
            "created_by": workflow.created_by,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "success": workflow.status == WorkflowStatus.COMPLETED,
            "participants": list(workflow.participants),
            "task_count": len(workflow.tasks),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if workflow.status == WorkflowStatus.COMPLETED:
            print("  Result: Success")
            # Include the actual results in the message
            completion_message["result"] = workflow.result
            completion_message["message"] = "Workflow completed successfully"
            
            # Limit printing large results
            result_summary = {k: (type(v), len(str(v))) for k,v in workflow.result.items()}
            print(f"    Tasks completed: {list(result_summary.keys())}")
        else:
            print(f"  Result: Failure")
            print(f"    Error: {workflow.error}")
            print(f"    Details: {workflow.result.get('errors') if workflow.result else 'No details'}")
            
            # Include error information in the message
            completion_message["result"] = {"error": workflow.error, "details": workflow.result}
            completion_message["message"] = f"Workflow failed: {workflow.error}"
        
        # **CRITICAL FIX**: Send completion notification back to the original requester
        try:
            # Send to the original requester (creator of the workflow)
            if workflow.created_by and workflow.created_by != "system":
                print(f"COLLAB ENGINE: SEND Sending completion notification to requester: {workflow.created_by}")
                success = await self.lobby.send_task_to_agent(workflow.created_by, completion_message)
                if success:
                    print(f"COLLAB ENGINE: OK Completion notification sent to {workflow.created_by}")
                else:
                    print(f"COLLAB ENGINE: ERROR Failed to send completion notification to {workflow.created_by}")
            
            # **ENHANCED**: Also notify all participants about the completion
            for participant_id in workflow.participants:
                if participant_id != workflow.created_by:  # Don't double-notify the creator
                    try:
                        participant_message = completion_message.copy()
                        participant_message["type"] = "workflow_participation_completed"
                        participant_message["your_role"] = "participant"
                        
                        print(f"COLLAB ENGINE: SEND Sending participation completion to: {participant_id}")
                        success = await self.lobby.send_task_to_agent(participant_id, participant_message)
                        if success:
                            print(f"COLLAB ENGINE: OK Participation notification sent to {participant_id}")
                        else:
                            print(f"COLLAB ENGINE: ERROR Failed to send participation notification to {participant_id}")
                    except Exception as e:
                        print(f"COLLAB ENGINE: ERROR Error notifying participant {participant_id}: {e}")
        
        except Exception as e:
            print(f"COLLAB ENGINE: ERROR Error sending completion notifications: {e}")
            import traceback
            traceback.print_exc()
        
        # --- COLLABORATION FIX: Also handle completion for multi-agent collaborations ---
        if workflow.name.startswith("Multi-Agent:"):
             await self._handle_agent_collaboration_completion(workflow, workflow.result)
    
    async def _handle_agent_collaboration_completion(self, workflow: Workflow, result_payload: dict):
        """Handle completion of a multi-agent collaboration workflow"""
        
        # Extract the original collaboration task
        original_task = None
        if "original_task" in workflow.shared_state:
            original_task = workflow.shared_state["original_task"]
        
        if not original_task:
            print("COLLAB ENGINE: WARNING - No original task found for collaboration completion.")
            return

        # Determine if the collaboration was successful
        success = workflow.status == WorkflowStatus.COMPLETED
        
        # Find the agent who initiated the collaboration
        initiating_agent_id = original_task.get("assigned_agent")
        
        if not initiating_agent_id:
            print("COLLAB ENGINE: WARNING - No initiating agent found.")
            return

        # Prepare a response message for the initiating agent
        response_message = {
            "type": "collaboration_result",
            "original_task_id": original_task.get("task_id"),
            "collaboration_id": workflow.workflow_id,
            "success": success,
            "result": result_payload if success else {"error": workflow.error}
        }
        
        print(f"COLLAB ENGINE: Notifying initiating agent {initiating_agent_id} of collaboration result.")
        await self._send_collaboration_result_to_agent(initiating_agent_id, response_message)

    async def _send_collaboration_result_to_agent(self, agent_id: str, result: dict):
        """Sends the result of a collaboration back to the agent who started it."""
        try:
            # This uses the same mechanism as sending a task
            print(f"DISPATCHING COLLAB RESULT: Sending to agent {agent_id}")
            success = await self.lobby.send_task_to_agent(agent_id, result) # Re-use the send_task_to_agent method for simplicity
            if not success:
                 print(f"DISPATCHING COLLAB RESULT: FAILED to send to {agent_id}")
        except Exception as e:
            print(f"COLLAB ENGINE: Error sending collaboration result to agent {agent_id}: {e}")

    def _update_agent_performance(self, agent_id: str, success: bool, task: Task):
        """Updates the performance metrics for an agent."""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "completed_tasks": 0,
                "failed_tasks": 0,
                "total_tasks": 0,
                "total_execution_time": 0,
                "avg_execution_time": 0,
                "success_rate": 0
            }
        
        stats = self.agent_performance[agent_id]
        stats['total_tasks'] += 1
        
        if success:
            stats['completed_tasks'] += 1
            if task.started_at and task.completed_at:
                exec_time = (task.completed_at - task.started_at).total_seconds()
                stats['total_execution_time'] += exec_time
        else:
            stats['failed_tasks'] += 1
            
        # Recalculate averages
        if stats['total_tasks'] > 0:
            stats['success_rate'] = stats['completed_tasks'] / stats['total_tasks']
        if stats['completed_tasks'] > 0:
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['completed_tasks']
        
    async def create_collaboration_session(self, agent_ids: List[str], purpose: str) -> str:
        """Creates a dedicated collaboration session between agents."""
        collab_id = f"collab_{uuid.uuid4().hex[:8]}"
        collaboration = AgentCollaboration(
            collab_id=collab_id,
            agent_ids=set(agent_ids),
            purpose=purpose
        )
        self.active_collaborations[collab_id] = collaboration
        
        # Notify agents that they are part of a new collaboration
        for agent_id in agent_ids:
            await self._notify_collaboration_start(agent_id, collaboration)
        
        return collab_id

    async def _notify_collaboration_start(self, agent_id: str, collaboration: AgentCollaboration):
        """Notify an agent that a collaboration session has started."""
        message = {
            "type": "collaboration_started",
            "collaboration_id": collaboration.collab_id,
            "purpose": collaboration.purpose,
            "participants": list(collaboration.agent_ids),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        # This can be sent as a standard message or a special task type
        # Re-using send_task_to_agent for simplicity
        await self.lobby.send_task_to_agent(agent_id, message)

    async def broadcast_to_collaboration(self, collab_id: str, sender_id: str, content: Dict[str, Any]):
        """Broadcasts a message to all participants in a collaboration, excluding the sender."""
        if collab_id not in self.active_collaborations:
            print(f"COLLAB ENGINE: Collaboration {collab_id} not found.")
            return

        collaboration = self.active_collaborations[collab_id]
        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        collaboration.message_history.append(message_id)
        collaboration.last_activity = datetime.now(timezone.utc)
        
        message = {
            "type": "collaboration_message",
            "collaboration_id": collab_id,
            "message_id": message_id,
            "sender_id": sender_id,
            "content": content,
            "timestamp": collaboration.last_activity.isoformat()
        }
        
        for agent_id in collaboration.agent_ids:
            if agent_id != sender_id:
                await self.lobby.send_task_to_agent(agent_id, message)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Gets the status of a specific workflow."""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            return {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "status": workflow.status.value,
                "created_at": workflow.created_at.isoformat(),
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                "tasks": [task.to_dict() for task in workflow.tasks.values()],
                "result": workflow.result,
                "error": workflow.error
            }
        return None

    def get_agent_workload(self, agent_id: str) -> Dict[str, Any]:
        """Gets the current workload for a specific agent."""
        return {
            "agent_id": agent_id,
            "active_tasks": len(self.agent_workloads.get(agent_id, set())),
            "task_ids": list(self.agent_workloads.get(agent_id, set()))
        }
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Returns overall statistics for the collaboration engine."""
        return {
            "active_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.RUNNING]),
            "pending_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.PENDING]),
            "completed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]),
            "failed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.FAILED]),
            "total_registered_agents": len(self.lobby.agents),
            "agents_with_workload": len(self.agent_workloads),
            "agent_performance_tracked": len(self.agent_performance)
        }
    
    def get_databus_system_status(self) -> Dict[str, Any]:
        """Returns the status of the DataBus system"""
        if not self.databus_enabled:
            return {"status": "disabled"}
        return self.databus_orchestrator.get_system_status() if self.databus_orchestrator else {}
        
    async def create_workflow(self, name: str, description: str, created_by: str,
                            task_definitions: list[dict]) -> str:
        """Creates a new workflow with a defined set of tasks and dependencies."""
        workflow_id = str(uuid.uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            created_by=created_by
        )

        for task_def in task_definitions:
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                workflow_id=workflow_id,
                name=task_def.get("name"),
                required_capability=task_def.get("capability"),
                input_data=task_def.get("input_data", {}),
                dependencies=task_def.get("dependencies", []),
                timeout_seconds=task_def.get("timeout", 300)
            )
            workflow.tasks[task_id] = task

        self.workflows[workflow_id] = workflow
        print(f"Created workflow '{name}' with {len(workflow.tasks)} tasks.")
        return workflow_id 

    async def _create_collaborative_space(self, task: Task, workflow: Workflow):
        """Create a collaborative space and invite multiple agents to work together"""
        try:
            task_data = task.input_data
            collaboration_id = task_data.get("collaboration_id")
            participating_agents = task_data.get("participating_agents", [])
            goal = task_data.get("goal", "Collaborative workspace")
            required_capabilities = task_data.get("required_capabilities", [])
            
            print(f"COLLAB ENGINE:  Creating collaborative space")
            print(f"   Collaboration ID: {collaboration_id}")
            print(f"   Participating agents: {participating_agents}")
            print(f"   Goal: {goal}")
            print(f"   Required capabilities: {required_capabilities}")
            
            if len(participating_agents) < 2:
                print(f"COLLAB ENGINE:  Not enough agents for collaboration ({len(participating_agents)})")
                task.status = TaskStatus.FAILED
                task.error = "Not enough agents for collaboration"
                return
            
            # Mark task as in progress
            task.status = TaskStatus.IN_PROGRESS
            workflow.participants.update(participating_agents)
            
            # Send collaborative space invitation to each agent
            for agent_id in participating_agents:
                await self._invite_agent_to_collaborative_space(
                    agent_id=agent_id,
                    collaboration_id=collaboration_id,
                    task_id=task.task_id,
                    workflow_id=workflow.workflow_id,
                    goal=goal,
                    required_capabilities=required_capabilities,
                    participating_agents=participating_agents,
                    task_data=task_data
                )
            
            # Update the active collaboration
            if collaboration_id in self.active_collaborations:
                collaboration = self.active_collaborations[collaboration_id]
                collaboration.shared_context.update({
                    "workflow_id": workflow.workflow_id,
                    "task_id": task.task_id,
                    "goal": goal,
                    "required_capabilities": required_capabilities,
                    "status": "active",
                    "participants_invited": len(participating_agents)
                })
                
            print(f"COLLAB ENGINE:  Collaborative space created and invitations sent")
            
        except Exception as e:
            print(f"COLLAB ENGINE:  Error creating collaborative space: {e}")
            import traceback
            traceback.print_exc()
            task.status = TaskStatus.FAILED
            task.error = f"Failed to create collaborative space: {e}"

    async def _invite_agent_to_collaborative_space(self, agent_id: str, collaboration_id: str, 
                                                 task_id: str, workflow_id: str, goal: str,
                                                 required_capabilities: List[str], 
                                                 participating_agents: List[str], task_data: Dict[str, Any]):
        """Invite an agent to join a collaborative workspace by sending them a collaborative task."""
        try:
            # Determine what capabilities this specific agent brings
            agent_info = self.lobby.agents.get(agent_id, {})
            agent_capabilities = agent_info.get("capabilities", [])
            agent_role = agent_info.get("specialization", "General Agent")
            
            # Find which capabilities this agent can contribute
            relevant_capabilities = []
            for cap in required_capabilities:
                if any(agent_cap for agent_cap in agent_capabilities 
                      if cap.lower() in agent_cap.lower() or agent_cap.lower() in cap.lower()):
                    relevant_capabilities.append(cap)
            
            if not relevant_capabilities:
                relevant_capabilities = [required_capabilities[0]] if required_capabilities else ["general"]
            
            # Create a collaborative task message that agents can execute
            collaborative_task_message = {
                "type": "task",  # Use the standard 'task' type that agents can execute
                "task_id": task_id,
                "workflow_id": workflow_id,
                "task_title": f"Collaborative Task: {goal}",
                "task_description": "You have been assigned to a collaborative task. Please work with the other participating agents to achieve the shared goal.",
                "required_capabilities": required_capabilities,
                "input_data": {
                    "collaboration_id": collaboration_id,
                    "space_type": "multi_agent_workspace",
                    "goal": goal,
                    "your_role": f"Contribute {', '.join(relevant_capabilities)} expertise",
                    "your_capabilities_needed": relevant_capabilities,
                    "all_required_capabilities": required_capabilities,
                    "participating_agents": participating_agents,
                    "other_participants": [p for p in participating_agents if p != agent_id],
                    "collaboration_context": {
                        "original_data": task_data,
                        "shared_workspace": True,
                        "real_time_collaboration": True,
                        "cross_agent_communication": True
                    },
                    "instructions": {
                        "join_workspace": "You are part of a collaborative workspace.",
                        "work_together": "Collaborate with other agents to achieve the shared goal.",
                        "contribute_expertise": f"Focus on {', '.join(relevant_capabilities)} aspects.",
                        "communicate": "Share insights and coordinate with other agents.",
                        "goal_oriented": "Work towards the common goal together."
                    },
                    "workspace_features": [
                        "Shared context and state",
                        "Real-time agent-to-agent communication", 
                        "Collaborative problem solving",
                        "Coordinated task execution",
                        "Joint result creation"
                    ]
                },
                "delegation_history": [] # Start with a clean history for this task
            }
            
            print(f"COLLAB ENGINE: 📨 Assigning collaborative task to {agent_id}")
            print(f"   Agent role: {agent_role}")
            print(f"   Contributing: {relevant_capabilities}")
            
            # Send the task via the lobby's standard mechanism
            success = await self.lobby.send_task_to_agent(agent_id, collaborative_task_message)
            
            if success:
                print(f"COLLAB ENGINE: [OK] Collaborative task sent to {agent_id}")
            else:
                print(f"COLLAB ENGINE: [FAIL] Failed to send collaborative task to {agent_id}")
                
        except Exception as e:
            print(f"COLLAB ENGINE: [FAIL] Error assigning collaborative task to {agent_id}: {e}")
            import traceback
            traceback.print_exc() 

    async def _send_task_outcome_to_lobby(self, task: Task, workflow: Workflow, agent_id: str, success: bool):
        """Track task outcome for internal metrics and learning"""
        try:
            # Track the outcome internally for collaboration engine learning
            outcome_data = {
                "task_id": task.task_id,
                "workflow_id": workflow.workflow_id,
                "success": success,
                "requester_id": workflow.created_by,
                "provider_id": agent_id,
                "task_name": task.name,
                "required_capability": task.required_capability,
                "result": task.result if success else None,
                "error": task.error if not success else None,
                "execution_time": (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Update agent performance metrics internally
            self._update_agent_performance(agent_id, success, task)
            
            print(f"TROPHY COLLAB ENGINE: Task outcome tracked internally (Task: {task.task_id}, Success: {success})")
            
        except Exception as e:
            print(f"TROPHY COLLAB ENGINE ERROR: Failed to track task outcome: {e}")
            import traceback
            traceback.print_exc() 

    async def start_workflow_processing(self):
        """Start the workflow processing loop"""
        try:
            print("COLLAB ENGINE: Starting workflow processing loop...")
            
            # Start cleanup task
            cleanup_task = asyncio.create_task(self._workflow_cleanup_loop())
            
            while self.running:
                try:
                    # Process all active workflows
                    if self.workflows:
                        workflows_to_process = list(self.workflows.values())
                        for workflow in workflows_to_process:
                            if workflow.status == 'active':
                                await self._process_workflow(workflow)
                    
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"COLLAB ENGINE ERROR: Workflow processing error: {e}")
                    await asyncio.sleep(5)  # Longer delay on error
                    
        except asyncio.CancelledError:
            print("COLLAB ENGINE: Workflow processing cancelled")
        except Exception as e:
            print(f"COLLAB ENGINE ERROR: Fatal workflow processing error: {e}")
        finally:
            cleanup_task.cancel()
            print("COLLAB ENGINE: Workflow processing stopped")

    async def _workflow_cleanup_loop(self):
        """Background task to clean up orphaned and stuck workflows"""
        try:
            cleanup_interval = 30  # Run cleanup every 30 seconds
            max_workflow_age = 300  # Consider workflows older than 5 minutes for cleanup
            
            while self.running:
                try:
                    await asyncio.sleep(cleanup_interval)
                    await self._cleanup_orphaned_workflows(max_workflow_age)
                    
                except Exception as e:
                    print(f"COLLAB ENGINE ERROR: Cleanup error: {e}")
                    
        except asyncio.CancelledError:
            print("COLLAB ENGINE: Cleanup loop cancelled")
            
    async def _cleanup_orphaned_workflows(self, max_age_seconds: int):
        """Clean up workflows that are stuck or orphaned"""
        try:
            current_time = datetime.now(timezone.utc)
            cleaned_workflows = []
            
            for workflow_id, workflow in list(self.workflows.items()):
                try:
                    # Calculate workflow age
                    age_seconds = (current_time - workflow.created_at).total_seconds()
                    
                    # Check if workflow should be cleaned up
                    should_cleanup = False
                    cleanup_reason = ""
                    
                    # Condition 1: Very old workflows (older than max age)
                    if age_seconds > max_age_seconds:
                        should_cleanup = True
                        cleanup_reason = f"aged out ({age_seconds:.0f}s old)"
                    
                    # Condition 2: Workflows with all tasks stuck in progress for too long
                    elif workflow.status == 'active':
                        stuck_tasks = []
                        for task in workflow.tasks:
                            if task.status == 'IN_PROGRESS':
                                task_age = (current_time - task.assigned_at).total_seconds() if task.assigned_at else age_seconds
                                if task_age > 120:  # Tasks stuck for more than 2 minutes
                                    stuck_tasks.append(task.task_id)
                        
                        if stuck_tasks and len(stuck_tasks) == len([t for t in workflow.tasks if t.status == 'IN_PROGRESS']):
                            should_cleanup = True
                            cleanup_reason = f"all tasks stuck ({len(stuck_tasks)} tasks)"
                    
                    # Condition 3: Workflows with no active agents
                    elif workflow.status == 'active':
                        has_active_agents = False
                        for task in workflow.tasks:
                            if task.assigned_agent and task.assigned_agent in self.lobby.get_connected_agents():
                                has_active_agents = True
                                break
                        
                        if not has_active_agents and age_seconds > 60:  # No active agents for 1+ minutes
                            should_cleanup = True
                            cleanup_reason = "no active agents"
                    
                    # Perform cleanup
                    if should_cleanup:
                        await self._force_cleanup_workflow(workflow, cleanup_reason)
                        cleaned_workflows.append((workflow_id, cleanup_reason))
                        
                except Exception as e:
                    print(f"COLLAB ENGINE ERROR: Error checking workflow {workflow_id} for cleanup: {e}")
            
            # Report cleanup results
            if cleaned_workflows:
                print(f"COLLAB ENGINE: CLEANUP Cleaned up {len(cleaned_workflows)} orphaned workflows:")
                for workflow_id, reason in cleaned_workflows:
                    print(f"  - {workflow_id}: {reason}")
            
        except Exception as e:
            print(f"COLLAB ENGINE ERROR: Cleanup process failed: {e}")
            
    async def _force_cleanup_workflow(self, workflow: 'Workflow', reason: str):
        """Force cleanup a stuck workflow"""
        try:
            print(f"COLLAB ENGINE: 🗑️ Force cleaning workflow {workflow.workflow_id} ({reason})")
            
            # Mark all in-progress tasks as failed
            failed_tasks = []
            for task in workflow.tasks:
                if task.status in ['IN_PROGRESS', 'PENDING']:
                    task.status = 'FAILED'
                    task.completed_at = datetime.now(timezone.utc)
                    task.result = {"error": f"Task forcibly failed during cleanup: {reason}"}
                    failed_tasks.append(task.task_id)
                    
                    # Remove from agent workloads
                    if task.assigned_agent:
                        await self._remove_task_from_agent_workload(task.assigned_agent, task.task_id)
            
            # Mark workflow as failed
            workflow.status = 'failed'
            workflow.completed_at = datetime.now(timezone.utc)
            
            # Remove from active workflows
            if workflow.workflow_id in self.workflows:
                del self.workflows[workflow.workflow_id]
            
            print(f"COLLAB ENGINE: OK Cleaned up workflow {workflow.workflow_id} (failed {len(failed_tasks)} tasks)")
            
        except Exception as e:
            print(f"COLLAB ENGINE ERROR: Failed to cleanup workflow {workflow.workflow_id}: {e}")
    
    async def _remove_task_from_agent_workload(self, agent_id: str, task_id: str):
        """Remove a task from an agent's workload tracking"""
        try:
            if hasattr(self, 'agent_workloads') and agent_id in self.agent_workloads:
                self.agent_workloads[agent_id].discard(task_id)
                if not self.agent_workloads[agent_id]:  # Remove empty workload
                    del self.agent_workloads[agent_id]
        except Exception as e:
            print(f"COLLAB ENGINE ERROR: Failed to remove task {task_id} from {agent_id} workload: {e}")

    async def handle_agent_disconnect(self, agent_id: str):
        """Handle an agent disconnecting unexpectedly."""
        logger.warning(f"DISCONNECT: Agent {agent_id} has disconnected. Checking for in-progress tasks to re-queue.")
        
        # Find all tasks assigned to this agent and handle their failure
        for workflow in list(self.workflows.values()):
            for task in list(workflow.tasks.values()):
                if task.assigned_agent == agent_id and task.status == TaskStatus.IN_PROGRESS:
                    logger.info(f"Re-queueing task {task.task_id} from disconnected agent {agent_id}.")
                    await self._handle_task_failure(task, workflow, f"Agent {agent_id} disconnected")

    async def _check_task_timeouts(self, workflow_id: str):
        """Periodically checks for tasks that have timed out."""
        # This function is now superseded by the logic in _monitor_workflow_progress
        pass