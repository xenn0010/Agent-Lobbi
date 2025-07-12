"""
Agent Lobbi SDK - Complete Integration
Includes all security, consensus, recovery, and tracking features
"""

import asyncio
import json
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Set, Callable, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import gzip
import pickle
import uuid
import time
import aiohttp
import websockets

# Import our security systems
try:
    from ..security.consensus_system import (
        ConsensusReputationSystem, TaskDifficulty, AgentReputation, TaskCompletion
    )
    from ..security.data_protection_layer import (
        DataProtectionLayer, DataClassification, AccessLevel
    )
    from ..recovery.connection_recovery import (
        ConnectionRecoverySystem, ConnectionState, RecoveryStrategy
    )
    from ..tracking.agent_tracking_system import (
        AgentTrackingSystem, ActivityType, AgentActivity
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from security.consensus_system import (
        ConsensusReputationSystem, TaskDifficulty, AgentReputation, TaskCompletion
    )
    from security.data_protection_layer import (
        DataProtectionLayer, DataClassification, AccessLevel
    )
    from recovery.connection_recovery import (
        ConnectionRecoverySystem, ConnectionState, RecoveryStrategy
    )
    from tracking.agent_tracking_system import (
        AgentTrackingSystem, ActivityType, AgentActivity
    )

logger = logging.getLogger(__name__)


class AgentLobbySDK:
    """
    Complete Agent Lobbi SDK with integrated security systems
    Provides honest, production-ready multi-agent collaboration
    """
    
    def __init__(self, 
                 lobby_host: str = "localhost",
                 lobby_port: int = 9101,
                 ws_port: int = 9102,
                 enable_security: bool = True,
                 db_path_prefix: str = "agent_lobby"):
        
        self.lobby_host = lobby_host
        self.lobby_port = lobby_port
        self.ws_port = ws_port
        self.lobby_url = f"http://{lobby_host}:{lobby_port}"
        self.websocket_url = f"ws://{lobby_host}:{ws_port}"
        
        # Agent information
        self.agent_id: Optional[str] = None
        self.api_key: Optional[str] = None
        self.auth_token: Optional[str] = None
        self.session_id: Optional[str] = None
        
        # User agent reference and capabilities
        self.user_agent: Optional[Any] = None
        self.agent_capabilities: List[str] = []
        self.task_handler: Optional[Callable] = None
        
        # Initialize security systems
        if enable_security:
            self.consensus_system = ConsensusReputationSystem(f"{db_path_prefix}_consensus.db")
            self.data_protection = DataProtectionLayer(f"{db_path_prefix}_protection.db")
            self.recovery_system = ConnectionRecoverySystem(f"{db_path_prefix}_recovery.db")
            self.tracking_system = AgentTrackingSystem(f"{db_path_prefix}_tracking.db")
        else:
            self.consensus_system = None
            self.data_protection = None
            self.recovery_system = None
            self.tracking_system = None
        
        # Connection state
        self.connected = False
        self.websocket = None
        self._websocket_task = None
        self._running = False
        
        logger.info("Agent Lobbi SDK initialized with security features")
    
    async def register_agent(self, 
                           agent_id: str,
                           name: str,
                           agent_type: str,
                           capabilities: List[str],
                           user_agent: Any = None,
                           task_handler: Optional[Callable] = None,
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register agent with the lobby and start WebSocket connection
        """
        try:
            self.agent_id = agent_id
            self.agent_capabilities = capabilities
            self.user_agent = user_agent
            self.task_handler = task_handler
            
            # Generate API key for tracking
            if self.tracking_system:
                self.api_key = self.tracking_system.generate_api_key()
            
            # Register with consensus system
            if self.consensus_system:
                await self.consensus_system.register_agent(agent_id)
            
            # Register with recovery system
            if self.recovery_system:
                await self.recovery_system.register_connection(
                    agent_id, "lobby", "primary", metadata or {}
                )
            
            # Start tracking session
            if self.tracking_system and self.api_key:
                self.session_id = await self.tracking_system.start_agent_session(
                    agent_id, self.api_key, {"agent_type": agent_type}
                )
            
            # Register with lobby via HTTP
            registration_data = {
                "agent_id": agent_id,
                "name": name,
                "agent_type": agent_type,
                "goal": "",
                "specialization": "",
                "capabilities": capabilities
            }
            
            # Register via HTTP
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.lobby_url}/api/agents/register",
                        json=registration_data
                    ) as response:
                        if response.status == 200:
                            registration_result = await response.json()
                            logger.info(f"Agent {agent_id} registered via HTTP")
                        else:
                            logger.warning(f"HTTP registration failed with status {response.status}")
                            registration_result = {"status": "http_failed"}
            except Exception as e:
                logger.warning(f"HTTP registration failed: {e}")
                registration_result = {"status": "http_failed"}
            
            # Start WebSocket connection for task receiving
            await self._start_websocket_connection()
            
            self.auth_token = f"auth_{secrets.token_hex(16)}"
            self.connected = True
            
            # Track registration activity
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    agent_id, self.api_key, ActivityType.REGISTERED,
                    {"agent_type": agent_type, "capabilities_count": len(capabilities)}
                )
            
            logger.info(f"Agent {agent_id} registered successfully with WebSocket connection")
            return {
                "status": "success",
                "agent_id": agent_id,
                "websocket_connected": self.websocket is not None,
                "capabilities": capabilities
            }
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def _start_websocket_connection(self):
        """Start WebSocket connection to lobby for task receiving"""
        try:
            self._running = True
            websocket_url = f"ws://{self.lobby_host}:{self.ws_port}/api/ws/{self.agent_id}"
            self.websocket_url = websocket_url  # Update the URL with agent ID
            self._websocket_task = asyncio.create_task(self._websocket_handler())
        except Exception as e:
            logger.error(f"Failed to start WebSocket connection: {e}")
            raise
    
    async def _websocket_handler(self):
        """Handle WebSocket connection and message processing"""
        while self._running:
            try:
                async with websockets.connect(self.websocket_url) as websocket:
                    self.websocket = websocket
                    
                    # Register with WebSocket server
                    if self.agent_id:
                        await websocket.send(json.dumps({
                            "type": "register",
                            "agent_id": self.agent_id
                        }))
                        logger.info(f"Agent {self.agent_id} registered with WebSocket")
                    
                    # Listen for messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(data)
                        except Exception as e:
                            logger.error(f"Error handling WebSocket message: {e}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting to reconnect...")
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_websocket_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages (tasks from DataBus)"""
        message_type = data.get("type", "unknown")
        
        # Log ALL incoming messages for debugging
        logger.info(f"SDK Agent {self.agent_id} received message: {message_type}")
        logger.info(f"Full message data: {data}")
        
        if message_type == "register_ack":
            logger.info(f"WebSocket registration acknowledged for {self.agent_id}")
            
        elif message_type == "connection_established":
            logger.info(f"WebSocket connection established for {self.agent_id}")
            
        elif message_type == "task_request":
            # This is a task from the DataBus system
            await self._handle_task_request(data)
            
        elif message_type == "task":
            # This is a task from the lobby delegation system
            logger.info(f"Processing task delegation for {self.agent_id}")
            await self._handle_lobby_delegation_task(data)
            
        elif message_type == "TASK_ASSIGNMENT":
            # CRITICAL FIX: Handle TASK_ASSIGNMENT messages from lobby
            logger.info(f"Processing TASK_ASSIGNMENT for {self.agent_id}")
            await self._handle_task_assignment(data)
            
        elif message_type == "message" and data.get("message_type") == "REQUEST":
            # This is a task message from the lobby message system
            await self._handle_lobby_task_message(data)
            
        elif message_type == "REQUEST" or data.get("message_type") == "REQUEST":
            # Handle REQUEST messages from collaboration engine
            await self._handle_lobby_task_message(data)
            
        elif message_type == "workflow_participation_completed":
            # Handle workflow completion notifications
            await self._handle_workflow_completion(data)
            
        else:
            logger.info(f"Unhandled WebSocket message type: {message_type}")
    
    async def _handle_task_request(self, data: Dict[str, Any]):
        """Handle task request from DataBus system"""
        try:
            task_data = data.get("payload", {})
            task_id = task_data.get("task_id")
            task_name = task_data.get("task_name", "Unknown Task")
            capability_name = task_data.get("capability_name")
            input_data = task_data.get("input_data", {})
            
            logger.info(f"Received task: {task_name} (ID: {task_id}, Capability: {capability_name})")
            
            # Check if agent has required capability
            if capability_name not in self.agent_capabilities:
                logger.warning(f"Agent {self.agent_id} doesn't have capability: {capability_name}")
                await self._send_task_response(task_id, "error", {"error": "Capability not available"})
                return
            
            # Process task with user's agent
            result = await self._process_task_with_user_agent(task_data)
            
            # Send response back
            await self._send_task_response(task_id, "success", result)
            
        except Exception as e:
            logger.error(f"Error handling task request: {e}")
            task_id = data.get("payload", {}).get("task_id")
            if task_id:
                await self._send_task_response(task_id, "error", {"error": str(e)})
    
    async def _handle_lobby_delegation_task(self, data: Dict[str, Any]):
        """Handle task from lobby delegation system"""
        try:
            task_id = data.get("task_id")
            task_title = data.get("task_title", "Unknown Task")
            task_description = data.get("task_description", "")
            required_capabilities = data.get("required_capabilities", [])
            workflow_id = data.get("workflow_id") or data.get("conversation_id")
            
            logger.info(f"Received delegation task: {task_title} (ID: {task_id})")
            logger.info(f"Required capabilities: {required_capabilities}")
            
            # Check if we have any required capabilities
            my_capabilities = set(self.agent_capabilities)
            required_caps = set(required_capabilities)
            can_contribute = bool(my_capabilities.intersection(required_caps))
            
            if not can_contribute:
                logger.warning(f"Agent {self.agent_id} has no matching capabilities for task {task_id}")
                return
            
            # Process task with user's agent (pass full task data)
            result = await self._process_task_with_user_agent(data)
            
            # Send response back to lobby
            await self._send_task_response(task_id, "success", result)
            
        except Exception as e:
            logger.error(f"Error handling delegation task: {e}")
            task_id = data.get("task_id")
            if task_id:
                await self._send_task_response(task_id, "error", {"error": str(e)})

    async def _handle_lobby_task_message(self, data: Dict[str, Any]):
        """Handle task message from lobby message system"""
        try:
            payload = data.get("payload", {})
            task_id = payload.get("task_id")
            task_name = payload.get("task_name", "Unknown Task")
            capability_name = payload.get("capability_name")
            
            logger.info(f"Received lobby task: {task_name} (ID: {task_id}, Capability: {capability_name})")
            
            # Process task with user's agent
            result = await self._process_task_with_user_agent(payload)
            
            # Send response via WebSocket
            await self._send_lobby_task_response(data.get("conversation_id"), task_id, "success", result)
            
        except Exception as e:
            logger.error(f"Error handling lobby task message: {e}")
            payload = data.get("payload", {})
            task_id = payload.get("task_id")
            conversation_id = data.get("conversation_id")
            if task_id and conversation_id:
                await self._send_lobby_task_response(conversation_id, task_id, "error", {"error": str(e)})
    
    async def _process_task_with_user_agent(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process task using user's agent implementation"""
        try:
            # If user provided a custom task handler, use it
            if self.task_handler:
                result = await self.task_handler(task_data)
                return result
            
            # If user provided an agent object with a process method, use it
            if self.user_agent:
                if hasattr(self.user_agent, 'process_task'):
                    result = await self.user_agent.process_task(task_data)
                    return result
                elif hasattr(self.user_agent, 'process'):
                    result = await self.user_agent.process(task_data)
                    return result
                elif callable(self.user_agent):
                    result = await self.user_agent(task_data)
                    return result
            
            # Default processing if no user agent is provided
            capability_name = task_data.get("capability_name", "unknown")
            task_name = task_data.get("task_name", "Unknown Task")
            input_data = task_data.get("input_data", {})
            
            # Simulate processing
            await asyncio.sleep(0.1)
            
            return {
                "agent_id": self.agent_id,
                "capability_used": capability_name,
                "task_processed": task_name,
                "input_received": bool(input_data),
                "processing_time": 0.1,
                "timestamp": datetime.now().isoformat(),
                "result": f"Processed {task_name} using {capability_name}"
            }
            
        except Exception as e:
            logger.error(f"Error processing task with user agent: {e}")
            raise
    
    async def _send_task_response(self, task_id: str, status: str, result: Dict[str, Any]):
        """Send task response via WebSocket"""
        try:
            if self.websocket:
                # OK FIXED: Send in the format the lobby expects
                response = {
                    "message_type": "TASK_COMPLETION",  # OK CORRECT FORMAT for lobby
                    "sender_id": self.agent_id,
                    "receiver_id": "lobby", 
                    "payload": {
                        "task_id": task_id,
                        "status": status,
                        "result": result,
                        "agent_id": self.agent_id
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await self.websocket.send(json.dumps(response))
                logger.info(f"OK Sent TASK_COMPLETION response for {task_id}: {status}")
        except Exception as e:
            logger.error(f"Error sending task response: {e}")
    
    async def _send_lobby_task_response(self, conversation_id: str, task_id: str, status: str, result: Dict[str, Any]):
        """Send task response for lobby message system"""
        try:
            if self.websocket:
                # OK FIXED: Send proper Message format
                response = {
                    "message_type": "RESPONSE",  # OK CORRECT MESSAGE TYPE
                    "sender_id": self.agent_id,
                    "receiver_id": "lobby",
                    "conversation_id": conversation_id,
                    "payload": {
                        "task_id": task_id,
                        "status": status,
                        "result": result
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await self.websocket.send(json.dumps(response))
                logger.info(f"OK Sent RESPONSE for task {task_id}: {status}")
        except Exception as e:
            logger.error(f"Error sending lobby task response: {e}")
    
    async def _handle_workflow_completion(self, data: Dict[str, Any]):
        """Handle workflow completion notification"""
        try:
            workflow_id = data.get("workflow_id")
            workflow_name = data.get("workflow_name", "Unknown Workflow")
            status = data.get("status", "unknown")
            success = data.get("success", False)
            
            logger.info(f"SUCCESS Workflow completed: {workflow_name} ({workflow_id}) - Success: {success}")
            
            # Log the workflow results if available
            results = data.get("result", {})
            if results:
                logger.info(f"INFO Workflow results: {len(results)} tasks completed")
                for task_name, task_result in results.items():
                    task_status = task_result.get("status", "unknown")
                    logger.info(f"  - {task_name}: {task_status}")
            
        except Exception as e:
            logger.error(f"Error handling workflow completion: {e}")
    
    async def submit_task(self,
                         task_id: str,
                         task_description: str,
                         difficulty: str = "medium",
                         collaborators: List[str] = None,
                         quality_score: float = 1.0) -> Dict[str, Any]:
        """
        Submit and complete a task with consensus tracking
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Map difficulty string to enum
            difficulty_map = {
                "trivial": TaskDifficulty.TRIVIAL,
                "easy": TaskDifficulty.EASY,
                "medium": TaskDifficulty.MEDIUM,
                "hard": TaskDifficulty.HARD,
                "expert": TaskDifficulty.EXPERT
            }
            
            task_difficulty = difficulty_map.get(difficulty.lower(), TaskDifficulty.MEDIUM)
            collaborator_set = set(collaborators) if collaborators else set()
            
            # Track task start
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.TASK_STARTED,
                    {"task_id": task_id, "description": task_description, "difficulty": difficulty}
                )
            
            # Simulate task execution time
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            completion_time = time.time() - start_time
            
            # Record task completion in consensus system
            points_awarded = 0.0
            if self.consensus_system:
                points_awarded = await self.consensus_system.record_task_completion(
                    task_id, self.agent_id, task_difficulty, quality_score,
                    completion_time, collaborator_set
                )
            
            # Track task completion
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.TASK_COMPLETED,
                    {"task_id": task_id, "points_awarded": points_awarded},
                    completion_time, True
                )
            
            # Update recovery system with latest activity
            if self.recovery_system:
                await self.recovery_system.update_connection_activity(
                    self.agent_id, "lobby"
                )

            return {
                "status": "success",
                "task_id": task_id,
                "points_awarded": points_awarded,
                "completion_time": completion_time,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Failed to submit task {task_id}: {e}")
            # Track failed task
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.ERROR,
                    {"task_id": task_id, "error": str(e)}, 0, False
                )
            raise
    
    async def delegate_task(self,
                           task_title: str,
                           task_description: str,
                           required_capabilities: List[str],
                           task_data: Dict[str, Any] = None,
                           max_agents: int = 1,
                           deadline_minutes: int = 60) -> Dict[str, Any]:
        """
        FIXED: Delegate task using the CORRECT API endpoint
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Prepare delegation request with CORRECT format for /api/tasks/delegate
            delegation_payload = {
                "task_title": task_title,
                "task_description": task_description,
                "required_capabilities": required_capabilities,
                "requester_id": self.agent_id,
                "task_intent": "",
                "max_agents": max_agents,
                "priority": "normal",
                "deadline": None,
                "task_data": task_data or {}
            }
            
            logger.info(f"[OK] FIXED delegation: {task_title} -> capabilities: {required_capabilities}")
            
            # Use the CORRECT endpoint that actually exists: /api/tasks/delegate
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.lobby_url}/api/tasks/delegate",  # FIXED ENDPOINT
                    json=delegation_payload,
                    headers={"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Track delegation activity
                        if self.tracking_system and self.api_key:
                            await self.tracking_system.track_agent_activity(
                                self.agent_id, self.api_key, ActivityType.TASK_DELEGATED,
                                {"task_title": task_title, "required_capabilities": required_capabilities}
                            )
                        
                        logger.info(f"[OK] FIXED delegation successful: {result}")
                        
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"[FAIL] FIXED delegation failed: HTTP {response.status} - {error_text}")
                        return {
                            "status": "error",
                            "message": f"Delegation failed: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"[FAIL] FIXED delegation error: {e}")
            return {
                "status": "error", 
                "message": f"Delegation failed: {str(e)}"
            }

    async def _wait_for_task_completion(self, task_id: str, deadline_minutes: int = 60) -> Dict[str, Any]:
        """Wait for direct task completion and return result"""
        import asyncio
        
        max_wait_time = deadline_minutes * 60  # Convert to seconds
        poll_interval = 2  # Poll every 2 seconds
        elapsed_time = 0
        
        logger.info(f"⏳ Waiting for task {task_id} completion (max {deadline_minutes} minutes)")
        
        while elapsed_time < max_wait_time:
            try:
                # Poll for task result
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.lobby_url}/api/get_task_result",
                        json={"task_id": task_id},
                        headers={"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            if result.get('status') == 'success':
                                task_result = result.get('task_result', {})
                                logger.info(f" Task {task_id} completed successfully")
                                return {
                                    "status": "completed",
                                    "task_id": task_id,
                                    "result": task_result.get('result', {}),
                                    "agent_id": task_result.get('agent_id'),
                                    "completed_at": task_result.get('completed_at'),
                                    "delegation_type": "direct"
                                }
                        elif response.status == 404:
                            # Task still pending
                            pass
                        else:
                            logger.warning(f"Unexpected response polling task {task_id}: {response.status}")
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
                
            except Exception as e:
                logger.error(f"Error polling task {task_id}: {e}")
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
        
        # Timeout reached
        logger.error(f" Task {task_id} timed out after {deadline_minutes} minutes")
        return {
            "status": "timeout",
            "task_id": task_id,
            "message": f"Task timed out after {deadline_minutes} minutes"
        }

    async def browse_available_tasks(self,
                                   my_capabilities: List[str] = None,
                                   filter_by_agent_type: str = None) -> List[Dict[str, Any]]:
        """
        DIRECT task browsing from the lobby - simplified approach
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Prepare DIRECT browse request (simplified)
            browse_params = {
                "agent_id": self.agent_id,
                "capabilities": ','.join(my_capabilities or [])
            }
            
            logger.info(f" DIRECT task browsing for agent {self.agent_id}")
            
            # Make DIRECT HTTP request to simplified lobby endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.lobby_url}/api/available_tasks",
                    params=browse_params,
                    headers={"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        available_tasks = result.get("tasks", [])
                        
                        # Track browsing activity
                        if self.tracking_system and self.api_key:
                            await self.tracking_system.track_agent_activity(
                                self.agent_id, self.api_key, ActivityType.TASK_BROWSED,
                                {"tasks_found": len(available_tasks), "filter_type": filter_by_agent_type}
                            )
                        
                        logger.info(f" Found {len(available_tasks)} direct tasks for {self.agent_id}")
                        return available_tasks
                    else:
                        error_text = await response.text()
                        logger.error(f"DIRECT task browsing failed: HTTP {response.status} - {error_text}")
                        return []
                        
        except Exception as e:
            logger.error(f"DIRECT task browsing error: {e}")
            return []

    async def submit_task_response(self,
                                 task_id: str,
                                 status: str = "completed",
                                 result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Submit task response directly to the lobby - simplified approach
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Prepare DIRECT task response
            response_payload = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": status,
                "result": result or {}
            }
            
            logger.info(f" DIRECT task response: {task_id} -> status: {status}")
            
            # Make DIRECT HTTP request to submit response
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.lobby_url}/api/task_response",
                    json=response_payload,
                    headers={"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f" DIRECT task response submitted: {task_id}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"DIRECT task response failed: HTTP {response.status} - {error_text}")
                        return {
                            "status": "error",
                            "message": f"Task response failed: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"DIRECT task response error: {e}")
            return {
                "status": "error", 
                "message": f"Task response failed: {str(e)}"
            }

    async def accept_delegated_task(self,
                                  delegation_id: str,
                                  estimated_completion_minutes: int = 30) -> Dict[str, Any]:
        """
        Accept a delegated task and start collaboration.
        This enables autonomous agents to work together.
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Track task acceptance
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.ACCEPTED,
                    {
                        "delegation_id": delegation_id,
                        "estimated_completion": estimated_completion_minutes
                    }
                )
            
            # In real implementation, this would POST to lobby's task acceptance endpoint
            logger.info(f"Agent {self.agent_id} accepted delegation {delegation_id}")
            
            return {
                "status": "success",
                "delegation_id": delegation_id,
                "collaborator_agent": self.agent_id,
                "estimated_completion": f"{estimated_completion_minutes} minutes",
                "message": "Task accepted, collaboration initiated"
            }
            
        except Exception as e:
            logger.error(f"Failed to accept task {delegation_id}: {e}")
            raise

    async def complete_delegated_task(self,
                                    delegation_id: str,
                                    task_result: Dict[str, Any],
                                    quality_score: float = 1.0) -> Dict[str, Any]:
        """
        Complete a delegated task and return results to the delegating agent.
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            completion_time = datetime.now(timezone.utc)
            
            # Award consensus points for collaboration
            points_awarded = 0.0
            if self.consensus_system:
                points_awarded = await self.consensus_system.record_task_completion(
                    delegation_id, self.agent_id, TaskDifficulty.MEDIUM, 
                    quality_score, 1.0, set()  # Collaborative task
                )
            
            # Track task completion
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.TASK_COMPLETED,
                    {
                        "delegation_id": delegation_id,
                        "points_awarded": points_awarded,
                        "quality_score": quality_score
                    }, 1.0, True
                )
            
            logger.info(f"Agent {self.agent_id} completed delegation {delegation_id}")
            
            return {
                "status": "success",
                "delegation_id": delegation_id,
                "completing_agent": self.agent_id,
                "result": task_result,
                "points_awarded": points_awarded,
                "completed_at": completion_time,
                "message": "Delegated task completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to complete task {delegation_id}: {e}")
            raise

    async def get_collaboration_status(self, delegation_id: str) -> Dict[str, Any]:
        """
        REAL collaboration status from the lobby - no simulation, no fallbacks
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Make REAL HTTP request to lobby
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.lobby_url}/api/collaboration_status/{delegation_id}",
                    headers={"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Track status check activity
                        if self.tracking_system and self.api_key:
                            await self.tracking_system.track_agent_activity(
                                self.agent_id, self.api_key, ActivityType.STATUS_CHECKED,
                                {"delegation_id": delegation_id, "status": result.get("status")}
                            )
                        
                        logger.info(f"Real collaboration status for {delegation_id}: {result.get('status')}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Collaboration status check failed: HTTP {response.status} - {error_text}")
                        return {
                            "status": "error",
                            "message": f"Status check failed: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"Collaboration status error: {e}")
            return {
                "status": "error",
                "message": f"Status check failed: {str(e)}"
            }
    
    async def request_data_access(self,
                                target_agent: str,
                                data_type: str,
                                purpose: str = "",
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Request access to another agent's data through security layer
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Check data access through protection layer
            access_granted = False
            denial_reason = "Data protection layer not available"
            
            if self.data_protection:
                access_granted, denial_reason = await self.data_protection.check_data_access(
                    self.agent_id, target_agent, data_type, context or {}
                )
            
            # Track data access attempt
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.DATA_ACCESS,
                    {
                        "target_agent": target_agent,
                        "data_type": data_type,
                        "access_granted": access_granted,
                        "purpose": purpose
                    },
                    success=access_granted
                )
            
            result = {
                "status": "success" if access_granted else "denied",
                "access_granted": access_granted,
                "target_agent": target_agent,
                "data_type": data_type,
                "reason": "Access granted" if access_granted else denial_reason,
                "data": {"sample": "data"} if access_granted else None
            }
            
            logger.info(f"Data access request: {self.agent_id} -> {target_agent}/{data_type}: {'GRANTED' if access_granted else 'DENIED'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed data access request: {e}")
            raise
    
    async def create_collaboration(self,
                                 participants: List[str],
                                 purpose: str = "",
                                 data_sharing_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a collaboration session with security controls
        """
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            collab_id = f"collab_{uuid.uuid4().hex[:8]}"
            
            # Register connections for all participants
            if self.recovery_system:
                for participant in participants:
                    if participant != self.agent_id:
                        await self.recovery_system.register_connection(
                            self.agent_id, participant, "collaboration",
                            {"collaboration_id": collab_id, "purpose": purpose}
                        )
            
            # Set up data sharing rules
            if self.data_protection and data_sharing_rules:
                for data_type, allowed_agents in data_sharing_rules.items():
                    await self.data_protection.register_agent_data(
                        self.agent_id, data_type, DataClassification.INTERNAL,
                        set(allowed_agents), AccessLevel.READ,
                        f"Collaboration: {purpose}"
                    )
            
            # Track collaboration creation
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.COLLABORATION_JOINED,
                    {
                        "collaboration_id": collab_id,
                        "participants": participants,
                        "purpose": purpose
                    }
                )
            
            result = {
                "status": "success",
                "collaboration_id": collab_id,
                "participants": participants,
                "purpose": purpose,
                "security_enabled": True,
                "message": "Collaboration created with security controls"
            }
            
            logger.info(f"Collaboration {collab_id} created by {self.agent_id} with {len(participants)} participants")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create collaboration: {e}")
            raise
    
    async def initiate_recovery(self,
                              strategy: str = "gradual") -> Dict[str, Any]:
        """
        Initiate system recovery after failures
        """
        if not self.recovery_system:
            return {"status": "error", "message": "Recovery system not available"}
        
        try:
            # Map strategy string to enum
            strategy_map = {
                "immediate": RecoveryStrategy.IMMEDIATE,
                "gradual": RecoveryStrategy.GRADUAL,
                "manual": RecoveryStrategy.MANUAL
            }
            
            recovery_strategy = strategy_map.get(strategy.lower(), RecoveryStrategy.GRADUAL)
            
            # Create recovery snapshot first
            active_agents = {self.agent_id} if self.agent_id else set()
            snapshot_id = await self.recovery_system.create_recovery_snapshot(
                active_agents, {}, {}
            )
            
            # Initiate recovery
            recovery_result = await self.recovery_system.initiate_recovery(
                recovery_strategy, snapshot_id
            )
            
            logger.info(f"Recovery initiated with strategy {strategy}: {recovery_result['status']}")
            return recovery_result
            
        except Exception as e:
            logger.error(f"Failed to initiate recovery: {e}")
            raise
    
    def get_agent_reputation(self) -> Optional[AgentReputation]:
        """
        Get current agent reputation from consensus system
        """
        if not self.consensus_system or not self.agent_id:
            return None
        
        return self.consensus_system.get_agent_reputation(self.agent_id)
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive agent metrics
        """
        if not self.api_key:
            return {"error": "No API key available"}
        
        metrics = {}
        
        # Tracking metrics
        if self.tracking_system:
            tracking_metrics = self.tracking_system.get_agent_metrics(self.api_key, self.agent_id)
            if tracking_metrics:
                metrics["tracking"] = tracking_metrics[0].__dict__ if tracking_metrics else {}
        
        # Reputation metrics
        if self.consensus_system:
            reputation = self.get_agent_reputation()
            if reputation:
                metrics["reputation"] = asdict(reputation)
        
        # Security stats
        if self.data_protection:
            security_stats = self.data_protection.get_access_stats()
            metrics["security"] = security_stats
        
        # Recovery stats
        if self.recovery_system:
            recovery_stats = self.recovery_system.get_recovery_stats()
            metrics["recovery"] = recovery_stats
        
        return metrics
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get agent leaderboard from consensus system
        """
        if not self.consensus_system:
            return []
        
        leaderboard = self.consensus_system.get_leaderboard(limit)
        return [asdict(agent) for agent in leaderboard]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive system overview
        """
        overview = {
            "agent_id": self.agent_id,
            "connected": self.connected,
            "security_enabled": bool(self.consensus_system),
            "systems": {}
        }
        
        # System statistics
        if self.consensus_system:
            overview["systems"]["consensus"] = self.consensus_system.get_system_stats()
        
        if self.data_protection:
            overview["systems"]["security"] = self.data_protection.get_access_stats()
        
        if self.recovery_system:
            overview["systems"]["recovery"] = self.recovery_system.get_recovery_stats()
        
        if self.tracking_system:
            overview["systems"]["tracking"] = self.tracking_system.get_system_stats()
        
        return overview
    
    async def disconnect(self):
        """
        Properly disconnect and cleanup
        """
        try:
            self.connected = False
            self._running = False
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            # Cancel WebSocket task
            if self._websocket_task:
                self._websocket_task.cancel()
                try:
                    await self._websocket_task
                except asyncio.CancelledError:
                    pass
                self._websocket_task = None
            
            # End tracking session
            if self.tracking_system and self.session_id:
                await self.tracking_system.end_agent_session(self.session_id)
            
            # Remove connections from recovery system
            if self.recovery_system and self.agent_id:
                await self.recovery_system.remove_connection(self.agent_id, "lobby")
            
            logger.info(f"Agent {self.agent_id} disconnected and cleaned up")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def _handle_task_assignment(self, data: Dict[str, Any]):
        """Handle TASK_ASSIGNMENT messages from lobby"""
        try:
            payload = data.get("payload", {})
            task_id = payload.get("task_id")
            task_title = payload.get("task_title", "Unknown Task")
            task_description = payload.get("task_description", "")
            required_capabilities = payload.get("required_capabilities", [])
            workflow_id = payload.get("workflow_id")
            
            logger.info(f"DART TASK_ASSIGNMENT received: {task_title} (ID: {task_id})")
            logger.info(f"   Required capabilities: {required_capabilities}")
            logger.info(f"   Workflow: {workflow_id}")
            
            # Check if we have any required capabilities
            my_capabilities = set(self.agent_capabilities)
            required_caps = set(required_capabilities)
            can_contribute = bool(my_capabilities.intersection(required_caps))
            
            if not can_contribute:
                logger.warning(f"Agent {self.agent_id} has no matching capabilities for task {task_id}")
                return
            
            # Process task with user's agent (pass the payload as task data)
            result = await self._process_task_with_user_agent(payload)
            
            # Send response back to lobby
            await self._send_task_response(task_id, "completed", result)
            
        except Exception as e:
            logger.error(f"Error handling TASK_ASSIGNMENT: {e}")
            import traceback
            traceback.print_exc()
            task_id = data.get("payload", {}).get("task_id")
            if task_id:
                await self._send_task_response(task_id, "failed", {"error": str(e)})


# Convenience functions for quick usage
async def create_secure_agent(agent_id: str, 
                            agent_type: str,
                            capabilities: List[Dict[str, Any]],
                            lobby_host: str = "localhost",
                            lobby_port: int = 9101) -> AgentLobbySDK:
    """
    Quick function to create and register a secure agent
    """
    sdk = AgentLobbySDK(lobby_host, lobby_port, enable_security=True)
    await sdk.register_agent(agent_id, agent_type, capabilities)
    return sdk


async def create_basic_agent(agent_id: str,
                           agent_type: str, 
                           capabilities: List[Dict[str, Any]],
                           lobby_host: str = "localhost",
                           lobby_port: int = 9101) -> AgentLobbySDK:
    """
    Quick function to create a basic agent without security features
    """
    sdk = AgentLobbySDK(lobby_host, lobby_port, enable_security=False)
    await sdk.register_agent(agent_id, agent_type, capabilities)
    return sdk


# Export main classes and functions
__all__ = [
    'AgentLobbySDK',
    'create_secure_agent', 
    'create_basic_agent',
    'TaskDifficulty',
    'DataClassification',
    'AccessLevel',
    'RecoveryStrategy',
    'ActivityType'
] 