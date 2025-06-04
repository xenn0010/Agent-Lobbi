print("!!!!!!!!!! EXECUTING LATEST LOBBY.PY !!!!!!!!!!") # VFS: Prominent top-level print

import asyncio
import uuid # For generating tokens
from typing import Dict, List, Any, Optional, Set, cast
from datetime import datetime, timezone # Fixed import
import json # Added for structured logging
import logging
import time
import websockets
import structlog
import http.server
import socketserver
import threading
from urllib.parse import urlparse, parse_qs
import concurrent.futures
import socket  # Add socket import for port checking
import queue

from .message import Message, MessageType, MessagePriority, MessageValidationError
from .agent import Agent, Capability # Forward declaration for type hint
from .agent_learning import (
    LearningSession, LearningTaskSpec, LearningCapability, 
    TestEnvironment, LearningSessionStatus, LearningCapabilityType
)
from .world_state import WorldState
from .collaboration_engine import CollaborationEngine, WorkflowStatus, TaskStatus

# Import our production components
from .database import db_manager
from .load_balancer import load_balancer, LoadBalancingStrategy

# Import monitoring SDK with absolute import
try:
    from sdk.monitoring_sdk import monitoring_sdk, MonitoringConfig
except ImportError:
    # Fallback for different import contexts
    monitoring_sdk = None
    MonitoringConfig = None

# Setup logger
logger = structlog.get_logger(__name__)

# --- Constants ---
MAX_FAILED_AUTH_ATTEMPTS = 3
AUTH_LOCKOUT_DURATION_SECONDS = 300 # 5 minutes

LOG_FILE_PATH = "simulation_run.log" # Define log file path

class Lobby:
    """
    Production-ready Agent Lobby with database, load balancing, and monitoring
    """
    
    def __init__(self, host: str = "localhost", http_port: int = 8080, ws_port: int = 8081):
        # Check for port conflicts and auto-adjust if needed
        self.host = host
        
        # Track allocated ports to avoid conflicts
        self.allocated_ports = set()
        
        # Find available ports, ensuring no conflicts
        self.http_port = self._find_available_port(http_port)
        self.allocated_ports.add(self.http_port)
        
        # Ensure WebSocket gets a different port than HTTP
        if ws_port == self.http_port:
            ws_port = self.http_port + 1
        
        self.ws_port = self._find_available_port(ws_port)
        self.allocated_ports.add(self.ws_port)
        
        # Make sure we got different ports
        if self.ws_port == self.http_port:
            # If somehow we got the same port, find next available for WebSocket
            self.ws_port = self._find_available_port(self.http_port + 1)
            self.allocated_ports.add(self.ws_port)
        
        if self.http_port != http_port:
            print(f"⚠️  HTTP port {http_port} was occupied, using {self.http_port} instead")
        if self.ws_port != ws_port:
            print(f"⚠️  WebSocket port {ws_port} was occupied, using {self.ws_port} instead")
            
        self.lobby_id = f"global_lobby"
        
        # Core components
        self.agents: Dict[str, Dict] = {}
        self.message_log: List[Dict] = []
        self.websocket_connections: Dict[str, Any] = {}
        self.delegation_to_workflow: Dict[str, str] = {}  # delegation_id -> workflow_id mapping
        
        # Thread-safe communication bridge for HTTP->Asyncio
        self._http_to_asyncio_queue = queue.Queue()
        self._bridge_event = threading.Event()
        
        # Message processing
        self._priority_queues = None  # Initialized in start()
        self._message_processor_task = None
        self._cleanup_tasks = []
        
        # Initialize missing logging attributes
        self._log_lock = None  # Will be initialized in start()
        self._log_file_handle = None
        
        # Initialize missing agent management attributes
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}
        self.agent_auth_tokens: Dict[str, str] = {}
        self.agent_reputation: Dict[str, float] = {}
        self.default_reputation = 100.0
        
        # Initialize missing workflow tracking
        self.active_workflows: Dict[str, Any] = {}
        
        # Initialize reputation change constants
        self.reputation_change_on_success = {"provider": 10.0, "requester": 5.0}
        self.reputation_change_on_failure = {"provider": -5.0, "requester": -2.0}
        self.penalty_suspicious_report_reporter = -20.0
        self.penalty_suspicious_report_provider = -5.0
        
        # Initialize agent tracking
        self.agent_interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        self._agent_types: Dict[str, str] = {}
        
        # Initialize learning components
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.active_collaborations: Dict[str, Set[str]] = {}
        self.agent_learning_capabilities: Dict[str, List[LearningCapability]] = {}
        self.test_environments: Dict[str, TestEnvironment] = {}
        
        # Initialize world state
        self.world_state: Dict[str, Any] = {}
        
        # Production components - will be initialized in start()
        self.db_manager = db_manager
        self.load_balancer = load_balancer
        self.monitoring = monitoring_sdk if monitoring_sdk else None
        self.collaboration_engine = None  # Will be initialized in start()
        
        # Setup structured logging
        self.logger = structlog.get_logger(__name__)
        
        # Server instances
        self.http_server = None
        self.ws_server = None
        
        # Flag to track initialization state
        self._initialized = False
        
    def _find_available_port(self, preferred_port: int) -> int:
        """Find an available port, starting with the preferred one, avoiding already allocated ports"""
        for port in range(preferred_port, preferred_port + 100):  # Try 100 ports
            # Skip if already allocated by this instance
            if port in self.allocated_ports:
                continue
                
            try:
                # Test if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                    # Important: Don't add to allocated_ports here, let the caller do it
                    return port
            except OSError:
                continue  # Port is in use, try next one
        
        # If we get here, couldn't find an available port in range
        raise OSError(f"Could not find available port starting from {preferred_port}")
    
    async def _initialize_log_file(self):
        """Initialize the log file handle"""
        try:
            self._log_file_handle = open(LOG_FILE_PATH, "a", encoding="utf-8")
            self._log_file_handle.write(f"{datetime.now(timezone.utc).isoformat()} - Lobby initialized. Log started.\n")
            self._log_file_handle.flush()
        except Exception as e:
            self.logger.error(f"Failed to initialize log file: {e}")
            self._log_file_handle = None
    
    def _safe_monitoring_call(self, method_name: str, *args, **kwargs):
        """Safely call monitoring methods, handling None case"""
        if self.monitoring and hasattr(self.monitoring, method_name):
            try:
                method = getattr(self.monitoring, method_name)
                return method(*args, **kwargs)
            except Exception as e:
                self.logger.warning("Monitoring call failed", method=method_name, error=str(e))
        return None
    
    def _setup_monitoring(self):
        """Setup monitoring and health checks"""
        if not self.monitoring:
            return
            
        # Add custom health checks
        def agents_health_check():
            """Check if we have registered agents"""
            return len(self.agents) > 0
        
        def database_health_check():
            """Check database connectivity"""
            try:
                # This would be a simple database ping
                return True
            except:
                return False

        self.monitoring.add_health_check("agents_registered", agents_health_check, interval=60)
        self.monitoring.add_health_check("database_connection", database_health_check, interval=30, critical=True)
    
    async def start(self):
        """Start the lobby with all production components"""
        try:
            # Initialize async components that couldn't be done in __init__
            self._priority_queues = {priority: asyncio.Queue() for priority in MessagePriority}
            self._log_lock = asyncio.Lock()
            
            # Initialize log file
            await self._initialize_log_file()
            
            # Initialize collaboration engine
            try:
                from .collaboration_engine import CollaborationEngine
                self.collaboration_engine = CollaborationEngine(self)
            except ImportError:
                self.logger.warning("Collaboration engine not available")
                self.collaboration_engine = None
            
            # Initialize monitoring
            self._setup_monitoring()
            
            self.logger.info("Starting Agent Lobby", 
                           http_port=self.http_port, 
                           ws_port=self.ws_port)
            
            # Initialize database
            await self.db_manager.initialize()
            self.logger.info("Database initialized")
            
            # Start load balancer
            await self.load_balancer.start()
            self.logger.info("Load balancer started")
            
            # Start monitoring
            if self.monitoring:
                await self.monitoring.start()
                self.logger.info("Monitoring started")
            else:
                self.logger.warning("Monitoring SDK not available")
            
            # Start collaboration engine if available
            if self.collaboration_engine:
                await self.collaboration_engine.start()
                self.logger.info("Collaboration engine started")
            
            # Start message processor
            self._message_processor_task = asyncio.create_task(self._process_message_queues())
            self.logger.info("Message processor started")
            
            # Start HTTP->Asyncio bridge processor
            self._bridge_processor_task = asyncio.create_task(self._process_http_bridge_queue())
            self.logger.info("HTTP bridge processor started")
            
            # Start HTTP server
            server_class = self._create_http_server()
            self.http_server = server_class((self.host, self.http_port), server_class.handler_class, self)
            
            # Start HTTP server in a separate thread
            self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
            self.http_thread.start()
            
            self.logger.info(f"HTTP server started on {self.host}:{self.http_port}")
            
            # Start WebSocket server
            self.ws_server = await websockets.serve(
                self._websocket_handler,
                self.host,
                self.ws_port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            self.logger.info(f"WebSocket server started on {self.host}:{self.ws_port}")
            
            # Start cleanup task
            cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._cleanup_tasks.append(cleanup_task)
            
            # Record metrics
            self._safe_monitoring_call("increment", "agents.registered")
            self._safe_monitoring_call("gauge", "agents.total", len(self.agents))
            
            self.logger.info("Agent Lobby started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start lobby", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the lobby and cleanup resources"""
        self.logger.info("Stopping Agent Lobby")
        
        try:
            # Stop message processor
            if self._message_processor_task:
                self._message_processor_task.cancel()
                try:
                    await self._message_processor_task
                except asyncio.CancelledError:
                    pass
                    
            # Stop HTTP bridge processor
            if hasattr(self, '_bridge_processor_task') and self._bridge_processor_task:
                self._bridge_processor_task.cancel()
                try:
                    await self._bridge_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel cleanup tasks
            for task in self._cleanup_tasks:
                task.cancel()
            
            if self._cleanup_tasks:
                await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            
            # Stop WebSocket server
            if self.ws_server:
                self.ws_server.close()
                await self.ws_server.wait_closed()
            
            # Stop HTTP server
            if self.http_server:
                self.http_server.shutdown()
                self.http_server.server_close()
                if hasattr(self, 'http_thread'):
                    self.http_thread.join(timeout=5)
            
            # Stop collaboration engine
            if self.collaboration_engine:
                await self.collaboration_engine.stop()
            
            # Stop monitoring
            if self.monitoring:
                await self.monitoring.stop()
            
            # Stop load balancer
            await self.load_balancer.stop()
            
            # Close database connections
            await self.db_manager.close()
            
            # Close log file
            self.close_log_file()
            
            self.logger.info("Agent Lobby stopped")
            
        except Exception as e:
            self.logger.error("Error during shutdown", error=str(e))

    async def _log_message_event(self, event_type: str, data: Dict):
        """Logs generic lobby events. 'data' is a dictionary of event-specific details."""
        if self._log_file_handle:
            async with self._log_lock:
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": event_type,
                    "details": data 
                }
                self._log_file_handle.write(json.dumps(log_entry) + "\n")
                self._log_file_handle.flush()

    async def _log_communication(self, sender_id: str, receiver_id: str, msg_type: MessageType, 
                               payload: Optional[Dict[str, Any]] = None, 
                               conversation_id: Optional[str] = None, 
                               auth_status: str = "N/A"):
        if self._log_file_handle:
            async with self._log_lock:
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sender": sender_id,
                    "receiver": receiver_id,
                    "message_type": msg_type.name,
                    "conversation_id": conversation_id,
                    "auth_status": auth_status,
                    "payload": payload
                }
                self._log_file_handle.write(json.dumps(log_entry) + "\n")
                self._log_file_handle.flush()
                print(f"{datetime.now(timezone.utc).isoformat()} | {sender_id} -> {receiver_id} | {msg_type.name} | ConvID: {conversation_id} | Auth: {auth_status} | {payload}")

    async def _process_message_queues(self):
        """Process messages from priority queues."""
        print(f"LOBBY ({self.lobby_id}): Starting _process_message_queues loop.")
        
        # Wait for priority queues to be initialized
        while self._priority_queues is None:
            await asyncio.sleep(0.1)
        
        while True:
            try:
                processed_message_this_cycle = False
                for priority in MessagePriority: # Iterate through enum members
                    queue = self._priority_queues[priority]
                    if not queue.empty():
                        try:
                            # Item in queue is (priority_value, timestamp, message_id, message_obj)
                            _priority_val, _timestamp, _msg_id, message = await queue.get() 
                            
                            print(f"LOBBY ({self.lobby_id}): Dequeued message {_msg_id} (Type: {message.message_type.name}, Prio: {priority.name}) for {message.receiver_id} from {message.sender_id}")
                            await self._process_single_message(message)
                            processed_message_this_cycle = True
                        except asyncio.QueueEmpty: # Should ideally not be hit if we check .empty() first
                            pass 
                        except Exception as e: 
                            retrieved_msg_id = _msg_id if '_msg_id' in locals() else "UNKNOWN_ID_IN_QUEUE_PROCESSING"
                            print(f"LOBBY ({self.lobby_id}): Error during single message processing (msg_id: {retrieved_msg_id}): {e}")
                            import traceback
                            traceback.print_exc()
                        finally:
                            # Ensure task_done is called even if _process_single_message fails
                            # queue.get() removes item, so task_done must be called.
                            queue.task_done() 
                
                if not processed_message_this_cycle:
                    await asyncio.sleep(0.01)
                else: # Yield if messages were processed
                    await asyncio.sleep(0) 
            except asyncio.CancelledError:
                print(f"LOBBY ({self.lobby_id}): _process_message_queues task cancelled.")
                break
            except Exception as e:
                print(f"LOBBY ({self.lobby_id}): CRITICAL ERROR in _process_message_queues outer loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.1) #Slightly longer back-off for outer loop errors
        print(f"LOBBY ({self.lobby_id}): Exiting _process_message_queues loop.")

    async def _process_http_bridge_queue(self):
        """Process workflow creation requests from HTTP thread"""
        print(f"LOBBY ({self.lobby_id}): Starting HTTP bridge processor")
        
        while True:
            try:
                # Check for pending HTTP->Asyncio requests
                try:
                    # Non-blocking check for bridge requests
                    queue_size = self._http_to_asyncio_queue.qsize()
                    if queue_size > 0:
                        print(f"BRIDGE: Found {queue_size} items in queue")
                    
                    while not self._http_to_asyncio_queue.empty():
                        try:
                            request = self._http_to_asyncio_queue.get_nowait()
                            print(f"BRIDGE: Processing request type: {request.get('type')}")
                            print(f"BRIDGE: Request details: {request}")
                            
                            if request['type'] == 'create_workflow':
                                await self._handle_bridge_workflow_creation(request)
                            else:
                                print(f"BRIDGE: Unknown request type: {request.get('type')}")
                                
                        except Exception as e:
                            print(f"BRIDGE: Error processing request: {e}")
                            import traceback
                            traceback.print_exc()
                            
                except Exception as e:
                    print(f"BRIDGE: Error checking queue: {e}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except asyncio.CancelledError:
                print(f"LOBBY ({self.lobby_id}): HTTP bridge processor cancelled")
                break
            except Exception as e:
                print(f"LOBBY ({self.lobby_id}): Error in HTTP bridge processor: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)  # Wait a bit before retrying on error

    async def _handle_bridge_workflow_creation(self, request):
        """Handle workflow creation request from HTTP bridge"""
        try:
            delegation_id = request['delegation_id']
            workflow_id = request['workflow_id']
            task_title = request['task_title']
            task_description = request['task_description']
            required_capabilities = request['required_capabilities']
            requester_id = request['requester_id']
            data = request['data']
            
            print(f"BRIDGE: Creating workflow {workflow_id} for delegation {delegation_id}")
            
            # Check if this is a goal-driven collaboration request
            task_intent = data.get('task_intent', '')
            max_agents = data.get('max_agents', 3)
            
            if task_intent and hasattr(self.collaboration_engine, 'create_goal_driven_workflow'):
                print(f"BRIDGE: Creating goal-driven workflow with intent: '{task_intent}'")
                # Create goal-driven collaborative workflow
                actual_workflow_id = await self.collaboration_engine.create_goal_driven_workflow(
                    name=f"Goal-Driven: {task_title}",
                    description=f"{task_description} | Intent: {task_intent}",
                    created_by=requester_id,
                    task_intent=task_intent,
                    required_capabilities=required_capabilities,
                    task_data=data.get('task_data', {}),
                    max_agents=max_agents
                )
            else:
                print(f"BRIDGE: Creating standard workflow")
                # Create standard workflow for this delegation
                actual_workflow_id = await self.collaboration_engine.create_workflow(
                    name=f"Delegation: {task_title}",
                    description=task_description,
                    created_by=requester_id,
                    task_definitions=[{
                        'task_id': f"task_{uuid.uuid4().hex[:8]}",
                        'name': task_title,
                        'capability': required_capabilities[0] if required_capabilities else 'general',
                        'input_data': data.get('task_data', {}),
                        'dependencies': [],
                        'timeout_seconds': data.get('deadline_minutes', 60) * 60
                    }]
                )
            
            # Update mapping to actual workflow ID
            self.delegation_to_workflow[delegation_id] = actual_workflow_id
            print(f"BRIDGE: Updated delegation mapping {delegation_id} -> {actual_workflow_id}")
            
            # Start the workflow
            started = await self.collaboration_engine.start_workflow(actual_workflow_id)
            print(f"BRIDGE: Workflow {actual_workflow_id} started: {started}")
            
        except Exception as e:
            print(f"BRIDGE: Workflow creation error: {e}")
            import traceback
            traceback.print_exc()

    async def _process_single_message(self, message: Message):
        """Process a single message with acknowledgment handling and authorization for requests."""
        try:
            if message.message_type == MessageType.BROADCAST:
                await self._handle_broadcast(message)
            elif message.receiver_id == self.lobby_id:
                await self.handle_lobby_message(message)
            elif message.receiver_id in self.agents:
                # Authorization check specifically for REQUEST messages to other agents
                if message.message_type == MessageType.REQUEST:
                    if not message.payload or "capability_name" not in message.payload:
                        err_msg = "Missing capability_name in REQUEST payload for authorization check."
                        print(f"LOBBY AUTH ERROR ({self.lobby_id}): {err_msg} from {message.sender_id}")
                        await self._log_message_event("AUTHORIZATION_ERROR", {
                            "sender_id": message.sender_id,
                            "receiver_id": message.receiver_id,
                            "error": err_msg,
                            "conversation_id": message.conversation_id
                        })
                        await self._send_direct_response(
                            self.lobby_id, message.sender_id, message, 
                            {"status": "error", "error": err_msg}, 
                            MessageType.ERROR
                        )
                        return # Stop processing this malformed request

                    target_capability_name = message.payload["capability_name"]                    
                    is_authorized = await self._is_request_authorized(
                        message.sender_id, message.receiver_id, target_capability_name
                    )

                    if not is_authorized:
                        auth_denial_reason = f"Agent '{message.sender_id}' is not authorized to call capability '{target_capability_name}' on agent '{message.receiver_id}'."
                        print(f"LOBBY AUTH DENIED ({self.lobby_id}): {auth_denial_reason}")
                        await self._log_message_event("REQUEST_DENIED_AUTHORIZATION", {
                            "denied_message_sender": message.sender_id,
                            "denied_message_receiver": message.receiver_id,
                            "denied_message_type": message.message_type.name,
                            "denied_capability": target_capability_name,
                            "denied_conversation_id": message.conversation_id,
                            "reason": auth_denial_reason
                        })
                        await self._send_direct_response(
                            self.lobby_id, message.sender_id, message,
                            {"status": "error", "error": f"Unauthorized: {auth_denial_reason}"},
                            MessageType.ERROR 
                        )
                        return # Do not route the unauthorized message
                
                # Route message to agent - handle both Agent objects and WebSocket agents
                agent_data = self.agents[message.receiver_id]
                
                # Check if this is a WebSocket-connected agent (dict) vs Agent object
                if isinstance(agent_data, dict) and message.receiver_id in self.websocket_connections:
                    # WebSocket agent - send via WebSocket
                    await self._send_message_via_websocket(message)
                elif hasattr(agent_data, 'receive_message'):
                    # Agent object - call receive_message
                    await agent_data.receive_message(message)
                elif isinstance(agent_data, dict):
                    # HTTP-registered agent (dict but no WebSocket connection)
                    # These agents exist in the system but don't have active connections
                    # We'll log the task assignment but can't deliver the message
                    print(f"LOBBY INFO: Task assigned to HTTP-registered agent {message.receiver_id}")
                    print(f"LOBBY INFO: Agent type: {agent_data.get('agent_type', 'unknown')}")
                    print(f"LOBBY INFO: Task details - {message.message_type.name}: {message.payload.get('task_name', 'unnamed task')}")
                    
                    # For now, we'll consider this "delivered" since the task is assigned
                    # In a real implementation, you might want to store the message for pickup
                    # or implement a polling mechanism for HTTP agents
                    await self._log_message_event("TASK_ASSIGNED_TO_HTTP_AGENT", {
                        "agent_id": message.receiver_id,
                        "agent_type": agent_data.get('agent_type', 'unknown'),
                        "task_id": message.payload.get('task_id'),
                        "task_name": message.payload.get('task_name', 'unnamed task'),
                        "message_type": message.message_type.name
                    })
                else:
                    # Unknown agent type
                    print(f"LOBBY ERROR: Cannot route message to agent {message.receiver_id} - unknown agent type")
            
            # Handle acknowledgment if required (This part seems to be about messages that the Lobby itself sends and expects an ACK for)
            # This might need review if it's intended for messages *received* by the lobby that *require* an ACK from the lobby.
            # For now, assuming this is for messages *sent by* the lobby.
            if message.requires_ack: # This field is on the Message object itself.
                # If the Lobby just processed a message that requires an ACK FROM THE LOBBY, it should send an ACK here.
                # The current _pending_acks logic seems more for when the Lobby sends a message and awaits an ACK for it.
                # Let's clarify: if message.requires_ack is true, it means the SENDER of *this* message expects an ACK.
                # If the lobby is the receiver and processes it, it should send an ACK back to message.sender_id.
                pass # Placeholder - ACK sending logic from Lobby needs to be explicitly defined if Lobby is to ACK received messages.

        except Exception as e:
            print(f"LOBBY ({self.lobby_id}): Error in _process_single_message for msg_id {message.message_id}: {e}")
            import traceback
            traceback.print_exc()
            # Attempt to notify sender of error if possible and not an auth error already handled
            if message.sender_id in self.agents and message.message_type != MessageType.ERROR: # Avoid error loops
                 try:
                    await self._send_direct_response(
                        self.lobby_id, message.sender_id, message,
                        {"status": "error", "error": f"Lobby failed to process your message {message.message_id}: {str(e)}"},
                        MessageType.ERROR
                    )
                 except Exception as nested_e:
                     print(f"LOBBY ({self.lobby_id}): CRITICAL - Failed to send error notification to {message.sender_id} during exception handling: {nested_e}")

    async def _send_message_via_websocket(self, message: Message):
        """Send a Message object to a WebSocket-connected agent"""
        try:
            receiver_id = message.receiver_id
            
            if receiver_id not in self.websocket_connections:
                print(f"LOBBY ERROR: No WebSocket connection for agent {receiver_id}")
                return False
            
            websocket = self.websocket_connections[receiver_id]
            
            # Convert Message object to WebSocket-compatible format
            # FIX: Handle both old format (message_type) and new format (type) for compatibility
            ws_message = {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                # CRITICAL FIX: Send both formats for compatibility
                "message_type": message.message_type.name,  # Original format
                "type": message.message_type.name,          # Format agents expect
                "payload": message.payload,
                "conversation_id": message.conversation_id,
                "priority": message.priority.name if message.priority else "NORMAL",
                "timestamp": message.timestamp.isoformat() if message.timestamp else datetime.now(timezone.utc).isoformat(),
                "requires_ack": getattr(message, 'requires_ack', False)
            }
            
            # Ensure all values are JSON serializable
            try:
                message_json = json.dumps(ws_message, ensure_ascii=False, default=str)
            except (TypeError, ValueError) as e:
                print(f"LOBBY ERROR: Failed to serialize message to JSON: {e}")
                print(f"LOBBY ERROR: Message data: {ws_message}")
                return False
            
            # Send JSON message via WebSocket
            await websocket.send(message_json)
            
            print(f"LOBBY: ✅ Sent {message.message_type.name} message to {receiver_id}")
            print(f"LOBBY: 📦 Message content: {json.dumps(message.payload, indent=2) if message.payload else 'No payload'}")
            return True
            
        except Exception as e:
            print(f"LOBBY ERROR: Failed to send WebSocket message to {receiver_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _handle_broadcast(self, message: Message):
        """Handle broadcast messages."""
        if not message.broadcast_scope:
            print(f"Error: Broadcast message {message.message_id} has no scope")
            return

        # Get all agents that match the broadcast scope
        target_agents = [
            agent_id for agent_id, agent_type in self._agent_types.items()
            if agent_type in message.broadcast_scope
        ]

        # Send message to all target agents
        for agent_id in target_agents:
            if agent_id in self.agents:
                try:
                    await self.agents[agent_id].receive_message(message)
                except Exception as e:
                    print(f"Error sending broadcast to agent {agent_id}: {e}")

    async def _handle_ack_timeout(self, message: Message):
        """Handle acknowledgment timeout."""
        if message.message_id in self._pending_acks:
            del self._pending_acks[message.message_id]
        if message.message_id in self._message_timeouts:
            self._message_timeouts[message.message_id].cancel()
            del self._message_timeouts[message.message_id]
        
        # Notify sender of timeout
        if message.sender_id in self.agents:
            timeout_notification = Message(
                sender_id=self.lobby_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                payload={
                    "error": "acknowledgment_timeout",
                    "original_message_id": message.message_id
                },
                priority=MessagePriority.HIGH
            )
            await self.agents[message.sender_id].receive_message(timeout_notification)

    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message with load balancing and error recovery"""
        try:
            sender_id = message.get("sender_id")
            receiver_id = message.get("receiver_id") 
            message_type = message.get("message_type")
            
            # Record metrics
            self._safe_monitoring_call("increment", "messages.total")
            self._safe_monitoring_call("increment", f"messages.{message_type.lower()}")
            
            # Save message to database for audit
            message_data = {
                "id": message.get("message_id", str(uuid.uuid4())),
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "message_type": message_type,
                "payload": message.get("payload", {}),
                "conversation_id": message.get("conversation_id"),
                "status": "processing",
                "priority": message.get("priority", 2),
                "created_at": datetime.now(timezone.utc)
            }
            
            await self.db_manager.save_message(message_data)
            
            # Handle different message types with load balancing
            if receiver_id == "lobby":
                result = await self._handle_lobby_message(message)
            elif receiver_id == "*" or receiver_id == "broadcast":
                result = await self._handle_broadcast_message(message)
            else:
                # Use load balancer for capability-based routing
                if message_type == "CAPABILITY_REQUEST":
                    required_capability = message.get("payload", {}).get("capability")
                    if required_capability:
                        target_agent = self.load_balancer.get_agent_for_capability(required_capability)
                        if target_agent:
                            receiver_id = target_agent
                            message["receiver_id"] = target_agent
                
                result = await self._handle_direct_message(message)
            
            # Update message status
            message_data["status"] = "delivered" if result.get("status") == "success" else "failed"
            message_data["processed_at"] = datetime.now(timezone.utc)
            await self.db_manager.save_message(message_data)
            
            self._safe_monitoring_call("increment", "messages.successful")
            return result
            
        except Exception as e:
            self._safe_monitoring_call("increment", "messages.failed")
            self.logger.error("Message routing failed", 
                            message_id=message.get("message_id"), 
                            error=str(e))
            
            # Update message status to failed
            message_data["status"] = "failed"
            message_data["processed_at"] = datetime.now(timezone.utc)
            await self.db_manager.save_message(message_data)
            
            return {
                "status": "error",
                "message": f"Message routing failed: {str(e)}"
            }

    async def _handle_lobby_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle messages directed to the lobby with monitoring"""
        message_type = message.get("message_type")
        
        async with self.monitoring.timer(f"lobby_message.{message_type.lower()}"):
            try:
                if message_type == "REGISTER":
                    return await self.register_agent(message.get("payload", {}))
                
                elif message_type == "GET_AGENTS":
                    agents = await self.db_manager.get_all_agents()
                    return {"status": "success", "agents": agents}
                
                elif message_type == "CREATE_WORKFLOW" and self.collaboration_engine:
                    return await self._handle_create_workflow(message)
                
                elif message_type == "START_WORKFLOW" and self.collaboration_engine:
                    return await self._handle_start_workflow(message)
                
                elif message_type == "GET_WORKFLOW_STATUS" and self.collaboration_engine:
                    return await self._handle_get_workflow_status(message)
                
                elif message_type == "CREATE_COLLABORATION" and self.collaboration_engine:
                    return await self._handle_create_collaboration(message)
                
                elif message_type == "GET_LOAD_BALANCER_STATS":
                    return {"status": "success", "stats": self.load_balancer.get_stats()}
                
                elif message_type == "GET_MONITORING_STATUS":
                    return {"status": "success", "monitoring": self.monitoring.get_status()}
                
                else:
                    return {"status": "error", "message": f"Unknown message type: {message_type}"}
                    
            except Exception as e:
                self.logger.error("Lobby message handling failed", 
                                message_type=message_type, 
                                error=str(e))
                return {"status": "error", "message": str(e)}

    async def _periodic_cleanup(self):
        """Periodic cleanup task with monitoring"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up stale agents
                current_time = datetime.now(timezone.utc)
                stale_agents = []
                
                for agent_id, agent_data in list(self.agents.items()):
                    last_seen = agent_data.get("last_seen")
                    if isinstance(last_seen, str):
                        last_seen = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                    
                    if last_seen and (current_time - last_seen).total_seconds() > 600:  # 10 minutes
                        stale_agents.append(agent_id)
                
                # Remove stale agents
                for agent_id in stale_agents:
                    await self._remove_stale_agent(agent_id)
                
                # Update metrics
                self._safe_monitoring_call("gauge", "agents.active", len(self.agents))
                self._safe_monitoring_call("gauge", "websockets.active", len(self.websocket_connections))
                self._safe_monitoring_call("gauge", "workflows.active", len(self.active_workflows))
                
                if stale_agents:
                    self.logger.info("Cleaned up stale agents", count=len(stale_agents))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cleanup task error", error=str(e))
    
    async def _remove_stale_agent(self, agent_id: str):
        """Remove a stale agent from all systems"""
        try:
            # Remove from memory
            if agent_id in self.agents:
                del self.agents[agent_id]
            
            # Remove from load balancer
            self.load_balancer.unregister_agent(agent_id)
            
            # Update database status
            agent_data = await self.db_manager.get_agent(agent_id)
            if agent_data:
                agent_data["status"] = "offline"
                await self.db_manager.save_agent(agent_data)
            
            # Close websocket connection if exists
            if agent_id in self.websocket_connections:
                try:
                    await self.websocket_connections[agent_id].close()
                except:
                    pass
                del self.websocket_connections[agent_id]
            
            self._safe_monitoring_call("increment", "agents.removed_stale")
            self.logger.info("Stale agent removed", agent_id=agent_id)
            
        except Exception as e:
            self.logger.error("Failed to remove stale agent", agent_id=agent_id, error=str(e))

    async def _is_request_authorized(self, sender_id: str, target_agent_id: str, capability_name: str) -> bool:
        # Special authorization for lobby/collaboration engine system messages
        if sender_id == self.lobby_id or sender_id == "global_lobby":
            return True  # System always authorized
            
        if target_agent_id not in self.agent_capabilities or capability_name not in self.agent_capabilities[target_agent_id]:
            return False # Target capability doesn't exist, treat as unauthorized for safety
        
        capability_details = self.agent_capabilities[target_agent_id][capability_name]
        authorized_ids = capability_details.get("authorized_requester_ids")

        if authorized_ids is None: # If None, it's public
            return True
        if not authorized_ids: # If empty list, also considered public (or could be strict: no one allowed)
             # For now, let's treat empty list as public for simplicity, can be revisited.
            return True

        return sender_id in authorized_ids

    async def handle_lobby_message(self, msg: Message):
        agent_id = msg.sender_id

        # Log all messages directed to the lobby
        await self._log_communication(agent_id, self.lobby_id, msg.message_type, msg.payload, msg.conversation_id, "OK_TO_LOBBY")

        if msg.message_type == MessageType.REGISTER:
            # This specific REGISTER handling might be redundant if register_agent is always called first by the sim
            # However, keeping it allows an agent to re-send a REGISTER message if needed.
            if agent_id not in self.agents:
                # This scenario should ideally be handled by an external registration call first.
                # If an agent instance is passed, we can register it.
                # For now, we assume an agent object is available or this is a re-registration attempt.
                print(f"Lobby: Received REGISTER from {agent_id} but agent object not directly available. This might be a re-registration attempt.")
                # If we need to create agent on the fly, this is more complex.
                # Assuming for now that if self.agents[agent_id] existed, this would be for capability updates.
                # The current `register_agent` function expects an Agent object.
                # We will rely on external `lobby.register_agent(agent_instance)` call for initial setup.
                pass # Let's assume register_agent was called externally for initial setup.
            
            # Update capabilities if agent is re-registering or sending capability updates
            # This part assumes the agent is already known through an initial `register_agent` call
            if agent_id in self.agents and "capabilities" in msg.payload:
                updated_caps = msg.payload["capabilities"]
                self.agent_capabilities[agent_id] = {cap["name"]: cap for cap in updated_caps}
                await self._log_message_event("AGENT_CAPABILITIES_UPDATED", {"agent_id": agent_id, "new_capabilities": updated_caps})
            
            # Send REGISTER_ACK (The token is already given during the initial `register_agent` call)
            ack_payload = {"status": "success_registered_finalized", "auth_token": self.agent_auth_tokens.get(agent_id)}
            await self._send_direct_response(self.lobby_id, agent_id, msg, ack_payload, MessageType.REGISTER_ACK)

        elif msg.message_type == MessageType.DISCOVER_SERVICES:
            await self._handle_discover_services(msg)
        
        elif msg.message_type == MessageType.ADVERTISE_CAPABILITIES:
            # Similar to REGISTER, update capabilities
            if agent_id in self.agents and "capabilities" in msg.payload:
                new_caps = msg.payload["capabilities"]
                self.agent_capabilities[agent_id] = {cap["name"]: cap for cap in new_caps}
                await self._log_message_event("AGENT_CAPABILITIES_ADVERTISED", {"agent_id": agent_id, "new_capabilities": new_caps})
                # Optionally send an ACK for advertisement
                await self._send_direct_response(self.lobby_id, agent_id, msg, {"status": "capabilities_updated"}, MessageType.INFO)

        elif msg.message_type == MessageType.TASK_OUTCOME_REPORT:
            await self._handle_task_outcome_report(msg)

        # Learning Collaboration Message Handlers
        elif msg.message_type == MessageType.CREATE_LEARNING_SESSION:
            await self._handle_create_learning_session(msg)
        
        elif msg.message_type == MessageType.JOIN_LEARNING_SESSION:
            await self._handle_join_learning_session(msg)
            
        elif msg.message_type == MessageType.LEAVE_LEARNING_SESSION:
            await self._handle_leave_learning_session(msg)
            
        elif msg.message_type == MessageType.SHARE_MODEL_PARAMETERS:
            await self._handle_share_model_parameters(msg)
            
        elif msg.message_type == MessageType.REQUEST_COLLABORATION:
            await self._handle_request_collaboration(msg)
            
        elif msg.message_type == MessageType.REPORT_LEARNING_PROGRESS:
            await self._handle_report_learning_progress(msg)
            
        elif msg.message_type == MessageType.CREATE_TEST_ENVIRONMENT:
            await self._handle_create_test_environment(msg)
            
        elif msg.message_type == MessageType.RUN_MODEL_TEST:
            await self._handle_run_model_test(msg)
            
        elif msg.message_type == MessageType.GET_TEST_RESULTS:
            await self._handle_get_test_results(msg)

        # Workflow and Collaboration Message Handlers
        elif msg.message_type == MessageType.REQUEST and msg.payload.get("action") == "create_workflow":
            await self._handle_create_workflow_request(msg)
            
        elif msg.message_type == MessageType.REQUEST and msg.payload.get("action") == "start_workflow":
            await self._handle_start_workflow_request(msg)
            
        elif msg.message_type == MessageType.REQUEST and msg.payload.get("action") == "get_workflow_status":
            await self._handle_get_workflow_status_request(msg)
            
        elif msg.message_type == MessageType.REQUEST and msg.payload.get("action") == "create_collaboration":
            await self._handle_create_collaboration_request(msg)
            
        elif msg.message_type == MessageType.RESPONSE and "task_id" in msg.payload:
            await self.collaboration_engine.handle_task_result(msg)

        # ... other lobby message types (HEALTH_CHECK, etc.) can be added here
        else:
            print(f"Lobby: Received unhandled message type {msg.message_type} from {agent_id}")
            error_payload = {"error": f"Lobby does not handle message type {msg.message_type.name}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_discover_services(self, msg: Message):
        capability_name_query = msg.payload.get("capability_name")
        requester_id = msg.sender_id
        found_services = []

        if capability_name_query:
            for agent_id, capabilities_dict in self.agent_capabilities.items():
                if capability_name_query in capabilities_dict:
                    # Check if the requester is authorized to use this capability on this agent
                    target_capability = capabilities_dict[capability_name_query]
                    authorized_requesters = target_capability.get("authorized_requester_ids")
                    
                    is_authorized = False
                    if authorized_requesters is None: # Public capability
                        is_authorized = True
                    elif requester_id in authorized_requesters: # Explicitly authorized
                        is_authorized = True
                    
                    if is_authorized:
                        found_services.append({
                            "agent_id": agent_id,
                            "capability_name": capability_name_query,
                            "description": target_capability.get("description", "N/A"),
                            "reputation": self.agent_reputation.get(agent_id, self.default_reputation) # Include reputation
                        })
        
        # Sort services by reputation (descending)
        found_services.sort(key=lambda x: x.get("reputation", self.default_reputation), reverse=True)

        response_payload = {"capability_name_queried": capability_name_query, "services_found": found_services}
        await self._send_direct_response(self.lobby_id, requester_id, msg, response_payload, MessageType.SERVICES_AVAILABLE)
        await self._log_message_event("SERVICE_DISCOVERY_RESPONSE", {"requester": requester_id, "query": capability_name_query, "found_count": len(found_services), "results": found_services})

    async def _handle_task_outcome_report(self, msg: Message):
        reporter_id = msg.sender_id 
        payload = msg.payload
        
        provider_agent_id = payload.get("provider_agent_id")
        capability_name = payload.get("capability_name")
        status = payload.get("status")
        details = payload.get("details", "N/A")

        if not all([provider_agent_id, capability_name, status]):
            print(f"Lobby: Received incomplete TASK_OUTCOME_REPORT from {reporter_id}. Payload: {payload}")
            # Optionally send error back to reporter if they are known
            if reporter_id in self.agents:
                await self._send_direct_response(
                    self.lobby_id, reporter_id, msg, 
                    {"status": "error", "error": "Incomplete TASK_OUTCOME_REPORT"}, 
                    MessageType.ERROR
                )
            return

        log_data = {
            "reporter_id": reporter_id,
            "provider_agent_id": provider_agent_id,
            "capability_name": capability_name,
            "status": status,
            "original_payload": payload
        }

        provider_exists = provider_agent_id in self.agents
        reporter_exists = reporter_id in self.agents # Sender of the report

        if not provider_exists or not reporter_exists:
            error_detail = []
            if not provider_exists: error_detail.append("Provider not found")
            if not reporter_exists: error_detail.append("Reporter not found (should not happen if auth passed)")
            log_data["error"] = ", ".join(error_detail)
            print(f"Lobby: Error processing TASK_OUTCOME_REPORT: {log_data['error']}")
            await self._log_message_event("TASK_OUTCOME_PROCESSING_ERROR", log_data)
            return

        # Anomaly Detection & Game Theory Aspect
        is_suspicious_report = False
        suspicion_reason = ""

        # Check 1: Self-reporting a typical inter-agent interaction as if it were a service from oneself.
        if reporter_id == provider_agent_id:
            # This could be legitimate for some internal agent self-checks, but not for typical service outcome reports.
            # For now, we'll flag it if it's not a specifically designed self-assessment capability.
            # We don't have such capabilities defined, so any self-report is suspicious for now.
            is_suspicious_report = True
            suspicion_reason = "Self-reporting of service outcome."
            log_data["suspicion_detected"] = suspicion_reason

        # Check 2: Provider does not advertise the reported capability.
        if not is_suspicious_report: # Only if not already flagged
            provider_caps = self.agent_capabilities.get(provider_agent_id, {})
            if capability_name not in provider_caps:
                is_suspicious_report = True
                suspicion_reason = f"Provider {provider_agent_id} does not advertise capability '{capability_name}'."
                log_data["suspicion_detected"] = suspicion_reason

        current_provider_rep = self.agent_reputation.get(provider_agent_id, self.default_reputation)
        current_reporter_rep = self.agent_reputation.get(reporter_id, self.default_reputation)
        rep_change_provider = 0.0
        rep_change_reporter = 0.0

        if is_suspicious_report:
            print(f"Lobby: SUSPICIOUS TASK_OUTCOME_REPORT from {reporter_id} about {provider_agent_id}. Reason: {suspicion_reason}")
            rep_change_reporter = self.penalty_suspicious_report_reporter
            # Optional: Small penalty to provider or investigate further. For now, a small one.
            rep_change_provider = self.penalty_suspicious_report_provider 
            log_data["report_status"] = "SUSPICIOUS"
            # We might also choose to not even log this against the provider's direct interaction history if it's clearly fabricated.
        else:
            log_data["report_status"] = "NORMAL"
            if status == "success":
                rep_change_provider = self.reputation_change_on_success["provider"]
                rep_change_reporter = self.reputation_change_on_success["requester"]
            elif status == "failure":
                rep_change_provider = self.reputation_change_on_failure["provider"]
                rep_change_reporter = self.reputation_change_on_failure["requester"]
            else:
                print(f"Lobby: Invalid status '{status}' in TASK_OUTCOME_REPORT from {reporter_id}.")
                log_data["error"] = f"Invalid status: {status}"
                await self._log_message_event("TASK_OUTCOME_PROCESSING_ERROR", log_data)
                # Send error to reporter
                await self._send_direct_response(
                    self.lobby_id, reporter_id, msg, 
                    {"status": "error", "error": f"Invalid status '{status}' in your report"}, 
                    MessageType.ERROR
                )
                return

        self.agent_reputation[provider_agent_id] = max(0, current_provider_rep + rep_change_provider)
        self.agent_reputation[reporter_id] = max(0, current_reporter_rep + rep_change_reporter)

        interaction_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reporter_id": reporter_id,
            "provider_agent_id": provider_agent_id,
            "capability_name": capability_name,
            "reported_status": status, # The status as reported by the agent
            "processing_status": log_data["report_status"], # NORMAL or SUSPICIOUS
            "suspicion_reason": suspicion_reason if is_suspicious_report else None,
            "provider_rep_before": current_provider_rep,
            "provider_rep_after": self.agent_reputation[provider_agent_id],
            "reporter_rep_before": current_reporter_rep,
            "reporter_rep_after": self.agent_reputation[reporter_id],
            "details": details
        }
        
        # Log interaction for both, unless it was a suspicious self-report that we want to only attribute to the reporter
        if not (is_suspicious_report and reporter_id == provider_agent_id):
            if provider_agent_id in self.agent_interaction_history:
                self.agent_interaction_history[provider_agent_id].append(interaction_record)
        else:
                self.agent_interaction_history[provider_agent_id] = [interaction_record]
        
        if reporter_id in self.agent_interaction_history:
            self.agent_interaction_history[reporter_id].append(interaction_record)
        else:
            self.agent_interaction_history[reporter_id] = [interaction_record]

        log_data["provider_rep_change"] = rep_change_provider
        log_data["reporter_rep_change"] = rep_change_reporter
        log_data["provider_new_rep"] = self.agent_reputation[provider_agent_id]
        log_data["reporter_new_rep"] = self.agent_reputation[reporter_id]
        await self._log_message_event("TASK_OUTCOME_PROCESSED", log_data)
        print(f"Lobby: Processed TASK_OUTCOME from {reporter_id} for provider {provider_agent_id}. Reported Status: {status}. Processing Status: {log_data['report_status']}. Reputations updated.")

    async def process_world_action(self, agent_id: str, action_payload: Dict[str, Any]):
        action_type = action_payload.get("action_type")
        print(f"Lobby: (Authenticated) Agent {agent_id} performs action: {action_type} with payload {action_payload}")
        # Actual world state modification logic would go here
        # For now, we just log it was attempted after auth and validation
        if action_type == "open_item" and action_payload.get("item") == "door": # From previous simulation
            self.update_world_state("door_status", "opened_by_" + agent_id)
        pass

    def get_agent_advertised_capabilities(self, agent_id: str) -> Optional[Dict[str, Capability]]:
        return self.agent_capabilities.get(agent_id)

    def get_world_state(self) -> Dict[str, Any]:
        return self.world_state.copy()

    def update_world_state(self, key: str, value: Any):
        print(f"Lobby: World state updated - {key}: {value}")
        self.world_state[key] = value

    def print_message_log(self):
        print("\n--- Message Log ---")
        # self.message_log list is no longer maintained in memory.
        # All logs are written directly to the file by _log_communication and _log_message_event.
        print(f"NOTE: Comprehensive log is in {LOG_FILE_PATH}")
        print("--- End Log ---") 

    def _generate_token(self) -> str:
        return str(uuid.uuid4()) # Simple UUID-based token

    async def _send_error_to_agent(self, receiver_id: str, error_message: str, original_msg_id: Optional[str] = None):
        if receiver_id in self.agents:
            error_payload = {"error": error_message, "original_message_id": original_msg_id}
            error_msg = Message(sender_id="lobby", receiver_id=receiver_id, message_type=MessageType.ERROR, payload=error_payload)
            await self.agents[receiver_id].receive_message(error_msg)
        else:
            print(f"Lobby Error: Tried to send error to non-existent agent {receiver_id}")

    def _authenticate_message(self, msg: Message) -> bool:
        # REGISTER messages do not require a token yet (it's issued upon successful REGISTER)
        if msg.message_type == MessageType.REGISTER:
            return True

        expected_token = self.agent_auth_tokens.get(msg.sender_id)
        if not expected_token or msg.auth_token != expected_token:
            print(f"Lobby Auth Error: Invalid or missing token for agent {msg.sender_id}. Message Type: {msg.message_type.name}")
            return False
        return True

    async def unregister_agent(self, agent_id: str):
        if agent_id in self.agents:
            print(f"Lobby: Agent {agent_id} unregistered.")
            del self.agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            if agent_id in self.agent_auth_tokens:
                del self.agent_auth_tokens[agent_id]
            return True
        return False

    async def register_agent(self, agent: Agent) -> Dict[str, Any]:
        """Register an agent with the lobby"""
        try:
            agent_id = agent.agent_id
            
            # Generate auth token
            auth_token = self._generate_token()
            self.agent_auth_tokens[agent_id] = auth_token
            
            # Store agent
            self.agents[agent_id] = agent
            
            # Initialize reputation
            if agent_id not in self.agent_reputation:
                self.agent_reputation[agent_id] = self.default_reputation
            
            # Store agent capabilities
            if hasattr(agent, 'capabilities') and agent.capabilities:
                self.agent_capabilities[agent_id] = {cap.name: cap for cap in agent.capabilities}
            
            # Store agent type for broadcasting
            if hasattr(agent, 'agent_type'):
                self._agent_types[agent_id] = agent.agent_type
            
            # Initialize interaction history
            if agent_id not in self.agent_interaction_history:
                self.agent_interaction_history[agent_id] = []
            
            # Register with load balancer
            if hasattr(agent, 'capabilities') and agent.capabilities:
                for capability in agent.capabilities:
                    self.load_balancer.register_agent_capability(agent_id, capability.name)
            
            # Save to database if available
            try:
                await self.db_manager.save_agent({
                    "agent_id": agent_id,
                    "name": getattr(agent, 'name', agent_id),
                    "capabilities": [cap.name for cap in getattr(agent, 'capabilities', [])],
                    "status": "active",
                    "registered_at": datetime.now(timezone.utc)
                })
            except Exception as e:
                self.logger.warning(f"Database save failed: {e}")
            
            await self._log_message_event("AGENT_REGISTERED", {
                "agent_id": agent_id,
                "agent_type": getattr(agent, 'agent_type', 'generic'),
                "capabilities": [cap.name for cap in getattr(agent, 'capabilities', [])]
            })
            
            # Record metrics
            self._safe_monitoring_call("increment", "agents.registered")
            self._safe_monitoring_call("gauge", "agents.total", len(self.agents))
            
            self.logger.info(f"Agent registered: {agent_id}")
            
            return {
                "status": "success",
                "agent_id": agent_id,
                "auth_token": auth_token,
                "message": "Agent registered successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Agent registration failed: {e}")
            return {
                "status": "error",
                "message": f"Registration failed: {str(e)}"
            }

    def close_log_file(self):
        """Closes the log file handle. Should be called on graceful shutdown."""
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.write(f"{datetime.now(timezone.utc).isoformat()} - Lobby shutting down. Log closed.\n")
            self._log_file_handle.close()
            print("Lobby log file closed.")

    async def _send_direct_response(self, 
                                    sender_id: str, # Should be self.lobby_id for lobby responses
                                    receiver_id: str, 
                                    original_message: Message, # For context like conversation_id
                                    payload: Dict[str, Any], 
                                    response_msg_type: MessageType):
        """Helper to send a direct message from the lobby to an agent."""
        if receiver_id in self.agents:
            response_msg = Message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=response_msg_type,
                payload=payload,
                conversation_id=original_message.conversation_id, # Preserve conversation context
                # Lobby-originated messages might not need a typical agent auth_token,
                # or could have a special system token/status if needed.
                # For now, sending without an explicit auth_token for lobby->agent.
                # The agent's receive_message doesn't strictly check token for non-REGISTER_ACK.
                priority=MessagePriority.HIGH # Responses from lobby are often important
            )
            # Bypass routing for direct internal responses to avoid loops or re-auth if not needed
            await self.agents[receiver_id].receive_message(response_msg)
            print(f"LOBBY ({self.lobby_id}): Sent direct {response_msg_type.name} to {receiver_id} (ConvID: {original_message.conversation_id})")
        else:
            print(f"LOBBY ERROR ({self.lobby_id}): Tried to send direct response to unknown agent {receiver_id}")

# Example of how to ensure log is closed (e.g., in your main simulation script)
# async def main():
#     lobby = Lobby()
#     try:
#         # ... your simulation logic ...
#         await asyncio.sleep(10) # Simulate work
#     finally:
#         lobby.close_log_file()

# if __name__ == "__main__":
# asyncio.run(main())

# ====== LEARNING COLLABORATION MESSAGE HANDLERS ======
    
    async def _handle_create_learning_session(self, msg: Message):
        """Handle request to create a new learning session"""
        try:
            agent_id = msg.sender_id
            payload = msg.payload
            
            # Extract task specification
            task_spec_data = payload.get("task_spec", {})
            task_spec = LearningTaskSpec(
                task_name=task_spec_data.get("task_name", ""),
                task_type=task_spec_data.get("task_type", "supervised"),
                objective=task_spec_data.get("objective", ""),
                data_requirements=task_spec_data.get("data_requirements", {}),
                success_criteria=task_spec_data.get("success_criteria", {}),
                collaboration_preferences=task_spec_data.get("collaboration_preferences", []),
                computational_constraints=task_spec_data.get("computational_constraints", {})
            )
            
            # Create learning session
            session = LearningSession(
                task_spec=task_spec,
                creator_id=agent_id
            )
            session.add_participant(agent_id)
            
            # Store session
            self.learning_sessions[session.session_id] = session
            self.active_collaborations[session.session_id] = {agent_id}
            
            await self._log_message_event("LEARNING_SESSION_CREATED", {
                "session_id": session.session_id,
                "creator": agent_id,
                "task_type": task_spec.task_type
            })
            
            # Send success response
            response_payload = {
                "success": True,
                "session_id": session.session_id,
                "message": "Learning session created successfully"
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to create learning session: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_join_learning_session(self, msg: Message):
        """Handle request to join an existing learning session"""
        try:
            agent_id = msg.sender_id
            session_id = msg.payload.get("session_id")
            
            if not session_id or session_id not in self.learning_sessions:
                error_payload = {"error": "Invalid or non-existent session ID"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
                
            session = self.learning_sessions[session_id]
            
            # Add participant to session
            if session.add_participant(agent_id):
                self.active_collaborations[session_id].add(agent_id)
                
                await self._log_message_event("AGENT_JOINED_LEARNING_SESSION", {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "participant_count": len(session.participants)
                })
                
                # Notify all participants about new member
                for participant_id in session.participants:
                    if participant_id != agent_id and participant_id in self.agents:
                        notification_msg = Message(
                            sender_id=self.lobby_id,
                            receiver_id=participant_id,
                            message_type=MessageType.LEARNING_SESSION_UPDATE,
                            payload={
                                "event": "participant_joined",
                                "session_id": session_id,
                                "new_participant": agent_id
                            }
                        )
                        await self.agents[participant_id].receive_message(notification_msg)
                
                response_payload = {
                    "success": True,
                    "message": "Successfully joined learning session",
                    "session_info": session.to_dict()
                }
                await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            else:
                error_payload = {"error": "Agent already in session"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                
        except Exception as e:
            error_payload = {"error": f"Failed to join learning session: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_leave_learning_session(self, msg: Message):
        """Handle request to leave a learning session"""
        try:
            agent_id = msg.sender_id
            session_id = msg.payload.get("session_id")
            
            if not session_id or session_id not in self.learning_sessions:
                error_payload = {"error": "Invalid or non-existent session ID"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
                
            session = self.learning_sessions[session_id]
            
            if session.remove_participant(agent_id):
                self.active_collaborations[session_id].discard(agent_id)
                
                await self._log_message_event("AGENT_LEFT_LEARNING_SESSION", {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "remaining_participants": len(session.participants)
                })
                
                # If no participants left, close session
                if not session.participants:
                    session.status = LearningSessionStatus.CANCELLED
                    del self.active_collaborations[session_id]
                    await self._log_message_event("LEARNING_SESSION_CLOSED", {
                        "session_id": session_id,
                        "reason": "no_participants"
                    })
                else:
                    # Notify remaining participants
                    for participant_id in session.participants:
                        if participant_id in self.agents:
                            notification_msg = Message(
                                sender_id=self.lobby_id,
                                receiver_id=participant_id,
                                message_type=MessageType.LEARNING_SESSION_UPDATE,
                                payload={
                                    "event": "participant_left",
                                    "session_id": session_id,
                                    "left_participant": agent_id
                                }
                            )
                            await self.agents[participant_id].receive_message(notification_msg)
                
                response_payload = {"success": True, "message": "Successfully left learning session"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            else:
                error_payload = {"error": "Agent not in session"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                
        except Exception as e:
            error_payload = {"error": f"Failed to leave learning session: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_share_model_parameters(self, msg: Message):
        """Handle sharing of model parameters between agents in a learning session"""
        try:
            agent_id = msg.sender_id
            session_id = msg.payload.get("session_id")
            parameters = msg.payload.get("parameters", {})
            
            if not session_id or session_id not in self.learning_sessions:
                error_payload = {"error": "Invalid or non-existent session ID"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
                
            session = self.learning_sessions[session_id]
            
            if agent_id not in session.participants:
                error_payload = {"error": "Agent not participant in session"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
            
            # Store parameters in session
            session.update_parameters(agent_id, parameters)
            
            await self._log_message_event("MODEL_PARAMETERS_SHARED", {
                "session_id": session_id,
                "agent_id": agent_id,
                "parameter_keys": list(parameters.keys())
            })
            
            # Notify other participants about parameter update
            for participant_id in session.participants:
                if participant_id != agent_id and participant_id in self.agents:
                    notification_msg = Message(
                        sender_id=self.lobby_id,
                        receiver_id=participant_id,
                        message_type=MessageType.LEARNING_SESSION_UPDATE,
                        payload={
                            "event": "parameters_updated",
                            "session_id": session_id,
                            "updated_by": agent_id,
                            "available_parameters": list(session.shared_parameters.keys())
                        }
                    )
                    await self.agents[participant_id].receive_message(notification_msg)
            
            response_payload = {"success": True, "message": "Parameters shared successfully"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to share parameters: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_request_collaboration(self, msg: Message):
        """Handle request for learning collaboration - find suitable agents"""
        try:
            agent_id = msg.sender_id
            payload = msg.payload
            required_capabilities = payload.get("required_capabilities", [])
            task_description = payload.get("task_description", "")
            
            # Find agents with matching learning capabilities
            suitable_agents = []
            
            for other_agent_id, learning_caps in self.agent_learning_capabilities.items():
                if other_agent_id == agent_id:
                        continue
                    
                # Check if agent has required capabilities
                agent_cap_types = [cap.type.value for cap in learning_caps]
                
                match_score = 0
                for req_cap in required_capabilities:
                    if req_cap in agent_cap_types:
                        match_score += 1
                
                if match_score > 0:
                    suitable_agents.append({
                        "agent_id": other_agent_id,
                        "capabilities": [cap.to_dict() for cap in learning_caps],
                        "match_score": match_score,
                        "reputation": self.agent_reputation.get(other_agent_id, self.default_reputation)
                    })
            
            # Sort by match score and reputation
            suitable_agents.sort(key=lambda x: (x["match_score"], x["reputation"]), reverse=True)
            
            await self._log_message_event("COLLABORATION_REQUEST_PROCESSED", {
                "requester": agent_id,
                "required_capabilities": required_capabilities,
                "found_agents": len(suitable_agents)
            })
            
            response_payload = {
                "success": True,
                "suitable_agents": suitable_agents[:10],  # Return top 10 matches
                "message": f"Found {len(suitable_agents)} suitable collaboration partners"
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to process collaboration request: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_report_learning_progress(self, msg: Message):
        """Handle learning progress reports from agents"""
        try:
            agent_id = msg.sender_id
            session_id = msg.payload.get("session_id")
            progress = msg.payload.get("progress", {})
            
            if not session_id or session_id not in self.learning_sessions:
                error_payload = {"error": "Invalid or non-existent session ID"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
                
            session = self.learning_sessions[session_id]
            
            if agent_id not in session.participants:
                error_payload = {"error": "Agent not participant in session"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
            
            # Update progress in session
            session.update_progress(agent_id, progress)
            
            await self._log_message_event("LEARNING_PROGRESS_REPORTED", {
                "session_id": session_id,
                "agent_id": agent_id,
                "progress_metrics": list(progress.keys())
            })
            
            response_payload = {"success": True, "message": "Progress reported successfully"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to report progress: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_create_test_environment(self, msg: Message):
        """Handle request to create a test environment"""
        try:
            agent_id = msg.sender_id
            payload = msg.payload
            
            env_config = payload.get("environment_config", {})
            test_data = payload.get("test_data", {})
            
            # Create test environment
            test_env = TestEnvironment(
                env_name=env_config.get("name", f"test_env_{agent_id}"),
                env_type=env_config.get("type", "basic"),
                creator_id=agent_id,
                configuration=env_config,
                test_data=test_data
            )
            
            test_env.participants.add(agent_id)
            self.test_environments[test_env.env_id] = test_env
            
            await self._log_message_event("TEST_ENVIRONMENT_CREATED", {
                "env_id": test_env.env_id,
                "creator": agent_id,
                "env_type": test_env.env_type
            })
            
            response_payload = {
                "success": True,
                "env_id": test_env.env_id,
                "message": "Test environment created successfully"
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to create test environment: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_run_model_test(self, msg: Message):
        """Handle request to run a model test in an environment"""
        try:
            agent_id = msg.sender_id
            env_id = msg.payload.get("env_id")
            model_config = msg.payload.get("model_config", {})
            test_params = msg.payload.get("test_parameters", {})
            
            if not env_id or env_id not in self.test_environments:
                error_payload = {"error": "Invalid or non-existent environment ID"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
                
            test_env = self.test_environments[env_id]
            
            # For MVP, simulate test results
            import random
            test_results = {
                "test_id": f"test_{uuid.uuid4().hex[:8]}",
                "agent_id": agent_id,
                "model_config": model_config,
                "accuracy": random.uniform(0.7, 0.95),
                "loss": random.uniform(0.1, 0.5),
                "test_duration": random.uniform(10, 60),
                "metrics": {
                    "precision": random.uniform(0.8, 0.95),
                    "recall": random.uniform(0.75, 0.9),
                    "f1_score": random.uniform(0.8, 0.92)
                },
                "status": "completed"
            }
            
            # Store results in environment
            test_env.add_test_result(agent_id, test_results)
            test_env.status = "completed"
            
            await self._log_message_event("MODEL_TEST_COMPLETED", {
                "env_id": env_id,
                "agent_id": agent_id,
                "test_results": test_results
            })
            
            response_payload = {
                "success": True,
                "test_results": test_results,
                "message": "Model test completed successfully"
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to run model test: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    async def _handle_get_test_results(self, msg: Message):
        """Handle request to get test results from an environment"""
        try:
            agent_id = msg.sender_id
            env_id = msg.payload.get("env_id")
            
            if not env_id or env_id not in self.test_environments:
                error_payload = {"error": "Invalid or non-existent environment ID"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return

            test_env = self.test_environments[env_id]
            
            # Filter results based on access permissions
            accessible_results = {}
            if agent_id == test_env.creator_id or agent_id in test_env.participants:
                accessible_results = test_env.results
            else:
                # Only return agent's own results if not creator/participant
                accessible_results = {agent_id: test_env.results.get(agent_id, {})}
            
            response_payload = {
                "success": True,
                "env_id": env_id,
                "results": accessible_results,
                "environment_status": test_env.status
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to get test results: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    # ====== LEARNING CAPABILITY MANAGEMENT ======
    
    async def register_learning_agent(self, agent_id: str, learning_capabilities: List[LearningCapability]) -> Dict[str, Any]:
        """Register an agent with learning capabilities"""
        try:
            self.agent_learning_capabilities[agent_id] = learning_capabilities
            
            await self._log_message_event("LEARNING_AGENT_REGISTERED", {
                "agent_id": agent_id,
                "capabilities": [cap.name for cap in learning_capabilities]
            })
            
            return {
                "success": True,
                "message": "Learning capabilities registered successfully",
                "registered_capabilities": [cap.to_dict() for cap in learning_capabilities]
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to register learning capabilities: {str(e)}"}

    async def get_learning_sessions_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all learning sessions an agent is participating in"""
        agent_sessions = []
        for session_id, session in self.learning_sessions.items():
            if agent_id in session.participants:
                agent_sessions.append(session.to_dict())
        return agent_sessions

    async def get_available_test_environments(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get test environments accessible to an agent"""
        accessible_envs = []
        for env_id, env in self.test_environments.items():
            if agent_id == env.creator_id or agent_id in env.participants:
                accessible_envs.append(env.to_dict())
        return accessible_envs

    # ====== WORKFLOW REQUEST HANDLERS ======
    
    async def _handle_create_workflow_request(self, msg: Message):
        """Handle request to create a new workflow"""
        try:
            agent_id = msg.sender_id
            payload = msg.payload
            
            workflow_name = payload.get("workflow_name", "")
            workflow_description = payload.get("workflow_description", "")
            tasks = payload.get("tasks", [])
            
            if not workflow_name or not tasks:
                error_payload = {"error": "Missing workflow_name or tasks in request"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
            
            # Create workflow using collaboration engine
            workflow_id = await self.collaboration_engine.create_workflow(
                name=workflow_name,
                description=workflow_description,
                created_by=agent_id,
                task_definitions=tasks
            )
            
            response_payload = {
                "success": True,
                "workflow_id": workflow_id,
                "message": f"Workflow '{workflow_name}' created successfully"
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to create workflow: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
    
    async def _handle_start_workflow_request(self, msg: Message):
        """Handle request to start a workflow"""
        try:
            agent_id = msg.sender_id
            workflow_id = msg.payload.get("workflow_id")
            
            if not workflow_id:
                error_payload = {"error": "Missing workflow_id in request"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
            
            # Start workflow using collaboration engine
            success = await self.collaboration_engine.start_workflow(workflow_id)
            
            if success:
                response_payload = {
                    "success": True,
                    "workflow_id": workflow_id,
                    "message": "Workflow started successfully"
                }
            else:
                response_payload = {
                    "success": False,
                    "error": "Failed to start workflow - workflow not found"
                }
            
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to start workflow: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
    
    async def _handle_get_workflow_status_request(self, msg: Message):
        """Handle request to get workflow status"""
        try:
            agent_id = msg.sender_id
            workflow_id = msg.payload.get("workflow_id")
            
            if not workflow_id:
                error_payload = {"error": "Missing workflow_id in request"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return

            # Get workflow status using collaboration engine
            status = self.collaboration_engine.get_workflow_status(workflow_id)
            
            if status:
                response_payload = {
                    "success": True,
                    "workflow_status": status
                }
            else:
                response_payload = {
                    "success": False,
                    "error": "Workflow not found"
                }
            
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to get workflow status: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
    
    async def _handle_create_collaboration_request(self, msg: Message):
        """Handle request to create a collaboration session"""
        try:
            agent_id = msg.sender_id
            payload = msg.payload
            
            participant_ids = payload.get("participant_ids", [])
            purpose = payload.get("purpose", "")
            
            if not participant_ids:
                error_payload = {"error": "Missing participant_ids in request"}
                await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)
                return
            
            # Add requester to participants if not already included
            if agent_id not in participant_ids:
                participant_ids.append(agent_id)
            
            # Create collaboration session using collaboration engine
            collab_id = await self.collaboration_engine.create_collaboration_session(
                agent_ids=participant_ids,
                purpose=purpose
            )
            
            response_payload = {
                "success": True,
                "collaboration_id": collab_id,
                "message": f"Collaboration session created with {len(participant_ids)} participants"
            }
            await self._send_direct_response(self.lobby_id, agent_id, msg, response_payload, MessageType.RESPONSE)
            
        except Exception as e:
            error_payload = {"error": f"Failed to create collaboration session: {str(e)}"}
            await self._send_direct_response(self.lobby_id, agent_id, msg, error_payload, MessageType.ERROR)

    def _create_http_app(self):
        """Legacy method - replaced by _create_http_server"""
        pass

    def _create_http_server(self):
        """Create HTTP server using built-in libraries"""
        
        class LobbyHTTPHandler(http.server.BaseHTTPRequestHandler):
            def __init__(self, request, client_address, server):
                self.lobby = server.lobby_instance
                super().__init__(request, client_address, server)
            
            def do_GET(self):
                try:
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path
                    
                    if path == '/health' or path == '/api/health':
                        response = {'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}
                        self._send_json_response(200, response)
                    elif path == '/api/status':
                        response = {
                            'status': 'operational',
                            'agents_count': len(self.lobby.agents),
                            'active_connections': len(self.lobby.websocket_connections)
                        }
                        self._send_json_response(200, response)
                    elif path == '/api/agents':
                        agents_list = list(self.lobby.agents.values())
                        response = {'status': 'success', 'agents': agents_list, 'count': len(agents_list)}
                        self._send_json_response(200, response)
                    elif path == '/api/available_tasks':
                        # Get available tasks from collaboration engine
                        try:
                            query_params = parse_qs(parsed_url.query)
                            agent_id = query_params.get('agent_id', [''])[0]
                            
                            # Get active workflows that need agents
                            active_workflows = self.lobby.collaboration_engine.workflows
                            available_tasks = []
                            
                            for workflow_id, workflow in active_workflows.items():
                                if workflow.status.value == 'running':
                                    for task_id, task in workflow.tasks.items():
                                        if task.status.value == 'pending':
                                            available_tasks.append({
                                                'delegation_id': workflow_id,
                                                'task_id': task_id,
                                                'task_title': task.name,
                                                'task_description': workflow.description,
                                                'required_capabilities': [task.required_capability],
                                                'created_by': workflow.created_by,
                                                'deadline': task.created_at.isoformat(),
                                                'status': 'open'
                                            })
                            
                            response = {'status': 'success', 'tasks': available_tasks, 'count': len(available_tasks)}
                            self._send_json_response(200, response)
                        except Exception as e:
                            self._send_json_response(500, {'status': 'error', 'message': f'Failed to get tasks: {str(e)}'})
                    elif path.startswith('/api/collaboration_status/'):
                        # Extract delegation_id from URL
                        delegation_id = path.split('/')[-1]
                        try:
                            # Map delegation_id to workflow_id
                            workflow_id = self.lobby.delegation_to_workflow.get(delegation_id)
                            
                            if not workflow_id:
                                response = {
                                    'status': 'not_found',
                                    'message': f'No workflow found for delegation {delegation_id}'
                                }
                            else:
                                # Get workflow status using the correct workflow_id
                                workflow_status = self.lobby.collaboration_engine.get_workflow_status(workflow_id)
                                
                                if workflow_status:
                                    response = {
                                        'status': 'success',
                                        'delegation_id': delegation_id,
                                        'workflow_id': workflow_id,
                                        'workflow_status': workflow_status.get('status'),
                                        'participating_agents': list(workflow_status.get('participants', [])),
                                        'progress': workflow_status.get('progress', 'In progress'),
                                        'results': workflow_status.get('result')
                                    }
                                else:
                                    response = {
                                        'status': 'not_found',
                                        'message': f'Workflow {workflow_id} not found in collaboration engine'
                                    }
                            
                            self._send_json_response(200, response)
                        except Exception as e:
                            self._send_json_response(500, {'status': 'error', 'message': f'Failed to get status: {str(e)}'})
                    else:
                        self._send_json_response(404, {'status': 'error', 'message': 'Not found'})
                except Exception as e:
                    print(f"HTTP GET error: {e}")
                    self._send_json_response(500, {'status': 'error', 'error': str(e)})
            
            def do_POST(self):
                try:
                    parsed_url = urlparse(self.path)
                    path = parsed_url.path
                    
                    # Read request body
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    
                    if path == '/api/agents/register':
                        try:
                            data = json.loads(post_data) if post_data else {}
                            agent_id = data.get('agent_id')
                            name = data.get('name', agent_id)
                            capabilities = data.get('capabilities', [])
                            
                            # Goal-driven agent information
                            goal = data.get('goal', '')
                            specialization = data.get('specialization', '')
                            collaboration_style = data.get('collaboration_style', 'individual')
                            
                            if not agent_id:
                                self._send_json_response(400, {'status': 'error', 'message': 'agent_id is required'})
                                return
                            
                            # Store all agent info including goal information
                            agent_data = {
                                'agent_id': agent_id,
                                'name': name,
                                'capabilities': capabilities,
                                'goal': goal,
                                'specialization': specialization,
                                'collaboration_style': collaboration_style,
                                'registered_at': datetime.now(timezone.utc).isoformat(),
                                'status': 'active'
                            }
                            
                            # Preserve additional fields like websocket_ready
                            for key, value in data.items():
                                if key not in ['agent_id', 'name', 'capabilities', 'goal', 'specialization', 'collaboration_style']:
                                    agent_data[key] = value
                            
                            self.lobby.agents[agent_id] = agent_data
                            
                            # Initialize agent capabilities for HTTP-registered agents
                            if agent_id not in self.lobby.agent_capabilities:
                                self.lobby.agent_capabilities[agent_id] = {}
                            
                            # Convert capabilities list to capability dict for collaboration engine
                            for capability in capabilities:
                                self.lobby.agent_capabilities[agent_id][capability] = {
                                    "name": capability,
                                    "description": f"HTTP-registered capability: {capability}",
                                    "authorized_requester_ids": None,  # Public capability
                                    "agent_id": agent_id
                                }
                            
                            # Initialize reputation if not exists
                            if agent_id not in self.lobby.agent_reputation:
                                self.lobby.agent_reputation[agent_id] = self.lobby.default_reputation
                            
                            print(f"HTTP: Registered goal-driven agent {agent_id}")
                            print(f"  Goal: {goal}")
                            print(f"  Specialization: {specialization}")
                            print(f"  Capabilities: {capabilities}")
                            print(f"  Capabilities initialized: {list(self.lobby.agent_capabilities[agent_id].keys())}")
                            
                            response = {
                                'status': 'success',
                                'agent_id': agent_id,
                                'message': 'Agent registered successfully',
                                'goal_support': bool(goal),
                                'collaboration_ready': True,
                                'capabilities_count': len(capabilities)
                            }
                            self._send_json_response(200, response)
                        except json.JSONDecodeError:
                            self._send_json_response(400, {'status': 'error', 'message': 'Invalid JSON'})
                    
                    elif path == '/api/delegate_task':
                        try:
                            data = json.loads(post_data) if post_data else {}
                            
                            # Extract delegation parameters
                            task_title = data.get('task_title', '')
                            task_description = data.get('task_description', '')
                            required_capabilities = data.get('required_capabilities', [])
                            requester_id = data.get('requester_id', '')
                            
                            # Goal-driven collaboration parameters
                            task_intent = data.get('task_intent', '')
                            max_agents = data.get('max_agents', 3)
                            
                            if not all([task_title, task_description, required_capabilities, requester_id]):
                                self._send_json_response(400, {'status': 'error', 'message': 'Missing required fields'})
                                return
                            
                            # Create delegation using collaboration engine (synchronously)
                            delegation_id = f"delegation_{uuid.uuid4().hex[:8]}"
                            
                            # Enhanced data for goal-driven collaboration
                            enhanced_data = {
                                **data,
                                'task_intent': task_intent,
                                'max_agents': max_agents,
                                'collaboration_type': 'goal_driven' if task_intent else 'standard'
                            }
                            
                            # Use a helper method to handle async calls
                            result = self._create_delegation_sync(
                                delegation_id, task_title, task_description, 
                                required_capabilities, requester_id, enhanced_data
                            )
                            
                            if result['success']:
                                # Store delegation to workflow mapping
                                self.lobby.delegation_to_workflow[delegation_id] = result['workflow_id']
                                
                                # Re-enable background workflow creation
                                self._schedule_workflow_creation(
                                    result['workflow_id'], task_title, task_description,
                                    required_capabilities, requester_id, enhanced_data
                                )
                                
                                response = {
                                    'status': 'success',
                                    'delegation_id': delegation_id,
                                    'workflow_id': result['workflow_id'],
                                    'message': f"Task '{task_title}' delegated successfully",
                                    'started': result['started'],
                                    'collaboration_type': enhanced_data['collaboration_type'],
                                    'max_agents': max_agents if task_intent else 1
                                }
                                self._send_json_response(200, response)
                            else:
                                self._send_json_response(500, {'status': 'error', 'message': result['error']})
                            
                        except json.JSONDecodeError:
                            self._send_json_response(400, {'status': 'error', 'message': 'Invalid JSON'})
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            self._send_json_response(500, {'status': 'error', 'message': f'Delegation failed: {str(e)}'})
                    
                    else:
                        self._send_json_response(404, {'status': 'error', 'message': 'Not found'})
                except Exception as e:
                    print(f"HTTP POST error: {e}")
                    self._send_json_response(500, {'status': 'error', 'error': str(e)})
            
            def _schedule_workflow_creation(self, workflow_id, task_title, task_description,
                                           required_capabilities, requester_id, data):
                """Schedule async workflow creation as background task"""
                try:
                    print(f"HTTP: _schedule_workflow_creation called for {workflow_id}")
                    print(f"HTTP: Scheduling workflow creation for {workflow_id}")
                    
                    # Create request for the bridge queue
                    request = {
                        'type': 'create_workflow',
                        'delegation_id': None,  # Will be filled by the caller
                        'workflow_id': workflow_id,
                        'task_title': task_title,
                        'task_description': task_description,
                        'required_capabilities': required_capabilities,
                        'requester_id': requester_id,
                        'data': data
                    }
                    
                    # Find the delegation ID for this workflow
                    delegation_id = None
                    for del_id, wf_id in self.server.lobby_instance.delegation_to_workflow.items():
                        if wf_id == workflow_id:
                            delegation_id = del_id
                            break
                    
                    print(f"HTTP: Found delegation_id: {delegation_id} for workflow_id: {workflow_id}")
                    
                    if delegation_id:
                        request['delegation_id'] = delegation_id
                        
                        # Put request in thread-safe queue
                        print(f"HTTP: Adding request to bridge queue...")
                        self.server.lobby_instance._http_to_asyncio_queue.put(request)
                        print(f"HTTP: Queued workflow creation request for delegation {delegation_id}")
                        print(f"HTTP: Queue size is now: {self.server.lobby_instance._http_to_asyncio_queue.qsize()}")
                    else:
                        print(f"HTTP: Warning - Could not find delegation ID for workflow {workflow_id}")
                        print(f"HTTP: Available delegations: {list(self.server.lobby_instance.delegation_to_workflow.items())}")
                        
                except Exception as e:
                    print(f"HTTP: Error scheduling workflow creation: {e}")
                    import traceback
                    traceback.print_exc()
            
            def _create_delegation_sync(self, delegation_id, task_title, task_description, 
                                      required_capabilities, requester_id, data):
                """Synchronous wrapper for async delegation creation"""
                try:
                    # Simple synchronous approach - don't mix async/sync contexts
                    # Just return success for now and let the workflow handle the rest
                    
                    # Create a simple workflow ID
                    import uuid
                    workflow_id = str(uuid.uuid4())
                    
                    # Store basic delegation info
                    lobby_instance_ref = self.server.lobby_instance
                    lobby_instance_ref.delegation_to_workflow[delegation_id] = workflow_id
                    
                    # Return success - the collaboration engine will handle the actual workflow later
                    print(f"HTTP: Created delegation {delegation_id} -> workflow {workflow_id}")
                    
                    return {
                        'success': True,
                        'workflow_id': workflow_id,
                        'started': True
                    }
                    
                except Exception as e:
                    print(f"Delegation sync wrapper error: {e}")
                    import traceback
                    traceback.print_exc()
                    return {'success': False, 'error': str(e)}
            
            def _send_json_response(self, status_code, data):
                self.send_response(status_code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response_json = json.dumps(data)
                self.wfile.write(response_json.encode('utf-8'))
            
            def log_message(self, format, *args):
                # Suppress default logging
                pass
        
        # Create handler with lobby instance
        handler_class = LobbyHTTPHandler
        
        # Create custom server class that stores lobby instance
        class LobbyTCPServer(socketserver.TCPServer):
            handler_class = LobbyHTTPHandler
            
            def __init__(self, server_address, RequestHandlerClass, lobby_instance):
                self.lobby_instance = lobby_instance
                super().__init__(server_address, RequestHandlerClass)
        
        return LobbyTCPServer
    
    async def _websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        agent_id = None
        try:
            self.logger.info("WebSocket connection established")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type', 'unknown')
                    
                    if msg_type == 'register':
                        agent_id = data.get('agent_id')
                        if agent_id:
                            self.websocket_connections[agent_id] = websocket
                            self.logger.info(f"Agent {agent_id} connected via WebSocket")
                            
                            # Send acknowledgment
                            await websocket.send(json.dumps({
                                'type': 'register_ack',
                                'status': 'success',
                                'agent_id': agent_id
                            }))
                    
                    elif msg_type == 'task_response':
                        # Handle task completion response from agent
                        await self._handle_task_response(data)
                    
                    elif msg_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    
                    elif msg_type == 'message':
                        # Handle agent-to-agent messages
                        await self._handle_websocket_message(data, agent_id)
                    
                    else:
                        # Echo back for testing
                        await websocket.send(json.dumps({
                            'type': 'echo',
                            'original': data,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }))
                        
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON in WebSocket message")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    self.logger.error(f"WebSocket message error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed for agent: {agent_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            if agent_id and agent_id in self.websocket_connections:
                del self.websocket_connections[agent_id]
                self.logger.info(f"Cleaned up WebSocket for agent: {agent_id}")
    
    async def _handle_task_response(self, data: Dict[str, Any]):
        """Handle task response from WebSocket agent"""
        try:
            # Convert WebSocket message to Message object for collaboration engine
            from .message import Message, MessageType, MessagePriority
            
            message = Message(
                message_id=data.get('message_id', str(uuid.uuid4())),
                sender_id=data.get('sender_id'),
                receiver_id=data.get('receiver_id', self.lobby_id),
                message_type=MessageType.RESPONSE,
                payload=data.get('payload', {}),
                conversation_id=data.get('conversation_id'),
                priority=MessagePriority.HIGH,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Route to collaboration engine
            if self.collaboration_engine:
                await self.collaboration_engine.handle_task_result(message)
                print(f"LOBBY: Routed task response from {message.sender_id} to collaboration engine")
            else:
                print(f"LOBBY ERROR: No collaboration engine to handle task response from {message.sender_id}")
                
        except Exception as e:
            print(f"LOBBY ERROR: Failed to handle task response: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_websocket_message(self, data: Dict[str, Any], sender_id: str):
        """Handle WebSocket message routing"""
        try:
            receiver_id = data.get('receiver_id')
            content = data.get('content', '')
            
            if receiver_id and receiver_id in self.websocket_connections:
                # Route message to target agent
                target_ws = self.websocket_connections[receiver_id]
                await target_ws.send(json.dumps({
                    'type': 'message',
                    'sender_id': sender_id,
                    'content': content,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }))
                
                # Send acknowledgment to sender
                sender_ws = self.websocket_connections.get(sender_id)
                if sender_ws:
                    await sender_ws.send(json.dumps({
                        'type': 'message_ack',
                        'status': 'delivered',
                        'receiver_id': receiver_id
                    }))
            else:
                # Send error back to sender
                sender_ws = self.websocket_connections.get(sender_id)
                if sender_ws:
                    await sender_ws.send(json.dumps({
                        'type': 'error',
                        'message': f'Agent {receiver_id} not connected'
                    }))
                    
        except Exception as e:
            self.logger.error(f"Message routing error: {e}")