print("!!!!!!!!!! EXECUTING LATEST LOBBY.PY !!!!!!!!!!") # VFS: Prominent top-level print

import asyncio
import uuid # For generating tokens
from typing import Dict, List, Any, Optional, Set, cast
import datetime # Added for timestamping logs
import json # Added for structured logging

from .message import Message, MessageType, MessagePriority, MessageValidationError
from .agent import Agent, Capability # Forward declaration for type hint

# --- Constants ---
MAX_FAILED_AUTH_ATTEMPTS = 3
AUTH_LOCKOUT_DURATION_SECONDS = 300 # 5 minutes

LOG_FILE_PATH = "simulation_run.log" # Define log file path

class Lobby:
    def __init__(self, world_state_path="world_state.json"):
        print("!!!!!!!!!! LOBBY __INIT__ CALLED - LATEST VERSION !!!!!!!!!!") # VFS: Prominent init print
        self.agents: Dict[str, Agent] = {}
        # agent_id -> capability_name -> capability_details
        self.agent_capabilities: Dict[str, Dict[str, Capability]] = {} 
        self.registered_agents: Set[str] = set() 
        self.agent_auth_tokens: Dict[str, str] = {} 
        self.lobby_id = "global_lobby" 
        self.failed_auth_attempts: Dict[str, int] = {} 
        self.auth_lockout_timers: Dict[str, asyncio.TimerHandle] = {}
        self._log_file_handle = open(LOG_FILE_PATH, "a", encoding="utf-8")
        self._log_lock = asyncio.Lock()
        # New: Message handling queues and tracking
        self._priority_queues: Dict[MessagePriority, asyncio.PriorityQueue] = {
            priority: asyncio.PriorityQueue() for priority in MessagePriority
        }
        self._pending_acks: Dict[str, asyncio.Event] = {}  # message_id -> Event
        self._message_timeouts: Dict[str, asyncio.TimerHandle] = {}  # message_id -> Timer
        self._agent_types: Dict[str, str] = {}  # agent_id -> agent_type
        self._message_processor_task: Optional[asyncio.Task] = None

        # Reputation System
        self.agent_reputation: Dict[str, float] = {} # agent_id -> reputation_score
        self.agent_interaction_history: Dict[str, List[Dict[str, Any]]] = {} # agent_id -> list of interactions
        self.default_reputation = 100.0
        self.reputation_change_on_success = {"provider": 5.0, "requester": 1.0}
        self.reputation_change_on_failure = {"provider": -10.0, "requester": -0.5} # Minor penalty for failed request initiation
        self.penalty_suspicious_report_reporter = -25.0 # Severe penalty for suspicious reporting
        self.penalty_suspicious_report_provider = -2.0 # Minor penalty for being subject of suspicious report, or none

    async def _log_message_event(self, event_type: str, data: Dict):
        """Logs generic lobby events. 'data' is a dictionary of event-specific details."""
        async with self._log_lock:
            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "event_type": event_type,
                "details": data 
            }
            self._log_file_handle.write(json.dumps(log_entry) + "\n")
            self._log_file_handle.flush()

    async def _log_communication(self, sender_id: str, receiver_id: str, msg_type: MessageType, 
                               payload: Optional[Dict[str, Any]] = None, 
                               conversation_id: Optional[str] = None, 
                               auth_status: str = "N/A"):
        async with self._log_lock:
            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "sender": sender_id,
                "receiver": receiver_id,
                "message_type": msg_type.name,
                "conversation_id": conversation_id,
                "auth_status": auth_status,
                "payload": payload
            }
            self._log_file_handle.write(json.dumps(log_entry) + "\n")
            self._log_file_handle.flush()
            print(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} | {sender_id} -> {receiver_id} | {msg_type.name} | ConvID: {conversation_id} | Auth: {auth_status} | {payload}")

    async def start(self):
        """Start the message processing loop."""
        self._message_processor_task = asyncio.create_task(self._process_message_queues())
        print(f"Lobby message processor started.")

    async def stop(self):
        """Stop the message processing loop."""
        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
        print(f"Lobby message processor stopped.")

    async def _process_message_queues(self):
        """Process messages from priority queues."""
        print(f"LOBBY ({self.lobby_id}): Starting _process_message_queues loop.")
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
                
                # If it's not a REQUEST or if it IS a REQUEST and it was authorized, send it to the agent.
                await self.agents[message.receiver_id].receive_message(message)
            
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

    async def route_message(self, message: Message):
        """Enhanced message routing with priority queues and validation."""
        try:
            # Validate message
            message.validate()
            
            # Authentication check
            if not self._authenticate_message(message):
                # If authentication fails, we should probably send an error response
                # back to the sender. For now, just returning as per original logic for non-REGISTER.
                # For REGISTER messages, _authenticate_message currently always returns True.
                # If it were to return False (e.g., for a banned agent trying to re-register),
                # we might want specific handling.
                print(f"LOBBY ({self.lobby_id}): Authentication failed for message {message.message_id} from {message.sender_id}. Type: {message.message_type.name}")
                # Consider sending an explicit auth error message back to the sender if not a REGISTER message
                if message.message_type != MessageType.REGISTER:
                    await self._send_error_to_agent(message.sender_id, "Authentication failed for your message.", message.message_id)
                return
            
            # Add to appropriate priority queue
            # Store (priority_value, timestamp, message_id, message_object)
            timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
            await self._priority_queues[message.priority].put(
                (message.priority.value, timestamp, message.message_id, message)
            )
            print(f"LOBBY ({self.lobby_id}): Queued message {message.message_id} with priority {message.priority.name}")
            
            # Handle acknowledgment if this is an ACK message
            if message.message_type == MessageType.ACK:
                ack_for = message.payload.get("ack_for")
                if ack_for in self._pending_acks:
                    self._pending_acks[ack_for].set()
                    if ack_for in self._message_timeouts:
                        self._message_timeouts[ack_for].cancel()
                        del self._message_timeouts[ack_for]
        except MessageValidationError as e:
            print(f"Message validation error: {e}")
            if message.sender_id in self.agents:
                error_msg = Message(
                    sender_id=self.lobby_id,
                    receiver_id=message.sender_id,
                    message_type=MessageType.ERROR,
                    payload={"error": f"Message validation failed: {str(e)}"},
                    priority=MessagePriority.HIGH
                )
                await self.agents[message.sender_id].receive_message(error_msg)

    async def register_agent(self, agent: Agent):
        """Enhanced agent registration with type tracking and initial reputation."""
        if agent.agent_id in self.agents:
            await self._log_message_event("AGENT_REGISTRATION_FAILED", {"agent_id": agent.agent_id, "reason": "Already registered"})
            return

            self.agents[agent.agent_id] = agent
            agent.lobby_ref = self

        # Store capabilities
        self.agent_capabilities[agent.agent_id] = {}
        for cap_dict in agent.get_capabilities():
            cap = cast(Capability, cap_dict)
            self.agent_capabilities[agent.agent_id][cap["name"]] = cap

        # Generate and store auth token
        token = str(uuid.uuid4())
        self.agent_auth_tokens[agent.agent_id] = token
        agent.auth_token = token
        self.registered_agents.add(agent.agent_id)

        # Store agent type (derived from agent class name or explicit type)
        agent_type = getattr(agent, "agent_type", agent.__class__.__name__)
        self._agent_types[agent.agent_id] = agent_type

        # Initialize reputation and interaction history
        self.agent_reputation[agent.agent_id] = self.default_reputation
        self.agent_interaction_history[agent.agent_id] = []

        await self._log_message_event("AGENT_REGISTERED", {
            "agent_id": agent.agent_id,
            "agent_type": agent_type,
            "initial_reputation": self.agent_reputation[agent.agent_id],
            "capabilities": agent.get_capabilities()
        })
        print(f"Agent {agent.agent_id} ({agent_type}) successfully registered. Token: {'present' if token else 'None'}. Initial Reputation: {self.agent_reputation[agent.agent_id]}")

    async def _is_request_authorized(self, sender_id: str, target_agent_id: str, capability_name: str) -> bool:
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
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
        # self.world_state[key] = value # world_state was not defined, commented out for now

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

    def close_log_file(self):
        """Closes the log file handle. Should be called on graceful shutdown."""
        if self._log_file_handle and not self._log_file_handle.closed:
            self._log_file_handle.write(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} - Lobby shutting down. Log closed.\n")
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