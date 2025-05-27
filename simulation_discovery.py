import asyncio
import uuid # Import uuid for conversation IDs
from typing import List, Dict, Any, Optional

from src.core.agent import Agent, Capability
from src.core.lobby import Lobby
from src.core.message import Message, MessageType


class SecureServiceProviderAgent(Agent):
    def __init__(self, agent_id: str, service_name: str, service_description: str, service_keywords: List[str], required_inputs: List[str]):
        self.service_name = service_name
        self.service_description = service_description
        self.service_keywords = service_keywords
        self.required_inputs = required_inputs # e.g., ["project_details", "distribution_model"]
        self.pending_clarifications: Dict[str, Dict[str, Any]] = {} # Store partial requests by conversation_id
        super().__init__(agent_id)

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": self.service_name,
                "description": self.service_description,
                "input_schema": {"type": "object", "properties": {key: {"type": "string"} for key in self.required_inputs}},
                "output_schema": {"type": "object", "properties": {"review_summary": {"type": "string"}, "recommendations": {"type": "array"}}},
                "keywords": self.service_keywords
            }
        ]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) AuthTokenPresent: {bool(msg.auth_token)} payload: {msg.payload}")
        
        if msg.message_type == MessageType.REQUEST:
            conv_id = msg.conversation_id or str(uuid.uuid4()) # Use existing or create new conv_id
            original_requester = msg.sender_id
            current_payload = msg.payload

            # Check if this is a response to a clarification we asked for
            if conv_id in self.pending_clarifications and "clarification_response" in current_payload:
                stored_request = self.pending_clarifications[conv_id]
                # Merge new clarification with stored data
                stored_request["data"].update(current_payload["clarification_response"])
                print(f"{self.agent_id} received clarification for ConvID {conv_id}: {current_payload['clarification_response']}")
                
                # Now check if all required inputs are present in the merged data
                missing_inputs = [req for req in self.required_inputs if req not in stored_request["data"]]
                if not missing_inputs:
                    print(f"{self.agent_id} all info received for ConvID {conv_id}. Processing final review.")
                    # Simulate final processing
                    await asyncio.sleep(0.5)
                    final_review_summary = f"Final review for {stored_request['data'].get('project_details', 'N/A')} (dist: {stored_request['data'].get('distribution_model', 'N/A')}) completed."
                    recommendations = ["Secure all endpoints."]
                    final_response_payload = {"review_summary": final_review_summary, "recommendations": recommendations}
                    await self.send_message(stored_request["original_requester"], MessageType.RESPONSE, final_response_payload, conversation_id=conv_id)
                    del self.pending_clarifications[conv_id] # Clean up
                else:
                    # Should not happen if clarification was specific, but as a fallback:
                    print(f"{self.agent_id} still missing inputs for ConvID {conv_id} after clarification: {missing_inputs}. This is unexpected.")
                    # Potentially send another clarification request or an error

            else: # This is an initial request for the service
                # Check for all required inputs in the initial request
                missing_inputs = [req for req in self.required_inputs if req not in current_payload]
                if not missing_inputs:
                    print(f"{self.agent_id} all info received in initial request for ConvID {conv_id}. Processing.")
                    await asyncio.sleep(0.5) # Simulate work
                    review_summary = f"Initial review for {current_payload.get('project_details', 'N/A')} completed."
                    response_payload = {"review_summary": review_summary, "recommendations": ["Initial checks OK."]}
                    await self.send_message(original_requester, MessageType.RESPONSE, response_payload, conversation_id=conv_id)
                else:
                    # Information is missing, ask for clarification
                    print(f"{self.agent_id} missing inputs for ConvID {conv_id}: {missing_inputs}. Requesting clarification.")
                    self.pending_clarifications[conv_id] = {"original_requester": original_requester, "data": current_payload.copy()}
                    clarification_request_payload = {"missing_input": missing_inputs[0]} # Ask for the first missing input
                    await self.send_message(original_requester, MessageType.REQUEST, clarification_request_payload, conversation_id=conv_id)
        
        elif msg.message_type == MessageType.REGISTER_ACK:
            if msg.payload.get("status") == "success_registered_finalized" and self.auth_token:
                print(f"{self.agent_id} successfully registered and token confirmed.")
            else:
                print(f"{self.agent_id} received REGISTER_ACK: {msg.payload}, token expected.")
        elif msg.message_type == MessageType.ERROR:
            print(f"ERROR for {self.agent_id}: {msg.payload.get('error')}")

    async def send_message(self, receiver_id: str, message_type: MessageType, payload: Optional[Dict[str, Any]] = None, conversation_id: Optional[str] = None):
        """Override to include conversation_id easily."""
        if not self.lobby_ref:
            print(f"Error: Agent {self.agent_id} not connected to a lobby.")
            return
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload or {},
            conversation_id=conversation_id
        )
        await self.lobby_ref.route_message(msg)

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} is up, offering: {self.service_name}. Waiting for token to be confirmed via REGISTER_ACK.")
        while not self.auth_token:
            await asyncio.sleep(0.1) # Wait until token is set by receive_message handling REGISTER_ACK
        print(f"{self.agent_id} token confirmed. Fully operational.")

        while True:
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.1)

class SecureServiceRequesterAgent(Agent):
    def __init__(self, agent_id: str, service_to_find_name: str, initial_request_payload: Dict[str, Any], clarifications_to_provide: Dict[str, Any]):
        self.service_to_find_name = service_to_find_name
        self.initial_request_payload = initial_request_payload
        self.clarifications_to_provide = clarifications_to_provide # e.g., {"distribution_model": "Commercial Proprietary"}
        self.active_conversation_id: Optional[str] = None
        self.request_sent_for_conv: Dict[str, bool] = {}
        self.discovered_provider_id: Optional[str] = None
        super().__init__(agent_id)

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "request_secure_services", 
                "description": "Can discover and request services securely, and handle clarification.",
                "input_schema": {},
                "output_schema": {},
                "keywords": ["client", "requester", "secure", "multi-step"]
            }
        ]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) AuthTokenPresent: {bool(msg.auth_token)} payload: {msg.payload}")
        conv_id = msg.conversation_id

        if msg.message_type == MessageType.SERVICES_AVAILABLE:
            if self.active_conversation_id and self.request_sent_for_conv.get(self.active_conversation_id):
                 return # Already processing a discovery for the active conversation

            services_found = msg.payload.get("services_found", [])
            if services_found:
                provider_info = services_found[0]
                self.discovered_provider_id = provider_info["agent_id"]
                print(f"{self.agent_id} discovered provider {self.discovered_provider_id} for '{self.service_to_find_name}'.")
                
                # Start a new conversation or use existing if somehow set
                self.active_conversation_id = self.active_conversation_id or str(uuid.uuid4())
                print(f"{self.agent_id} initiating task with ConvID: {self.active_conversation_id}")
                await self.send_message(self.discovered_provider_id, MessageType.REQUEST, self.initial_request_payload, conversation_id=self.active_conversation_id)
                self.request_sent_for_conv[self.active_conversation_id] = True
            else:
                print(f"{self.agent_id} could not find any providers for service: {self.service_to_find_name}")
        
        elif msg.message_type == MessageType.REQUEST: # This is a clarification request from the provider
            if not conv_id:
                print(f"{self.agent_id} received a REQUEST without a conversation_id. Ignoring.")
                return
            
            missing_input = msg.payload.get("missing_input")
            if missing_input and missing_input in self.clarifications_to_provide:
                clarification_payload = {"clarification_response": {missing_input: self.clarifications_to_provide[missing_input]}}
                print(f"{self.agent_id} providing clarification for {missing_input} (ConvID: {conv_id}).")
                await self.send_message(msg.sender_id, MessageType.RESPONSE, clarification_payload, conversation_id=conv_id)
            else:
                print(f"{self.agent_id} received unhandled clarification request or missing info for ConvID {conv_id}: {msg.payload}")

        elif msg.message_type == MessageType.RESPONSE: # This is the final response from the provider
            if not conv_id:
                print(f"{self.agent_id} received a RESPONSE without a conversation_id. Logging.")
            print(f"--- {self.agent_id} (ConvID: {conv_id}) received FINAL SECURE RESPONSE: {msg.payload} ---")
            if conv_id and conv_id in self.request_sent_for_conv:
                del self.request_sent_for_conv[conv_id] # Conversation complete
            if conv_id == self.active_conversation_id:
                self.active_conversation_id = None # Reset for potential new tasks

        elif msg.message_type == MessageType.REGISTER_ACK:
            if msg.payload.get("status") == "success_registered_finalized" and self.auth_token:
                print(f"{self.agent_id} successfully registered and token confirmed.")
            else:
                print(f"{self.agent_id} received REGISTER_ACK: {msg.payload}, token expected.")
        elif msg.message_type == MessageType.ERROR:
            print(f"ERROR for {self.agent_id}: {msg.payload.get('error')}")

    async def send_message(self, receiver_id: str, message_type: MessageType, payload: Optional[Dict[str, Any]] = None, conversation_id: Optional[str] = None):
        """Override to include conversation_id easily."""
        if not self.lobby_ref:
            print(f"Error: Agent {self.agent_id} not connected to a lobby.")
            return
        # Use the provided conversation_id or the agent's active one
        effective_conv_id = conversation_id or self.active_conversation_id
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload or {},
            conversation_id=effective_conv_id
        )
        await self.lobby_ref.route_message(msg)

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} is up. Waiting for token to be confirmed via REGISTER_ACK.")
        while not self.auth_token:
            await asyncio.sleep(0.1) # Wait until token is set
        print(f"{self.agent_id} token confirmed. Fully operational.")
        
        await asyncio.sleep(0.5) # Stagger discovery slightly
        if not self.active_conversation_id: # Only discover if not already in a conversation
            discovery_payload = {"capability_name": self.service_to_find_name}
            print(f"{self.agent_id} attempting to discover service by NAME: '{self.service_to_find_name}'")
            await self.send_message("lobby", MessageType.DISCOVER_SERVICES, discovery_payload)
            # The rest of the interaction (sending initial request) happens upon receiving SERVICES_AVAILABLE

        while True:
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.1)

async def main_secure_conversational_simulation():
    lobby = Lobby()

    # Service Provider (Legal Agent)
    legal_service_agent = SecureServiceProviderAgent(
        agent_id="secure_legal_001", 
        service_name="SecureLicenseReview", 
        service_description="Provides license reviews, requires auth, may need clarification.",
        service_keywords=["secure", "legal", "license"],
        required_inputs=["project_details", "licenses_list", "distribution_model"] # It needs these three things
    )

    # Service Requester (Project Manager Agent)
    # This PM will initially forget to provide the 'distribution_model'
    project_manager_beta = SecureServiceRequesterAgent(
        agent_id="secure_pm_beta",
        service_to_find_name="SecureLicenseReview",
        initial_request_payload={"project_details": "Project Phoenix", "licenses_list": ["MIT", "Apache-2.0", "GPL-3.0"]},
        clarifications_to_provide={"distribution_model": "Commercial SaaS Offering"} # This is what it will provide when asked
    )
    
    # Another Requester that provides all info upfront
    project_manager_gamma = SecureServiceRequesterAgent(
        agent_id="secure_pm_gamma",
        service_to_find_name="SecureLicenseReview",
        initial_request_payload={
            "project_details": "Project Griffin", 
            "licenses_list": ["BSD-3-Clause", "LGPL-2.1"], 
            "distribution_model": "Internal Corporate Use Only"
        },
        clarifications_to_provide={} # Won't be asked for clarification
    )

    all_agents = [legal_service_agent, project_manager_beta, project_manager_gamma]

    for agent in all_agents:
        await lobby.register_agent(agent)

    tasks = []
    for agent in all_agents:
        if agent.agent_id == "secure_pm_beta":
            # Let rogue agent try to send a message quickly before its token might be fully processed (or simulate no token)
            async def run_rogue_then_normal():
                print(f"{agent.agent_id} is up. Attempting to operate without waiting for token confirmation.")
                # This agent's send_message will initially use self.auth_token which is None
                await agent.send_message("lobby", MessageType.DISCOVER_SERVICES, {"capability_name": "SecureLicenseReview"})
                await asyncio.sleep(0.5) # Give time for the unauthenticated message to be processed and rejected
                await agent.run() # Then proceed with normal run loop (which waits for token)
            tasks.append(asyncio.create_task(run_rogue_then_normal()))
        else:
            tasks.append(asyncio.create_task(agent.run()))

    simulation_duration = 10 # seconds
    await asyncio.sleep(simulation_duration)

    print("\n--- Simulation Time Ended ---")
    for task in tasks:
        task.cancel()
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        agent_name = all_agents[i].agent_id
        if isinstance(result, asyncio.CancelledError):
            print(f"Task for agent {agent_name} cancelled gracefully.")
        elif isinstance(result, Exception):
            print(f"Task for agent {agent_name} raised an exception: {result}")

    lobby.print_message_log()

if __name__ == "__main__":
    asyncio.run(main_secure_conversational_simulation()) 