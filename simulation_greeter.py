import asyncio
import uuid
from typing import List, Dict, Any, Optional

from src.core.agent import Agent, Capability
from src.core.lobby import Lobby
from src.core.message import Message, MessageType
from src.test_agents.simple_greeter_agent import SimpleGreeterAgent # Import the new agent

class GreeterTestUI(Agent):
    """A simple UI agent to test the SimpleGreeterAgent."""
    def __init__(self, agent_id: str, name_to_greet: str):
        super().__init__(agent_id)
        self.name_to_greet = name_to_greet
        self.greeter_agent_id: Optional[str] = None
        self.active_discovery_conv_id: Optional[str] = None
        self.active_greeting_request_conv_id: Optional[str] = None
        self.final_greeting: Optional[str] = None
        self.task_initiated = False
        self.task_completed = False

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "test_greeter_ui",
                "description": "Initiates a greeting request to a SimpleGreeterAgent.",
                "input_schema": {}, 
                "output_schema": {},
                "keywords": ["test_ui", "greeting_test"]
            }
        ]

    async def _initiate_greeting_sequence(self):
        if self.task_initiated: return
        self.task_initiated = True

        print(f"{self.agent_id}: Attempting to discover SimpleGreeterAgent (capability: 'greet_user').")
        self.active_discovery_conv_id = str(uuid.uuid4())
        discovery_payload = {"capability_name": "greet_user"}
        await self.send_message("lobby", MessageType.DISCOVER_SERVICES, discovery_payload, conversation_id=self.active_discovery_conv_id)

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) payload: {msg.payload}")

        if msg.message_type == MessageType.SERVICES_AVAILABLE and msg.conversation_id == self.active_discovery_conv_id:
            self.active_discovery_conv_id = None
            services = msg.payload.get("services_found", [])
            if services:
                self.greeter_agent_id = services[0]["agent_id"]
                print(f"{self.agent_id}: Discovered SimpleGreeterAgent: {self.greeter_agent_id}")
                
                self.active_greeting_request_conv_id = str(uuid.uuid4())
                request_payload = {
                    "capability_name": "greet_user", # Specify the capability being requested
                    "name": self.name_to_greet
                }
                print(f"{self.agent_id}: Requesting greeting for '{self.name_to_greet}' from {self.greeter_agent_id}.")
                await self.send_message(
                    self.greeter_agent_id,
                    MessageType.REQUEST,
                    request_payload,
                    conversation_id=self.active_greeting_request_conv_id
                )
            else:
                print(f"{self.agent_id}: ERROR - Could not find any SimpleGreeterAgent. Task aborted.")
                self.final_greeting = "Error: SimpleGreeterAgent not found"
                self.task_completed = True

        elif msg.message_type == MessageType.RESPONSE and msg.conversation_id == self.active_greeting_request_conv_id:
            self.active_greeting_request_conv_id = None
            greeting_received = msg.payload.get("greeting", "Error: No greeting in response.")
            print(f"--- {self.agent_id} RECEIVED GREETING ---: {greeting_received}")
            self.final_greeting = greeting_received
            self.task_completed = True
        
        elif msg.message_type == MessageType.ERROR and msg.sender_id == self.greeter_agent_id and msg.conversation_id == self.active_greeting_request_conv_id:
            error_detail = msg.payload.get('error', 'No specific error detail provided.')
            print(f"{self.agent_id}: ERROR - Received error from SimpleGreeterAgent: {error_detail}")
            self.final_greeting = f"Error from Greeter: {error_detail}"
            self.task_completed = True

        elif msg.message_type == MessageType.REGISTER_ACK:
            if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} is up. Waiting for token.")
        while not self.auth_token: # Wait for token
            await asyncio.sleep(0.1)
        print(f"{self.agent_id} token confirmed. Operational.")

        await asyncio.sleep(0.2) # Give other agents a moment
        await self._initiate_greeting_sequence()

        while not self.task_completed:
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                if self.task_completed: break
            await asyncio.sleep(0.1)
        
        print(f"--- {self.agent_id} Final Greeting Summary ---")
        if self.final_greeting:
            print(f"  UI requested greeting for: '{self.name_to_greet}'")
            print(f"  Greeting received: '{self.final_greeting}'")
        else:
            print(f"  Task did not complete or no greeting received.")
        print(f"--- {self.agent_id} Finished ---")

async def main_greeter_simulation():
    lobby = Lobby()

    test_ui_agent = GreeterTestUI(
        agent_id="greeter_ui_001",
        name_to_greet="Galaxy Explorer"
    )
    greeter_agent = SimpleGreeterAgent(agent_id="simple_greeter_A1")

    all_agents = [test_ui_agent, greeter_agent]

    for agent in all_agents:
        await lobby.register_agent(agent)

    tasks = [asyncio.create_task(agent.run()) for agent in all_agents]

    simulation_duration = 15 
    print(f"--- Starting Greeter Simulation for {simulation_duration} seconds ---")
    
    start_time = asyncio.get_event_loop().time()
    while True:
        await asyncio.sleep(1)
        if test_ui_agent.task_completed: 
            print("--- GreeterTestUI has completed its task. Ending simulation early. ---")
            break
        if asyncio.get_event_loop().time() - start_time > simulation_duration:
            print("--- Greeter Simulation Time Ended (Timeout) ---")
            break
            
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
    asyncio.run(main_greeter_simulation()) 