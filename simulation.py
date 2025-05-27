import asyncio
from typing import List, Dict, Any

from src.core.agent import Agent
from src.core.lobby import Lobby
from src.core.message import Message, MessageType


class KeyAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.has_key = False

    def get_capabilities(self) -> List[str]:
        return ["find_key", "pickup_key", "give_key"]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} with payload {msg.payload}")
        if msg.message_type == MessageType.REQUEST:
            if msg.payload.get("task") == "get_key_status":
                await self.send_message(msg.sender_id, MessageType.RESPONSE, {"has_key": self.has_key})
        elif msg.message_type == MessageType.INFO:
            if msg.payload.get("item_found") == "key_location":
                # In a real scenario, agent would decide to go pick it up
                print(f"{self.agent_id} notes key location: {msg.payload.get('location')}")
                # Simulate picking up the key if found
                if not self.has_key:
                     print(f"{self.agent_id} is attempting to pick up the key.")
                     await self.send_message("lobby", MessageType.ACTION, {"action_type": "pickup_item", "item": "key"})

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        # Simulate finding the key
        await asyncio.sleep(1) # Simulate time to find key
        print(f"{self.agent_id}: I found the key at (x:1, y:1)!")
        self.has_key = True
        self.lobby_ref.update_world_state("key_holder", self.agent_id)
        self.lobby_ref.update_world_state("key_location", "(x:1, y:1)")

        # Inform DoorAgent about having the key
        await self.send_message("door_agent", MessageType.INFO, {"item": "key", "status": "acquired"})

        while True:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                pass # No messages
            await asyncio.sleep(0.1)

class DoorAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.door_opened = False

    def get_capabilities(self) -> List[str]:
        return ["open_door"]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} with payload {msg.payload}")
        if msg.message_type == MessageType.INFO:
            if msg.payload.get("item") == "key" and msg.payload.get("status") == "acquired":
                print(f"{self.agent_id}: Received key from {msg.sender_id}. Attempting to open door.")
                # Simulate opening the door
                await self.send_message("lobby", MessageType.ACTION, {"action_type": "open_item", "item": "door"})
                self.door_opened = True
                self.lobby_ref.update_world_state("door_status", "opened")
                print(f"\n{self.agent_id}: The door is now OPENED! Collaboration successful!\n")

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        while not self.door_opened:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                if self.lobby_ref.get_world_state().get("key_holder") and not self.door_opened:
                    # Proactively ask for the key if we know someone has it
                    key_holder_agent = self.lobby_ref.get_world_state().get("key_holder")
                    if key_holder_agent:
                         print(f"{self.agent_id}: Requesting key from {key_holder_agent}")
                         await self.send_message(key_holder_agent, MessageType.REQUEST, {"task": "request_key"} )

            await asyncio.sleep(0.1) # Check for messages periodically
        print(f"{self.agent_id} exiting run loop as door is opened.")

async def main_simulation():
    lobby = Lobby()

    # Initialize world state
    lobby.update_world_state("key_location", None)
    lobby.update_world_state("key_holder", None)
    lobby.update_world_state("door_status", "closed")

    key_agent = KeyAgent("key_agent")
    door_agent = DoorAgent("door_agent")

    # Manually register agents with the lobby for the simulation
    # In a real system, agents might connect over a network and self-register.
    await lobby.register_agent(key_agent, key_agent.get_capabilities())
    await lobby.register_agent(door_agent, door_agent.get_capabilities())

    # Start agent tasks
    key_agent_task = asyncio.create_task(key_agent.run())
    door_agent_task = asyncio.create_task(door_agent.run())

    # Let the simulation run for a certain amount of time or until a condition is met
    simulation_duration = 10 # seconds
    start_time = asyncio.get_event_loop().time()
    while True:
        await asyncio.sleep(1)
        current_time = asyncio.get_event_loop().time()
        if lobby.get_world_state().get("door_status") == "opened":
            print("Simulation End: Door has been opened.")
            break
        if current_time - start_time > simulation_duration:
            print("Simulation End: Timed out.")
            break
    
    # Cancel tasks and wait for them to finish
    key_agent_task.cancel()
    door_agent_task.cancel()
    try:
        await key_agent_task
    except asyncio.CancelledError:
        print("KeyAgent task cancelled.")
    try:
        await door_agent_task
    except asyncio.CancelledError:
        print("DoorAgent task cancelled.")

    lobby.print_message_log()

if __name__ == "__main__":
    asyncio.run(main_simulation()) 