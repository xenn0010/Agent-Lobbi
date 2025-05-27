import asyncio
from typing import List, Dict, Any, Optional
import uuid

from ..core.agent import Agent, Capability
from ..core.message import Message, MessageType

class SimpleGreeterAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        # Instructions are implicitly defined by the agent's purpose and how it uses its tools
        # and the prompts it sends to the LLM.

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "greet_user",
                "description": "Receives a name and returns a personalized greeting.",
                "input_schema": {"type": "object", "properties": {
                    "name": {"type": "string", "description": "The name of the person to greet."}
                }},
                "output_schema": {"type": "object", "properties": {
                    "greeting": {"type": "string", "description": "The personalized greeting."}
                }},
                "keywords": ["greeting", "hello", "welcome"]
            }
        ]

    async def _generate_greeting_tool(self, name: str) -> str:
        """
        Tool: Generates a polite and slightly creative greeting for a given name using the LLM.
        """
        prompt = f"You are a friendly assistant. Generate a polite and creative greeting for the name '{name}'. The greeting should be a single sentence. For example, if the name is 'World', a good greeting might be 'Hello World, it's a pleasure to meet you!' or 'Greetings World, wishing you a fantastic day!'. Respond with ONLY the greeting itself."
        
        print(f"{self.agent_id}: Using LLM to generate greeting for '{name}'. Prompt: '{prompt}'")
        greeting = await self._invoke_llm(prompt=prompt)
        # Basic cleanup for LLM response
        greeting = greeting.strip().strip('"') 
        if not greeting: # Fallback
            greeting = f"Hello {name}, nice to meet you!"
        return greeting

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) payload: {msg.payload}")

        if msg.message_type == MessageType.REQUEST and msg.payload.get("capability_name") == "greet_user":
            name_to_greet = msg.payload.get("name")
            if not name_to_greet:
                error_payload = {"error": "'name' not provided in payload for greet_user capability."}
                await self.send_message(msg.sender_id, MessageType.ERROR, error_payload, conversation_id=msg.conversation_id)
                return

            print(f"{self.agent_id}: Received request to greet '{name_to_greet}'.")
            
            generated_greeting = await self._generate_greeting_tool(name_to_greet)
            
            response_payload = {"greeting": generated_greeting}
            print(f"{self.agent_id}: Sending greeting response for '{name_to_greet}': {response_payload}")
            await self.send_message(msg.sender_id, MessageType.RESPONSE, response_payload, conversation_id=msg.conversation_id)
        
        elif msg.message_type == MessageType.REGISTER_ACK:
             if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")


    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} (SimpleGreeterAgent) is up, awaiting requests. Waiting for token.")
        while not self.auth_token: # Wait until token is received
            await asyncio.sleep(0.1)
        print(f"{self.agent_id} token confirmed. Operational.")

        while True: 
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                pass 
            await asyncio.sleep(0.1)

# To test this agent, we'll need a simple simulation script.
# We can create a 'simulation_greeter.py' similar to 'simulation_item_search.py'
# that registers this agent and a 'UserInterfaceAgent' that sends it a greeting request. 