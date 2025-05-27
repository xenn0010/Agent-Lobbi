import asyncio
import uuid
from typing import List, Dict, Any, Optional

from src.core.agent import Agent, Capability
from src.core.lobby import Lobby
from src.core.message import Message, MessageType


class FactProviderAgent(Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.fact_generation_prompt_template = "Provide a concise fact for the question: "

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "provide_fact",
                "description": "Provides a factual answer to a question using its knowledge base (simulated LLM).",
                "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"fact": {"type": "string"}}},
                "keywords": ["fact", "information", "knowledge", "question", "answer"]
            }
        ]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) payload: {msg.payload}")
        if msg.message_type == MessageType.REQUEST and msg.payload.get("task") == "get_fact":
            question = msg.payload.get("question")
            if not question:
                await self.send_message(msg.sender_id, MessageType.ERROR, {"error": "No question provided"}, conversation_id=msg.conversation_id)
                return

            # Use the mocked LLM to generate a fact
            full_prompt = f"{self.fact_generation_prompt_template}{question}"
            fact = await self._invoke_llm(prompt=full_prompt)
            
            # The mocked LLM returns "LLM mock response to: '[prompt]'", so we'll parse it slightly
            # In a real scenario, the LLM would directly provide the fact.
            parsed_fact = fact.replace(f"LLM mock response to: '{self.fact_generation_prompt_template}", "").replace(question + "'", "").strip()
            if not parsed_fact: # Fallback if parsing fails
                parsed_fact = f"The capital of France is Paris. (Derived from '{question}' by {self.agent_id})"


            response_payload = {"fact": parsed_fact, "original_question": question}
            await self.send_message(msg.sender_id, MessageType.RESPONSE, response_payload, conversation_id=msg.conversation_id)
        elif msg.message_type == MessageType.REGISTER_ACK:
             if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} ({self.get_capabilities()[0]['name']}) is up. Waiting for token.")
        while not self.auth_token: # Wait for registration to be fully acknowledged with token
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


class KnowledgeSeekerAgent(Agent):
    def __init__(self, agent_id: str, questions_to_ask: List[str]):
        super().__init__(agent_id)
        self.questions_to_ask = questions_to_ask
        self.learned_facts: Dict[str, str] = {} # Stores question: fact
        self.pending_fact_requests: Dict[str, str] = {} # Stores conversation_id: question
        self.fact_provider_id: Optional[str] = None

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "seek_knowledge",
                "description": "Seeks factual information from other agents and learns from it.",
                "input_schema": {}, # No direct external input to trigger its core loop
                "output_schema": {},
                "keywords": ["learning", "knowledge_acquisition", "curiosity"]
            }
        ]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) payload: {msg.payload}")
        
        if msg.message_type == MessageType.SERVICES_AVAILABLE and msg.payload.get("discovered_for_capability") == "provide_fact":
            services = msg.payload.get("services_found", [])
            if services:
                self.fact_provider_id = services[0]["agent_id"]
                print(f"{self.agent_id} discovered FactProviderAgent: {self.fact_provider_id}")
                await self._ask_next_question() # Start asking questions
            else:
                print(f"{self.agent_id} could not find any agent with 'provide_fact' capability.")

        elif msg.message_type == MessageType.RESPONSE:
            if msg.conversation_id in self.pending_fact_requests:
                question = self.pending_fact_requests.pop(msg.conversation_id)
                fact = msg.payload.get("fact")
                if fact:
                    self.learned_facts[question] = fact
                    print(f"--- {self.agent_id} LEARNED: Q: '{question}' A: '{fact}' ---")
                    # Try to ask the next question
                    await self._ask_next_question() 
                else:
                    print(f"{self.agent_id} received response for '{question}' but no fact was found in payload: {msg.payload}")
            else:
                print(f"{self.agent_id} received unexpected response for ConvID {msg.conversation_id}")
        
        elif msg.message_type == MessageType.ERROR and msg.sender_id == self.fact_provider_id:
            print(f"{self.agent_id} received error from {self.fact_provider_id}: {msg.payload.get('error')}")
            if msg.conversation_id in self.pending_fact_requests:
                self.pending_fact_requests.pop(msg.conversation_id) # Clear pending request
            await self._ask_next_question() # Try next question

        elif msg.message_type == MessageType.REGISTER_ACK:
             if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")


    async def _ask_next_question(self):
        if not self.fact_provider_id:
            print(f"{self.agent_id}: No fact provider known. Cannot ask questions yet.")
            return

        if self.questions_to_ask:
            question = self.questions_to_ask.pop(0) # Get the next question from the list
            if question in self.learned_facts:
                print(f"{self.agent_id} already knows the answer to: '{question}'. It is: '{self.learned_facts[question]}'. Skipping.")
                await self._ask_next_question() # Ask the next one
                return

            print(f"{self.agent_id} is asking FactProvider ({self.fact_provider_id}) the question: '{question}'")
            conv_id = str(uuid.uuid4())
            self.pending_fact_requests[conv_id] = question
            await self.send_message(
                receiver_id=self.fact_provider_id,
                message_type=MessageType.REQUEST,
                payload={"task": "get_fact", "question": question},
                conversation_id=conv_id
            )
        else:
            print(f"{self.agent_id} has asked all its questions. Current knowledge: {len(self.learned_facts)} facts.")
            print(f"Learned facts by {self.agent_id}: {self.learned_facts}")


    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} is up. Waiting for token.")
        while not self.auth_token: # Wait for registration to be fully acknowledged with token
            await asyncio.sleep(0.1)
        print(f"{self.agent_id} token confirmed. Operational.")

        # Discover the FactProviderAgent
        await asyncio.sleep(0.5) # Give lobby a moment
        discovery_payload = {"capability_name": "provide_fact"}
        print(f"{self.agent_id} attempting to discover service by NAME: 'provide_fact'")
        # We'll use a conversation ID for discovery too, to track it if needed, though not strictly necessary for this agent's logic
        await self.send_message("lobby", MessageType.DISCOVER_SERVICES, discovery_payload, conversation_id=str(uuid.uuid4()))
        
        # The rest of the logic (asking questions) is triggered by receiving SERVICES_AVAILABLE

        while True:
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                # If all questions are asked and no more pending requests, it can just idle or exit
                if not self.questions_to_ask and not self.pending_fact_requests:
                    print(f"{self.agent_id} has completed its tasks and is now idle.")
                    await asyncio.sleep(5) # Idle for a bit then break for simulation end
                    break 
            await asyncio.sleep(0.1)

async def main_knowledge_sharing_simulation():
    lobby = Lobby()

    # Agents
    provider_agent = FactProviderAgent(agent_id="fact_master_007")
    
    seeker_agent_alice = KnowledgeSeekerAgent(
        agent_id="learner_alice", 
        questions_to_ask=[
            "What is the capital of France?",
            "How tall is Mount Everest?",
            "What is the chemical symbol for water?"
        ]
    )
    seeker_agent_bob = KnowledgeSeekerAgent(
        agent_id="learner_bob",
        questions_to_ask=[
            "What is the capital of France?", # Bob will ask one same question as Alice
            "Who painted the Mona Lisa?"
        ]
    )

    all_agents = [provider_agent, seeker_agent_alice, seeker_agent_bob]

    for agent in all_agents:
        # In this simulation, we're not focusing on the unauthenticated message scenario.
        # So, we'll directly register and let the run() method wait for token confirmation.
        await lobby.register_agent(agent) # Capabilities are fetched by Lobby during registration

    tasks = [asyncio.create_task(agent.run()) for agent in all_agents]

    simulation_duration = 15  # seconds
    print(f"--- Starting Knowledge Sharing Simulation for {simulation_duration} seconds ---")
    
    # Let tasks run, but also check if seekers have completed their work
    start_time = asyncio.get_event_loop().time()
    while True:
        await asyncio.sleep(1)
        if all(not agent.questions_to_ask and not agent.pending_fact_requests for agent in [seeker_agent_alice, seeker_agent_bob]):
            print("--- All seeker agents have completed their tasks. ---")
            break
        if asyncio.get_event_loop().time() - start_time > simulation_duration:
            print("--- Simulation Time Ended (Timeout) ---")
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
    print(f"
--- Final Learned Facts by {seeker_agent_alice.agent_id}: ---")
    for q, a in seeker_agent_alice.learned_facts.items():
        print(f"  Q: {q}
  A: {a}")
    print(f"
--- Final Learned Facts by {seeker_agent_bob.agent_id}: ---")
    for q, a in seeker_agent_bob.learned_facts.items():
        print(f"  Q: {q}
  A: {a}")

if __name__ == "__main__":
    asyncio.run(main_knowledge_sharing_simulation()) 