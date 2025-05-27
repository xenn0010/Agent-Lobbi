from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, TypedDict
import asyncio
import aiohttp
import json
import random
import os # Import os
from dotenv import load_dotenv # Import dotenv

from .message import Message, MessageType

# Load environment variables from .env file at the module level
load_dotenv()

class Capability(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    keywords: List[str]
    authorized_requester_ids: Optional[List[str]] # ADDED for granular authorization

class Agent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self.lobby_ref: Optional[Any] = None # Will be set by Lobby
        self.knowledge: Dict[str, Any] = {} # Simple knowledge base
        self.advertised_capabilities: List[Capability] = self.get_capabilities()
        self.auth_token: Optional[str] = None # ADDED to store the token
        self.ollama_url = "http://localhost:11434/api/generate"  # Default Ollama API endpoint
        self.ollama_model = "gemma:2b"  # Changed default model to gemma:2b
        # agent_type should be defined in subclasses for better Lobby tracking
        self.agent_type: str = self.__class__.__name__ 

    async def send_message(self, receiver_id: str, msg_type: MessageType, payload: Dict[str, Any] = None, conversation_id: Optional[str] = None):
        if payload is None:
            payload = {}
        
        # Automatically use the Lobby's actual ID if the agent is trying to send to the generic "lobby"
        actual_receiver_id = receiver_id
        if receiver_id == "lobby" and self.lobby_ref and hasattr(self.lobby_ref, 'lobby_id'):
            actual_receiver_id = self.lobby_ref.lobby_id

        message = Message(
            sender_id=self.agent_id,
            receiver_id=actual_receiver_id, # Use actual_receiver_id
            message_type=msg_type,
            payload=payload,
            auth_token=self.auth_token, 
            conversation_id=conversation_id
        )
        if self.lobby_ref:
            await self.lobby_ref.route_message(message)
        else:
            print(f"Warning: Agent {self.agent_id} has no lobby reference to send message.")

    async def receive_message(self, msg: Message):
        # Store token if received in REGISTER_ACK
        if msg.message_type == MessageType.REGISTER_ACK and msg.payload.get("status") == "success_registered_finalized":
            received_token = msg.payload.get("auth_token")
            if received_token:
                self.auth_token = received_token
                print(f"Agent {self.agent_id} received and stored auth_token.")
            else:
                print(f"Agent {self.agent_id} received REGISTER_ACK but no token was found in payload.")
        elif msg.message_type == MessageType.ERROR and msg.payload.get("error") and "Authentication failed" in msg.payload.get("error", ""):
            print(f"CRITICAL AUTH ERROR for Agent {self.agent_id}: {msg.payload.get('error')}. Agent might be compromised or lobby token mismatch.")
            # Potentially, the agent could try to re-register or stop sending messages.
            # For now, it will just log the critical error.

        await self._message_queue.put(msg)

    @abstractmethod
    async def process_incoming_message(self, msg: Message):
        """Logic for how the agent reacts to a message."""
        pass

    @abstractmethod
    async def run(self):
        """Main loop for the agent's operation."""
        pass

    async def register_with_lobby(self, lobby_ref: 'Lobby'): # Type hint for Lobby
        self.lobby_ref = lobby_ref
        self.advertised_capabilities = self.get_capabilities()
        
        # Ensure the REGISTER message targets the Lobby's actual ID
        lobby_actual_id = "lobby" # Default, will be overridden if lobby_ref has specific ID
        if self.lobby_ref and hasattr(self.lobby_ref, 'lobby_id'):
            lobby_actual_id = self.lobby_ref.lobby_id

        payload = {"capabilities": self.advertised_capabilities}
        # REGISTER message itself does not send a token; token is received in REGISTER_ACK
        register_message = Message(
            sender_id=self.agent_id,
            receiver_id=lobby_actual_id, # Use the Lobby's actual ID
            message_type=MessageType.REGISTER,
            payload=payload,
            auth_token=None # No token for initial registration message
        )
        # The Agent class itself should not call lobby.register_agent().
        # The simulation script or main environment should call lobby.register_agent(agent_instance) first,
        # which sets agent.lobby_ref and agent.auth_token.
        # Then, the agent's run() method can call self.register_with_lobby() which actually SENDS the REGISTER message.
        # This current structure is a bit mixed up.
        # For now, we assume lobby_ref is set and we send the message.
        if self.lobby_ref:
            await self.lobby_ref.route_message(register_message)
        else:
            print(f"ERROR: Agent {self.agent_id} cannot send REGISTER message, no lobby_ref.")

    @abstractmethod
    def get_capabilities(self) -> List[Capability]:
        pass

    async def _report_task_outcome(
        self,
        provider_agent_id: str,
        capability_name: str,
        status: str, # "success" or "failure"
        details: str,
        original_conversation_id: Optional[str]
    ):
        """Helper method to send a TASK_OUTCOME_REPORT to the Lobby."""
        if not self.lobby_ref:
            print(f"{self.agent_id} ({self.agent_type}): Cannot report task outcome, no lobby reference.")
            return

        if status not in ["success", "failure"]:
            print(f"{self.agent_id} ({self.agent_type}): Invalid status '{status}' for task outcome report. Must be 'success' or 'failure'.")
            # Fallback to failure if status is invalid to ensure a report is made, though this indicates an agent logic error.
            status = "failure"
            details = f"Internal Error: Invalid status provided to _report_task_outcome. Original detail: {details}"
            
        report_payload = {
            "provider_agent_id": provider_agent_id,
            "capability_name": capability_name,
            "status": status,
            "details": details
        }
        
        print(f"{self.agent_id} ({self.agent_type}): Sending TASK_OUTCOME_REPORT to lobby about provider '{provider_agent_id}' for capability '{capability_name}'. Status: {status}")
        await self.send_message(
            receiver_id="lobby", 
            msg_type=MessageType.TASK_OUTCOME_REPORT, 
            payload=report_payload, 
            conversation_id=original_conversation_id # Use original conv_id for context in Lobby
        )

    async def _invoke_llm(self, prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None, temperature: float = 0.2) -> str:
        """
        Makes a call to the Ollama API using the specified model.
        Includes a temperature parameter for controlling response randomness.
        Falls back to a mock response if the API call fails.
        """
        print(f"Agent {self.agent_id} is invoking Ollama LLM with prompt: '{prompt}', temp: {temperature}")
        
        try:
            # Prepare the request payload for Ollama
            request_data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            # If there's conversation history, format it for the model
            if conversation_history:
                print(f"Agent {self.agent_id} LLM call includes conversation history of {len(conversation_history)} turns.")
                formatted_history = "\n".join([f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in conversation_history])
                request_data["prompt"] = f"{formatted_history}\n\nuser: {prompt}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.ollama_url, json=request_data) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        return response_json.get("response", f"Error: Unexpected Ollama response format")
                    else:
                        error_text = await response.text()
                        print(f"Agent {self.agent_id} Ollama API error ({response.status}): {error_text}")
                        # Fall back to mock response on error
                        return f"[Ollama API Error - Fallback Mock] Response to: '{prompt}'"
                        
        except Exception as e:
            print(f"Agent {self.agent_id} exception when calling Ollama API: {str(e)}")
            # Fall back to mock response on exception
            return f"[Ollama API Exception - Fallback Mock] Response to: '{prompt}'"

    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a web search using the Google Custom Search API.
        Requires GOOGLE_API_KEY and GOOGLE_CSE_ID to be set in the .env file.
        """
        print(f"Agent {self.agent_id} attempting REAL web search (Google Custom Search) for: '{query}'")

        api_key = os.getenv("GOOGLE_API_KEY")
        cse_id = os.getenv("GOOGLE_CSE_ID")

        if not api_key or not cse_id:
            print(f"Agent {self.agent_id}: ERROR - GOOGLE_API_KEY or GOOGLE_CSE_ID not found in .env file. Falling back to mock search.")
            # Fallback to a very simple mock response if keys are missing
            return [
                {"title": f"Mock Result for {query}", "link": "http://example.com/mock", "snippet": "API key or CSE ID missing."}
            ]

        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": 5  # Requesting 5 search results
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    search_data = await response.json()
                    
                    results = []
                    if "items" in search_data:
                        for item in search_data["items"]:
                            results.append({
                                "title": item.get("title", "No Title"),
                                "link": item.get("link", "No Link"),
                                "snippet": item.get("snippet", "No Snippet")
                            })
                    
                    if not results:
                        print(f"Agent {self.agent_id}: Google Search - No results found for '{query}'.")
                        return [{"title": "No results found", "link": "", "snippet": f"Query: {query}"}]
                        
                    print(f"Agent {self.agent_id}: Google Search - Found {len(results)} results for '{query}'.")
                    return results

        except aiohttp.ClientError as e:
            print(f"Agent {self.agent_id}: ERROR during Google Custom Search API call: {e}")
            return [{"title": "API Error", "link": "", "snippet": str(e)}]
        except json.JSONDecodeError:
            print(f"Agent {self.agent_id}: ERROR decoding JSON response from Google Custom Search API.")
            return [{"title": "API Response Error", "link": "", "snippet": "Could not parse JSON response."}]
        except Exception as e:
            print(f"Agent {self.agent_id}: UNEXPECTED ERROR during web search: {e}")
            return [{"title": "Unexpected Search Error", "link": "", "snippet": str(e)}]

    def __str__(self):
        return f"Agent({self.agent_id})" 