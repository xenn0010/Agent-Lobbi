import asyncio
import uuid
from typing import List, Dict, Any, Optional, TypedDict, cast

from ..core.agent import Agent, Capability
from ..core.message import Message, MessageType

# VFS: Import jsonschema for validation (will be used later in Lobby or Agent base class)
# For now, we just define the schemas here.
# from jsonschema import validate, ValidationError

# Define TypedDicts for structured capability schemas if not already general
class CapabilityInputSchema(TypedDict):
    type: str
    properties: Dict[str, Any]
    required: List[str]

class FoundItem(TypedDict):
    name: str
    description: str
    regular_price: Optional[float] # Made price optional and float
    source_website: str
    url: str

class PriceDetails(TypedDict): # Re-using from PriceHunterAgent_v2 for consistency
    item_name: str
    cheapest_price: float # Made price float
    source: str
    url: str

class CapabilityOutputSchemaSuccess(TypedDict):
    type: str
    properties: Dict[str, Any]
    required: List[str]

class CapabilityOutputSchemaError(TypedDict):
    type: str
    properties: Dict[str, Any]
    required: List[str]

class ItemFinderAgent(Agent):  # Agent A1 - Test Agent for AgentLobby
    """
    ItemFinderAgent (A1): 
    Purpose: Finds a specific item on a target website and then triggers a price hunt.
    Follows the AI Agent Design Guide.

    III. Agent Design Foundations:
        - Model: Inherits LLM usage from core.Agent (e.g., gemma3:1b via Ollama).
        - Tools: 
            1. _generate_search_query_tool (LLM-based): Creates optimized web search queries.
            2. _execute_web_search_tool (simulated): Simulates performing a web search.
            3. _check_relevance_tool (LLM-based): Checks if a found item matches the request.
        - Instructions: 
            - Primary Goal: Given an item and a website, find the item, confirm its relevance, 
              and if relevant, discover and request a price hunt from a PriceHunterAgent.
            - Iterative Search: Make up to 'max_attempts' to find a relevant item.
            - Query Generation: Use LLM to generate search queries. Inform LLM of past failed queries.
            - Relevance Check: Use LLM to confirm if a found item is relevant to the original request.
            - Handoff: If a relevant item is found, discover a PriceHunterAgent and request it to find the cheapest price.

    V. Defining Tools (Details):
        - _generate_search_query_tool:
            - Input: item_name (str), target_website (str), previous_failed_queries (List[str])
            - Output: search_query (str)
            - Description: Prompts LLM to create an effective search query. Informs LLM about queries that previously failed to yield relevant results.
        - _execute_web_search_tool (simulated by self._search_web):
            - Input: search_query (str)
            - Output: List[Dict[str, Any]] (simulated search results)
            - Description: Simulates calling a web search engine with the given query.
        - _check_relevance_tool:
            - Input: item_to_find (str), target_website (str), search_query (str), found_item_title (str), found_item_snippet (str)
            - Output: is_relevant (bool)
            - Description: Prompts LLM to determine if the found item details match the user's original item request.

    VI. Configuring Instructions (Implicitly via logic and prompts):
        The agent's run loop and message processing logic, combined with the prompts sent to the LLM for 
        query generation and relevance checking, constitute its operational instructions.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.item_to_find: Optional[str] = None
        self.target_website: Optional[str] = None
        self.item_details_on_site: Optional[Dict[str, Any]] = None
        self.cheapest_price_info: Optional[Dict[str, Any]] = None
        self.price_hunter_agent_id: Optional[str] = None
        self.active_discovery_conv_id: Optional[str] = None
        self.active_price_request_conv_id: Optional[str] = None
        self.task_in_progress = False
        self.current_requester_id: Optional[str] = None
        self.current_requester_conv_id: Optional[str] = None
        self.max_search_attempts = 3
        # Enhanced status tracking
        self.status = {
            "state": "initialized",  # initialized, searching, discovering_price_hunter, requesting_price, completed, error
            "search_attempts": 0,
            "last_error": None,
            "start_time": None,
            "end_time": None,
            "llm_errors": 0,
            "search_errors": 0,
            "price_request_errors": 0
        }
        # Error recovery settings
        self.retry_config = {
            "max_llm_retries": 2,
            "max_search_retries": 2,
            "max_price_request_retries": 2,
            "retry_delay": 1.0  # seconds
        }

    def get_capabilities(self) -> List[Capability]:
        input_schema_item_search = {
            "type": "object",
            "properties": {
                "item_to_find": {"type": "string", "description": "The name or description of the item to search for."},
                "target_website": {"type": "string", "description": "The specific website to search on (e.g., 'amazon.com', 'bestbuy.com')."}
            },
            "required": ["item_to_find", "target_website"]
        }

        output_schema_item_search = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "item_found_on_site": {
                    "type": ["object", "null"], # Can be null if not found or error
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "regular_price": {"type": ["number", "null"]},
                        "source_website": {"type": "string"},
                        "url": {"type": "string"}
                    },
                    "required": ["name", "description", "source_website", "url"] # regular_price is optional
                },
                "cheapest_price_globally": {
                    "type": ["object", "null"], # Can be null if not found or error
                     "properties": {
                        "item_name": {"type": "string"},
                        "cheapest_price": {"type": "number"},
                        "source": {"type": "string"},
                        "url": {"type": "string"}
                    },
                    "required": ["item_name", "cheapest_price", "source", "url"]
                },
                "message": {"type": "string", "description": "A summary message of the operation's outcome."}
            },
            "required": ["status", "message"] # item_found_on_site and cheapest_price_globally depend on status
        }

        return [
            Capability(
                name="initiate_item_search_v2",
                description="(Test Agent) Initiates an iterative, LLM-guided search for an item on a website and then finds its cheapest price.",
                input_schema=cast(CapabilityInputSchema, input_schema_item_search), # VFS: Cast to specific type
                output_schema=cast(Dict[str, Any], output_schema_item_search), # VFS: Cast to generic Dict for now, can be more specific
                keywords=["item_search", "price_comparison", "product_finder", "test_agent"],
                authorized_requester_ids=None  # Publicly callable
            )
        ]

    def _update_status(self, new_state: str, error: Optional[str] = None):
        """Update agent status with timestamp and optional error."""
        import time
        self.status["state"] = new_state
        if error:
            self.status["last_error"] = error
        if new_state == "searching":
            self.status["search_attempts"] += 1
        elif new_state == "completed":
            self.status["end_time"] = time.time()
        print(f"{self.agent_id} Status Update: {new_state}" + (f" (Error: {error})" if error else ""))

    async def _handle_llm_error(self, operation: str, error: str) -> Optional[str]:
        """Handle LLM-related errors with retries."""
        self.status["llm_errors"] += 1
        if self.status["llm_errors"] <= self.retry_config["max_llm_retries"]:
            print(f"{self.agent_id}: LLM error in {operation}. Attempt {self.status['llm_errors']}/{self.retry_config['max_llm_retries']}. Retrying...")
            await asyncio.sleep(self.retry_config["retry_delay"])
            return None  # Signal to retry
        else:
            error_msg = f"Maximum LLM retries exceeded for {operation}: {error}"
            self._update_status("error", error_msg)
            return error_msg

    async def _handle_search_error(self, operation: str, error: str) -> Optional[str]:
        """Handle web search related errors with retries."""
        self.status["search_errors"] += 1
        if self.status["search_errors"] <= self.retry_config["max_search_retries"]:
            print(f"{self.agent_id}: Search error in {operation}. Attempt {self.status['search_errors']}/{self.retry_config['max_search_retries']}. Retrying...")
            await asyncio.sleep(self.retry_config["retry_delay"])
            return None  # Signal to retry
        else:
            error_msg = f"Maximum search retries exceeded for {operation}: {error}"
            self._update_status("error", error_msg)
            return error_msg

    async def _generate_search_query_tool_action(self, item_name: str, target_website: str, previous_failed_queries: List[str]) -> Optional[str]:
        try:
            query_prompt_parts = [
                f"I need to find an item called '{item_name}' on the website '{target_website}'.",
                "Please generate the most effective search query string to use for a web search engine, focusing the search on that specific site."
            ]
            if previous_failed_queries:
                failed_queries_str = ", ".join([f"'{q}'" for q in previous_failed_queries])
                query_prompt_parts.append(f"Previous attempts with queries like [{failed_queries_str}] did not yield relevant results. Generate a DIFFERENT and NOVEL search query.")
            query_prompt_parts.append("Respond with ONLY the search query string itself, without any preamble or explanation.")
            
            query_generation_prompt = " ".join(query_prompt_parts)
            
            print(f"{self.agent_id}: TOOL[_generate_search_query_tool_action] - Asking LLM.")
            llm_response = await self._invoke_llm(prompt=query_generation_prompt, temperature=0.2)
            
            if not llm_response or llm_response.startswith("[Ollama API"):
                error = await self._handle_llm_error("query generation", "Failed to get valid response from LLM")
                if error:
                    return None
                return await self._generate_search_query_tool_action(item_name, target_website, previous_failed_queries)
            
            search_query = llm_response.strip().replace("\n", "")
            return search_query
        except Exception as e:
            error = await self._handle_llm_error("query generation", str(e))
            if error:
                return None
            return await self._generate_search_query_tool_action(item_name, target_website, previous_failed_queries)

    async def _check_relevance_tool_action(self, item_to_find_orig: str, target_website_orig: str, used_query: str, found_title: str, found_snippet: str) -> Optional[bool]:
        try:
            relevance_prompt = (
                f"I am looking for a specific product: '{self.item_to_find}'. "
                f"I searched on the website '{self.target_website}' using the search query '{used_query}'. "
                f"The search returned an item titled '{found_title}' with this description/snippet: '{found_snippet}'. "
                f"Based on this title and snippet, does this found item appear to be the specific product '{self.item_to_find}' "
                f"or a very close variant (e.g., different color, newer model, storage size, or a direct sales page for it)? "
                f"Answer with only the word YES or the word NO."
            )
            print(f"{self.agent_id}: TOOL[_check_relevance_tool_action] - Asking LLM for relevance.")
            llm_response = await self._invoke_llm(prompt=relevance_prompt, temperature=0.1)
            
            if not llm_response or llm_response.startswith("[Ollama API"):
                error = await self._handle_llm_error("relevance check", "Failed to get valid response from LLM")
                if error:
                    return None
                return await self._check_relevance_tool_action(item_to_find_orig, target_website_orig, used_query, found_title, found_snippet)
            
            cleaned_response = llm_response.strip().upper()
            return "YES" in cleaned_response
        except Exception as e:
            error = await self._handle_llm_error("relevance check", str(e))
            if error:
                return None
            return await self._check_relevance_tool_action(item_to_find_orig, target_website_orig, used_query, found_title, found_snippet)

    async def _start_item_search_flow(self, item_to_find: str, target_website: str, requester_id: str, requester_conv_id: str):
        if self.task_in_progress:
            await self._send_final_response("error", f"Task already in progress for {self.current_requester_id}")
            return

        import time
        self.status["start_time"] = time.time()
        self.item_to_find = item_to_find
        self.target_website = target_website
        self.current_requester_id = requester_id
        self.current_requester_conv_id = requester_conv_id
        self.task_in_progress = True
        self.item_details_on_site = None
        self.cheapest_price_info = None
        self.price_hunter_agent_id = None
        self.active_discovery_conv_id = None
        self.active_price_request_conv_id = None
        
        self._update_status("searching")
        print(f"{self.agent_id}: Initiating search for '{self.item_to_find}' on '{self.target_website}'")

        previous_failed_queries = []
        llm_found_relevant_item = False

        for attempt in range(1, self.max_search_attempts + 1):
            print(f"{self.agent_id}: Search attempt {attempt}/{self.max_search_attempts}")

            search_query = await self._generate_search_query_tool_action(self.item_to_find, self.target_website, previous_failed_queries)
            if not search_query:
                error_msg = "Failed to generate search query after retries"
                self._update_status("error", error_msg)
                await self._send_final_response("error", error_msg)
                return

            if search_query in previous_failed_queries:
                search_query = f'{self.item_to_find} "{self.item_to_find}" site:{self.target_website} attempt:{attempt}'
            
            if search_query not in previous_failed_queries:
                previous_failed_queries.append(search_query)
            
            try:
                mock_site_results = await self._search_web(search_query)
            except Exception as e:
                error = await self._handle_search_error("web search", str(e))
                if error:
                    await self._send_final_response("error", error)
                    return
                continue

            if mock_site_results:
                found_item_title = mock_site_results[0].get("title", "Unknown Title")
                found_item_snippet = mock_site_results[0].get("snippet", "No description.")

                is_relevant = await self._check_relevance_tool_action(
                    self.item_to_find, self.target_website, search_query, found_item_title, found_item_snippet
                )
                
                if is_relevant is None:  # LLM error occurred
                    error_msg = "Failed to check item relevance after retries"
                    self._update_status("error", error_msg)
                    await self._send_final_response("error", error_msg)
                    return

                if is_relevant:
                    print(f"{self.agent_id}: Found relevant item: '{found_item_title}'")
                    mock_regular_price = 100 + len(self.item_to_find)
                    self.item_details_on_site = {
                        "name": found_item_title,
                        "description": found_item_snippet,
                        "regular_price": mock_regular_price,
                        "source_website": self.target_website,
                        "url": mock_site_results[0].get("url", f"http://{self.target_website}/{self.item_to_find.replace(' ','_')}")
                    }
                    llm_found_relevant_item = True
                    break

            if attempt < self.max_search_attempts:
                await asyncio.sleep(0.1)

        if not llm_found_relevant_item:
            error_msg = f"Could not find relevant item after {self.max_search_attempts} attempts"
            self._update_status("error", error_msg)
            await self._send_final_response("error", error_msg)
            return

        # Continue with price discovery...
        self._update_status("discovering_price_hunter")
        self.active_discovery_conv_id = str(uuid.uuid4()) 
        discovery_payload = {"capability_name": "find_cheapest_item_price_v2"} # CORRECTED capability name
        print(f"{self.agent_id}: Attempting to discover PriceHunterAgent (capability: find_cheapest_item_price_v2).")
        await self.send_message("lobby", MessageType.DISCOVER_SERVICES, discovery_payload, conversation_id=self.active_discovery_conv_id)

    async def _send_final_response(self, status: str, message: Optional[str] = None):
        if not self.current_requester_id or not self.current_requester_conv_id:
            print(f"{self.agent_id}: ERROR - No current requester to send final response to.")
            self.task_in_progress = False
            return

        response_payload = {
            "status": status,
            "item_found_on_site": self.item_details_on_site,
            "cheapest_price_globally": self.cheapest_price_info if self.cheapest_price_info and self.cheapest_price_info.get("status") != "error" else None,
            "message": message or (self.cheapest_price_info.get("message") if self.cheapest_price_info and self.cheapest_price_info.get("status") == "error" else "Search complete.")
        }
        print(f"{self.agent_id}: Sending final response to {self.current_requester_id} (ConvID: {self.current_requester_conv_id}): {response_payload}")
        await self.send_message(self.current_requester_id, MessageType.RESPONSE, response_payload, conversation_id=self.current_requester_conv_id)
        self.task_in_progress = False
        self.current_requester_id = None
        self.current_requester_conv_id = None
        self.item_to_find = None
        self.target_website = None

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}) payload: {msg.payload}")

        if msg.message_type == MessageType.REQUEST and msg.payload.get("capability_name") == self.get_capabilities()[0]["name"]: # Match dynamically
            item = msg.payload.get("item_to_find")
            website = msg.payload.get("target_website")
            if item and website:
                await self._start_item_search_flow(item, website, msg.sender_id, msg.conversation_id)
            else:
                await self.send_message(msg.sender_id, MessageType.ERROR, {"error": "Missing item_to_find or target_website in request"}, conversation_id=msg.conversation_id)
            return

        if msg.message_type == MessageType.SERVICES_AVAILABLE and msg.conversation_id == self.active_discovery_conv_id:
            self.active_discovery_conv_id = None 
            services = msg.payload.get("services_found", [])
            if services:
                self.price_hunter_agent_id = services[0]["agent_id"]
                print(f"{self.agent_id}: Discovered PriceHunterAgent: {self.price_hunter_agent_id}")
                if self.item_details_on_site:
                    self.active_price_request_conv_id = str(uuid.uuid4()) 
                    request_payload = {
                        "capability_name": "find_cheapest_item_price_v2", # CORRECTED capability name
                        "item_name": self.item_details_on_site["name"],
                        "regular_price": self.item_details_on_site["regular_price"]
                    }
                    print(f"{self.agent_id}: Requesting cheapest price for '{self.item_details_on_site['name']}' from {self.price_hunter_agent_id} (capability: find_cheapest_item_price_v2).")
                    await self.send_message(self.price_hunter_agent_id, MessageType.REQUEST, request_payload, conversation_id=self.active_price_request_conv_id)
                else: 
                     print(f"{self.agent_id}: ERROR - No item details to ask price for.")
                     await self._send_final_response(status="error", message="Internal error: Item details missing before price hunting.")
            else:
                print(f"{self.agent_id}: ERROR - Could not find PriceHunterAgent.")
                self.cheapest_price_info = {"status": "error", "message": "PriceHunterAgent not found"}
                await self._send_final_response(status="error", message="PriceHunterAgent not found.")

        elif msg.message_type == MessageType.RESPONSE and msg.sender_id == self.price_hunter_agent_id and msg.conversation_id == self.active_price_request_conv_id:
            self.active_price_request_conv_id = None
            if msg.payload.get("status") == "success":
                self.cheapest_price_info = msg.payload.get("price_details")
                print(f"--- {self.agent_id} INTERNAL: Got cheapest price: {self.cheapest_price_info} ---")
                await self._send_final_response(status="success")
            else:
                error_detail = msg.payload.get('error', 'No specific error detail provided.')
                print(f"{self.agent_id}: ERROR - PriceHunterAgent reported error: {error_detail}")
                self.cheapest_price_info = {"status": "error", "message": f"PriceHunterAgent error: {error_detail}"}
                await self._send_final_response(status="error", message=f"PriceHunterAgent error: {error_detail}")

        elif msg.message_type == MessageType.ERROR and msg.sender_id == self.price_hunter_agent_id and msg.conversation_id == self.active_price_request_conv_id:
            error_detail = msg.payload.get('error', 'No specific error detail provided.')
            print(f"{self.agent_id}: ERROR - Received error message from PriceHunterAgent: {error_detail}")
            self.cheapest_price_info = {"status": "error", "message": f"PriceHunterAgent communication error: {error_detail}"}
            self.active_price_request_conv_id = None
            await self._send_final_response(status="error", message=f"PriceHunterAgent communication error: {error_detail}")
        
        elif msg.message_type == MessageType.REGISTER_ACK:
             if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} (ItemFinderAgent V2) is up, awaiting requests. Waiting for token.")
        while not self.auth_token:
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