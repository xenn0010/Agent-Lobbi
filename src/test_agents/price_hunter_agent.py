import asyncio
from typing import List, Dict, Any, Optional
import random
import time

from ..core.agent import Agent, Capability
from ..core.message import Message, MessageType

class PriceHunterAgent(Agent): # Agent A2 - Test Agent for AgentLobby
    """
    PriceHunterAgent (A2):
    Purpose: Finds the cheapest price for a given item name across the internet (simulated).
    Follows the AI Agent Design Guide.

    III. Agent Design Foundations:
        - Model: Inherits LLM usage from core.Agent.
        - Tools:
            1. _extract_product_name_tool (LLM-based): Cleans/extracts the core product name from potentially messy input.
            2. _find_best_price_tool (simulated): Simulates searching various vendors for the best price.
        - Instructions:
            - Primary Goal: Given an item name (and optionally a regular price), find the best deal for it.
            - Name Extraction: Use LLM to refine the input item name for more effective searching.
            - Price Search: Use the refined name to search for prices (currently simulated with mock logic).

    V. Defining Tools (Details):
        - _extract_product_name_tool:
            - Input: item_name_request (str)
            - Output: cleaned_item_name (str)
            - Description: Prompts LLM to extract the specific, searchable product name from a potentially longer or less clean string.
        - _find_best_price_tool (simulated by self._search_web and mock pricing logic):
            - Input: cleaned_item_name (str), regular_price (Optional[float])
            - Output: Dict representing price details (price, source, url) or error.
            - Description: Simulates searching for the item and calculating/finding a discounted price.

    VI. Configuring Instructions (Implicitly via logic and prompts):
        The agent's logic and LLM prompts guide its behavior.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.active_request_conv_id: Optional[str] = None
        self.original_item_name_request: Optional[str] = None
        # Price history cache for learning
        self.price_history: Dict[str, List[Dict[str, Any]]] = {}
        # Market trends simulation
        self.market_factors = {
            "electronics": {"base_discount": 0.15, "seasonal_multiplier": 1.2},
            "clothing": {"base_discount": 0.25, "seasonal_multiplier": 1.1},
            "books": {"base_discount": 0.20, "seasonal_multiplier": 1.0},
            "default": {"base_discount": 0.10, "seasonal_multiplier": 1.0}
        }

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "find_cheapest_item_price_v2",
                "description": "(Test Agent) Finds the cheapest price for a given item name across the internet (LLM-assisted name cleaning and price extraction).",
                "input_schema": {'type': "object", "properties": {
                    "item_name": {'type': "string", "description": "The name of the item to find the price for."},
                    "regular_price": {'type': "number", "description": "Optional regular price for context/better mock calculation"}
                }},
                "output_schema": {'type': "object", "properties": {
                    "status": {'type': "string", "enum": ["success", "not_found", "error"]},
                    "price_details": {'type': "object", "properties": {
                        "item_name": {'type': "string"},
                        "cheapest_price": {'type': "number"},
                        "source": {'type': "string"},
                        "url": {'type': "string"}
                    }},
                    "error": {'type': "string"}
                }},
                "keywords": ["price", "cheapest", "deal", "discount", "test_agent", "v2"],
                "authorized_requester_ids": ["item_finder_A1_v2"]
            }
        ]

    async def _extract_product_name_tool_action(self, raw_item_name: str) -> str:
        prompt = (
            f"From the following text, extract only the specific, core product name: '{raw_item_name}'. "
            f"For example, if the text is 'Mock Web Result 1 for Best Ever Laptop X1000', you should respond with 'Best Ever Laptop X1000'. "
            f"Respond with ONLY the extracted product name itself, without any preamble or explanation."
        )
        print(f"{self.agent_id}: TOOL[_extract_product_name_tool_action] - Asking LLM. Prompt: '{prompt}'")
        llm_response = await self._invoke_llm(prompt=prompt, temperature=0.2) # Low temp for factual extraction
        extracted_name = llm_response.strip().replace("\\n", "")
        if not extracted_name or extracted_name.startswith("[Ollama API"): # Handle LLM error
            print(f"{self.agent_id}: TOOL[_extract_product_name_tool_action] - LLM failed to extract name, using original: '{raw_item_name}'")
            extracted_name = raw_item_name # Fallback to raw name
        else:
            print(f"{self.agent_id}: TOOL[_extract_product_name_tool_action] - LLM raw response: '{llm_response}', Cleaned: '{extracted_name}'")
        return extracted_name

    async def _perform_diversified_searches(self, cleaned_item_name: str) -> List[Dict[str, Any]]:
        """
        Performs multiple targeted and general web searches to find price information.
        """
        all_search_results = []
        
        # 1. Targeted site searches
        targeted_sites = ["ebay.com", "newegg.com"] # Example sites
        for site in targeted_sites:
            query = f"{cleaned_item_name} site:{site}"
            print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - Searching on {site} with query: '{query}'")
            results = await self._search_web(query) # Uses real Google Search inherited from Agent
            if results and not (len(results) == 1 and results[0]['title'] == "No results found"):
                print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - Found {len(results)} results on {site}.")
                all_search_results.extend(results)
            else:
                print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - No results found on {site} for '{query}'.")

        # 2. General deal/price searches
        general_search_patterns = [
            "cheapest {item_name}",
            "used {item_name} price",
            "{item_name} deals",
            "{item_name} sale"
        ]
        for pattern in general_search_patterns:
            query = pattern.format(item_name=cleaned_item_name)
            print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - Performing general search: '{query}'")
            results = await self._search_web(query)
            if results and not (len(results) == 1 and results[0]['title'] == "No results found"):
                print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - Found {len(results)} results for general query '{query}'.")
                all_search_results.extend(results)
            else:
                print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - No results found for general query '{query}'.")
        
        print(f"{self.agent_id}: TOOL[_perform_diversified_searches] - Total raw search results gathered: {len(all_search_results)}")
        return all_search_results

    async def _find_best_price_tool_action(self, item_name_from_a1: str, regular_price: Optional[float] = None) -> Dict[str, Any]:
        # Step 1: Use LLM to clean/extract the core product name
        cleaned_item_name = await self._extract_product_name_tool_action(item_name_from_a1)
        
        # Step 2: Perform diversified web searches using the cleaned name
        # This now uses the real _search_web method inherited from the core Agent class.
        print(f"{self.agent_id}: TOOL[_find_best_price_tool_action] - Starting diversified search for '{cleaned_item_name}'.")
        raw_search_results = await self._perform_diversified_searches(cleaned_item_name)

        if not raw_search_results:
            print(f"{self.agent_id}: TOOL[_find_best_price_tool_action] - No search results from diversified search. Cannot determine price.")
            return {"status": "not_found", "error": "No search results found from diversified search."}

        # Step 3: Mock Price Selection (To be replaced with LLM-based extraction and analysis)
        # For now, we'll just simulate finding a price from the gathered results.
        # This mock logic is very basic and needs to be replaced.
        print(f"{self.agent_id}: TOOL[_find_best_price_tool_action] - Applying MOCK price selection to {len(raw_search_results)} results.")
        
        # Super simple mock: try to find one with "deal" or "sale" in title/snippet, or take first one.
        best_deal_mock = None
        if raw_search_results:
            for res in raw_search_results:
                title_lower = res.get('title', '').lower()
                snippet_lower = res.get('snippet', '').lower()
                if "deal" in title_lower or "sale" in title_lower or "discount" in title_lower:
                    best_deal_mock = res
                    break
            if not best_deal_mock: # just take the first if no obvious "deal"
                best_deal_mock = raw_search_results[0]

        if best_deal_mock:
            # Mock a price reduction
            mock_price = (regular_price * random.uniform(0.7, 0.95)) if regular_price else random.uniform(50, 500)
            price_details = {
                "item_name": cleaned_item_name, 
                "cheapest_price": round(mock_price, 2),
                "source": best_deal_mock.get('link', "Unknown Source (from diversified search)"), # Use link as source
                "url": best_deal_mock.get('link', "#") # Use link as URL
            }
            print(f"{self.agent_id}: TOOL[_find_best_price_tool_action] - MOCK best price found: {price_details}")
            return {"status": "success", "price_details": price_details}
        else:
            print(f"{self.agent_id}: TOOL[_find_best_price_tool_action] - MOCK: Could not determine a best price from search results.")
            return {"status": "not_found", "error": "Mock logic could not determine a best price."}

    def _calculate_market_adjusted_price(self, item_name: str, regular_price: float) -> float:
        """Calculate price based on market factors and item category."""
        # Simple category detection based on keywords
        category = "default"
        item_lower = item_name.lower()
        if any(kw in item_lower for kw in ["phone", "laptop", "computer", "headphone", "camera"]):
            category = "electronics"
        elif any(kw in item_lower for kw in ["shirt", "pants", "dress", "shoes"]):
            category = "clothing"
        elif any(kw in item_lower for kw in ["book", "novel", "textbook"]):
            category = "books"

        # Get market factors for category
        factors = self.market_factors.get(category, self.market_factors["default"])
        base_discount = factors["base_discount"]
        seasonal_multiplier = factors["seasonal_multiplier"]

        # Calculate base discounted price
        base_discount_amount = regular_price * base_discount
        
        # Apply seasonal adjustment (mock seasonal effect based on current month)
        current_month = time.localtime().tm_mon
        seasonal_adjustment = abs(((current_month % 6) - 3) / 10)  # Creates a wave pattern through the year
        
        # Calculate final price with all adjustments
        final_discount = base_discount_amount * (1 + seasonal_adjustment) * seasonal_multiplier
        
        # Ensure minimum discount is at least 5% and maximum is 40%
        final_discount = min(regular_price * 0.40, max(regular_price * 0.05, final_discount))
        
        # Round to nearest .99
        final_price = round(regular_price - final_discount, 2)
        final_price = round(final_price - 0.01, 2)

        return max(1.0, final_price)  # Ensure price is never below $1

    def _get_competitive_price(self, item_name: str, regular_price: float) -> float:
        """Calculate competitive price based on price history and market factors."""
        if item_name in self.price_history:
            # Calculate average historical price
            historical_prices = [p["price"] for p in self.price_history[item_name]]
            avg_historical = sum(historical_prices) / len(historical_prices)
            
            # Use historical data to influence pricing
            if avg_historical < regular_price * 0.7:  # If historical prices are very low
                competitive_price = min(regular_price * 0.75, avg_historical * 1.1)  # Stay competitive but profitable
            else:
                # Normal competitive pricing
                competitive_price = self._calculate_market_adjusted_price(item_name, regular_price)
        else:
            # No history, use market-based calculation
            competitive_price = self._calculate_market_adjusted_price(item_name, regular_price)
        
        return round(competitive_price, 2)

    def _update_price_history(self, item_name: str, price: float, source: str):
        """Update price history for learning."""
        if item_name not in self.price_history:
            self.price_history[item_name] = []
        
        # Add new price data
        self.price_history[item_name].append({
            "price": price,
            "source": source,
            "timestamp": time.time()
        })
        
        # Keep only last 10 prices to prevent unlimited growth
        if len(self.price_history[item_name]) > 10:
            self.price_history[item_name] = self.price_history[item_name][-10:]

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id})")
        
        if msg.message_type == MessageType.REQUEST and msg.payload.get("capability_name") == self.get_capabilities()[0]["name"]:
            original_item_name_request = msg.payload.get("item_name")
            regular_price = msg.payload.get("regular_price")

            if not original_item_name_request:
                await self.send_message(msg.sender_id, MessageType.ERROR, {"error": "'item_name' not provided"}, conversation_id=msg.conversation_id)
                return

            print(f"{self.agent_id}: Received request to find cheapest price for '{original_item_name_request}' (regular price: {regular_price}).")

            cleaned_item_name = await self._extract_product_name_tool_action(original_item_name_request)
            
            # Get search results
            search_query_for_price = f"cheapest {cleaned_item_name}"
            print(f"{self.agent_id}: Searching web with cleaned query for price: '{search_query_for_price}'")
            mock_results = await self._search_web(search_query_for_price)

            if mock_results:
                if regular_price is not None:
                    # Calculate competitive price using our enhanced logic
                    mock_cheapest_price = self._get_competitive_price(cleaned_item_name, regular_price)
                else:
                    # If no regular price provided, use a base price estimation
                    estimated_regular_price = 100 + (len(cleaned_item_name) * 2)  # Simple mock estimation
                    mock_cheapest_price = self._get_competitive_price(cleaned_item_name, estimated_regular_price)

                # Create mock source based on best search result
                best_result = mock_results[0]
                mock_source = best_result.get("link", "DiscountDealz.com (mock)")
                
                # Update price history for learning
                self._update_price_history(cleaned_item_name, mock_cheapest_price, mock_source)

                price_details = {
                    "item_name": cleaned_item_name, 
                    "cheapest_price": mock_cheapest_price,
                    "source": mock_source,
                    "url": best_result.get("link", f"http://mockdealz.com/{cleaned_item_name.replace(' ','_')}")
                }
                response_payload = {"status": "success", "price_details": price_details}
                print(f"{self.agent_id}: Found cheapest price details: {price_details}")
            else:
                response_payload = {"status": "not_found", "error": f"Could not find any price for '{cleaned_item_name}'."}
                print(f"{self.agent_id}: Could not find any deals for '{cleaned_item_name}'.")
            
            await self.send_message(msg.sender_id, MessageType.RESPONSE, response_payload, conversation_id=msg.conversation_id)

        elif msg.message_type == MessageType.REGISTER_ACK:
             if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} (PriceHunterAgent V2) is up. Waiting for token.")
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