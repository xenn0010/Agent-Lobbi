import asyncio
import uuid
from typing import List, Dict, Any, Optional
import time

from src.core.agent import Agent, Capability
from src.core.lobby import Lobby
from src.core.message import Message, MessageType
# Updated imports for the refactored test agents
from src.test_agents.item_finder_agent import ItemFinderAgent 
from src.test_agents.price_hunter_agent import PriceHunterAgent


class UserInterfaceAgent(Agent):
    agent_type = "UserInterfaceAgent" # For Lobby tracking

    def __init__(self, agent_id: str, item_to_request: str, website_to_search: str):
        super().__init__(agent_id)
        self.item_to_request = item_to_request
        self.website_to_search = website_to_search
        self.item_finder_agent_id: Optional[str] = None
        self.active_discovery_conv_id: Optional[str] = None
        self.active_item_request_conv_id: Optional[str] = None
        self.final_result: Optional[Dict[str, Any]] = None
        self.task_initiated = False
        self.task_completed = False
        self.task_completion_event = asyncio.Event()
        # New status tracking
        self.status = {
            "state": "initialized",  # initialized, discovering, requesting, completed, error
            "last_error": None,
            "discovery_attempts": 0,
            "request_attempts": 0,
            "start_time": None,
            "end_time": None
        }

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "user_item_requester_v2",
                "description": "Simulates a user requesting an item search and price comparison using V2 test agents.",
                "input_schema": {}, 
                "output_schema": {},
                "keywords": ["user_interface", "item_request", "test_agent_ui"],
                "authorized_requester_ids": None
            }
        ]

    def _update_status(self, new_state: str, error: Optional[str] = None):
        """Update agent status with timestamp and optional error."""
        self.status["state"] = new_state
        if error:
            self.status["last_error"] = error
        if new_state == "discovering":
            self.status["discovery_attempts"] += 1
        elif new_state == "requesting":
            self.status["request_attempts"] += 1
        elif new_state == "completed":
            self.status["end_time"] = time.time()
        print(f"{self.agent_id} ({self.agent_type}) Status Update: {new_state}" + (f" (Error: {error})" if error else ""))

    async def _initiate_search_sequence(self):
        if self.task_initiated:
            return
        
        self.status["start_time"] = time.time()
        self.task_initiated = True
        self._update_status("discovering")

        print(f"{self.agent_id} ({self.agent_type}): Attempting to discover ItemFinderAgent (capability: 'initiate_item_search_v2').")
        self.active_discovery_conv_id = str(uuid.uuid4())
        discovery_payload = {"capability_name": "initiate_item_search_v2"}
        await self.send_message("lobby", MessageType.DISCOVER_SERVICES, discovery_payload, conversation_id=self.active_discovery_conv_id)

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id} ({self.agent_type}) received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id})")

        if msg.message_type == MessageType.SERVICES_AVAILABLE and msg.conversation_id == self.active_discovery_conv_id:
            self.active_discovery_conv_id = None
            services = msg.payload.get("services_found", [])
            
            if services:
                # For now, pick the first service (which should be highest reputation)
                self.item_finder_agent_id = services[0]["agent_id"]
                print(f"{self.agent_id} ({self.agent_type}): Discovered ItemFinderAgent: {self.item_finder_agent_id} (Reputation: {services[0].get('reputation')})")
                self._update_status("requesting")
                
                self.active_item_request_conv_id = str(uuid.uuid4())
                request_payload = {
                    "capability_name": "initiate_item_search_v2",
                    "item_to_find": self.item_to_request,
                    "target_website": self.website_to_search
                }
                print(f"{self.agent_id} ({self.agent_type}): Requesting item search for '{self.item_to_request}' on '{self.website_to_search}'")
                await self.send_message(
                    self.item_finder_agent_id,
                    MessageType.REQUEST,
                    request_payload,
                    conversation_id=self.active_item_request_conv_id
                )
            else:
                error_msg = "Could not find any ItemFinderAgent V2"
                self._update_status("error", error_msg)
                self.final_result = {"status": "error", "message": error_msg}
                self.task_completed = True
                self.task_completion_event.set()

        elif msg.message_type == MessageType.RESPONSE and msg.conversation_id == self.active_item_request_conv_id:
            provider_agent_id = msg.sender_id # This is the ItemFinderAgent
            original_capability_requested = "initiate_item_search_v2" # Hardcoded for this agent's specific request
            self.active_item_request_conv_id = None
            
            task_status_for_report: str
            report_details: str

            if msg.payload.get("status") == "success":
                self._update_status("completed")
                task_status_for_report = "success"
                report_details = f"Successfully received item details and price from {provider_agent_id}."
            else:
                error_message = msg.payload.get('message', 'Unknown error from provider')
                self._update_status("error", f"Response indicates failure: {error_message}")
                task_status_for_report = "failure"
                report_details = f"ItemFinderAgent {provider_agent_id} reported failure: {error_message}"
            
            self.final_result = msg.payload
            self.task_completed = True

            # Use the helper method to send TASK_OUTCOME_REPORT
            await self._report_task_outcome(
                provider_agent_id=provider_agent_id,
                capability_name=original_capability_requested,
                status=task_status_for_report,
                details=report_details,
                original_conversation_id=msg.conversation_id
            )
            
            self.task_completion_event.set()
        
        elif msg.message_type == MessageType.ERROR and msg.conversation_id == self.active_item_request_conv_id:
            provider_agent_id = msg.sender_id # Could be lobby or the agent itself if it sent an error for the request
            original_capability_requested = "initiate_item_search_v2"
            error_detail = msg.payload.get('error', 'No specific error detail provided.')
            self._update_status("error", error_detail)
            self.final_result = {"status": "error", "message": f"Communication error for request: {error_detail}"}
            self.task_completed = True

            # Use the helper method to send TASK_OUTCOME_REPORT
            await self._report_task_outcome(
                provider_agent_id=provider_agent_id, 
                capability_name=original_capability_requested,
                status="failure",
                details=f"Received ERROR message from {provider_agent_id} regarding request: {error_detail}",
                original_conversation_id=msg.conversation_id
            )

            self.task_completion_event.set()

        elif msg.message_type == MessageType.REGISTER_ACK:
            if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} ({self.agent_type}) successfully registered. Token: {'present' if self.auth_token else 'absent'}")

    async def run(self):
        # Ensure agent_type is set if not overridden by subclass, for base class consistency
        if not hasattr(self, 'agent_type') or self.agent_type == Agent.__name__:
            self.agent_type = self.__class__.__name__
            
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id} ({self.agent_type}) is up. Waiting for token.")
        while not self.auth_token:
            await asyncio.sleep(0.1)
        print(f"{self.agent_id} ({self.agent_type}) token confirmed. Operational.")

        await asyncio.sleep(0.2)
        await self._initiate_search_sequence()

        while not self.task_completed:
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                if self.task_completed:
                    break
                # Check for timeout conditions
                if (time.time() - self.status["start_time"]) > 30:  # 30 second timeout
                    self._update_status("error", "Operation timed out")
                    self.final_result = {"status": "error", "message": "Operation timed out"}
                    self.task_completed = True
                    self.task_completion_event.set()
                    break
            await asyncio.sleep(0.1)
        
        print(f"\n--- {self.agent_id} ({self.agent_type}) Final Result Summary ---")
        print(f"  Status: {self.status['state']}")
        print(f"  Discovery attempts: {self.status['discovery_attempts']}")
        print(f"  Request attempts: {self.status['request_attempts']}")
        if self.status['start_time'] and self.status['end_time']:
            duration = self.status['end_time'] - self.status['start_time']
            print(f"  Duration: {duration:.2f} seconds")
        if self.final_result:
            print(f"  User Request: Item '{self.item_to_request}' on site '{self.website_to_search}'")
            print(f"  Outcome: {self.final_result}")
        if self.status['last_error']:
            print(f"  Last Error: {self.status['last_error']}")
        print(f"--- {self.agent_id} ({self.agent_type}) Finished ---\n")


class RogueAgent(Agent):
    """An agent that will try to make an unauthorized call and send false reports."""
    agent_type = "RogueAgent" # For Lobby tracking

    def __init__(self, agent_id: str, price_hunter_id_target: str, innocent_agent_id_target: str):
        super().__init__(agent_id)
        self.price_hunter_id_target = price_hunter_id_target
        self.innocent_agent_id_target = innocent_agent_id_target # Target for false report
        self.unauthorized_call_conv_id = str(uuid.uuid4())
        self.false_report_conv_id = str(uuid.uuid4()) # Separate conv_id for false report
        self.has_made_unauthorized_attempt = False
        self.has_sent_false_report = False
        self.response_received_event = asyncio.Event()
        self.all_malicious_acts_done_event = asyncio.Event()

    def get_capabilities(self) -> List[Capability]:
        return [
            {
                "name": "rogue_caller_v2",
                "description": "Attempts an unauthorized call and sends false outcome reports.",
                "input_schema": {},
                "output_schema": {},
                "keywords": ["test", "rogue", "security"],
                "authorized_requester_ids": None # Public for test setup
            }
        ]

    async def _attempt_unauthorized_call(self):
        if self.lobby_ref and self.auth_token and not self.has_made_unauthorized_attempt:
            await asyncio.sleep(2) # Give lobby time to register price_hunter
            print(f"{self.agent_id}: Attempting UNAUTHORIZED call to {self.price_hunter_id_target} for 'find_cheapest_item_price_v2'")
            request_payload = {
                "capability_name": "find_cheapest_item_price_v2", # PriceHunter's actual capability
                "item_name": "PhantomDevice",
                "regular_price": 150
            }
            await self.send_message(
                self.price_hunter_id_target, 
                MessageType.REQUEST, 
                request_payload, 
                conversation_id=self.unauthorized_call_conv_id
            )
            self.has_made_unauthorized_attempt = True
            # We don't set response_received_event here yet, wait for the response or error from lobby
        elif not self.auth_token:
            print(f"{self.agent_id}: Cannot attempt unauthorized call, auth_token not yet received.")
            # If auth fails, this agent might not be able to do much, so set event to prevent hanging test
            if not self.response_received_event.is_set(): self.response_received_event.set() 
            if not self.all_malicious_acts_done_event.is_set(): self.all_malicious_acts_done_event.set()

    async def _send_false_task_outcome_report(self):
        if self.lobby_ref and self.auth_token and not self.has_sent_false_report:
            await asyncio.sleep(1) # Short delay after unauthorized attempt response
            print(f"{self.agent_id}: INSIDE _send_false_task_outcome_report: About to send to Lobby about {self.innocent_agent_id_target}.")
            
            false_report_payload = {
                "provider_agent_id": self.innocent_agent_id_target, 
                "capability_name": "some_generic_capability", 
                "status": "failure", 
                "details": "RogueAgent claims this agent failed to deliver as promised (this is a false report)."
            }
            await self.send_message(
                "lobby", 
                MessageType.TASK_OUTCOME_REPORT,
                false_report_payload,
                conversation_id=self.false_report_conv_id
            )
            print(f"{self.agent_id}: INSIDE _send_false_task_outcome_report: False TASK_OUTCOME_REPORT supposedly sent.")
            self.has_sent_false_report = True
            
            if not self.all_malicious_acts_done_event.is_set():
                print(f"{self.agent_id}: INSIDE _send_false_task_outcome_report: Setting all_malicious_acts_done_event.")
                self.all_malicious_acts_done_event.set()
            else:
                print(f"{self.agent_id}: INSIDE _send_false_task_outcome_report: all_malicious_acts_done_event was ALREADY set.")

        elif not self.auth_token:
            print(f"{self.agent_id}: INSIDE _send_false_task_outcome_report: Cannot send, auth_token not yet received.")
            if not self.all_malicious_acts_done_event.is_set(): self.all_malicious_acts_done_event.set()
        elif self.has_sent_false_report:
            print(f"{self.agent_id}: INSIDE _send_false_task_outcome_report: Already sent, not sending again.")
            # Ensure event is set if this path is taken for some reason post-initial send.
            if not self.all_malicious_acts_done_event.is_set(): self.all_malicious_acts_done_event.set()

    async def run(self):
        await self.register_with_lobby(self.lobby_ref)
        print(f"{self.agent_id}: {self.agent_type} up. Waiting for token.")
        token_wait_cycles = 0
        max_token_wait_cycles = 200 # Approx 10 seconds
        while not self.auth_token and token_wait_cycles < max_token_wait_cycles:
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=0.05)
                if msg_item: await self.process_incoming_message(msg_item)
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                pass
            token_wait_cycles +=1
            await asyncio.sleep(0.01)

        if not self.auth_token:
            print(f"{self.agent_id}: CRITICAL - Did not receive auth_token. RogueAgent cannot operate.")
            self.response_received_event.set() # Unblock any waiting part of the test
            self.all_malicious_acts_done_event.set()
            return
        
        print(f"{self.agent_id}: Token confirmed. Operational. Starting malicious sequence.")

        # Perform malicious acts in sequence
        asyncio.create_task(self._attempt_unauthorized_call()) # Launch as a task so run loop can continue

        # Main loop to process incoming messages and wait for malicious acts to complete
        start_time = time.time()
        processing_loop_timeout = 20.0 # Timeout for this agent's specific operational phase

        while not self.all_malicious_acts_done_event.is_set():
            if time.time() - start_time > processing_loop_timeout:
                print(f"{self.agent_id}: Timed out in main processing loop waiting for malicious acts.")
                break
            try:
                msg_item = await asyncio.wait_for(self._message_queue.get(), timeout=0.5) 
                if msg_item:
                    await self.process_incoming_message(msg_item)
                    self._message_queue.task_done()
            except asyncio.TimeoutError:
                # This timeout is for self._message_queue.get(), not for all_malicious_acts_done_event
                pass # Continue loop if no message, check all_malicious_acts_done_event again
            
            await asyncio.sleep(0.01) # Yield control briefly

        # Final check and logging for all_malicious_acts_done_event
        if self.all_malicious_acts_done_event.is_set():
            print(f"{self.agent_id}: all_malicious_acts_done_event is set. Proceeding to finish.")
        else:
            print(f"{self.agent_id}: all_malicious_acts_done_event was NOT set before exiting processing loop.")

        # Original wait_for logic, now as a final confirmation or if loop exited due to timeout
        try:
            # This timeout should be shorter as the loop above already has a timeout
            await asyncio.wait_for(self.all_malicious_acts_done_event.wait(), timeout=5.0) 
            print(f"{self.agent_id}: Confirmed all malicious acts completed via event after loop.")
        except asyncio.TimeoutError:
            print(f"{self.agent_id}: Timed out waiting for all_malicious_acts_done_event (final check). Ensure it was set.")
        finally:
            if not self.response_received_event.is_set(): 
                print(f"{self.agent_id}: Setting response_received_event in finally block of run().")
                self.response_received_event.set()
            if not self.all_malicious_acts_done_event.is_set(): # If still not set, force it to unblock simulation
                print(f"{self.agent_id}: Forcing all_malicious_acts_done_event in finally block of run() to prevent sim hang.")
                self.all_malicious_acts_done_event.set()

        print(f"{self.agent_id}: RogueAgent run loop finished.")

    async def process_incoming_message(self, msg: Message):
        print(f"{self.agent_id}: Received: {msg.message_type.name} from {msg.sender_id} (ConvID: {msg.conversation_id}), AuthToken: {msg.auth_token}, Payload: {msg.payload}")

        if msg.conversation_id == self.unauthorized_call_conv_id:
            # This is the response/error from the unauthorized call attempt
            print(f"{self.agent_id}: Handling message for unauthorized_call_conv_id: {msg.conversation_id}")
            is_blocked_error = msg.message_type == MessageType.ERROR and msg.payload and "Unauthorized" in msg.payload.get("error", "")
            
            if is_blocked_error:
                print(f"{self.agent_id}: UNAUTHORIZED CALL SUCCESSFULLY BLOCKED (ERROR msg type with 'Unauthorized' in error field) by Lobby. Error: {msg.payload.get('error')}")
            elif msg.payload and msg.payload.get("status") == "error" and "Unauthorized" in msg.payload.get("error", ""):
                print(f"{self.agent_id}: UNAUTHORIZED CALL SUCCESSFULLY BLOCKED (payload status error with 'Unauthorized' in error field) by Lobby. Error: {msg.payload.get('error')}")
            else:
                print(f"{self.agent_id}: UNEXPECTED RESPONSE to unauthorized call. Type: {msg.message_type.name}, Payload: {msg.payload}")
            
            # Now that the unauthorized call attempt has resolved (blocked or not), proceed to send false report
            if not self.has_sent_false_report:
                print(f"{self.agent_id}: Triggering _send_false_task_outcome_report.")
                asyncio.create_task(self._send_false_task_outcome_report())
            else:
                print(f"{self.agent_id}: False report already attempted or sent, not triggering again.")
            
            if not self.response_received_event.is_set(): 
                print(f"{self.agent_id}: Setting response_received_event for unauthorized call.")
                self.response_received_event.set()

        elif msg.message_type == MessageType.REGISTER_ACK:
            if msg.payload.get("status") == "success_registered_finalized":
                print(f"{self.agent_id} successfully registered. Token: {'present' if self.auth_token else 'absent'}")
        
        elif msg.message_type == MessageType.INFO and msg.sender_id == "lobby":
            # Could be an ack for the false task outcome report if lobby sends one (currently it doesn't for reports)
            print(f"{self.agent_id}: Received INFO from lobby: {msg.payload}")
            if msg.payload.get("status") == "task_outcome_report_received": # Hypothetical ack from lobby
                if not self.all_malicious_acts_done_event.is_set(): self.all_malicious_acts_done_event.set()

        elif msg.message_type == MessageType.ERROR and msg.sender_id == "lobby":
            print(f"{self.agent_id}: Received ERROR from lobby: {msg.payload.get('error')}")
            # If this error is related to the false report, we can consider acts done.
            if msg.conversation_id == self.false_report_conv_id: # Check if error is for the false report
                 if not self.all_malicious_acts_done_event.is_set(): self.all_malicious_acts_done_event.set()


# Main simulation function using the refactored test agents
async def main_item_search_simulation_v2():
    print("###########################################################")
    print("###### SIMULATION_ITEM_SEARCH.PY - main_item_search_simulation_v2 STARTING ######")
    print("###########################################################")
    print("\n=== Starting Enhanced Item Search Simulation V2 ===\n")
    lobby = Lobby()

    # Start the Lobby's message processing task
    await lobby.start() # Ensure the lobby's processing loop is running

    # Define test items with varying characteristics
    test_items = [
        {
            "name": "Bose QuietComfort 45 headphones",
            "website": "amazon.com",
            "category": "electronics"
        },
        {
            "name": "Nike Air Zoom Pegasus 38",
            "website": "nike.com",
            "category": "clothing"
        },
        {
            "name": "Python Programming: A Modern Approach",
            "website": "barnesandnoble.com",
            "category": "books"
        }
    ]

    # Create and register agents
    ui_agents = []
    for i, item in enumerate(test_items):
        ui_agent = UserInterfaceAgent(
            agent_id=f"user_interface_{i+1:03d}",
            item_to_request=item["name"],
            website_to_search=item["website"]
        )
        ui_agents.append(ui_agent)

    item_finder_A1_v2 = ItemFinderAgent(agent_id="item_finder_A1_v2")
    price_hunter_A2_v2 = PriceHunterAgent(agent_id="price_hunter_A2_v2")
    # Modify RogueAgent instantiation to include a target for false reporting
    # Let's pick item_finder_A1_v2 as the innocent target for the false report
    rogue_agent = RogueAgent(agent_id="rogue_007", 
                             price_hunter_id_target="price_hunter_A2_v2", 
                             innocent_agent_id_target="item_finder_A1_v2")

    all_agents = [item_finder_A1_v2, price_hunter_A2_v2, rogue_agent] + ui_agents

    # Register all agents with the lobby. This will now send REGISTER messages
    # that the (now started) lobby processor can handle.
    for agent in all_agents:
        await lobby.register_agent(agent)

    # Start all agents' main run loops
    agent_tasks = [asyncio.create_task(agent.run(), name=f"{agent.agent_id}_run") for agent in all_agents]

    print("\n--- Starting Enhanced Item Search Tests ---\n")
    try:
        ui_event_wait_tasks = [asyncio.create_task(agent.task_completion_event.wait(), name=f"{agent.agent_id}_ui_event") for agent in ui_agents]
        rogue_event_wait_task = asyncio.create_task(rogue_agent.all_malicious_acts_done_event.wait(), name="rogue_event_wait")

        # Define the core tasks that the simulation depends on for its main phase
        # These include the agents' main run loops, the UI agents' completion events, 
        # the rogue agent's completion event, and the lobby's message processor.
        critical_tasks = agent_tasks + ui_event_wait_tasks + [rogue_event_wait_task]
        if lobby._message_processor_task:
            critical_tasks.append(lobby._message_processor_task)
        else:
            print("CRITICAL WARNING: Lobby message processor task not found before gathering.")

        print(f"--- Monitoring {len(critical_tasks)} critical tasks ---")
        # Use asyncio.wait to monitor tasks, allows checking for exceptions if one finishes early
        # We expect most of these to run for the duration or until their specific events are set.
        # If a task errors out, FIRST_COMPLETED will let us inspect it.
        # If all run to completion of events/timeout, that's also handled.
        
        # We'll use a longer timeout here to ensure simulation has ample time.
        # The internal timeouts within agents (e.g., for LLM calls) are separate.
        simulation_timeout = 120.0 
        
        done, pending = await asyncio.wait(critical_tasks, timeout=simulation_timeout, return_when=asyncio.ALL_COMPLETED)
        
        # Check for tasks that completed with an exception
        for task in done:
            if task.cancelled():
                print(f"Task {task.get_name()} was cancelled.")
            elif task.exception():
                exc = task.exception()
                print(f"XXX TASK {task.get_name()} FAILED WITH EXCEPTION: {exc} XXX")
                # To print the full traceback for the failed task:
                # import traceback
                # traceback.print_exception(type(exc), exc, exc.__traceback__)
                # We might want to re-raise to stop the simulation if a critical task fails.
                # For now, just logging. Consider re-raising if needed for stricter error handling.
                # raise exc 

        if not pending and not any(task.exception() for task in done):
            print("\n--- All monitored tasks completed successfully (or via their own logic/events) ---")
        elif pending: #implies timeout from asyncio.wait if return_when=ALL_COMPLETED
             print(f"\nXXX SIMULATION REACHED MAIN TIMEOUT ({simulation_timeout}s) XXX")
             for task in pending:
                print(f"Task {task.get_name()} was still pending and will be cancelled.")
                task.cancel()


        print("\n=== Final Results Summary (including RogueAgent outcome) ===\n")
        for i, ui_agent in enumerate(ui_agents):
            item = test_items[i]
            print(f"\nTest {i+1}: {item['name']} on {item['website']}")
            print(f"Category: {item['category']}")
            print(f"Status: {ui_agent.status['state']}")
            print(f"Discovery attempts: {ui_agent.status['discovery_attempts']}")
            print(f"Request attempts: {ui_agent.status['request_attempts']}")
            
            if ui_agent.status['start_time'] and ui_agent.status['end_time']:
                duration = ui_agent.status['end_time'] - ui_agent.status['start_time']
                print(f"Duration: {duration:.2f} seconds")
            
            if ui_agent.final_result:
                if ui_agent.final_result.get("status") == "success":
                    price_details = ui_agent.final_result.get("price_details", {})
                    print(f"Found Price: ${price_details.get('cheapest_price', 'N/A')}")
                    print(f"Source: {price_details.get('source', 'N/A')}")
                else:
                    print(f"Error: {ui_agent.final_result.get('message', 'Unknown error')}")
            
            if ui_agent.status['last_error']:
                print(f"Last Error: {ui_agent.status['last_error']}")
            print("-" * 50)

    except asyncio.TimeoutError: # This might be caught if asyncio.gather was used with a timeout and it triggered
        print("\nXXX SIMULATION ENCOUNTERED ASYNCIO.TIMEOUT (possibly from gather) XXX\n")
    except Exception as e:
        print(f"\nXXX SIMULATION FAILED WITH AN UNHANDLED EXCEPTION IN MAIN TRY BLOCK: {e} XXX\n")
    finally:
        # Cancel all agent tasks
        for task in agent_tasks:
            task.cancel()
        
        try:
            await asyncio.gather(*agent_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

        print("\n=== Enhanced Item Search Simulation V2 Complete ===\n")

if __name__ == "__main__":
    asyncio.run(main_item_search_simulation_v2()) 