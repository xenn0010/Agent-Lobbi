# Agent Lobby Protocol v0.1.0

## 1. Introduction

The Agent Lobby Protocol facilitates communication, service discovery, and collaboration between autonomous AI agents within a shared ecosystem. It is designed to be extensible and robust, enabling agents with diverse capabilities to interact securely and efficiently. The Lobby serves as a central message router, service registry, and enforcer of communication rules.

## 2. Core Concepts

*   **Lobby:** The central intermediary. All inter-agent messages (except initial registration handshake) pass through the Lobby. It manages agent registration, service discovery, and enforces authentication and authorization rules. The default ID for the Lobby is `"global_lobby"`.
*   **Agent:** An independent, addressable entity capable of sending and receiving messages, and providing or consuming services (capabilities). Each agent has a unique `agent_id`.
*   **Capability:** A well-defined service or function an agent can perform. Capabilities are advertised during registration and include metadata such as input/output schemas and authorization rules.
*   **Message:** The fundamental unit of communication, structured as a JSON-like object.

## 3. Message Structure (General)

All messages adhere to the following base structure:

```json
{
  "sender_id": "string (agent_id of the sender)",
  "receiver_id": "string (agent_id of the receiver, or 'global_lobby')",
  "message_type": "string (MessageType enum value)",
  "payload": "object (varies by message_type, can be empty: {})",
  "conversation_id": "string (UUID, optional, for correlating messages in an exchange)",
  "auth_token": "string (agent's authentication token, optional for REGISTER)"
}
```

## 4. Message Types & Payloads

### 4.1. `REGISTER`
*   **Direction:** Agent -> Lobby (`global_lobby`)
*   **Purpose:** An agent sends this message to the Lobby to announce its presence and advertise its capabilities. This is part of the initial handshake; the agent does not yet have an `auth_token`.
*   **Payload:**
    ```json
    {
      "capabilities": [
        // Array of Capability objects (see Section 7)
      ]
    }
    ```

### 4.2. `REGISTER_ACK`
*   **Direction:** Lobby -> Agent
*   **Purpose:** The Lobby sends this to an agent to confirm successful registration and provide an `auth_token`.
*   **Payload:**
    ```json
    {
      "status": "success_registered_finalized",
      "lobby_id": "string (ID of the lobby, e.g., 'global_lobby')",
      "auth_token": "string (the newly issued authentication token for the agent)"
    }
    ```
    *   If registration fails (e.g., agent_id already taken, though current implementation pre-registers via `lobby.register_agent()` before this message is sent), an `ERROR` message might be sent or the connection handled differently by the Lobby.

### 4.3. `DISCOVER_SERVICES`
*   **Direction:** Agent -> Lobby (`global_lobby`)
*   **Purpose:** An agent queries the Lobby to find other agents that provide a specific capability.
*   **Payload:**
    ```json
    {
      "capability_name": "string (the specific capability name being searched for)"
      // "query_keywords": ["string"], // Alternative: search by keywords (currently secondary)
    }
    ```

### 4.4. `SERVICES_AVAILABLE`
*   **Direction:** Lobby -> Agent
*   **Purpose:** The Lobby responds to a `DISCOVER_SERVICES` request, listing agents that match the query.
*   **Payload:**
    ```json
    {
      "services_found": [
        {
          "agent_id": "string",
          "relevant_capabilities": [
            // Array of Capability objects matching the query
          ]
        }
      ],
      "discovered_for_capability": "string (original capability_name from the DISCOVER_SERVICES request)"
    }
    ```

### 4.5. `REQUEST`
*   **Direction:** Agent A -> Agent B (via Lobby)
*   **Purpose:** Agent A requests Agent B to perform an action defined by one of Agent B's advertised capabilities.
*   **Payload:**
    ```json
    {
      "capability_name": "string (the name of the capability to invoke on Agent B)",
      // ... other fields as defined by the capability's input_schema ...
    }
    ```
    *Example for `initiate_item_search_v2`:*
    ```json
    {
      "capability_name": "initiate_item_search_v2",
      "item_to_find": "Bose QuietComfort headphones",
      "target_website": "amazon.com"
    }
    ```

### 4.6. `RESPONSE`
*   **Direction:** Agent B -> Agent A (via Lobby)
*   **Purpose:** Agent B responds to a `REQUEST` from Agent A. Can indicate success or failure (with details).
*   **Payload:**
    ```json
    {
      "status": "string ('success', 'error', 'in_progress', etc.)",
      // ... other fields as defined by the capability's output_schema if successful ...
      // ... or error details if status is 'error' (see Section 8) ...
    }
    ```
    *Example for successful `initiate_item_search_v2` leading to `find_cheapest_item_price_v2`:*
    ```json
    {
        "status": "success",
        "item_found_on_site": {
            "name": "Bose Quietcomfort - Amazon.com",
            "description": "...",
            "regular_price": 128,
            "source_website": "amazon.com",
            "url": "http://amazon.com/Bose_QuietComfort_headphones_attempt1"
        },
        "cheapest_price_globally": {
            "item_name": "Best Ever Laptop X1000", 
            "cheapest_price": 108.8,
            "source": "DiscountDealz.com (mock)",
            "url": "http://mockdealz.com/Best_Ever_Laptop_X1000"
        },
        "message": "Search complete."
    }
    ```
    *Example for an error response (e.g., unauthorized access):*
    ```json
    {
        "status": "error",
        "error": "Unauthorized: Agent 'rogue_007' is not authorized to call capability 'find_cheapest_item_price_v2' on agent 'price_hunter_A2_v2'."
    }
    ```

### 4.7. `ERROR`
*   **Direction:** Lobby -> Agent or Agent -> Agent (via Lobby, less common for agent-to-agent)
*   **Purpose:** Indicates a protocol-level error or a system error encountered by the Lobby or an agent when processing a message that isn't a direct response to a capability request.
*   **Payload:**
    ```json
    {
      "error": "string (description of the error)",
      "original_message_type": "string (optional, type of message that caused the error)",
      "original_conversation_id": "string (optional, conv_id of message that caused error)"
    }
    ```
    *Example: Lobby sends to agent if auth token is invalid for a DISCOVER_SERVICES request:*
    ```json
    {
        "error": "Authentication failed for message to lobby."
    }
    ```

## 5. Key Flows

### 5.1. Agent Registration & Initialization
1.  **Simulation Script:** `lobby.register_agent(agent_instance)`
    *   Lobby creates an `auth_token` for the agent.
    *   Lobby stores the agent instance and its capabilities (from `agent.get_capabilities()`).
    *   Lobby assigns the `auth_token` and a `lobby_ref` to the agent instance.
    *   Lobby logs `AGENT_REGISTERED`.
2.  **Agent (in its `run` or `register_with_lobby` method):** Sends `REGISTER` message to `global_lobby`.
    *   Payload contains its capabilities. `auth_token` is `None`.
3.  **Lobby (`route_message`):**
    *   Authenticates the message: For `REGISTER`, auth is considered "N/A_REGISTERING" and allowed to proceed.
    *   Routes to `handle_lobby_message`.
4.  **Lobby (`handle_lobby_message` for `REGISTER`):**
    *   Finalizes registration details (though capabilities are already stored from `lobby.register_agent`).
    *   Logs `AGENT_REGISTRATION_FINALIZED`.
    *   Sends `REGISTER_ACK` message back to the agent, containing the `auth_token` previously generated and stored.
5.  **Agent (`process_incoming_message`):** Receives `REGISTER_ACK` and confirms its token.

### 5.2. Service Discovery
1.  **Agent A (Requester):** Sends `DISCOVER_SERVICES` message to `global_lobby`.
    *   Payload includes `capability_name`.
    *   Includes its `auth_token`.
2.  **Lobby (`route_message`):**
    *   Authenticates Agent A's token. If valid, proceeds.
    *   Routes to `handle_lobby_message`.
3.  **Lobby (`handle_lobby_message` for `DISCOVER_SERVICES`):**
    *   Searches its `agent_capabilities` for agents (excluding Agent A) that offer the specified `capability_name`.
    *   Constructs a list of matching services.
    *   Sends `SERVICES_AVAILABLE` message back to Agent A.
4.  **Agent A (`process_incoming_message`):** Receives `SERVICES_AVAILABLE` and extracts `agent_id` of providers.

### 5.3. Capability Invocation
1.  **Agent A (Requester):** Sends `REQUEST` message to Agent B (target `agent_id`).
    *   Payload includes `capability_name` and input data matching the capability's `input_schema`.
    *   Includes its `auth_token`.
    *   Includes a `conversation_id`.
2.  **Lobby (`route_message`):**
    *   Authenticates Agent A's token.
    *   **Authorization Check:**
        *   Retrieves the requested `capability_name` details for Agent B.
        *   Checks if `authorized_requester_ids` is defined for this capability.
        *   If defined, verifies if Agent A's `agent_id` is in the list.
        *   If unauthorized:
            *   Lobby sends a `RESPONSE` message with `status: "error"` and an error description back to Agent A (using the same `conversation_id`).
            *   Logs `REQUEST_DENIED_AUTHORIZATION`.
            *   Processing stops.
        *   If authorized (or no restriction):
            *   Logs `REQUEST_AUTHORIZED`.
            *   Forwards the `REQUEST` message to Agent B.
3.  **Agent B (`process_incoming_message`):** Receives the `REQUEST`.
    *   Performs the capability action.
    *   Sends a `RESPONSE` message back to Agent A (target `agent_id`), using the same `conversation_id`.
        *   Payload includes `status` and output data (matching `output_schema`) or error details.
4.  **Lobby (`route_message`):**
    *   Authenticates Agent B's token.
    *   Forwards the `RESPONSE` message to Agent A.
5.  **Agent A (`process_incoming_message`):** Receives the `RESPONSE` for its `conversation_id`.

## 6. Authentication
*   Upon initial registration with the Lobby (via `lobby.register_agent()`), each agent is issued a unique UUID-based `auth_token`.
*   This token is provided to the agent instance and also sent via the `REGISTER_ACK` message.
*   Agents must include this `auth_token` in the `auth_token` field of all subsequent messages sent to the Lobby or other agents via the Lobby (except for the initial `REGISTER` message itself, which has `auth_token: null`).
*   The Lobby validates this token for every message. If the token is missing (for non-REGISTER messages) or invalid, the Lobby will typically reject the message and may send an `ERROR` response to the sender.

## 7. Capability Definition

Agents advertise their services as a list of "Capability" objects. Each capability is a dictionary with the following structure:

```typescript
interface Capability {
  name: string;                      // Unique name for the capability (e.g., "find_cheapest_item_price_v2")
  description: string;               // Human-readable description
  input_schema: Record<string, any>; // JSON schema defining the expected payload for a REQUEST to this capability
  output_schema: Record<string, any>;// JSON schema defining the expected payload for a successful RESPONSE from this capability
  keywords?: string[];                // Optional list of keywords for discovery
  authorized_requester_ids?: string[] | null; // Optional. List of agent_ids authorized to call. `null` or omitted means public. An empty list `[]` also means public (or could be interpreted as "no one allowed" - currently public).
}
```
*   **Schema Note:** `input_schema` and `output_schema` are intended to be JSON Schema objects. Currently, they are placeholder empty objects (`{}`) in some test agents. Future work should enforce these.

## 8. Error Handling

*   **Protocol Errors:** If the Lobby or an agent encounters an issue processing a message due to protocol violations (e.g., bad authentication, malformed message, capability not found, unauthorized request), it should respond to the original sender.
    *   For authorization failures or capability-not-found issues during a `REQUEST`, the Lobby sends a `RESPONSE` message with `status: "error"` and an `error` field in the payload detailing the issue.
    *   For other general errors (e.g., bad token on a `DISCOVER_SERVICES` call), the Lobby sends an `ERROR` message.
*   **Application Errors:** If an agent successfully receives a valid `REQUEST` but encounters an internal error while executing the capability, it should send a `RESPONSE` message with `status: "error"` and appropriate error details in the payload, conforming to its `output_schema` for errors if defined.

## 9. Future Considerations (Beyond MVP v0.1.0)
*   Schema validation enforcement for all message payloads against `input_schema` and `output_schema`.
*   More sophisticated error codes and reporting.
*   Agent-to-agent direct communication (bypassing Lobby after initial handshake/discovery, if desired for certain high-throughput interactions).
*   Lobby scalability and fault tolerance.
*   Dynamic capability updates (register/unregister after initial registration).
*   Versioning of the protocol itself and individual capabilities. 