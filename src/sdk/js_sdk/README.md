# Agent Lobbi JavaScript SDK (@agent-lobby/sdk)

This SDK provides a JavaScript client for interacting with the Agent Lobbi ecosystem, suitable for Node.js environments.

## Installation

Navigate to the `src/sdk/js_sdk` directory and run:

```bash
npm install
# or
yarn install
```

To use it in your project, you can install it locally:

```bash
# Assuming your project is outside this sdk directory
npm install path/to/agent_lobby/src/sdk/js_sdk
# or add to your package.json dependencies:
# "@agent-lobby/sdk": "file:../path/to/agent_lobby/src/sdk/js_sdk"
```

If published to a registry (e.g., npmjs.com):
```bash
npm install @agent-lobby/sdk
```

## Usage

```javascript
const { EnhancedEcosystemClientJS, SDKConfigJS } = require('@agent-lobby/sdk');
const { v4: uuidv4 } = require('uuid'); // For generating unique IDs if needed

async function main() {
    console.log("Starting JS SDK example usage...");

    // Configuration can be loaded from environment variables or a config file
    // For this example, we use environment variables or defaults.
    const config = new SDKConfigJS({
        agentId: process.env.AGENT_ID || `my_js_agent_${uuidv4()}`,
        agentType: process.env.AGENT_TYPE || "ExampleNodeAgent",
        apiKey: process.env.API_KEY || "your-secret-api-key", // Replace with actual key
        lobbyHttpUrl: process.env.LOBBY_HTTP_URL || "http://localhost:8088",
        lobbyWsUrl: process.env.LOBBY_WS_URL || "ws://localhost:8088",
        capabilities: [{ name: "nodejs_echo_service", version: "1.0" }],
        reconnectInterval: 5000,      // ms, interval for WebSocket reconnection attempts
        registrationTimeout: 10000,   // ms, timeout for HTTP registration call
        websocketTimeout: 10000       // ms, timeout for WebSocket connection handshake
    });

    const client = new EnhancedEcosystemClientJS(config);

    // Register a wildcard handler for all messages
    client.registerMessageHandler('*', (message) => {
        console.log(`NodeAgent (${client.config.agentId}) received message:`, JSON.stringify(message, null, 2));
        // Add your custom message processing logic here
    });

    try {
        await client.start(); // Registers and connects WebSocket

        // It might take a moment for the WebSocket to connect fully after start() resolves
        // A more robust approach would be to await an 'open' or 'connected' event from the client if exposed.
        await new Promise(resolve => setTimeout(resolve, 2000));

        if (client.isConnected) {
            console.log(`NodeAgent (${client.config.agentId}) is connected and registered.`);

            // Send a test message
            const testMessage = client.createMessage(
                client.lobbyId || "lobby_main", // Target lobby or another agent
                "SIMPLE_MESSAGE",
                { content: "Hello from Node.js SDK!" }
            );
            await client.sendMessage(testMessage);
            console.log("Test message sent.");

            // Example: Request-response (if your lobby/agent supports this pattern)
            try {
                const response = await client.sendRequest(
                    client.lobbyId || "lobby_main", 
                    "GET_INFO_REQUEST", 
                    { item: "agent_status" }, 
                    5000 // 5-second timeout for this request
                );
                console.log("GET_INFO_RESPONSE:", response);
            } catch (error) {
                console.error("GET_INFO_REQUEST failed or timed out:", error.message);
            }

            console.log("NodeAgent will listen for messages for 20 seconds...");
            await new Promise(resolve => setTimeout(resolve, 20000));

        } else {
            console.error(`NodeAgent (${client.config.agentId}) failed to connect.`);
        }

    } catch (error) {
        console.error("Error during Node.js SDK usage:", error.message, error.stack);
    } finally {
        console.log("Stopping Node.js agent client...");
        await client.stop();
        console.log("Node.js agent client stopped.");
    }
}

if (require.main === module) {
    // This allows running the example directly using `node index.js` (if this README was an executable script)
    // For actual usage, import and call main() from your application script.
    main().catch(e => console.error("Unhandled error in main execution:", e));
}
```

## Dependencies

This SDK requires the following Node.js packages:
- `ws` (for WebSocket client)
- `uuid` (for generating unique IDs)
- `node-fetch` (for HTTP requests, especially in Node.js versions < 18)
- `yaml` (if using the example config loading from the original SDK file, not directly used by the package itself if config is passed in)

Install them in your project: `npm install ws uuid node-fetch yaml`

## Configuration Notes

The `EnhancedEcosystemClientJS` expects an `SDKConfigJS` instance. This configuration includes:
- `agentId`: Unique ID for your agent.
- `agentType`: Type of your agent (e.g., "GreeterAgentNode").
- `apiKey`: API key for authentication with the Lobby (ensure this is kept secure).
- `lobbyHttpUrl`: Full HTTP URL for the Lobby's registration endpoint (e.g., `http://localhost:8088`).
- `lobbyWsUrl`: Full WebSocket URL for the Lobby (e.g., `ws://localhost:8088`).
- `capabilities`: Array of objects describing agent capabilities.
- `reconnectInterval` (optional): Milliseconds between WebSocket reconnection attempts.
- `registrationTimeout` (optional): Milliseconds for HTTP registration request timeout.
- `websocketTimeout` (optional): Milliseconds for WebSocket connection handshake timeout.

Ensure these values, especially the URLs and API key, match your Agent Lobbi server configuration. 