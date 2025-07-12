# Agent Lobbi Go SDK

This SDK provides a Go client for interacting with the Agent Lobbi ecosystem.

## Prerequisites

- Go (version 1.18 or higher recommended)

## Installation & Setup

1.  Navigate to the `src/sdk/go_sdk` directory in your terminal.
2.  Initialize the Go module (if not already done). Replace `yourusername/agent_lobby` with your actual repository path if you plan to publish this module:
    ```bash
    go mod init github.com/yourusername/agent_lobby/sdk/go_sdk
    ```
3.  Tidy the dependencies to fetch `gorilla/websocket` and other potential future dependencies:
    ```bash
    go mod tidy
    ```

## Usage

Import the SDK into your Go project:

```go
import (
    "log"
    "time"
    "os"
    "os/signal"
    "syscall"

    "github.com/yourusername/agent_lobby/sdk/go_sdk" // Update with your module path
)

func main() {
    logger := log.New(os.Stdout, "GoAgent: ", log.LstdFlags|log.Lshortfile)

    // Configuration typically loaded from a config file or environment variables
    // For example, if you use a config/config.yaml similar to other SDKs:
    // cfg, err := LoadConfigFromYAML("../../../../config/config.yaml") // Adjust path
    // if err != nil {
    //    logger.Fatalf("Failed to load config: %v", err)
    // }
    // agentID := cfg.SDK.AgentIDGoExample // or generate one
    // apiKey := cfg.SDK.APIKeyGoExample
    // lobbyHTTPURL := fmt.Sprintf("http://%s:%d", cfg.Lobby.Host, cfg.Lobby.HTTPPort)
    // lobbyWSURL := fmt.Sprintf("ws://%s:%d", cfg.Lobby.Host, cfg.Lobby.HTTPPort)

    // For this example, using placeholders or environment variables:
    agentID := os.Getenv("AGENT_ID")
    if agentID == "" {
        agentID = "go_agent_" + go_sdk.GenerateUUID()[:8]
    }
    apiKey := os.Getenv("API_KEY")
    if apiKey == "" {
        apiKey = "go-sdk-secret-key" // Replace with your actual API key
    }
    lobbyHTTPURL := os.Getenv("LOBBY_HTTP_URL")
    if lobbyHTTPURL == "" {
        lobbyHTTPURL = "http://localhost:8088"
    }
    lobbyWSURL := os.Getenv("LOBBY_WS_URL")
    if lobbyWSURL == "" {
        lobbyWSURL = "ws://localhost:8088"
    }

    config := go_sdk.SDKConfigGo{
        AgentID:      agentID,
        AgentType:    "GoExampleAgent",
        APIKey:       apiKey,
        LobbyHTTPURL: lobbyHTTPURL,
        LobbyWSURL:   lobbyWSURL,
        Capabilities: []go_sdk.Capability{
            {Name: "go_echo_service", Version: "1.0"},
            {Name: "go_status_reporter", Version: "1.1"},
        },
        ReconnectInterval:   5 * time.Second,
        RegistrationTimeout: 10 * time.Second,
        WebsocketTimeout:    10 * time.Second,
        Logger:              logger,
    }

    client, err := go_sdk.NewEnhancedEcosystemClientGo(config)
    if err != nil {
        logger.Fatalf("Failed to create Go SDK client: %v", err)
    }

    // Register a message handler for a specific type
    client.RegisterMessageHandler("LOBBY_ANNOUNCEMENT", func(msg go_sdk.Message) {
        logger.Printf("Received LOBBY_ANNOUNCEMENT: %+v\n", msg.Payload)
        // Process announcement
    })

    // Register a wildcard handler for all other messages
    client.RegisterMessageHandler("*", func(msg go_sdk.Message) {
        logger.Printf("Go Agent (%s) received generic message: Type=%s, Payload=%+v\n", client.Config.AgentID, msg.MessageType, msg.Payload)
    })

    // Start the client (registers and connects WebSocket)
    err = client.Start()
    if err != nil {
        logger.Fatalf("Failed to start Go SDK client: %v", err)
    }

    // Wait for connection (optional, depends on how `Start` is implemented or if events are used)
    time.Sleep(2 * time.Second)

    if client.IsConnected() {
        logger.Printf("Go Agent (%s) connected and registered. Lobby ID: %s\n", client.Config.AgentID, client.LobbyID())

        // Example: Send a message
        echoPayload := map[string]interface{}{"content": "Hello from Go SDK!"}
        echoMessage := client.CreateMessage(client.LobbyID(), "ECHO_REQUEST_GO", echoPayload, "")
        err = client.SendMessage(echoMessage)
        if err != nil {
            logger.Printf("Error sending ECHO_REQUEST_GO: %v\n", err)
        } else {
            logger.Println("Test ECHO_REQUEST_GO message sent.")
        }

        // Example: Send a request and wait for a response
        getTimePayload := map[string]interface{}{"timezone": "PST"}
        responseMsg, err := client.SendRequest(client.LobbyID(), "GET_TIME_GO", getTimePayload, 5*time.Second)
        if err != nil {
            logger.Printf("GET_TIME_GO request failed: %v\n", err)
        } else {
            logger.Printf("Received GET_TIME_GO response: %+v\n", responseMsg)
        }

        logger.Println("Go Agent will stay alive to listen for messages. Press Ctrl+C to stop.")
        // Keep the agent running until interrupted
        sigChan := make(chan os.Signal, 1)
        signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
        <-sigChan // Block until a signal is received

    } else {
        logger.Printf("Go Agent (%s) failed to connect.\n", client.Config.AgentID)
    }

    logger.Println("Stopping Go SDK client...")
    client.Stop()
    logger.Println("Go SDK client stopped.")
}

// You would need a LoadConfigFromYAML function if you want to load from project's config.yaml
// Example (highly simplified, needs error handling and proper struct mapping):
// type ProjectLobbyConfig struct { Host string; HTTPPort int `yaml:"http_port"`}
// type ProjectSDKConfig struct { AgentIDGoExample string `yaml:"agent_id_go_example"`; APIKeyGoExample string `yaml:"api_key_go_example"` }
// type ProjectConfig struct { Lobby ProjectLobbyConfig; SDK ProjectSDKConfig }
// func LoadConfigFromYAML(path string) (*ProjectConfig, error) { /* ... read and parse YAML ... */ return nil, nil }

```

## SDK Structure (ecosystem_sdk.go)

Refer to `ecosystem_sdk.go` for the `SDKConfigGo`, `EnhancedEcosystemClientGo`, `Message` structures, and available methods.

Key components:
- `NewEnhancedEcosystemClientGo(config SDKConfigGo)`: Constructor for the client.
- `client.Start()`: Initiates registration and WebSocket connection.
- `client.Stop()`: Disconnects and cleans up.
- `client.SendMessage(msg Message)`: Sends a pre-formatted message.
- `client.SendRequest(receiverID, requestType string, payload map[string]interface{}, timeout time.Duration)`: Sends a request and waits for a response.
- `client.CreateMessage(...)`: Helper to construct valid messages.
- `client.RegisterMessageHandler(messageType string, handler MessageHandlerFunc)`: Registers callbacks for incoming messages.
- `client.IsConnected() bool`: Checks WebSocket connection status.

## Configuration

The `SDKConfigGo` struct requires fields similar to other SDKs:
- `AgentID`, `AgentType`, `APIKey`
- `LobbyHTTPURL`, `LobbyWSURL`
- `Capabilities` (slice of `Capability` structs)
- Timeouts and intervals (`ReconnectInterval`, `RegistrationTimeout`, `WebsocketTimeout`)
- `Logger` (an instance of `*log.Logger` for logging)

Ensure these are correctly set for your Lobby environment. 