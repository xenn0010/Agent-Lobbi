package go_sdk

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// Message mirrors the structure in PROTOCOL.md
type Message struct {
	MessageID       string      `json:"message_id"`
	Timestamp       string      `json:"timestamp"`
	SenderID        string      `json:"sender_id"`
	ReceiverID      string      `json:"receiver_id"`
	MessageType     string      `json:"message_type"`
	Payload         interface{} `json:"payload"` // Can be map[string]interface{} or a specific struct
	ConversationID  string      `json:"conversation_id,omitempty"`
	AuthToken       *string     `json:"auth_token,omitempty"`      // Pointer to allow null
	Priority        int         `json:"priority,omitempty"`
	RequiresAck     bool        `json:"requires_ack,omitempty"`
	AckTimeout      *int        `json:"ack_timeout,omitempty"`     // Pointer to allow null
	BroadcastScope  *string     `json:"broadcast_scope,omitempty"` // Pointer to allow null
}

// Capability mirrors the structure for agent capabilities
type Capability struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema interface{} `json:"input_schema,omitempty"`
	OutputSchema interface{} `json:"output_schema,omitempty"`
}

// SDKConfig holds the configuration for the Go SDK client
type SDKConfig struct {
	AgentID             string
	AgentType           string
	APIKey              string
	LobbyHTTPURL        string // e.g., "http://localhost:8080"
	LobbyWSURL          string // e.g., "ws://localhost:8081"
	Capabilities        []Capability
	ReconnectInterval   time.Duration
	RegistrationTimeout time.Duration
	WebsocketTimeout    time.Duration // Timeout for the WebSocket handshake
	DefaultRequestTimeout time.Duration
	Logger              *log.Logger
}

// EnhancedEcosystemClientGo is the Go SDK client for the Agent Lobbi
type EnhancedEcosystemClientGo struct {
	config         SDKConfig
	authToken      string
	lobbyID        string
	wsConn         *websocket.Conn
	isConnected    bool
	messageHandlers map[string][]func(msg Message) // message_type -> list of handlers
	pendingRequests map[string]chan Message      // conversation_id -> channel for response/error
	mutex          sync.RWMutex
	shutdownChan   chan struct{}
	wg             sync.WaitGroup
	logger         *log.Logger
	shouldReconnect bool
}

// NewSDKConfig creates a new SDKConfig with default values
func NewSDKConfig(agentID, agentType, apiKey, lobbyHTTPURL, lobbyWSURL string) (*SDKConfig, error) {
	if agentID == "" { return nil, errors.New("agentID cannot be empty") }
	if agentType == "" { return nil, errors.New("agentType cannot be empty") }
	if apiKey == "" { return nil, errors.New("apiKey cannot be empty") }
	if lobbyHTTPURL == "" { return nil, errors.New("lobbyHTTPURL cannot be empty") }
	if lobbyWSURL == "" { return nil, errors.New("lobbyWSURL cannot be empty") }

	return &SDKConfig{
		AgentID:             agentID,
		AgentType:           agentType,
		APIKey:              apiKey,
		LobbyHTTPURL:        lobbyHTTPURL,
		LobbyWSURL:          lobbyWSURL,
		Capabilities:        make([]Capability, 0),
		ReconnectInterval:   5 * time.Second,
		RegistrationTimeout: 10 * time.Second,
		WebsocketTimeout:    10 * time.Second,
		DefaultRequestTimeout: 5 * time.Second,
		Logger:              log.New(os.Stdout, fmt.Sprintf("GoSDK [%s] - ", agentID), log.LstdFlags),
	}, nil
}

// NewEnhancedEcosystemClientGo creates a new instance of the Go SDK client
func NewEnhancedEcosystemClientGo(config SDKConfig) *EnhancedEcosystemClientGo {
	lgr := config.Logger
	if lgr == nil {
		lgr = log.New(os.Stdout, fmt.Sprintf("GoSDK [%s] - ", config.AgentID), log.LstdFlags)
	}
	return &EnhancedEcosystemClientGo{
		config:         config,
		messageHandlers: make(map[string][]func(msg Message)),
		pendingRequests: make(map[string]chan Message),
		shutdownChan:   make(chan struct{}),
		logger:         lgr,
		shouldReconnect: true,
	}
}

func (c *EnhancedEcosystemClientGo) logf(format string, v ...interface{}) {
	c.logger.Printf(format, v...)
}

// RegisterAgent performs HTTP registration with the lobby
func (c *EnhancedEcosystemClientGo) registerAgent(ctx context.Context) error {
	registerURL := fmt.Sprintf("%s/api/register", c.config.LobbyHTTPURL)

	// Construct registration payload based on PROTOCOL.md
	// The protocol implies the sender_id is part of the payload for REGISTER message.
	regPayload := map[string]interface{}{
		"sender_id":    c.config.AgentID,
		"agent_id":     c.config.AgentID, // Often redundant but some systems might expect it nested
		"agent_type":   c.config.AgentType,
		"capabilities": c.config.Capabilities,
	}

	jsonPayload, err := json.Marshal(regPayload)
	if err != nil {
		return fmt.Errorf("failed to marshal registration payload: %w", err)
	}

	c.logf("Attempting registration to %s", registerURL)
	reqCtx, cancel := context.WithTimeout(ctx, c.config.RegistrationTimeout)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(reqCtx, "POST", registerURL, bytes.NewBuffer(jsonPayload))
	if err != nil {
		return fmt.Errorf("failed to create registration request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-API-Key", c.config.APIKey)

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("registration request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body) // Read body for logging, even on error
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("registration failed with status %d: %s", resp.StatusCode, string(body))
	}

	var ackMsg Message
	if err := json.Unmarshal(body, &ackMsg); err != nil {
		return fmt.Errorf("failed to unmarshal registration ACK: %w. Body: %s", err, string(body))
	}

	if ackMsg.MessageType == "REGISTER_ACK" {
		if payload, ok := ackMsg.Payload.(map[string]interface{}); ok {
			if status, ok := payload["status"].(string); ok && status == "success" {
				if token, ok := payload["auth_token"].(string); ok {
					c.mutex.Lock()
					c.authToken = token
					if lobbyID, ok := payload["lobby_id"].(string); ok {
						c.lobbyID = lobbyID
					}
					c.mutex.Unlock()
					c.logf("Registration successful. Lobby ID: %s, Auth Token: %s", c.lobbyID, c.authToken)
					return nil
				} else {
					return fmt.Errorf("auth_token missing or not a string in REGISTER_ACK payload. Payload: %+v", payload)
				}
			}
		}
	}
	return fmt.Errorf("registration ACK invalid or status not success: %+v", ackMsg)
}

// ConnectWebSocket establishes the WebSocket connection
func (c *EnhancedEcosystemClientGo) connectWebSocket(ctx context.Context) error {
	c.mutex.RLock()
	token := c.authToken
	c.mutex.RUnlock()
	if token == "" {
		return errors.New("cannot connect WebSocket: auth token not available")
	}

	wsURL := fmt.Sprintf("%s/ws/%s?token=%s", c.config.LobbyWSURL, c.config.AgentID, token)
	c.logf("Connecting to WebSocket: %s", wsURL)

	dialer := websocket.Dialer{
		HandshakeTimeout: c.config.WebsocketTimeout,
	}

	conn, resp, err := dialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		errMsg := fmt.Sprintf("WebSocket dial error: %s", err.Error())
		if resp != nil {
			bodyBytes, _ := ioutil.ReadAll(resp.Body)
			resp.Body.Close()
			errMsg = fmt.Sprintf("%s. Response Status: %s, Body: %s", errMsg, resp.Status, string(bodyBytes))
		}
		return errors.New(errMsg)
	}

	c.mutex.Lock()
	c.wsConn = conn
	c.isConnected = true
	c.mutex.Unlock()
	c.logf("WebSocket connection established.")

	c.wg.Add(1)
	go c.readLoop()

	return nil
}

func (c *EnhancedEcosystemClientGo) readLoop() {
	defer c.wg.Done()
	defer func() {
		c.mutex.Lock()
		if c.wsConn != nil {
			c.wsConn.Close()
			c.wsConn = nil
		}
		c.isConnected = false
		c.mutex.Unlock()
		c.logf("Read loop terminated.")
	}()

	for {
		select {
		case <-c.shutdownChan:
			return
		default:
			c.mutex.RLock()
			conn := c.wsConn
			c.mutex.RUnlock()
			if conn == nil {
				// Connection lost, attempt reconnect if enabled
				// Reconnect logic will be handled by the Start method's loop
				c.logf("Read loop: connection is nil, exiting to allow reconnect.")
				return // Exit readLoop, Start will handle reconnect
			}

			conn.SetReadDeadline(time.Now().Add(c.config.ReconnectInterval * 2)) // Example deadline
			_, messageBytes, err := conn.ReadMessage()
			if err != nil {
				c.logf("Read error: %v. Connection may be closed.", err)
				// Notify connection loss to trigger reconnect logic in Start() more explicitly if desired
				// or simply let the loop in Start() handle it after readLoop exits.
				c.mutex.Lock()
				c.isConnected = false // Mark as not connected
				if c.wsConn != nil {
					c.wsConn.Close() // Ensure it's closed
					c.wsConn = nil
				}
				c.mutex.Unlock()
				return // Exit readLoop to trigger reconnection in Start
			}

			var msg Message
			if err := json.Unmarshal(messageBytes, &msg); err != nil {
				c.logf("Failed to unmarshal incoming message: %v. Body: %s", err, string(messageBytes))
				continue
			}
			c.logf("Received message: Type %s, ID %s", msg.MessageType, msg.MessageID)
			c.handleIncomingMessage(msg)
		}
	}
}

func (c *EnhancedEcosystemClientGo) handleIncomingMessage(msg Message) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	// Handle direct request-response matching
	if ch, ok := c.pendingRequests[msg.ConversationID]; ok {
		select {
		case ch <- msg: // Send if channel is ready
		default: // Avoid blocking if channel not ready (e.g. timeout already occurred)
			c.logf("Pending request channel for conv %s not ready or already handled.", msg.ConversationID)
		}
		// Note: The channel itself is not closed here; sendRequest will handle its lifecycle.
		// It is removed from pendingRequests by sendRequest on timeout or reception.
	}

	// Generic message type handlers
	if handlers, ok := c.messageHandlers[msg.MessageType]; ok {
		for _, handler := range handlers {
			go func(h func(Message), m Message) { // Run handler in a new goroutine
				defer func() {
					if r := recover(); r != nil {
						c.logf("Panic recovered in message handler for type %s: %v", m.MessageType, r)
					}
				}()
				h(m)
			}(handler, msg)
		}
	}
	if handlers, ok := c.messageHandlers["*"]; ok { // Wildcard handler
		for _, handler := range handlers {
			go func(h func(Message), m Message) {
				defer func() {
					if r := recover(); r != nil {
						c.logf("Panic recovered in wildcard message handler: %v", r)
					}
				}()
				h(m)
			}(handler, msg)
		}
	}
}

// Start initiates the connection and message handling loops.
// It will attempt to register and connect, and will retry on failure if shouldReconnect is true.
func (c *EnhancedEcosystemClientGo) Start(ctx context.Context) error {
	c.mutex.Lock()
	c.shouldReconnect = true
	c.mutex.Unlock()

	c.logf("Starting client...")

	go func() {
		for {
			select {
			case <-c.shutdownChan:
				c.logf("Client shutdown initiated, exiting start loop.")
				return
			case <-ctx.Done():
				c.logf("Context cancelled, shutting down client.")
				c.Stop() // Ensure proper shutdown
				return
			default:
				c.mutex.RLock()
				isConnected := c.isConnected
				shouldReconnect := c.shouldReconnect
				c.mutex.RUnlock()

				if !isConnected && shouldReconnect {
					c.logf("Not connected, attempting to register and connect...")
					if err := c.registerAgent(ctx); err != nil {
						c.logf("Registration attempt failed: %v. Retrying in %s...", err, c.config.ReconnectInterval)
						time.Sleep(c.config.ReconnectInterval)
						continue
					}

					if err := c.connectWebSocket(ctx); err != nil {
						c.logf("WebSocket connection attempt failed: %v. Retrying in %s...", err, c.config.ReconnectInterval)
						c.mutex.Lock() // ensure wsConn is closed if partially opened
						if c.wsConn != nil {
							c.wsConn.Close()
							c.wsConn = nil
						}
						c.isConnected = false
						c.mutex.Unlock()
						time.Sleep(c.config.ReconnectInterval)
						continue
					}
					c.logf("Successfully re-established connection.")
				} else if !shouldReconnect && !isConnected {
					c.logf("Not connected and reconnection is disabled. Exiting start loop.")
					return
				}
			}
			// If connected, just wait or do periodic health checks if needed
			time.Sleep(1 * time.Second) // Check connection status periodically
		}
	}()
	return nil
}

// Stop gracefully shuts down the client
func (c *EnhancedEcosystemClientGo) Stop() {
	c.logf("Stopping client...")
	c.mutex.Lock()
	c.shouldReconnect = false
	close(c.shutdownChan) // Signal all loops to exit
	c.mutex.Unlock()

	c.mutex.RLock() // Protect wsConn access
	conn := c.wsConn
	c.mutex.RUnlock()

	if conn != nil {
		// Attempt to send a close message
		deadline := time.Now().Add(5 * time.Second)
		err := conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, "Client stopping"), deadline)
		if err != nil {
			c.logf("Error sending close message: %v", err)
		}
		conn.Close() // Force close
	}

	c.mutex.Lock()
	c.isConnected = false
	// Clean up pending requests
	for convID, ch := range c.pendingRequests {
		// Sending error on channel might panic if receiver already gone, use non-blocking send or just close
		// Or simply log and remove
		close(ch) // Close the channel to signal waiters
		delete(c.pendingRequests, convID)
	}
	c.mutex.Unlock()

	c.wg.Wait() // Wait for readLoop to finish
	c.logf("Client stopped.")
}

// CreateMessage utility to construct a message object
func (c *EnhancedEcosystemClientGo) CreateMessage(receiverID, messageType string, payload interface{}, conversationID string) Message {
	msgID, _ := uuid.NewRandom()
	convID := conversationID
	if convID == "" {
		convUUID, _ := uuid.NewRandom()
		convID = convUUID.String()
	}

	return Message{
		MessageID:      msgID.String(),
		Timestamp:      time.Now().UTC().Format(time.RFC3339Nano),
		SenderID:       c.config.AgentID,
		ReceiverID:     receiverID,
		MessageType:    messageType,
		Payload:        payload,
		ConversationID: convID,
		Priority:       2, // Default priority
	}
}

// SendMessage sends a message over the WebSocket connection
func (c *EnhancedEcosystemClientGo) SendMessage(msg Message) error {
	c.mutex.RLock()
	if !c.isConnected || c.wsConn == nil {
		c.mutex.RUnlock()
		return errors.New("cannot send message: not connected")
	}
	conn := c.wsConn
	c.mutex.RUnlock()

	jsonMsg, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	if err := conn.WriteMessage(websocket.TextMessage, jsonMsg); err != nil {
		c.logf("Failed to send message: %v. Connection may be lost.", err)
		// Trigger reconnect explicitly or rely on readLoop to detect closure
		c.mutex.Lock()
		c.isConnected = false
		if c.wsConn != nil { c.wsConn.Close(); c.wsConn = nil }
		c.mutex.Unlock()
		return fmt.Errorf("failed to write message: %w", err)
	}
	c.logf("Sent message: Type %s to %s, ConvID %s", msg.MessageType, msg.ReceiverID, msg.ConversationID)
	return nil
}

// SendRequest sends a request message and waits for a response or error with a timeout.
func (c *EnhancedEcosystemClientGo) SendRequest(ctx context.Context, receiverID, requestType string, payload interface{}) (Message, error) {
	convIDUUID, _ := uuid.NewRandom()
	conversationID := convIDUUID.String()
	msg := c.CreateMessage(receiverID, requestType, payload, conversationID)

	respChan := make(chan Message, 1) // Buffered channel of 1

	c.mutex.Lock()
	c.pendingRequests[conversationID] = respChan
	c.mutex.Unlock()

	defer func() {
		c.mutex.Lock()
		close(c.pendingRequests[conversationID]) // Close the channel
		delete(c.pendingRequests, conversationID)
		c.mutex.Unlock()
	}()

	if err := c.SendMessage(msg); err != nil {
		return Message{}, fmt.Errorf("failed to send request message: %w", err)
	}

	// Use context for timeout if provided, otherwise default timeout
	timeout := c.config.DefaultRequestTimeout
	var reqCtx context.Context
	var cancel context.CancelFunc
	if ctx != nil {
		if deadline, ok := ctx.Deadline(); ok {
			timeout = time.Until(deadline)
			if timeout < 0 {
				timeout = c.config.DefaultRequestTimeout // or return error if already past deadline
			}
		}
		reqCtx, cancel = context.WithTimeout(ctx, timeout)
	} else {
		reqCtx, cancel = context.WithTimeout(context.Background(), timeout)
	}
	defer cancel()

	select {
	case response := <-respChan:
		if response.MessageType == "ERROR" {
			return response, fmt.Errorf("received error response from lobby: %+v", response.Payload)
		}
		return response, nil
	case <-reqCtx.Done(): // This handles timeout
		return Message{}, fmt.Errorf("request timed out for conversation %s: %w", conversationID, reqCtx.Err())
	}
}

// RegisterMessageHandler adds a handler function for a specific message type or "*" for all types.
func (c *EnhancedEcosystemClientGo) RegisterMessageHandler(messageType string, handler func(msg Message)) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.messageHandlers[messageType] = append(c.messageHandlers[messageType], handler)
	c.logf("Registered handler for message type: %s", messageType)
}

// Example of how to start and use the Go SDK client
/*
func main() {
    // Trap SIGINT to trigger a shutdown.
    signals := make(chan os.Signal, 1)
    signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

    cfg, err := go_sdk.NewSDKConfig(
        "go_agent_007",
        "GoAgent",
        "test_api_key",
        "http://localhost:8088",
        "ws://localhost:8081",
    )
    if err != nil {
        log.Fatalf("Failed to create SDK config: %v", err)
    }
    cfg.Capabilities = []go_sdk.Capability{{
        Name: "go_time_asker", Description: "Asks for time in Go!",
    }}
    // cfg.Logger = log.New(os.Stdout, "MyGoAgent - ", log.LstdFlags|log.Lshortfile)

    client := go_sdk.NewEnhancedEcosystemClientGo(*cfg)

    // Register a general handler for INFO messages
    client.RegisterMessageHandler("INFO", func(msg go_sdk.Message) {
        client.Logf("INFO Handler received: %+v\n", msg)
    })
    client.RegisterMessageHandler("RESPONSE", func(msg go_sdk.Message) {
        client.Logf("RESPONSE Handler received: %+v\n", msg)
    })
     client.RegisterMessageHandler("ERROR", func(msg go_sdk.Message) {
        client.Logf("ERROR Handler received: %+v\n", msg)
    })


    ctx, cancel := context.WithCancel(context.Background())
    defer cancel() // Ensure resources are cleaned up

    if err := client.Start(ctx); err != nil {
        log.Fatalf("Failed to start client: %v", err)
    }

    // Allow time for connection
    time.Sleep(2 * time.Second)

    // Check if connected - needs a method like IsConnected()
    // For now, we assume connection if Start() didn't error out immediately for demo

    // Send an INFO message
    infoMsg := client.CreateMessage(client.GetLobbyID(), "INFO", map[string]string{"data": "Hello from Go agent!"}, "")
    if err := client.SendMessage(infoMsg); err != nil {
        client.Logf("Failed to send INFO message: %v", err)
    }

    // Send a REQUEST for time
    timeRequestPayload := map[string]string{"action": "get_time"}
    go func() {
        respMsg, err := client.SendRequest(context.TODO(), client.GetLobbyID(), "REQUEST", timeRequestPayload)
        if err != nil {
            client.Logf("Time request failed: %v", err)
        } else {
            client.Logf("Time request successful: %+v", respMsg)
        }
    }()

    // Send an unknown request
     go func() {
        unknownRequestPayload := map[string]string{"action": "do_something_silly"}
        respMsg, err := client.SendRequest(context.TODO(), client.GetLobbyID(), "REQUEST", unknownRequestPayload)
        if err != nil {
            client.Logf("Unknown action request failed (as expected for mock): %v", err)
        } else {
            client.Logf("Unknown action request successful (unexpected for mock): %+v", respMsg)
        }
    }()


    <-signals // Wait for shutdown signal
    log.Println("Shutdown signal received, stopping client...")
    client.Stop()
    log.Println("Client stopped.")
}

// Helper to get LobbyID (should be added to client)
func (c *EnhancedEcosystemClientGo) GetLobbyID() string {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    return c.lobbyID
}
*/ 