#!/usr/bin/env python3
"""
CORE AGENT LOBBI
================
This is the central nervous system of the agent collaboration framework.
It has been corrected to properly start and manage WebSocket connections.
"""

import asyncio
import json
import logging
import time
import websockets
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import uuid
import sys
import os
from dataclasses import asdict

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.message import Message, MessageType, MessagePriority
    from core.collaboration_engine import CollaborationEngine, WorkflowStatus
except ImportError:
    print("CRITICAL: Could not import core components. Make sure you are running from the project root.", file=sys.stderr)
    sys.exit(1)

# Create placeholder classes for missing components
class SecurityModule:
    """Placeholder security module."""
    def __init__(self):
        pass

class ConsensusSystem:
    """Placeholder consensus system."""
    def __init__(self):
        pass

class DataProtectionLayer:
    """Placeholder data protection layer."""
    def __init__(self):
        pass

class ConnectionRecovery:
    """Placeholder connection recovery system."""
    def __init__(self):
        pass

class AgentTracking:
    """Placeholder agent tracking system."""
    def __init__(self):
        pass

logger = logging.getLogger(__name__)

class Lobby:
    """The main lobby for agent collaboration. This is the corrected version."""
    
    def __init__(self, host: str = "localhost", http_port: int = 8080, ws_port: int = 8081):
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.lobby_id = f"lobby_{uuid.uuid4().hex[:8]}"
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.tasks: List[asyncio.Task] = []
        self.http_server = None
        self.ws_server = None
        self.running = True

        self._priority_queues: List[asyncio.PriorityQueue] = [asyncio.PriorityQueue() for _ in range(5)]
        
        # Centralized Connection Management for live WebSocket agents
        self.live_agent_connections: Dict[str, Any] = {}
        self.agent_connections = self.live_agent_connections  # Alias for compatibility
        
        # **FIX: Add delegation tracking for API bridge**
        self.delegation_to_workflow = {}  # Maps delegation_id to workflow_id
        
        # Initialize components
        self.collaboration_engine = CollaborationEngine(self)
        self.security_module = SecurityModule()
        self.consensus_system = ConsensusSystem()
        self.data_protection_layer = DataProtectionLayer()
        self.connection_recovery = ConnectionRecovery()
        self.agent_tracking = AgentTracking()
        # Store path information during WebSocket handshake
        self.websocket_paths = {}

    async def start(self):
        """Start all lobby services, including the crucial WebSocket server."""
        logger.info("START Starting Agent Lobbi services...")
        self.tasks.append(asyncio.create_task(self._process_message_queues()))
        self.tasks.append(asyncio.create_task(self.collaboration_engine.start()))
        
        # Check if ports are available before starting
        if not await self._check_port_availability(self.http_port):
            logger.error(f"ERROR Port {self.http_port} is already in use. Please check for existing instances.")
            logger.info("TIP Try running: netstat -an | findstr :9101 (Windows) or lsof -i :9101 (Linux/Mac)")
            raise Exception(f"Port {self.http_port} is already in use")
        
        if not await self._check_port_availability(self.ws_port):
            logger.error(f"ERROR Port {self.ws_port} is already in use. Please check for existing instances.")
            logger.info("TIP Try running: netstat -an | findstr :9102 (Windows) or lsof -i :9102 (Linux/Mac)")
            raise Exception(f"Port {self.ws_port} is already in use")
        
        try:
            # Start HTTP server with better error handling
            logger.info(f"SERVER Starting HTTP server on {self.host}:{self.http_port}")
            try:
                self.http_server = await asyncio.start_server(
                    self.handle_http_request, self.host, self.http_port
                )
                logger.info(f"OK HTTP server started successfully on {self.host}:{self.http_port}")
            except OSError as e:
                if e.errno == 10048:  # Windows "Address already in use"
                    logger.error(f"ERROR HTTP port {self.http_port} is already in use")
                    logger.info("TIP Another instance of Agent Lobbi might be running")
                    raise Exception(f"HTTP port {self.http_port} is already in use")
                else:
                    logger.error(f"ERROR Failed to start HTTP server: {e}")
                    raise
            
            # Start WebSocket server with better error handling
            logger.info(f"CONNECT Starting WebSocket server on {self.host}:{self.ws_port}")
            try:
                self.ws_server = await websockets.serve(
                    self.handle_websocket_connection,
                    self.host,
                    self.ws_port,
                    process_request=self.process_websocket_request
                )
                logger.info(f"OK WebSocket server started successfully on {self.host}:{self.ws_port}")
            except OSError as e:
                if e.errno == 10048:  # Windows "Address already in use"
                    logger.error(f"ERROR WebSocket port {self.ws_port} is already in use")
                    logger.info("TIP Another instance of Agent Lobbi might be running")
                    # Clean up HTTP server if it was started
                    if self.http_server:
                        self.http_server.close()
                        await self.http_server.wait_closed()
                    raise Exception(f"WebSocket port {self.ws_port} is already in use")
                else:
                    logger.error(f"ERROR Failed to start WebSocket server: {e}")
                    # Clean up HTTP server if it was started
                    if self.http_server:
                        self.http_server.close()
                        await self.http_server.wait_closed()
                    raise
            
            logger.info("OK All Agent Lobbi services started successfully!")
            logger.info(f"   SERVER HTTP API available at: http://{self.host}:{self.http_port}")
            logger.info(f"   CONNECT WebSocket server available at: ws://{self.host}:{self.ws_port}")
            
        except Exception as e:
            logger.error(f"ERROR Failed to start Agent Lobbi services: {e}")
            # Clean up any partially started services
            await self._cleanup_servers()
            raise

    async def _check_port_availability(self, port: int) -> bool:
        """Check if a port is available for binding"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, port))
            sock.close()
            return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    async def _cleanup_servers(self):
        """Clean up any partially started servers"""
        if self.http_server:
            try:
                logger.info("CLEANUP Cleaning up HTTP server...")
                self.http_server.close()
                await self.http_server.wait_closed()
            except Exception as e:
                logger.error(f"Error cleaning up HTTP server: {e}")
        
        if self.ws_server:
            try:
                logger.info("CLEANUP Cleaning up WebSocket server...")
                self.ws_server.close()
                await self.ws_server.wait_closed()
            except Exception as e:
                logger.error(f"Error cleaning up WebSocket server: {e}")



    async def stop(self):
        """Stop all lobby services gracefully."""
        logger.info("STOP Stopping Agent Lobbi services...")
        self.running = False
        if self.http_server:
            logger.info("SERVER Closing HTTP server...")
            self.http_server.close()
            await self.http_server.wait_closed()
        if self.ws_server:
            logger.info("CONNECT Closing WebSocket server...")
            self.ws_server.close()
            await self.ws_server.wait_closed()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("OK Lobby stopped.")

    async def shutdown(self):
        """Gracefully shuts down the lobby and all its components. For Railway."""
        logger.info("Initiating Lobby shutdown sequence for Railway deployment...")
        await self.stop()
        logger.info("Lobby has been shut down successfully.")

    async def handle_http_request(self, reader, writer):
        """Handles HTTP requests for the lobby API."""
        try:
            # Read the request line
            request_line_bytes = await reader.readuntil(b'\r\n')
            request_line = request_line_bytes.decode().strip()
            
            if not request_line:
                self._send_http_response(writer, 400, {"status": "error", "message": "Invalid request"})
                return
            
            try:
                method, path, _ = request_line.split(' ', 2)
            except ValueError:
                self._send_http_response(writer, 400, {"status": "error", "message": "Invalid request line"})
                return

            # Read headers
            headers = {}
            while True:
                header_line = await reader.readuntil(b'\r\n')
                header_line = header_line.decode().strip()
                if not header_line:  # Empty line indicates end of headers
                    break
                if ':' in header_line:
                    key, value = header_line.split(':', 1)
                    headers[key.strip().lower()] = value.strip()

            # Handle OPTIONS request for CORS
            if method == 'OPTIONS':
                self._send_http_response(writer, 200, {})
                return

            # Health Check
            if method == 'GET' and path in ['/health', '/api/health']:
                health_data = {
                    "status": "ok", 
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "lobby_id": self.lobby_id,
                    "agents_count": len(self.agents),
                    "http_port": self.http_port,
                    "ws_port": self.ws_port
                }
                self._send_http_response(writer, 200, health_data)
                return

            # A2A Protocol: Agent Discovery Endpoint
            if method == 'GET' and path == '/.well-known/agent.json':
                agent_card = {
                    "name": "Agent Lobby Hub",
                    "description": "Enhanced Agent Lobby with neuromorphic learning and A2A+ protocol support",
                    "version": "2.0.0",
                    "url": f"http://{self.host}:{self.http_port}",
                    "capabilities": {
                        "streaming": True,
                        "pushNotifications": True,
                        "neuromorphic_learning": True,
                        "collective_intelligence": True,
                        "reputation_system": True,
                        "multi_agent_collaboration": True,
                        "real_time_metrics": True
                    },
                    "skills": [
                        "agent_coordination",
                        "task_delegation", 
                        "workflow_orchestration",
                        "reputation_management",
                        "collective_intelligence"
                    ],
                    "authentication": {
                        "schemes": ["bearer", "api_key"]
                    },
                    "endpoints": {
                        "register": f"http://{self.host}:{self.http_port}/api/agents/register",
                        "discover": f"http://{self.host}:{self.http_port}/api/a2a/discover",
                        "delegate": f"http://{self.host}:{self.http_port}/api/a2a/delegate",
                        "communicate": f"http://{self.host}:{self.http_port}/api/a2a/communicate",
                        "status": f"http://{self.host}:{self.http_port}/api/a2a/status"
                    },
                    "extensions": {
                        "agent_lobby": {
                            "platform": "Agent Lobby",
                            "enhanced_features": [
                                "Neuromorphic agent selection",
                                "Collective intelligence",
                                "Reputation-based routing",
                                "Real-time collaboration",
                                "Advanced metrics and analytics"
                            ],
                            "agent_count": len(self.agents),
                            "protocols": ["Agent Lobby Native", "A2A", "WebSocket", "HTTP"]
                        }
                    }
                }
                self._send_http_response(writer, 200, agent_card)
                return

            # A2A Protocol: Agent Discovery
            if method == 'GET' and path == '/api/a2a/discover':
                a2a_agents = []
                for agent_id, agent_data in self.agents.items():
                    a2a_agent = {
                        "agent_id": agent_id,
                        "name": agent_data.get("name", agent_id),
                        "description": f"Agent Lobby enhanced agent: {agent_data.get('agent_type', 'unknown')}",
                        "capabilities": agent_data.get("capabilities", []),
                        "agent_type": agent_data.get("agent_type"),
                        "url": f"http://{self.host}:{self.http_port}/api/a2a/agents/{agent_id}",
                        "registered_at": agent_data.get("registered_at"),
                        "status": "online" if agent_id in self.live_agent_connections else "offline",
                        "protocols": ["A2A", "Agent Lobby Native"],
                        "extensions": {
                            "agent_lobby": {
                                "reputation_score": 0.85,  # TODO: Get from reputation system
                                "total_tasks_completed": 0,  # TODO: Get from tracking system
                                "success_rate": 0.95  # TODO: Get from metrics
                            }
                        }
                    }
                    a2a_agents.append(a2a_agent)
                
                discovery_response = {
                    "agents": a2a_agents,
                    "total_count": len(a2a_agents),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "lobby_info": {
                        "lobby_id": self.lobby_id,
                        "platform": "Agent Lobby A2A+",
                        "version": "2.0.0"
                    }
                }
                self._send_http_response(writer, 200, discovery_response)
                return

            # A2A Protocol: Individual Agent Info
            if method == 'GET' and path.startswith('/api/a2a/agents/'):
                agent_id = path.split('/')[-1]
                if agent_id in self.agents:
                    agent_data = self.agents[agent_id]
                    agent_info = {
                        "agent_id": agent_id,
                        "name": agent_data.get("name", agent_id),
                        "description": f"Agent Lobby enhanced agent: {agent_data.get('agent_type', 'unknown')}",
                        "capabilities": agent_data.get("capabilities", []),
                        "agent_type": agent_data.get("agent_type"),
                        "registered_at": agent_data.get("registered_at"),
                        "status": "online" if agent_id in self.live_agent_connections else "offline",
                        "protocols": ["A2A", "Agent Lobby Native"],
                        "endpoints": {
                            "communicate": f"http://{self.host}:{self.http_port}/api/a2a/communicate/{agent_id}",
                            "delegate": f"http://{self.host}:{self.http_port}/api/a2a/delegate/{agent_id}"
                        },
                        "extensions": {
                            "agent_lobby": {
                                "reputation_score": 0.85,
                                "total_tasks_completed": 0,
                                "success_rate": 0.95,
                                "neuromorphic_features": True,
                                "collective_intelligence": True
                            }
                        }
                    }
                    self._send_http_response(writer, 200, agent_info)
                else:
                    self._send_http_response(writer, 404, {"status": "error", "message": f"Agent {agent_id} not found"})
                return

            # Get all agents
            if method == 'GET' and path == '/api/agents':
                self._send_http_response(writer, 200, {"agents": list(self.agents.values())})
                return

            # Register agent
            if method == 'POST' and path == '/api/agents/register':
                # Read body if present
                content_length = int(headers.get('content-length', 0))
                body_data = {}
                
                if content_length > 0:
                    body_bytes = await reader.readexactly(content_length)
                    try:
                        body_data = json.loads(body_bytes.decode())
                    except json.JSONDecodeError:
                        self._send_http_response(writer, 400, {"status": "error", "message": "Invalid JSON"})
                        return
                
                if body_data and 'agent_id' in body_data:
                    result = await self.register_agent(body_data)
                    self._send_http_response(writer, 200, result)
                else:
                    self._send_http_response(writer, 400, {"status": "error", "message": "Invalid registration data"})
                return

            # A2A Protocol: Task Delegation
            if method == 'POST' and path.startswith('/api/a2a/delegate'):
                content_length = int(headers.get('content-length', 0))
                body_data = {}
                
                if content_length > 0:
                    body_bytes = await reader.readexactly(content_length)
                    try:
                        body_data = json.loads(body_bytes.decode())
                    except json.JSONDecodeError:
                        self._send_http_response(writer, 400, {"status": "error", "message": "Invalid JSON"})
                        return

                # Extract target agent from path if specified
                target_agent_id = None
                if path != '/api/a2a/delegate':
                    target_agent_id = path.split('/')[-1]

                task_result = await self.handle_a2a_delegation(body_data, target_agent_id)
                status_code = 200 if task_result.get("status") == "success" else 400
                self._send_http_response(writer, status_code, task_result)
                return

            # A2A Protocol: Agent Communication
            if method == 'POST' and path.startswith('/api/a2a/communicate'):
                content_length = int(headers.get('content-length', 0))
                body_data = {}
                
                if content_length > 0:
                    body_bytes = await reader.readexactly(content_length)
                    try:
                        body_data = json.loads(body_bytes.decode())
                    except json.JSONDecodeError:
                        self._send_http_response(writer, 400, {"status": "error", "message": "Invalid JSON"})
                        return

                # Extract target agent from path if specified
                target_agent_id = None
                if path != '/api/a2a/communicate':
                    target_agent_id = path.split('/')[-1]

                comm_result = await self.handle_a2a_communication(body_data, target_agent_id)
                status_code = 200 if comm_result.get("status") == "success" else 400
                self._send_http_response(writer, status_code, comm_result)
                return

            # A2A Protocol: Task Polling for HTTP-only agents
            if method == 'GET' and path.startswith('/api/a2a/tasks/'):
                agent_id = path.split('/')[-1]
                tasks_result = await self.handle_a2a_task_polling(agent_id)
                self._send_http_response(writer, 200, tasks_result)
                return

            # A2A Protocol: Status Check
            if method == 'GET' and path == '/api/a2a/status':
                status_data = {
                    "lobby_status": "operational",
                    "lobby_id": self.lobby_id,
                    "platform": "Agent Lobby A2A+",
                    "version": "2.0.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_count": len(self.agents),
                    "active_connections": len(self.live_agent_connections),
                    "protocols_supported": ["A2A", "Agent Lobby Native", "WebSocket", "HTTP"],
                    "capabilities": {
                        "neuromorphic_learning": True,
                        "collective_intelligence": True,
                        "reputation_system": True,
                        "real_time_metrics": True,
                        "advanced_collaboration": True
                    }
                }
                self._send_http_response(writer, 200, status_data)
                return

            # Default: Not Found
            self._send_http_response(writer, 404, {"status": "error", "message": "Not Found"})
            
        except asyncio.IncompleteReadError:
            logger.warning("Client disconnected before request completed")
        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            try:
                if not writer.is_closing():
                    self._send_http_response(writer, 500, {"status": "error", "message": "Internal Server Error"})
            except Exception:
                pass  # Ignore errors when trying to send error response
        finally:
            try:
                if not writer.is_closing():
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
            except Exception:
                pass  # Ignore cleanup errors

    def _send_http_response(self, writer, status_code, body):
        """Send an HTTP response."""
        try:
            status_messages = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"}
            status_text = status_messages.get(status_code, "OK")
            
            response_body = json.dumps(body)
            response_headers = [
                f"HTTP/1.1 {status_code} {status_text}",
                "Content-Type: application/json",
                "Access-Control-Allow-Origin: *",
                "Access-Control-Allow-Methods: GET, POST, OPTIONS",
                "Access-Control-Allow-Headers: Content-Type",
                f"Content-Length: {len(response_body)}",
                "Connection: close",
                ""  # Empty line to separate headers from body
            ]
            
            response = "\r\n".join(response_headers) + "\r\n" + response_body
            writer.write(response.encode())
            
        except Exception as e:
            logger.error(f"Error sending HTTP response: {e}")
            # Send minimal error response
            error_response = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n"
            writer.write(error_response.encode())

    async def handle_websocket_connection(self, websocket: Any):
        """Handle new WebSocket connections, including the health check."""
        logger.info(f"CONNECT NEW WebSocket connection attempt.")
        
        try:
            # Extract path from our stored paths
            connection_id = id(websocket)
            path = self.websocket_paths.get(connection_id)
            
            if not path:
                logger.warning("CONNECT WebSocket connection rejected: No path found")
                return
                
            logger.info(f"CONNECT Processing WebSocket path: {path}")
            
            # **FIX: Extract agent_id from query parameters**
            agent_id = None
            if '?' in path:
                path_part, query_part = path.split('?', 1)
                # Parse query parameters
                query_params = {}
                for param in query_part.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        query_params[key] = value
                
                agent_id = query_params.get('agent_id')
                logger.info(f"CONNECT Extracted agent_id from query: {agent_id}")
            
            # **FALLBACK: Try to extract from path (for backward compatibility)**
            if not agent_id:
                try:
                    _, agent_id = path.rsplit('/', 1)
                    if '?' in agent_id:
                        agent_id = agent_id.split('?', 1)[0]
                    logger.info(f"CONNECT Extracted agent_id from path: {agent_id}")
                except ValueError:
                    pass
            
            if not agent_id:
                logger.warning(f"CONNECT WebSocket connection rejected: No agent_id found in path '{path}'")
                return
                
            logger.info(f"CONNECT Final agent_id: {agent_id}")
        except Exception as e:
            logger.error(f"CONNECT WebSocket connection error: {e}")
            return

        # Check if agent is registered
        if agent_id not in self.agents:
            logger.warning(f"CONNECT Agent {agent_id} not registered, but allowing connection")
            # Don't return here - allow unregistered agents to connect
            
        # Register the live connection
        logger.info(f"CONNECT Registering live connection for agent: {agent_id}")
        await self.register_live_connection(agent_id, websocket)
        
        try:
            logger.info(f"CONNECT WebSocket connection established for agent: {agent_id}")
            logger.info(f"CONNECT Starting message listener for agent: {agent_id}")
            
            # Keep the connection alive and listen for messages
            async for message in websocket:
                logger.info(f"CONNECT Received message from {agent_id}: {message}")
                try:
                    data = json.loads(message)
                    await self.handle_websocket_message(agent_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"CONNECT Invalid JSON from {agent_id}: {message}")
                except Exception as e:
                    logger.error(f"CONNECT Error handling message from {agent_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"WebSocket connection closed for agent {agent_id}: {e}")
        except Exception as e:
            logger.error(f"Error in WebSocket handler for agent {agent_id}: {e}", exc_info=True)
        finally:
            logger.info(f"CONNECT Cleaning up connection for agent: {agent_id}")
            await self.unregister_live_connection(agent_id)
            # **FIX:** Notify the collaboration engine about the disconnect
            if self.collaboration_engine:
                await self.collaboration_engine.handle_agent_disconnect(agent_id)

    async def handle_websocket_message(self, agent_id: str, data: dict):
        """Handle different types of WebSocket messages from agents."""
        message_type_str = data.get("message_type")
        logger.info(f"📨 Processing {message_type_str} message from {agent_id}")
        logger.info(f"📨 Full message data: {data}")

        if message_type_str == "TASK_COMPLETION":
            logger.info(f"OK Received TASK_COMPLETION from {agent_id}")
            
            # Enhanced handling for task completion
            try:
                # OK FIXED: Handle both new structured format and legacy formats
                if "payload" in data:
                    # New structured format
                    payload = data["payload"]
                    task_id = payload.get("task_id")
                    status = payload.get("status")
                    result = payload.get("result", {})
                else:
                    # Legacy format support
                    task_id = data.get("task_id")
                    status = data.get("status")
                    result = data.get("result", {})
                
                logger.info(f"OK Task completion details: Task {task_id}, Status: {status}")
                
                # Create proper Message object for collaboration engine
                message_for_engine = Message(
                    sender_id=agent_id,
                    receiver_id="lobby",
                    message_type=MessageType.TASK_COMPLETION,
                    payload={
                        "task_id": task_id,
                        "status": status,
                        "result": result,
                        "agent_id": agent_id
                    }
                )
                
                if self.collaboration_engine:
                    logger.info(f"OK Forwarding task result to collaboration engine.")
                    await self.collaboration_engine.handle_task_result(message_for_engine)
                else:
                    logger.warning("WARNING Collaboration engine not available to handle task result.")
                    
            except Exception as e:
                logger.error(f"ERROR Failed to process TASK_COMPLETION message: {e}", exc_info=True)

        elif message_type_str == "RESPONSE":
            # Handle response messages from message system
            logger.info(f"OK Received RESPONSE from {agent_id}")
            try:
                # Forward to collaboration engine as task completion
                message_for_engine = Message(
                    sender_id=agent_id,
                    receiver_id=data.get("receiver_id", "lobby"),
                    message_type=MessageType.TASK_COMPLETION,
                    payload=data.get("payload", {}),
                    conversation_id=data.get("conversation_id")
                )
                
                if self.collaboration_engine:
                    await self.collaboration_engine.handle_task_result(message_for_engine)
                
            except Exception as e:
                logger.error(f"ERROR Failed to process RESPONSE message: {e}", exc_info=True)

        elif message_type_str == "HEARTBEAT":
            logger.debug(f"💓 Heartbeat from {agent_id}")
            # SDK handles heartbeats client-side, no response needed from lobby.
            
        # OK LEGACY SUPPORT: Handle old message formats
        elif data.get("type") == "task_response":
            logger.info(f"OK Received legacy task_response from {agent_id}")
            try:
                # Convert legacy format to new format
                message_for_engine = Message(
                    sender_id=agent_id,
                    receiver_id="lobby",
                    message_type=MessageType.TASK_COMPLETION,
                    payload={
                        "task_id": data.get("task_id"),
                        "status": data.get("status"),
                        "result": data.get("result", {}),
                        "agent_id": agent_id
                    }
                )
                
                if self.collaboration_engine:
                    await self.collaboration_engine.handle_task_result(message_for_engine)
                    
            except Exception as e:
                logger.error(f"ERROR Failed to process legacy task_response: {e}", exc_info=True)
                
        elif data.get("type") == "register":
            logger.info(f"OK Received agent registration from WebSocket: {agent_id}")
            # Agent is registering via WebSocket - acknowledge
            try:
                await self.send_websocket_message(agent_id, {
                    "type": "register_ack",
                    "message": f"Agent {agent_id} registered successfully"
                })
            except Exception as e:
                logger.error(f"ERROR Failed to send registration ack: {e}")
            
        else:
            logger.warning(f"CONNECT Unknown or unhandled message type from {agent_id}: {message_type_str or data.get('type', 'unknown')}")

    async def send_websocket_message(self, agent_id: str, message: dict):
        """Send a message to an agent via WebSocket."""
        if agent_id in self.live_agent_connections:
            try:
                websocket = self.live_agent_connections[agent_id]
                await websocket.send(json.dumps(message))
                logger.debug(f"SEND Sent message to {agent_id}: {message}")
            except Exception as e:
                logger.error(f"ERROR Failed to send message to {agent_id}: {e}")
        else:
            logger.warning(f"WARNING Cannot send message to {agent_id}: No active connection")

    async def _handle_agent_ws_message(self, agent_id: str, message_str: str):
        """Handles messages received from a connected agent."""
        try:
            data = json.loads(message_str)
            logger.info(f"Received message from agent '{agent_id}': {data.get('type')}")
            
            # If the agent reports a completed task, feed it into the lobby's system
            if data.get("type") == "TASK_COMPLETED":
                task_message = Message(
                    sender_id=agent_id,
                    receiver_id=self.lobby_id,
                    message_type=MessageType.TASK_COMPLETED,
                    payload=data,
                )
                # Put it into the main queue for consistent processing
                await self._priority_queues[task_message.priority.value].put((
                    task_message.priority.value, time.time(), task_message.message_id, task_message
                ))
        except Exception as e:
            logger.error(f"Error processing message from agent {agent_id}: {e}", exc_info=True)

    async def register_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Registers an agent and initializes their data."""
        agent_id = agent_data.get("agent_id")
        logger.info(f"🤖 AGENT REGISTRATION REQUEST: {agent_id}")
        
        if not agent_id:
            logger.error("🤖 Registration failed: Missing agent_id")
            return {"status": "error", "message": "Missing agent_id"}
        if agent_id in self.agents:
            logger.warning(f"🤖 Registration failed: Agent {agent_id} already registered")
            return {"status": "error", "message": f"Agent {agent_id} already registered"}
        
        # Ensure 'name' is included in the stored agent data
        agent_info = {
            "agent_id": agent_id,
            "name": agent_data.get("name", agent_id), # Fallback to agent_id if name is not provided
            "agent_type": agent_data.get("agent_type"),
            "capabilities": agent_data.get("capabilities", []),
            "callback_url": agent_data.get("callback_url"),  # Store callback URL for HTTP fallback
            "registered_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.agents[agent_id] = agent_info
        logger.info(f"OK AGENT REGISTERED: {agent_id}")
        logger.info(f"   TASK Name: {agent_info['name']}")
        logger.info(f"   🏷️  Type: {agent_info['agent_type']}")
        logger.info(f"   DART Capabilities: {agent_info['capabilities']}")
        logger.info(f"   INFO Total agents: {len(self.agents)}")
        return {"status": "success", "message": f"Agent {agent_id} registered successfully"}

    async def register_live_connection(self, agent_id: str, websocket: Any):
        """Registers an active WebSocket connection."""
        self.live_agent_connections[agent_id] = websocket
        logger.info(f"OK LIVE CONNECTION REGISTERED: {agent_id}")
        logger.info(f"   INFO Total live connections: {len(self.live_agent_connections)}")
        logger.info(f"   🔗 Connected agents: {list(self.live_agent_connections.keys())}")

    async def unregister_live_connection(self, agent_id: str):
        """Unregisters a closed WebSocket connection."""
        if agent_id in self.live_agent_connections:
            del self.live_agent_connections[agent_id]
            logger.info(f"ERROR LIVE CONNECTION UNREGISTERED: {agent_id}")
            logger.info(f"   INFO Total live connections: {len(self.live_agent_connections)}")
            logger.info(f"   🔗 Connected agents: {list(self.live_agent_connections.keys())}")

    async def send_task_to_agent(self, agent_id: str, task_message: Dict[str, Any]) -> bool:
        """Sends a task message to a specific agent via WebSocket with HTTP fallback."""
        logger.info(f"SEND ATTEMPTING TO SEND TASK to agent: {agent_id}")
        logger.info(f"   🔗 Available connections: {list(self.live_agent_connections.keys())}")
        logger.info(f"   TASK Task message structure: {list(task_message.keys())}")
        
        # First try WebSocket delivery (preferred)
        if agent_id in self.live_agent_connections:
            websocket = self.live_agent_connections.get(agent_id)
            try:
                logger.info(f"SEND Sending task via WebSocket to {agent_id}")
                await websocket.send(json.dumps(task_message))
                
                # Extract task_id from the correct location in the message
                task_id = (
                    task_message.get('payload', {}).get('task_id') or  # New format: payload.task_id
                    task_message.get('task', {}).get('task_id') or     # Legacy format: task.task_id  
                    task_message.get('task_id') or                     # Direct format: task_id
                    'N/A'
                )
                
                logger.info(f"OK TASK SENT SUCCESSFULLY: Task {task_id} to agent {agent_id}")
                logger.info(f"   TASK Message type: {task_message.get('message_type', task_message.get('type', 'unknown'))}")
                return True
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"ERROR TASK SEND FAILED: Connection closed for agent {agent_id}")
                await self.unregister_live_connection(agent_id) # Clean up dead connection
                # Fall through to HTTP fallback
            except Exception as e:
                logger.error(f"ERROR TASK SEND ERROR for agent {agent_id}: {e}", exc_info=True)
                # Fall through to HTTP fallback
        
        # HTTP fallback for A2A protocol compatibility 
        if agent_id in self.agents:
            logger.info(f"🔄 Attempting HTTP fallback for agent {agent_id}")
            
            # For A2A protocol, we simulate successful task delivery
            # This allows HTTP-only agents to be compatible with A2A delegation
            if task_message.get("protocol") == "A2A":
                task_id = (
                    task_message.get('payload', {}).get('task_id') or
                    task_message.get('task_id') or
                    'N/A'
                )
                
                logger.info(f"✅ A2A HTTP FALLBACK: Task {task_id} queued for agent {agent_id}")
                logger.info(f"   📋 Agent {agent_id} is registered but offline (HTTP-only)")
                logger.info(f"   🔄 Task will be available when agent connects or polls")
                
                # Store task for when agent connects or polls
                if not hasattr(self, 'pending_a2a_tasks'):
                    self.pending_a2a_tasks = {}
                
                if agent_id not in self.pending_a2a_tasks:
                    self.pending_a2a_tasks[agent_id] = []
                
                self.pending_a2a_tasks[agent_id].append({
                    "task_message": task_message,
                    "queued_at": datetime.now(timezone.utc).isoformat()
                })
                
                return True  # Success for A2A compatibility
            
            logger.warning(f"ERROR TASK SEND FAILED: No WebSocket connection for agent {agent_id}")
            logger.info(f"   🔄 Agent {agent_id} is registered but not connected via WebSocket")
            logger.info(f"   📋 This is expected for HTTP-only test agents")
            return False
        else:
            logger.error(f"ERROR Agent {agent_id} not even registered!")
            return False

    async def _process_message_queues(self):
        """The main message processing loop for the lobby."""
        while self.running:
            try:
                for pq in self._priority_queues:
                    if not pq.empty():
                        _, _, _, message = await pq.get()
                        await self._process_single_message(message)
                await asyncio.sleep(0.01) # Prevent busy-waiting
            except Exception as e:
                logger.error(f"CRITICAL ERROR in message processing loop: {e}", exc_info=True)

    async def _process_single_message(self, message: Message):
        """Routes a single message to its destination."""
        try:
            if message.receiver_id in self.agents:
                # This is a message for an agent, send it via WebSocket
                await self._send_message_via_websocket(message)
            elif message.receiver_id == self.lobby_id:
                # This is a message for the lobby itself (e.g., task completion)
                logger.info(f"Lobby received internal message: {message.message_type}")
            else:
                logger.warning(f"Message receiver '{message.receiver_id}' not found.")
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}", exc_info=True)

    async def _send_message_via_websocket(self, message: Message):
        """Sends a message to a connected agent."""
        websocket = self.live_agent_connections.get(message.receiver_id)
        if websocket:
            try:
                serialized = message.to_dict()
                # Add a top-level 'type' field for SDK compatibility
                serialized["type"] = "message"
                await websocket.send(json.dumps(serialized))
                logger.info(f"Sent message {message.message_type} to {message.receiver_id}")
            except Exception as e:
                logger.error(f"Failed to send WebSocket message to {message.receiver_id}: {e}")
        else:
            logger.warning(f"Cannot send message: No live connection for agent {message.receiver_id}.")

    async def process_websocket_request(self, connection, request):
        """Process WebSocket request to extract path information."""
        logger.info(f"CONNECT WebSocket handshake - Path: {request.path}")
        # Store the path for this connection
        self.websocket_paths[id(connection)] = request.path
        return None  # Allow connection to proceed

    async def handle_a2a_delegation(self, task_data: Dict[str, Any], target_agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle A2A protocol task delegation"""
        try:
            logger.info(f"🔄 A2A Task delegation received: {task_data}")
            
            # Extract task information from A2A format
            task_title = task_data.get("title", task_data.get("task", "A2A Task"))
            task_description = task_data.get("description", task_data.get("prompt", ""))
            required_capabilities = task_data.get("required_capabilities", task_data.get("skills", []))
            sender_id = task_data.get("sender_id", task_data.get("from", "a2a_client"))
            
            if not task_description:
                return {
                    "status": "error",
                    "message": "Task description is required",
                    "error_code": "MISSING_DESCRIPTION"
                }
            
            # Create unique task ID for A2A
            task_id = f"a2a_{uuid.uuid4().hex[:8]}"
            
            # If target agent specified, try to delegate directly
            if target_agent_id:
                if target_agent_id not in self.agents:
                    return {
                        "status": "error",
                        "message": f"Target agent {target_agent_id} not found",
                        "error_code": "AGENT_NOT_FOUND"
                    }
                
                # Format task for Agent Lobby
                task_message = {
                    "message_id": task_id,
                    "task_id": task_id,
                    "sender_id": sender_id,
                    "task_title": task_title,
                    "task_description": task_description,
                    "required_capabilities": required_capabilities,
                    "protocol": "A2A",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "payload": {
                        "original_a2a_message": task_data,
                        "task": task_description,
                        "input_data": task_data.get("input", {}),
                        "context": task_data.get("context", {})
                    }
                }
                
                # Try to send task to specific agent
                success = await self.send_task_to_agent(target_agent_id, task_message)
                
                if success:
                    return {
                        "status": "success",
                        "task_id": task_id,
                        "assigned_to": target_agent_id,
                        "message": f"Task delegated to {target_agent_id}",
                        "protocol": "A2A",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to delegate task to {target_agent_id}",
                        "error_code": "DELEGATION_FAILED"
                    }
            
            # Auto-select best agent based on capabilities
            selected_agent = await self._select_best_agent_for_a2a_task(required_capabilities)
            
            if not selected_agent:
                return {
                    "status": "error",
                    "message": "No suitable agent found for this task",
                    "error_code": "NO_SUITABLE_AGENT",
                    "required_capabilities": required_capabilities,
                    "available_agents": len(self.agents)
                }
            
            # Format and send task
            task_message = {
                "message_id": task_id,
                "task_id": task_id,
                "sender_id": sender_id,
                "task_title": task_title,
                "task_description": task_description,
                "required_capabilities": required_capabilities,
                "protocol": "A2A",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "original_a2a_message": task_data,
                    "task": task_description,
                    "input_data": task_data.get("input", {}),
                    "context": task_data.get("context", {})
                }
            }
            
            success = await self.send_task_to_agent(selected_agent, task_message)
            
            if success:
                return {
                    "status": "success",
                    "task_id": task_id,
                    "assigned_to": selected_agent,
                    "message": f"Task delegated to {selected_agent}",
                    "protocol": "A2A",
                    "agent_info": {
                        "agent_id": selected_agent,
                        "capabilities": self.agents[selected_agent].get("capabilities", [])
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Failed to delegate task to {selected_agent}",
                    "error_code": "DELEGATION_FAILED"
                }
                
        except Exception as e:
            logger.error(f"Error in A2A delegation: {e}")
            return {
                "status": "error",
                "message": f"Internal error during task delegation: {str(e)}",
                "error_code": "INTERNAL_ERROR"
            }

    async def handle_a2a_communication(self, message_data: Dict[str, Any], target_agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle A2A protocol agent-to-agent communication"""
        try:
            logger.info(f"💬 A2A Communication received: {message_data}")
            
            # Extract message information
            message_text = message_data.get("message", message_data.get("content", ""))
            sender_id = message_data.get("sender_id", message_data.get("from", "a2a_client"))
            message_type = message_data.get("type", "communication")
            
            if not message_text:
                return {
                    "status": "error",
                    "message": "Message content is required",
                    "error_code": "MISSING_CONTENT"
                }
            
            # If target agent specified, send directly
            if target_agent_id:
                if target_agent_id not in self.agents:
                    return {
                        "status": "error",
                        "message": f"Target agent {target_agent_id} not found",
                        "error_code": "AGENT_NOT_FOUND"
                    }
                
                # Create message for WebSocket
                ws_message = {
                    "type": "A2A_COMMUNICATION",
                    "message_id": f"a2a_comm_{uuid.uuid4().hex[:8]}",
                    "sender_id": sender_id,
                    "content": message_text,
                    "protocol": "A2A",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "original_message": message_data
                }
                
                # Send via WebSocket if agent is connected
                if target_agent_id in self.live_agent_connections:
                    await self.send_websocket_message(target_agent_id, ws_message)
                    return {
                        "status": "success",
                        "message": f"Message sent to {target_agent_id}",
                        "recipient": target_agent_id,
                        "protocol": "A2A",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Agent {target_agent_id} is not online",
                        "error_code": "AGENT_OFFLINE"
                    }
            
            # Broadcast to all online agents if no target specified
            online_agents = list(self.live_agent_connections.keys())
            
            if not online_agents:
                return {
                    "status": "error",
                    "message": "No agents are currently online",
                    "error_code": "NO_ONLINE_AGENTS"
                }
            
            broadcast_message = {
                "type": "A2A_BROADCAST",
                "message_id": f"a2a_broadcast_{uuid.uuid4().hex[:8]}",
                "sender_id": sender_id,
                "content": message_text,
                "protocol": "A2A",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original_message": message_data
            }
            
            sent_count = 0
            for agent_id in online_agents:
                try:
                    await self.send_websocket_message(agent_id, broadcast_message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send broadcast to {agent_id}: {e}")
            
            return {
                "status": "success",
                "message": f"Message broadcast to {sent_count} agents",
                "recipients": online_agents[:sent_count],
                "protocol": "A2A",
                "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                
        except Exception as e:
            logger.error(f"Error in A2A communication: {e}")
            return {
                "status": "error",
                "message": f"Internal error during communication: {str(e)}",
                "error_code": "INTERNAL_ERROR"
            }

    async def handle_a2a_task_polling(self, agent_id: str) -> Dict[str, Any]:
        """Handle A2A task polling for HTTP-only agents"""
        try:
            logger.info(f"📥 A2A task polling request from agent: {agent_id}")
            
            # Check if agent is registered
            if agent_id not in self.agents:
                return {
                    "status": "error",
                    "message": f"Agent {agent_id} not registered",
                    "error_code": "AGENT_NOT_FOUND",
                    "tasks": []
                }
            
            # Initialize pending tasks if not exists
            if not hasattr(self, 'pending_a2a_tasks'):
                self.pending_a2a_tasks = {}
            
            # Get pending tasks for this agent
            pending_tasks = self.pending_a2a_tasks.get(agent_id, [])
            
            if not pending_tasks:
                return {
                    "status": "success",
                    "message": "No pending tasks",
                    "agent_id": agent_id,
                    "tasks": [],
                    "count": 0,
                    "protocol": "A2A",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Format tasks for A2A response
            formatted_tasks = []
            for task_entry in pending_tasks:
                task_message = task_entry["task_message"]
                formatted_task = {
                    "task_id": task_message.get("task_id"),
                    "title": task_message.get("task_title"),
                    "description": task_message.get("task_description"),
                    "required_capabilities": task_message.get("required_capabilities", []),
                    "payload": task_message.get("payload", {}),
                    "sender_id": task_message.get("sender_id"),
                    "queued_at": task_entry["queued_at"],
                    "protocol": "A2A"
                }
                formatted_tasks.append(formatted_task)
            
            # Clear the pending tasks after polling (delivered)
            self.pending_a2a_tasks[agent_id] = []
            
            logger.info(f"📤 Delivered {len(formatted_tasks)} A2A tasks to agent {agent_id}")
            
            return {
                "status": "success",
                "message": f"Retrieved {len(formatted_tasks)} pending tasks",
                "agent_id": agent_id,
                "tasks": formatted_tasks,
                "count": len(formatted_tasks),
                "protocol": "A2A",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in A2A task polling: {e}")
            return {
                "status": "error",
                "message": f"Internal error during task polling: {str(e)}",
                "error_code": "INTERNAL_ERROR",
                "tasks": []
            }

    async def _select_best_agent_for_a2a_task(self, required_capabilities: List[str]) -> Optional[str]:
        """Select the best agent for an A2A task based on capabilities and availability"""
        try:
            if not self.agents:
                return None
            
            # Score agents based on capability match and availability
            agent_scores = []
            
            for agent_id, agent_data in self.agents.items():
                agent_capabilities = agent_data.get("capabilities", [])
                
                # Calculate capability match score
                if required_capabilities:
                    matches = sum(1 for cap in required_capabilities if cap in agent_capabilities)
                    capability_score = matches / len(required_capabilities)
                else:
                    capability_score = 1.0  # If no specific capabilities required
                
                # Bonus for online agents
                online_bonus = 0.2 if agent_id in self.live_agent_connections else 0.0
                
                # TODO: Add reputation score when available
                reputation_score = 0.1  # Placeholder
                
                total_score = capability_score + online_bonus + reputation_score
                
                agent_scores.append((agent_id, total_score, capability_score))
            
            # Sort by score (highest first)
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return best agent if they have at least some capability match
            if agent_scores and agent_scores[0][1] > 0:
                selected_agent = agent_scores[0][0]
                logger.info(f"Selected agent {selected_agent} for A2A task (score: {agent_scores[0][1]:.2f})")
                return selected_agent
            
            return None
            
        except Exception as e:
            logger.error(f"Error selecting agent for A2A task: {e}")
            return None
