#!/usr/bin/env python3
"""
ENHANCED A2A API BRIDGE - ULTIMATE CONNECTOR WITH METRICS
=========================================================
Top-notch A2A-enabled API bridge with integrated metrics dashboard,
real-time monitoring, and comprehensive connector capabilities.
"""

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import httpx
from contextlib import asynccontextmanager

# Import lobby components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lobby import Lobby
from core.message import Message, MessageType, MessagePriority

logger = logging.getLogger(__name__)

# ================================
# ENHANCED MODELS
# ================================

class A2ATaskRequest(BaseModel):
    """A2A protocol task request"""
    title: str
    description: str
    required_capabilities: List[str] = []
    input: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    sender_id: Optional[str] = "a2a_bridge_client"

class A2ACommunicationRequest(BaseModel):
    """A2A protocol communication request"""
    message: str
    sender_id: Optional[str] = "a2a_bridge_client"
    type: Optional[str] = "communication"

class MetricsRequest(BaseModel):
    """Metrics query request"""
    metric_type: str
    time_range: Optional[str] = "1h"
    agent_id: Optional[str] = None
    group_by: Optional[str] = None

# ================================
# ENHANCED A2A API BRIDGE
# ================================

class EnhancedA2AAPIBridge:
    """
    Ultimate A2A-enabled API bridge with integrated metrics dashboard
    and comprehensive connector capabilities
    """
    
    def __init__(self, lobby_instance: Optional[Lobby] = None, 
                 lobby_host: str = "localhost", 
                 lobby_http_port: int = 8080, 
                 lobby_ws_port: int = 8081):
        self.lobby_host = lobby_host
        self.lobby_http_port = lobby_http_port
        self.lobby_ws_port = lobby_ws_port
        self.lobby_http_url = f"http://{lobby_host}:{lobby_http_port}"
        self.lobby_ws_url = f"ws://{lobby_host}:{lobby_ws_port}"
        
        # Lobby instance for direct integration
        self.lobby_instance: Optional[Lobby] = lobby_instance
        
        # Enhanced metrics and monitoring
        self.metrics = {
            "requests_total": 0,
            "a2a_requests": 0,
            "native_requests": 0,
            "websocket_connections": 0,
            "active_tasks": 0,
            "response_times": [],
            "error_count": 0,
            "agent_activity": {},
            "protocol_distribution": {"a2a": 0, "native": 0},
            "start_time": datetime.now(timezone.utc)
        }
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.metrics_subscribers: List[str] = []
        
        # HTTP client for external requests
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Initialize FastAPI app
        self.app = self._create_fastapi_app()
        
        logger.info("🚀 Enhanced A2A API Bridge initialized")
        logger.info(f"   🌐 A2A Protocol: ENABLED")
        logger.info(f"   📊 Metrics Dashboard: ENABLED")
        logger.info(f"   🔗 Lobby Connection: {self.lobby_http_url}")

    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application with A2A and metrics support"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.startup()
            yield
            await self.shutdown()
        
        app = FastAPI(
            title="Enhanced A2A API Bridge",
            description="Ultimate A2A-enabled connector with integrated metrics dashboard",
            version="3.0.0-a2a-enhanced",
            lifespan=lifespan
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ================================
        # ROOT & STATUS ENDPOINTS
        # ================================
        
        @app.get("/")
        async def root():
            """Enhanced root endpoint with A2A capabilities"""
            return {
                "service": "Enhanced A2A API Bridge",
                "version": "3.0.0-a2a-enhanced",
                "status": "operational",
                "capabilities": {
                    "a2a_protocol": True,
                    "native_protocol": True,
                    "metrics_dashboard": True,
                    "real_time_monitoring": True,
                    "websocket_support": True
                },
                "endpoints": {
                    "a2a_discovery": "/.well-known/agent.json",
                    "a2a_delegate": "/api/a2a/delegate",
                    "a2a_communicate": "/api/a2a/communicate",
                    "metrics_dashboard": "/metrics/dashboard",
                    "metrics_api": "/api/metrics",
                    "real_time_metrics": "/api/ws/metrics"
                },
                "documentation": "/docs",
                "metrics": await self._get_current_metrics()
            }

        @app.get("/api/health")
        async def health_check():
            """Enhanced health check with A2A status"""
            lobby_status = await self._check_lobby_health()
            return {
                "bridge_status": "operational",
                "lobby_connection": lobby_status,
                "a2a_enabled": True,
                "metrics_enabled": True,
                "websocket_connections": len(self.websocket_connections),
                "requests_processed": self.metrics["requests_total"],
                "uptime_seconds": (datetime.now(timezone.utc) - self.metrics["start_time"]).total_seconds(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # ================================
        # A2A PROTOCOL ENDPOINTS
        # ================================
        
        @app.get("/.well-known/agent.json")
        async def a2a_agent_discovery():
            """A2A protocol agent discovery - Bridge Agent Card"""
            self._track_request("a2a")
            
            return {
                "name": "Enhanced Agent Lobby Bridge",
                "description": "A2A-enabled API bridge with native Agent Lobby integration and metrics dashboard",
                "version": "3.0.0",
                "url": f"http://{self.lobby_host}:{self.lobby_http_port}",
                "capabilities": {
                    "streaming": True,
                    "pushNotifications": True,
                    "bridge_mode": True,
                    "metrics_dashboard": True,
                    "multi_protocol": True,
                    "real_time_monitoring": True
                },
                "protocols": ["A2A", "Agent Lobby Native", "HTTP", "WebSocket"],
                "extensions": {
                    "agent_lobby": {
                        "bridge_version": "3.0.0",
                        "native_integration": True,
                        "metrics_endpoint": "/api/metrics",
                        "dashboard_url": "/metrics/dashboard"
                    }
                },
                "endpoints": {
                    "delegate": "/api/a2a/delegate",
                    "communicate": "/api/a2a/communicate",
                    "discover": "/api/a2a/discover",
                    "status": "/api/a2a/status"
                }
            }

        @app.get("/api/a2a/status")
        async def a2a_status():
            """A2A protocol status endpoint"""
            self._track_request("a2a")
            
            if self.lobby_instance:
                agent_count = len(self.lobby_instance.agents)
                online_agents = len(self.lobby_instance.live_agent_connections)
            else:
                agent_count = 0
                online_agents = 0
            
            return {
                "status": "operational",
                "protocol": "A2A",
                "bridge_version": "3.0.0",
                "agent_count": agent_count,
                "online_agents": online_agents,
                "requests_processed": self.metrics["a2a_requests"],
                "capabilities": ["task_delegation", "communication", "discovery", "metrics"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        @app.get("/api/a2a/discover")
        async def a2a_discover_agents():
            """A2A protocol agent discovery endpoint"""
            self._track_request("a2a")
            
            if not self.lobby_instance:
                return {"agents": [], "count": 0}
            
            # Format agents for A2A discovery
            a2a_agents = []
            for agent_id, agent_data in self.lobby_instance.agents.items():
                a2a_agent = {
                    "agent_id": agent_id,
                    "name": agent_data.get("name", agent_id),
                    "description": f"Agent Lobby agent: {agent_data.get('agent_type', 'unknown')}",
                    "capabilities": agent_data.get("capabilities", []),
                    "status": "online" if agent_id in self.lobby_instance.live_agent_connections else "offline",
                    "protocols": ["A2A", "Agent Lobby Native"],
                    "url": f"http://{self.lobby_host}:{self.lobby_http_port}/api/a2a/agents/{agent_id}"
                }
                a2a_agents.append(a2a_agent)
            
            return {
                "agents": a2a_agents,
                "count": len(a2a_agents),
                "bridge_info": {
                    "service": "Enhanced A2A API Bridge",
                    "version": "3.0.0"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        @app.post("/api/a2a/delegate")
        async def a2a_delegate_task(request: A2ATaskRequest):
            """A2A protocol task delegation"""
            start_time = time.time()
            self._track_request("a2a")
            
            try:
                if not self.lobby_instance:
                    raise HTTPException(status_code=503, detail="Lobby instance not available")
                
                # Convert A2A request to lobby format and delegate
                result = await self.lobby_instance.handle_a2a_delegation(
                    {
                        "title": request.title,
                        "description": request.description,
                        "required_capabilities": request.required_capabilities,
                        "input": request.input or {},
                        "context": request.context or {},
                        "sender_id": request.sender_id
                    }
                )
                
                # Track metrics
                response_time = time.time() - start_time
                self._track_response_time(response_time)
                self.metrics["active_tasks"] += 1
                
                return result
                
            except Exception as e:
                self._track_error()
                logger.error(f"A2A delegation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/a2a/communicate")
        async def a2a_communicate(request: A2ACommunicationRequest):
            """A2A protocol communication"""
            self._track_request("a2a")
            
            try:
                if not self.lobby_instance:
                    raise HTTPException(status_code=503, detail="Lobby instance not available")
                
                # Convert A2A communication to lobby format
                result = await self.lobby_instance.handle_a2a_communication(
                    {
                        "message": request.message,
                        "sender_id": request.sender_id,
                        "type": request.type
                    }
                )
                
                return result
                
            except Exception as e:
                self._track_error()
                logger.error(f"A2A communication failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/a2a/tasks/{agent_id}")
        async def a2a_poll_tasks(agent_id: str):
            """A2A protocol task polling for HTTP-only agents"""
            self._track_request("a2a")
            
            try:
                if not self.lobby_instance:
                    raise HTTPException(status_code=503, detail="Lobby instance not available")
                
                result = await self.lobby_instance.handle_a2a_task_polling(agent_id)
                return result
                
            except Exception as e:
                self._track_error()
                logger.error(f"A2A task polling failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ================================
        # METRICS DASHBOARD ENDPOINTS
        # ================================
        
        @app.get("/metrics/dashboard", response_class=HTMLResponse)
        async def metrics_dashboard():
            """Integrated metrics dashboard"""
            return self._generate_metrics_dashboard_html()

        @app.get("/api/metrics")
        async def get_metrics():
            """Get current metrics data"""
            return await self._get_current_metrics()

        @app.get("/api/metrics/agents")
        async def get_agent_metrics():
            """Get agent-specific metrics"""
            if not self.lobby_instance:
                return {"agents": [], "total": 0}
            
            agent_metrics = []
            for agent_id, agent_data in self.lobby_instance.agents.items():
                metrics = {
                    "agent_id": agent_id,
                    "name": agent_data.get("name", agent_id),
                    "status": "online" if agent_id in self.lobby_instance.live_agent_connections else "offline",
                    "capabilities": agent_data.get("capabilities", []),
                    "tasks_completed": self.metrics["agent_activity"].get(agent_id, 0),
                    "last_activity": agent_data.get("last_activity"),
                    "reputation": getattr(self.lobby_instance, 'agent_reputation', {}).get(agent_id, 0.5)
                }
                agent_metrics.append(metrics)
            
            return {
                "agents": agent_metrics,
                "total": len(agent_metrics),
                "online": len(self.lobby_instance.live_agent_connections),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        @app.get("/api/metrics/system")
        async def get_system_metrics():
            """Get system performance metrics"""
            uptime = (datetime.now(timezone.utc) - self.metrics["start_time"]).total_seconds()
            
            return {
                "uptime_seconds": uptime,
                "requests_total": self.metrics["requests_total"],
                "requests_per_second": self.metrics["requests_total"] / max(uptime, 1),
                "protocol_distribution": self.metrics["protocol_distribution"],
                "active_connections": len(self.websocket_connections),
                "active_tasks": self.metrics["active_tasks"],
                "error_rate": self.metrics["error_count"] / max(self.metrics["requests_total"], 1),
                "average_response_time": sum(self.metrics["response_times"][-100:]) / max(len(self.metrics["response_times"][-100:]), 1),
                "memory_usage": self._get_memory_usage(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        # ================================
        # REAL-TIME WEBSOCKET ENDPOINTS
        # ================================
        
        @app.websocket("/api/ws/metrics")
        async def metrics_websocket(websocket: WebSocket):
            """Real-time metrics via WebSocket"""
            await websocket.accept()
            client_id = str(uuid.uuid4())
            self.websocket_connections[client_id] = websocket
            self.metrics_subscribers.append(client_id)
            self.metrics["websocket_connections"] += 1
            
            try:
                # Send initial metrics
                await websocket.send_json(await self._get_current_metrics())
                
                # Keep connection alive and send periodic updates
                while True:
                    await asyncio.sleep(5)  # Update every 5 seconds
                    metrics = await self._get_current_metrics()
                    await websocket.send_json({
                        "type": "metrics_update",
                        "data": metrics,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
            except WebSocketDisconnect:
                pass
            finally:
                if client_id in self.websocket_connections:
                    del self.websocket_connections[client_id]
                if client_id in self.metrics_subscribers:
                    self.metrics_subscribers.remove(client_id)
                self.metrics["websocket_connections"] -= 1

        @app.websocket("/api/ws/{client_id}")
        async def general_websocket(websocket: WebSocket, client_id: str):
            """General WebSocket endpoint with lobby integration"""
            await websocket.accept()
            self.websocket_connections[client_id] = websocket
            
            # Register with lobby if available
            if self.lobby_instance:
                await self.lobby_instance.register_live_connection(client_id, websocket)
            
            try:
                while True:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(client_id, data)
            except WebSocketDisconnect:
                pass
            finally:
                if client_id in self.websocket_connections:
                    del self.websocket_connections[client_id]
                if self.lobby_instance:
                    await self.lobby_instance.unregister_live_connection(client_id)

        return app

    # ================================
    # HELPER METHODS
    # ================================
    
    def _track_request(self, protocol: str):
        """Track incoming request metrics"""
        self.metrics["requests_total"] += 1
        if protocol == "a2a":
            self.metrics["a2a_requests"] += 1
            self.metrics["protocol_distribution"]["a2a"] += 1
        else:
            self.metrics["native_requests"] += 1
            self.metrics["protocol_distribution"]["native"] += 1

    def _track_response_time(self, response_time: float):
        """Track response time metrics"""
        self.metrics["response_times"].append(response_time)
        # Keep only last 1000 response times
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]

    def _track_error(self):
        """Track error metrics"""
        self.metrics["error_count"] += 1

    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get comprehensive current metrics"""
        uptime = (datetime.now(timezone.utc) - self.metrics["start_time"]).total_seconds()
        
        return {
            "bridge_info": {
                "version": "3.0.0-a2a-enhanced",
                "uptime_seconds": uptime,
                "start_time": self.metrics["start_time"].isoformat()
            },
            "requests": {
                "total": self.metrics["requests_total"],
                "a2a": self.metrics["a2a_requests"],
                "native": self.metrics["native_requests"],
                "per_second": self.metrics["requests_total"] / max(uptime, 1),
                "error_count": self.metrics["error_count"],
                "error_rate": self.metrics["error_count"] / max(self.metrics["requests_total"], 1)
            },
            "performance": {
                "average_response_time": sum(self.metrics["response_times"][-100:]) / max(len(self.metrics["response_times"][-100:]), 1) if self.metrics["response_times"] else 0,
                "active_tasks": self.metrics["active_tasks"],
                "websocket_connections": len(self.websocket_connections)
            },
            "lobby_status": {
                "connected": self.lobby_instance is not None,
                "agent_count": len(self.lobby_instance.agents) if self.lobby_instance else 0,
                "online_agents": len(self.lobby_instance.live_agent_connections) if self.lobby_instance else 0
            },
            "protocols": self.metrics["protocol_distribution"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    async def _check_lobby_health(self) -> bool:
        """Check if lobby is healthy"""
        if self.lobby_instance:
            return True
        
        try:
            if not self.http_client:
                return False
            response = await self.http_client.get(f"{self.lobby_http_url}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"message": "psutil not available"}

    def _generate_metrics_dashboard_html(self) -> str:
        """Generate HTML for metrics dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced A2A API Bridge - Metrics Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; }
        .status-online { color: #28a745; }
        .status-offline { color: #dc3545; }
        .chart-container { height: 200px; background: #f8f9fa; border-radius: 5px; margin-top: 10px; }
        .update-time { text-align: center; color: #666; margin-top: 20px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced A2A API Bridge</h1>
            <p>Real-time metrics dashboard for A2A-enabled connector</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="total-requests">-</div>
                <div class="metric-label">Total Requests</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="a2a-requests">-</div>
                <div class="metric-label">A2A Requests</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="active-agents">-</div>
                <div class="metric-label">Active Agents</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="response-time">-</div>
                <div class="metric-label">Avg Response Time (ms)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="error-rate">-</div>
                <div class="metric-label">Error Rate (%)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="websocket-connections">-</div>
                <div class="metric-label">WebSocket Connections</div>
            </div>
        </div>
        
        <div class="update-time" id="last-update">Last updated: -</div>
    </div>

    <script>
        // Connect to metrics WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/ws/metrics`;
        const ws = new WebSocket(wsUrl);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateMetrics(data.data || data);
        };
        
        function updateMetrics(metrics) {
            document.getElementById('total-requests').textContent = metrics.requests?.total || 0;
            document.getElementById('a2a-requests').textContent = metrics.requests?.a2a || 0;
            document.getElementById('active-agents').textContent = metrics.lobby_status?.online_agents || 0;
            document.getElementById('response-time').textContent = 
                Math.round((metrics.performance?.average_response_time || 0) * 1000);
            document.getElementById('error-rate').textContent = 
                Math.round((metrics.requests?.error_rate || 0) * 100);
            document.getElementById('websocket-connections').textContent = 
                metrics.performance?.websocket_connections || 0;
            document.getElementById('last-update').textContent = 
                'Last updated: ' + new Date().toLocaleTimeString();
        }
        
        // Initial metrics load
        fetch('/api/metrics')
            .then(response => response.json())
            .then(data => updateMetrics(data));
    </script>
</body>
</html>
        """

    async def _handle_websocket_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type")
        
        if message_type == "ping":
            await self.websocket_connections[client_id].send_json({"type": "pong"})
        elif message_type == "get_metrics":
            metrics = await self._get_current_metrics()
            await self.websocket_connections[client_id].send_json({
                "type": "metrics_response",
                "data": metrics
            })

    async def startup(self):
        """Initialize the enhanced bridge"""
        logger.info("🚀 Starting Enhanced A2A API Bridge...")
        
        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=100)
        )
        
        logger.info("✅ Enhanced A2A API Bridge started successfully")
        logger.info("   🌐 A2A Protocol: ACTIVE")
        logger.info("   📊 Metrics Dashboard: ACTIVE")
        logger.info("   🔗 Lobby Integration: ACTIVE")

    async def shutdown(self):
        """Cleanup resources"""
        logger.info("🛑 Shutting down Enhanced A2A API Bridge...")
        
        if self.http_client:
            await self.http_client.aclose()
        
        # Close all WebSocket connections
        for websocket in self.websocket_connections.values():
            try:
                await websocket.close()
            except:
                pass
        
        logger.info("✅ Enhanced A2A API Bridge shutdown complete")

# ================================
# FACTORY FUNCTION
# ================================

def create_enhanced_a2a_bridge(
    lobby_instance: Optional[Lobby] = None,
    lobby_host: str = "localhost",
    lobby_http_port: int = 8080,
    lobby_ws_port: int = 8081
) -> FastAPI:
    """
    Factory function to create the Enhanced A2A API Bridge
    
    Args:
        lobby_instance: Direct lobby instance for integration
        lobby_host: Hostname of the Agent Lobby
        lobby_http_port: HTTP port of the Agent Lobby  
        lobby_ws_port: WebSocket port of the Agent Lobby
    
    Returns:
        Configured FastAPI application with A2A and metrics support
    """
    bridge = EnhancedA2AAPIBridge(
        lobby_instance=lobby_instance,
        lobby_host=lobby_host,
        lobby_http_port=lobby_http_port,
        lobby_ws_port=lobby_ws_port
    )
    return bridge.app

# ================================
# STANDALONE SERVER
# ================================

async def start_enhanced_a2a_bridge_server(
    bridge_host: str = "localhost",
    bridge_port: int = 8090,
    lobby_instance: Optional[Lobby] = None,
    lobby_host: str = "localhost", 
    lobby_http_port: int = 8080,
    lobby_ws_port: int = 8081
):
    """Start the Enhanced A2A API Bridge server"""
    import uvicorn
    
    bridge = EnhancedA2AAPIBridge(
        lobby_instance=lobby_instance,
        lobby_host=lobby_host,
        lobby_http_port=lobby_http_port,
        lobby_ws_port=lobby_ws_port
    )
    
    config = uvicorn.Config(
        bridge.app,
        host=bridge_host,
        port=bridge_port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(start_enhanced_a2a_bridge_server()) 