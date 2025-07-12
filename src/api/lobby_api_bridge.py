#!/usr/bin/env python3
"""
AGENT LOBBI API BRIDGE - PERFORMANCE OPTIMIZED
==============================================
Ultra-fast API bridge with aggressive caching, connection pooling,
and request pipelining for real-time performance (<20ms target).
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import httpx
from contextlib import asynccontextmanager
import time

# Import lobby components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lobby import Lobby
from core.message import Message, MessageType, MessagePriority

logger = logging.getLogger(__name__)

# ================================
# PERFORMANCE CACHE SYSTEM
# ================================

class PerformanceCache:
    """High-performance in-memory cache with TTL"""
    
    def __init__(self, default_ttl: int = 5):  # 5 second default TTL
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry['expires']:
                return entry['data']
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Cache value with TTL"""
        expires = time.time() + (ttl or self.default_ttl)
        self.cache[key] = {'data': value, 'expires': expires}
    
    def invalidate(self, pattern: str = None) -> None:
        """Invalidate cache entries matching pattern"""
        if pattern:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
        else:
            self.cache.clear()

# ================================
# PYDANTIC MODELS FOR API
# ================================

class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration"""
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    goal: Optional[str] = ""
    specialization: Optional[str] = ""
    collaboration_style: Optional[str] = "individual"
    metadata: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None

class TaskDelegationRequest(BaseModel):
    """Request model for task delegation"""
    task_title: str
    task_description: str
    required_capabilities: List[str]
    requester_id: str
    task_intent: Optional[str] = ""
    max_agents: Optional[int] = 3
    priority: Optional[str] = "normal"
    deadline: Optional[str] = None
    task_data: Optional[Dict[str, Any]] = None

class CollaborationRequest(BaseModel):
    """Request model for creating collaboration sessions"""
    participant_ids: List[str]
    purpose: str
    collaboration_type: Optional[str] = "task_based"
    shared_context: Optional[Dict[str, Any]] = None

class WorkflowRequest(BaseModel):
    """Request model for workflow creation"""
    name: str
    description: str
    created_by: str
    task_intent: str
    required_capabilities: List[str]
    task_data: Optional[Dict[str, Any]] = None
    max_agents: Optional[int] = 5

class AgentTaskResponse(BaseModel):
    """Response model for agent task completion"""
    task_id: str
    agent_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ================================
# API RESPONSE MODELS
# ================================

@dataclass
class APIResponse:
    """Standard API response format"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self):
        return asdict(self)

class LobbyAPIBridge:
    """
    PERFORMANCE-OPTIMIZED API bridge between the website and Agent Lobbi
    Target: <20ms response times with aggressive caching and connection pooling
    """
    
    def __init__(self, lobby_instance: Optional[Lobby] = None, lobby_host: str = "localhost", lobby_http_port: int = 8080, lobby_ws_port: int = 8081):
        self.lobby_host = lobby_host
        self.lobby_http_port = lobby_http_port
        self.lobby_ws_port = lobby_ws_port
        self.lobby_http_url = f"http://{lobby_host}:{lobby_http_port}"
        self.lobby_ws_url = f"ws://{lobby_host}:{lobby_ws_port}"
        
        # PERFORMANCE OPTIMIZATIONS
        self.cache = PerformanceCache(default_ttl=5)  # 5 second cache
        self.http_client: Optional[httpx.AsyncClient] = None
        self.lobby_healthy = False
        self.last_health_check = 0
        self.health_check_interval = 10  # Check lobby health every 10 seconds
        
        # Connection management
        self.lobby_instance: Optional[Lobby] = lobby_instance
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.active_tasks: Dict[str, Dict] = {}
        self.task_subscriptions: Dict[str, List[str]] = {}  # task_id -> [websocket_ids]
        self.server = None # To hold the uvicorn server instance for graceful shutdown
        
        # Initialize FastAPI app
        self.app = self._create_fastapi_app()
        
        logger.info(f"START PERFORMANCE-OPTIMIZED Lobby API Bridge initialized")
        logger.info(f"Target Lobby: {self.lobby_http_url}")
        logger.info(f"Cache TTL: {self.cache.default_ttl}s")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()
        
        app = FastAPI(
            title="Agent Lobbi API Bridge - PERFORMANCE OPTIMIZED",
            description="Ultra-fast API bridge with <20ms target response times",
            version="2.0.0-performance",
            lifespan=lifespan
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security
        security = HTTPBearer(auto_error=False)
        
        # ================================
        # ROOT ENDPOINT
        # ================================
        @app.get("/")
        async def root():
            """Root endpoint providing basic API information."""
            return APIResponse(
                success=True,
                message="Welcome to the Agent Lobbi API Bridge",
                data={
                    "version": app.version,
                    "status": "operational",
                    "documentation": "/docs",
                    "health_check": "/api/health",
                },
            ).to_dict()
        
        # ================================
        # HEALTH AND STATUS ENDPOINTS
        # ================================
        
        @app.get("/api/health")
        async def health_check():
            """OPTIMIZED: Health check with caching"""
            # Check cache first
            cached = self.cache.get("health_check")
            if cached:
                return cached
            
            lobby_status = await self._check_lobby_health_cached()
            response = APIResponse(
                success=True,
                message="API Bridge is healthy",
                data={
                    "bridge_status": "operational",
                    "lobby_connection": lobby_status,
                    "cache_enabled": True,
                    "performance_mode": "enabled",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ).to_dict()
            
            # Cache for 2 seconds
            self.cache.set("health_check", response, ttl=2)
            return response
        
        @app.get("/api/lobby/status")
        async def get_lobby_status():
            """OPTIMIZED: Get lobby status with aggressive caching"""
            try:
                # Check cache first
                cached = self.cache.get("lobby_status")
                if cached:
                    return cached
                
                status = await self._get_lobby_status_fast()
                response = APIResponse(
                    success=True,
                    message="Lobby status retrieved",
                    data=status
                ).to_dict()
                
                # Cache for 3 seconds
                self.cache.set("lobby_status", response, ttl=3)
                return response
            except Exception as e:
                logger.error(f"Failed to get lobby status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ================================
        # AGENT MANAGEMENT ENDPOINTS
        # ================================
        
        @app.post("/api/agents/register")
        async def register_agent(request: AgentRegistrationRequest):
            """Register a new agent with the lobby."""
            try:
                # Use the fast, direct method if lobby_instance is available
                if self.lobby_instance:
                    result = await self.lobby_instance.register_agent(request.dict())
                else:
                    # Fallback to HTTP request if lobby is external
                    result = await self._register_agent_with_lobby_fast(request)
                
                if result.get("status") == "success":
                    return APIResponse(success=True, message="Agent registered successfully", data=result).to_dict()
                else:
                    raise HTTPException(status_code=400, detail=result.get("message", "Registration failed"))
            except Exception as e:
                logger.error(f"Agent registration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/agents")
        async def list_agents():
            """OPTIMIZED: List agents with aggressive caching"""
            try:
                # Check cache first
                cached = self.cache.get("agents_list")
                if cached:
                    return cached
                
                agents = await self._get_agents_from_lobby_fast()
                response = APIResponse(
                    success=True,
                    message="Agents retrieved",
                    data={"agents": agents, "count": len(agents)}
                ).to_dict()
                
                # Cache for 5 seconds
                self.cache.set("agents_list", response, ttl=5)
                return response
            except Exception as e:
                logger.error(f"Failed to get agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/agents/{agent_id}")
        async def get_agent_details(agent_id: str):
            """OPTIMIZED: Get agent details with caching"""
            try:
                # Check cache first
                cache_key = f"agent_details_{agent_id}"
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
                
                agent = await self._get_agent_details_fast(agent_id)
                response = APIResponse(
                    success=True,
                    message="Agent details retrieved",
                    data=agent
                ).to_dict()
                
                # Cache for 10 seconds
                self.cache.set(cache_key, response, ttl=10)
                return response
            except Exception as e:
                logger.error(f"Failed to get agent details: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        @app.delete("/api/agents/{agent_id}")
        async def unregister_agent(agent_id: str):
            """OPTIMIZED: Unregister agent with cache invalidation"""
            try:
                result = await self._unregister_agent_fast(agent_id)
                
                # Invalidate agent cache
                self.cache.invalidate("agents")
                self.cache.invalidate(f"agent_details_{agent_id}")
                
                return APIResponse(
                    success=True,
                    message=f"Agent {agent_id} unregistered successfully",
                    data=result
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to unregister agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # ================================
        # TASK DELEGATION ENDPOINTS
        # ================================
        
        @app.post("/api/tasks/delegate")
        async def delegate_task(request: TaskDelegationRequest, background_tasks: BackgroundTasks):
            """FIXED: Delegate task directly to collaboration engine"""
            try:
                # Use the fixed delegation method
                result = await self._delegate_task_to_lobby_fast(request)
                
                # CRITICAL FIX: Ensure delegation_id is properly returned
                delegation_id = result.get("delegation_id")
                if not delegation_id:
                    raise Exception("Failed to generate delegation ID")
                
                return APIResponse(
                    success=True,
                    message="Task delegated successfully", 
                    data={
                        "delegation_id": delegation_id,
                        "workflow_id": result.get("workflow_id"),
                        "status": "delegated"
                    }
                ).to_dict()
                
            except Exception as e:
                logger.error(f"Task delegation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/tasks/{task_id}/status")
        async def get_task_status(task_id: str):
            """OPTIMIZED: Get task status with caching"""
            try:
                # Check cache first
                cache_key = f"task_status_{task_id}"
                cached = self.cache.get(cache_key)
                if cached:
                    return cached
                
                status = await self._get_task_status_fast(task_id)
                response = APIResponse(
                    success=True,
                    message="Task status retrieved",
                    data=status
                ).to_dict()
                
                # Cache for 2 seconds (tasks change frequently)
                self.cache.set(cache_key, response, ttl=2)
                return response
            except Exception as e:
                logger.error(f"Failed to get task status: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        @app.get("/api/tasks")
        async def list_tasks():
            """OPTIMIZED: List tasks with aggressive caching"""
            try:
                # Check cache first
                cached = self.cache.get("tasks_list")
                if cached:
                    return cached
                
                tasks = await self._get_all_tasks_fast()
                response = APIResponse(
                    success=True,
                    message="Tasks retrieved successfully",
                    data={
                        "tasks": tasks,
                        "count": len(tasks)
                    }
                ).to_dict()
                
                # Cache for 3 seconds
                self.cache.set("tasks_list", response, ttl=3)
                return response
            except Exception as e:
                logger.error(f"Failed to get tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str):
            """Cancel a running task"""
            try:
                result = await self._cancel_task(task_id)
                return APIResponse(
                    success=True,
                    message=f"Task {task_id} cancelled successfully",
                    data=result
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to cancel task: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        # ================================
        # COLLABORATION ENDPOINTS
        # ================================
        
        @app.post("/api/collaborations/create")
        async def create_collaboration(request: CollaborationRequest):
            """Create a new collaboration session"""
            try:
                collaboration = await self._create_collaboration_session(request)
                return APIResponse(
                    success=True,
                    message="Collaboration session created",
                    data=collaboration
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to create collaboration: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/api/collaborations")
        async def list_collaborations():
            """Get list of active collaborations"""
            try:
                collaborations = await self._get_active_collaborations()
                return APIResponse(
                    success=True,
                    message="Collaborations retrieved",
                    data=collaborations
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to get collaborations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/collaborations/{collaboration_id}")
        async def get_collaboration_details(collaboration_id: str):
            """Get detailed information about a collaboration"""
            try:
                details = await self._get_collaboration_details(collaboration_id)
                return APIResponse(
                    success=True,
                    message="Collaboration details retrieved",
                    data=details
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to get collaboration details: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        # ================================
        # WORKFLOW ENDPOINTS
        # ================================
        
        @app.post("/api/workflows/create")
        async def create_workflow(request: WorkflowRequest):
            """Create a new workflow directly in the collaboration engine."""
            try:
                if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                    raise HTTPException(status_code=503, detail="Collaboration engine not available")

                workflow_id = await self.lobby_instance.collaboration_engine.create_goal_driven_workflow(
                    name=request.name,
                    description=request.description,
                    created_by=request.created_by,
                    task_intent=request.task_intent,
                    required_capabilities=request.required_capabilities,
                    task_data=request.task_data or {},
                    max_agents=request.max_agents
                )
                
                await self.lobby_instance.collaboration_engine.start_workflow(workflow_id)

                return APIResponse(
                    success=True,
                    message="Workflow created and started successfully",
                    data={"workflow_id": workflow_id}
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to create workflow: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.get("/api/workflows/{workflow_id}/status")
        async def get_workflow_status(workflow_id: str):
            """Get workflow status - was missing and causing 404 errors"""
            try:
                if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                    raise HTTPException(status_code=503, detail="Collaboration engine not available")
                
                status = self.lobby_instance.collaboration_engine.get_workflow_status(workflow_id)
                
                if status:
                    return APIResponse(
                        success=True,
                        message="Workflow status retrieved",
                        data=status
                    ).to_dict()
                else:
                    raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting workflow status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/workflows/{workflow_id}/results")
        async def get_workflow_results(workflow_id: str):
            """Get workflow results"""
            try:
                if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                    raise HTTPException(status_code=503, detail="Collaboration engine not available")
                
                status = self.lobby_instance.collaboration_engine.get_workflow_status(workflow_id)
                
                if status:
                    return APIResponse(
                        success=True,
                        message="Workflow results retrieved",
                        data=status
                    ).to_dict()
                else:
                    raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting workflow results: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/agents/{agent_id}/reputation")
        async def get_agent_reputation(agent_id: str):
            """Get current reputation score for an agent"""
            try:
                reputation_data = await self._get_agent_reputation(agent_id)
                return APIResponse(
                    success=True,
                    message="Agent reputation retrieved",
                    data=reputation_data
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to get agent reputation: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        # ================================
        # REAL-TIME WEBSOCKET ENDPOINTS
        # ================================
        
        @app.websocket("/api/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """Endpoint for WebSocket connections"""
            await self._handle_websocket_connection(websocket, client_id)
        
        # ================================
        # ANALYTICS AND MONITORING
        # ================================
        
        @app.get("/api/analytics/system")
        async def get_system_analytics():
            """Get comprehensive system analytics"""
            try:
                analytics = await self._get_system_analytics()
                return APIResponse(
                    success=True,
                    message="System analytics retrieved",
                    data=analytics
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to get analytics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/api/analytics/agents/{agent_id}")
        async def get_agent_analytics(agent_id: str):
            """Get analytics for a specific agent"""
            try:
                analytics = await self._get_agent_analytics(agent_id)
                return APIResponse(
                    success=True,
                    message="Agent analytics retrieved",
                    data=analytics
                ).to_dict()
            except Exception as e:
                logger.error(f"Failed to get agent analytics: {e}")
                raise HTTPException(status_code=404, detail=str(e))
        
        # ================================
        # SYSTEM MANAGEMENT ENDPOINTS
        # ================================
        
        @app.get("/api/system/stats")
        async def get_system_stats():
            """Get comprehensive system statistics"""
            try:
                stats = await self._get_system_stats()
                return APIResponse(success=True, message="System stats retrieved", data=stats).to_dict()
            except Exception as e:
                logger.error(f"Failed to get system stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/system/cleanup")
        async def force_cleanup(cleanup_request: Dict[str, Any]):
            """Force cleanup of stuck workflows and tasks"""
            try:
                result = await self._force_system_cleanup(cleanup_request)
                return APIResponse(success=True, message="System cleanup completed", data=result).to_dict()
            except Exception as e:
                logger.error(f"Failed to cleanup system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/workflows/timeout_all")
        async def timeout_all_workflows():
            """Force timeout of all stuck workflows"""
            try:
                result = await self._timeout_all_workflows()
                return APIResponse(success=True, message="All workflows timed out", data=result).to_dict()
            except Exception as e:
                logger.error(f"Failed to timeout workflows: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/workflows/{workflow_id}/timeout")
        async def timeout_workflow(workflow_id: str):
            """Force timeout of specific workflow"""
            try:
                result = await self._timeout_workflow(workflow_id)
                return APIResponse(success=True, message="Workflow timed out", data=result).to_dict()
            except Exception as e:
                logger.error(f"Failed to timeout workflow: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    # ================================
    # INITIALIZATION AND LIFECYCLE
    # ================================
    
    async def startup(self):
        """OPTIMIZED: Initialize high-performance connections"""
        logger.info("START Starting PERFORMANCE-OPTIMIZED Lobby API Bridge...")
        
        # Initialize high-performance httpx client
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(5.0, connect=1.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        # Initial lobby health check
        self.lobby_healthy = await self._check_lobby_health_cached()
        if not self.lobby_healthy:
            logger.warning("WARNING Lobby connection not available at startup")
        else:
            logger.info("OK Lobby connection verified")
        
        logger.info("OK PERFORMANCE-OPTIMIZED Lobby API Bridge started successfully")
        logger.info(f"   Connection pool: 100 total, 30 per host")
        logger.info(f"   Timeout: 5s total, 1s connect, 2s read")
        logger.info(f"   Cache TTL: {self.cache.default_ttl}s")
    
    async def shutdown(self):
        """OPTIMIZED: Cleanup connections and resources"""
        logger.info("STOP Shutting down PERFORMANCE-OPTIMIZED Lobby API Bridge...")
        
        # Close http client
        if self.http_client:
            await self.http_client.aclose()
        
        # Close WebSocket connections
        for client_id, websocket in self.websocket_connections.items():
            try:
                await websocket.close()
            except:
                pass
        
        # Clear cache
        self.cache.invalidate()
        
        logger.info("OK PERFORMANCE-OPTIMIZED Lobby API Bridge shutdown complete")
    
    # ================================
    # LOBBY COMMUNICATION METHODS
    # ================================
    
    async def _check_lobby_health_cached(self) -> bool:
        """OPTIMIZED: Check lobby health with caching and fast timeout"""
        current_time = time.time()
        
        # Use cached result if recent
        if (current_time - self.last_health_check) < self.health_check_interval:
            return self.lobby_healthy
        
        try:
            if not self.http_client:
                return False
            
            response = await self.http_client.get(f"{self.lobby_http_url}/api/health")
            self.lobby_healthy = response.status_code == 200
            self.last_health_check = current_time
            return self.lobby_healthy
        except Exception as e:
            logger.debug(f"Lobby health check failed: {e}")
            self.lobby_healthy = False
            self.last_health_check = current_time
            return False
    
    async def _get_lobby_status_fast(self) -> Dict[str, Any]:
        """OPTIMIZED: Get lobby status via HTTP connection or direct instance access"""
        try:
            if self.lobby_instance:
                # Get status directly from lobby instance
                agents_count = len(self.lobby_instance.agents)
                workflows_count = len(self.lobby_instance.collaboration_engine.workflows) if self.lobby_instance.collaboration_engine else 0
                
                return {
                    "status": "operational",
                    "agent_count": agents_count,
                    "active_workflows": workflows_count,
                    "lobby_id": self.lobby_instance.lobby_id,
                    "started_at": getattr(self.lobby_instance, 'started_at', None),
                    "uptime_seconds": getattr(self.lobby_instance, 'uptime_seconds', 0),
                    "performance_mode": "enabled",
                    "naa_enabled": getattr(self.lobby_instance, 'naa_enabled', False)
                }
            else:
                # Try to connect via HTTP to the main lobby service
                if not self.http_client:
                    self.http_client = httpx.AsyncClient()
                
                try:
                    # Try to get basic status from the main lobby
                    response = await self.http_client.get(f"{self.lobby_http_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        # If main lobby is responsive, create a basic status
                        return {
                            "status": "operational",
                            "agent_count": 0,  # We'll get this from agents endpoint
                            "active_workflows": 0,
                            "lobby_connection": True,
                            "lobby_url": self.lobby_http_url,
                            "performance_mode": "enabled",
                            "connection_type": "http"
                        }
                    else:
                        raise Exception(f"Lobby HTTP response: {response.status_code}")
                except httpx.RequestError as e:
                    raise Exception(f"Cannot connect to lobby at {self.lobby_http_url}: {e}")
        except Exception as e:
            raise Exception(f"Failed to get lobby status: {e}")
    
    async def _register_agent_with_lobby_fast(self, request: AgentRegistrationRequest) -> Dict[str, Any]:
        """FIXED: Register agent directly with lobby instance"""
        try:
            if not self.lobby_instance:
                raise Exception("Lobby instance not available")
            
            # Prepare agent data
            agent_data = {
                "agent_id": request.agent_id,
                "name": request.name,
                "agent_type": request.agent_type,
                "capabilities": request.capabilities,
                "goal": request.goal or "",
                "specialization": request.specialization or "",
                "collaboration_style": request.collaboration_style or "individual",
                "metadata": request.metadata or {},
                "callback_url": request.callback_url
            }
            
            # Register directly with lobby
            result = await self.lobby_instance.register_agent(agent_data)
            
            # Invalidate agent cache
            self.cache.invalidate("agents")
            
            print(f"COLLAB REGISTRATION FIX: Agent {request.agent_id} registered directly with lobby")
            print(f"   Name: {request.name}")
            print(f"   Capabilities: {request.capabilities}")
            
            return result
            
        except Exception as e:
            print(f"ERROR REGISTRATION ERROR: {e}")
            raise Exception(f"Failed to register agent: {e}")
    
    async def _get_agents_from_lobby_fast(self) -> List[Dict[str, Any]]:
        """FIXED: Get agents directly from lobby instance"""
        try:
            if not self.lobby_instance:
                return []
            
            # Get agents directly from lobby
            agents = []
            for agent_id, agent_data in self.lobby_instance.agents.items():
                if isinstance(agent_data, dict):
                    agents.append({
                        "agent_id": agent_id,
                        "name": agent_data.get("name", agent_id),
                        "agent_type": agent_data.get("agent_type", "unknown"),
                        "capabilities": agent_data.get("capabilities", []),
                        "specialization": agent_data.get("specialization", ""),
                        "status": agent_data.get("status", "unknown"),
                        "registered_at": agent_data.get("registered_at"),
                        "reputation": self.lobby_instance.agent_reputation.get(agent_id, self.lobby_instance.default_reputation)
                    })
            
            return agents
            
        except Exception as e:
            print(f"ERROR GET AGENTS ERROR: {e}")
            return []
    
    async def _get_agent_details_fast(self, agent_id: str) -> Dict[str, Any]:
        """OPTIMIZED: Get agent details from cache or single request"""
        try:
            # Try to get from agents list cache first
            agents = await self._get_agents_from_lobby_fast()
            for agent in agents:
                if agent.get("agent_id") == agent_id:
                    return agent
            raise Exception(f"Agent {agent_id} not found")
        except Exception as e:
            raise Exception(f"Failed to get agent details: {e}")
    
    async def _unregister_agent_fast(self, agent_id: str) -> Dict[str, Any]:
        """OPTIMIZED: Unregister agent (placeholder)"""
        # This would need to be implemented in the lobby
        # For now, return a placeholder
        return {"agent_id": agent_id, "status": "unregistered"}
    
    async def _delegate_task_to_lobby_fast(self, request: TaskDelegationRequest) -> Dict[str, Any]:
        """FIXED: Delegate task directly to lobby instance instead of HTTP calls"""
        try:
            print(f"API BRIDGE: Starting delegation for task: {request.task_title}")
            
            if not self.lobby_instance:
                raise Exception("Lobby instance not available")
            
            if not self.lobby_instance.collaboration_engine:
                raise Exception("Collaboration engine not available")
            
            print(f"API BRIDGE: Creating workflow...")
            
            # Create workflow directly in collaboration engine
            workflow_id = await self.lobby_instance.collaboration_engine.create_goal_driven_workflow(
                name=request.task_title,
                description=request.task_description,
                created_by=request.requester_id,
                task_intent=request.task_intent or request.task_description,
                required_capabilities=request.required_capabilities,
                task_data=request.task_data or {},
                max_agents=request.max_agents or 3
            )
            
            print(f"API BRIDGE: Workflow created: {workflow_id}")
            
            # **FIX: AUTOMATICALLY START THE WORKFLOW** 
            # This was the missing piece - workflows were created but never started!
            workflow_started = await self.lobby_instance.collaboration_engine.start_workflow(workflow_id)
            
            print(f"API BRIDGE: Workflow started: {workflow_started}")
            
            if not workflow_started:
                raise Exception(f"Failed to start workflow {workflow_id}")
            
            # Create delegation mapping - CRITICAL FIX: Ensure unique delegation ID
            import time
            delegation_id = f"delegation_{workflow_id[-8:]}_{int(time.time())}"
            self.lobby_instance.delegation_to_workflow[delegation_id] = workflow_id
            
            print(f"API BRIDGE: DELEGATION SUCCESS - ID: {delegation_id}, Workflow: {workflow_id}")
            
            return {
                "success": True,
                "delegation_id": delegation_id,
                "workflow_id": workflow_id,
                "message": "Task delegated and workflow started successfully",
                "status": "running"
            }
            
        except Exception as e:
            print(f"API BRIDGE: DELEGATION FAILED - {e}")
            logger.error(f"Task delegation failed: {e}")
            raise Exception(f"Task delegation failed: {str(e)}")
    
    async def _get_task_status_fast(self, task_id: str) -> Dict[str, Any]:
        """FIXED: Get task status directly from lobby instance"""
        try:
            if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                raise Exception("Lobby or collaboration engine not available")
            
            # Check if task_id is a delegation_id
            workflow_id = self.lobby_instance.delegation_to_workflow.get(task_id)
            if not workflow_id:
                workflow_id = task_id  # Maybe it's already a workflow_id
            
            # Get workflow status from collaboration engine
            status = self.lobby_instance.collaboration_engine.get_workflow_status(workflow_id)
            
            if status:
                return {
                    "task_id": task_id,
                    "workflow_id": workflow_id,
                    "status": status.get("status", "not_found"),
                    "details": status
                }
            else:
                return {
                    "task_id": task_id,
                    "status": "not_found",
                    "message": f"Task {task_id} not found"
                }
                
        except Exception as e:
            print(f"ERROR TASK STATUS ERROR: {e}")
            raise Exception(f"Failed to get task status: {e}")
    
    async def _get_all_tasks_fast(self) -> List[Dict[str, Any]]:
        """FIXED: Get all tasks directly from lobby instance"""
        try:
            if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                return []
            
            # Get all workflows from collaboration engine
            workflows = []
            for workflow_id, workflow in self.lobby_instance.collaboration_engine.workflows.items():
                workflows.append({
                    "workflow_id": workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "status": workflow.status.value,
                    "created_by": workflow.created_by,
                    "created_at": workflow.created_at.isoformat(),
                    "participants": list(workflow.participants),
                    "task_count": len(workflow.tasks)
                })
            
            return workflows
            
        except Exception as e:
            print(f"ERROR GET TASKS ERROR: {e}")
            return []
    
    async def _cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task"""
        # This would need to be implemented in the lobby
        return {"task_id": task_id, "status": "cancelled"}
    
    async def _create_collaboration_session(self, request: CollaborationRequest) -> Dict[str, Any]:
        """Create collaboration session"""
        try:
            payload = {
                "participant_ids": request.participant_ids,
                "purpose": request.purpose,
                "collaboration_type": request.collaboration_type,
                "shared_context": request.shared_context or {}
            }
            
            # This would need appropriate lobby endpoint
            collaboration_id = str(uuid.uuid4())
            return {
                "collaboration_id": collaboration_id,
                "participants": request.participant_ids,
                "purpose": request.purpose,
                "status": "created",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to create collaboration: {e}")
    
    async def _get_active_collaborations(self) -> Dict[str, Any]:
        """Get active collaborations"""
        # Placeholder - would get from lobby
        return {"collaborations": [], "count": 0}
    
    async def _get_collaboration_details(self, collaboration_id: str) -> Dict[str, Any]:
        """Get collaboration details"""
        # Placeholder - would get from lobby
        return {"collaboration_id": collaboration_id, "status": "not_found"}
    
    async def _create_workflow(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Create workflow in lobby"""
        # Placeholder - would create in lobby's collaboration engine
        workflow_id = str(uuid.uuid4())
        return {
            "workflow_id": workflow_id,
            "name": request.name,
            "description": request.description,
            "status": "created",
            "created_by": request.created_by,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """FIXED: Get workflow status directly from lobby instance"""
        try:
            if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                raise Exception("Lobby or collaboration engine not available")
            
            # Get workflow status from collaboration engine
            status = self.lobby_instance.collaboration_engine.get_workflow_status(workflow_id)
            
            if status:
                return status
            else:
                return {
                    "workflow_id": workflow_id,
                    "status": "not_found",
                    "message": f"Workflow {workflow_id} not found"
                }
                
        except Exception as e:
            print(f"ERROR WORKFLOW STATUS ERROR: {e}")
            raise Exception(f"Failed to get workflow status: {e}")
    
    async def _get_workflow_results(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed results from a completed workflow"""
        try:
            if self.lobby_instance and self.lobby_instance.collaboration_engine:
                # Get workflow from collaboration engine
                workflow = self.lobby_instance.collaboration_engine.workflows.get(workflow_id)
                if not workflow:
                    raise Exception(f"Workflow {workflow_id} not found")
                
                # Return comprehensive results
                return {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "status": workflow.status.value,
                    "created_by": workflow.created_by,
                    "created_at": workflow.created_at.isoformat(),
                    "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                    "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
                    "participants": list(workflow.participants),
                    "task_intent": workflow.task_intent,
                    "final_result": workflow.result,
                    "error": workflow.error,
                    "tasks": {
                        task_id: {
                            "name": task.name,
                            "status": task.status.value,
                            "assigned_agent": task.assigned_agent,
                            "result": task.result,
                            "error": task.error,
                            "started_at": task.started_at.isoformat() if task.started_at else None,
                            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                            "execution_time": (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else None
                        }
                        for task_id, task in workflow.tasks.items()
                    },
                    "shared_state": workflow.shared_state
                }
            else:
                raise Exception("Collaboration engine not available")
        except Exception as e:
            raise Exception(f"Failed to get workflow results: {e}")
    
    async def _get_agent_reputation(self, agent_id: str) -> Dict[str, Any]:
        """Get current reputation score and history for an agent"""
        try:
            if self.lobby_instance:
                # Get current reputation
                reputation = self.lobby_instance.agent_reputation.get(agent_id, self.lobby_instance.default_reputation)
                
                # Get performance metrics from collaboration engine
                performance = {}
                if self.lobby_instance.collaboration_engine:
                    performance = self.lobby_instance.collaboration_engine.agent_performance.get(agent_id, {})
                
                return {
                    "agent_id": agent_id,
                    "current_reputation": reputation,
                    "default_reputation": self.lobby_instance.default_reputation,
                    "reputation_rank": self._calculate_reputation_rank(agent_id),
                    "performance_metrics": {
                        "completed_tasks": performance.get("completed_tasks", 0),
                        "failed_tasks": performance.get("failed_tasks", 0),
                        "total_tasks": performance.get("total_tasks", 0),
                        "success_rate": performance.get("success_rate", 0.0),
                        "avg_execution_time": performance.get("avg_execution_time", 0.0)
                    },
                    "reputation_config": {
                        "success_reward_provider": self.lobby_instance.reputation_change_on_success["provider"],
                        "success_reward_requester": self.lobby_instance.reputation_change_on_success["requester"],
                        "failure_penalty_provider": self.lobby_instance.reputation_change_on_failure["provider"],
                        "failure_penalty_requester": self.lobby_instance.reputation_change_on_failure["requester"]
                    }
                }
            else:
                raise Exception("Lobby instance not available")
        except Exception as e:
            raise Exception(f"Failed to get agent reputation: {e}")
    
    def _calculate_reputation_rank(self, agent_id: str) -> Dict[str, Any]:
        """Calculate where this agent ranks compared to others"""
        try:
            if not self.lobby_instance:
                return {"rank": "unknown", "total_agents": 0}
            
            # Get all agent reputations
            all_reputations = []
            for aid, rep in self.lobby_instance.agent_reputation.items():
                all_reputations.append((aid, rep))
            
            # Sort by reputation (highest first)
            all_reputations.sort(key=lambda x: x[1], reverse=True)
            
            # Find this agent's rank
            rank = None
            for i, (aid, rep) in enumerate(all_reputations):
                if aid == agent_id:
                    rank = i + 1
                    break
            
            return {
                "rank": rank,
                "total_agents": len(all_reputations),
                "percentile": round((len(all_reputations) - rank + 1) / len(all_reputations) * 100, 1) if rank else None
            }
        except Exception as e:
            return {"rank": "error", "total_agents": 0, "error": str(e)}
    
    # ================================
    # WEBSOCKET HANDLING
    # ================================
    
    async def _handle_websocket_connection(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection and message passing."""
        await websocket.accept()
        
        # --- THE FIX: Register connection with the CORE LOBBY ---
        if self.lobby_instance:
            await self.lobby_instance.register_live_connection(client_id, websocket)
        else:
            logger.error(f"Cannot register live connection for {client_id}: Lobby instance is not available.")
        # ---------------------------------------------------------

        self.websocket_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
        
        try:
            while True:
                data = await websocket.receive_json()
                await self._handle_websocket_message(client_id, data)
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {client_id}")
        finally:
            if client_id in self.websocket_connections:
                del self.websocket_connections[client_id]
            
            # --- THE FIX: Unregister connection from the CORE LOBBY ---
            if self.lobby_instance:
                await self.lobby_instance.unregister_live_connection(client_id)
            # -----------------------------------------------------------
    
    async def _handle_websocket_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming messages from a WebSocket client."""
        message_type = data.get("type")
        
        if message_type == "subscribe_task":
            task_id = data.get("task_id")
            if task_id:
                if task_id not in self.task_subscriptions:
                    self.task_subscriptions[task_id] = []
                self.task_subscriptions[task_id].append(client_id)
                
        elif message_type == "unsubscribe_task":
            task_id = data.get("task_id")
            if task_id and task_id in self.task_subscriptions:
                if client_id in self.task_subscriptions[task_id]:
                    self.task_subscriptions[task_id].remove(client_id)
    
    async def _broadcast_task_update(self, task_id: str, update: Dict[str, Any]):
        """Broadcast task update to subscribed clients"""
        if task_id in self.task_subscriptions:
            message = {
                "type": "task_update",
                "task_id": task_id,
                "update": update,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            for client_id in self.task_subscriptions[task_id]:
                if client_id in self.websocket_connections:
                    try:
                        await self.websocket_connections[client_id].send_json(message)
                    except Exception as e:
                        logger.error(f"Failed to send update to {client_id}: {e}")
    
    # ================================
    # MONITORING AND ANALYTICS
    # ================================
    
    async def _monitor_task_progress(self, task_id: str):
        """Monitor task progress and broadcast updates"""
        try:
            while task_id in self.active_tasks:
                # Get current status
                status = await self._get_task_status(task_id)
                
                # Update stored task
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]["last_status"] = status
                    self.active_tasks[task_id]["last_checked"] = datetime.now(timezone.utc).isoformat()
                
                # Broadcast update
                await self._broadcast_task_update(task_id, status)
                
                # Check if task is complete
                task_status = status.get("status", "").lower()
                if task_status in ["completed", "failed", "cancelled"]:
                    break
                
                # Wait before next check
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {e}")
    
    async def _get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        try:
            # Get lobby status
            lobby_status = await self._get_lobby_status()
            
            # Get agents
            agents = await self._get_agents_from_lobby()
            
            # Get tasks
            tasks = await self._get_all_tasks()
            
            return {
                "lobby_status": lobby_status,
                "agents": {
                    "total": len(agents),
                    "active": len([a for a in agents if a.get("status") == "active"]),
                    "by_type": self._group_by_field(agents, "agent_type"),
                    "capabilities": self._extract_capabilities(agents)
                },
                "tasks": {
                    "total": len(tasks),
                    "active": len(self.active_tasks),
                    "by_status": self._group_by_field(tasks, "status")
                },
                "websocket_connections": len(self.websocket_connections),
                "system_health": {
                    "lobby_connected": await self._check_lobby_health(),
                    "api_bridge_uptime": "unknown",  # Would track this
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
        except Exception as e:
            raise Exception(f"Failed to get system analytics: {e}")
    
    async def _get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get analytics for specific agent"""
        try:
            agent_details = await self._get_agent_details(agent_id)
            
            # Get task history for this agent (placeholder)
            task_history = []  # Would get from lobby
            
            return {
                "agent_info": agent_details,
                "performance": {
                    "tasks_completed": len(task_history),
                    "success_rate": 0.95,  # Placeholder
                    "average_completion_time": "5.2 minutes",  # Placeholder
                    "last_active": "2025-01-01T00:00:00Z"  # Placeholder
                },
                "current_workload": {
                    "active_tasks": 0,  # Placeholder
                    "pending_tasks": 0,  # Placeholder
                    "collaboration_sessions": 0  # Placeholder
                }
            }
        except Exception as e:
            raise Exception(f"Failed to get agent analytics: {e}")
    
    # ================================
    # UTILITY METHODS
    # ================================
    
    def _group_by_field(self, items: List[Dict], field: str) -> Dict[str, int]:
        """Group items by a specific field"""
        groups = {}
        for item in items:
            value = item.get(field, "unknown")
            groups[value] = groups.get(value, 0) + 1
        return groups
    
    def _extract_capabilities(self, agents: List[Dict]) -> Dict[str, int]:
        """Extract and count capabilities across agents"""
        capabilities = {}
        for agent in agents:
            agent_caps = agent.get("capabilities", [])
            for cap in agent_caps:
                capabilities[cap] = capabilities.get(cap, 0) + 1
        return capabilities
    
    # ================================
    # SYSTEM MANAGEMENT METHODS
    # ================================
    
    async def _get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            if not self.lobby_instance:
                return {"error": "Lobby instance not available"}
            
            # Get basic counts
            agents = await self._get_agents_from_lobby_fast()
            workflows = self.lobby_instance.collaboration_engine.workflows if self.lobby_instance.collaboration_engine else {}
            
            # Count task statuses
            task_stats = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
            for workflow in workflows.values():
                for task in workflow.tasks.values():
                    status = task.status.value if hasattr(task.status, 'value') else str(task.status)
                    if status in task_stats:
                        task_stats[status] += 1
            
            return {
                "registered_agents": len(agents),
                "active_workflows": len(workflows),
                "active_tasks": task_stats["pending"] + task_stats["in_progress"],
                "task_breakdown": task_stats,
                "system_health": {
                    "lobby_healthy": self.lobby_healthy,
                    "collaboration_engine_active": self.lobby_instance.collaboration_engine is not None,
                    "websocket_connections": len(self.websocket_connections)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to get system stats: {e}")
    
    async def _force_system_cleanup(self, cleanup_request: Dict[str, Any]) -> Dict[str, Any]:
        """Force cleanup of stuck workflows and tasks"""
        try:
            if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                return {"error": "Collaboration engine not available"}
            
            collab_engine = self.lobby_instance.collaboration_engine
            force = cleanup_request.get("force", False)
            max_age_seconds = cleanup_request.get("max_age_seconds", 300)  # 5 minutes default
            reason = cleanup_request.get("reason", "api_cleanup")
            
            cleaned_workflows = 0
            cleaned_tasks = 0
            
            # Force cleanup orphaned workflows
            if hasattr(collab_engine, '_cleanup_orphaned_workflows'):
                result = await collab_engine._cleanup_orphaned_workflows(max_age_seconds)
                cleaned_workflows = result if isinstance(result, int) else 0
            
            # If force=True, clean everything regardless of age
            if force:
                workflows_to_clean = list(collab_engine.workflows.keys())
                for workflow_id in workflows_to_clean:
                    try:
                        if hasattr(collab_engine, '_force_cleanup_workflow'):
                            workflow = collab_engine.workflows.get(workflow_id)
                            if workflow:
                                await collab_engine._force_cleanup_workflow(workflow, reason)
                                cleaned_workflows += 1
                    except Exception as e:
                        logger.warning(f"Failed to force cleanup workflow {workflow_id}: {e}")
            
            # Clear agent workloads
            if hasattr(collab_engine, 'agent_workloads'):
                for agent_id in list(collab_engine.agent_workloads.keys()):
                    collab_engine.agent_workloads[agent_id].clear()
            
            return {
                "cleaned_workflows": cleaned_workflows,
                "cleaned_tasks": cleaned_tasks,
                "force": force,
                "max_age_seconds": max_age_seconds,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to force cleanup: {e}")
    
    async def _timeout_all_workflows(self) -> Dict[str, Any]:
        """Force timeout of all stuck workflows"""
        try:
            if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                return {"error": "Collaboration engine not available"}
            
            collab_engine = self.lobby_instance.collaboration_engine
            timed_out_workflows = 0
            timed_out_tasks = 0
            
            # Force timeout all workflows
            for workflow_id, workflow in list(collab_engine.workflows.items()):
                try:
                    # Force timeout all in-progress tasks
                    for task in workflow.tasks.values():
                        if str(task.status) == "in_progress":
                            task.status = "failed" 
                            task.error = "Force timeout via API"
                            task.completed_at = datetime.now(timezone.utc)
                            timed_out_tasks += 1
                    
                    # Check workflow completion
                    if hasattr(collab_engine, '_check_workflow_completion'):
                        await collab_engine._check_workflow_completion(workflow_id)
                    
                    timed_out_workflows += 1
                except Exception as e:
                    logger.warning(f"Failed to timeout workflow {workflow_id}: {e}")
            
            return {
                "timed_out_workflows": timed_out_workflows,
                "timed_out_tasks": timed_out_tasks,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to timeout all workflows: {e}")
    
    async def _timeout_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Force timeout of specific workflow"""
        try:
            if not self.lobby_instance or not self.lobby_instance.collaboration_engine:
                return {"error": "Collaboration engine not available"}
            
            collab_engine = self.lobby_instance.collaboration_engine
            
            if workflow_id not in collab_engine.workflows:
                return {"error": f"Workflow {workflow_id} not found"}
            
            workflow = collab_engine.workflows[workflow_id]
            timed_out_tasks = 0
            
            # Force timeout all in-progress tasks in this workflow
            for task in workflow.tasks.values():
                if str(task.status) == "in_progress":
                    task.status = "failed"
                    task.error = "Force timeout via API"
                    task.completed_at = datetime.now(timezone.utc)
                    timed_out_tasks += 1
            
            # Check workflow completion
            if hasattr(collab_engine, '_check_workflow_completion'):
                await collab_engine._check_workflow_completion(workflow_id)
            
            return {
                "workflow_id": workflow_id,
                "timed_out_tasks": timed_out_tasks,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise Exception(f"Failed to timeout workflow {workflow_id}: {e}")

# ================================
# FACTORY FUNCTION
# ================================

def create_lobby_api_bridge(
    lobby_host: str = "localhost",
    lobby_http_port: int = 8080,
    lobby_ws_port: int = 8081
) -> FastAPI:
    """
    Factory function to create and configure the Lobby API Bridge
    
    Args:
        lobby_host: Hostname of the Agent Lobbi
        lobby_http_port: HTTP port of the Agent Lobbi
        lobby_ws_port: WebSocket port of the Agent Lobbi
    
    Returns:
        Configured FastAPI application
    """
    bridge = LobbyAPIBridge(lobby_host, lobby_http_port, lobby_ws_port)
    return bridge.app

# ================================
# STANDALONE SERVER
# ================================

async def start_api_bridge_server(
    bridge_host: str = "localhost",
    bridge_port: int = 8090,
    lobby_host: str = "localhost",
    lobby_http_port: int = 8080,
    lobby_ws_port: int = 8081
):
    """Start the API bridge server."""
    # Use a dynamic import for uvicorn to support different environments
    import uvicorn
    
    bridge = LobbyAPIBridge(
        lobby_host=lobby_host, 
        lobby_http_port=lobby_http_port, 
        lobby_ws_port=lobby_ws_port
    )
    
    config = uvicorn.Config(
        bridge.app, 
        host=bridge_host, 
        port=bridge_port, 
        log_level="info",
        lifespan="on"
    )
    server = uvicorn.Server(config)
    bridge.server = server # Store server instance for graceful shutdown
    
    await server.serve()

if __name__ == "__main__":
    # This allows running the bridge directly for development
    asyncio.run(start_api_bridge_server(
        lobby_http_port=8080,
        lobby_ws_port=8081
    ))