#!/usr/bin/env python3
"""
ENHANCED LOBBY INTEGRATION WITH DATA BUS + TRAFFIC LIGHTS
========================================================
This module integrates the new orchestration system with existing Agent Lobby.

Key Features:
1. Backward compatibility with existing APIs
2. Enhanced multi-agent collaboration
3. Real-time traffic monitoring dashboard
4. Automatic agent pool discovery
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .data_bus_orchestrator import (
    WorkflowOrchestrator, 
    WorkflowFactory, 
    StageDefinition,
    StandardizedMessage
)

logger = logging.getLogger(__name__)

class EnhancedAgentLobby:
    """Enhanced Agent Lobby with Data Bus + Traffic Light orchestration"""
    
    def __init__(self, original_lobby):
        self.original_lobby = original_lobby
        self.orchestrator = WorkflowOrchestrator(original_lobby)
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.workflow_patterns: Dict[str, Any] = {}
        
        logger.info("🎯 Enhanced Agent Lobby initialized")
        logger.info("   Data Bus + Traffic Light orchestration enabled")
        
    async def initialize(self):
        """Initialize the enhanced lobby system"""
        # Discover agent capabilities from registered agents
        await self._discover_agent_capabilities()
        
        # Setup common workflow patterns
        await self._setup_workflow_patterns()
        
        # Integrate with original lobby message handling
        await self._integrate_message_handling()
        
        logger.info("🎯 Enhanced Agent Lobby fully operational")
        
    async def _discover_agent_capabilities(self):
        """Discover capabilities of registered agents"""
        # Get agents from the database
        db = self.original_lobby.db
        agents = await db.get_all_agents()
        
        capability_pools = {}
        
        for agent in agents:
            agent_id = agent.get("agent_id")
            capabilities = agent.get("capabilities", [])
            
            if not agent_id or not capabilities:
                continue
                
            self.agent_capabilities[agent_id] = capabilities
            
            # Group agents by capability
            for capability in capabilities:
                if capability not in capability_pools:
                    capability_pools[capability] = []
                capability_pools[capability].append(agent_id)
                
        logger.info(f"🔍 Discovered {len(self.agent_capabilities)} agents")
        for capability, agents in capability_pools.items():
            logger.info(f"   {capability}: {len(agents)} agents")
            
        return capability_pools
        
    async def _setup_workflow_patterns(self):
        """Setup common workflow patterns"""
        capability_pools = await self._discover_agent_capabilities()
        
        # META Analysis Pattern
        if all(cap in capability_pools for cap in ["financial_analysis", "data_analysis", "content_creation"]):
            meta_stages = WorkflowFactory.create_meta_analysis_workflow(
                self.orchestrator, 
                capability_pools
            )
            self.workflow_patterns["meta_analysis"] = {
                "stages": [stage.stage_name for stage in meta_stages],
                "entry_stage": "financial_analysis",
                "description": "Comprehensive META stock analysis with multi-agent collaboration"
            }
            logger.info("🏭 META analysis workflow pattern ready")
            
        # General Analysis Pattern (3-stage pipeline)
        if len(capability_pools) >= 2:
            capabilities = list(capability_pools.keys())
            stages = []
            
            # Create sequential stages from available capabilities
            for i, capability in enumerate(capabilities[:3]):  # Max 3 stages
                outputs_to = [capabilities[i+1]] if i+1 < len(capabilities) else []
                dependencies = [capabilities[i-1]] if i > 0 else []
                
                stage = StageDefinition(
                    stage_name=capability,
                    required_capability=capability,
                    agent_pool=capability_pools[capability],
                    max_concurrent=2,  # Allow parallel processing
                    dependencies=dependencies,
                    outputs_to=outputs_to
                )
                stages.append(stage)
                self.orchestrator.register_stage(stage)
                
            if stages:
                self.workflow_patterns["general_analysis"] = {
                    "stages": [stage.stage_name for stage in stages],
                    "entry_stage": stages[0].stage_name,
                    "description": f"General analysis pipeline: {' → '.join([s.stage_name for s in stages])}"
                }
                logger.info("🏭 General analysis workflow pattern ready")
    
    async def _integrate_message_handling(self):
        """Integrate with original lobby message handling"""
        # Store original message handler
        self.original_process_message = self.original_lobby._process_single_message
        
        # Replace with enhanced handler
        self.original_lobby._process_single_message = self._enhanced_message_handler
        
        logger.info("🔌 Message handling integration complete")
    
    async def _enhanced_message_handler(self, message):
        """Enhanced message handler that supports both old and new systems"""
        # Check if this is a response to an orchestrated workflow
        task_id = message.payload.get("task_id")
        workflow_id = message.conversation_id
        
        if (task_id and workflow_id and 
            hasattr(message, 'message_type') and 
            message.message_type.name == "RESPONSE"):
            
            # This is likely a response to an orchestrated task
            response = {
                "status": "success" if message.payload.get("status") == "success" else "failed",
                "result": message.payload.get("result"),
                "error": message.payload.get("error")
            }
            
            handled = await self.orchestrator.handle_agent_response(task_id, response, workflow_id)
            if handled:
                logger.info(f"🎛️ Orchestrator handled response for task {task_id}")
                return
        
        # Fall back to original message handling
        await self.original_process_message(message)
    
    # ============================================
    # PUBLIC API METHODS
    # ============================================
    
    async def create_orchestrated_workflow(self, workflow_type: str, goal: str, 
                                         initial_data: Dict[str, Any], 
                                         requester_id: str) -> str:
        """Create a new orchestrated workflow"""
        if workflow_type not in self.workflow_patterns:
            available = list(self.workflow_patterns.keys())
            raise ValueError(f"Unknown workflow type '{workflow_type}'. Available: {available}")
        
        pattern = self.workflow_patterns[workflow_type]
        entry_stage = pattern["entry_stage"]
        
        workflow_id = await self.orchestrator.start_workflow(
            workflow_name=f"{workflow_type.title()} Workflow",
            goal=goal,
            initial_data=initial_data,
            entry_stage=entry_stage,
            requester_id=requester_id
        )
        
        logger.info(f"🎯 Created orchestrated workflow: {workflow_id}")
        logger.info(f"   Type: {workflow_type}")
        logger.info(f"   Goal: {goal}")
        
        return workflow_id
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an orchestrated workflow"""
        return self.orchestrator.get_workflow_status(workflow_id)
    
    async def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system status dashboard"""
        orchestrator_status = self.orchestrator.get_system_status()
        
        # Get original lobby stats
        try:
            lobby_stats = {
                "total_agents": len(self.agent_capabilities),
                "capabilities": list(set().union(*self.agent_capabilities.values())) if self.agent_capabilities else [],
                "workflow_patterns": len(self.workflow_patterns)
            }
        except Exception as e:
            lobby_stats = {"error": str(e)}
        
        return {
            "enhanced_lobby": {
                "status": "operational",
                "agents": lobby_stats,
                "orchestrator": orchestrator_status,
                "available_patterns": list(self.workflow_patterns.keys())
            },
            "traffic_lights": orchestrator_status.get("traffic_lights", {}),
            "active_workflows": orchestrator_status.get("active_workflows", 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def list_available_patterns(self) -> Dict[str, Any]:
        """List available workflow patterns"""
        return {
            pattern_name: {
                "description": pattern["description"],
                "stages": pattern["stages"],
                "entry_stage": pattern["entry_stage"]
            }
            for pattern_name, pattern in self.workflow_patterns.items()
        }
    
    # ============================================
    # BACKWARD COMPATIBILITY
    # ============================================
    
    async def create_goal_driven_workflow(self, goal: str, required_capabilities: List[str], 
                                        max_agents: int = 3, deadline_minutes: int = 15,
                                        requester_id: str = "system") -> str:
        """
        Backward compatible method that creates orchestrated workflows
        This replaces the broken original method with working orchestration
        """
        
        logger.info(f"🔄 Converting legacy workflow request to orchestrated workflow")
        logger.info(f"   Goal: {goal}")
        logger.info(f"   Required capabilities: {required_capabilities}")
        
        # Determine best workflow pattern
        workflow_type = self._select_best_pattern(required_capabilities)
        
        if not workflow_type:
            # Create a custom workflow for these capabilities
            workflow_type = await self._create_custom_workflow(required_capabilities)
        
        # Create initial data from legacy parameters
        initial_data = {
            "goal": goal,
            "required_capabilities": required_capabilities,
            "max_agents": max_agents,
            "deadline_minutes": deadline_minutes,
            "legacy_request": True,
            "stock_symbol": "META",  # Default for compatibility
            "analysis_type": "comprehensive"
        }
        
        workflow_id = await self.create_orchestrated_workflow(
            workflow_type=workflow_type,
            goal=goal,
            initial_data=initial_data,
            requester_id=requester_id
        )
        
        logger.info(f"✅ Legacy workflow converted to orchestrated workflow: {workflow_id}")
        return workflow_id
    
    def _select_best_pattern(self, required_capabilities: List[str]) -> Optional[str]:
        """Select the best workflow pattern for given capabilities"""
        # Check for exact matches first
        for pattern_name, pattern in self.workflow_patterns.items():
            pattern_capabilities = pattern["stages"]
            if set(required_capabilities).issubset(set(pattern_capabilities)):
                logger.info(f"🎯 Selected pattern '{pattern_name}' for capabilities {required_capabilities}")
                return pattern_name
        
        # Check for partial matches
        best_match = None
        best_score = 0
        
        for pattern_name, pattern in self.workflow_patterns.items():
            pattern_capabilities = pattern["stages"]
            match_score = len(set(required_capabilities) & set(pattern_capabilities))
            
            if match_score > best_score:
                best_score = match_score
                best_match = pattern_name
        
        if best_match and best_score > 0:
            logger.info(f"🎯 Selected partial match pattern '{best_match}' (score: {best_score}/{len(required_capabilities)})")
            
        return best_match
    
    async def _create_custom_workflow(self, required_capabilities: List[str]) -> str:
        """Create a custom workflow for specific capabilities"""
        capability_pools = await self._discover_agent_capabilities()
        
        # Filter to only available capabilities
        available_capabilities = [cap for cap in required_capabilities if cap in capability_pools]
        
        if not available_capabilities:
            raise ValueError(f"No agents available for capabilities: {required_capabilities}")
        
        # Create sequential workflow
        stages = []
        for i, capability in enumerate(available_capabilities):
            outputs_to = [available_capabilities[i+1]] if i+1 < len(available_capabilities) else []
            dependencies = [available_capabilities[i-1]] if i > 0 else []
            
            stage = StageDefinition(
                stage_name=capability,
                required_capability=capability,
                agent_pool=capability_pools[capability],
                max_concurrent=1,
                dependencies=dependencies,
                outputs_to=outputs_to
            )
            stages.append(stage)
            self.orchestrator.register_stage(stage)
        
        # Register custom pattern
        pattern_name = f"custom_{'_'.join(available_capabilities)}"
        self.workflow_patterns[pattern_name] = {
            "stages": [stage.stage_name for stage in stages],
            "entry_stage": stages[0].stage_name,
            "description": f"Custom workflow: {' → '.join(available_capabilities)}"
        }
        
        logger.info(f"🏭 Created custom workflow pattern: {pattern_name}")
        return pattern_name
    
    # ============================================
    # PROXY METHODS FOR COMPATIBILITY
    # ============================================
    
    def __getattr__(self, name):
        """Proxy all other methods to original lobby for compatibility"""
        if hasattr(self.original_lobby, name):
            return getattr(self.original_lobby, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


# ============================================
# INTEGRATION UTILITIES
# ============================================

async def enhance_existing_lobby(lobby):
    """Enhance an existing lobby with Data Bus + Traffic Light orchestration"""
    enhanced = EnhancedAgentLobby(lobby)
    await enhanced.initialize()
    
    logger.info("🎯 Lobby enhancement complete")
    logger.info("   ✅ Data Bus orchestration enabled")
    logger.info("   ✅ Traffic Light stage management enabled")
    logger.info("   ✅ Multi-agent collaboration fixed")
    logger.info("   ✅ Backward compatibility maintained")
    
    return enhanced

def create_workflow_dashboard_api():
    """Create REST API endpoints for workflow monitoring"""
    # This would create FastAPI endpoints for monitoring
    # Keeping it simple for now, but could be extended
    return {
        "endpoints": [
            "GET /workflows/{workflow_id}/status",
            "GET /workflows/patterns",
            "POST /workflows/create",
            "GET /system/dashboard",
            "GET /traffic-lights/status"
        ]
    } 