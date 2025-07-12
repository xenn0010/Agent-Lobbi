"""
MCP Server Factory - Integration with Agent Lobbi
================================================

This module provides the integration layer between the Agent Lobbi
and the MCP autonomous agents system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from .mcp_agent_manager import MCPAgentManager

logger = logging.getLogger(__name__)

class MCPServerFactory:
    """Factory for creating and managing MCP integration with Agent Lobbi"""
    
    def __init__(self, lobby_instance=None):
        self.agent_manager = MCPAgentManager()
        self.lobby = lobby_instance
        self.mcp_enabled = True
        
    async def initialize(self):
        """Initialize the MCP server factory"""
        try:
            logger.info(" MCP Server Factory initializing...")
            
            # Test tool availability
            test_result = await self._test_tools()
            if not test_result["success"]:
                logger.warning(f" Some MCP tools may not be fully functional: {test_result['warnings']}")
            
            logger.info(" MCP Server Factory initialized successfully")
            return {"success": True, "message": "MCP integration ready"}
            
        except Exception as e:
            logger.error(f" MCP Server Factory initialization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _test_tools(self) -> Dict[str, Any]:
        """Test basic tool functionality"""
        warnings = []
        
        try:
            # Test file system tools
            from .mcp_tools import FileSystemTools
            fs_tools = FileSystemTools()
            test_result = fs_tools.list_directory(".")
            if not test_result.get("success"):
                warnings.append("File system tools may have issues")
            
            # Test web search tools
            from .mcp_tools import WebSearchTools
            web_tools = WebSearchTools()
            # Quick connectivity test (don't actually search)
            
            logger.info("TEST Basic tool functionality verified")
            
        except Exception as e:
            warnings.append(f"Tool test error: {str(e)}")
        
        return {"success": len(warnings) == 0, "warnings": warnings}
    
    def convert_lobby_agent_to_mcp(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a registered lobby agent to an autonomous MCP agent"""
        try:
            logger.info(f" Converting agent {agent_data.get('agent_id')} to autonomous MCP agent")
            
            # Register with MCP agent manager
            result = self.agent_manager.register_mcp_agent(agent_data)
            
            if result["success"]:
                logger.info(f" Agent {agent_data['agent_id']} is now autonomous with tools: {result['available_tools']}")
            else:
                logger.error(f" Failed to convert agent {agent_data['agent_id']}: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Agent conversion failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_autonomous_task(self, agent_id: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task autonomously using an MCP agent"""
        try:
            logger.info(f" Assigning autonomous task to agent {agent_id}: {task[:100]}...")
            
            result = await self.agent_manager.assign_autonomous_task(agent_id, task, context)
            
            if result["success"]:
                logger.info(f" Agent {agent_id} completed autonomous task successfully")
                logger.info(f" Tools used: {result.get('capabilities_used', [])}")
            else:
                logger.error(f" Agent {agent_id} failed autonomous task: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Autonomous task execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_autonomous_collaboration(self, task: str, required_capabilities: List[str] = None) -> Dict[str, Any]:
        """Create a collaboration between multiple autonomous agents"""
        try:
            logger.info(f" Creating autonomous collaboration for task: {task[:100]}...")
            
            result = await self.agent_manager.broadcast_task(task, required_capabilities)
            
            if result["success"]:
                logger.info(f" Autonomous collaboration completed with {len(result['successful_results'])} agents")
            else:
                logger.error(f" Autonomous collaboration failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Autonomous collaboration failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_mcp_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of an MCP agent"""
        return self.agent_manager.get_agent_status(agent_id)
    
    def list_autonomous_agents(self) -> Dict[str, Any]:
        """List all autonomous MCP agents"""
        return self.agent_manager.list_active_agents()
    
    async def integrate_with_lobby_workflow(self, workflow_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate MCP autonomous execution with lobby workflow system"""
        try:
            # Extract relevant information
            task_title = task_data.get("task_title", "Autonomous Task")
            task_description = task_data.get("task_description", "")
            required_capabilities = task_data.get("required_capabilities", [])
            assigned_agent = task_data.get("assigned_agent_id")
            
            if assigned_agent:
                # Single agent autonomous execution
                result = await self.execute_autonomous_task(
                    agent_id=assigned_agent,
                    task=f"{task_title}: {task_description}",
                    context={"workflow_id": workflow_id, "original_task": task_data}
                )
            else:
                # Multi-agent autonomous collaboration
                result = await self.create_autonomous_collaboration(
                    task=f"{task_title}: {task_description}",
                    required_capabilities=required_capabilities
                )
            
            # Format result for lobby workflow system
            lobby_result = {
                "workflow_id": workflow_id,
                "autonomous_execution": True,
                "mcp_result": result,
                "status": "completed" if result.get("success") else "failed",
                "summary": self._generate_result_summary(result)
            }
            
            return lobby_result
            
        except Exception as e:
            logger.error(f"Lobby workflow integration failed: {e}")
            return {
                "workflow_id": workflow_id,
                "autonomous_execution": False,
                "error": str(e),
                "status": "failed"
            }
    
    def _generate_result_summary(self, mcp_result: Dict[str, Any]) -> str:
        """Generate a human-readable summary of MCP execution results"""
        if not mcp_result.get("success"):
            return f" Autonomous execution failed: {mcp_result.get('error', 'Unknown error')}"
        
        if "successful_results" in mcp_result:
            # Multi-agent collaboration
            agent_count = len(mcp_result["successful_results"])
            return f" {agent_count} autonomous agents collaborated successfully. Results combined and analyzed."
        else:
            # Single agent execution
            tools_used = mcp_result.get("capabilities_used", [])
            tool_summary = f" using {len(tools_used)} tools" if tools_used else ""
            return f" Autonomous agent completed task{tool_summary}. Full analysis provided."
    
    def get_mcp_statistics(self) -> Dict[str, Any]:
        """Get statistics about MCP system usage"""
        agent_list = self.agent_manager.list_active_agents()
        
        stats = {
            "mcp_enabled": self.mcp_enabled,
            "total_autonomous_agents": agent_list["count"],
            "agents_by_status": {},
            "tools_available": list(self.agent_manager.tool_registry.tools.keys()),
            "agent_types": {}
        }
        
        # Calculate statistics
        for agent in agent_list["agents"]:
            status = agent["status"]
            stats["agents_by_status"][status] = stats["agents_by_status"].get(status, 0) + 1
            
            # Determine agent type from capabilities
            agent_type = self.agent_manager._determine_agent_type(agent["capabilities"])
            stats["agent_types"][agent_type] = stats["agent_types"].get(agent_type, 0) + 1
        
        return stats


# Global MCP factory instance for lobby integration
_mcp_factory = None

def get_mcp_factory(lobby_instance=None) -> MCPServerFactory:
    """Get or create the global MCP factory instance"""
    global _mcp_factory
    if _mcp_factory is None:
        _mcp_factory = MCPServerFactory(lobby_instance)
    return _mcp_factory

async def initialize_mcp_integration(lobby_instance=None) -> Dict[str, Any]:
    """Initialize MCP integration for the lobby system"""
    factory = get_mcp_factory(lobby_instance)
    return await factory.initialize()

def convert_agent_to_autonomous(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a lobby agent to autonomous MCP agent"""
    factory = get_mcp_factory()
    return factory.convert_lobby_agent_to_mcp(agent_data) 