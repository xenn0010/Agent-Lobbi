"""
MCP Agent Manager - Autonomous Agent Orchestration
=================================================

This module transforms registered Ollama models into autonomous agents
with real capabilities through MCP (Model Context Protocol) integration.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests

from .mcp_tools import MCPToolRegistry

logger = logging.getLogger(__name__)

class MCPAgent:
    """A real autonomous agent powered by Ollama + MCP tools"""
    
    def __init__(self, agent_id: str, name: str, model: str, capabilities: List[str], tools: Dict[str, Any]):
        self.agent_id = agent_id
        self.name = name
        self.model = model
        self.capabilities = capabilities
        self.tools = tools
        self.ollama_url = "http://localhost:11434"
        self.conversation_history = []
        self.status = "idle"
        self.current_task = None
        
    async def think_and_act(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main autonomous agent loop - think, plan, and execute"""
        try:
            self.status = "thinking"
            self.current_task = task
            
            # Build the agent's prompt with available tools
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(task, context)
            
            # Get initial response from Ollama
            response = await self._call_ollama(system_prompt, user_prompt)
            
            if not response.get("success"):
                return {"success": False, "error": "Failed to get response from Ollama"}
            
            agent_response = response["content"]
            
            # Parse for tool usage
            tool_usage = self._parse_tool_usage(agent_response)
            
            results = []
            if tool_usage:
                # Execute tools
                self.status = "executing"
                for tool_call in tool_usage:
                    tool_result = await self._execute_tool(tool_call)
                    results.append(tool_result)
                    
                    # Give results back to agent for further processing
                    if tool_result.get("success"):
                        follow_up_prompt = f"Tool {tool_call['tool']}.{tool_call['method']} executed successfully. Result: {json.dumps(tool_result)}\n\nPlease analyze this result and continue with your task."
                        follow_up = await self._call_ollama("Continue with your analysis.", follow_up_prompt)
                        if follow_up.get("success"):
                            agent_response += f"\n\n**Analysis of {tool_call['tool']} result:**\n{follow_up['content']}"
            
            self.status = "idle"
            self.current_task = None
            
            return {
                "success": True,
                "agent_id": self.agent_id,
                "task": task,
                "response": agent_response,
                "tool_results": results,
                "capabilities_used": [tool["tool"] for tool in tool_usage]
            }
            
        except Exception as e:
            self.status = "error"
            logger.error(f"Agent {self.agent_id} failed task: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with agent's identity and available tools"""
        available_tools = list(self.tools.keys())
        
        return f"""You are {self.name} (ID: {self.agent_id}), an autonomous AI agent powered by the {self.model} model.

Your capabilities: {', '.join(self.capabilities)}

Available Tools:
{self._format_tools_description()}

When you need to use a tool, format your request like this:
TOOL_CALL: tool_name.method_name(param1="value1", param2="value2")

For example:
- To search the web: TOOL_CALL: web_search.web_search(query="AI trends 2024")
- To read a file: TOOL_CALL: file_system.read_file(file_path="data.txt")
- To execute code: TOOL_CALL: code_execution.execute_python(code="print('Hello World')")

You are autonomous - make decisions, use tools when needed, and provide comprehensive responses.
Always think step by step and explain your reasoning."""

    def _format_tools_description(self) -> str:
        """Format available tools for the agent"""
        descriptions = {
            "file_system": "Read/write files, list directories",
            "web_search": "Search the web, make HTTP requests",
            "code_execution": "Execute Python code, validate syntax",
            "database": "Store and query data",
            "analytics": "Analyze text, generate summaries",
            "creative": "Generate content outlines, format text"
        }
        
        tool_list = []
        for tool_name in self.tools.keys():
            desc = descriptions.get(tool_name, "Tool functionality")
            tool_list.append(f"  - {tool_name}: {desc}")
        
        return "\n".join(tool_list)
    
    def _build_user_prompt(self, task: str, context: Dict[str, Any] = None) -> str:
        """Build user prompt with task and context"""
        prompt = f"Task: {task}\n"
        
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}\n"
        
        prompt += f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        prompt += "\n\nPlease complete this task using your capabilities and available tools. Be autonomous and thorough."
        
        return prompt
    
    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call Ollama API to get agent's response"""
        try:
            payload = {
                "model": self.model,
                "prompt": f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}\n\nASSISTANT:",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return {"success": True, "content": data.get("response", "")}
            else:
                return {"success": False, "error": f"Ollama returned status {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": f"Ollama call failed: {str(e)}"}
    
    def _parse_tool_usage(self, response: str) -> List[Dict[str, Any]]:
        """Parse agent response for tool usage requests"""
        tool_calls = []
        lines = response.split('\n')
        
        for line in lines:
            if "TOOL_CALL:" in line:
                try:
                    tool_call_str = line.split("TOOL_CALL:")[1].strip()
                    # Parse tool_name.method_name(params)
                    if '(' in tool_call_str:
                        tool_method, params_str = tool_call_str.split('(', 1)
                        params_str = params_str.rstrip(')')
                        
                        tool_name, method_name = tool_method.split('.', 1)
                        
                        # Simple parameter parsing (for demo - would need more robust parsing)
                        params = {}
                        if params_str:
                            # Basic parameter parsing - handles simple cases
                            for param in params_str.split(','):
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    key = key.strip()
                                    value = value.strip().strip('"\'')
                                    params[key] = value
                        
                        tool_calls.append({
                            "tool": tool_name.strip(),
                            "method": method_name.strip(),
                            "params": params
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse tool call: {line} - {e}")
        
        return tool_calls
    
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        try:
            tool_name = tool_call["tool"]
            method_name = tool_call["method"]
            params = tool_call["params"]
            
            if tool_name not in self.tools:
                return {"success": False, "error": f"Tool {tool_name} not available"}
            
            tool = self.tools[tool_name]
            
            if not hasattr(tool, method_name):
                return {"success": False, "error": f"Method {method_name} not found"}
            
            method = getattr(tool, method_name)
            result = method(**params)
            
            logger.info(f"Agent {self.agent_id} executed {tool_name}.{method_name}")
            return {
                "success": True,
                "tool": tool_name,
                "method": method_name,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}


class MCPAgentManager:
    """Manages autonomous MCP agents for the lobby system"""
    
    def __init__(self):
        self.tool_registry = MCPToolRegistry()
        self.active_agents: Dict[str, MCPAgent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        
    def register_mcp_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a registered Ollama agent into an autonomous MCP agent"""
        try:
            agent_id = agent_data["agent_id"]
            name = agent_data.get("name", f"Agent {agent_id}")
            model = agent_data.get("ollama_model", "llama3.2")
            capabilities = agent_data.get("capabilities", [])
            
            # Determine agent type for tool access
            agent_type = self._determine_agent_type(capabilities)
            
            # Get tools for this agent type
            tools = self.tool_registry.get_tools_for_agent(agent_type)
            
            # Create MCP agent
            mcp_agent = MCPAgent(
                agent_id=agent_id,
                name=name,
                model=model,
                capabilities=capabilities,
                tools=tools
            )
            
            self.active_agents[agent_id] = mcp_agent
            
            logger.info(f"Created autonomous MCP agent: {agent_id} with tools: {list(tools.keys())}")
            
            return {
                "success": True,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "available_tools": list(tools.keys()),
                "capabilities": capabilities
            }
            
        except Exception as e:
            logger.error(f"Failed to register MCP agent: {e}")
            return {"success": False, "error": str(e)}
    
    def _determine_agent_type(self, capabilities: List[str]) -> str:
        """Determine agent type based on capabilities"""
        if any(cap in capabilities for cap in ["data_analysis", "research", "reporting"]):
            return "analyst_agent"
        elif any(cap in capabilities for cap in ["creative_writing", "storytelling", "content_creation"]):
            return "creative_agent"
        elif any(cap in capabilities for cap in ["code_analysis", "technical_review", "problem_solving"]):
            return "tech_agent"
        else:
            return "general_agent"
    
    async def assign_autonomous_task(self, agent_id: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assign a task to an autonomous agent"""
        try:
            if agent_id not in self.active_agents:
                return {"success": False, "error": f"Agent {agent_id} not found"}
            
            agent = self.active_agents[agent_id]
            
            if agent.status != "idle":
                return {"success": False, "error": f"Agent {agent_id} is busy (status: {agent.status})"}
            
            # Start autonomous task execution
            task_coroutine = agent.think_and_act(task, context)
            task_future = asyncio.create_task(task_coroutine)
            
            self.agent_tasks[agent_id] = task_future
            
            # Wait for completion (or could return immediately for async operation)
            result = await task_future
            
            # Clean up task reference
            if agent_id in self.agent_tasks:
                del self.agent_tasks[agent_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to assign task to agent {agent_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current status of an agent"""
        if agent_id not in self.active_agents:
            return {"error": "Agent not found"}
        
        agent = self.active_agents[agent_id]
        
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "model": agent.model,
            "status": agent.status,
            "current_task": agent.current_task,
            "capabilities": agent.capabilities,
            "available_tools": list(agent.tools.keys())
        }
    
    def list_active_agents(self) -> Dict[str, Any]:
        """List all active MCP agents"""
        agents = []
        for agent_id, agent in self.active_agents.items():
            agents.append({
                "agent_id": agent_id,
                "name": agent.name,
                "model": agent.model,
                "status": agent.status,
                "capabilities": agent.capabilities,
                "tool_count": len(agent.tools)
            })
        
        return {"agents": agents, "count": len(agents)}
    
    async def broadcast_task(self, task: str, required_capabilities: List[str] = None) -> Dict[str, Any]:
        """Broadcast a task to multiple capable agents for collaborative execution"""
        try:
            suitable_agents = []
            
            for agent_id, agent in self.active_agents.items():
                if agent.status == "idle":
                    if not required_capabilities or any(cap in agent.capabilities for cap in required_capabilities):
                        suitable_agents.append(agent_id)
            
            if not suitable_agents:
                return {"success": False, "error": "No suitable agents available"}
            
            # Assign task to all suitable agents
            tasks = []
            for agent_id in suitable_agents[:3]:  # Limit to 3 agents for now
                task_coroutine = self.assign_autonomous_task(agent_id, task)
                tasks.append(task_coroutine)
            
            # Wait for all agents to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Agent {suitable_agents[i]}: {str(result)}")
                elif result.get("success"):
                    successful_results.append(result)
                else:
                    errors.append(f"Agent {suitable_agents[i]}: {result.get('error', 'Unknown error')}")
            
            return {
                "success": True,
                "task": task,
                "participating_agents": suitable_agents,
                "successful_results": successful_results,
                "errors": errors,
                "collaboration_summary": f"{len(successful_results)} agents completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Broadcast task failed: {e}")
            return {"success": False, "error": str(e)} 