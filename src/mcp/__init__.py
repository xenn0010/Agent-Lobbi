"""
MCP (Model Context Protocol) Integration for Agent Lobbi
========================================================

This module implements Model Context Protocol to turn registered Ollama agents
into real autonomous agents with actual capabilities through standardized tool access.

MCP enables agents to:
- Access file systems
- Query databases
- Make web requests  
- Execute code
- Use external APIs
- Collaborate with other agents

Architecture:
- MCP Servers: Each agent type has specialized tool servers
- MCP Client: Lobby communicates with agent MCP servers
- Tool Registry: Standardized tool access across all agents
"""

from .mcp_agent_manager import MCPAgentManager
from .mcp_server_factory import MCPServerFactory
from .mcp_tools import (
    FileSystemTools,
    WebSearchTools,
    CodeExecutionTools,
    DatabaseTools,
    AnalyticsTools,
    CreativeTools
)

__version__ = "1.0.0"
__all__ = [
    "MCPAgentManager",
    "MCPServerFactory", 
    "FileSystemTools",
    "WebSearchTools",
    "CodeExecutionTools",
    "DatabaseTools",
    "AnalyticsTools",
    "CreativeTools"
] 