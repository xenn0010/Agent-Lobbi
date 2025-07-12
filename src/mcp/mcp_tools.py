"""
MCP Tools - Real Capabilities for Autonomous Agents
===================================================

This module implements standardized tools that turn Ollama agents into 
real autonomous agents with actual capabilities beyond just text generation.
"""

import os
import json
import subprocess
import requests
import sqlite3
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FileSystemTools:
    """File system access tools for agents"""
    
    def __init__(self, allowed_paths: List[str] = None):
        self.allowed_paths = allowed_paths or [os.getcwd()]
        
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read a file and return its contents"""
        try:
            if not self._is_path_allowed(file_path):
                return {"error": "Access denied to this path"}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"success": True, "content": content, "file_path": file_path}
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file"""
        try:
            if not self._is_path_allowed(file_path):
                return {"error": "Access denied to this path"}
                
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"success": True, "message": f"File written to {file_path}"}
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
    
    def list_directory(self, dir_path: str) -> Dict[str, Any]:
        """List contents of a directory"""
        try:
            if not self._is_path_allowed(dir_path):
                return {"error": "Access denied to this path"}
                
            items = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                items.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
            return {"success": True, "items": items, "directory": dir_path}
        except Exception as e:
            return {"error": f"Failed to list directory: {str(e)}"}
    
    def _is_path_allowed(self, file_path: str) -> bool:
        """Check if file path is within allowed directories"""
        abs_path = os.path.abspath(file_path)
        return any(abs_path.startswith(os.path.abspath(allowed)) for allowed in self.allowed_paths)


class WebSearchTools:
    """Web search and HTTP request tools for agents"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        
    def web_search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Search the web for information"""
        try:
            # Using DuckDuckGo Instant Answer API for free web search
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(search_url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract results from DuckDuckGo response
                if data.get('Abstract'):
                    results.append({
                        "title": data.get('Heading', 'Search Result'),
                        "snippet": data.get('Abstract'),
                        "url": data.get('AbstractURL', '')
                    })
                
                # Add related topics
                for topic in data.get('RelatedTopics', [])[:num_results-1]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            "title": topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related',
                            "snippet": topic.get('Text', ''),
                            "url": topic.get('FirstURL', '')
                        })
                
                return {"success": True, "query": query, "results": results}
            else:
                return {"error": f"Search failed with status {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}
    
    def http_request(self, url: str, method: str = "GET", data: Dict = None, headers: Dict = None) -> Dict[str, Any]:
        """Make HTTP requests"""
        try:
            response = requests.request(
                method=method,
                url=url,
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:5000]  # Limit content size
            }
        except Exception as e:
            return {"error": f"HTTP request failed: {str(e)}"}


class CodeExecutionTools:
    """Code execution tools for technical agents"""
    
    def __init__(self, allowed_languages: List[str] = None):
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
    
    def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            if "python" not in self.allowed_languages:
                return {"error": "Python execution not allowed"}
            
            # Create a temporary file for execution
            temp_file = f"temp_agent_code_{os.getpid()}.py"
            
            with open(temp_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up
            os.remove(temp_file)
            
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timed out"}
        except Exception as e:
            return {"error": f"Code execution failed: {str(e)}"}
    
    def validate_code(self, code: str, language: str) -> Dict[str, Any]:
        """Validate code syntax without executing"""
        try:
            if language == "python":
                compile(code, '<string>', 'exec')
                return {"success": True, "message": "Python code is syntactically valid"}
            else:
                return {"success": True, "message": f"Syntax validation for {language} not implemented"}
        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}


class DatabaseTools:
    """Database access tools for data analysis agents"""
    
    def __init__(self, db_path: str = "agent_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT,
                        data_type TEXT,
                        data_value TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def store_data(self, agent_id: str, data_type: str, data_value: str) -> Dict[str, Any]:
        """Store data in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO agent_data (agent_id, data_type, data_value) VALUES (?, ?, ?)",
                    (agent_id, data_type, data_value)
                )
                conn.commit()
            return {"success": True, "message": "Data stored successfully"}
        except Exception as e:
            return {"error": f"Failed to store data: {str(e)}"}
    
    def query_data(self, agent_id: str = None, data_type: str = None) -> Dict[str, Any]:
        """Query data from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM agent_data WHERE 1=1"
                params = []
                
                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)
                
                if data_type:
                    query += " AND data_type = ?"
                    params.append(data_type)
                
                query += " ORDER BY timestamp DESC LIMIT 100"
                
                cursor = conn.execute(query, params)
                rows = [dict(row) for row in cursor.fetchall()]
                
            return {"success": True, "data": rows}
        except Exception as e:
            return {"error": f"Failed to query data: {str(e)}"}


class AnalyticsTools:
    """Analytics and data processing tools for analyst agents"""
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for basic metrics"""
        try:
            words = text.split()
            sentences = text.split('.')
            
            analysis = {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_words_per_sentence": len(words) / max(len(sentences), 1),
                "unique_words": len(set(word.lower() for word in words)),
                "readability_score": min(100, max(0, 206.835 - 1.015 * (len(words) / max(len(sentences), 1)) - 84.6 * (sum(len(word) for word in words) / len(words))))
            }
            
            return {"success": True, "analysis": analysis}
        except Exception as e:
            return {"error": f"Text analysis failed: {str(e)}"}
    
    def generate_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from data"""
        try:
            if not data:
                return {"error": "No data provided"}
            
            summary = {
                "total_records": len(data),
                "fields": list(data[0].keys()) if data else [],
                "sample_record": data[0] if data else None
            }
            
            return {"success": True, "summary": summary}
        except Exception as e:
            return {"error": f"Summary generation failed: {str(e)}"}


class CreativeTools:
    """Creative and content generation tools for creative agents"""
    
    def generate_content_outline(self, topic: str, content_type: str = "article") -> Dict[str, Any]:
        """Generate content outline structure"""
        try:
            outlines = {
                "article": [
                    f"Introduction to {topic}",
                    f"Key aspects of {topic}",
                    f"Analysis and insights about {topic}",
                    f"Practical applications of {topic}",
                    f"Conclusion and future outlook"
                ],
                "story": [
                    f"Setting introduction involving {topic}",
                    f"Character development around {topic}",
                    f"Conflict or challenge related to {topic}",
                    f"Resolution and growth through {topic}"
                ],
                "report": [
                    f"Executive summary of {topic}",
                    f"Background and context of {topic}",
                    f"Detailed analysis of {topic}",
                    f"Recommendations based on {topic}",
                    f"Appendices and references"
                ]
            }
            
            outline = outlines.get(content_type, outlines["article"])
            
            return {
                "success": True,
                "topic": topic,
                "content_type": content_type,
                "outline": outline
            }
        except Exception as e:
            return {"error": f"Outline generation failed: {str(e)}"}
    
    def format_content(self, content: str, format_type: str = "markdown") -> Dict[str, Any]:
        """Format content in specified format"""
        try:
            if format_type == "markdown":
                # Basic markdown formatting
                formatted = content.replace('\n\n', '\n\n---\n\n')
                formatted = f"# Content\n\n{formatted}\n\n*Generated by Creative Agent*"
            elif format_type == "html":
                # Basic HTML formatting
                formatted = f"<html><body><h1>Content</h1><p>{content.replace(chr(10), '</p><p>')}</p></body></html>"
            else:
                formatted = content
            
            return {
                "success": True,
                "original_content": content,
                "formatted_content": formatted,
                "format_type": format_type
            }
        except Exception as e:
            return {"error": f"Content formatting failed: {str(e)}"}


class MCPToolRegistry:
    """Registry for all MCP tools - manages which tools are available to which agents"""
    
    def __init__(self):
        self.tools = {
            "file_system": FileSystemTools(),
            "web_search": WebSearchTools(),
            "code_execution": CodeExecutionTools(),
            "database": DatabaseTools(),
            "analytics": AnalyticsTools(),
            "creative": CreativeTools()
        }
        
        # Define which tools each agent type can access
        self.agent_tool_mapping = {
            "analyst_agent": ["file_system", "web_search", "database", "analytics"],
            "creative_agent": ["file_system", "web_search", "creative", "analytics"],
            "tech_agent": ["file_system", "web_search", "code_execution", "database"],
            "general_agent": ["file_system", "web_search", "analytics"]
        }
    
    def get_tools_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get available tools for a specific agent type"""
        agent_tools = self.agent_tool_mapping.get(agent_type, ["file_system", "web_search"])
        return {tool_name: self.tools[tool_name] for tool_name in agent_tools if tool_name in self.tools}
    
    def execute_tool(self, agent_type: str, tool_name: str, method: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool method for an agent"""
        try:
            available_tools = self.get_tools_for_agent(agent_type)
            
            if tool_name not in available_tools:
                return {"error": f"Tool '{tool_name}' not available for agent type '{agent_type}'"}
            
            tool = available_tools[tool_name]
            
            if not hasattr(tool, method):
                return {"error": f"Method '{method}' not found in tool '{tool_name}'"}
            
            method_func = getattr(tool, method)
            result = method_func(**kwargs)
            
            logger.info(f"Agent {agent_type} executed {tool_name}.{method}")
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"error": f"Tool execution failed: {str(e)}"} 