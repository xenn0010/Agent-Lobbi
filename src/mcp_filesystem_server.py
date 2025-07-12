#!/usr/bin/env python3

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Basic MCP server implementation
class MCPServer:
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).resolve()
        self.tools = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "create_directory": self.create_directory,
            "delete_file": self.delete_file,
            "search_files": self.search_files,
        }
    
    def _validate_path(self, path: str) -> Path:
        """Ensure path is within workspace and resolve it."""
        full_path = (self.workspace_root / path).resolve()
        if not str(full_path).startswith(str(self.workspace_root)):
            raise ValueError(f"Path {path} is outside workspace")
        return full_path
    
    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents."""
        try:
            file_path = self._validate_path(path)
            if not file_path.exists():
                return {"error": f"File {path} does not exist"}
            
            content = file_path.read_text(encoding='utf-8')
            return {
                "content": content,
                "path": str(file_path.relative_to(self.workspace_root)),
                "size": len(content)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to file."""
        try:
            file_path = self._validate_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            return {
                "success": True,
                "path": str(file_path.relative_to(self.workspace_root)),
                "size": len(content)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def list_files(self, path: str = ".", pattern: str = "*") -> Dict[str, Any]:
        """List files in directory."""
        try:
            dir_path = self._validate_path(path)
            if not dir_path.exists():
                return {"error": f"Directory {path} does not exist"}
            
            files = []
            for item in dir_path.glob(pattern):
                rel_path = item.relative_to(self.workspace_root)
                files.append({
                    "name": item.name,
                    "path": str(rel_path),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            
            return {"files": files}
        except Exception as e:
            return {"error": str(e)}
    
    async def create_directory(self, path: str) -> Dict[str, Any]:
        """Create directory."""
        try:
            dir_path = self._validate_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "path": str(dir_path.relative_to(self.workspace_root))
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def delete_file(self, path: str) -> Dict[str, Any]:
        """Delete file or directory."""
        try:
            file_path = self._validate_path(path)
            if not file_path.exists():
                return {"error": f"Path {path} does not exist"}
            
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
            
            return {
                "success": True,
                "path": str(file_path.relative_to(self.workspace_root))
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def search_files(self, query: str, path: str = ".", file_extensions: List[str] = None) -> Dict[str, Any]:
        """Search for text in files."""
        try:
            search_path = self._validate_path(path)
            results = []
            
            if file_extensions is None:
                file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.txt', '.md']
            
            for file_path in search_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in file_extensions:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if query.lower() in content.lower():
                            # Find line numbers
                            lines = content.split('\n')
                            matches = []
                            for i, line in enumerate(lines, 1):
                                if query.lower() in line.lower():
                                    matches.append({
                                        "line": i,
                                        "content": line.strip()
                                    })
                            
                            results.append({
                                "path": str(file_path.relative_to(self.workspace_root)),
                                "matches": matches
                            })
                    except:
                        continue  # Skip files that can't be read
            
            return {"results": results}
        except Exception as e:
            return {"error": str(e)}
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "initialize":
            return {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "filesystem-server",
                    "version": "1.0.0"
                }
            }
        
        elif method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read the contents of a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Path to the file"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Path to the file"},
                                "content": {"type": "string", "description": "Content to write"}
                            },
                            "required": ["path", "content"]
                        }
                    },
                    {
                        "name": "list_files",
                        "description": "List files in a directory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Directory path", "default": "."},
                                "pattern": {"type": "string", "description": "Glob pattern", "default": "*"}
                            }
                        }
                    },
                    {
                        "name": "create_directory",
                        "description": "Create a directory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Directory path"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "delete_file",
                        "description": "Delete a file or directory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Path to delete"}
                            },
                            "required": ["path"]
                        }
                    },
                    {
                        "name": "search_files",
                        "description": "Search for text in files",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                                "path": {"type": "string", "description": "Directory to search", "default": "."},
                                "file_extensions": {"type": "array", "items": {"type": "string"}, "description": "File extensions to search"}
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if tool_name in self.tools:
                result = await self.tools[tool_name](**arguments)
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        
        return {"error": f"Unknown method: {method}"}

async def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else "."
    server = MCPServer(workspace)
    
    # Simple stdio-based MCP server
    while True:
        try:
            line = input()
            if not line:
                break
            
            request = json.loads(line)
            response = await server.handle_request(request)
            
            # Add request ID if present
            if "id" in request:
                response["id"] = request["id"]
            
            print(json.dumps(response))
            sys.stdout.flush()
        
        except EOFError:
            break
        except Exception as e:
            error_response = {"error": str(e)}
            print(json.dumps(error_response))
            sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())