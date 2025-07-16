"""
Agent Lobbi CLI
===============
Command-line interface for Agent Lobbi operations.
"""

import argparse
import asyncio
import json
import sys
from typing import List

from .client import Agent, Capability
from .utils import setup_logging, validate_agent_id, validate_capabilities

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Agent Lobbi - Multi-Agent Collaboration Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a simple agent
  agent-lobbi start my_agent --type Assistant --capabilities analysis writing

  # Test connection to lobby
  agent-lobbi test --host localhost --port 8098

  # Show package information
  agent-lobbi info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start an agent')
    start_parser.add_argument('agent_id', help='Agent identifier')
    start_parser.add_argument('--type', default='Assistant', help='Agent type')
    start_parser.add_argument('--capabilities', nargs='+', required=True, help='Agent capabilities')
    start_parser.add_argument('--host', default='localhost', help='Lobby host')
    start_parser.add_argument('--port', type=int, default=8098, help='Lobby port')
    start_parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test connection to lobby')
    test_parser.add_argument('--host', default='localhost', help='Lobby host')
    test_parser.add_argument('--port', type=int, default=8098, help='Lobby port')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show package information')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')
    
    return parser

async def start_agent(agent_id: str, agent_type: str, capabilities: List[str], 
                     host: str, port: int, debug: bool = False):
    """Start an agent with the given parameters"""
    
    if not validate_agent_id(agent_id):
        print(f"Error: Invalid agent ID '{agent_id}'")
        return False
    
    if not validate_capabilities(capabilities):
        print(f"Error: Invalid capabilities list")
        return False
    
    print(f"Starting agent {agent_id} ({agent_type})")
    print(f"Capabilities: {', '.join(capabilities)}")
    print(f"Connecting to {host}:{port}")
    
    # Create agent
    agent = Agent(
        agent_id=agent_id,
        agent_type=agent_type,
        capabilities=capabilities,
        lobby_host=host,
        lobby_port=port,
        debug=debug
    )
    
    # Simple message handler
    @agent.on_message
    async def handle_message(message):
        print(f"Received message: {message.payload}")
        return {"status": "received", "agent": agent_id}
    
    # Start agent
    if await agent.start():
        print(f"Agent {agent_id} started successfully!")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping agent...")
            await agent.stop()
            print("Agent stopped.")
            return True
    else:
        print(f"Failed to start agent {agent_id}")
        return False

async def test_connection(host: str, port: int):
    """Test connection to the lobby"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/api/lobby/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Connected to Agent Lobbi at {host}:{port}")
                    print(f"Status: {data}")
                    return True
                else:
                    print(f"❌ Connection failed: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def show_info():
    """Show package information"""
    from . import __version__, __description__, __url__
    
    print("Agent Lobbi - Multi-Agent Collaboration Platform")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print(f"Homepage: {__url__}")
    print()
    print("Features:")
    print("• Multi-agent collaboration")
    print("• Security and consensus systems")
    print("• Real-time WebSocket communication")
    print("• Automatic recovery and monitoring")
    print("• MCP protocol support")

def show_version():
    """Show version information"""
    from . import __version__
    print(f"agent-lobbi {__version__}")

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set up logging
    setup_logging("DEBUG" if hasattr(args, 'debug') and args.debug else "INFO")
    
    if args.command == 'start':
        success = asyncio.run(start_agent(
            args.agent_id, args.type, args.capabilities,
            args.host, args.port, args.debug
        ))
        sys.exit(0 if success else 1)
        
    elif args.command == 'test':
        success = asyncio.run(test_connection(args.host, args.port))
        sys.exit(0 if success else 1)
        
    elif args.command == 'info':
        show_info()
        
    elif args.command == 'version':
        show_version()

if __name__ == '__main__':
    main() 