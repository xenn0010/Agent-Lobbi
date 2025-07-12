#!/usr/bin/env python3
"""
Command Line Interface for Agent Lobbi SDK

This module provides a CLI for common Agent Lobbi operations.
"""

import asyncio
import click
import json
import logging
from typing import List, Optional

from .client import AgentLobbiClient, create_agent, Capability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--api-key', required=True, help='Agent Lobbi API key')
@click.option('--lobby-url', default='http://localhost:8092', help='Agent Lobbi URL')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, api_key, lobby_url, debug):
    """Agent Lobbi SDK Command Line Interface"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['api_key'] = api_key
    ctx.obj['lobby_url'] = lobby_url

@cli.command()
@click.pass_context
async def health(ctx):
    """Check Agent Lobbi health status"""
    api_key = ctx.obj['api_key']
    lobby_url = ctx.obj['lobby_url']
    
    try:
        async with AgentLobbiClient(api_key, lobby_url) as client:
            result = await client.health_check()
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.pass_context
async def list_agents(ctx):
    """List all registered agents"""
    api_key = ctx.obj['api_key']
    lobby_url = ctx.obj['lobby_url']
    
    try:
        async with AgentLobbiClient(api_key, lobby_url) as client:
            agents = await client.list_agents()
            click.echo(json.dumps(agents, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option('--task-name', required=True, help='Name of the task')
@click.option('--task-description', required=True, help='Description of the task')
@click.option('--capabilities', required=True, help='Required capabilities (comma-separated)')
@click.option('--task-data', help='Task data as JSON string')
@click.option('--max-agents', default=1, help='Maximum number of agents')
@click.option('--timeout', default=30, help='Timeout in minutes')
@click.pass_context
async def delegate_task(ctx, task_name, task_description, capabilities, task_data, max_agents, timeout):
    """Delegate a task to available agents"""
    api_key = ctx.obj['api_key']
    lobby_url = ctx.obj['lobby_url']
    
    try:
        # Parse capabilities
        capability_list = [cap.strip() for cap in capabilities.split(',')]
        
        # Parse task data
        task_data_dict = {}
        if task_data:
            task_data_dict = json.loads(task_data)
        
        async with AgentLobbiClient(api_key, lobby_url) as client:
            result = await client.delegate_task(
                task_name=task_name,
                task_description=task_description,
                required_capabilities=capability_list,
                task_data=task_data_dict,
                max_agents=max_agents,
                timeout_minutes=timeout
            )
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option('--task-id', required=True, help='Task ID to check')
@click.pass_context
async def task_status(ctx, task_id):
    """Get status of a specific task"""
    api_key = ctx.obj['api_key']
    lobby_url = ctx.obj['lobby_url']
    
    try:
        async with AgentLobbiClient(api_key, lobby_url) as client:
            result = await client.get_task_status(task_id)
            click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

@cli.command()
@click.option('--agent-type', required=True, help='Type of agent')
@click.option('--capabilities', required=True, help='Agent capabilities (comma-separated)')
@click.option('--agent-id', help='Custom agent ID')
@click.pass_context
async def run_agent(ctx, agent_type, capabilities, agent_id):
    """Run a basic agent with specified capabilities"""
    api_key = ctx.obj['api_key']
    lobby_url = ctx.obj['lobby_url']
    
    try:
        # Parse capabilities
        capability_list = [cap.strip() for cap in capabilities.split(',')]
        
        # Create agent
        agent = await create_agent(
            api_key=api_key,
            agent_type=agent_type,
            capabilities=capability_list,
            agent_id=agent_id,
            lobby_url=lobby_url,
            debug=True
        )
        
        # Simple message handler
        @agent.on_message
        async def handle_message(message):
            click.echo(f"Received message: {message.message_type.name}")
            click.echo(f"Payload: {json.dumps(message.payload, indent=2)}")
            
            # Echo back the payload
            return {
                "success": True,
                "echo": message.payload,
                "message": "Message received and processed"
            }
        
        # Start agent
        click.echo(f"Starting agent: {agent.agent_id}")
        success = await agent.start()
        
        if success:
            click.echo("Agent started successfully! Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                click.echo("Stopping agent...")
        else:
            click.echo("Failed to start agent", err=True)
        
        await agent.stop()
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)

def main():
    """Main entry point for CLI"""
    # Convert async commands to sync
    def make_sync(async_func):
        def sync_func(*args, **kwargs):
            return asyncio.run(async_func(*args, **kwargs))
        return sync_func
    
    # Convert async commands
    health.callback = make_sync(health.callback)
    list_agents.callback = make_sync(list_agents.callback)
    delegate_task.callback = make_sync(delegate_task.callback)
    task_status.callback = make_sync(task_status.callback)
    run_agent.callback = make_sync(run_agent.callback)
    
    cli()

if __name__ == '__main__':
    main() 