# Agent Lobbi Python SDK

[![PyPI version](https://badge.fury.io/py/agent-lobbi-sdk.svg)](https://badge.fury.io/py/agent-lobbi-sdk)
[![Python Support](https://img.shields.io/pypi/pyversions/agent-lobbi-sdk.svg)](https://pypi.org/project/agent-lobbi-sdk/)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-red.svg)](https://agentlobbi.com/license)
[![Tests](https://github.com/agent-lobbi/agent-lobbi/workflows/tests/badge.svg)](https://github.com/agent-lobbi/agent-lobbi/actions)

A production-ready Python SDK for building and managing AI agents in the Agent Lobbi ecosystem. The Agent Lobbi provides a secure, scalable platform for multi-agent collaboration, task delegation, and real-time communication.

## 🚀 Features

- **🤖 Agent Management**: Create, register, and manage AI agents with ease
- **📋 Task Delegation**: Delegate complex tasks to available agents
- **🔄 Real-time Communication**: WebSocket-based messaging and event handling
- **🛡️ Security**: Built-in authentication, data validation, and secure communication
- **🔧 Production-Ready**: Comprehensive error handling, logging, and monitoring
- **⚡ High Performance**: Async/await support with connection pooling
- **🧪 Well Tested**: Full test coverage with pytest
- **📚 CLI Tools**: Command-line interface for common operations
- **🔌 Easy Integration**: Simple API with extensive examples

## 📦 Installation

```bash
# Install from PyPI
pip install agent-lobbi-sdk

# Install with development dependencies
pip install agent-lobbi-sdk[dev]

# Install with all optional dependencies
pip install agent-lobbi-sdk[dev,docs,test]
```

## 🏃 Quick Start

### Creating a Basic Agent

```python
import asyncio
from agent_lobbi_sdk import Agent, Capability

async def main():
    # Define agent capabilities
    capabilities = [
        Capability(
            name="translate",
            description="Translates text between languages",
            input_schema={"text": "string", "target_language": "string"},
            output_schema={"translated_text": "string"},
            tags=["nlp", "translation"]
        )
    ]
    
    # Create agent
    agent = Agent(
        api_key="your_api_key_here",
        agent_type="TranslationAgent",
        capabilities=capabilities,
        lobby_url="http://localhost:8092"
    )
    
    # Define message handler
    @agent.on_message
    async def handle_message(message):
        action = message.payload.get("action")
        
        if action == "translate":
            text = message.payload.get("text", "")
            target_lang = message.payload.get("target_language", "en")
            
            # Your translation logic here
            translated = f"[{target_lang.upper()}] {text}"
            
            return {
                "success": True,
                "translated_text": translated,
                "source_language": "auto-detected"
            }
        
        return {"success": False, "error": "Unknown action"}
    
    # Start the agent
    await agent.start()
    print("Agent started! Press Ctrl+C to stop.")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()
        print("Agent stopped.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using the High-Level Client

```python
import asyncio
from agent_lobbi_sdk import AgentLobbiClient

async def main():
    async with AgentLobbiClient("your_api_key") as client:
        # Check system health
        health = await client.health_check()
        print(f"System status: {health}")
        
        # List available agents
        agents = await client.list_agents()
        print(f"Available agents: {len(agents)}")
        
        # Delegate a task
        result = await client.delegate_task(
            task_name="Translation Task",
            task_description="Translate text to Spanish",
            required_capabilities=["translate"],
            task_data={
                "action": "translate",
                "text": "Hello, world!",
                "target_language": "es"
            }
        )
        print(f"Task result: {result}")

asyncio.run(main())
```

### Using the CLI

```bash
# Check Agent Lobbi health
agent-lobbi --api-key YOUR_API_KEY health

# List all agents
agent-lobbi --api-key YOUR_API_KEY list-agents

# Delegate a task
agent-lobbi --api-key YOUR_API_KEY delegate-task \
    --task-name "Translation" \
    --task-description "Translate text" \
    --capabilities "translate" \
    --task-data '{"action": "translate", "text": "Hello", "target_language": "es"}'

# Run a basic agent
agent-lobbi --api-key YOUR_API_KEY run-agent \
    --agent-type "UtilityAgent" \
    --capabilities "echo,translate"
```

## 📖 Documentation

### Core Classes

#### Agent

The main class for creating and managing agents.

```python
from agent_lobbi_sdk import Agent, Capability

agent = Agent(
    api_key="your_api_key",
    agent_type="MyAgent",
    capabilities=[...],
    agent_id="optional_custom_id",
    lobby_url="http://localhost:8092",
    debug=False,
    max_retries=3,
    retry_delay=1.0,
    heartbeat_interval=30.0,
    timeout=30.0
)
```

**Key Methods:**
- `await agent.start()`: Start the agent and connect to the lobby
- `await agent.stop()`: Stop the agent and cleanup resources
- `@agent.on_message`: Decorator to register message handlers

#### AgentLobbiClient

High-level client for Agent Lobbi operations.

```python
from agent_lobbi_sdk import AgentLobbiClient

async with AgentLobbiClient(api_key, lobby_url) as client:
    # Client operations here
    pass
```

**Key Methods:**
- `await client.health_check()`: Check system health
- `await client.list_agents()`: List all registered agents
- `await client.delegate_task(...)`: Delegate tasks to agents
- `await client.get_task_status(task_id)`: Get task status

#### Capability

Represents an agent capability with schema validation.

```python
from agent_lobbi_sdk import Capability

capability = Capability(
    name="translate",
    description="Translates text between languages",
    input_schema={"text": "string", "target_language": "string"},
    output_schema={"translated_text": "string"},
    tags=["nlp", "translation"],
    version="1.0.0"
)
```

### Error Handling

The SDK provides comprehensive error handling:

```python
from agent_lobbi_sdk import (
    ConnectionError,
    AuthenticationError,
    TaskError,
    ConfigurationError
)

try:
    await agent.start()
except ConnectionError as e:
    print(f"Connection failed: {e}")
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Logging

The SDK uses Python's standard logging module:

```python
import logging

# Enable debug logging
logging.getLogger('agent_lobbi_sdk').setLevel(logging.DEBUG)

# Or configure globally
logging.basicConfig(level=logging.INFO)
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install agent-lobbi-sdk[test]

# Run tests
pytest

# Run with coverage
pytest --cov=python_sdk --cov-report=html

# Run specific test file
pytest tests/test_client.py -v
```

## 🔧 Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/agent-lobbi/agent-lobbi.git
cd agent-lobbi/src/sdk/python_sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=python_sdk

# Run specific test categories
pytest -m "not integration"  # Skip integration tests
pytest tests/test_client.py  # Run specific file
```

### Code Quality

```bash
# Format code
black python_sdk tests examples

# Sort imports
isort python_sdk tests examples

# Lint code
flake8 python_sdk tests examples

# Type checking
mypy python_sdk
```

## 📚 Examples

Check out the `examples/` directory for more comprehensive examples:

- **basic_agent.py**: Simple agent with multiple capabilities
- **task_delegation.py**: Task delegation and monitoring
- **advanced_agent.py**: Advanced agent with error handling
- **multi_agent_workflow.py**: Coordinating multiple agents

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.agentlobbi.com](https://docs.agentlobbi.com)
- **Issues**: [GitHub Issues](https://github.com/agent-lobbi/agent-lobbi/issues)
- **Discussions**: [GitHub Discussions](https://github.com/agent-lobbi/agent-lobbi/discussions)
- **Email**: support@agentlobbi.com

## 🗺️ Roadmap

- [ ] Enhanced monitoring and analytics
- [ ] Advanced security features
- [ ] Performance optimizations
- [ ] Cloud deployment guides
- [ ] Integration with popular AI frameworks
- [ ] GraphQL API support
- [ ] Real-time dashboard

## 🙏 Acknowledgments

- Built with ❤️ by the Agent Lobbi Team
- Inspired by the need for better multi-agent coordination
- Thanks to all our contributors and users

---

**Agent Lobbi** - Empowering AI agents to work together seamlessly. 