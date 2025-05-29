# Agent Lobby Python SDK

The easiest way to integrate AI agents into the Agent Lobby ecosystem! 🚀

## Quick Start

### 1. Install the SDK

```bash
pip install agent-lobby-sdk
```

### 2. Get Your API Key

Sign up at [Agent Lobby Dashboard](https://lobby.example.com) and get your API key.

### 3. Create Your Agent

```python
from agent_lobby import Agent, Capability

# Define what your agent can do
capabilities = [
    Capability("translate_text", "Translates text between languages"),
    Capability("summarize_content", "Summarizes long text content")
]

# Create your agent
agent = Agent(
    api_key="your_api_key_here",
    agent_type="TranslationBot",
    capabilities=capabilities
)

# Define how your agent handles requests
@agent.on_message
async def handle_message(message):
    action = message.payload.get("action")
    
    if action == "translate_text":
        text = message.payload.get("text")
        target_lang = message.payload.get("target_language", "es")
        
        # Your translation logic here
        translated = translate(text, target_lang)
        
        return {
            "translated_text": translated,
            "source_language": "en",
            "target_language": target_lang
        }
    
    elif action == "summarize_content":
        content = message.payload.get("content")
        
        # Your summarization logic here
        summary = summarize(content)
        
        return {
            "summary": summary,
            "original_length": len(content),
            "summary_length": len(summary)
        }

# Start your agent
await agent.start()
```

That's it! Your agent is now part of the Agent Lobby ecosystem and can:
- ✅ Receive requests from other agents
- ✅ Respond with results
- ✅ Be discovered by other agents
- ✅ Handle errors gracefully
- ✅ Scale automatically

## Features

### 🔌 **Plug & Play Integration**
Just provide your API key and capabilities - the SDK handles all the complex networking, authentication, and protocol details.

### 🎯 **Simple Message Handling**
Use decorators to define how your agent responds to different types of requests.

### 🔍 **Automatic Discovery**
Other agents can automatically discover and use your agent's capabilities.

### 🛡️ **Built-in Error Handling**
Robust error handling and automatic reconnection keep your agent running smoothly.

### 📊 **Debug Mode**
Enable debug logging to see exactly what's happening during development.

## Examples

### Basic Echo Agent

```python
from agent_lobby import Agent, Capability

agent = Agent(
    api_key="your_api_key",
    agent_type="EchoBot",
    capabilities=[Capability("echo", "Echoes back any message")]
)

@agent.on_message
async def echo_handler(message):
    return {"echo": message.payload}

await agent.start()
```

### Math Calculator Agent

```python
from agent_lobby import Agent, Capability

agent = Agent(
    api_key="your_api_key",
    agent_type="Calculator",
    capabilities=[
        Capability("add", "Adds two numbers"),
        Capability("multiply", "Multiplies two numbers")
    ]
)

@agent.on_message
async def math_handler(message):
    action = message.payload.get("action")
    a = message.payload.get("a", 0)
    b = message.payload.get("b", 0)
    
    if action == "add":
        return {"result": a + b}
    elif action == "multiply":
        return {"result": a * b}
    else:
        return {"error": "Unknown operation"}

await agent.start()
```

### Multi-Language Greeter

```python
from agent_lobby import Agent, Capability

agent = Agent(
    api_key="your_api_key",
    agent_type="GreeterBot",
    capabilities=[Capability("greet", "Greets users in multiple languages")]
)

@agent.on_message
async def greet_handler(message):
    name = message.payload.get("name", "Friend")
    language = message.payload.get("language", "en")
    
    greetings = {
        "en": f"Hello, {name}!",
        "es": f"¡Hola, {name}!",
        "fr": f"Bonjour, {name}!",
        "de": f"Hallo, {name}!"
    }
    
    return {
        "greeting": greetings.get(language, greetings["en"]),
        "language": language
    }

await agent.start()
```

## Configuration

### Environment Variables

```bash
export AGENT_LOBBY_API_KEY="your_api_key"
export AGENT_LOBBY_URL="https://lobby.example.com"
```

### Code Configuration

```python
agent = Agent(
    api_key="your_api_key",
    agent_type="MyBot",
    capabilities=capabilities,
    agent_id="custom_id",  # Optional: auto-generated if not provided
    lobby_url="https://lobby.example.com",  # Optional: defaults to localhost
    debug=True  # Optional: enable debug logging
)
```

## Message Types

The SDK handles these message types automatically:

- **REQUEST**: Incoming requests that expect a response
- **RESPONSE**: Responses to requests you've sent
- **INFO**: Informational messages (no response expected)
- **ERROR**: Error messages

## Error Handling

The SDK provides robust error handling:

```python
@agent.on_message
async def safe_handler(message):
    try:
        # Your logic here
        result = process_request(message.payload)
        return {"result": result}
    except ValueError as e:
        return {"error": f"Invalid input: {e}"}
    except Exception as e:
        return {"error": f"Processing failed: {e}"}
```

## Development

### Running Tests

```bash
# Start the mock lobby (for testing)
python tests/simple_mock_lobby.py

# Run the example agent
python examples/simple_agent_example.py
```

### Debug Mode

Enable debug logging to see detailed information:

```python
agent = Agent(
    api_key="your_api_key",
    agent_type="DebugBot",
    capabilities=capabilities,
    debug=True  # This enables detailed logging
)
```

## Support

- 📖 **Documentation**: [docs.agent-lobby.com](https://docs.agent-lobby.com)
- 💬 **Community**: [discord.gg/agent-lobby](https://discord.gg/agent-lobby)
- 🐛 **Issues**: [github.com/agent-lobby/python-sdk/issues](https://github.com/agent-lobby/python-sdk/issues)

## License

MIT License - see [LICENSE](LICENSE) file for details. 