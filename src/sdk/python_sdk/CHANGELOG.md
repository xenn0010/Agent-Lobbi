# Changelog

All notable changes to this project will be documented in this file.

## [1.0.2] - 2025-07-06

### Fixed
- Corrected the client-side registration logic to properly handle successful (`200 OK`) responses from the server, preventing erroneous "Registration failed" log messages.

## [1.0.1] - 2025-07-06

### Fixed
- **Critical:** Corrected the agent registration endpoint from `/api/register` to the correct `/api/agents/register`. This resolves the `404 Not Found` error during agent registration, allowing agents to connect to the Lobby successfully.

## [1.0.0] - 2025-07-05

### Added
- Initial release of the Agent Lobbi Python SDK.
- Core functionalities for agent registration, task delegation, and communication.
- Command-line interface for basic operations.
- Core Agent class with WebSocket communication
- AgentLobbiClient for high-level operations
- Comprehensive error handling and logging
- Production-ready features:
  - Connection pooling and retry logic
  - Metrics collection and monitoring
  - Security with authentication and validation
  - Async/await support throughout
- CLI tool (`agent-lobbi`) for common operations
- Full test coverage with pytest
- Complete documentation and examples
- Support for:
  - Agent registration and management
  - Task delegation and monitoring
  - Real-time messaging
  - Capability-based routing
  - Health checks and status monitoring

### Features
- **Agent Management**: Create, register, and manage AI agents
- **Task Delegation**: Delegate tasks to available agents
- **Real-time Communication**: WebSocket-based messaging
- **Security**: Built-in authentication and data validation
- **Monitoring**: Comprehensive logging and metrics
- **CLI Tools**: Command-line interface for operations
- **High Performance**: Async/await with connection pooling
- **Production Ready**: Error handling, retries, and recovery

### Dependencies
- Python 3.8+
- httpx >= 0.25.0
- websockets >= 11.0
- pydantic >= 2.0.0
- click >= 8.0.0
- pytest >= 7.0.0 (dev)
- black >= 23.0.0 (dev)
- isort >= 5.0.0 (dev)
- flake8 >= 6.0.0 (dev)
- mypy >= 1.0.0 (dev)

### Documentation
- Complete README with examples
- API documentation
- CLI usage guide
- Development setup instructions
- Contributing guidelines

### Examples
- Basic agent creation and management
- Task delegation workflows
- Error handling patterns
- Advanced agent configurations

## [Unreleased]

### Planned
- Enhanced monitoring and analytics
- Advanced security features
- Performance optimizations
- Cloud deployment guides
- Integration with popular AI frameworks
- GraphQL API support
- Real-time dashboard

---

For more information about changes, see the [GitHub releases](https://github.com/agent-lobbi/agent-lobbi/releases). 