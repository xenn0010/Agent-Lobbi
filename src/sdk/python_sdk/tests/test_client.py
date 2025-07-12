"""
Test suite for Agent Lobbi SDK client module
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from python_sdk.client import (
    Agent,
    Capability,
    Message,
    MessageType,
    AgentLobbiClient,
    ConnectionError,
    AuthenticationError,
    TaskError,
    ConfigurationError,
    create_agent,
    quick_task_delegation
)


class TestCapability:
    """Test cases for Capability class"""
    
    def test_capability_creation(self):
        """Test basic capability creation"""
        cap = Capability("translate", "Translates text")
        assert cap.name == "translate"
        assert cap.description == "Translates text"
        assert cap.version == "1.0.0"
        assert cap.tags == []
    
    def test_capability_validation(self):
        """Test capability validation"""
        with pytest.raises(ConfigurationError):
            Capability("", "description")
        
        with pytest.raises(ConfigurationError):
            Capability("name", "")
        
        with pytest.raises(ConfigurationError):
            Capability(None, "description")
    
    def test_capability_to_dict(self):
        """Test capability serialization"""
        cap = Capability(
            "translate",
            "Translates text",
            input_schema={"text": "string"},
            output_schema={"translated": "string"},
            tags=["nlp", "translation"]
        )
        
        result = cap.to_dict()
        expected = {
            "name": "translate",
            "description": "Translates text",
            "input_schema": {"text": "string"},
            "output_schema": {"translated": "string"},
            "tags": ["nlp", "translation"],
            "version": "1.0.0"
        }
        assert result == expected


class TestMessage:
    """Test cases for Message class"""
    
    def test_message_creation(self):
        """Test basic message creation"""
        msg = Message("agent1", "agent2", MessageType.REQUEST)
        assert msg.sender_id == "agent1"
        assert msg.receiver_id == "agent2"
        assert msg.message_type == MessageType.REQUEST
        assert msg.priority == 2
        assert msg.retry_count == 0
    
    def test_message_validation(self):
        """Test message validation"""
        with pytest.raises(ConfigurationError):
            Message("", "agent2", MessageType.REQUEST)
        
        with pytest.raises(ConfigurationError):
            Message("agent1", "", MessageType.REQUEST)
    
    def test_message_serialization(self):
        """Test message to_dict and from_dict"""
        msg = Message(
            "agent1",
            "agent2", 
            MessageType.REQUEST,
            payload={"action": "test"},
            conversation_id="conv123"
        )
        
        msg_dict = msg.to_dict()
        assert msg_dict["sender_id"] == "agent1"
        assert msg_dict["receiver_id"] == "agent2"
        assert msg_dict["message_type"] == "REQUEST"
        assert msg_dict["payload"] == {"action": "test"}
        assert msg_dict["conversation_id"] == "conv123"
        
        # Test deserialization
        reconstructed = Message.from_dict(msg_dict)
        assert reconstructed.sender_id == msg.sender_id
        assert reconstructed.receiver_id == msg.receiver_id
        assert reconstructed.message_type == msg.message_type
        assert reconstructed.payload == msg.payload


class TestAgent:
    """Test cases for Agent class"""
    
    def test_agent_creation(self):
        """Test basic agent creation"""
        capabilities = [Capability("test", "Test capability")]
        agent = Agent(
            api_key="test_key",
            agent_type="TestAgent",
            capabilities=capabilities
        )
        
        assert agent.api_key == "test_key"
        assert agent.agent_type == "TestAgent"
        assert len(agent.capabilities) == 1
        assert agent.capabilities[0].name == "test"
    
    def test_agent_validation(self):
        """Test agent input validation"""
        capabilities = [Capability("test", "Test capability")]
        
        with pytest.raises(ConfigurationError):
            Agent("", "TestAgent", capabilities)
        
        with pytest.raises(ConfigurationError):
            Agent("test_key", "", capabilities)
        
        with pytest.raises(ConfigurationError):
            Agent("test_key", "TestAgent", [])
    
    def test_agent_url_parsing(self):
        """Test agent URL parsing"""
        capabilities = [Capability("test", "Test capability")]
        agent = Agent(
            api_key="test_key",
            agent_type="TestAgent",
            capabilities=capabilities,
            lobby_url="http://example.com:8080"
        )
        
        assert agent.lobby_host == "example.com"
        assert agent.lobby_port == 8080
        assert agent.websocket_url == "ws://example.com:8081"
    
    @pytest.mark.asyncio
    async def test_agent_start_stop(self):
        """Test agent start and stop"""
        capabilities = [Capability("test", "Test capability")]
        agent = Agent(
            api_key="test_key",
            agent_type="TestAgent",
            capabilities=capabilities
        )
        
        # Mock the registration and websocket connection
        with patch.object(agent, '_register_with_lobby', return_value=True), \
             patch.object(agent, '_connect_websocket', return_value=True), \
             patch.object(agent, '_start_background_tasks'):
            
            result = await agent.start()
            assert result is True
            assert agent._running is True
        
        await agent.stop()
        assert agent._running is False
    
    def test_message_handler_decorator(self):
        """Test message handler decorator"""
        capabilities = [Capability("test", "Test capability")]
        agent = Agent(
            api_key="test_key",
            agent_type="TestAgent",
            capabilities=capabilities
        )
        
        @agent.on_message
        async def test_handler(message):
            return {"result": "handled"}
        
        assert "default" in agent._message_handlers
        assert agent._message_handlers["default"] == test_handler


class TestAgentLobbiClient:
    """Test cases for AgentLobbiClient class"""
    
    def test_client_creation(self):
        """Test client creation"""
        client = AgentLobbiClient("test_key", "http://localhost:8092")
        assert client.api_key == "test_key"
        assert client.lobby_url == "http://localhost:8092"
        assert client.timeout == 30.0
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as async context manager"""
        client = AgentLobbiClient("test_key")
        
        async with client as c:
            assert c._http_client is not None
        
        # Should be closed after context
        assert client._http_client is None or client._http_client.is_closed
    
    @pytest.mark.asyncio
    async def test_list_agents(self):
        """Test listing agents"""
        client = AgentLobbiClient("test_key")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"agents": [{"id": "agent1"}]}
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            async with client as c:
                c._http_client = mock_client.return_value.__aenter__.return_value
                agents = await c.list_agents()
                
                assert len(agents) == 1
                assert agents[0]["id"] == "agent1"
    
    @pytest.mark.asyncio
    async def test_delegate_task(self):
        """Test task delegation"""
        client = AgentLobbiClient("test_key")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"task_id": "task123", "status": "delegated"}
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            async with client as c:
                c._http_client = mock_client.return_value.__aenter__.return_value
                result = await c.delegate_task(
                    "Test Task",
                    "Test description",
                    ["capability1"]
                )
                
                assert result["task_id"] == "task123"
                assert result["status"] == "delegated"
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check"""
        client = AgentLobbiClient("test_key")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"status": "healthy"}
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
            
            async with client as c:
                c._http_client = mock_client.return_value.__aenter__.return_value
                result = await c.health_check()
                
                assert result["status"] == "healthy"


class TestConvenienceFunctions:
    """Test cases for convenience functions"""
    
    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test create_agent convenience function"""
        agent = await create_agent(
            api_key="test_key",
            agent_type="TestAgent",
            capabilities=["translate", "summarize"]
        )
        
        assert agent.api_key == "test_key"
        assert agent.agent_type == "TestAgent"
        assert len(agent.capabilities) == 2
        assert agent.capabilities[0].name == "translate"
        assert agent.capabilities[1].name == "summarize"
    
    @pytest.mark.asyncio
    async def test_quick_task_delegation(self):
        """Test quick_task_delegation convenience function"""
        with patch('python_sdk.client.AgentLobbiClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.delegate_task.return_value = {"task_id": "task123"}
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            result = await quick_task_delegation(
                api_key="test_key",
                task_name="Test Task",
                task_description="Test description",
                required_capabilities=["translate"]
            )
            
            assert result["task_id"] == "task123"


class TestErrorHandling:
    """Test cases for error handling"""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling"""
        client = AgentLobbiClient("test_key")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
            
            async with client as c:
                c._http_client = mock_client.return_value.__aenter__.return_value
                with pytest.raises(ConnectionError):
                    await c.health_check()
    
    @pytest.mark.asyncio
    async def test_task_error_handling(self):
        """Test task error handling"""
        client = AgentLobbiClient("test_key")
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Task failed")
            
            async with client as c:
                c._http_client = mock_client.return_value.__aenter__.return_value
                with pytest.raises(TaskError):
                    await c.delegate_task("Test", "Description", ["cap1"])


if __name__ == "__main__":
    pytest.main([__file__]) 