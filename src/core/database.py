"""
Database abstraction layer supporting both MongoDB and PostgreSQL.
Provides unified interface for agent ecosystem data persistence.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from enum import Enum
import json

# MongoDB imports
import motor.motor_asyncio
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# PostgreSQL imports
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

# Pydantic for data validation
from pydantic import BaseModel, Field
from typing import Optional

# Configuration
from dataclasses import dataclass


class DatabaseType(Enum):
    MONGODB = "mongodb"
    POSTGRESQL = "postgresql"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_type: DatabaseType
    host: str = "localhost"
    port: int = None
    username: str = None
    password: str = None
    database: str = "agent_ecosystem"
    
    # Connection pool settings
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    
    # MongoDB specific
    replica_set: Optional[str] = None
    auth_source: str = "admin"
    
    # PostgreSQL specific
    ssl_mode: str = "prefer"
    
    def __post_init__(self):
        if self.port is None:
            self.port = 27017 if self.db_type == DatabaseType.MONGODB else 5432


# Pydantic models for data validation
class AgentRegistration(BaseModel):
    agent_id: str
    agent_type: str
    capabilities: List[Dict[str, Any]]
    status: str = "active"
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageRecord(BaseModel):
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    conversation_id: Optional[str] = None
    timestamp: datetime
    status: str = "sent"
    priority: int = 2
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationRecord(BaseModel):
    conversation_id: str
    participants: List[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InteractionRecord(BaseModel):
    interaction_id: str
    initiator_id: str
    target_id: str
    interaction_type: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    status: str = "in_progress"
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# SQLAlchemy models for PostgreSQL
class Base(DeclarativeBase):
    pass


class AgentModel(Base):
    __tablename__ = "agents"
    
    agent_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    agent_type: Mapped[str] = mapped_column(String(100), nullable=False)
    capabilities: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active")
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})


class MessageModel(Base):
    __tablename__ = "messages"
    
    message_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    sender_id: Mapped[str] = mapped_column(String(255), nullable=False)
    receiver_id: Mapped[str] = mapped_column(String(255), nullable=False)
    message_type: Mapped[str] = mapped_column(String(50), nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(255))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="sent")
    priority: Mapped[int] = mapped_column(Integer, default=2)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})


class ConversationModel(Base):
    __tablename__ = "conversations"
    
    conversation_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    participants: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="active")
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})


class InteractionModel(Base):
    __tablename__ = "interactions"
    
    interaction_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    initiator_id: Mapped[str] = mapped_column(String(255), nullable=False)
    target_id: Mapped[str] = mapped_column(String(255), nullable=False)
    interaction_type: Mapped[str] = mapped_column(String(100), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(50), default="in_progress")
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})


class DatabaseManager:
    """Unified database manager supporting both MongoDB and PostgreSQL"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database connections
        self._mongo_client: Optional[AsyncIOMotorClient] = None
        self._mongo_db: Optional[AsyncIOMotorDatabase] = None
        self._pg_engine = None
        self._pg_session_factory = None
        
        self._connected = False
    
    async def connect(self):
        """Establish database connections"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                await self._connect_mongodb()
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                await self._connect_postgresql()
            
            self._connected = True
            self.logger.info(f"Connected to {self.config.db_type.value} database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connections"""
        try:
            if self.config.db_type == DatabaseType.MONGODB and self._mongo_client:
                self._mongo_client.close()
            elif self.config.db_type == DatabaseType.POSTGRESQL and self._pg_engine:
                await self._pg_engine.dispose()
            
            self._connected = False
            self.logger.info("Disconnected from database")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from database: {e}")
    
    async def _connect_mongodb(self):
        """Connect to MongoDB"""
        connection_string = self._build_mongo_connection_string()
        
        self._mongo_client = AsyncIOMotorClient(
            connection_string,
            maxPoolSize=self.config.max_connections,
            minPoolSize=self.config.min_connections,
            serverSelectionTimeoutMS=self.config.connection_timeout * 1000
        )
        
        # Test connection
        await self._mongo_client.admin.command('ping')
        self._mongo_db = self._mongo_client[self.config.database]
        
        # Create indexes
        await self._create_mongo_indexes()
    
    async def _connect_postgresql(self):
        """Connect to PostgreSQL"""
        connection_string = self._build_pg_connection_string()
        
        self._pg_engine = create_async_engine(
            connection_string,
            pool_size=self.config.max_connections,
            max_overflow=0,
            pool_timeout=self.config.connection_timeout,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create tables
        async with self._pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._pg_session_factory = async_sessionmaker(
            self._pg_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def _build_mongo_connection_string(self) -> str:
        """Build MongoDB connection string"""
        auth_part = ""
        if self.config.username and self.config.password:
            auth_part = f"{self.config.username}:{self.config.password}@"
        
        replica_part = ""
        if self.config.replica_set:
            replica_part = f"?replicaSet={self.config.replica_set}&authSource={self.config.auth_source}"
        
        return f"mongodb://{auth_part}{self.config.host}:{self.config.port}/{replica_part}"
    
    def _build_pg_connection_string(self) -> str:
        """Build PostgreSQL connection string"""
        auth_part = ""
        if self.config.username and self.config.password:
            auth_part = f"{self.config.username}:{self.config.password}@"
        
        return (f"postgresql+asyncpg://{auth_part}{self.config.host}:{self.config.port}/"
                f"{self.config.database}?sslmode={self.config.ssl_mode}")
    
    async def _create_mongo_indexes(self):
        """Create MongoDB indexes for better performance"""
        # Agent indexes
        await self._mongo_db.agents.create_index("agent_id", unique=True)
        await self._mongo_db.agents.create_index("agent_type")
        await self._mongo_db.agents.create_index("status")
        await self._mongo_db.agents.create_index("last_seen")
        
        # Message indexes
        await self._mongo_db.messages.create_index("message_id", unique=True)
        await self._mongo_db.messages.create_index("sender_id")
        await self._mongo_db.messages.create_index("receiver_id")
        await self._mongo_db.messages.create_index("conversation_id")
        await self._mongo_db.messages.create_index("timestamp")
        await self._mongo_db.messages.create_index("message_type")
        
        # Conversation indexes
        await self._mongo_db.conversations.create_index("conversation_id", unique=True)
        await self._mongo_db.conversations.create_index("participants")
        await self._mongo_db.conversations.create_index("created_at")
        
        # Interaction indexes
        await self._mongo_db.interactions.create_index("interaction_id", unique=True)
        await self._mongo_db.interactions.create_index("initiator_id")
        await self._mongo_db.interactions.create_index("target_id")
        await self._mongo_db.interactions.create_index("started_at")
    
    # Agent operations
    async def register_agent(self, agent: AgentRegistration) -> bool:
        """Register a new agent"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self._mongo_db.agents.insert_one(agent.model_dump())
                return result.acknowledged
            else:
                async with self._pg_session_factory() as session:
                    db_agent = AgentModel(**agent.model_dump())
                    session.add(db_agent)
                    await session.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                doc = await self._mongo_db.agents.find_one({"agent_id": agent_id})
                return AgentRegistration(**doc) if doc else None
            else:
                async with self._pg_session_factory() as session:
                    result = await session.get(AgentModel, agent_id)
                    if result:
                        return AgentRegistration(
                            agent_id=result.agent_id,
                            agent_type=result.agent_type,
                            capabilities=result.capabilities,
                            status=result.status,
                            registered_at=result.registered_at,
                            last_seen=result.last_seen,
                            metadata=result.metadata
                        )
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get agent {agent_id}: {e}")
            return None
    
    async def update_agent_last_seen(self, agent_id: str) -> bool:
        """Update agent's last seen timestamp"""
        try:
            now = datetime.now(timezone.utc)
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self._mongo_db.agents.update_one(
                    {"agent_id": agent_id},
                    {"$set": {"last_seen": now}}
                )
                return result.modified_count > 0
            else:
                async with self._pg_session_factory() as session:
                    agent = await session.get(AgentModel, agent_id)
                    if agent:
                        agent.last_seen = now
                        await session.commit()
                        return True
                    return False
        except Exception as e:
            self.logger.error(f"Failed to update last seen for agent {agent_id}: {e}")
            return False
    
    async def get_active_agents(self) -> List[AgentRegistration]:
        """Get all active agents"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                cursor = self._mongo_db.agents.find({"status": "active"})
                docs = await cursor.to_list(length=None)
                return [AgentRegistration(**doc) for doc in docs]
            else:
                async with self._pg_session_factory() as session:
                    from sqlalchemy import select
                    result = await session.execute(
                        select(AgentModel).where(AgentModel.status == "active")
                    )
                    agents = result.scalars().all()
                    return [
                        AgentRegistration(
                            agent_id=agent.agent_id,
                            agent_type=agent.agent_type,
                            capabilities=agent.capabilities,
                            status=agent.status,
                            registered_at=agent.registered_at,
                            last_seen=agent.last_seen,
                            metadata=agent.metadata
                        ) for agent in agents
                    ]
        except Exception as e:
            self.logger.error(f"Failed to get active agents: {e}")
            return []
    
    # Message operations
    async def store_message(self, message: MessageRecord) -> bool:
        """Store a message"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self._mongo_db.messages.insert_one(message.model_dump())
                return result.acknowledged
            else:
                async with self._pg_session_factory() as session:
                    db_message = MessageModel(**message.model_dump())
                    session.add(db_message)
                    await session.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to store message {message.message_id}: {e}")
            return False
    
    async def get_conversation_messages(self, conversation_id: str, limit: int = 100) -> List[MessageRecord]:
        """Get messages for a conversation"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                cursor = self._mongo_db.messages.find(
                    {"conversation_id": conversation_id}
                ).sort("timestamp", 1).limit(limit)
                docs = await cursor.to_list(length=None)
                return [MessageRecord(**doc) for doc in docs]
            else:
                async with self._pg_session_factory() as session:
                    from sqlalchemy import select
                    result = await session.execute(
                        select(MessageModel)
                        .where(MessageModel.conversation_id == conversation_id)
                        .order_by(MessageModel.timestamp)
                        .limit(limit)
                    )
                    messages = result.scalars().all()
                    return [
                        MessageRecord(
                            message_id=msg.message_id,
                            sender_id=msg.sender_id,
                            receiver_id=msg.receiver_id,
                            message_type=msg.message_type,
                            payload=msg.payload,
                            conversation_id=msg.conversation_id,
                            timestamp=msg.timestamp,
                            status=msg.status,
                            priority=msg.priority,
                            metadata=msg.metadata
                        ) for msg in messages
                    ]
        except Exception as e:
            self.logger.error(f"Failed to get conversation messages: {e}")
            return []
    
    # Conversation operations
    async def create_conversation(self, conversation: ConversationRecord) -> bool:
        """Create a new conversation"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self._mongo_db.conversations.insert_one(conversation.model_dump())
                return result.acknowledged
            else:
                async with self._pg_session_factory() as session:
                    db_conversation = ConversationModel(**conversation.model_dump())
                    session.add(db_conversation)
                    await session.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to create conversation {conversation.conversation_id}: {e}")
            return False
    
    # Interaction operations
    async def create_interaction(self, interaction: InteractionRecord) -> bool:
        """Create a new interaction record"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                result = await self._mongo_db.interactions.insert_one(interaction.model_dump())
                return result.acknowledged
            else:
                async with self._pg_session_factory() as session:
                    db_interaction = InteractionModel(**interaction.model_dump())
                    session.add(db_interaction)
                    await session.commit()
                    return True
        except Exception as e:
            self.logger.error(f"Failed to create interaction {interaction.interaction_id}: {e}")
            return False
    
    async def complete_interaction(self, interaction_id: str, result: Dict[str, Any]) -> bool:
        """Mark an interaction as completed"""
        try:
            now = datetime.now(timezone.utc)
            if self.config.db_type == DatabaseType.MONGODB:
                update_result = await self._mongo_db.interactions.update_one(
                    {"interaction_id": interaction_id},
                    {"$set": {"completed_at": now, "status": "completed", "result": result}}
                )
                return update_result.modified_count > 0
            else:
                async with self._pg_session_factory() as session:
                    interaction = await session.get(InteractionModel, interaction_id)
                    if interaction:
                        interaction.completed_at = now
                        interaction.status = "completed"
                        interaction.result = result
                        await session.commit()
                        return True
                    return False
        except Exception as e:
            self.logger.error(f"Failed to complete interaction {interaction_id}: {e}")
            return False
    
    # Analytics and monitoring
    async def get_agent_stats(self, agent_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get agent statistics for the specified time period"""
        try:
            from datetime import timedelta
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            if self.config.db_type == DatabaseType.MONGODB:
                # Messages sent
                sent_count = await self._mongo_db.messages.count_documents({
                    "sender_id": agent_id,
                    "timestamp": {"$gte": since}
                })
                
                # Messages received
                received_count = await self._mongo_db.messages.count_documents({
                    "receiver_id": agent_id,
                    "timestamp": {"$gte": since}
                })
                
                # Interactions initiated
                interactions_initiated = await self._mongo_db.interactions.count_documents({
                    "initiator_id": agent_id,
                    "started_at": {"$gte": since}
                })
                
                # Interactions participated
                interactions_participated = await self._mongo_db.interactions.count_documents({
                    "target_id": agent_id,
                    "started_at": {"$gte": since}
                })
                
            else:
                async with self._pg_session_factory() as session:
                    from sqlalchemy import select, func
                    
                    # Messages sent
                    sent_result = await session.execute(
                        select(func.count(MessageModel.message_id))
                        .where(MessageModel.sender_id == agent_id)
                        .where(MessageModel.timestamp >= since)
                    )
                    sent_count = sent_result.scalar()
                    
                    # Messages received
                    received_result = await session.execute(
                        select(func.count(MessageModel.message_id))
                        .where(MessageModel.receiver_id == agent_id)
                        .where(MessageModel.timestamp >= since)
                    )
                    received_count = received_result.scalar()
                    
                    # Interactions initiated
                    init_result = await session.execute(
                        select(func.count(InteractionModel.interaction_id))
                        .where(InteractionModel.initiator_id == agent_id)
                        .where(InteractionModel.started_at >= since)
                    )
                    interactions_initiated = init_result.scalar()
                    
                    # Interactions participated
                    part_result = await session.execute(
                        select(func.count(InteractionModel.interaction_id))
                        .where(InteractionModel.target_id == agent_id)
                        .where(InteractionModel.started_at >= since)
                    )
                    interactions_participated = part_result.scalar()
            
            return {
                "agent_id": agent_id,
                "period_hours": hours,
                "messages_sent": sent_count,
                "messages_received": received_count,
                "interactions_initiated": interactions_initiated,
                "interactions_participated": interactions_participated,
                "total_activity": sent_count + received_count + interactions_initiated + interactions_participated
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get agent stats for {agent_id}: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                # Ping MongoDB
                await self._mongo_client.admin.command('ping')
                
                # Get database stats
                stats = await self._mongo_db.command("dbStats")
                
                return {
                    "status": "healthy",
                    "database_type": "mongodb",
                    "database_name": self.config.database,
                    "collections": stats.get("collections", 0),
                    "data_size": stats.get("dataSize", 0),
                    "storage_size": stats.get("storageSize", 0)
                }
            else:
                # Test PostgreSQL connection
                async with self._pg_session_factory() as session:
                    result = await session.execute("SELECT 1")
                    result.scalar()
                
                return {
                    "status": "healthy",
                    "database_type": "postgresql",
                    "database_name": self.config.database
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_type": self.config.db_type.value
            }


# Factory function for easy database manager creation
def create_database_manager(
    db_type: str,
    host: str = "localhost",
    port: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "agent_ecosystem",
    **kwargs
) -> DatabaseManager:
    """Factory function to create a database manager"""
    
    db_type_enum = DatabaseType.MONGODB if db_type.lower() == "mongodb" else DatabaseType.POSTGRESQL
    
    config = DatabaseConfig(
        db_type=db_type_enum,
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        **kwargs
    )
    
    return DatabaseManager(config) 