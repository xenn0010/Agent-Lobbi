"""
Production Database Layer for Agent Lobbi
Supports PostgreSQL (production) and SQLite (development)
"""
import asyncio
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import json
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    HAS_STRUCTLOG = False
    import logging
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, Boolean, JSON, select, update, delete
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

logger = structlog.get_logger(__name__) if HAS_STRUCTLOG else logging.getLogger(__name__)

class Base(DeclarativeBase):
    """Base class for all database models"""
    pass

class Agent(Base):
    __tablename__ = "agents"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_type: Mapped[str] = mapped_column(String, nullable=False)
    capabilities: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String, default="online")
    reputation: Mapped[float] = mapped_column(Integer, default=100.0)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    agent_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    name: Mapped[str] = mapped_column(String, nullable=True)  # Add name field

class Workflow(Base):
    __tablename__ = "workflows"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="created")
    tasks: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    shared_state: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    participants: Mapped[List[str]] = mapped_column(JSON, default=list)
    result: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

class Message(Base):
    __tablename__ = "messages"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    sender_id: Mapped[str] = mapped_column(String, nullable=False)
    receiver_id: Mapped[str] = mapped_column(String, nullable=False)
    message_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    conversation_id: Mapped[str] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    priority: Mapped[int] = mapped_column(Integer, default=2)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

class AgentMetrics(Base):
    __tablename__ = "agent_metrics"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    metric_type: Mapped[str] = mapped_column(String, nullable=False)  # task_completion, error_rate, etc
    value: Mapped[float] = mapped_column(Integer, nullable=False)
    metric_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())

class DatabaseManager:
    """Production-ready database manager with async support and Redis optimization"""
    
    def __init__(self, database_url: Optional[str] = None, enable_redis: bool = True):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
        self.enable_redis = enable_redis
        self.redis_db = None
        
        # Try to import Redis database with improved error handling
        if enable_redis:
            try:
                from .redis_database import RedisAgentDatabase, REDIS_AVAILABLE
                if REDIS_AVAILABLE:
                    self.redis_db = RedisAgentDatabase()
                    logger.info("Redis database module loaded successfully")
                else:
                    logger.warning("Redis module loaded but Redis not available, falling back to SQLite")
                    self.redis_db = None
            except ImportError as e:
                logger.warning("Redis module not available, falling back to SQLite for all operations", error=str(e))
                self.redis_db = None
            except Exception as e:
                logger.warning("Redis initialization failed, falling back to SQLite", error=str(e))
                self.redis_db = None
        
    def _get_database_url(self) -> str:
        """Get database URL from environment or default to SQLite"""
        # Check for PostgreSQL first (production)
        if pg_url := os.getenv("DATABASE_URL"):
            return pg_url
        
        # Check for individual PostgreSQL components
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "agent_lobby")
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_password = os.getenv("POSTGRES_PASSWORD", "")
        
        if pg_password:
            return f"postgresql+asyncpg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        
        # Fallback to SQLite for development
        return "sqlite+aiosqlite:///./agent_lobby.db"
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Initialize Redis first (if enabled)
            if self.redis_db:
                try:
                    await self.redis_db.initialize()
                    logger.info("Redis database initialized for high-performance operations")
                except Exception as e:
                    logger.warning("Redis initialization failed, falling back to SQLite", error=str(e))
                    self.redis_db = None
            
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            db_type = "Hybrid (Redis + SQLite)" if self.redis_db else "SQLite only"
            logger.info("Database initialized successfully", url=self.database_url, type=db_type)
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    # Agent Operations (Redis-optimized)
    async def save_agent(self, agent_data: Dict[str, Any]) -> bool:
        """Save or update agent data (Redis-first with SQLite fallback)"""
        # Try Redis first for high performance
        if self.redis_db:
            try:
                success = await self.redis_db.save_agent(agent_data)
                if success:
                    logger.debug("Agent saved to Redis", agent_id=agent_data.get("id"))
                    return True
            except Exception as e:
                logger.warning("Redis save failed, falling back to SQLite", error=str(e))
        
        # Fallback to SQLite
        try:
            async with self.get_session() as session:
                # Check if agent exists
                result = await session.execute(select(Agent).where(Agent.id == agent_data["id"]))
                existing_agent = result.scalar_one_or_none()
                
                if existing_agent:
                    # Update existing agent
                    await session.execute(
                        update(Agent).where(Agent.id == agent_data["id"]).values(**agent_data)
                    )
                else:
                    # Create new agent
                    agent = Agent(**agent_data)
                    session.add(agent)
                
                return True
        except Exception as e:
            logger.error("Failed to save agent", agent_id=agent_data.get("id"), error=str(e))
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID (Redis-first with SQLite fallback)"""
        # Try Redis first for high performance
        if self.redis_db:
            try:
                agent_data = await self.redis_db.get_agent(agent_id)
                if agent_data:
                    logger.debug("Agent retrieved from Redis", agent_id=agent_id)
                    return agent_data
            except Exception as e:
                logger.warning("Redis get failed, falling back to SQLite", error=str(e))
        
        # Fallback to SQLite
        try:
            async with self.get_session() as session:
                result = await session.execute(select(Agent).where(Agent.id == agent_id))
                agent = result.scalar_one_or_none()
                
                if agent:
                    return {
                        "id": agent.id,
                        "agent_type": agent.agent_type,
                        "capabilities": agent.capabilities,
                        "status": agent.status,
                        "reputation": agent.reputation,
                        "last_seen": agent.last_seen,
                        "metadata": agent.agent_metadata,
                        "created_at": agent.created_at
                    }
                return None
        except Exception as e:
            logger.error("Failed to get agent", agent_id=agent_id, error=str(e))
            return None
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all active agents (Redis-first with SQLite fallback)"""
        # Try Redis first for high performance
        if self.redis_db:
            try:
                agents = await self.redis_db.get_all_agents()
                if agents is not None:  # Empty list is valid
                    logger.debug("All agents retrieved from Redis", count=len(agents))
                    return agents
            except Exception as e:
                logger.warning("Redis get_all failed, falling back to SQLite", error=str(e))
        
        # Fallback to SQLite
        try:
            async with self.get_session() as session:
                result = await session.execute(select(Agent).where(Agent.status == "online"))
                agents = result.scalars().all()
                
                return [
                    {
                        "id": agent.id,
                        "agent_type": agent.agent_type,
                        "capabilities": agent.capabilities,
                        "status": agent.status,
                        "reputation": agent.reputation,
                        "last_seen": agent.last_seen,
                        "metadata": agent.agent_metadata
                    }
                    for agent in agents
                ]
        except Exception as e:
            logger.error("Failed to get all agents", error=str(e))
            return []
    
    # Workflow Operations
    async def save_workflow(self, workflow_data: Dict[str, Any]) -> bool:
        """Save or update workflow"""
        try:
            async with self.get_session() as session:
                result = await session.execute(select(Workflow).where(Workflow.id == workflow_data["id"]))
                existing_workflow = result.scalar_one_or_none()
                
                if existing_workflow:
                    await session.execute(
                        update(Workflow).where(Workflow.id == workflow_data["id"]).values(**workflow_data)
                    )
                else:
                    workflow = Workflow(**workflow_data)
                    session.add(workflow)
                
                return True
        except Exception as e:
            logger.error("Failed to save workflow", workflow_id=workflow_data.get("id"), error=str(e))
            return False
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        try:
            async with self.get_session() as session:
                result = await session.execute(select(Workflow).where(Workflow.id == workflow_id))
                workflow = result.scalar_one_or_none()
                
                if workflow:
                    return {
                        "id": workflow.id,
                        "name": workflow.name,
                        "description": workflow.description,
                        "created_by": workflow.created_by,
                        "status": workflow.status,
                        "tasks": workflow.tasks,
                        "shared_state": workflow.shared_state,
                        "participants": workflow.participants,
                        "result": workflow.result,
                        "error": workflow.error,
                        "created_at": workflow.created_at,
                        "started_at": workflow.started_at,
                        "completed_at": workflow.completed_at
                    }
                return None
        except Exception as e:
            logger.error("Failed to get workflow", workflow_id=workflow_id, error=str(e))
            return None
    
    # Message Operations
    async def save_message(self, message_data: Dict[str, Any]) -> bool:
        """Save message for audit trail"""
        try:
            async with self.get_session() as session:
                message = Message(**message_data)
                session.add(message)
                return True
        except Exception as e:
            logger.error("Failed to save message", message_id=message_data.get("id"), error=str(e))
            return False
    
    # Metrics Operations
    async def save_agent_metric(self, agent_id: str, metric_type: str, value: float, metadata: Dict[str, Any] = None) -> bool:
        """Save agent performance metric"""
        try:
            async with self.get_session() as session:
                metric = AgentMetrics(
                    agent_id=agent_id,
                    metric_type=metric_type,
                    value=value,
                    metric_metadata=metadata or {}
                )
                session.add(metric)
                return True
        except Exception as e:
            logger.error("Failed to save metric", agent_id=agent_id, metric_type=metric_type, error=str(e))
            return False
    
    async def get_agent_metrics(self, agent_id: str, metric_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent metrics"""
        try:
            async with self.get_session() as session:
                query = select(AgentMetrics).where(AgentMetrics.agent_id == agent_id)
                if metric_type:
                    query = query.where(AgentMetrics.metric_type == metric_type)
                query = query.order_by(AgentMetrics.timestamp.desc()).limit(limit)
                
                result = await session.execute(query)
                metrics = result.scalars().all()
                
                return [
                    {
                        "id": metric.id,
                        "agent_id": metric.agent_id,
                        "metric_type": metric.metric_type,
                        "value": metric.value,
                        "metadata": metric.metric_metadata,
                        "timestamp": metric.timestamp
                    }
                    for metric in metrics
                ]
        except Exception as e:
            logger.error("Failed to get agent metrics", agent_id=agent_id, error=str(e))
            return []

# Global database manager instance
db_manager = DatabaseManager() 