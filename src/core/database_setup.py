"""
Database setup and management utilities.
Handles database initialization, migrations, and maintenance for both MongoDB and PostgreSQL.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json

# Database imports
import motor.motor_asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

# Our imports
from .database import DatabaseManager, DatabaseConfig, DatabaseType, Base
from .config import get_config


class DatabaseSetupError(Exception):
    """Raised when database setup fails"""
    pass


class DatabaseSetup:
    """Database setup and management utility"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def initialize_database(self, force: bool = False) -> bool:
        """Initialize the database with required tables/collections"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                return await self._initialize_mongodb(force)
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                return await self._initialize_postgresql(force)
            else:
                raise DatabaseSetupError(f"Unsupported database type: {self.config.db_type}")
        
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise DatabaseSetupError(f"Database initialization failed: {e}")
    
    async def _initialize_mongodb(self, force: bool = False) -> bool:
        """Initialize MongoDB database"""
        connection_string = self._build_mongo_connection_string()
        client = motor.motor_asyncio.AsyncIOMotorClient(connection_string)
        
        try:
            # Test connection
            await client.admin.command('ping')
            self.logger.info("MongoDB connection successful")
            
            db = client[self.config.database]
            
            # Create collections if they don't exist
            collections = await db.list_collection_names()
            required_collections = ['agents', 'messages', 'conversations', 'interactions', 'capabilities']
            
            for collection_name in required_collections:
                if collection_name not in collections or force:
                    await db.create_collection(collection_name)
                    self.logger.info(f"Created collection: {collection_name}")
            
            # Create indexes
            await self._create_mongo_indexes(db, force)
            
            # Insert initial data if needed
            await self._insert_initial_mongo_data(db, force)
            
            self.logger.info("MongoDB initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"MongoDB initialization failed: {e}")
            raise
        finally:
            client.close()
    
    async def _initialize_postgresql(self, force: bool = False) -> bool:
        """Initialize PostgreSQL database"""
        # First, create database if it doesn't exist
        await self._create_postgresql_database()
        
        # Create tables using SQLAlchemy
        connection_string = self._build_pg_connection_string()
        engine = create_async_engine(connection_string)
        
        try:
            async with engine.begin() as conn:
                if force:
                    # Drop all tables if force is True
                    await conn.run_sync(Base.metadata.drop_all)
                    self.logger.info("Dropped all existing tables")
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                self.logger.info("Created all tables")
                
                # Create indexes
                await self._create_pg_indexes(conn, force)
                
                # Insert initial data
                await self._insert_initial_pg_data(conn, force)
            
            self.logger.info("PostgreSQL initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")
            raise
        finally:
            await engine.dispose()
    
    async def _create_postgresql_database(self):
        """Create PostgreSQL database if it doesn't exist"""
        # Connect to default postgres database to create our database
        default_config = self.config.__dict__.copy()
        default_config['database'] = 'postgres'
        
        connection_string = self._build_pg_connection_string(database='postgres')
        
        try:
            conn = await asyncpg.connect(connection_string)
            
            # Check if database exists
            result = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self.config.database
            )
            
            if not result:
                # Create database
                await conn.execute(f'CREATE DATABASE "{self.config.database}"')
                self.logger.info(f"Created database: {self.config.database}")
            else:
                self.logger.info(f"Database {self.config.database} already exists")
            
            await conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create PostgreSQL database: {e}")
            raise
    
    async def _create_mongo_indexes(self, db, force: bool = False):
        """Create MongoDB indexes"""
        indexes = {
            'agents': [
                [('agent_id', 1)],
                [('agent_type', 1)],
                [('status', 1)],
                [('last_seen', -1)],
                [('registered_at', -1)]
            ],
            'messages': [
                [('message_id', 1)],
                [('sender_id', 1)],
                [('receiver_id', 1)],
                [('conversation_id', 1)],
                [('timestamp', -1)],
                [('status', 1)],
                [('priority', -1)],
                [('sender_id', 1), ('timestamp', -1)],
                [('receiver_id', 1), ('timestamp', -1)]
            ],
            'conversations': [
                [('conversation_id', 1)],
                [('participants', 1)],
                [('status', 1)],
                [('created_at', -1)],
                [('updated_at', -1)]
            ],
            'interactions': [
                [('interaction_id', 1)],
                [('initiator_id', 1)],
                [('target_id', 1)],
                [('interaction_type', 1)],
                [('status', 1)],
                [('started_at', -1)],
                [('completed_at', -1)]
            ],
            'capabilities': [
                [('agent_id', 1)],
                [('capability_name', 1)],
                [('capability_type', 1)],
                [('status', 1)],
                [('agent_id', 1), ('capability_name', 1)]
            ]
        }
        
        for collection_name, collection_indexes in indexes.items():
            collection = db[collection_name]
            
            # Get existing indexes
            existing_indexes = await collection.list_indexes().to_list(length=None)
            existing_index_keys = {tuple(idx['key'].items()) for idx in existing_indexes if 'key' in idx}
            
            for index_spec in collection_indexes:
                index_key = tuple(index_spec)
                if index_key not in existing_index_keys or force:
                    try:
                        await collection.create_index(index_spec)
                        self.logger.info(f"Created index {index_spec} on {collection_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create index {index_spec} on {collection_name}: {e}")
    
    async def _create_pg_indexes(self, conn, force: bool = False):
        """Create PostgreSQL indexes"""
        indexes = [
            # Agents table
            "CREATE INDEX IF NOT EXISTS idx_agents_type ON agents(agent_type)",
            "CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)",
            "CREATE INDEX IF NOT EXISTS idx_agents_last_seen ON agents(last_seen DESC)",
            "CREATE INDEX IF NOT EXISTS idx_agents_registered_at ON agents(registered_at DESC)",
            
            # Messages table
            "CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_receiver ON messages(receiver_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(status)",
            "CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(priority DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_sender_time ON messages(sender_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_receiver_time ON messages(receiver_id, timestamp DESC)",
            
            # Conversations table
            "CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)",
            
            # Interactions table
            "CREATE INDEX IF NOT EXISTS idx_interactions_initiator ON interactions(initiator_id)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_target ON interactions(target_id)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_started ON interactions(started_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_completed ON interactions(completed_at DESC)"
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(text(index_sql))
                self.logger.info(f"Created index: {index_sql.split()[-1]}")
            except Exception as e:
                self.logger.warning(f"Failed to create index: {e}")
    
    async def _insert_initial_mongo_data(self, db, force: bool = False):
        """Insert initial data into MongoDB"""
        # Insert system agent if it doesn't exist
        system_agent = {
            'agent_id': 'system',
            'agent_type': 'system',
            'capabilities': [{'name': 'system_management', 'type': 'internal'}],
            'status': 'active',
            'registered_at': datetime.now(timezone.utc),
            'last_seen': datetime.now(timezone.utc),
            'metadata': {'description': 'System management agent'}
        }
        
        existing_system = await db.agents.find_one({'agent_id': 'system'})
        if not existing_system or force:
            await db.agents.replace_one(
                {'agent_id': 'system'}, 
                system_agent, 
                upsert=True
            )
            self.logger.info("Inserted/updated system agent")
    
    async def _insert_initial_pg_data(self, conn, force: bool = False):
        """Insert initial data into PostgreSQL"""
        # Insert system agent if it doesn't exist
        system_agent_sql = """
        INSERT INTO agents (agent_id, agent_type, capabilities, status, registered_at, last_seen, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (agent_id) DO UPDATE SET
            last_seen = EXCLUDED.last_seen,
            metadata = EXCLUDED.metadata
        """
        
        await conn.execute(
            text(system_agent_sql),
            'system',
            'system',
            json.dumps([{'name': 'system_management', 'type': 'internal'}]),
            'active',
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
            json.dumps({'description': 'System management agent'})
        )
        self.logger.info("Inserted/updated system agent")
    
    def _build_mongo_connection_string(self) -> str:
        """Build MongoDB connection string"""
        if self.config.username and self.config.password:
            auth = f"{self.config.username}:{self.config.password}@"
        else:
            auth = ""
        
        connection_string = f"mongodb://{auth}{self.config.host}:{self.config.port}/{self.config.database}"
        
        params = []
        if self.config.replica_set:
            params.append(f"replicaSet={self.config.replica_set}")
        if self.config.auth_source:
            params.append(f"authSource={self.config.auth_source}")
        
        if params:
            connection_string += "?" + "&".join(params)
        
        return connection_string
    
    def _build_pg_connection_string(self, database: Optional[str] = None) -> str:
        """Build PostgreSQL connection string"""
        db_name = database or self.config.database
        
        if self.config.username and self.config.password:
            auth = f"{self.config.username}:{self.config.password}@"
        else:
            auth = ""
        
        return f"postgresql+asyncpg://{auth}{self.config.host}:{self.config.port}/{db_name}"
    
    async def backup_database(self, backup_path: str) -> bool:
        """Create a database backup"""
        try:
            if self.config.db_type == DatabaseType.MONGODB:
                return await self._backup_mongodb(backup_path)
            elif self.config.db_type == DatabaseType.POSTGRESQL:
                return await self._backup_postgresql(backup_path)
            else:
                raise DatabaseSetupError(f"Backup not supported for {self.config.db_type}")
        
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    async def _backup_mongodb(self, backup_path: str) -> bool:
        """Backup MongoDB database"""
        import subprocess
        
        cmd = [
            'mongodump',
            '--host', f"{self.config.host}:{self.config.port}",
            '--db', self.config.database,
            '--out', backup_path
        ]
        
        if self.config.username:
            cmd.extend(['--username', self.config.username])
        if self.config.password:
            cmd.extend(['--password', self.config.password])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"MongoDB backup completed: {backup_path}")
                return True
            else:
                self.logger.error(f"MongoDB backup failed: {result.stderr}")
                return False
        except FileNotFoundError:
            self.logger.error("mongodump command not found. Please install MongoDB tools.")
            return False
    
    async def _backup_postgresql(self, backup_path: str) -> bool:
        """Backup PostgreSQL database"""
        import subprocess
        
        env = {}
        if self.config.password:
            env['PGPASSWORD'] = self.config.password
        
        cmd = [
            'pg_dump',
            '-h', self.config.host,
            '-p', str(self.config.port),
            '-d', self.config.database,
            '-f', backup_path
        ]
        
        if self.config.username:
            cmd.extend(['-U', self.config.username])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode == 0:
                self.logger.info(f"PostgreSQL backup completed: {backup_path}")
                return True
            else:
                self.logger.error(f"PostgreSQL backup failed: {result.stderr}")
                return False
        except FileNotFoundError:
            self.logger.error("pg_dump command not found. Please install PostgreSQL client tools.")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            db_manager = DatabaseManager(self.config)
            await db_manager.connect()
            
            health_info = await db_manager.health_check()
            await db_manager.disconnect()
            
            return {
                'status': 'healthy',
                'database_type': self.config.db_type.value,
                'database_info': health_info
            }
        
        except Exception as e:
            return {
                'status': 'unhealthy',
                'database_type': self.config.db_type.value,
                'error': str(e)
            }


async def setup_database_from_config(config_file: Optional[str] = None, force: bool = False) -> bool:
    """Setup database using configuration file"""
    try:
        # Load configuration
        config = get_config(config_file)
        
        # Create database config
        db_config = DatabaseConfig(
            db_type=DatabaseType(config.database.type),
            host=config.database.host,
            port=config.database.port,
            username=config.database.username,
            password=config.database.password,
            database=config.database.database,
            min_connections=config.database.min_connections,
            max_connections=config.database.max_connections,
            connection_timeout=config.database.connection_timeout,
            replica_set=config.database.replica_set,
            auth_source=config.database.auth_source,
            ssl_mode=config.database.ssl_mode
        )
        
        # Setup database
        setup = DatabaseSetup(db_config)
        return await setup.initialize_database(force)
        
    except Exception as e:
        logging.error(f"Database setup failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database setup utility")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--force", action="store_true", help="Force recreate database")
    parser.add_argument("--backup", help="Create backup at specified path")
    parser.add_argument("--health-check", action="store_true", help="Perform health check")
    
    args = parser.parse_args()
    
    async def main():
        if args.health_check:
            config = get_config(args.config)
            db_config = DatabaseConfig(
                db_type=DatabaseType(config.database.type),
                host=config.database.host,
                port=config.database.port,
                username=config.database.username,
                password=config.database.password,
                database=config.database.database
            )
            setup = DatabaseSetup(db_config)
            health = await setup.health_check()
            print(json.dumps(health, indent=2, default=str))
            return
        
        if args.backup:
            config = get_config(args.config)
            db_config = DatabaseConfig(
                db_type=DatabaseType(config.database.type),
                host=config.database.host,
                port=config.database.port,
                username=config.database.username,
                password=config.database.password,
                database=config.database.database
            )
            setup = DatabaseSetup(db_config)
            success = await setup.backup_database(args.backup)
            sys.exit(0 if success else 1)
        
        success = await setup_database_from_config(args.config, args.force)
        sys.exit(0 if success else 1)
    
    asyncio.run(main()) 