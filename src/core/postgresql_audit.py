#!/usr/bin/env python3
"""
PostgreSQL Audit Logging System for Agent Lobbi
Enterprise-grade audit logging with database persistence for compliance
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
import structlog
from dataclasses import dataclass

from .security import SecurityAuditEvent, SecurityLevel

logger = structlog.get_logger(__name__)

@dataclass
class PostgreSQLConfig:
    """PostgreSQL configuration for audit logging"""
    host: str = "localhost"
    port: int = 5432
    database: str = "agent_lobby_audit"
    username: str = "lobby_user"
    password: str = "secure_password"
    pool_min_conn: int = 1
    pool_max_conn: int = 20

class PostgreSQLAuditLogger:
    """Enterprise PostgreSQL audit logging system"""
    
    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self.connection_pool = None
        self.connected = False
        
    def initialize(self):
        """Initialize PostgreSQL connection and create tables"""
        try:
            # Create connection pool
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                self.config.pool_min_conn,
                self.config.pool_max_conn,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                cursor_factory=RealDictCursor
            )
            
            # Create audit tables if they don't exist
            self._create_audit_tables()
            
            self.connected = True
            logger.info("PostgreSQL audit logger initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize PostgreSQL audit logger", error=str(e))
            self.connected = False
            # Don't raise - allow fallback to file logging
    
    def _create_audit_tables(self):
        """Create audit logging tables"""
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS security_audit_events (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            event_type VARCHAR(100) NOT NULL,
            user_id VARCHAR(100) NOT NULL,
            ip_address INET NOT NULL,
            risk_level VARCHAR(20) NOT NULL,
            details JSONB NOT NULL,
            session_id VARCHAR(100),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON security_audit_events (timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_event_type ON security_audit_events (event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_risk_level ON security_audit_events (risk_level);
        
        CREATE TABLE IF NOT EXISTS auth_audit_events (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            agent_id VARCHAR(100) NOT NULL,
            auth_method VARCHAR(50) NOT NULL,
            success BOOLEAN NOT NULL,
            ip_address INET NOT NULL,
            failure_reason VARCHAR(200),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_auth_timestamp ON auth_audit_events (timestamp);
        CREATE INDEX IF NOT EXISTS idx_auth_agent_id ON auth_audit_events (agent_id);
        """
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(create_tables_sql)
                conn.commit()
                logger.info("Audit tables created successfully")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Failed to create audit tables", error=str(e))
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def log_security_event(self, event: SecurityAuditEvent, session_id: str = None):
        """Log security audit event to PostgreSQL"""
        if not self.connected:
            logger.warning("PostgreSQL audit logger not connected, skipping")
            return
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO security_audit_events 
                    (timestamp, event_type, user_id, ip_address, risk_level, details, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    event.timestamp,
                    event.event_type,
                    event.user_id,
                    event.ip_address,
                    event.risk_level.value,
                    json.dumps(event.details),
                    session_id
                ))
                conn.commit()
                logger.debug("Security event logged to PostgreSQL", event_type=event.event_type)
                
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Failed to log security event to PostgreSQL", error=str(e))
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def log_auth_event(self, agent_id: str, auth_method: str, success: bool, 
                      ip_address: str, failure_reason: str = None):
        """Log authentication event"""
        if not self.connected:
            return
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO auth_audit_events 
                    (agent_id, auth_method, success, ip_address, failure_reason)
                    VALUES (%s, %s, %s, %s, %s)
                """, (agent_id, auth_method, success, ip_address, failure_reason))
                conn.commit()
                
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("Failed to log auth event", error=str(e))
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def get_security_events(self, start_time: datetime = None, end_time: datetime = None,
                           event_types: List[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get security events with filters"""
        if not self.connected:
            return []
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            with conn.cursor() as cursor:
                where_conditions = []
                params = []
                
                if start_time:
                    where_conditions.append("timestamp >= %s")
                    params.append(start_time)
                
                if end_time:
                    where_conditions.append("timestamp <= %s")
                    params.append(end_time)
                
                if event_types:
                    where_conditions.append("event_type = ANY(%s)")
                    params.append(event_types)
                
                where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                query = f"""
                    SELECT id, timestamp, event_type, user_id, ip_address, risk_level, details
                    FROM security_audit_events
                    {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT %s
                """
                params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            logger.error("Failed to get security events", error=str(e))
            return []
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def close(self):
        """Close all database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connected = False
            logger.info("PostgreSQL audit logger closed")

# Database connection utility
def create_postgresql_audit_logger(config: PostgreSQLConfig = None) -> PostgreSQLAuditLogger:
    """Create and initialize PostgreSQL audit logger"""
    config = config or PostgreSQLConfig()
    logger_instance = PostgreSQLAuditLogger(config)
    logger_instance.initialize()
    return logger_instance 