"""
Data Protection Layer
Prevents unauthorized access, rival company data theft, and trojan horse exploits
Honest implementation focused on practical security measures
"""

import asyncio
import json
import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import sqlite3
import re

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class DataAccessRule:
    """Rule defining who can access what data"""
    owner_agent: str
    data_type: str
    classification: DataClassification
    allowed_agents: Set[str]
    access_level: AccessLevel
    expiry_time: Optional[str] = None
    purpose: str = ""
    audit_required: bool = True


@dataclass
class AccessAttempt:
    """Record of data access attempt"""
    timestamp: str
    requesting_agent: str
    target_agent: str
    data_type: str
    access_granted: bool
    denial_reason: str = ""
    ip_address: str = ""
    user_agent: str = ""


class DataProtectionLayer:
    """
    Practical data protection - no overhyped security theater
    Focus on real threats: unauthorized access, data theft, malicious agents
    """
    
    def __init__(self, db_path: str = "data_protection.db"):
        self.db_path = db_path
        self.access_rules: Dict[str, List[DataAccessRule]] = {}
        self.access_attempts: List[AccessAttempt] = []
        self.banned_agents: Set[str] = set()
        self.suspicious_patterns: Dict[str, List[Dict]] = {}
        
        # Security thresholds - practical values
        self.max_failed_attempts = 5
        self.max_access_frequency = 50  # requests per minute
        self.suspicious_keywords = [
            'training_data', 'model_weights', 'proprietary', 'confidential',
            'api_key', 'secret', 'password', 'private_key'
        ]
        
        self._initialize_database()
        self._load_default_rules()
    
    def _initialize_database(self):
        """Initialize database for access control"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Access rules table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_agent TEXT,
                    data_type TEXT,
                    classification TEXT,
                    allowed_agents TEXT,
                    access_level INTEGER,
                    expiry_time TEXT,
                    purpose TEXT,
                    audit_required BOOLEAN,
                    created_at TEXT
                )
            ''')
            
            # Access attempts table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    requesting_agent TEXT,
                    target_agent TEXT,
                    data_type TEXT,
                    access_granted BOOLEAN,
                    denial_reason TEXT,
                    ip_address TEXT,
                    user_agent TEXT
                )
            ''')
            
            # Banned agents table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS banned_agents (
                    agent_id TEXT PRIMARY KEY,
                    reason TEXT,
                    banned_at TEXT,
                    banned_by TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Data protection database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize data protection database: {e}")
    
    def _load_default_rules(self):
        """Load sensible default access rules"""
        # Default: agents can only access their own data
        # Public data is accessible to all
        # Confidential data requires explicit permission
        pass
    
    async def register_agent_data(self, 
                                owner_agent: str,
                                data_type: str,
                                classification: DataClassification,
                                allowed_agents: Set[str] = None,
                                access_level: AccessLevel = AccessLevel.READ,
                                purpose: str = "") -> bool:
        """Register data and set access rules"""
        
        rule = DataAccessRule(
            owner_agent=owner_agent,
            data_type=data_type,
            classification=classification,
            allowed_agents=allowed_agents or set(),
            access_level=access_level,
            purpose=purpose
        )
        
        if owner_agent not in self.access_rules:
            self.access_rules[owner_agent] = []
        
        self.access_rules[owner_agent].append(rule)
        await self._save_access_rule(rule)
        
        logger.info(f"Registered data access rule: {owner_agent}/{data_type} "
                   f"({classification.value}, level: {access_level.name})")
        return True
    
    async def check_data_access(self, 
                              requesting_agent: str,
                              target_agent: str,
                              data_type: str,
                              request_context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Check if requesting_agent can access target_agent's data
        Returns (allowed, reason)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        context = request_context or {}
        
        # Log the access attempt
        attempt = AccessAttempt(
            timestamp=timestamp,
            requesting_agent=requesting_agent,
            target_agent=target_agent,
            data_type=data_type,
            access_granted=False,
            ip_address=context.get('ip_address', ''),
            user_agent=context.get('user_agent', '')
        )
        
        # Check if requesting agent is banned
        if requesting_agent in self.banned_agents:
            attempt.denial_reason = "Agent is banned"
            await self._log_access_attempt(attempt)
            return False, "Access denied: Agent is banned"
        
        # Check for suspicious behavior
        if await self._is_suspicious_request(requesting_agent, target_agent, data_type, context):
            attempt.denial_reason = "Suspicious behavior detected"
            await self._log_access_attempt(attempt)
            await self._record_suspicious_activity(requesting_agent, context)
            return False, "Access denied: Suspicious behavior detected"
        
        # Check access rules
        if target_agent in self.access_rules:
            for rule in self.access_rules[target_agent]:
                if rule.data_type == data_type or rule.data_type == "*":
                    # Check if access is allowed
                    if self._check_rule_permission(rule, requesting_agent):
                        attempt.access_granted = True
                        await self._log_access_attempt(attempt)
                        logger.info(f"Data access granted: {requesting_agent} -> {target_agent}/{data_type}")
                        return True, "Access granted"
        
        # Default deny
        attempt.denial_reason = "No matching access rule found"
        await self._log_access_attempt(attempt)
        logger.warning(f"Data access denied: {requesting_agent} -> {target_agent}/{data_type}")
        return False, "Access denied: No permission found"
    
    def _check_rule_permission(self, rule: DataAccessRule, requesting_agent: str) -> bool:
        """Check if a specific rule allows access"""
        # Public data is accessible to all
        if rule.classification == DataClassification.PUBLIC:
            return True
        
        # Owner always has access
        if requesting_agent == rule.owner_agent:
            return True
        
        # Check explicit permissions
        if requesting_agent in rule.allowed_agents:
            # Check if rule has expired
            if rule.expiry_time:
                expiry = datetime.fromisoformat(rule.expiry_time)
                if datetime.now(timezone.utc) > expiry:
                    return False
            return True
        
        return False
    
    async def _is_suspicious_request(self, 
                                   requesting_agent: str,
                                   target_agent: str, 
                                   data_type: str,
                                   context: Dict[str, Any]) -> bool:
        """Detect suspicious access patterns"""
        
        # Check for suspicious keywords in data type
        for keyword in self.suspicious_keywords:
            if keyword.lower() in data_type.lower():
                logger.warning(f"Suspicious data type requested: {data_type} by {requesting_agent}")
                return True
        
        # Check request frequency (simple rate limiting)
        recent_requests = [
            attempt for attempt in self.access_attempts[-100:]  # Last 100 attempts
            if attempt.requesting_agent == requesting_agent and
            (datetime.now(timezone.utc) - datetime.fromisoformat(attempt.timestamp)).total_seconds() < 60
        ]
        
        if len(recent_requests) > self.max_access_frequency:
            logger.warning(f"High request frequency from {requesting_agent}: {len(recent_requests)} requests/min")
            return True
        
        # Check for mass data access attempts (potential data harvesting)
        unique_targets = set()
        for attempt in recent_requests:
            unique_targets.add(attempt.target_agent)
        
        if len(unique_targets) > 10:  # Accessing data from many different agents quickly
            logger.warning(f"Mass data access detected from {requesting_agent}: {len(unique_targets)} targets")
            return True
        
        # Check for pattern matching rival company behavior
        if await self._check_rival_company_patterns(requesting_agent, context):
            return True
        
        return False
    
    async def _check_rival_company_patterns(self, requesting_agent: str, context: Dict[str, Any]) -> bool:
        """Check for patterns indicating rival company data harvesting"""
        
        # Check if agent is requesting data types typically used for model training
        training_related_requests = [
            attempt for attempt in self.access_attempts[-50:]
            if attempt.requesting_agent == requesting_agent and
            any(keyword in attempt.data_type.lower() for keyword in ['dataset', 'training', 'model', 'weights'])
        ]
        
        if len(training_related_requests) > 5:
            logger.warning(f"Potential model training data harvesting by {requesting_agent}")
            return True
        
        # Check user agent for automated tools
        user_agent = context.get('user_agent', '').lower()
        suspicious_agents = ['bot', 'crawler', 'scraper', 'automated', 'python-requests', 'curl']
        if any(agent in user_agent for agent in suspicious_agents):
            logger.warning(f"Suspicious user agent from {requesting_agent}: {user_agent}")
            return True
        
        return False
    
    async def _record_suspicious_activity(self, agent_id: str, context: Dict[str, Any]):
        """Record suspicious activity for further analysis"""
        if agent_id not in self.suspicious_patterns:
            self.suspicious_patterns[agent_id] = []
        
        self.suspicious_patterns[agent_id].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': context,
            'severity': 'medium'
        })
        
        # Auto-ban if too many suspicious activities
        if len(self.suspicious_patterns[agent_id]) >= 3:
            await self.ban_agent(agent_id, "Multiple suspicious activities", "system")
    
    async def ban_agent(self, agent_id: str, reason: str, banned_by: str):
        """Ban an agent from accessing any data"""
        self.banned_agents.add(agent_id)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO banned_agents (agent_id, reason, banned_at, banned_by)
                VALUES (?, ?, ?, ?)
            ''', (agent_id, reason, datetime.now(timezone.utc).isoformat(), banned_by))
            conn.commit()
            conn.close()
            
            logger.warning(f"Agent banned: {agent_id} (reason: {reason}, by: {banned_by})")
            
        except Exception as e:
            logger.error(f"Failed to ban agent: {e}")
    
    async def grant_temporary_access(self, 
                                   owner_agent: str,
                                   requesting_agent: str,
                                   data_type: str,
                                   duration_hours: int = 24,
                                   purpose: str = ""):
        """Grant temporary access to specific data"""
        expiry_time = (datetime.now(timezone.utc) + timedelta(hours=duration_hours)).isoformat()
        
        rule = DataAccessRule(
            owner_agent=owner_agent,
            data_type=data_type,
            classification=DataClassification.INTERNAL,
            allowed_agents={requesting_agent},
            access_level=AccessLevel.READ,
            expiry_time=expiry_time,
            purpose=purpose
        )
        
        if owner_agent not in self.access_rules:
            self.access_rules[owner_agent] = []
        
        self.access_rules[owner_agent].append(rule)
        await self._save_access_rule(rule)
        
        logger.info(f"Temporary access granted: {requesting_agent} -> {owner_agent}/{data_type} "
                   f"(expires: {expiry_time})")
    
    async def _log_access_attempt(self, attempt: AccessAttempt):
        """Log access attempt to database and memory"""
        self.access_attempts.append(attempt)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO access_attempts 
                (timestamp, requesting_agent, target_agent, data_type, 
                 access_granted, denial_reason, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                attempt.timestamp,
                attempt.requesting_agent,
                attempt.target_agent,
                attempt.data_type,
                attempt.access_granted,
                attempt.denial_reason,
                attempt.ip_address,
                attempt.user_agent
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log access attempt: {e}")
    
    async def _save_access_rule(self, rule: DataAccessRule):
        """Save access rule to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO access_rules 
                (owner_agent, data_type, classification, allowed_agents, 
                 access_level, expiry_time, purpose, audit_required, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.owner_agent,
                rule.data_type,
                rule.classification.value,
                json.dumps(list(rule.allowed_agents)),
                rule.access_level.value,
                rule.expiry_time,
                rule.purpose,
                rule.audit_required,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save access rule: {e}")
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics"""
        total_attempts = len(self.access_attempts)
        granted = sum(1 for a in self.access_attempts if a.access_granted)
        denied = total_attempts - granted
        
        return {
            'total_access_attempts': total_attempts,
            'access_granted': granted,
            'access_denied': denied,
            'success_rate': (granted / total_attempts * 100) if total_attempts > 0 else 0,
            'banned_agents': len(self.banned_agents),
            'suspicious_patterns': len(self.suspicious_patterns),
            'active_rules': sum(len(rules) for rules in self.access_rules.values())
        }
    
    def get_agent_access_history(self, agent_id: str, limit: int = 50) -> List[AccessAttempt]:
        """Get access history for a specific agent"""
        return [
            attempt for attempt in self.access_attempts[-limit:]
            if attempt.requesting_agent == agent_id or attempt.target_agent == agent_id
        ] 