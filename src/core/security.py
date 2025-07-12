#!/usr/bin/env python3
"""
Enterprise Security Module for Agent Lobbi
Implements authentication, rate limiting, encryption, audit logging, and CORS
"""

import asyncio
import hashlib
import hmac
try:
    import jwt
except ImportError:
    # Fallback - disable JWT functionality
    jwt = None
import time
import json
import ssl
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import secrets
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from collections import defaultdict, deque
import structlog
import re
import ipaddress
from urllib.parse import urlparse
import os

# Try to import monitoring - graceful degradation if not available
try:
    from .monitoring import get_metrics
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

logger = structlog.get_logger(__name__)

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthMethod(Enum):
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    MUTUAL_TLS = "mutual_tls"

@dataclass
class SecurityConfig:
    """Comprehensive security configuration"""
    # Authentication
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(64))
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    api_key_length: int = 32
    require_auth: bool = True
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    global_rate_limit: int = 1000  # requests per minute
    per_agent_rate_limit: int = 100  # requests per minute
    burst_allowance: int = 50  # burst capacity
    
    # Encryption
    enable_encryption: bool = True
    encryption_key: Optional[bytes] = None
    
    # TLS/SSL
    tls_enabled: bool = True
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Security Headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    })
    
    # Audit Logging
    audit_enabled: bool = True
    audit_log_file: str = "security_audit.log"
    
    # Input Validation
    max_message_size: int = 1024 * 1024  # 1MB
    max_payload_depth: int = 10
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"<script.*?>.*?</script>",  # XSS
        r"javascript:",              # XSS
        r"on\w+\s*=",               # Event handlers
        r"eval\s*\(",               # Code injection
        r"exec\s*\(",               # Code injection
    ])

class SecurityAuditEvent:
    """Security audit event"""
    def __init__(self, event_type: str, user_id: str, ip_address: str, 
                 details: Dict[str, Any], risk_level: SecurityLevel = SecurityLevel.LOW):
        self.event_type = event_type
        self.user_id = user_id
        self.ip_address = ip_address
        self.details = details
        self.risk_level = risk_level
        self.timestamp = datetime.now(timezone.utc)

class RateLimiter:
    """Token bucket rate limiter with burst support"""
    
    def __init__(self, rate_limit: int, burst_capacity: int = None):
        self.rate_limit = rate_limit  # tokens per minute
        self.burst_capacity = burst_capacity or rate_limit
        self.buckets: Dict[str, Dict[str, Any]] = {}
        
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = time.time()
        
        # Initialize bucket if it doesn't exist
        if identifier not in self.buckets:
            self.buckets[identifier] = {
                'tokens': self.burst_capacity,
                'last_refill': now
            }
        
        bucket = self.buckets[identifier]
        
        # Calculate tokens to add since last refill
        time_passed = now - bucket['last_refill']
        tokens_to_add = (time_passed / 60.0) * self.rate_limit
        
        # Update bucket
        bucket['tokens'] = min(
            self.burst_capacity,
            bucket['tokens'] + tokens_to_add
        )
        bucket['last_refill'] = now
        
        # Check if request can be allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        
        return False
    
    def get_wait_time(self, identifier: str) -> float:
        """Get wait time until next request is allowed"""
        # Initialize bucket if it doesn't exist
        if identifier not in self.buckets:
            return 0.0
        
        bucket = self.buckets[identifier]
        if bucket['tokens'] >= 1:
            return 0.0
        
        # Calculate time until next token
        return (1 - bucket['tokens']) * (60.0 / self.rate_limit)

class InputValidator:
    """Advanced input validation and sanitization"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE) 
                               for pattern in config.blocked_patterns]
    
    def validate_message(self, message: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate message for security threats"""
        try:
            # Check message size
            message_size = len(json.dumps(message))
            if message_size > self.config.max_message_size:
                return False, f"Message too large: {message_size} bytes"
            
            # Check payload depth
            if self._get_dict_depth(message) > self.config.max_payload_depth:
                return False, "Message payload too deeply nested"
            
            # Check for malicious patterns
            message_str = json.dumps(message)
            for pattern in self.blocked_patterns:
                if pattern.search(message_str):
                    return False, f"Malicious pattern detected: {pattern.pattern}"
            
            # Validate required fields
            required_fields = ['sender_id', 'receiver_id', 'message_type']
            for field in required_fields:
                if field not in message:
                    return False, f"Missing required field: {field}"
                if not isinstance(message[field], str) or not message[field].strip():
                    return False, f"Invalid {field}: must be non-empty string"
            
            # Validate sender_id format (alphanumeric + underscore + dash)
            if not re.match(r'^[a-zA-Z0-9_-]+$', message['sender_id']):
                return False, "Invalid sender_id format"
            
            # Validate message_type
            valid_types = [
                'REGISTER', 'REQUEST', 'RESPONSE', 'ACTION', 'INFO', 
                'ERROR', 'DISCOVER_SERVICES', 'BROADCAST', 'ACK'
            ]
            if message['message_type'] not in valid_types:
                return False, f"Invalid message_type: {message['message_type']}"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def sanitize_string(self, text: str) -> str:
        """Sanitize string input"""
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        text = text[:1000]
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\n', '\r', '\t']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        return text.strip()
    
    def _get_dict_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionaries"""
        if not isinstance(obj, dict):
            return depth
        
        if not obj:
            return depth + 1
        
        return max(self._get_dict_depth(value, depth + 1) for value in obj.values())

class EncryptionManager:
    """Enterprise-grade encryption for all sensitive data"""

    KEY_FILE = "encryption.key"
    
    def __init__(self, config: SecurityConfig):
        """Initialize encryption with a secure key from env var or file."""
        self.config = config
        self.fernet = None
        
        if config.enable_encryption:
            self.key = self._load_or_generate_key(config.encryption_key)
            self.fernet = Fernet(self.key)

    def _load_or_generate_key(self, config_key: Optional[bytes]) -> bytes:
        """Loads key from env var, config, file, or generates a new one."""
        # 1. From environment variable (highest priority)
        env_key = os.environ.get("ENCRYPTION_KEY")
        if env_key:
            logger.info("Loading encryption key from environment variable.")
            return env_key.encode()

        # 2. From security config
        if config_key:
            logger.info("Loading encryption key from security config.")
            return config_key

        # 3. From key file
        if os.path.exists(self.KEY_FILE):
            logger.info(f"Loading encryption key from file: {self.KEY_FILE}")
            with open(self.KEY_FILE, "rb") as f:
                return f.read()

        # 4. Generate, save, and return a new key
        logger.warning(
            f"No persistent key found. Generating new key and saving to {self.KEY_FILE}"
        )
        logger.warning(
            "IMPORTANT: For production, set the ENCRYPTION_KEY environment variable for better security."
        )
        new_key = Fernet.generate_key()
        try:
            with open(self.KEY_FILE, "wb") as f:
                f.write(new_key)
            logger.info(f"New key saved successfully to {self.KEY_FILE}")
        except IOError as e:
            logger.error(
                f"Failed to save new encryption key to {self.KEY_FILE}", error=str(e)
            )
            logger.error("The key will be ephemeral for this session only.")

        return new_key
    
    def encrypt_message(self, message: str) -> str:
        """Encrypt message payload"""
        if not self.fernet:
            return message
        
        try:
            encrypted = self.fernet.encrypt(message.encode())
            return encrypted.decode()
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            return message
    
    def decrypt_message(self, encrypted_message: str) -> str:
        """Decrypt message payload"""
        if not self.fernet:
            return encrypted_message
        
        try:
            decrypted = self.fernet.decrypt(encrypted_message.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            return encrypted_message

class AuthenticationManager:
    """Multi-method authentication system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.jwt_secret = config.jwt_secret
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Dict[str, datetime] = {}
        
    def generate_api_key(self, agent_id: str, permissions: List[str] = None) -> str:
        """Generate new API key for agent"""
        api_key = secrets.token_urlsafe(self.config.api_key_length)
        
        self.api_keys[api_key] = {
            'agent_id': agent_id,
            'permissions': permissions or ['read', 'write'],
            'created_at': datetime.now(timezone.utc),
            'last_used': None,
            'active': True
        }
        
        logger.info("Generated API key", agent_id=agent_id)
        return api_key
    
    def generate_jwt_token(self, agent_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token for agent"""
        if jwt is None:
            # JWT not available - return API key instead
            return self.generate_api_key(agent_id, permissions)
        
        payload = {
            'agent_id': agent_id,
            'permissions': permissions or ['read', 'write'],
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(hours=self.config.jwt_expiry_hours)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.config.jwt_algorithm)
        logger.info("Generated JWT token", agent_id=agent_id)
        return token
    
    def validate_api_key(self, api_key: str, ip_address: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate API key"""
        if self._is_ip_blocked(ip_address):
            return False, "IP address blocked", None
        
        if api_key not in self.api_keys:
            self._record_failed_attempt(ip_address)
            return False, "Invalid API key", None
        
        key_data = self.api_keys[api_key]
        if not key_data['active']:
            return False, "API key disabled", None
        
        # Update last used
        key_data['last_used'] = datetime.now(timezone.utc)
        
        return True, None, key_data
    
    def validate_jwt_token(self, token: str, ip_address: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Validate JWT token"""
        if jwt is None:
            # JWT not available - try as API key instead
            return self.validate_api_key(token, ip_address)
        
        if self._is_ip_blocked(ip_address):
            return False, "IP address blocked", None
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.config.jwt_algorithm])
            return True, None, payload
        except jwt.ExpiredSignatureError:
            self._record_failed_attempt(ip_address)
            return False, "Token expired", None
        except jwt.InvalidTokenError:
            self._record_failed_attempt(ip_address)
            return False, "Invalid token", None
    
    def _record_failed_attempt(self, ip_address: str):
        """Record failed authentication attempt"""
        now = datetime.now(timezone.utc)
        self.failed_attempts[ip_address].append(now)
        
        # Clean old attempts (older than 1 hour)
        self.failed_attempts[ip_address] = [
            attempt for attempt in self.failed_attempts[ip_address]
            if now - attempt < timedelta(hours=1)
        ]
        
        # Block IP if too many failures
        if len(self.failed_attempts[ip_address]) >= 5:
            self.blocked_ips[ip_address] = now + timedelta(hours=1)
            logger.warning("IP blocked due to failed attempts", ip=ip_address)
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        if ip_address in self.blocked_ips:
            if datetime.now(timezone.utc) > self.blocked_ips[ip_address]:
                del self.blocked_ips[ip_address]
                return False
            return True
        return False

class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_events: deque = deque(maxlen=10000)
        self.log_file = None
        
        if config.audit_enabled:
            try:
                self.log_file = open(config.audit_log_file, 'a', encoding='utf-8')
            except Exception as e:
                logger.error("Failed to open audit log file", error=str(e))
    
    def log_event(self, event: SecurityAuditEvent):
        """Log security audit event"""
        self.audit_events.append(event)
        
        if self.log_file:
            try:
                audit_entry = {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'ip_address': event.ip_address,
                    'risk_level': event.risk_level.value,
                    'details': event.details
                }
                
                self.log_file.write(json.dumps(audit_entry) + '\n')
                self.log_file.flush()
                
                # Log high-risk events to main logger
                if event.risk_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                    logger.warning("High-risk security event", **audit_entry)
                    
            except Exception as e:
                logger.error("Failed to write audit log", error=str(e))
    
    def get_recent_events(self, count: int = 100, risk_level: SecurityLevel = None) -> List[SecurityAuditEvent]:
        """Get recent audit events"""
        events = list(self.audit_events)
        
        if risk_level:
            events = [e for e in events if e.risk_level == risk_level]
        
        return events[-count:]
    
    def close(self):
        """Close audit log file"""
        if self.log_file:
            self.log_file.close()

class SecurityManager:
    """Comprehensive security management system"""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.rate_limiter = RateLimiter(
            self.config.global_rate_limit,
            self.config.burst_allowance
        )
        self.per_agent_limiters: Dict[str, RateLimiter] = {}
        self.input_validator = InputValidator(self.config)
        self.encryption_manager = EncryptionManager(self.config)
        self.auth_manager = AuthenticationManager(self.config)
        self.audit_logger = SecurityAuditLogger(self.config)
        
        # Initialize metrics if available
        self.metrics = get_metrics() if MONITORING_AVAILABLE else None
        
    def check_rate_limit(self, identifier: str, agent_id: str = None) -> Tuple[bool, float]:
        """Check rate limits for global and per-agent"""
        # Check global rate limit
        if not self.rate_limiter.is_allowed(identifier):
            wait_time = self.rate_limiter.get_wait_time(identifier)
            if self.metrics:
                self.metrics.record_rate_limit_hit("global", "global")
                self.metrics.record_rate_limit_wait("global", wait_time)
            return False, wait_time
        
        # Check per-agent rate limit
        if agent_id:
            if agent_id not in self.per_agent_limiters:
                self.per_agent_limiters[agent_id] = RateLimiter(
                    self.config.per_agent_rate_limit,
                    self.config.burst_allowance
                )
            
            if not self.per_agent_limiters[agent_id].is_allowed(agent_id):
                wait_time = self.per_agent_limiters[agent_id].get_wait_time(agent_id)
                if self.metrics:
                    self.metrics.record_rate_limit_hit("per_agent", agent_id)
                    self.metrics.record_rate_limit_wait("per_agent", wait_time)
                return False, wait_time
        
        return True, 0.0
    
    def validate_and_sanitize_message(self, message: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Validate and sanitize incoming message"""
        start_time = time.time()
        
        # Validate message
        valid, error = self.input_validator.validate_message(message)
        if not valid:
            if self.metrics:
                self.metrics.record_validation_check("message", "failed", time.time() - start_time)
            return False, error, message
        
        # Sanitize string fields
        sanitized = message.copy()
        for key, value in sanitized.items():
            if isinstance(value, str):
                sanitized[key] = self.input_validator.sanitize_string(value)
        
        if self.metrics:
            self.metrics.record_validation_check("message", "success", time.time() - start_time)
        
        return True, None, sanitized
    
    def authenticate_request(self, auth_header: str, ip_address: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Authenticate request using multiple methods"""
        if not self.config.require_auth:
            return True, None, {'agent_id': 'anonymous', 'permissions': ['read']}
        
        if not auth_header:
            return False, "Authentication required", None
        
        # Try different authentication methods
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            valid, error, data = self.auth_manager.validate_jwt_token(token, ip_address)
            if self.metrics:
                agent_id = data.get('agent_id', 'unknown') if data else 'unknown'
                result = "success" if valid else "failed"
                self.metrics.record_auth_attempt("jwt", result, agent_id)
                if not valid:
                    self.metrics.record_auth_failure("jwt", error or "unknown", ip_address)
            return valid, error, data
        elif auth_header.startswith('ApiKey '):
            api_key = auth_header[7:]
            valid, error, data = self.auth_manager.validate_api_key(api_key, ip_address)
            if self.metrics:
                agent_id = data.get('agent_id', 'unknown') if data else 'unknown'
                result = "success" if valid else "failed"
                self.metrics.record_auth_attempt("api_key", result, agent_id)
                if not valid:
                    self.metrics.record_auth_failure("api_key", error or "unknown", ip_address)
            return valid, error, data
        else:
            return False, "Invalid authentication method", None
    
    def get_cors_headers(self, origin: str = None) -> Dict[str, str]:
        """Get CORS headers for response"""
        if not self.config.cors_enabled:
            return {}
        
        headers = {}
        
        # Check if origin is allowed
        if origin and (self.config.cors_origins == ["*"] or origin in self.config.cors_origins):
            headers['Access-Control-Allow-Origin'] = origin
        elif "*" in self.config.cors_origins:
            headers['Access-Control-Allow-Origin'] = "*"
        
        headers['Access-Control-Allow-Methods'] = ", ".join(self.config.cors_methods)
        headers['Access-Control-Allow-Headers'] = ", ".join(self.config.cors_headers)
        headers['Access-Control-Max-Age'] = "86400"
        
        return headers
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for response"""
        return self.config.security_headers.copy()
    
    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure connections"""
        if not self.config.tls_enabled or not self.config.cert_file:
            return None
        
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(self.config.cert_file, self.config.key_file)
            
            if self.config.ca_file:
                context.load_verify_locations(self.config.ca_file)
                context.verify_mode = ssl.CERT_REQUIRED
            
            return context
        except Exception as e:
            logger.error("Failed to create SSL context", error=str(e))
            return None
    
    def log_security_event(self, event_type: str, user_id: str, ip_address: str, 
                          details: Dict[str, Any], risk_level: SecurityLevel = SecurityLevel.LOW):
        """Log security event"""
        event = SecurityAuditEvent(event_type, user_id, ip_address, details, risk_level)
        self.audit_logger.log_event(event)
        
        # Record metrics
        if self.metrics:
            self.metrics.record_security_event(event_type, risk_level.value)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            'config': {
                'auth_required': self.config.require_auth,
                'rate_limiting_enabled': self.config.rate_limit_enabled,
                'encryption_enabled': self.config.enable_encryption,
                'tls_enabled': self.config.tls_enabled,
                'cors_enabled': self.config.cors_enabled,
                'audit_enabled': self.config.audit_enabled
            },
            'stats': {
                'blocked_ips': len(self.auth_manager.blocked_ips),
                'active_api_keys': len([k for k in self.auth_manager.api_keys.values() if k['active']]),
                'recent_audit_events': len(self.audit_logger.audit_events),
                'high_risk_events': len([e for e in self.audit_logger.audit_events 
                                       if e.risk_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]])
            },
            'health': {
                'rate_limiter_active': True,
                'audit_logger_active': self.audit_logger.log_file is not None,
                'encryption_active': self.encryption_manager.fernet is not None
            }
        }
    
    def cleanup(self):
        """Cleanup security resources"""
        self.audit_logger.close()

# Global security manager instance
security_manager = SecurityManager() 