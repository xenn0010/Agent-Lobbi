"""
Security and authentication system for the agent ecosystem.
Provides JWT authentication, role-based access control, rate limiting, and security monitoring.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

# JWT and cryptography
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt
import base64

# Pydantic for data validation
from pydantic import BaseModel, Field, validator
from typing import Optional


class Permission(Enum):
    """System permissions"""
    # Agent permissions
    REGISTER_AGENT = "register_agent"
    SEND_MESSAGE = "send_message"
    RECEIVE_MESSAGE = "receive_message"
    INITIATE_INTERACTION = "initiate_interaction"
    RESPOND_TO_INTERACTION = "respond_to_interaction"
    
    # Lobby permissions
    MANAGE_AGENTS = "manage_agents"
    VIEW_SYSTEM_STATUS = "view_system_status"
    MANAGE_CONVERSATIONS = "manage_conversations"
    VIEW_METRICS = "view_metrics"
    
    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    SYSTEM_ADMIN = "system_admin"
    VIEW_LOGS = "view_logs"


class Role(Enum):
    """System roles with associated permissions"""
    AGENT = "agent"
    LOBBY = "lobby"
    ADMIN = "admin"
    VIEWER = "viewer"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.AGENT: {
        Permission.REGISTER_AGENT,
        Permission.SEND_MESSAGE,
        Permission.RECEIVE_MESSAGE,
        Permission.INITIATE_INTERACTION,
        Permission.RESPOND_TO_INTERACTION
    },
    Role.LOBBY: {
        Permission.MANAGE_AGENTS,
        Permission.VIEW_SYSTEM_STATUS,
        Permission.MANAGE_CONVERSATIONS,
        Permission.SEND_MESSAGE,
        Permission.RECEIVE_MESSAGE
    },
    Role.ADMIN: {
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.SYSTEM_ADMIN,
        Permission.VIEW_LOGS,
        Permission.VIEW_METRICS,
        Permission.MANAGE_AGENTS,
        Permission.VIEW_SYSTEM_STATUS,
        Permission.MANAGE_CONVERSATIONS
    },
    Role.VIEWER: {
        Permission.VIEW_SYSTEM_STATUS,
        Permission.VIEW_METRICS
    }
}


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # JWT settings
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 7
    
    # Encryption settings
    encryption_key: Optional[bytes] = None
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # Password settings
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    
    # Session settings
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5
    
    # Security monitoring
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    def __post_init__(self):
        if self.encryption_key is None:
            # Generate a new encryption key
            password = self.jwt_secret_key.encode()
            salt = b'agent_ecosystem_salt'  # In production, use a random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.encryption_key = key


class User(BaseModel):
    """User model for authentication"""
    user_id: str
    username: str
    email: Optional[str] = None
    password_hash: str
    roles: List[Role] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False
    
    def has_role(self, role: Role) -> bool:
        """Check if user has a specific role"""
        return role in self.roles
    
    def is_locked(self) -> bool:
        """Check if user account is locked"""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until


class AuthToken(BaseModel):
    """Authentication token model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    permissions: List[str]


class RateLimitEntry:
    """Rate limiting entry"""
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
    
    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit"""
        now = time.time()
        
        # Remove old requests outside the window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.window_seconds]
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def time_until_reset(self) -> float:
        """Get time until rate limit resets"""
        if not self.requests:
            return 0.0
        
        oldest_request = min(self.requests)
        return max(0.0, self.window_seconds - (time.time() - oldest_request))


class SecurityEvent(BaseModel):
    """Security event for monitoring"""
    event_id: str
    event_type: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical


class SecurityManager:
    """Central security manager"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self.fernet = Fernet(config.encryption_key)
        
        # Rate limiting
        self.rate_limits: Dict[str, RateLimitEntry] = {}
        
        # Active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Security events
        self.security_events: List[SecurityEvent] = []
        
        # User storage (in production, this would be a database)
        self.users: Dict[str, User] = {}
        
        # Initialize default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_password = "admin123"  # In production, generate a secure password
        password_hash = self.hash_password(admin_password)
        
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@agent-ecosystem.local",
            password_hash=password_hash,
            roles=[Role.ADMIN]
        )
        
        self.users["admin"] = admin_user
        self.logger.info("Created default admin user (username: admin, password: admin123)")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> tuple[bool, List[str]]:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_user(self, username: str, password: str, email: Optional[str] = None, 
                   roles: List[Role] = None) -> tuple[bool, str]:
        """Create a new user"""
        try:
            # Check if user already exists
            if any(user.username == username for user in self.users.values()):
                return False, "Username already exists"
            
            # Validate password
            is_valid, errors = self.validate_password_strength(password)
            if not is_valid:
                return False, "; ".join(errors)
            
            # Create user
            user_id = secrets.token_urlsafe(16)
            password_hash = self.hash_password(password)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles or [Role.AGENT]
            )
            
            self.users[user_id] = user
            
            # Log security event
            self.log_security_event(
                event_type="user_created",
                user_id=user_id,
                details={"username": username, "roles": [r.value for r in user.roles]}
            )
            
            return True, user_id
            
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return False, str(e)
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: Optional[str] = None) -> tuple[bool, Optional[User], str]:
        """Authenticate a user"""
        try:
            # Find user by username
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                self.log_security_event(
                    event_type="login_failed",
                    details={"username": username, "reason": "user_not_found"},
                    ip_address=ip_address,
                    severity="warning"
                )
                return False, None, "Invalid username or password"
            
            # Check if user is locked
            if user.is_locked():
                self.log_security_event(
                    event_type="login_failed",
                    user_id=user.user_id,
                    details={"username": username, "reason": "account_locked"},
                    ip_address=ip_address,
                    severity="warning"
                )
                return False, None, "Account is locked due to too many failed attempts"
            
            # Check if user is active
            if not user.is_active:
                self.log_security_event(
                    event_type="login_failed",
                    user_id=user.user_id,
                    details={"username": username, "reason": "account_inactive"},
                    ip_address=ip_address,
                    severity="warning"
                )
                return False, None, "Account is inactive"
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock account if too many failures
                if user.failed_login_attempts >= self.config.max_failed_attempts:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(
                        minutes=self.config.lockout_duration_minutes
                    )
                
                self.log_security_event(
                    event_type="login_failed",
                    user_id=user.user_id,
                    details={
                        "username": username, 
                        "reason": "invalid_password",
                        "failed_attempts": user.failed_login_attempts
                    },
                    ip_address=ip_address,
                    severity="warning"
                )
                
                return False, None, "Invalid username or password"
            
            # Successful authentication
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(timezone.utc)
            
            self.log_security_event(
                event_type="login_success",
                user_id=user.user_id,
                details={"username": username},
                ip_address=ip_address
            )
            
            return True, user, "Authentication successful"
            
        except Exception as e:
            self.logger.error(f"Error during authentication: {e}")
            return False, None, "Authentication error"
    
    def generate_tokens(self, user: User) -> AuthToken:
        """Generate JWT access and refresh tokens"""
        now = datetime.now(timezone.utc)
        
        # Access token payload
        access_payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for role in user.roles 
                          for perm in ROLE_PERMISSIONS.get(role, set())],
            "iat": now,
            "exp": now + timedelta(hours=self.config.jwt_expiration_hours),
            "type": "access"
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user.user_id,
            "iat": now,
            "exp": now + timedelta(days=self.config.jwt_refresh_expiration_days),
            "type": "refresh"
        }
        
        # Generate tokens
        access_token = jwt.encode(
            access_payload, 
            self.config.jwt_secret_key, 
            algorithm=self.config.jwt_algorithm
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            "user_id": user.user_id,
            "created_at": now,
            "last_activity": now,
            "access_token": access_token,
            "refresh_token": refresh_token
        }
        
        return AuthToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.jwt_expiration_hours * 3600,
            user_id=user.user_id,
            permissions=[perm.value for role in user.roles 
                        for perm in ROLE_PERMISSIONS.get(role, set())]
        )
    
    def verify_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]], str]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != "access":
                return False, None, "Invalid token type"
            
            # Check if user still exists and is active
            user_id = payload.get("user_id")
            user = self.users.get(user_id)
            
            if not user or not user.is_active:
                return False, None, "User not found or inactive"
            
            return True, payload, "Token valid"
            
        except jwt.ExpiredSignatureError:
            return False, None, "Token has expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error verifying token: {e}")
            return False, None, "Token verification error"
    
    def refresh_token(self, refresh_token: str) -> tuple[bool, Optional[AuthToken], str]:
        """Refresh an access token using a refresh token"""
        try:
            payload = jwt.decode(
                refresh_token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check token type
            if payload.get("type") != "refresh":
                return False, None, "Invalid token type"
            
            # Get user
            user_id = payload.get("user_id")
            user = self.users.get(user_id)
            
            if not user or not user.is_active:
                return False, None, "User not found or inactive"
            
            # Generate new tokens
            new_tokens = self.generate_tokens(user)
            
            self.log_security_event(
                event_type="token_refreshed",
                user_id=user_id
            )
            
            return True, new_tokens, "Token refreshed successfully"
            
        except jwt.ExpiredSignatureError:
            return False, None, "Refresh token has expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid refresh token: {str(e)}"
        except Exception as e:
            self.logger.error(f"Error refreshing token: {e}")
            return False, None, "Token refresh error"
    
    def check_rate_limit(self, identifier: str) -> tuple[bool, float]:
        """Check rate limit for an identifier (IP, user, etc.)"""
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = RateLimitEntry(
                self.config.rate_limit_requests,
                self.config.rate_limit_window_seconds
            )
        
        rate_limit = self.rate_limits[identifier]
        is_allowed = rate_limit.is_allowed()
        time_until_reset = rate_limit.time_until_reset()
        
        if not is_allowed:
            self.log_security_event(
                event_type="rate_limit_exceeded",
                details={"identifier": identifier},
                severity="warning"
            )
        
        return is_allowed, time_until_reset
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def log_security_event(self, event_type: str, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                          details: Dict[str, Any] = None, severity: str = "info"):
        """Log a security event"""
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Log to system logger
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.INFO)
        
        self.logger.log(
            log_level,
            f"Security event: {event_type}",
            extra={
                "event_id": event.event_id,
                "user_id": user_id,
                "ip_address": ip_address,
                "details": details
            }
        )
        
        # Keep only recent events (last 1000)
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def get_security_events(self, limit: int = 100, 
                           event_type: Optional[str] = None,
                           user_id: Optional[str] = None,
                           severity: Optional[str] = None) -> List[SecurityEvent]:
        """Get security events with optional filtering"""
        events = self.security_events
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now(timezone.utc)
        timeout = timedelta(minutes=self.config.session_timeout_minutes)
        
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if now - session["last_activity"] > timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            self.log_security_event(
                event_type="sessions_cleaned",
                details={"expired_count": len(expired_sessions)}
            )
    
    def revoke_user_sessions(self, user_id: str):
        """Revoke all sessions for a user"""
        revoked_sessions = []
        for session_id, session in list(self.active_sessions.items()):
            if session["user_id"] == user_id:
                del self.active_sessions[session_id]
                revoked_sessions.append(session_id)
        
        if revoked_sessions:
            self.log_security_event(
                event_type="user_sessions_revoked",
                user_id=user_id,
                details={"revoked_count": len(revoked_sessions)}
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security system status"""
        now = datetime.now(timezone.utc)
        
        # Count events by severity in last 24 hours
        recent_events = [e for e in self.security_events 
                        if now - e.timestamp < timedelta(hours=24)]
        
        event_counts = {}
        for event in recent_events:
            event_counts[event.severity] = event_counts.get(event.severity, 0) + 1
        
        return {
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "locked_users": len([u for u in self.users.values() if u.is_locked()]),
            "active_sessions": len(self.active_sessions),
            "rate_limited_identifiers": len(self.rate_limits),
            "recent_events_24h": len(recent_events),
            "event_counts_24h": event_counts,
            "last_cleanup": now.isoformat()
        }


# Decorator for requiring authentication
def require_auth(permission: Optional[Permission] = None):
    """Decorator to require authentication and optionally a specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be implemented based on your web framework
            # For now, it's a placeholder showing the concept
            
            # Extract token from request headers
            # Verify token
            # Check permissions
            # Call original function if authorized
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Factory function for easy security manager creation
def create_security_manager(
    jwt_secret_key: str,
    jwt_expiration_hours: int = 24,
    rate_limit_requests: int = 100,
    **kwargs
) -> SecurityManager:
    """Factory function to create a security manager"""
    
    config = SecurityConfig(
        jwt_secret_key=jwt_secret_key,
        jwt_expiration_hours=jwt_expiration_hours,
        rate_limit_requests=rate_limit_requests,
        **kwargs
    )
    
    return SecurityManager(config) 