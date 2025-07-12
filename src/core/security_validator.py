#!/usr/bin/env python3
"""
Security Validation Module for Agent Lobbi
Provides comprehensive input validation, sanitization, and security checks
"""

import re
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import urllib.parse

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    security_warnings: List[str] = None
    
    def __post_init__(self):
        if self.security_warnings is None:
            self.security_warnings = []

class SecurityValidator:
    """Comprehensive security validation for Agent Lobbi"""
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',               # JavaScript protocol
        r'vbscript:',                # VBScript protocol
        r'on\w+\s*=',               # Event handlers
        r'expression\s*\(',         # CSS expressions
        r'url\s*\(',                # CSS urls
        r'@import',                 # CSS imports
        r'<iframe[^>]*>',           # iframes
        r'<object[^>]*>',           # objects
        r'<embed[^>]*>',            # embeds
        r'<link[^>]*>',             # links
        r'<meta[^>]*>',             # meta tags
    ]
    
    # SQL injection patterns - Refined to reduce false positives on normal text
    SQL_INJECTION_PATTERNS = [
        r'\b(SELECT\s.*?FROM\s\w|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|DROP\s+(TABLE|DATABASE)|CREATE\s+(TABLE|DATABASE)|ALTER\s+(TABLE|DATABASE)|EXEC\s|UNION\s+SELECT)',
        r';\s*--(?:[^\r\n]*)',              # SQL comment after semicolon
        r'\b(OR|AND)\s+["\']?\w+["\']?\s*=\s*["\']?\w+["\']?', # More specific boolean checks
    ]
    
    # Command injection patterns - Refined to reduce false positives
    COMMAND_INJECTION_PATTERNS = [
        r'\b(sudo|su|chmod|chown|rm|mv|cp|cat|less|more|head|tail|nc|netcat|wget|curl|ping|nslookup|dig)\s',
        r'[;&|`$]\s*\b',  # Shell command separators followed by a word boundary
        r'\b\w+\s*(\(\s*\)\s*\{.*\})', # Function definitions in shell `func() { ... }`
        r'(\$\(|\`\w)', # Command substitution
    ]
    
    # Safe characters for different contexts
    SAFE_CHARS = {
        'agent_id': re.compile(r'^[a-zA-Z0-9_-]{1,64}$'),
        'task_title': re.compile(r'^[a-zA-Z0-9\s\.\-_,!?]{1,200}$'),
        'capability': re.compile(r'^[a-zA-Z0-9_-]{1,50}$'),
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        'domain': re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    }
    
    def __init__(self):
        self.blocked_attempts = {}  # Track blocked attempts per IP
        self.rate_limits = {}       # Track rate limiting
    
    def validate_task_delegation(self, data: Dict[str, Any], client_ip: str = None) -> ValidationResult:
        """Validate task delegation request"""
        errors = []
        warnings = []
        sanitized = {}
        
        # Required fields
        required_fields = ['task_title', 'task_description', 'required_capabilities', 'requester_id']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(False, errors, security_warnings=warnings)
        
        # Validate task_title
        title_result = self._validate_text_field(
            data['task_title'], 'task_title', max_length=200
        )
        if not title_result.is_valid:
            errors.extend(title_result.errors)
        else:
            sanitized['task_title'] = title_result.sanitized_data
        
        # Validate task_description
        desc_result = self._validate_text_field(
            data['task_description'], 'task_description', max_length=2000
        )
        if not desc_result.is_valid:
            errors.extend(desc_result.errors)
        else:
            sanitized['task_description'] = desc_result.sanitized_data
        
        # Validate required_capabilities
        cap_result = self._validate_capabilities(data['required_capabilities'])
        if not cap_result.is_valid:
            errors.extend(cap_result.errors)
        else:
            sanitized['required_capabilities'] = cap_result.sanitized_data
        
        # Validate requester_id
        requester_result = self._validate_agent_id(data['requester_id'])
        if not requester_result.is_valid:
            errors.extend(requester_result.errors)
        else:
            sanitized['requester_id'] = requester_result.sanitized_data
        
        # Optional fields
        if 'task_data' in data:
            task_data_result = self._validate_task_data(data['task_data'])
            if task_data_result.is_valid:
                sanitized['task_data'] = task_data_result.sanitized_data
            else:
                warnings.extend(task_data_result.errors)
                sanitized['task_data'] = {}
        
        # Rate limiting check
        if client_ip:
            rate_check = self._check_rate_limit(client_ip, 'task_delegation')
            if not rate_check:
                errors.append("Rate limit exceeded")
                warnings.append(f"Suspicious activity from IP {client_ip}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized,
            security_warnings=warnings
        )
    
    def validate_agent_registration(self, data: Dict[str, Any], client_ip: str = None) -> ValidationResult:
        """Validate agent registration data"""
        errors = []
        warnings = []
        sanitized = {}
        
        # Required fields
        required_fields = ['agent_id', 'name', 'capabilities']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(False, errors, security_warnings=warnings)
        
        # Validate agent_id
        agent_id_result = self._validate_agent_id(data['agent_id'])
        if not agent_id_result.is_valid:
            errors.extend(agent_id_result.errors)
        else:
            sanitized['agent_id'] = agent_id_result.sanitized_data
        
        # Validate name
        name_result = self._validate_text_field(data['name'], 'name', max_length=100)
        if not name_result.is_valid:
            errors.extend(name_result.errors)
        else:
            sanitized['name'] = name_result.sanitized_data
        
        # Validate capabilities
        cap_result = self._validate_capabilities(data['capabilities'])
        if not cap_result.is_valid:
            errors.extend(cap_result.errors)
        else:
            sanitized['capabilities'] = cap_result.sanitized_data
        
        # Optional fields validation
        optional_fields = ['goal', 'specialization', 'collaboration_style', 'description']
        for field in optional_fields:
            if field in data:
                field_result = self._validate_text_field(data[field], field, max_length=500)
                if field_result.is_valid:
                    sanitized[field] = field_result.sanitized_data
                else:
                    warnings.extend(field_result.errors)
                    sanitized[field] = ""
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized,
            security_warnings=warnings
        )
    
    def _validate_text_field(self, value: Any, field_name: str, max_length: int = 1000) -> ValidationResult:
        """Validate and sanitize text fields"""
        errors = []
        warnings = []
        
        # Type check
        if not isinstance(value, str):
            errors.append(f"{field_name} must be a string")
            return ValidationResult(False, errors)
        
        # Length check
        if len(value) > max_length:
            errors.append(f"{field_name} too long (max {max_length} characters)")
            return ValidationResult(False, errors)
        
        if len(value.strip()) == 0:
            errors.append(f"{field_name} cannot be empty")
            return ValidationResult(False, errors)
        
        # Security checks
        sanitized_value = self._sanitize_input(value)
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"Potentially dangerous content detected in {field_name}")
                warnings.append(f"XSS pattern detected: {pattern}")
                break
        
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"SQL injection attempt detected in {field_name}")
                warnings.append(f"SQL injection pattern: {pattern}")
                break
        
        # Check for command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"Command injection attempt detected in {field_name}")
                warnings.append(f"Command injection pattern: {pattern}")
                break
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized_value,
            security_warnings=warnings
        )
    
    def _validate_agent_id(self, agent_id: Any) -> ValidationResult:
        """Validate agent ID format"""
        if not isinstance(agent_id, str):
            return ValidationResult(False, ["Agent ID must be a string"])
        
        if not self.SAFE_CHARS['agent_id'].match(agent_id):
            return ValidationResult(
                False, 
                ["Agent ID must contain only alphanumeric characters, hyphens, and underscores (1-64 chars)"]
            )
        
        return ValidationResult(True, [], sanitized_data=agent_id)
    
    def _validate_capabilities(self, capabilities: Any) -> ValidationResult:
        """Validate capabilities list"""
        if not isinstance(capabilities, list):
            return ValidationResult(False, ["Capabilities must be a list"])
        
        if len(capabilities) == 0:
            return ValidationResult(False, ["At least one capability is required"])
        
        if len(capabilities) > 20:
            return ValidationResult(False, ["Too many capabilities (max 20)"])
        
        sanitized_caps = []
        errors = []
        
        for cap in capabilities:
            if not isinstance(cap, str):
                errors.append("All capabilities must be strings")
                continue
            
            if not self.SAFE_CHARS['capability'].match(cap):
                errors.append(f"Invalid capability format: {cap}")
                continue
            
            sanitized_caps.append(cap.lower())
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized_caps
        )
    
    def _validate_task_data(self, task_data: Any) -> ValidationResult:
        """Validate task data dictionary"""
        if not isinstance(task_data, dict):
            return ValidationResult(False, ["Task data must be a dictionary"])
        
        # Size limit for task data
        try:
            json_size = len(json.dumps(task_data))
            if json_size > 10000:  # 10KB limit
                return ValidationResult(False, ["Task data too large (max 10KB)"])
        except (TypeError, ValueError):
            return ValidationResult(False, ["Task data contains non-serializable content"])
        
        # Sanitize all string values in the dictionary
        sanitized_data = self._deep_sanitize_dict(task_data)
        
        return ValidationResult(True, [], sanitized_data=sanitized_data)
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text"""
        # HTML escape
        sanitized = html.escape(text)
        
        # URL decode to prevent double encoding attacks
        sanitized = urllib.parse.unquote(sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _deep_sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary values"""
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            clean_key = self._sanitize_input(str(key))
            
            if isinstance(value, str):
                sanitized[clean_key] = self._sanitize_input(value)
            elif isinstance(value, dict):
                sanitized[clean_key] = self._deep_sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[clean_key] = [
                    self._sanitize_input(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[clean_key] = value
        
        return sanitized
    
    def _check_rate_limit(self, client_ip: str, endpoint: str, limit: int = 60, window: int = 300) -> bool:
        """Check rate limiting (requests per window)"""
        current_time = datetime.now(timezone.utc)
        key = f"{client_ip}:{endpoint}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old timestamps
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if (current_time - timestamp).total_seconds() < window
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[key]) >= limit:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            return False
        
        # Add current timestamp
        self.rate_limits[key].append(current_time)
        return True
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], client_ip: str = None):
        """Log security events for monitoring"""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "client_ip": client_ip,
            "details": details
        }
        
        logger.warning(f"SECURITY EVENT: {json.dumps(event)}")
        
        # Track repeat offenders
        if client_ip:
            if client_ip not in self.blocked_attempts:
                self.blocked_attempts[client_ip] = 0
            self.blocked_attempts[client_ip] += 1

# Global validator instance
security_validator = SecurityValidator() 