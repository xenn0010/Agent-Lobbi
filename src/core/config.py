"""
Configuration management system for the agent ecosystem.
Provides environment-based configuration, validation, hot reloading, and secure secret management.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union, Type, get_type_hints
from datetime import datetime, timezone
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
import asyncio
import threading
import time
# Optional watchdog import for file monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    WATCHDOG_AVAILABLE = False

# Environment and secrets
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import base64

# Pydantic for validation
from pydantic import BaseModel, Field, validator, ValidationError


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "mongodb"  # mongodb or postgresql
    host: str = "localhost"
    port: int = 27017
    username: Optional[str] = None
    password: Optional[str] = None
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


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 7
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # Password requirements
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


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enabled: bool = True
    metrics_port: int = 8000
    health_check_interval: int = 30
    
    # Prometheus settings
    prometheus_enabled: bool = True
    prometheus_namespace: str = "agent_ecosystem"
    
    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"  # json or text
    log_file: Optional[str] = None
    log_rotation: bool = True
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    
    # Telemetry settings
    telemetry_enabled: bool = False
    jaeger_endpoint: Optional[str] = None
    
    # Performance tracking
    performance_tracking_enabled: bool = True
    performance_window_size: int = 100


@dataclass
class LobbyConfig:
    """Lobby server configuration"""
    host: str = "localhost"
    http_port: int = 8080
    websocket_port: int = 8081
    
    # WebSocket settings
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10
    websocket_close_timeout: int = 10
    
    # Message handling
    message_queue_size: int = 1000
    message_timeout: int = 30
    max_message_size: int = 1024 * 1024  # 1MB
    
    # Agent management
    agent_timeout: int = 300  # 5 minutes
    max_agents: int = 1000
    agent_heartbeat_interval: int = 60


@dataclass
class RedisConfig:
    """Redis configuration for caching and pub/sub"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0
    
    # Connection pool settings
    max_connections: int = 20
    connection_timeout: int = 5
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    key_prefix: str = "agent_ecosystem:"


@dataclass
class APIConfig:
    """API configuration"""
    enabled: bool = True
    host: str = "localhost"
    port: int = 8000
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30
    
    # Documentation
    docs_enabled: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


class AgentEcosystemConfig(BaseModel):
    """Main configuration class for the agent ecosystem"""
    
    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    lobby: LobbyConfig = Field(default_factory=LobbyConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Feature flags
    features: Dict[str, bool] = Field(default_factory=lambda: {
        "database_persistence": True,
        "security_enabled": True,
        "monitoring_enabled": True,
        "redis_caching": False,
        "api_enabled": True,
        "websocket_compression": False,
        "message_encryption": False
    })
    
    # Custom settings
    custom: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        validate_assignment = True
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('database')
    def validate_database_config(cls, v):
        if v.type not in ["mongodb", "postgresql"]:
            raise ValueError("Database type must be 'mongodb' or 'postgresql'")
        
        if v.type == "mongodb" and v.port == 5432:
            v.port = 27017
        elif v.type == "postgresql" and v.port == 27017:
            v.port = 5432
        
        return v
    
    @validator('security')
    def validate_security_config(cls, v):
        if len(v.jwt_secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_feature(self, feature_name: str, default: bool = False) -> bool:
        """Get feature flag value"""
        return self.features.get(feature_name, default)
    
    def set_feature(self, feature_name: str, enabled: bool):
        """Set feature flag value"""
        self.features[feature_name] = enabled


# Only define ConfigFileHandler if watchdog is available
if WATCHDOG_AVAILABLE:
    class ConfigFileHandler(FileSystemEventHandler):
        """File system event handler for configuration hot reloading"""
        
        def __init__(self, config_manager):
            self.config_manager = config_manager
            self.logger = logging.getLogger(__name__)
        
        def on_modified(self, event):
            if not event.is_directory and event.src_path in self.config_manager.watched_files:
                self.logger.info(f"Configuration file changed: {event.src_path}")
                asyncio.create_task(self.config_manager.reload_config())
else:
    # Dummy handler when watchdog is not available
    class ConfigFileHandler:
        def __init__(self, config_manager):
            pass


class ConfigManager:
    """Configuration manager with hot reloading and environment support"""
    
    def __init__(self, 
                 config_dir: str = "config",
                 env_file: str = ".env",
                 enable_hot_reload: bool = True):
        
        self.config_dir = Path(config_dir)
        self.env_file = env_file
        self.enable_hot_reload = enable_hot_reload
        
        self.logger = logging.getLogger(__name__)
        self.config: Optional[AgentEcosystemConfig] = None
        self.watched_files: List[str] = []
        self.observer: Optional[Observer] = None
        self.reload_callbacks: List[callable] = []
        
        # Load environment variables
        load_dotenv(env_file)
        
        # Initialize encryption for secrets
        self._init_encryption()
        
        # Load initial configuration
        self.load_config()
        
        # Start file watching if enabled and available
        if enable_hot_reload and WATCHDOG_AVAILABLE:
            self.start_watching()
        elif enable_hot_reload and not WATCHDOG_AVAILABLE:
            self.logger.warning("Hot reload requested but watchdog not available")
    
    def _init_encryption(self):
        """Initialize encryption for sensitive configuration values"""
        encryption_key = os.getenv("CONFIG_ENCRYPTION_KEY")
        if not encryption_key:
            # Generate a new key for development
            encryption_key = Fernet.generate_key().decode()
            self.logger.warning(
                f"No CONFIG_ENCRYPTION_KEY found, generated new key: {encryption_key}"
            )
        
        try:
            self.fernet = Fernet(encryption_key.encode())
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            self.fernet = None
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value"""
        if self.fernet:
            return self.fernet.encrypt(value.encode()).decode()
        return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value"""
        if self.fernet:
            try:
                return self.fernet.decrypt(encrypted_value.encode()).decode()
            except Exception:
                # If decryption fails, assume it's not encrypted
                return encrypted_value
        return encrypted_value
    
    def load_config(self) -> AgentEcosystemConfig:
        """Load configuration from files and environment variables"""
        try:
            # Determine environment
            env = os.getenv("ENVIRONMENT", "development").lower()
            environment = Environment(env)
            
            # Load base configuration
            config_data = self._load_config_files(environment)
            
            # Override with environment variables
            config_data = self._apply_env_overrides(config_data)
            
            # Create and validate configuration
            self.config = AgentEcosystemConfig(**config_data)
            
            self.logger.info(f"Configuration loaded for environment: {environment.value}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Return default configuration
            self.config = AgentEcosystemConfig()
            return self.config
    
    def _load_config_files(self, environment: Environment) -> Dict[str, Any]:
        """Load configuration from YAML/JSON files"""
        config_data = {}
        
        # Load base configuration
        base_files = ["config.yaml", "config.yml", "config.json"]
        for filename in base_files:
            file_path = self.config_dir / filename
            if file_path.exists():
                config_data.update(self._load_file(file_path))
                self.watched_files.append(str(file_path))
                break
        
        # Load environment-specific configuration
        env_files = [
            f"config.{environment.value}.yaml",
            f"config.{environment.value}.yml",
            f"config.{environment.value}.json"
        ]
        
        for filename in env_files:
            file_path = self.config_dir / filename
            if file_path.exists():
                env_config = self._load_file(file_path)
                config_data = self._deep_merge(config_data, env_config)
                self.watched_files.append(str(file_path))
                break
        
        return config_data
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        
        # Define environment variable mappings
        env_mappings = {
            # Database
            "DATABASE_TYPE": "database.type",
            "DATABASE_HOST": "database.host",
            "DATABASE_PORT": "database.port",
            "DATABASE_USERNAME": "database.username",
            "DATABASE_PASSWORD": "database.password",
            "DATABASE_NAME": "database.database",
            
            # Security
            "JWT_SECRET_KEY": "security.jwt_secret_key",
            "JWT_EXPIRATION_HOURS": "security.jwt_expiration_hours",
            
            # Lobby
            "LOBBY_HOST": "lobby.host",
            "LOBBY_HTTP_PORT": "lobby.http_port",
            "LOBBY_WEBSOCKET_PORT": "lobby.websocket_port",
            
            # Monitoring
            "MONITORING_ENABLED": "monitoring.enabled",
            "METRICS_PORT": "monitoring.metrics_port",
            "LOG_LEVEL": "monitoring.log_level",
            
            # Redis
            "REDIS_ENABLED": "redis.enabled",
            "REDIS_HOST": "redis.host",
            "REDIS_PORT": "redis.port",
            "REDIS_PASSWORD": "redis.password",
            
            # API
            "API_ENABLED": "api.enabled",
            "API_HOST": "api.host",
            "API_PORT": "api.port",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle encrypted values
                if env_var.endswith("_PASSWORD") or env_var.endswith("_SECRET_KEY"):
                    value = self.decrypt_value(value)
                
                # Convert types
                value = self._convert_env_value(value)
                
                # Set nested configuration value
                self._set_nested_value(config_data, config_path, value)
        
        return config_data
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON values
        if value.startswith(("{", "[")):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String value
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = path.split(".")
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def start_watching(self):
        """Start watching configuration files for changes"""
        if not self.enable_hot_reload or not self.watched_files or not WATCHDOG_AVAILABLE:
            return
        
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            
            # Watch the config directory
            self.observer.schedule(handler, str(self.config_dir), recursive=False)
            self.observer.start()
            
            self.logger.info("Configuration file watching started")
            
        except Exception as e:
            self.logger.error(f"Failed to start configuration file watching: {e}")
    
    def stop_watching(self):
        """Stop watching configuration files"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("Configuration file watching stopped")
    
    async def reload_config(self):
        """Reload configuration from files"""
        try:
            old_config = self.config
            new_config = self.load_config()
            
            # Notify callbacks of configuration change
            for callback in self.reload_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_config, new_config)
                    else:
                        callback(old_config, new_config)
                except Exception as e:
                    self.logger.error(f"Error in config reload callback: {e}")
            
            self.logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {e}")
    
    def add_reload_callback(self, callback: callable):
        """Add a callback to be called when configuration is reloaded"""
        self.reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: callable):
        """Remove a configuration reload callback"""
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
    
    def get_config(self) -> AgentEcosystemConfig:
        """Get the current configuration"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def save_config(self, config: AgentEcosystemConfig, filename: str = None):
        """Save configuration to file"""
        if filename is None:
            filename = f"config.{config.environment.value}.yaml"
        
        file_path = self.config_dir / filename
        
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary and save
            config_dict = config.dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self, config_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration data"""
        try:
            AgentEcosystemConfig(**config_data)
            return True, []
        except ValidationError as e:
            errors = [f"{'.'.join(map(str, error['loc']))}: {error['msg']}" 
                     for error in e.errors()]
            return False, errors
    
    def get_secret(self, key: str, default: str = None) -> Optional[str]:
        """Get a secret value from environment or configuration"""
        # Try environment variable first
        value = os.getenv(key, default)
        if value:
            return self.decrypt_value(value)
        return default
    
    def __del__(self):
        """Cleanup when the config manager is destroyed"""
        self.stop_watching()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> AgentEcosystemConfig:
    """Get the current configuration"""
    return get_config_manager().get_config()


def initialize_config(config_dir: str = "config", 
                     env_file: str = ".env",
                     enable_hot_reload: bool = True) -> ConfigManager:
    """Initialize the global configuration manager"""
    global _config_manager
    _config_manager = ConfigManager(
        config_dir=config_dir,
        env_file=env_file,
        enable_hot_reload=enable_hot_reload
    )
    return _config_manager 