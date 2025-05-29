#!/usr/bin/env python3
"""
Production Main Entry Point for Agent Lobby
Handles configuration, logging, graceful shutdown, and service orchestration
"""
import asyncio
import signal
import sys
import os
import argparse
from typing import Optional
import structlog
from contextlib import asynccontextmanager

# Import our production components
from core.lobby import Lobby
from core.database import db_manager
from core.load_balancer import load_balancer
from sdk.monitoring_sdk import monitoring_sdk, MonitoringConfig

# Configure structured logging
def setup_logging(debug: bool = False):
    """Setup structured logging with appropriate level"""
    log_level = "DEBUG" if debug else os.getenv("LOG_LEVEL", "INFO")
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

class AgentLobbyService:
    """Main service orchestrator for Agent Lobby"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.lobby: Optional[Lobby] = None
        self.shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start all services"""
        try:
            self.logger.info("🚀 Starting Agent Lobby Service", config=self.config)
            
            # Initialize lobby with configuration
            self.lobby = Lobby(
                host=self.config.get("host", "0.0.0.0"),
                http_port=self.config.get("http_port", 8080),
                ws_port=self.config.get("ws_port", 8081)
            )
            
            # Start the lobby (this initializes all components)
            await self.lobby.start()
            
            self.logger.info("✅ Agent Lobby Service started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error("❌ Failed to start Agent Lobby Service", error=str(e))
            raise
    
    async def stop(self):
        """Gracefully stop all services"""
        try:
            self.logger.info("🛑 Stopping Agent Lobby Service")
            
            if self.lobby:
                await self.lobby.stop()
            
            self.logger.info("✅ Agent Lobby Service stopped gracefully")
            
        except Exception as e:
            self.logger.error("❌ Error during shutdown", error=str(e))
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info("📡 Received shutdown signal", signal=signum)
        self.shutdown_event.set()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Agent Lobby - Multi-Agent Collaboration Platform")
    
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), 
                       help="Host to bind to")
    parser.add_argument("--http-port", type=int, default=int(os.getenv("PORT", "8080")), 
                       help="HTTP port")
    parser.add_argument("--ws-port", type=int, default=int(os.getenv("WS_PORT", "8081")), 
                       help="WebSocket port")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    parser.add_argument("--config-file", 
                       help="Path to configuration file")
    
    return parser.parse_args()

def load_config(args) -> dict:
    """Load configuration from environment and arguments"""
    config = {
        "host": args.host,
        "http_port": args.http_port,
        "ws_port": args.ws_port,
        "debug": args.debug,
        
        # Database configuration
        "database_url": os.getenv("DATABASE_URL"),
        "postgres_host": os.getenv("POSTGRES_HOST", "localhost"),
        "postgres_port": int(os.getenv("POSTGRES_PORT", "5432")),
        "postgres_db": os.getenv("POSTGRES_DB", "agent_lobby"),
        "postgres_user": os.getenv("POSTGRES_USER", "postgres"),
        "postgres_password": os.getenv("POSTGRES_PASSWORD", ""),
        
        # Redis configuration
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        
        # Monitoring configuration
        "enable_monitoring": os.getenv("ENABLE_MONITORING", "true").lower() == "true",
        "metrics_interval": int(os.getenv("METRICS_INTERVAL", "30")),
        "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "15")),
        
        # Load balancer configuration
        "load_balancer_strategy": os.getenv("LOAD_BALANCER_STRATEGY", "performance_based"),
        "circuit_breaker_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
        
        # Security configuration
        "api_key_required": os.getenv("API_KEY_REQUIRED", "false").lower() == "true",
        "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        
        # Performance configuration
        "max_connections": int(os.getenv("MAX_CONNECTIONS", "1000")),
        "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "30")),
    }
    
    # Load from config file if specified
    if args.config_file and os.path.exists(args.config_file):
        import json
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config

async def health_check_endpoint():
    """Simple health check for container orchestration"""
    from aiohttp import web
    
    async def health(request):
        """Health check endpoint"""
        try:
            # Check database connectivity
            # This is a simplified check - in production you'd check all critical services
            health_status = {
                "status": "healthy",
                "timestamp": structlog.processors.TimeStamper(fmt="iso")(),
                "version": os.getenv("VERSION", "unknown"),
                "services": {
                    "database": "healthy",  # Would check actual DB connection
                    "load_balancer": "healthy",
                    "monitoring": "healthy"
                }
            }
            return web.json_response(health_status)
        except Exception as e:
            return web.json_response(
                {"status": "unhealthy", "error": str(e)}, 
                status=503
            )
    
    app = web.Application()
    app.router.add_get('/health', health)
    return app

async def main():
    """Main entry point"""
    # Parse arguments and load configuration
    args = parse_arguments()
    config = load_config(args)
    
    # Setup logging
    setup_logging(config["debug"])
    logger = structlog.get_logger(__name__)
    
    logger.info("🎯 Agent Lobby Starting", version=os.getenv("VERSION", "dev"))
    
    # Create service instance
    service = AgentLobbyService(config)
    
    # Setup signal handlers for graceful shutdown
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, service.signal_handler)
    
    try:
        # Start the service
        await service.start()
    except KeyboardInterrupt:
        logger.info("🔄 Received keyboard interrupt")
    except Exception as e:
        logger.error("💥 Fatal error", error=str(e))
        sys.exit(1)
    finally:
        # Ensure graceful shutdown
        await service.stop()
        logger.info("👋 Agent Lobby shutdown complete")

if __name__ == "__main__":
    # Handle different Python versions and event loop policies
    if sys.platform == "win32":
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1) 