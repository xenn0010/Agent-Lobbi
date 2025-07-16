#!/usr/bin/env python3
"""
RAILWAY-OPTIMIZED AGENT LOBBY WITH A2A BRIDGE
=============================================
Railway-optimized launcher for Agent Lobby with integrated A2A API bridge,
properly configured for cloud deployment with environment variable support.
"""

import asyncio
import logging
import signal
import sys
import os
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lobby import Lobby
from api.enhanced_a2a_api_bridge import EnhancedA2AAPIBridge

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class RailwayAgentLobbySystem:
    """Railway-optimized Agent Lobby system with integrated A2A bridge and metrics"""
    
    def __init__(self):
        self.lobby: Lobby = None
        self.a2a_bridge: EnhancedA2AAPIBridge = None
        self.lobby_task = None
        self.bridge_task = None
        self.running = False
        
        # Railway environment configuration
        self.host = os.getenv("HOST", "0.0.0.0")  # Railway requires 0.0.0.0
        self.port = int(os.getenv("PORT", 8080))  # Railway provides PORT
        self.lobby_http_port = int(os.getenv("LOBBY_HTTP_PORT", self.port))
        self.lobby_ws_port = int(os.getenv("LOBBY_WS_PORT", self.port + 1))
        
        # Database configuration
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        
        logger.info(f"🎯 Railway Configuration:")
        logger.info(f"   Host: {self.host}")
        logger.info(f"   Primary Port: {self.port}")
        logger.info(f"   Lobby HTTP Port: {self.lobby_http_port}")
        logger.info(f"   Lobby WS Port: {self.lobby_ws_port}")
        logger.info(f"   Database: {'✅ Configured' if self.database_url else '❌ Missing'}")
        logger.info(f"   Redis: {'✅ Configured' if self.redis_url else '❌ Missing'}")
        
    async def start_system(self):
        """Start the complete system with lobby and A2A bridge"""
        logger.info("🚀 Starting Railway Agent Lobby System...")
        logger.info("   🏢 Agent Lobby Core")
        logger.info("   🌐 A2A Protocol Bridge") 
        logger.info("   📊 Metrics Dashboard")
        logger.info("   🔗 WebSocket Real-time")
        
        try:
            # Step 1: Initialize and start the core lobby
            logger.info("📡 Initializing Agent Lobby Core...")
            self.lobby = Lobby(
                host=self.host, 
                http_port=self.lobby_http_port, 
                ws_port=self.lobby_ws_port
            )
            
            # Start lobby in background (non-blocking for Railway)
            self.lobby_task = asyncio.create_task(self.lobby.start())
            await asyncio.sleep(2)  # Give lobby time to initialize
            
            logger.info("✅ Agent Lobby Core: OPERATIONAL")
            
            # Step 2: Initialize A2A bridge with lobby integration
            logger.info("🌐 Initializing Enhanced A2A API Bridge...")
            self.a2a_bridge = EnhancedA2AAPIBridge(
                lobby_instance=self.lobby,  # Direct integration
                lobby_host=self.host,
                lobby_http_port=self.lobby_http_port,
                lobby_ws_port=self.lobby_ws_port
            )
            
            # Step 3: Start A2A bridge server on Railway's assigned port
            logger.info("🚀 Starting A2A API Bridge Server...")
            config = uvicorn.Config(
                self.a2a_bridge.app,
                host=self.host,
                port=self.port,  # Use Railway's assigned port
                log_level=os.getenv("LOG_LEVEL", "info").lower(),
                access_log=True,
                workers=int(os.getenv("MAX_WORKERS", 1))  # Railway works better with 1 worker
            )
            
            server = uvicorn.Server(config)
            self.bridge_task = asyncio.create_task(server.serve())
            await asyncio.sleep(1)  # Give bridge time to start
            
            logger.info("✅ Enhanced A2A API Bridge: OPERATIONAL")
            
            # Step 4: Display system status
            self._display_system_status()
            
            self.running = True
            logger.info("🎉 Railway Agent Lobby System: FULLY OPERATIONAL")
            
            # Keep system running
            await self._keep_system_running()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Railway Agent Lobby System: {e}")
            await self.shutdown()
            
    def _display_system_status(self):
        """Display current system status"""
        logger.info("📊 SYSTEM STATUS DASHBOARD")
        logger.info("="*50)
        logger.info(f"🌐 A2A API Bridge: http://{self.host}:{self.port}")
        logger.info(f"🏢 Agent Lobby HTTP: http://{self.host}:{self.lobby_http_port}")
        logger.info(f"🔗 Agent Lobby WebSocket: ws://{self.host}:{self.lobby_ws_port}")
        logger.info(f"📈 Health Check: http://{self.host}:{self.port}/health")
        logger.info(f"📊 Metrics: http://{self.host}:{self.port}/metrics")
        logger.info(f"📚 API Docs: http://{self.host}:{self.port}/docs")
        logger.info("="*50)
            
    async def _keep_system_running(self):
        """Keep the system running and monitor health"""
        logger.info("🔄 System monitoring active...")
        
        while self.running:
            try:
                # Health check every 60 seconds (Railway-friendly)
                await asyncio.sleep(60)
                
                # Check if lobby is still running
                if self.lobby_task and self.lobby_task.done():
                    logger.error("❌ Lobby task has stopped!")
                    break
                
                # Check if bridge is still running  
                if self.bridge_task and self.bridge_task.done():
                    logger.error("❌ A2A Bridge task has stopped!")
                    break
                
                # Log periodic status (Railway-friendly logging)
                if hasattr(self.lobby, 'agents'):
                    agent_count = len(self.lobby.agents)
                    online_count = len(self.lobby.live_agent_connections)
                    logger.info(f"💓 Health Check: {agent_count} agents, {online_count} online")
                
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
                break
                
        logger.info("🔄 System monitoring stopped")
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Initiating Railway Agent Lobby System shutdown...")
        self.running = False
        
        # Cancel bridge task
        if self.bridge_task and not self.bridge_task.done():
            logger.info("🛑 Stopping A2A Bridge...")
            self.bridge_task.cancel()
            try:
                await self.bridge_task
            except asyncio.CancelledError:
                pass
        
        # Cancel lobby task
        if self.lobby_task and not self.lobby_task.done():
            logger.info("🛑 Stopping Agent Lobby...")
            self.lobby_task.cancel()
            try:
                await self.lobby_task
            except asyncio.CancelledError:
                pass
        
        if self.lobby:
            await self.lobby.shutdown()
            
        logger.info("✅ Railway Agent Lobby System shutdown complete")

async def main():
    """Main entry point"""
    system = RailwayAgentLobbySystem()
    
    # Handle shutdown signals (Railway-compatible)
    def signal_handler(signum, frame):
        logger.info(f"📡 Received signal {signum}, initiating shutdown...")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await system.start_system()
    except KeyboardInterrupt:
        logger.info("📡 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Railway Agent Lobby System stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1) 