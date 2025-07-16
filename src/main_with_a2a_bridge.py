#!/usr/bin/env python3
"""
ENHANCED AGENT LOBBY WITH A2A BRIDGE & METRICS
==============================================
Main launcher for Agent Lobby with integrated A2A API bridge,
metrics dashboard, and comprehensive connector capabilities.
"""

import asyncio
import logging
import signal
import sys
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lobby import Lobby
from api.enhanced_a2a_api_bridge import EnhancedA2AAPIBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class EnhancedAgentLobbySystem:
    """Enhanced Agent Lobby system with integrated A2A bridge and metrics"""
    
    def __init__(self):
        self.lobby: Lobby = None
        self.a2a_bridge: EnhancedA2AAPIBridge = None
        self.lobby_task = None
        self.bridge_task = None
        self.running = False
        
    async def start_system(self):
        """Start the complete system with lobby and A2A bridge"""
        logger.info("🚀 Starting Enhanced Agent Lobby System...")
        logger.info("   🏢 Agent Lobby Core")
        logger.info("   🌐 A2A Protocol Bridge") 
        logger.info("   📊 Metrics Dashboard")
        logger.info("   🔗 WebSocket Real-time")
        
        try:
            # Step 1: Initialize and start the core lobby
            logger.info("📡 Initializing Agent Lobby Core...")
            self.lobby = Lobby(host="localhost", http_port=8080, ws_port=8081)
            
            # Start lobby in background
            self.lobby_task = asyncio.create_task(self.lobby.start())
            await asyncio.sleep(2)  # Give lobby time to initialize
            
            logger.info("✅ Agent Lobby Core: OPERATIONAL")
            
            # Step 2: Initialize A2A bridge with lobby integration
            logger.info("🌐 Initializing Enhanced A2A API Bridge...")
            self.a2a_bridge = EnhancedA2AAPIBridge(
                lobby_instance=self.lobby,  # Direct integration
                lobby_host="localhost",
                lobby_http_port=8080,
                lobby_ws_port=8081
            )
            
            # Step 3: Start A2A bridge server
            logger.info("🚀 Starting A2A API Bridge Server...")
            config = uvicorn.Config(
                self.a2a_bridge.app,
                host="localhost",
                port=8090,
                log_level="info",
                access_log=False
            )
            
            server = uvicorn.Server(config)
            self.bridge_task = asyncio.create_task(server.serve())
            await asyncio.sleep(1)  # Give bridge time to start
            
            logger.info("✅ Enhanced A2A API Bridge: OPERATIONAL")
            
            # Step 4: Display system status
            self._display_system_status()
            
            self.running = True
            logger.info("🎉 Enhanced Agent Lobby System: FULLY OPERATIONAL")
            
            # Keep system running
            await self._keep_system_running()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Enhanced Agent Lobby System: {e}")
            await self.shutdown()
            
    async def _keep_system_running(self):
        """Keep the system running and monitor health"""
        logger.info("🔄 System monitoring active...")
        
        while self.running:
            try:
                # Health check every 30 seconds
                await asyncio.sleep(30)
                
                # Check if lobby is still running
                if self.lobby_task and self.lobby_task.done():
                    logger.error("❌ Lobby task has stopped!")
                    break
                
                # Check if bridge is still running  
                if self.bridge_task and self.bridge_task.done():
                    logger.error("❌ A2A Bridge task has stopped!")
                    break
                
                # Log periodic status
                if hasattr(self.lobby, 'agents'):
                    agent_count = len(self.lobby.agents)
                    online_count = len(self.lobby.live_agent_connections)
                    logger.info(f"📊 Status: {agent_count} agents registered, {online_count} online")
                
            except Exception as e:
                logger.error(f"⚠️ Health check error: {e}")
                
    def _display_system_status(self):
        """Display comprehensive system status"""
        print("\n" + "="*80)
        print("🎉 ENHANCED AGENT LOBBY SYSTEM - OPERATIONAL")
        print("="*80)
        print()
        print("🏢 CORE LOBBY:")
        print(f"   📡 HTTP Server:     http://localhost:8080")
        print(f"   🔌 WebSocket:       ws://localhost:8081") 
        print(f"   🆔 Lobby ID:        {self.lobby.lobby_id}")
        print()
        print("🌐 A2A API BRIDGE:")
        print(f"   🔗 Bridge API:      http://localhost:8090")
        print(f"   📋 A2A Discovery:   http://localhost:8090/.well-known/agent.json")
        print(f"   📊 Metrics API:     http://localhost:8090/api/metrics")
        print(f"   📈 Dashboard:       http://localhost:8090/metrics/dashboard")
        print()
        print("🚀 CAPABILITIES:")
        print("   ✅ A2A Protocol:         Full compliance + enhanced features")
        print("   ✅ Native Protocol:      Agent Lobby native with NAA/LAM")
        print("   ✅ WebSocket Support:    Real-time communication")
        print("   ✅ HTTP Fallback:        Compatible with HTTP-only agents")
        print("   ✅ Metrics Dashboard:    Real-time performance monitoring")
        print("   ✅ Multi-Agent Collab:   Advanced collaboration engine")
        print()
        print("📱 QUICK LINKS:")
        print("   🌐 API Documentation:   http://localhost:8090/docs")
        print("   📊 Live Metrics:        http://localhost:8090/metrics/dashboard")
        print("   🔍 A2A Status:          http://localhost:8090/api/a2a/status")
        print("   💻 Lobby Health:        http://localhost:8080/api/health")
        print()
        print("="*80)
        print("🎯 System ready for agent registration and collaboration!")
        print("="*80)
        print()
        
    async def shutdown(self):
        """Gracefully shutdown the entire system"""
        logger.info("🛑 Shutting down Enhanced Agent Lobby System...")
        self.running = False
        
        # Shutdown A2A bridge
        if self.bridge_task and not self.bridge_task.done():
            logger.info("   🌐 Stopping A2A Bridge...")
            self.bridge_task.cancel()
            try:
                await self.bridge_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown lobby
        if self.lobby_task and not self.lobby_task.done():
            logger.info("   🏢 Stopping Agent Lobby...")
            self.lobby_task.cancel()
            try:
                await self.lobby_task
            except asyncio.CancelledError:
                pass
        
        if self.lobby:
            await self.lobby.shutdown()
            
        logger.info("✅ Enhanced Agent Lobby System shutdown complete")

async def main():
    """Main entry point"""
    system = EnhancedAgentLobbySystem()
    
    # Handle shutdown signals
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
        print("\n👋 Enhanced Agent Lobby System stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1) 