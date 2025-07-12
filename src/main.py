#!/usr/bin/env python3
"""
CORE AGENT LOBBI RUNNER
=======================
This script starts the central Agent Lobbi server.
It is responsible for managing agents, tasks, and the collaboration engine.
The API Bridge is a separate component that runs in its own process.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

from core.lobby import Lobby

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point to start the Lobby server."""
    logger.info("Initializing Core Agent Lobbi System...")

    # Read ports from environment variables, with standardized defaults
    http_port = int(os.environ.get("LOBBY_HTTP_PORT", 8080))
    ws_port = int(os.environ.get("LOBBY_WS_PORT", 8081))

    lobby = Lobby(http_port=http_port, ws_port=ws_port)
    
    try:
        await lobby.start()
        logger.info(f"Core Agent Lobbi started successfully on HTTP:{http_port} and WS:{ws_port}")
        # Keep the lobby running indefinitely
        await asyncio.Event().wait()
    except Exception as e:
        logger.critical(f"A critical error occurred in the Lobby: {e}", exc_info=True)
    finally:
        logger.info("Shutting down Core Agent Lobbi...")
        await lobby.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Lobby shutdown requested by user.")
    except Exception as e:
        logger.error(f"Failed to run Lobby: {e}", exc_info=True)