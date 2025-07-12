#!/usr/bin/env python3
"""
Module entry point for Agent Lobbi
This allows running the package with 'python -m src'
"""

if __name__ == "__main__":
    from .main import main
    import asyncio
    import sys
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSTOP Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1) 