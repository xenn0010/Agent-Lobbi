"""
World State Management for Agent Lobbi
Handles persistent state that agents can read/write to
"""
import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import threading


class WorldState:
    """Manages persistent world state that agents can interact with"""
    
    def __init__(self, state_file_path: str = "world_state.json"):
        self.state_file_path = Path(state_file_path)
        self.state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._load_state()
    
    def _load_state(self):
        """Load state from file"""
        with self._lock:
            if self.state_file_path.exists():
                try:
                    with open(self.state_file_path, 'r') as f:
                        self.state = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"WORLD_STATE: Error loading state file: {e}")
                    self.state = {}
            else:
                self.state = {}
                self._save_state()  # Create the file
    
    def _save_state(self):
        """Save state to file"""
        with self._lock:
            try:
                with open(self.state_file_path, 'w') as f:
                    json.dump(self.state, f, indent=2, default=str)
            except IOError as e:
                print(f"WORLD_STATE: Error saving state file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from world state"""
        with self._lock:
            return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a value in world state"""
        with self._lock:
            self.state[key] = value
            self._save_state()
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple values in world state"""
        with self._lock:
            self.state.update(updates)
            self._save_state()
    
    def delete(self, key: str) -> bool:
        """Delete a key from world state"""
        with self._lock:
            if key in self.state:
                del self.state[key]
                self._save_state()
                return True
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all world state"""
        with self._lock:
            return self.state.copy()
    
    def clear(self):
        """Clear all world state"""
        with self._lock:
            self.state = {}
            self._save_state() 