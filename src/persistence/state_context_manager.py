"""
Advanced State Persistence and Context Management System
Ensures agents can resume exactly where they left off when tasks are interrupted
"""

import asyncio
import json
import pickle
import gzip
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import uuid
from collections import defaultdict
import redis
import sqlite3
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class StateType(Enum):
    """Types of state that can be persisted"""
    WORKFLOW_STATE = "workflow"
    COLLABORATION_STATE = "collaboration"
    TASK_STATE = "task"
    AGENT_STATE = "agent"
    CONVERSATION_STATE = "conversation"
    CONTEXT_STATE = "context"


class StateStatus(Enum):
    """Status of persisted states"""
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class StateSnapshot:
    """A complete snapshot of state at a point in time"""
    snapshot_id: str
    state_type: StateType
    entity_id: str  # workflow_id, collaboration_id, task_id, etc.
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # State management
    status: StateStatus = StateStatus.ACTIVE
    parent_snapshot_id: Optional[str] = None
    checkpoint_sequence: int = 0
    
    # Context tracking
    participants: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    related_states: Set[str] = field(default_factory=set)
    
    # Versioning
    version: str = "1.0"
    schema_version: str = "1.0"
    
    @property
    def is_expired(self) -> bool:
        """Check if snapshot has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age of snapshot in seconds"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        data['state_type'] = self.state_type.value
        data['status'] = self.status.value
        data['participants'] = list(self.participants)
        data['dependencies'] = list(self.dependencies)
        data['related_states'] = list(self.related_states)
        return data


@dataclass
class ContextFrame:
    """A frame of context for resuming operations"""
    frame_id: str
    entity_id: str
    frame_type: str  # "workflow_step", "collaboration_point", "task_checkpoint"
    
    # Context data
    local_state: Dict[str, Any] = field(default_factory=dict)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    execution_stack: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resumption information
    resume_point: str = ""
    next_actions: List[Dict[str, Any]] = field(default_factory=list)
    pending_responses: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    required_agents: Set[str] = field(default_factory=set)
    available_agents: Set[str] = field(default_factory=set)
    waiting_for: Set[str] = field(default_factory=set)
    
    # Temporal tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_state(self, key: str, value: Any, shared: bool = False):
        """Update state within the context frame"""
        if shared:
            self.shared_state[key] = value
        else:
            self.local_state[key] = value
        self.last_updated = datetime.now(timezone.utc)


class StateContextManager:
    """
    Advanced state persistence and context management system.
    
    Features:
    - Multi-level state snapshots
    - Intelligent context preservation
    - Automatic resumption points
    - Dependency tracking
    - Efficient compression and storage
    - Redis for fast access, SQLite for durability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Storage backends
        self.redis_client: Optional[redis.Redis] = None
        self.sqlite_db: Optional[str] = None
        self.filesystem_root: Optional[Path] = None
        
        # Memory cache
        self.memory_cache: Dict[str, StateSnapshot] = {}
        self.context_frames: Dict[str, ContextFrame] = {}
        
        # State tracking
        self.active_states: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> snapshot_ids
        self.state_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.snapshots_created = 0
        self.resumptions_executed = 0
        
        # Locks for thread safety
        self._locks: Dict[str, asyncio.Lock] = defaultdict(lambda: asyncio.Lock())
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage backends"""
        # Redis for fast access
        redis_config = self.config.get('redis', {})
        if redis_config.get('enabled', True):
            try:
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=False  # We'll handle encoding ourselves
                )
                self.redis_client.ping()
                logger.info("Redis backend initialized for state management")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.redis_client = None
        
        # SQLite for durability
        sqlite_config = self.config.get('sqlite', {})
        if sqlite_config.get('enabled', True):
            self.sqlite_db = sqlite_config.get('path', 'state_persistence.db')
            self._initialize_sqlite()
        
        # Filesystem for large states
        fs_config = self.config.get('filesystem', {})
        if fs_config.get('enabled', True):
            self.filesystem_root = Path(fs_config.get('root', './state_storage'))
            self.filesystem_root.mkdir(parents=True, exist_ok=True)
    
    def _initialize_sqlite(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        # State snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS state_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                state_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_accessed TEXT NOT NULL,
                parent_snapshot_id TEXT,
                checkpoint_sequence INTEGER,
                participants TEXT,
                dependencies TEXT,
                related_states TEXT,
                metadata TEXT,
                state_data_compressed BLOB
            )
        ''')
        
        # Create indexes for state_snapshots
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_entity_id ON state_snapshots(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_state_type ON state_snapshots(state_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_status ON state_snapshots(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON state_snapshots(created_at)')
        
        # Context frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_frames (
                frame_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                frame_type TEXT NOT NULL,
                resume_point TEXT,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                required_agents TEXT,
                available_agents TEXT,
                waiting_for TEXT,
                frame_data_compressed BLOB
            )
        ''')
        
        # Create indexes for context_frames
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_entity_id ON context_frames(entity_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_frame_type ON context_frames(frame_type)')
        
        conn.commit()
        conn.close()
        logger.info("SQLite backend initialized for state persistence")
    
    async def create_snapshot(self, state_type: StateType, entity_id: str, 
                            state_data: Dict[str, Any], metadata: Dict[str, Any] = None,
                            expires_in_hours: Optional[int] = None) -> str:
        """Create a new state snapshot"""
        snapshot_id = str(uuid.uuid4())
        
        # Set expiration
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)
        
        # Create snapshot
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            state_type=state_type,
            entity_id=entity_id,
            state_data=state_data,
            metadata=metadata or {},
            expires_at=expires_at
        )
        
        # Store snapshot
        await self._store_snapshot(snapshot)
        
        # Update tracking
        self.active_states[entity_id].add(snapshot_id)
        self.snapshots_created += 1
        
        logger.info(f"Created state snapshot {snapshot_id} for {entity_id} ({state_type.value})")
        return snapshot_id
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Retrieve a state snapshot"""
        # Check memory cache first
        if snapshot_id in self.memory_cache:
            snapshot = self.memory_cache[snapshot_id]
            snapshot.last_accessed = datetime.now(timezone.utc)
            self.cache_hits += 1
            return snapshot
        
        self.cache_misses += 1
        
        # Try Redis
        if self.redis_client:
            try:
                data = self.redis_client.get(f"snapshot:{snapshot_id}")
                if data:
                    snapshot = self._deserialize_snapshot(data)
                    if snapshot and not snapshot.is_expired:
                        self.memory_cache[snapshot_id] = snapshot
                        snapshot.last_accessed = datetime.now(timezone.utc)
                        return snapshot
            except Exception as e:
                logger.warning(f"Failed to retrieve from Redis: {e}")
        
        # Try SQLite
        if self.sqlite_db:
            snapshot = await self._load_snapshot_from_sqlite(snapshot_id)
            if snapshot and not snapshot.is_expired:
                self.memory_cache[snapshot_id] = snapshot
                # Cache in Redis if available
                if self.redis_client:
                    try:
                        serialized = self._serialize_snapshot(snapshot)
                        self.redis_client.setex(f"snapshot:{snapshot_id}", 3600, serialized)
                    except Exception as e:
                        logger.warning(f"Failed to cache in Redis: {e}")
                return snapshot
        
        return None
    
    async def update_snapshot(self, snapshot_id: str, state_data: Dict[str, Any], 
                            metadata: Dict[str, Any] = None) -> bool:
        """Update an existing snapshot"""
        snapshot = await self.get_snapshot(snapshot_id)
        if not snapshot:
            return False
        
        # Update data
        snapshot.state_data = state_data
        if metadata:
            snapshot.metadata.update(metadata)
        snapshot.last_accessed = datetime.now(timezone.utc)
        
        # Store updated snapshot
        await self._store_snapshot(snapshot)
        return True
    
    async def create_checkpoint(self, entity_id: str, checkpoint_data: Dict[str, Any]) -> str:
        """Create a checkpoint for resumable operations"""
        checkpoint_id = f"checkpoint_{entity_id}_{int(datetime.now(timezone.utc).timestamp())}"
        
        # Determine the next checkpoint sequence
        existing_snapshots = [
            await self.get_snapshot(sid) for sid in self.active_states[entity_id]
        ]
        existing_snapshots = [s for s in existing_snapshots if s]
        
        sequence = max([s.checkpoint_sequence for s in existing_snapshots], default=0) + 1
        
        # Create snapshot with checkpoint data
        snapshot_id = await self.create_snapshot(
            StateType.TASK_STATE,
            entity_id,
            checkpoint_data,
            metadata={'checkpoint_id': checkpoint_id, 'sequence': sequence}
        )
        
        snapshot = await self.get_snapshot(snapshot_id)
        if snapshot:
            snapshot.checkpoint_sequence = sequence
            await self._store_snapshot(snapshot)
        
        logger.info(f"Created checkpoint {checkpoint_id} for {entity_id} (sequence: {sequence})")
        return checkpoint_id
    
    async def resume_from_checkpoint(self, entity_id: str, 
                                   checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Resume from the latest or specified checkpoint"""
        entity_snapshots = []
        for snapshot_id in self.active_states[entity_id]:
            snapshot = await self.get_snapshot(snapshot_id)
            if snapshot and not snapshot.is_expired:
                entity_snapshots.append(snapshot)
        
        if not entity_snapshots:
            return None
        
        # Find the appropriate checkpoint
        target_snapshot = None
        if checkpoint_id:
            # Find specific checkpoint
            for snapshot in entity_snapshots:
                if snapshot.metadata.get('checkpoint_id') == checkpoint_id:
                    target_snapshot = snapshot
                    break
        else:
            # Find latest checkpoint
            target_snapshot = max(entity_snapshots, key=lambda s: s.checkpoint_sequence)
        
        if not target_snapshot:
            return None
        
        self.resumptions_executed += 1
        logger.info(f"Resuming {entity_id} from checkpoint {target_snapshot.metadata.get('checkpoint_id')}")
        
        return {
            'snapshot_id': target_snapshot.snapshot_id,
            'state_data': target_snapshot.state_data,
            'metadata': target_snapshot.metadata,
            'checkpoint_sequence': target_snapshot.checkpoint_sequence,
            'created_at': target_snapshot.created_at
        }
    
    async def create_context_frame(self, entity_id: str, frame_type: str, 
                                 local_state: Dict[str, Any] = None,
                                 shared_state: Dict[str, Any] = None) -> str:
        """Create a context frame for detailed resumption"""
        frame_id = str(uuid.uuid4())
        
        frame = ContextFrame(
            frame_id=frame_id,
            entity_id=entity_id,
            frame_type=frame_type,
            local_state=local_state or {},
            shared_state=shared_state or {}
        )
        
        self.context_frames[frame_id] = frame
        await self._store_context_frame(frame)
        
        logger.info(f"Created context frame {frame_id} for {entity_id} ({frame_type})")
        return frame_id
    
    async def get_context_frame(self, frame_id: str) -> Optional[ContextFrame]:
        """Retrieve a context frame"""
        # Check memory first
        if frame_id in self.context_frames:
            return self.context_frames[frame_id]
        
        # Load from storage
        frame = await self._load_context_frame(frame_id)
        if frame:
            self.context_frames[frame_id] = frame
        
        return frame
    
    async def update_context_frame(self, frame_id: str, **updates) -> bool:
        """Update a context frame"""
        frame = await self.get_context_frame(frame_id)
        if not frame:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(frame, key):
                setattr(frame, key, value)
        
        frame.last_updated = datetime.now(timezone.utc)
        await self._store_context_frame(frame)
        return True
    
    async def add_resumption_context(self, entity_id: str, context: Dict[str, Any]) -> bool:
        """Add context information for resumption"""
        # Find or create the latest context frame
        entity_frames = [
            frame for frame in self.context_frames.values()
            if frame.entity_id == entity_id
        ]
        
        if entity_frames:
            # Update latest frame
            latest_frame = max(entity_frames, key=lambda f: f.last_updated)
            latest_frame.execution_stack.append(context)
            latest_frame.last_updated = datetime.now(timezone.utc)
            await self._store_context_frame(latest_frame)
        else:
            # Create new frame
            frame_id = await self.create_context_frame(entity_id, "auto_context")
            frame = await self.get_context_frame(frame_id)
            if frame:
                frame.execution_stack.append(context)
                await self._store_context_frame(frame)
        
        return True
    
    async def get_resumption_plan(self, entity_id: str) -> Dict[str, Any]:
        """Get a comprehensive resumption plan for an entity"""
        # Get all snapshots
        snapshots = []
        for snapshot_id in self.active_states[entity_id]:
            snapshot = await self.get_snapshot(snapshot_id)
            if snapshot and not snapshot.is_expired:
                snapshots.append(snapshot)
        
        # Get context frames
        frames = [
            frame for frame in self.context_frames.values()
            if frame.entity_id == entity_id
        ]
        
        if not snapshots and not frames:
            return {"error": "No recoverable state found"}
        
        # Build resumption plan
        latest_snapshot = max(snapshots, key=lambda s: s.checkpoint_sequence) if snapshots else None
        latest_frame = max(frames, key=lambda f: f.last_updated) if frames else None
        
        plan = {
            "entity_id": entity_id,
            "resumable": True,
            "last_checkpoint": None,
            "context_available": False,
            "required_agents": set(),
            "missing_dependencies": set(),
            "recommended_actions": []
        }
        
        if latest_snapshot:
            plan["last_checkpoint"] = {
                "snapshot_id": latest_snapshot.snapshot_id,
                "checkpoint_sequence": latest_snapshot.checkpoint_sequence,
                "created_at": latest_snapshot.created_at.isoformat(),
                "state_summary": self._summarize_state(latest_snapshot.state_data)
            }
        
        if latest_frame:
            plan["context_available"] = True
            plan["required_agents"] = latest_frame.required_agents
            plan["missing_dependencies"] = latest_frame.waiting_for
            
            # Generate recommended actions
            if latest_frame.next_actions:
                plan["recommended_actions"] = latest_frame.next_actions
            else:
                # Infer actions from context
                plan["recommended_actions"] = self._infer_next_actions(latest_frame)
        
        return plan
    
    def _summarize_state(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of state data"""
        summary = {
            "total_keys": len(state_data),
            "data_types": {},
            "size_estimate": len(str(state_data))
        }
        
        for key, value in state_data.items():
            value_type = type(value).__name__
            if value_type not in summary["data_types"]:
                summary["data_types"][value_type] = 0
            summary["data_types"][value_type] += 1
        
        return summary
    
    def _infer_next_actions(self, frame: ContextFrame) -> List[Dict[str, Any]]:
        """Infer next actions from context frame"""
        actions = []
        
        # Check for pending responses
        if frame.pending_responses:
            actions.append({
                "type": "await_responses",
                "description": f"Wait for {len(frame.pending_responses)} pending responses",
                "details": list(frame.pending_responses.keys())
            })
        
        # Check for missing agents
        missing_agents = frame.required_agents - frame.available_agents
        if missing_agents:
            actions.append({
                "type": "locate_agents",
                "description": f"Locate {len(missing_agents)} required agents",
                "details": list(missing_agents)
            })
        
        # Check execution stack
        if frame.execution_stack:
            last_execution = frame.execution_stack[-1]
            if last_execution.get("status") == "in_progress":
                actions.append({
                    "type": "resume_execution",
                    "description": f"Resume {last_execution.get('operation', 'operation')}",
                    "details": last_execution
                })
        
        return actions
    
    async def _store_snapshot(self, snapshot: StateSnapshot):
        """Store snapshot across all configured backends"""
        # Memory cache
        self.memory_cache[snapshot.snapshot_id] = snapshot
        
        # Redis cache
        if self.redis_client:
            try:
                serialized = self._serialize_snapshot(snapshot)
                # Store with TTL based on expiration
                ttl = 3600  # Default 1 hour
                if snapshot.expires_at:
                    ttl = int((snapshot.expires_at - datetime.now(timezone.utc)).total_seconds())
                    ttl = max(300, ttl)  # Minimum 5 minutes
                
                self.redis_client.setex(f"snapshot:{snapshot.snapshot_id}", ttl, serialized)
            except Exception as e:
                logger.warning(f"Failed to store snapshot in Redis: {e}")
        
        # SQLite persistence
        if self.sqlite_db:
            await self._store_snapshot_in_sqlite(snapshot)
    
    def _serialize_snapshot(self, snapshot: StateSnapshot) -> bytes:
        """Serialize snapshot for storage"""
        data = snapshot.to_dict()
        json_data = json.dumps(data, default=str)
        return gzip.compress(json_data.encode())
    
    def _deserialize_snapshot(self, data: bytes) -> Optional[StateSnapshot]:
        """Deserialize snapshot from storage"""
        try:
            json_data = gzip.decompress(data).decode()
            data_dict = json.loads(json_data)
            
            # Convert back to StateSnapshot
            snapshot = StateSnapshot(
                snapshot_id=data_dict['snapshot_id'],
                state_type=StateType(data_dict['state_type']),
                entity_id=data_dict['entity_id'],
                state_data=data_dict['state_data'],
                metadata=data_dict['metadata']
            )
            
            # Restore datetime fields
            snapshot.created_at = datetime.fromisoformat(data_dict['created_at'].replace('Z', '+00:00'))
            snapshot.last_accessed = datetime.fromisoformat(data_dict['last_accessed'].replace('Z', '+00:00'))
            if data_dict.get('expires_at'):
                snapshot.expires_at = datetime.fromisoformat(data_dict['expires_at'].replace('Z', '+00:00'))
            
            # Restore other fields
            snapshot.status = StateStatus(data_dict['status'])
            snapshot.parent_snapshot_id = data_dict.get('parent_snapshot_id')
            snapshot.checkpoint_sequence = data_dict.get('checkpoint_sequence', 0)
            snapshot.participants = set(data_dict['participants'])
            snapshot.dependencies = set(data_dict['dependencies'])
            snapshot.related_states = set(data_dict['related_states'])
            
            return snapshot
        except Exception as e:
            logger.error(f"Failed to deserialize snapshot: {e}")
            return None
    
    async def _store_snapshot_in_sqlite(self, snapshot: StateSnapshot):
        """Store snapshot in SQLite database"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            # Compress state data
            state_data_compressed = gzip.compress(json.dumps(snapshot.state_data).encode())
            
            cursor.execute('''
                INSERT OR REPLACE INTO state_snapshots (
                    snapshot_id, state_type, entity_id, status, created_at,
                    expires_at, last_accessed, parent_snapshot_id, checkpoint_sequence,
                    participants, dependencies, related_states, metadata, state_data_compressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.snapshot_id,
                snapshot.state_type.value,
                snapshot.entity_id,
                snapshot.status.value,
                snapshot.created_at.isoformat(),
                snapshot.expires_at.isoformat() if snapshot.expires_at else None,
                snapshot.last_accessed.isoformat(),
                snapshot.parent_snapshot_id,
                snapshot.checkpoint_sequence,
                json.dumps(list(snapshot.participants)),
                json.dumps(list(snapshot.dependencies)),
                json.dumps(list(snapshot.related_states)),
                json.dumps(snapshot.metadata),
                state_data_compressed
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store snapshot in SQLite: {e}")
    
    async def _load_snapshot_from_sqlite(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load snapshot from SQLite database"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM state_snapshots WHERE snapshot_id = ?
            ''', (snapshot_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Decompress and reconstruct snapshot
            state_data = json.loads(gzip.decompress(row[13]).decode())
            
            snapshot = StateSnapshot(
                snapshot_id=row[0],
                state_type=StateType(row[1]),
                entity_id=row[2],
                state_data=state_data,
                metadata=json.loads(row[12])
            )
            
            snapshot.status = StateStatus(row[3])
            snapshot.created_at = datetime.fromisoformat(row[4])
            if row[5]:
                snapshot.expires_at = datetime.fromisoformat(row[5])
            snapshot.last_accessed = datetime.fromisoformat(row[6])
            snapshot.parent_snapshot_id = row[7]
            snapshot.checkpoint_sequence = row[8] or 0
            snapshot.participants = set(json.loads(row[9]))
            snapshot.dependencies = set(json.loads(row[10]))
            snapshot.related_states = set(json.loads(row[11]))
            
            return snapshot
        except Exception as e:
            logger.error(f"Failed to load snapshot from SQLite: {e}")
            return None
    
    async def _store_context_frame(self, frame: ContextFrame):
        """Store context frame in SQLite"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            # Serialize frame data
            frame_data = {
                'local_state': frame.local_state,
                'shared_state': frame.shared_state,
                'execution_stack': frame.execution_stack,
                'next_actions': frame.next_actions,
                'pending_responses': frame.pending_responses
            }
            frame_data_compressed = gzip.compress(json.dumps(frame_data).encode())
            
            cursor.execute('''
                INSERT OR REPLACE INTO context_frames (
                    frame_id, entity_id, frame_type, resume_point, created_at,
                    last_updated, required_agents, available_agents, waiting_for,
                    frame_data_compressed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                frame.frame_id,
                frame.entity_id,
                frame.frame_type,
                frame.resume_point,
                frame.created_at.isoformat(),
                frame.last_updated.isoformat(),
                json.dumps(list(frame.required_agents)),
                json.dumps(list(frame.available_agents)),
                json.dumps(list(frame.waiting_for)),
                frame_data_compressed
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store context frame: {e}")
    
    async def _load_context_frame(self, frame_id: str) -> Optional[ContextFrame]:
        """Load context frame from SQLite"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM context_frames WHERE frame_id = ?
            ''', (frame_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Decompress frame data
            frame_data = json.loads(gzip.decompress(row[9]).decode())
            
            frame = ContextFrame(
                frame_id=row[0],
                entity_id=row[1],
                frame_type=row[2],
                local_state=frame_data['local_state'],
                shared_state=frame_data['shared_state'],
                execution_stack=frame_data['execution_stack'],
                next_actions=frame_data['next_actions'],
                pending_responses=frame_data['pending_responses']
            )
            
            frame.resume_point = row[3]
            frame.created_at = datetime.fromisoformat(row[4])
            frame.last_updated = datetime.fromisoformat(row[5])
            frame.required_agents = set(json.loads(row[6]))
            frame.available_agents = set(json.loads(row[7]))
            frame.waiting_for = set(json.loads(row[8]))
            
            return frame
        except Exception as e:
            logger.error(f"Failed to load context frame: {e}")
            return None
    
    async def cleanup_expired_states(self):
        """Clean up expired states and snapshots"""
        current_time = datetime.now(timezone.utc)
        
        # Clean memory cache
        expired_snapshots = [
            sid for sid, snapshot in self.memory_cache.items()
            if snapshot.is_expired
        ]
        
        for sid in expired_snapshots:
            del self.memory_cache[sid]
        
        # Clean SQLite
        if self.sqlite_db:
            try:
                conn = sqlite3.connect(self.sqlite_db)
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM state_snapshots 
                    WHERE expires_at < ?
                ''', (current_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                logger.info(f"Cleaned up {deleted_count + len(expired_snapshots)} expired state snapshots")
            except Exception as e:
                logger.error(f"Failed to cleanup SQLite states: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics"""
        return {
            "performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_ratio": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "snapshots_created": self.snapshots_created,
                "resumptions_executed": self.resumptions_executed
            },
            "storage": {
                "memory_snapshots": len(self.memory_cache),
                "context_frames": len(self.context_frames),
                "active_entities": len(self.active_states),
                "redis_available": self.redis_client is not None,
                "sqlite_available": self.sqlite_db is not None
            },
            "system": {
                "total_dependencies": sum(len(deps) for deps in self.state_dependencies.values()),
                "average_dependencies_per_entity": sum(len(deps) for deps in self.state_dependencies.values()) / max(1, len(self.state_dependencies))
            }
        } 