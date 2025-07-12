"""
Advanced Context Manager for Seamless Task Resumption
Provides granular state tracking, dependency resolution, and intelligent resumption
"""

import asyncio
import json
import gzip
import pickle
import uuid
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import redis
import logging

logger = logging.getLogger(__name__)


class ContextType(Enum):
    CONVERSATION = "conversation"
    TASK_EXECUTION = "task_execution"
    COLLABORATION_SESSION = "collaboration_session"
    AGENT_STATE = "agent_state"
    WORKFLOW_STEP = "workflow_step"
    DECISION_POINT = "decision_point"
    RESOURCE_ACCESS = "resource_access"


class ResumptionStrategy(Enum):
    EXACT_CONTINUATION = "exact_continuation"
    INTELLIGENT_RECONSTRUCTION = "intelligent_reconstruction"
    FALLBACK_TO_CHECKPOINT = "fallback_to_checkpoint"
    FRESH_START_WITH_CONTEXT = "fresh_start_with_context"


@dataclass
class ContextFrame:
    """Detailed context frame for precise resumption"""
    frame_id: str
    agent_id: str
    context_type: ContextType
    timestamp: str
    parent_frame_id: Optional[str] = None
    
    # Execution state
    current_step: str = ""
    step_progress: float = 0.0
    local_variables: Dict[str, Any] = None
    execution_stack: List[Dict] = None
    pending_operations: List[Dict] = None
    
    # Collaboration context
    active_collaborators: Set[str] = None
    shared_resources: Dict[str, Any] = None
    communication_history: List[Dict] = None
    dependency_graph: Dict[str, List[str]] = None
    
    # Decision and reasoning context
    decision_history: List[Dict] = None
    reasoning_chain: List[str] = None
    available_options: List[Dict] = None
    selected_strategy: str = ""
    
    # Metadata
    priority: int = 0
    estimated_completion_time: Optional[str] = None
    required_capabilities: List[str] = None
    external_dependencies: List[str] = None
    
    def __post_init__(self):
        if self.local_variables is None:
            self.local_variables = {}
        if self.execution_stack is None:
            self.execution_stack = []
        if self.pending_operations is None:
            self.pending_operations = []
        if self.active_collaborators is None:
            self.active_collaborators = set()
        if self.shared_resources is None:
            self.shared_resources = {}
        if self.communication_history is None:
            self.communication_history = []
        if self.dependency_graph is None:
            self.dependency_graph = {}
        if self.decision_history is None:
            self.decision_history = []
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.available_options is None:
            self.available_options = []
        if self.required_capabilities is None:
            self.required_capabilities = []
        if self.external_dependencies is None:
            self.external_dependencies = []


class DependencyResolver:
    """Intelligent dependency resolution for context resumption"""
    
    def __init__(self):
        self.dependency_cache: Dict[str, Dict] = {}
        self.resolution_strategies: Dict[str, Any] = {}
    
    async def analyze_dependencies(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Analyze all dependencies for a context frame"""
        dependencies = {
            'agent_dependencies': [],
            'resource_dependencies': [],
            'state_dependencies': [],
            'external_dependencies': [],
            'resolution_plan': []
        }
        
        # Analyze agent dependencies
        for collaborator in context_frame.active_collaborators:
            dep_info = await self._check_agent_availability(collaborator)
            dependencies['agent_dependencies'].append({
                'agent_id': collaborator,
                'status': dep_info['status'],
                'last_seen': dep_info.get('last_seen'),
                'alternative_agents': dep_info.get('alternatives', [])
            })
        
        # Analyze resource dependencies
        for resource_id, resource_info in context_frame.shared_resources.items():
            resource_status = await self._check_resource_availability(resource_id, resource_info)
            dependencies['resource_dependencies'].append({
                'resource_id': resource_id,
                'type': resource_info.get('type', 'unknown'),
                'status': resource_status['status'],
                'recovery_options': resource_status.get('recovery_options', [])
            })
        
        # Analyze state dependencies
        for step in context_frame.execution_stack:
            if step.get('depends_on'):
                state_status = await self._check_state_dependency(step['depends_on'])
                dependencies['state_dependencies'].append({
                    'dependency': step['depends_on'],
                    'status': state_status['status'],
                    'recovery_method': state_status.get('recovery_method')
                })
        
        # Create resolution plan
        dependencies['resolution_plan'] = await self._create_resolution_plan(dependencies)
        
        return dependencies
    
    async def _check_agent_availability(self, agent_id: str) -> Dict[str, Any]:
        """Check if an agent is available for collaboration"""
        # Simulate agent availability check
        return {
            'status': 'available',  # available, unavailable, busy
            'last_seen': datetime.now(timezone.utc).isoformat(),
            'alternatives': []  # Could suggest similar agents
        }
    
    async def _check_resource_availability(self, resource_id: str, resource_info: Dict) -> Dict[str, Any]:
        """Check if a resource is still available"""
        return {
            'status': 'available',  # available, unavailable, modified
            'recovery_options': ['recreate', 'use_alternative']
        }
    
    async def _check_state_dependency(self, dependency: str) -> Dict[str, Any]:
        """Check if a state dependency can be resolved"""
        return {
            'status': 'resolved',  # resolved, missing, corrupted
            'recovery_method': 'restore_from_backup'
        }
    
    async def _create_resolution_plan(self, dependencies: Dict[str, Any]) -> List[Dict]:
        """Create a plan to resolve all dependencies"""
        plan = []
        
        # Handle missing agents
        for agent_dep in dependencies['agent_dependencies']:
            if agent_dep['status'] != 'available':
                plan.append({
                    'action': 'wait_for_agent' if agent_dep['status'] == 'busy' else 'find_alternative',
                    'target': agent_dep['agent_id'],
                    'priority': 'high',
                    'timeout': 300  # 5 minutes
                })
        
        # Handle missing resources
        for resource_dep in dependencies['resource_dependencies']:
            if resource_dep['status'] != 'available':
                plan.append({
                    'action': 'restore_resource',
                    'target': resource_dep['resource_id'],
                    'method': resource_dep['recovery_options'][0] if resource_dep['recovery_options'] else 'recreate',
                    'priority': 'medium'
                })
        
        # Handle state dependencies
        for state_dep in dependencies['state_dependencies']:
            if state_dep['status'] != 'resolved':
                plan.append({
                    'action': 'restore_state',
                    'target': state_dep['dependency'],
                    'method': state_dep.get('recovery_method', 'restore_from_backup'),
                    'priority': 'high'
                })
        
        return sorted(plan, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])


class IntelligentResumptionEngine:
    """Engine for intelligent task resumption with context reconstruction"""
    
    def __init__(self):
        self.dependency_resolver = DependencyResolver()
        self.resumption_strategies: Dict[ResumptionStrategy, Any] = {
            ResumptionStrategy.EXACT_CONTINUATION: self._exact_continuation,
            ResumptionStrategy.INTELLIGENT_RECONSTRUCTION: self._intelligent_reconstruction,
            ResumptionStrategy.FALLBACK_TO_CHECKPOINT: self._fallback_to_checkpoint,
            ResumptionStrategy.FRESH_START_WITH_CONTEXT: self._fresh_start_with_context
        }
    
    async def determine_resumption_strategy(self, context_frame: ContextFrame) -> ResumptionStrategy:
        """Determine the best resumption strategy based on context analysis"""
        
        # Analyze dependencies
        dependencies = await self.dependency_resolver.analyze_dependencies(context_frame)
        
        # Check how much context is intact
        context_integrity = await self._assess_context_integrity(context_frame, dependencies)
        
        # Determine strategy based on integrity and dependencies
        if context_integrity['score'] > 0.9 and not dependencies['resolution_plan']:
            return ResumptionStrategy.EXACT_CONTINUATION
        elif context_integrity['score'] > 0.7:
            return ResumptionStrategy.INTELLIGENT_RECONSTRUCTION
        elif context_integrity['score'] > 0.4:
            return ResumptionStrategy.FALLBACK_TO_CHECKPOINT
        else:
            return ResumptionStrategy.FRESH_START_WITH_CONTEXT
    
    async def _assess_context_integrity(self, context_frame: ContextFrame, dependencies: Dict) -> Dict[str, Any]:
        """Assess how much of the original context is still intact"""
        integrity_factors = {
            'agent_availability': 0.0,
            'resource_availability': 0.0,
            'state_consistency': 0.0,
            'execution_continuity': 0.0
        }
        
        # Agent availability
        available_agents = sum(1 for dep in dependencies['agent_dependencies'] if dep['status'] == 'available')
        total_agents = len(dependencies['agent_dependencies']) or 1
        integrity_factors['agent_availability'] = available_agents / total_agents
        
        # Resource availability
        available_resources = sum(1 for dep in dependencies['resource_dependencies'] if dep['status'] == 'available')
        total_resources = len(dependencies['resource_dependencies']) or 1
        integrity_factors['resource_availability'] = available_resources / total_resources
        
        # State consistency
        resolved_states = sum(1 for dep in dependencies['state_dependencies'] if dep['status'] == 'resolved')
        total_states = len(dependencies['state_dependencies']) or 1
        integrity_factors['state_consistency'] = resolved_states / total_states
        
        # Execution continuity (how far through the task were we?)
        integrity_factors['execution_continuity'] = context_frame.step_progress
        
        # Calculate overall score
        weights = {'agent_availability': 0.3, 'resource_availability': 0.2, 'state_consistency': 0.3, 'execution_continuity': 0.2}
        overall_score = sum(integrity_factors[factor] * weights[factor] for factor in weights)
        
        return {
            'score': overall_score,
            'factors': integrity_factors,
            'recommendation': 'high_integrity' if overall_score > 0.8 else 'medium_integrity' if overall_score > 0.5 else 'low_integrity'
        }
    
    async def _exact_continuation(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Resume exactly where we left off"""
        return {
            'strategy': 'exact_continuation',
            'resumption_point': context_frame.current_step,
            'local_variables': context_frame.local_variables,
            'execution_stack': context_frame.execution_stack,
            'collaborators': list(context_frame.active_collaborators),
            'confidence': 0.95
        }
    
    async def _intelligent_reconstruction(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Intelligently reconstruct context with some adaptation"""
        # Find the best stable resumption point
        stable_step = await self._find_stable_resumption_point(context_frame)
        
        # Reconstruct necessary context
        reconstructed_context = await self._reconstruct_context(context_frame, stable_step)
        
        return {
            'strategy': 'intelligent_reconstruction',
            'resumption_point': stable_step,
            'reconstructed_context': reconstructed_context,
            'adaptations_made': ['agent_substitution', 'resource_recreation'],
            'confidence': 0.8
        }
    
    async def _fallback_to_checkpoint(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Fall back to the most recent stable checkpoint"""
        checkpoint = await self._find_latest_checkpoint(context_frame)
        
        return {
            'strategy': 'fallback_to_checkpoint',
            'checkpoint_id': checkpoint['checkpoint_id'],
            'resumption_point': checkpoint['step'],
            'lost_progress': context_frame.step_progress - checkpoint.get('progress', 0),
            'confidence': 0.6
        }
    
    async def _fresh_start_with_context(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Start fresh but with available context as background"""
        available_context = await self._extract_usable_context(context_frame)
        
        return {
            'strategy': 'fresh_start_with_context',
            'available_context': available_context,
            'recommendations': ['use_previous_reasoning', 'leverage_partial_results'],
            'confidence': 0.4
        }
    
    async def _find_stable_resumption_point(self, context_frame: ContextFrame) -> str:
        """Find the most stable point to resume from"""
        # Look for completed steps that don't depend on missing resources
        for i, step in enumerate(reversed(context_frame.execution_stack)):
            if step.get('status') == 'completed' and step.get('stable', True):
                return step['step_id']
        
        return context_frame.current_step
    
    async def _reconstruct_context(self, context_frame: ContextFrame, resumption_point: str) -> Dict[str, Any]:
        """Reconstruct context for resumption"""
        return {
            'variables': {k: v for k, v in context_frame.local_variables.items() if not k.startswith('_temp')},
            'collaborators': await self._find_available_collaborators(context_frame.active_collaborators),
            'resources': await self._recreate_missing_resources(context_frame.shared_resources)
        }
    
    async def _find_latest_checkpoint(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Find the latest stable checkpoint"""
        # Simulate finding a checkpoint
        return {
            'checkpoint_id': f"checkpoint_{uuid.uuid4().hex[:8]}",
            'step': 'step_5',
            'progress': 0.6,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _extract_usable_context(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Extract any context that's still usable"""
        return {
            'reasoning_history': context_frame.reasoning_chain,
            'decision_history': context_frame.decision_history,
            'partial_results': {k: v for k, v in context_frame.local_variables.items() if k.startswith('result_')},
            'learned_preferences': context_frame.shared_resources.get('preferences', {})
        }
    
    async def _find_available_collaborators(self, original_collaborators: Set[str]) -> List[str]:
        """Find available collaborators or suitable substitutes"""
        # Simulate finding available collaborators
        return list(original_collaborators)
    
    async def _recreate_missing_resources(self, original_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Recreate or substitute missing resources"""
        return original_resources


class AdvancedContextManager:
    """Main context manager with advanced resumption capabilities"""
    
    def __init__(self, storage_config: Dict[str, Any]):
        self.storage_config = storage_config
        self.dependency_resolver = DependencyResolver()
        self.resumption_engine = IntelligentResumptionEngine()
        
        # Storage backends
        self.sqlite_conn = None
        self.redis_client = None
        
        # Context tracking
        self.active_contexts: Dict[str, ContextFrame] = {}
        self.context_history: List[ContextFrame] = []
        
    async def initialize(self):
        """Initialize storage backends"""
        if self.storage_config.get('sqlite', {}).get('enabled', True):
            await self._initialize_sqlite()
        
        if self.storage_config.get('redis', {}).get('enabled', False):
            await self._initialize_redis()
    
    async def _initialize_sqlite(self):
        """Initialize SQLite with enhanced schema"""
        db_path = self.storage_config.get('sqlite', {}).get('path', 'contexts.db')
        self.sqlite_conn = sqlite3.connect(db_path)
        
        # Create enhanced tables
        self.sqlite_conn.execute('''
            CREATE TABLE IF NOT EXISTS context_frames (
                frame_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                context_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parent_frame_id TEXT,
                current_step TEXT,
                step_progress REAL,
                priority INTEGER,
                estimated_completion TEXT,
                frame_data BLOB,
                dependencies TEXT,
                INDEX(agent_id),
                INDEX(context_type),
                INDEX(timestamp)
            )
        ''')
        
        self.sqlite_conn.execute('''
            CREATE TABLE IF NOT EXISTS resumption_logs (
                log_id TEXT PRIMARY KEY,
                frame_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                confidence REAL,
                execution_time REAL,
                timestamp TEXT NOT NULL,
                details TEXT,
                INDEX(frame_id),
                INDEX(timestamp)
            )
        ''')
        
        self.sqlite_conn.commit()
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        redis_config = self.storage_config.get('redis', {})
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            decode_responses=True
        )
    
    async def create_context_frame(self, 
                                 agent_id: str, 
                                 context_type: ContextType, 
                                 initial_data: Dict[str, Any] = None) -> str:
        """Create a new context frame"""
        frame_id = str(uuid.uuid4())
        
        context_frame = ContextFrame(
            frame_id=frame_id,
            agent_id=agent_id,
            context_type=context_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **(initial_data or {})
        )
        
        # Store in memory and persistent storage
        self.active_contexts[frame_id] = context_frame
        await self._persist_context_frame(context_frame)
        
        logger.info(f"Created context frame {frame_id} for agent {agent_id}")
        return frame_id
    
    async def update_context_frame(self, frame_id: str, updates: Dict[str, Any]):
        """Update an existing context frame"""
        if frame_id not in self.active_contexts:
            raise ValueError(f"Context frame {frame_id} not found")
        
        context_frame = self.active_contexts[frame_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(context_frame, key):
                setattr(context_frame, key, value)
        
        # Update timestamp
        context_frame.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Persist changes
        await self._persist_context_frame(context_frame)
    
    async def resume_from_context(self, frame_id: str) -> Dict[str, Any]:
        """Resume execution from a context frame"""
        # Load context frame if not in memory
        if frame_id not in self.active_contexts:
            context_frame = await self._load_context_frame(frame_id)
            if not context_frame:
                raise ValueError(f"Context frame {frame_id} not found")
            self.active_contexts[frame_id] = context_frame
        else:
            context_frame = self.active_contexts[frame_id]
        
        # Determine resumption strategy
        strategy = await self.resumption_engine.determine_resumption_strategy(context_frame)
        
        # Execute resumption
        start_time = time.time()
        resumption_result = await self.resumption_engine.resumption_strategies[strategy](context_frame)
        execution_time = time.time() - start_time
        
        # Log resumption attempt
        await self._log_resumption_attempt(frame_id, strategy, resumption_result, execution_time)
        
        logger.info(f"Resumed context {frame_id} using strategy {strategy.value}")
        return resumption_result
    
    async def _persist_context_frame(self, context_frame: ContextFrame):
        """Persist context frame to storage"""
        if self.sqlite_conn:
            # Serialize complex data
            frame_data = gzip.compress(pickle.dumps(asdict(context_frame)))
            dependencies = json.dumps(context_frame.dependency_graph)
            
            self.sqlite_conn.execute('''
                INSERT OR REPLACE INTO context_frames 
                (frame_id, agent_id, context_type, timestamp, parent_frame_id, 
                 current_step, step_progress, priority, estimated_completion, 
                 frame_data, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                context_frame.frame_id,
                context_frame.agent_id,
                context_frame.context_type.value,
                context_frame.timestamp,
                context_frame.parent_frame_id,
                context_frame.current_step,
                context_frame.step_progress,
                context_frame.priority,
                context_frame.estimated_completion_time,
                frame_data,
                dependencies
            ))
            self.sqlite_conn.commit()
        
        # Also cache in Redis if available
        if self.redis_client:
            await self._cache_in_redis(context_frame)
    
    async def _cache_in_redis(self, context_frame: ContextFrame):
        """Cache context frame in Redis for fast access"""
        key = f"context:{context_frame.frame_id}"
        value = json.dumps(asdict(context_frame), default=str)
        
        await self.redis_client.setex(key, 3600, value)  # 1 hour expiry
    
    async def _load_context_frame(self, frame_id: str) -> Optional[ContextFrame]:
        """Load context frame from storage"""
        # Try Redis first
        if self.redis_client:
            cached = await self.redis_client.get(f"context:{frame_id}")
            if cached:
                data = json.loads(cached)
                return ContextFrame(**data)
        
        # Fall back to SQLite
        if self.sqlite_conn:
            cursor = self.sqlite_conn.execute(
                'SELECT frame_data FROM context_frames WHERE frame_id = ?',
                (frame_id,)
            )
            row = cursor.fetchone()
            if row:
                frame_data = pickle.loads(gzip.decompress(row[0]))
                return ContextFrame(**frame_data)
        
        return None
    
    async def _log_resumption_attempt(self, frame_id: str, strategy: ResumptionStrategy, result: Dict, execution_time: float):
        """Log resumption attempt for analysis"""
        log_id = str(uuid.uuid4())
        
        if self.sqlite_conn:
            self.sqlite_conn.execute('''
                INSERT INTO resumption_logs 
                (log_id, frame_id, strategy, success, confidence, execution_time, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_id,
                frame_id,
                strategy.value,
                result.get('confidence', 0) > 0.5,
                result.get('confidence', 0),
                execution_time,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(result)
            ))
            self.sqlite_conn.commit()
    
    def get_active_contexts(self, agent_id: Optional[str] = None) -> List[ContextFrame]:
        """Get active contexts, optionally filtered by agent"""
        contexts = list(self.active_contexts.values())
        if agent_id:
            contexts = [ctx for ctx in contexts if ctx.agent_id == agent_id]
        return contexts
    
    async def cleanup_expired_contexts(self, max_age_hours: int = 24):
        """Clean up expired contexts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        expired_frames = []
        for frame_id, context_frame in list(self.active_contexts.items()):
            frame_time = datetime.fromisoformat(context_frame.timestamp)
            if frame_time < cutoff_time:
                expired_frames.append(frame_id)
        
        for frame_id in expired_frames:
            del self.active_contexts[frame_id]
            if self.redis_client:
                await self.redis_client.delete(f"context:{frame_id}")
        
        logger.info(f"Cleaned up {len(expired_frames)} expired context frames") 