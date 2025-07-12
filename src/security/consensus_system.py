"""
Consensus-Based Agent Reputation System
Honest implementation - rewards based on actual task completion and collaboration quality
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import hashlib

logger = logging.getLogger(__name__)


class TaskDifficulty(Enum):
    TRIVIAL = 1.0
    EASY = 1.5
    MEDIUM = 2.0
    HARD = 3.0
    EXPERT = 5.0


class CollaborationQuality(Enum):
    POOR = 0.5
    FAIR = 1.0
    GOOD = 1.5
    EXCELLENT = 2.0


@dataclass
class TaskCompletion:
    """Record of a completed task or subtask"""
    task_id: str
    agent_id: str
    subtask_id: Optional[str] = None
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    completion_time: float = 0.0  # seconds
    quality_score: float = 1.0  # 0.0 to 1.0
    collaboration_agents: Set[str] = None
    collaboration_quality: CollaborationQuality = CollaborationQuality.FAIR
    timestamp: str = ""
    verified_by: Set[str] = None  # Agents that verified this completion
    
    def __post_init__(self):
        if self.collaboration_agents is None:
            self.collaboration_agents = set()
        if self.verified_by is None:
            self.verified_by = set()
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class AgentReputation:
    """Agent reputation metrics"""
    agent_id: str
    total_points: float = 0.0
    completed_tasks: int = 0
    collaboration_score: float = 1.0
    reliability_score: float = 1.0
    security_violations: int = 0
    last_active: str = ""
    trust_level: str = "UNVERIFIED"  # UNVERIFIED, BASIC, TRUSTED, EXPERT
    
    def __post_init__(self):
        if not self.last_active:
            self.last_active = datetime.now(timezone.utc).isoformat()


class ConsensusReputationSystem:
    """
    Honest consensus system - no exaggerated claims
    Simply tracks task completion and calculates reputation based on real metrics
    """
    
    def __init__(self, db_path: str = "consensus.db"):
        self.db_path = db_path
        self.agent_reputations: Dict[str, AgentReputation] = {}
        self.task_completions: List[TaskCompletion] = []
        self.pending_verifications: Dict[str, TaskCompletion] = {}
        
        # Simple scoring weights - no complex algorithms
        self.scoring_weights = {
            'base_completion': 10.0,
            'difficulty_multiplier': True,
            'quality_multiplier': True,
            'collaboration_bonus': 5.0,
            'verification_bonus': 2.0
        }
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Agent reputations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_reputations (
                    agent_id TEXT PRIMARY KEY,
                    total_points REAL,
                    completed_tasks INTEGER,
                    collaboration_score REAL,
                    reliability_score REAL,
                    security_violations INTEGER,
                    last_active TEXT,
                    trust_level TEXT
                )
            ''')
            
            # Task completions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS task_completions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    agent_id TEXT,
                    subtask_id TEXT,
                    difficulty TEXT,
                    completion_time REAL,
                    quality_score REAL,
                    collaboration_agents TEXT,
                    collaboration_quality TEXT,
                    timestamp TEXT,
                    verified_by TEXT,
                    points_awarded REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Consensus database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize consensus database: {e}")
    
    async def register_agent(self, agent_id: str) -> bool:
        """Register a new agent in the reputation system"""
        if agent_id in self.agent_reputations:
            return False
        
        reputation = AgentReputation(agent_id=agent_id)
        self.agent_reputations[agent_id] = reputation
        
        # Persist to database
        await self._save_agent_reputation(reputation)
        logger.info(f"Registered agent {agent_id} in consensus system")
        return True
    
    async def record_task_completion(self, 
                                   task_id: str,
                                   agent_id: str,
                                   difficulty: TaskDifficulty = TaskDifficulty.MEDIUM,
                                   quality_score: float = 1.0,
                                   completion_time: float = 0.0,
                                   collaborators: Set[str] = None,
                                   subtask_id: Optional[str] = None) -> float:
        """
        Record task completion and calculate points
        Returns points awarded
        """
        if agent_id not in self.agent_reputations:
            await self.register_agent(agent_id)
        
        # Create task completion record
        completion = TaskCompletion(
            task_id=task_id,
            agent_id=agent_id,
            subtask_id=subtask_id,
            difficulty=difficulty,
            completion_time=completion_time,
            quality_score=max(0.0, min(1.0, quality_score)),  # Clamp to 0-1
            collaboration_agents=collaborators or set()
        )
        
        # Calculate points awarded
        points = await self._calculate_points(completion)
        
        # Update agent reputation
        reputation = self.agent_reputations[agent_id]
        reputation.total_points += points
        reputation.completed_tasks += 1
        reputation.last_active = datetime.now(timezone.utc).isoformat()
        
        # Update collaboration scores for all involved agents
        if collaborators:
            await self._update_collaboration_scores(agent_id, collaborators, quality_score)
        
        # Store completion
        self.task_completions.append(completion)
        await self._save_task_completion(completion, points)
        await self._save_agent_reputation(reputation)
        
        logger.info(f"Agent {agent_id} completed task {task_id}, awarded {points:.2f} points")
        return points
    
    async def _calculate_points(self, completion: TaskCompletion) -> float:
        """Calculate points for task completion - straightforward formula"""
        base_points = self.scoring_weights['base_completion']
        
        # Apply difficulty multiplier
        points = base_points * completion.difficulty.value
        
        # Apply quality multiplier
        points *= completion.quality_score
        
        # Collaboration bonus
        if completion.collaboration_agents:
            collab_bonus = len(completion.collaboration_agents) * self.scoring_weights['collaboration_bonus']
            points += collab_bonus
        
        return round(points, 2)
    
    async def _update_collaboration_scores(self, 
                                         primary_agent: str, 
                                         collaborators: Set[str], 
                                         task_quality: float):
        """Update collaboration scores for all agents involved"""
        for collaborator in collaborators:
            if collaborator in self.agent_reputations:
                reputation = self.agent_reputations[collaborator]
                # Simple moving average for collaboration score
                current_score = reputation.collaboration_score
                new_score = (current_score * 0.8) + (task_quality * 0.2)
                reputation.collaboration_score = new_score
                await self._save_agent_reputation(reputation)
    
    async def verify_task_completion(self, 
                                   task_id: str, 
                                   verifier_agent: str,
                                   verification_passed: bool) -> bool:
        """Allow agents to verify each other's task completions"""
        # Find the task completion
        for completion in self.task_completions:
            if completion.task_id == task_id:
                if verification_passed:
                    completion.verified_by.add(verifier_agent)
                    
                    # Award verification bonus to the original agent
                    if completion.agent_id in self.agent_reputations:
                        reputation = self.agent_reputations[completion.agent_id]
                        bonus = self.scoring_weights['verification_bonus']
                        reputation.total_points += bonus
                        await self._save_agent_reputation(reputation)
                        
                        logger.info(f"Task {task_id} verified by {verifier_agent}, "
                                  f"{bonus} bonus points awarded to {completion.agent_id}")
                    return True
        
        return False
    
    def get_agent_reputation(self, agent_id: str) -> Optional[AgentReputation]:
        """Get current reputation for an agent"""
        return self.agent_reputations.get(agent_id)
    
    def get_leaderboard(self, limit: int = 10) -> List[AgentReputation]:
        """Get top agents by total points"""
        sorted_agents = sorted(
            self.agent_reputations.values(),
            key=lambda x: x.total_points,
            reverse=True
        )
        return sorted_agents[:limit]
    
    def get_collaboration_network(self, agent_id: str) -> Dict[str, int]:
        """Get agents this agent has collaborated with and frequency"""
        collaborations = {}
        
        for completion in self.task_completions:
            if completion.agent_id == agent_id:
                for collaborator in completion.collaboration_agents:
                    collaborations[collaborator] = collaborations.get(collaborator, 0) + 1
            elif agent_id in completion.collaboration_agents:
                collaborations[completion.agent_id] = collaborations.get(completion.agent_id, 0) + 1
        
        return collaborations
    
    async def record_security_violation(self, agent_id: str, violation_type: str, severity: str):
        """Record security violation - reduces reputation"""
        if agent_id in self.agent_reputations:
            reputation = self.agent_reputations[agent_id]
            reputation.security_violations += 1
            
            # Reduce points based on severity
            penalties = {'low': 5.0, 'medium': 15.0, 'high': 50.0, 'critical': 100.0}
            penalty = penalties.get(severity, 10.0)
            reputation.total_points = max(0, reputation.total_points - penalty)
            
            # Reduce reliability score
            reputation.reliability_score *= 0.9
            
            await self._save_agent_reputation(reputation)
            logger.warning(f"Security violation recorded for {agent_id}: {violation_type} "
                         f"(severity: {severity}, penalty: {penalty} points)")
    
    async def _save_agent_reputation(self, reputation: AgentReputation):
        """Save agent reputation to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO agent_reputations 
                (agent_id, total_points, completed_tasks, collaboration_score, 
                 reliability_score, security_violations, last_active, trust_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                reputation.agent_id,
                reputation.total_points,
                reputation.completed_tasks,
                reputation.collaboration_score,
                reputation.reliability_score,
                reputation.security_violations,
                reputation.last_active,
                reputation.trust_level
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save agent reputation: {e}")
    
    async def _save_task_completion(self, completion: TaskCompletion, points_awarded: float):
        """Save task completion to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT INTO task_completions 
                (task_id, agent_id, subtask_id, difficulty, completion_time, 
                 quality_score, collaboration_agents, collaboration_quality, 
                 timestamp, verified_by, points_awarded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                completion.task_id,
                completion.agent_id,
                completion.subtask_id,
                completion.difficulty.name,
                completion.completion_time,
                completion.quality_score,
                json.dumps(list(completion.collaboration_agents)),
                completion.collaboration_quality.name,
                completion.timestamp,
                json.dumps(list(completion.verified_by)),
                points_awarded
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save task completion: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_agents': len(self.agent_reputations),
            'total_tasks_completed': len(self.task_completions),
            'average_reputation': sum(r.total_points for r in self.agent_reputations.values()) / len(self.agent_reputations) if self.agent_reputations else 0,
            'active_agents_24h': len([r for r in self.agent_reputations.values() 
                                    if (datetime.now(timezone.utc) - datetime.fromisoformat(r.last_active)).total_seconds() < 86400]),
            'security_violations': sum(r.security_violations for r in self.agent_reputations.values())
        } 