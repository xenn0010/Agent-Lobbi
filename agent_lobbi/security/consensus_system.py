"""
Consensus System for Agent Lobbi
===============================
Distributed consensus and reputation management for multi-agent collaboration.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
import time

class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class AgentReputation:
    agent_id: str
    reputation_score: float
    task_count: int
    success_rate: float
    last_activity: float = None

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = time.time()

@dataclass
class TaskCompletion:
    """Represents a completed task for consensus validation"""
    task_id: str
    agent_id: str
    success: bool
    completion_time: float
    difficulty: TaskDifficulty
    result: Any = None

    def __post_init__(self):
        if self.completion_time is None:
            self.completion_time = time.time()

class ConsensusReputationSystem:
    """
    Distributed consensus system for agent reputation and task validation.
    
    This is a placeholder implementation for PyPI packaging.
    In production, this would implement:
    - Byzantine fault tolerance
    - Proof-of-contribution consensus
    - Reputation-based voting
    - Anti-sybil mechanisms
    """
    
    def __init__(self, db_path: str = "consensus.db"):
        self.db_path = db_path
        self.agent_reputations: Dict[str, AgentReputation] = {}
        self.consensus_threshold = 0.67  # 67% agreement required
        
    async def register_agent(self, agent_id: str, initial_reputation: float = 50.0):
        """Register a new agent in the consensus system"""
        reputation = AgentReputation(
            agent_id=agent_id,
            reputation_score=initial_reputation,
            task_count=0,
            success_rate=0.0
        )
        self.agent_reputations[agent_id] = reputation
        return reputation
    
    def get_agent_reputation(self, agent_id: str) -> Optional[AgentReputation]:
        """Get reputation data for an agent"""
        return self.agent_reputations.get(agent_id)
    
    def update_reputation(self, agent_id: str, task_success: bool, task_difficulty: TaskDifficulty):
        """Update agent reputation based on task completion"""
        if agent_id not in self.agent_reputations:
            return False
        
        reputation = self.agent_reputations[agent_id]
        reputation.task_count += 1
        
        # Calculate new success rate
        if task_success:
            reputation.success_rate = ((reputation.success_rate * (reputation.task_count - 1)) + 1) / reputation.task_count
        else:
            reputation.success_rate = (reputation.success_rate * (reputation.task_count - 1)) / reputation.task_count
        
        # Adjust reputation score based on difficulty and success
        difficulty_multiplier = {
            TaskDifficulty.EASY: 1.0,
            TaskDifficulty.MEDIUM: 1.5,
            TaskDifficulty.HARD: 2.0
        }.get(task_difficulty, 1.0)
        
        if task_success:
            reputation.reputation_score += 5 * difficulty_multiplier
        else:
            reputation.reputation_score -= 2 * difficulty_multiplier
        
        # Keep reputation in bounds
        reputation.reputation_score = max(0, min(100, reputation.reputation_score))
        reputation.last_activity = time.time()
        
        return True
    
    def get_consensus_weight(self, agent_id: str) -> float:
        """Get voting weight for an agent based on reputation"""
        reputation = self.get_agent_reputation(agent_id)
        if not reputation:
            return 0.0
        
        # Weight based on reputation score and task experience
        base_weight = reputation.reputation_score / 100.0
        experience_bonus = min(0.5, reputation.task_count / 100.0)
        success_bonus = reputation.success_rate * 0.3
        
        return base_weight + experience_bonus + success_bonus
    
    def validate_agent_action(self, agent_id: str, action: str) -> bool:
        """Validate if an agent can perform a specific action"""
        reputation = self.get_agent_reputation(agent_id)
        if not reputation:
            return False
        
        # Basic validation based on reputation
        if reputation.reputation_score < 20:
            return False  # Too low reputation
        
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        if not self.agent_reputations:
            return {
                'total_agents': 0,
                'average_reputation': 0.0,
                'consensus_health': 'unknown'
            }
        
        total_agents = len(self.agent_reputations)
        avg_reputation = sum(r.reputation_score for r in self.agent_reputations.values()) / total_agents
        healthy_agents = sum(1 for r in self.agent_reputations.values() if r.reputation_score > 40)
        
        health = 'healthy' if healthy_agents / total_agents > 0.8 else 'moderate' if healthy_agents / total_agents > 0.5 else 'poor'
        
        return {
            'total_agents': total_agents,
            'average_reputation': round(avg_reputation, 2),
            'healthy_agents': healthy_agents,
            'consensus_health': health,
            'threshold': self.consensus_threshold
        }

# Global instance for easy access
consensus_system = ConsensusReputationSystem()

def get_consensus_system() -> ConsensusReputationSystem:
    """Get the global consensus system instance"""
    return consensus_system 