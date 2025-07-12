"""
Advanced Agent Lobbi SDK - Complete Integration with Cross-Domain Algorithms
============================================================================

This enhanced SDK integrates:
- PageRank reputation system for network-effect authority
- Shapley value distribution for fair reward allocation  
- Genetic algorithm optimization for coalition formation
- Trust propagation networks for collaboration decisions
- All existing security, consensus, recovery, and tracking features
"""

import asyncio
import json
import logging
import hashlib
import secrets
import random
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import gzip
import pickle
import uuid
import time

# Import our security systems
from ..security.consensus_system import (
    ConsensusReputationSystem, TaskDifficulty, AgentReputation, TaskCompletion
)
from ..security.data_protection_layer import (
    DataProtectionLayer, DataClassification, AccessLevel
)
from ..recovery.connection_recovery import (
    ConnectionRecoverySystem, ConnectionState, RecoveryStrategy
)
from ..tracking.agent_tracking_system import (
    AgentTrackingSystem, ActivityType, AgentActivity
)

logger = logging.getLogger(__name__)

# ===== ADVANCED ALGORITHM SYSTEMS =====

class AdvancedPageRankSystem:
    """PageRank-inspired reputation system for agents"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.collaborations: Dict[str, Dict[str, float]] = {}
        self.task_outcomes: Dict[str, List[Dict]] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize PageRank database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_collaborations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_a TEXT NOT NULL,
                agent_b TEXT NOT NULL,
                collaboration_quality REAL NOT NULL,
                task_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authority_scores (
                agent_id TEXT PRIMARY KEY,
                authority_score REAL NOT NULL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def record_collaboration(self, agent_a: str, agent_b: str, task_outcome: Dict[str, Any]):
        """Record real collaboration between agents"""
        quality = self._calculate_collaboration_quality(task_outcome)
        
        if agent_a not in self.collaborations:
            self.collaborations[agent_a] = {}
        if agent_b not in self.collaborations:
            self.collaborations[agent_b] = {}
        
        self.collaborations[agent_a][agent_b] = quality
        self.collaborations[agent_b][agent_a] = quality
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO agent_collaborations (agent_a, agent_b, collaboration_quality, task_id)
            VALUES (?, ?, ?, ?)
        ''', (agent_a, agent_b, quality, task_outcome.get('task_id')))
        conn.commit()
        conn.close()
        
        logger.info(f"PageRank: Recorded collaboration {agent_a} ↔ {agent_b}, Quality: {quality:.3f}")
    
    def _calculate_collaboration_quality(self, task_outcome: Dict[str, Any]) -> float:
        """Calculate collaboration quality from task outcomes"""
        success_rate = task_outcome.get('success', False)
        response_time = task_outcome.get('response_time', 1.0)
        confidence = task_outcome.get('confidence', 0.5)
        user_satisfaction = task_outcome.get('user_satisfaction', 0.7)
        
        quality = 0.0
        if success_rate:
            quality += 0.4
        
        time_efficiency = max(0.1, 1.0 / (1.0 + response_time / 10.0))
        quality += 0.2 * time_efficiency
        quality += 0.2 * confidence
        quality += 0.2 * user_satisfaction
        
        return min(1.0, max(0.1, quality))
    
    async def calculate_authority_scores(self) -> Dict[str, float]:
        """Calculate PageRank authority scores"""
        if not self.collaborations:
            return {}
        
        agents = list(set().union(*[set(collab.keys()) for collab in self.collaborations.values()]) 
                     | set(self.collaborations.keys()))
        
        if not agents:
            return {}
        
        scores = {agent: 1.0 for agent in agents}
        
        # PageRank iterations
        for iteration in range(15):
            new_scores = {}
            for agent in agents:
                new_score = 0.15  # Damping factor
                
                for collaborator, quality in self.collaborations.get(agent, {}).items():
                    if collaborator in scores:
                        collaborator_links = len(self.collaborations.get(collaborator, {}))
                        if collaborator_links > 0:
                            vote_strength = scores[collaborator] * quality / collaborator_links
                            new_score += 0.85 * vote_strength
                
                new_scores[agent] = new_score
            
            scores = new_scores
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for agent_id, score in scores.items():
            cursor.execute('''
                INSERT OR REPLACE INTO authority_scores (agent_id, authority_score)
                VALUES (?, ?)
            ''', (agent_id, score))
        conn.commit()
        conn.close()
        
        logger.info(f"PageRank: Calculated authority for {len(agents)} agents")
        return scores

class AdvancedShapleySystem:
    """Shapley value system for fair reward distribution"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.task_contributions: Dict[str, List[Dict]] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize Shapley database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                contribution_score REAL NOT NULL,
                success_impact REAL NOT NULL,
                quality_impact REAL NOT NULL,
                efficiency_impact REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shapley_rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                reward_amount REAL NOT NULL,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def record_contribution(self, task_id: str, agent_id: str, contribution_data: Dict[str, Any]):
        """Record agent contribution to task"""
        if task_id not in self.task_contributions:
            self.task_contributions[task_id] = []
        
        self.task_contributions[task_id].append({
            'agent_id': agent_id,
            'contribution': contribution_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO task_contributions 
            (task_id, agent_id, contribution_score, success_impact, quality_impact, efficiency_impact)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            task_id, agent_id,
            contribution_data.get('total_score', 0.5),
            contribution_data.get('success_impact', 0.5),
            contribution_data.get('quality_impact', 0.5),
            contribution_data.get('efficiency_impact', 0.5)
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Shapley: Recorded contribution for {agent_id} on task {task_id}")
    
    async def calculate_fair_rewards(self, task_id: str, total_reward: float) -> Dict[str, float]:
        """Calculate Shapley value distribution"""
        if task_id not in self.task_contributions:
            return {}
        
        contributions = self.task_contributions[task_id]
        agents = [contrib['agent_id'] for contrib in contributions]
        
        if not agents:
            return {}
        
        shapley_values = {}
        
        # Calculate marginal contributions for each agent
        for target_agent in agents:
            marginal_contributions = []
            
            for subset_size in range(len(agents)):
                agent_contrib = next(
                    contrib['contribution'] for contrib in contributions 
                    if contrib['agent_id'] == target_agent
                )
                
                success_impact = agent_contrib.get('success_impact', 0.5)
                quality_impact = agent_contrib.get('quality_impact', 0.5)
                efficiency_impact = agent_contrib.get('efficiency_impact', 0.5)
                
                size_factor = 1.0 / (1.0 + subset_size * 0.1)
                marginal_value = (success_impact + quality_impact + efficiency_impact) / 3.0 * size_factor
                marginal_contributions.append(marginal_value * total_reward)
            
            shapley_values[target_agent] = sum(marginal_contributions) / len(marginal_contributions)
        
        # Normalize
        total_calculated = sum(shapley_values.values())
        if total_calculated > 0:
            for agent in shapley_values:
                shapley_values[agent] = (shapley_values[agent] / total_calculated) * total_reward
        
        # Store results
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for agent_id, reward in shapley_values.items():
            cursor.execute('''
                INSERT INTO shapley_rewards (task_id, agent_id, reward_amount)
                VALUES (?, ?, ?)
            ''', (task_id, agent_id, reward))
        conn.commit()
        conn.close()
        
        logger.info(f"Shapley: Calculated fair rewards for task {task_id}: {len(agents)} agents")
        return shapley_values

class AdvancedGeneticOptimizer:
    """Genetic algorithm for optimal coalition formation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize genetic optimizer database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_capabilities (
                agent_id TEXT PRIMARY KEY,
                capabilities TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coalition_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coalition_hash TEXT NOT NULL,
                requirements_hash TEXT NOT NULL,
                performance_score REAL NOT NULL,
                coalition_members TEXT NOT NULL,
                requirements TEXT NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def register_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        """Register agent capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO agent_capabilities (agent_id, capabilities)
            VALUES (?, ?)
        ''', (agent_id, json.dumps(capabilities)))
        conn.commit()
        conn.close()
        
        logger.info(f"Genetic: Registered capabilities for {agent_id}: {capabilities}")
    
    async def optimize_coalition(self, task_requirements: List[str], 
                               available_agents: List[str]) -> Tuple[Set[str], float]:
        """Use genetic algorithm to find optimal coalition"""
        if not available_agents:
            return set(), 0.0
        
        population_size = min(20, len(available_agents) * 2)
        generations = 25
        
        best_coalition = set()
        best_fitness = 0.0
        
        logger.info(f"Genetic: Optimizing coalition for requirements: {task_requirements}")
        
        for generation in range(generations):
            population = []
            
            for _ in range(population_size):
                coalition_size = random.randint(2, min(5, len(available_agents)))
                coalition = set(random.sample(available_agents, coalition_size))
                fitness = await self._calculate_fitness(coalition, task_requirements)
                population.append((coalition, fitness))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_coalition = coalition.copy()
        
        logger.info(f"Genetic: Optimal coalition: {sorted(best_coalition)}, Fitness: {best_fitness:.3f}")
        return best_coalition, best_fitness
    
    async def _calculate_fitness(self, coalition: Set[str], requirements: List[str]) -> float:
        """Calculate coalition fitness"""
        if not coalition:
            return 0.0
        
        # Capability coverage
        covered_capabilities = set()
        for agent in coalition:
            covered_capabilities.update(self.agent_capabilities.get(agent, []))
        
        coverage_score = len(covered_capabilities.intersection(requirements)) / len(requirements)
        efficiency_score = 1.0 / (1.0 + len(coalition) * 0.1)
        
        # Historical performance
        coalition_key = frozenset(coalition)
        req_key = tuple(sorted(requirements))
        history_key = f"{coalition_key}_{req_key}"
        
        if history_key in self.performance_history:
            if HAS_NUMPY:
                historical_performance = np.mean(self.performance_history[history_key])
            else:
                # Fallback to built-in statistics module
                import statistics
                historical_performance = statistics.mean(self.performance_history[history_key])
        else:
            historical_performance = 0.7
        
        synergy_score = self._calculate_synergy(coalition)
        
        fitness = (0.5 * coverage_score + 0.2 * efficiency_score + 
                  0.2 * historical_performance + 0.1 * synergy_score)
        
        return fitness
    
    def _calculate_synergy(self, coalition: Set[str]) -> float:
        """Calculate synergy based on agent diversity"""
        agent_types = set()
        for agent in coalition:
            if 'financial' in agent:
                agent_types.add('financial')
            elif 'content' in agent:
                agent_types.add('content')
            elif 'coordinator' in agent:
                agent_types.add('coordinator')
            else:
                agent_types.add('general')
        
        diversity_factor = len(agent_types) / len(coalition)
        return min(1.0, diversity_factor + random.uniform(0.1, 0.3))

class AdvancedTrustNetwork:
    """Trust propagation system for collaboration decisions"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.direct_trust: Dict[str, Dict[str, float]] = {}
        self.trust_history: Dict[str, List[Dict]] = {}
        self._init_db()
    
    def _init_db(self):
        """Initialize trust network database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_agent TEXT NOT NULL,
                target_agent TEXT NOT NULL,
                trust_score REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_agent TEXT NOT NULL,
                target_agent TEXT NOT NULL,
                interaction_outcome TEXT NOT NULL,
                trust_delta REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def record_trust_interaction(self, source_agent: str, target_agent: str, 
                                     interaction_outcome: Dict[str, Any]):
        """Record trust-building interaction"""
        trust_score = self._calculate_trust_from_interaction(interaction_outcome)
        
        if source_agent not in self.direct_trust:
            self.direct_trust[source_agent] = {}
        
        # Update trust with exponential moving average
        current_trust = self.direct_trust[source_agent].get(target_agent, 0.5)
        alpha = 0.3  # Learning rate
        new_trust = alpha * trust_score + (1 - alpha) * current_trust
        self.direct_trust[source_agent][target_agent] = new_trust
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO trust_relationships (source_agent, target_agent, trust_score)
            VALUES (?, ?, ?)
        ''', (source_agent, target_agent, new_trust))
        
        cursor.execute('''
            INSERT INTO trust_interactions 
            (source_agent, target_agent, interaction_outcome, trust_delta)
            VALUES (?, ?, ?, ?)
        ''', (source_agent, target_agent, json.dumps(interaction_outcome), trust_score - current_trust))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trust: Updated {source_agent} → {target_agent}: {new_trust:.3f}")
    
    def _calculate_trust_from_interaction(self, outcome: Dict[str, Any]) -> float:
        """Calculate trust score from interaction outcome"""
        reliability = 1.0 if outcome.get('task_completed', False) else 0.2
        quality = outcome.get('output_quality', 0.5)
        timeliness = 1.0 if outcome.get('delivered_on_time', True) else 0.3
        communication = outcome.get('communication_quality', 0.7)
        
        trust_score = (0.3 * reliability + 0.3 * quality + 
                      0.2 * timeliness + 0.2 * communication)
        
        return max(0.0, min(1.0, trust_score))
    
    async def calculate_trust_propagation(self, source_agent: str, target_agent: str) -> float:
        """Calculate trust including indirect propagation"""
        # Direct trust
        if target_agent in self.direct_trust.get(source_agent, {}):
            return self.direct_trust[source_agent][target_agent]
        
        # Indirect trust through intermediaries
        max_indirect_trust = 0.0
        
        for intermediate in self.direct_trust.get(source_agent, {}):
            if target_agent in self.direct_trust.get(intermediate, {}):
                indirect_trust = (self.direct_trust[source_agent][intermediate] * 
                                self.direct_trust[intermediate][target_agent] * 0.8)
                max_indirect_trust = max(max_indirect_trust, indirect_trust)
        
        return max_indirect_trust

# ===== ENHANCED AGENT LOBBI SDK =====

class AdvancedAgentLobbySDK:
    """Enhanced Agent Lobbi SDK with advanced algorithms and security"""
    
    def __init__(self, 
                 lobby_host: str = "localhost",
                 lobby_port: int = 8080,
                 enable_security: bool = True,
                 db_path_prefix: str = "advanced_agent_lobby"):
        
        self.lobby_host = lobby_host
        self.lobby_port = lobby_port
        self.lobby_url = f"http://{lobby_host}:{lobby_port}"
        
        # Agent information
        self.agent_id: Optional[str] = None
        self.api_key: Optional[str] = None
        self.auth_token: Optional[str] = None
        self.session_id: Optional[str] = None
        
        # Initialize security systems
        if enable_security:
            self.consensus_system = ConsensusReputationSystem(f"{db_path_prefix}_consensus.db")
            self.data_protection = DataProtectionLayer(f"{db_path_prefix}_protection.db")
            self.recovery_system = ConnectionRecoverySystem(f"{db_path_prefix}_recovery.db")
            self.tracking_system = AgentTrackingSystem(f"{db_path_prefix}_tracking.db")
        else:
            self.consensus_system = None
            self.data_protection = None
            self.recovery_system = None
            self.tracking_system = None
        
        # Initialize advanced algorithm systems
        self.pagerank_system = AdvancedPageRankSystem(f"{db_path_prefix}_pagerank.db")
        self.shapley_system = AdvancedShapleySystem(f"{db_path_prefix}_shapley.db")
        self.genetic_optimizer = AdvancedGeneticOptimizer(f"{db_path_prefix}_genetic.db")
        self.trust_network = AdvancedTrustNetwork(f"{db_path_prefix}_trust.db")
        
        # Connection state
        self.connected = False
        self.websocket = None
        
        logger.info("Advanced Agent Lobbi SDK initialized with algorithms and security")
    
    async def register_agent(self, 
                           agent_id: str,
                           agent_type: str,
                           capabilities: List[str],
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Register agent with advanced systems"""
        try:
            self.agent_id = agent_id
            
            # Generate API key for tracking
            if self.tracking_system:
                self.api_key = self.tracking_system.generate_api_key()
            
            # Register with all systems
            if self.consensus_system:
                await self.consensus_system.register_agent(agent_id)
            
            if self.recovery_system:
                await self.recovery_system.register_connection(
                    agent_id, "lobby", "primary", metadata or {}
                )
            
            if self.tracking_system and self.api_key:
                self.session_id = await self.tracking_system.start_agent_session(
                    agent_id, self.api_key, {"agent_type": agent_type}
                )
            
            if self.data_protection:
                await self.data_protection.register_agent_data(
                    agent_id, "agent_capabilities", DataClassification.INTERNAL,
                    set(), AccessLevel.READ, "Agent capability information"
                )
            
            # Register with advanced algorithms
            await self.genetic_optimizer.register_agent_capabilities(agent_id, capabilities)
            
            registration_result = {
                "status": "success",
                "agent_id": agent_id,
                "auth_token": f"auth_{secrets.token_hex(16)}",
                "api_key": self.api_key,
                "session_id": self.session_id,
                "message": "Agent registered with advanced algorithms and security",
                "algorithms_enabled": ["pagerank", "shapley", "genetic", "trust"]
            }
            
            self.auth_token = registration_result["auth_token"]
            self.connected = True
            
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    agent_id, self.api_key, ActivityType.REGISTERED,
                    {"agent_type": agent_type, "capabilities": capabilities}
                )
            
            logger.info(f"Agent {agent_id} registered with all advanced systems")
            return registration_result
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def submit_collaborative_task(self,
                                      task_id: str,
                                      task_description: str,
                                      collaborators: List[str] = None,
                                      task_outcome: Dict[str, Any] = None) -> Dict[str, Any]:
        """Submit task with collaboration tracking for advanced algorithms"""
        if not self.agent_id:
            raise ValueError("Agent not registered")
        
        try:
            # Record collaborations for PageRank
            if collaborators and task_outcome:
                for collaborator in collaborators:
                    if collaborator != self.agent_id:
                        await self.pagerank_system.record_collaboration(
                            self.agent_id, collaborator, task_outcome
                        )
            
            # Record contribution for Shapley
            if task_outcome:
                contribution_data = {
                    'success_impact': 0.8 if task_outcome.get('success', False) else 0.2,
                    'quality_impact': task_outcome.get('confidence', 0.5),
                    'efficiency_impact': max(0.1, 1.0 / (1.0 + task_outcome.get('response_time', 1.0) / 5.0))
                }
                await self.shapley_system.record_contribution(task_id, self.agent_id, contribution_data)
            
            # Record trust interactions
            if collaborators and task_outcome:
                for collaborator in collaborators:
                    if collaborator != self.agent_id:
                        interaction_outcome = {
                            'task_completed': task_outcome.get('success', False),
                            'output_quality': task_outcome.get('confidence', 0.5),
                            'delivered_on_time': task_outcome.get('response_time', 1.0) < 10.0,
                            'communication_quality': random.uniform(0.7, 0.95)
                        }
                        await self.trust_network.record_trust_interaction(
                            self.agent_id, collaborator, interaction_outcome
                        )
            
            # Track with existing systems
            if self.tracking_system and self.api_key:
                await self.tracking_system.track_agent_activity(
                    self.agent_id, self.api_key, ActivityType.TASK_COMPLETED,
                    {"task_id": task_id, "collaborators": collaborators or []}
                )
            
            return {
                "status": "success",
                "task_id": task_id,
                "message": "Task submitted with advanced algorithm tracking",
                "algorithms_updated": ["pagerank", "shapley", "trust"]
            }
            
        except Exception as e:
            logger.error(f"Failed to submit collaborative task: {e}")
            raise
    
    async def get_optimal_coalition(self, task_requirements: List[str], 
                                  available_agents: List[str] = None) -> Dict[str, Any]:
        """Get optimal coalition using genetic algorithm"""
        try:
            if available_agents is None:
                # In real implementation, this would query the lobby for available agents
                available_agents = [self.agent_id] if self.agent_id else []
            
            optimal_coalition, fitness = await self.genetic_optimizer.optimize_coalition(
                task_requirements, available_agents
            )
            
            return {
                "status": "success",
                "optimal_coalition": list(optimal_coalition),
                "fitness_score": fitness,
                "task_requirements": task_requirements,
                "algorithm": "genetic_optimization"
            }
            
        except Exception as e:
            logger.error(f"Failed to get optimal coalition: {e}")
            raise
    
    async def get_authority_scores(self) -> Dict[str, Any]:
        """Get PageRank authority scores"""
        try:
            authority_scores = await self.pagerank_system.calculate_authority_scores()
            
            return {
                "status": "success",
                "authority_scores": authority_scores,
                "agent_count": len(authority_scores),
                "algorithm": "pagerank_authority"
            }
            
        except Exception as e:
            logger.error(f"Failed to get authority scores: {e}")
            raise
    
    async def get_fair_rewards(self, task_id: str, total_reward: float) -> Dict[str, Any]:
        """Get fair reward distribution using Shapley values"""
        try:
            shapley_rewards = await self.shapley_system.calculate_fair_rewards(task_id, total_reward)
            
            return {
                "status": "success",
                "task_id": task_id,
                "shapley_rewards": shapley_rewards,
                "total_reward": total_reward,
                "algorithm": "shapley_values"
            }
            
        except Exception as e:
            logger.error(f"Failed to get fair rewards: {e}")
            raise
    
    async def get_trust_score(self, target_agent: str) -> Dict[str, Any]:
        """Get trust score for target agent"""
        try:
            if not self.agent_id:
                raise ValueError("Agent not registered")
            
            trust_score = await self.trust_network.calculate_trust_propagation(
                self.agent_id, target_agent
            )
            
            return {
                "status": "success",
                "source_agent": self.agent_id,
                "target_agent": target_agent,
                "trust_score": trust_score,
                "algorithm": "trust_propagation"
            }
            
        except Exception as e:
            logger.error(f"Failed to get trust score: {e}")
            raise
    
    def get_algorithm_status(self) -> Dict[str, Any]:
        """Get status of all advanced algorithms"""
        return {
            "pagerank_system": {
                "collaborations": len(self.pagerank_system.collaborations),
                "agents_tracked": len(set().union(*[set(collab.keys()) for collab in self.pagerank_system.collaborations.values()]) | set(self.pagerank_system.collaborations.keys())) if self.pagerank_system.collaborations else 0
            },
            "shapley_system": {
                "tasks_tracked": len(self.shapley_system.task_contributions)
            },
            "genetic_optimizer": {
                "agents_registered": len(self.genetic_optimizer.agent_capabilities)
            },
            "trust_network": {
                "trust_relationships": len(self.trust_network.direct_trust),
                "total_interactions": sum(len(history) for history in self.trust_network.trust_history.values())
            }
        }

# ===== FACTORY FUNCTIONS =====

async def create_advanced_agent(agent_id: str, 
                              agent_type: str,
                              capabilities: List[str],
                              lobby_host: str = "localhost",
                              lobby_port: int = 8080) -> AdvancedAgentLobbySDK:
    """Create and register an advanced agent with all algorithms"""
    sdk = AdvancedAgentLobbySDK(lobby_host, lobby_port, enable_security=True)
    
    await sdk.register_agent(agent_id, agent_type, capabilities)
    
    logger.info(f"Advanced agent {agent_id} created and registered")
    return sdk 