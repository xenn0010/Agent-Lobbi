"""
Nash Equilibrium-Based Security System for Agent Lobbi
Uses game theory to create an iron-clad security architecture that naturally discourages bad actors
"""

import asyncio
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import json
import logging
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in the ecosystem"""
    UNTRUSTED = 0     # New/unknown agents
    OBSERVED = 1      # Under observation
    BASIC = 2         # Basic trust established
    VERIFIED = 3      # Verified good behavior
    TRUSTED = 4       # High trust
    CORE = 5          # Core system agents


class SecurityEvent(Enum):
    """Types of security events"""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    MALICIOUS_BEHAVIOR = "malicious_behavior"
    REPUTATION_BOOST = "reputation_boost"
    REPUTATION_PENALTY = "reputation_penalty"
    CONSENSUS_VIOLATION = "consensus_violation"
    COLLABORATION_SUCCESS = "collaboration_success"
    COLLABORATION_FAILURE = "collaboration_failure"


@dataclass
class GameTheoryParameters:
    """Game theory parameters for the security system"""
    # Cooperation incentives
    cooperation_reward: float = 2.0
    collaboration_bonus: float = 1.5
    trust_building_reward: float = 1.0
    
    # Defection penalties
    malicious_penalty: float = -10.0
    unreliable_penalty: float = -3.0
    isolation_penalty: float = -5.0
    
    # Nash equilibrium parameters
    reputation_decay_rate: float = 0.95  # Daily decay
    trust_threshold: float = 0.6
    cooperation_threshold: float = 0.7
    
    # Security parameters
    consensus_requirement: float = 0.67  # 2/3 majority
    isolation_threshold: float = -20.0   # Auto-isolation score
    rehabilitation_threshold: float = 10.0  # Recovery threshold


@dataclass
class AgentReputation:
    """Comprehensive agent reputation tracking"""
    agent_id: str
    trust_score: float = 0.5  # Initial neutral trust
    cooperation_score: float = 0.5
    reliability_score: float = 0.5
    security_score: float = 0.5
    
    # Historical tracking
    successful_interactions: int = 0
    failed_interactions: int = 0
    malicious_attempts: int = 0
    consensus_violations: int = 0
    
    # Behavioral patterns
    interaction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    collaboration_partners: Set[str] = field(default_factory=set)
    
    # Trust relationships
    vouches_for: Set[str] = field(default_factory=set)
    vouched_by: Set[str] = field(default_factory=set)
    
    # Temporal tracking
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reputation_events: List[Dict] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall reputation score"""
        weights = {
            'trust': 0.3,
            'cooperation': 0.25,
            'reliability': 0.25,
            'security': 0.2
        }
        return (
            self.trust_score * weights['trust'] +
            self.cooperation_score * weights['cooperation'] +
            self.reliability_score * weights['reliability'] +
            self.security_score * weights['security']
        )
    
    @property
    def trust_level(self) -> TrustLevel:
        """Determine trust level based on scores"""
        score = self.overall_score
        if score >= 0.9:
            return TrustLevel.CORE
        elif score >= 0.8:
            return TrustLevel.TRUSTED
        elif score >= 0.6:
            return TrustLevel.VERIFIED
        elif score >= 0.4:
            return TrustLevel.BASIC
        elif score >= 0.2:
            return TrustLevel.OBSERVED
        else:
            return TrustLevel.UNTRUSTED


@dataclass
class SecurityConsensus:
    """Consensus mechanism for security decisions"""
    decision_id: str
    subject_agent_id: str
    decision_type: str  # "isolate", "rehabilitate", "trust_boost", etc.
    proposed_by: str
    
    # Voting
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    abstentions: Set[str] = field(default_factory=set)
    
    # Requirements
    required_consensus: float = 0.67
    minimum_voters: int = 3
    
    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def total_votes(self) -> int:
        return len(self.votes_for) + len(self.votes_against)
    
    @property
    def consensus_reached(self) -> bool:
        if self.total_votes < self.minimum_voters:
            return False
        
        for_ratio = len(self.votes_for) / self.total_votes
        return for_ratio >= self.required_consensus
    
    @property
    def decision_outcome(self) -> Optional[bool]:
        """True if approved, False if rejected, None if pending"""
        if not self.consensus_reached:
            return None
        return len(self.votes_for) > len(self.votes_against)


class NashEquilibriumSecurityEngine:
    """
    Revolutionary security engine based on Nash equilibrium game theory.
    
    Core Principles:
    1. Cooperation is the Nash equilibrium (most beneficial strategy)
    2. Defection is automatically punished by the collective
    3. Trust is earned through consistent good behavior
    4. Malicious actors are naturally isolated
    5. Recovery is possible through rehabilitation
    """
    
    def __init__(self, parameters: GameTheoryParameters = None):
        self.params = parameters or GameTheoryParameters()
        
        # Reputation system
        self.agent_reputations: Dict[str, AgentReputation] = {}
        
        # Consensus system
        self.pending_decisions: Dict[str, SecurityConsensus] = {}
        self.security_events: List[Dict] = []
        
        # Game theory matrices
        self.payoff_matrix = self._initialize_payoff_matrix()
        
        # Security monitoring
        self.threat_patterns: Dict[str, List] = defaultdict(list)
        self.isolation_list: Set[str] = set()
        
        # Collaboration tracking
        self.collaboration_graph = defaultdict(set)
        self.successful_collaborations: Dict[Tuple[str, str], int] = defaultdict(int)
        
        logger.info("Nash Equilibrium Security Engine initialized")
    
    def _initialize_payoff_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize the game theory payoff matrix"""
        return {
            # Both cooperate (Nash equilibrium)
            ('cooperate', 'cooperate'): (self.params.cooperation_reward, self.params.cooperation_reward),
            
            # One defects
            ('cooperate', 'defect'): (-1.0, self.params.malicious_penalty * 0.5),
            ('defect', 'cooperate'): (self.params.malicious_penalty * 0.5, -1.0),
            
            # Both defect (worst outcome)
            ('defect', 'defect'): (self.params.malicious_penalty, self.params.malicious_penalty)
        }
    
    async def register_agent(self, agent_id: str, initial_trust: float = 0.5) -> AgentReputation:
        """Register a new agent in the security system"""
        if agent_id in self.agent_reputations:
            return self.agent_reputations[agent_id]
        
        reputation = AgentReputation(
            agent_id=agent_id,
            trust_score=initial_trust
        )
        
        self.agent_reputations[agent_id] = reputation
        
        await self._log_security_event(
            SecurityEvent.AUTHENTICATION_SUCCESS,
            agent_id,
            {"action": "registration", "initial_trust": initial_trust}
        )
        
        logger.info(f"Registered agent {agent_id} with initial trust {initial_trust}")
        return reputation
    
    async def validate_interaction(self, initiator: str, target: str, action: str, context: Dict[str, Any] = None) -> Tuple[bool, str, float]:
        """
        Validate an interaction using game theory and reputation
        Returns: (allowed, reason, confidence)
        """
        context = context or {}
        
        # Check if agents are registered
        if initiator not in self.agent_reputations:
            await self.register_agent(initiator)
        if target not in self.agent_reputations:
            await self.register_agent(target)
        
        initiator_rep = self.agent_reputations[initiator]
        target_rep = self.agent_reputations[target]
        
        # Check isolation status
        if initiator in self.isolation_list:
            return False, f"Agent {initiator} is isolated due to malicious behavior", 0.95
        
        if target in self.isolation_list:
            return False, f"Target agent {target} is isolated", 0.90
        
        # Calculate interaction payoff
        payoff = self._calculate_interaction_payoff(initiator_rep, target_rep, action, context)
        
        # Security rules based on trust levels
        min_trust_required = self._get_minimum_trust_for_action(action)
        
        if initiator_rep.trust_score < min_trust_required:
            return False, f"Insufficient trust level for action '{action}'", 0.85
        
        # Advanced threat detection
        threat_score = await self._detect_threats(initiator, target, action, context)
        
        if threat_score > 0.7:
            return False, f"High threat probability detected: {threat_score:.2f}", threat_score
        
        # Check collaboration history
        collaboration_score = self._get_collaboration_score(initiator, target)
        
        # Final decision based on Nash equilibrium
        cooperation_benefit = payoff + collaboration_score
        confidence = min(0.99, max(0.6, cooperation_benefit))
        
        allowed = cooperation_benefit > 0 and threat_score < 0.5
        reason = "Approved based on game theory analysis" if allowed else "Blocked by security analysis"
        
        return allowed, reason, confidence
    
    def _calculate_interaction_payoff(self, initiator_rep: AgentReputation, target_rep: AgentReputation, 
                                    action: str, context: Dict[str, Any]) -> float:
        """Calculate the expected payoff of an interaction"""
        # Base payoff from cooperation
        base_payoff = self.params.cooperation_reward
        
        # Adjust based on reputation scores
        reputation_factor = (initiator_rep.overall_score + target_rep.overall_score) / 2
        trust_factor = (initiator_rep.trust_score + target_rep.trust_score) / 2
        
        # Action-specific adjustments
        action_multiplier = {
            'message': 1.0,
            'capability_request': 1.2,
            'workflow_collaboration': 1.5,
            'data_sharing': 1.3,
            'admin_action': 0.8
        }.get(action, 1.0)
        
        return base_payoff * reputation_factor * trust_factor * action_multiplier
    
    def _get_minimum_trust_for_action(self, action: str) -> float:
        """Get minimum trust required for different actions"""
        return {
            'message': 0.2,
            'capability_request': 0.4,
            'workflow_collaboration': 0.5,
            'data_sharing': 0.6,
            'admin_action': 0.8
        }.get(action, 0.3)
    
    async def _detect_threats(self, initiator: str, target: str, action: str, context: Dict[str, Any]) -> float:
        """Advanced threat detection using pattern analysis"""
        threat_score = 0.0
        
        # Rate limiting detection
        current_time = time.time()
        agent_patterns = self.threat_patterns[initiator]
        
        # Remove old patterns (sliding window)
        agent_patterns[:] = [p for p in agent_patterns if current_time - p['timestamp'] < 300]  # 5 min window
        
        # Check rate limiting
        if len(agent_patterns) > 100:  # More than 100 actions in 5 minutes
            threat_score += 0.5
        
        # Pattern analysis
        if len(agent_patterns) > 10:
            recent_actions = [p['action'] for p in agent_patterns[-10:]]
            
            # Repetitive behavior detection
            if len(set(recent_actions)) == 1:
                threat_score += 0.3
            
            # Targeting pattern detection
            recent_targets = [p.get('target') for p in agent_patterns[-10:]]
            if recent_targets.count(target) > 5:
                threat_score += 0.4
        
        # Add current pattern
        agent_patterns.append({
            'timestamp': current_time,
            'action': action,
            'target': target,
            'context': context
        })
        
        # Behavioral anomaly detection
        initiator_rep = self.agent_reputations.get(initiator)
        if initiator_rep:
            # Check for sudden behavior changes
            if len(initiator_rep.interaction_history) > 5:
                recent_success_rate = sum(1 for h in list(initiator_rep.interaction_history)[-5:] if h.get('success', False)) / 5
                if recent_success_rate < 0.2:  # Very low recent success rate
                    threat_score += 0.3
        
        return min(1.0, threat_score)
    
    def _get_collaboration_score(self, agent1: str, agent2: str) -> float:
        """Get collaboration score between two agents"""
        pair = tuple(sorted([agent1, agent2]))
        successful = self.successful_collaborations.get(pair, 0)
        
        # Bonus for successful past collaborations
        return min(0.5, successful * 0.1)
    
    async def record_interaction_outcome(self, initiator: str, target: str, action: str, 
                                       success: bool, context: Dict[str, Any] = None):
        """Record the outcome of an interaction for reputation updates"""
        context = context or {}
        
        initiator_rep = self.agent_reputations.get(initiator)
        target_rep = self.agent_reputations.get(target)
        
        if not initiator_rep or not target_rep:
            return
        
        # Update interaction counts
        if success:
            initiator_rep.successful_interactions += 1
            target_rep.successful_interactions += 1
            
            # Update collaboration tracking
            pair = tuple(sorted([initiator, target]))
            self.successful_collaborations[pair] += 1
            
            # Cooperation rewards (Nash equilibrium)
            await self._apply_reputation_change(initiator, 'cooperation_success', self.params.cooperation_reward * 0.1)
            await self._apply_reputation_change(target, 'cooperation_success', self.params.cooperation_reward * 0.1)
            
        else:
            initiator_rep.failed_interactions += 1
            target_rep.failed_interactions += 1
            
            # Analyze failure reason
            failure_reason = context.get('failure_reason', 'unknown')
            if failure_reason in ['malicious', 'attack', 'fraud']:
                await self._handle_malicious_behavior(initiator, target, action, context)
            else:
                # Minor reliability penalty
                await self._apply_reputation_change(initiator, 'reliability_issue', self.params.unreliable_penalty * 0.1)
        
        # Record in interaction history
        interaction_record = {
            'timestamp': datetime.now(timezone.utc),
            'target': target,
            'action': action,
            'success': success,
            'context': context
        }
        
        initiator_rep.interaction_history.append(interaction_record)
        
        await self._log_security_event(
            SecurityEvent.COLLABORATION_SUCCESS if success else SecurityEvent.COLLABORATION_FAILURE,
            initiator,
            {
                'target': target,
                'action': action,
                'success': success,
                'context': context
            }
        )
    
    async def _handle_malicious_behavior(self, perpetrator: str, victim: str, action: str, context: Dict[str, Any]):
        """Handle detected malicious behavior"""
        perpetrator_rep = self.agent_reputations.get(perpetrator)
        if not perpetrator_rep:
            return
        
        perpetrator_rep.malicious_attempts += 1
        
        # Immediate penalties
        penalty = self.params.malicious_penalty
        await self._apply_reputation_change(perpetrator, 'malicious_behavior', penalty)
        
        # Escalation based on severity
        if perpetrator_rep.malicious_attempts >= 3:
            # Initiate consensus for isolation
            await self._initiate_security_consensus(
                perpetrator,
                'isolate',
                victim,
                {'reason': 'repeated_malicious_behavior', 'context': context}
            )
        
        await self._log_security_event(
            SecurityEvent.MALICIOUS_BEHAVIOR,
            perpetrator,
            {
                'victim': victim,
                'action': action,
                'malicious_attempts': perpetrator_rep.malicious_attempts,
                'context': context
            }
        )
    
    async def _apply_reputation_change(self, agent_id: str, reason: str, change: float):
        """Apply reputation change with proper weighting"""
        reputation = self.agent_reputations.get(agent_id)
        if not reputation:
            return
        
        # Distribute change across different scores based on reason
        if 'cooperation' in reason:
            reputation.cooperation_score = max(0, min(1, reputation.cooperation_score + change * 0.6))
            reputation.trust_score = max(0, min(1, reputation.trust_score + change * 0.4))
        elif 'malicious' in reason:
            reputation.security_score = max(0, min(1, reputation.security_score + change * 0.8))
            reputation.trust_score = max(0, min(1, reputation.trust_score + change * 0.2))
        elif 'reliability' in reason:
            reputation.reliability_score = max(0, min(1, reputation.reliability_score + change))
        else:
            # General trust change
            reputation.trust_score = max(0, min(1, reputation.trust_score + change))
        
        # Record the event
        reputation.reputation_events.append({
            'timestamp': datetime.now(timezone.utc),
            'reason': reason,
            'change': change,
            'new_score': reputation.overall_score
        })
        
        # Check for automatic isolation
        if reputation.overall_score < self.params.isolation_threshold:
            await self._auto_isolate_agent(agent_id, reason)
    
    async def _initiate_security_consensus(self, subject_agent: str, decision_type: str, 
                                         proposed_by: str, context: Dict[str, Any]):
        """Initiate a security consensus decision"""
        decision_id = str(uuid.uuid4())
        
        consensus = SecurityConsensus(
            decision_id=decision_id,
            subject_agent_id=subject_agent,
            decision_type=decision_type,
            proposed_by=proposed_by
        )
        
        self.pending_decisions[decision_id] = consensus
        
        # Notify eligible voters (trusted agents)
        eligible_voters = [
            agent_id for agent_id, rep in self.agent_reputations.items()
            if rep.trust_level.value >= TrustLevel.VERIFIED.value and agent_id != subject_agent
        ]
        
        logger.info(f"Initiated security consensus {decision_id} for {subject_agent} ({decision_type})")
        
        # Auto-vote from system if clear malicious behavior
        if context.get('reason') == 'repeated_malicious_behavior':
            await self._cast_consensus_vote(decision_id, 'system', True)
        
        return decision_id
    
    async def _cast_consensus_vote(self, decision_id: str, voter_id: str, vote_for: bool):
        """Cast a vote in a security consensus"""
        consensus = self.pending_decisions.get(decision_id)
        if not consensus or consensus.is_expired:
            return False
        
        # Remove from other vote sets
        consensus.votes_for.discard(voter_id)
        consensus.votes_against.discard(voter_id)
        consensus.abstentions.discard(voter_id)
        
        # Add to appropriate set
        if vote_for:
            consensus.votes_for.add(voter_id)
        else:
            consensus.votes_against.add(voter_id)
        
        # Check if consensus reached
        if consensus.consensus_reached:
            await self._execute_consensus_decision(decision_id)
        
        return True
    
    async def _execute_consensus_decision(self, decision_id: str):
        """Execute a consensus decision"""
        consensus = self.pending_decisions.get(decision_id)
        if not consensus:
            return
        
        decision = consensus.decision_outcome
        if decision is None:
            return
        
        if decision:  # Approved
            if consensus.decision_type == 'isolate':
                await self._isolate_agent(consensus.subject_agent_id, 'consensus_decision')
            elif consensus.decision_type == 'rehabilitate':
                await self._rehabilitate_agent(consensus.subject_agent_id)
            elif consensus.decision_type == 'trust_boost':
                await self._apply_reputation_change(consensus.subject_agent_id, 'consensus_trust_boost', 0.2)
        
        # Clean up
        del self.pending_decisions[decision_id]
        
        logger.info(f"Executed consensus decision {decision_id}: {consensus.decision_type} for {consensus.subject_agent_id} - {'APPROVED' if decision else 'REJECTED'}")
    
    async def _auto_isolate_agent(self, agent_id: str, reason: str):
        """Automatically isolate an agent"""
        self.isolation_list.add(agent_id)
        
        await self._log_security_event(
            SecurityEvent.CONSENSUS_VIOLATION,
            agent_id,
            {'action': 'auto_isolation', 'reason': reason}
        )
        
        logger.warning(f"Auto-isolated agent {agent_id} due to {reason}")
    
    async def _isolate_agent(self, agent_id: str, reason: str):
        """Isolate an agent by consensus"""
        self.isolation_list.add(agent_id)
        
        # Apply isolation penalty
        await self._apply_reputation_change(agent_id, 'isolation', self.params.isolation_penalty)
        
        await self._log_security_event(
            SecurityEvent.CONSENSUS_VIOLATION,
            agent_id,
            {'action': 'consensus_isolation', 'reason': reason}
        )
        
        logger.warning(f"Isolated agent {agent_id} by consensus: {reason}")
    
    async def _rehabilitate_agent(self, agent_id: str):
        """Rehabilitate an isolated agent"""
        self.isolation_list.discard(agent_id)
        
        # Apply rehabilitation bonus
        await self._apply_reputation_change(agent_id, 'rehabilitation', self.params.rehabilitation_threshold * 0.1)
        
        await self._log_security_event(
            SecurityEvent.REPUTATION_BOOST,
            agent_id,
            {'action': 'rehabilitation'}
        )
        
        logger.info(f"Rehabilitated agent {agent_id}")
    
    async def _log_security_event(self, event_type: SecurityEvent, agent_id: str, details: Dict[str, Any]):
        """Log a security event"""
        event = {
            'timestamp': datetime.now(timezone.utc),
            'event_type': event_type.value,
            'agent_id': agent_id,
            'details': details
        }
        
        self.security_events.append(event)
        
        # Keep only recent events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
    
    async def periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Reputation decay
        current_time = datetime.now(timezone.utc)
        
        for reputation in self.agent_reputations.values():
            # Calculate time since last activity
            time_diff = (current_time - reputation.last_activity).total_seconds()
            days_inactive = time_diff / (24 * 3600)
            
            if days_inactive > 1:
                # Apply decay
                decay_factor = self.params.reputation_decay_rate ** days_inactive
                reputation.trust_score *= decay_factor
                reputation.cooperation_score *= decay_factor
                reputation.reliability_score *= decay_factor
        
        # Clean up expired consensus decisions
        expired_decisions = [
            decision_id for decision_id, consensus in self.pending_decisions.items()
            if consensus.is_expired
        ]
        
        for decision_id in expired_decisions:
            del self.pending_decisions[decision_id]
        
        logger.info(f"Periodic maintenance: Applied decay to {len(self.agent_reputations)} agents, cleaned up {len(expired_decisions)} expired decisions")
    
    def get_agent_security_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive security status for an agent"""
        reputation = self.agent_reputations.get(agent_id)
        if not reputation:
            return {"error": "Agent not found"}
        
        return {
            "agent_id": agent_id,
            "trust_level": reputation.trust_level.name,
            "overall_score": reputation.overall_score,
            "scores": {
                "trust": reputation.trust_score,
                "cooperation": reputation.cooperation_score,
                "reliability": reputation.reliability_score,
                "security": reputation.security_score
            },
            "statistics": {
                "successful_interactions": reputation.successful_interactions,
                "failed_interactions": reputation.failed_interactions,
                "malicious_attempts": reputation.malicious_attempts,
                "consensus_violations": reputation.consensus_violations
            },
            "status": {
                "isolated": agent_id in self.isolation_list,
                "active_collaborations": len(reputation.collaboration_partners),
                "vouches_received": len(reputation.vouched_by),
                "vouches_given": len(reputation.vouches_for)
            },
            "last_activity": reputation.last_activity.isoformat()
        }
    
    def get_system_security_overview(self) -> Dict[str, Any]:
        """Get overall system security overview"""
        total_agents = len(self.agent_reputations)
        isolated_agents = len(self.isolation_list)
        
        trust_distribution = defaultdict(int)
        for rep in self.agent_reputations.values():
            trust_distribution[rep.trust_level.name] += 1
        
        return {
            "total_agents": total_agents,
            "isolated_agents": isolated_agents,
            "pending_decisions": len(self.pending_decisions),
            "recent_events": len([e for e in self.security_events if (datetime.now(timezone.utc) - e['timestamp']).total_seconds() < 3600]),
            "trust_distribution": dict(trust_distribution),
            "collaboration_pairs": len(self.successful_collaborations),
            "threat_patterns_monitored": len(self.threat_patterns),
            "security_health": "HEALTHY" if isolated_agents / max(1, total_agents) < 0.05 else "COMPROMISED"
        } 