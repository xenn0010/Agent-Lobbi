"""
Advanced Security Hardening Layer
Adds cryptographic verification, anti-tampering, and enterprise-grade threat detection
"""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

logger = logging.getLogger(__name__)


class CryptographicVerification:
    """Cryptographic verification for agent messages and state"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = secrets.token_bytes(32)
        
        self.master_key = master_key
        self.cipher_suite = self._create_cipher_suite()
        self.message_signatures: Dict[str, str] = {}
        
    def _create_cipher_suite(self) -> Fernet:
        """Create encryption cipher suite"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'agent_lobby_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def sign_message(self, message: Dict[str, Any], agent_id: str) -> str:
        """Create cryptographic signature for message"""
        message_string = json.dumps(message, sort_keys=True)
        signature = hmac.new(
            self.master_key,
            f"{agent_id}:{message_string}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        self.message_signatures[message.get('message_id', '')] = signature
        return signature
    
    def verify_message_integrity(self, message: Dict[str, Any], agent_id: str, signature: str) -> bool:
        """Verify message hasn't been tampered with"""
        expected_signature = self.sign_message(message, agent_id)
        return hmac.compare_digest(signature, expected_signature)
    
    def encrypt_sensitive_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data)
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data)


class AdvancedThreatDetection:
    """Advanced threat detection with machine learning-inspired patterns"""
    
    def __init__(self):
        self.agent_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds = {
            'message_frequency': 100,  # messages per minute
            'payload_size': 1024 * 1024,  # 1MB
            'repeated_patterns': 0.8,  # 80% similarity
            'target_diversity': 0.1,  # only targeting 10% of agents
            'timing_patterns': 0.9  # 90% of messages at same time intervals
        }
        
    async def analyze_agent_behavior(self, agent_id: str, recent_actions: List[Dict]) -> Dict[str, float]:
        """Analyze agent behavior for anomalies"""
        if len(recent_actions) < 5:
            return {'overall_threat_score': 0.0}
        
        threat_indicators = {}
        
        # 1. Message frequency analysis
        time_window = 60  # 1 minute
        current_time = time.time()
        recent_messages = [
            action for action in recent_actions 
            if current_time - action.get('timestamp', 0) < time_window
        ]
        
        freq_score = len(recent_messages) / self.anomaly_thresholds['message_frequency']
        threat_indicators['frequency_anomaly'] = min(1.0, freq_score)
        
        # 2. Payload size analysis
        avg_payload_size = sum(len(str(action.get('payload', ''))) for action in recent_actions) / len(recent_actions)
        size_score = avg_payload_size / self.anomaly_thresholds['payload_size']
        threat_indicators['payload_size_anomaly'] = min(1.0, size_score)
        
        # 3. Pattern repetition analysis
        action_types = [action.get('action_type', '') for action in recent_actions]
        unique_actions = set(action_types)
        repetition_score = 1.0 - (len(unique_actions) / len(action_types))
        threat_indicators['pattern_repetition'] = repetition_score
        
        # 4. Target diversity analysis
        targets = [action.get('target', '') for action in recent_actions if action.get('target')]
        unique_targets = set(targets)
        if targets:
            diversity_score = len(unique_targets) / len(targets)
            threat_indicators['target_diversity'] = 1.0 - diversity_score
        else:
            threat_indicators['target_diversity'] = 0.0
        
        # 5. Calculate overall threat score
        weights = {
            'frequency_anomaly': 0.3,
            'payload_size_anomaly': 0.2,
            'pattern_repetition': 0.25,
            'target_diversity': 0.25
        }
        
        overall_score = sum(
            threat_indicators[indicator] * weights[indicator]
            for indicator in weights
        )
        
        threat_indicators['overall_threat_score'] = min(1.0, overall_score)
        
        return threat_indicators


class SecurityConsensusEngine:
    """Enhanced consensus engine with cryptographic voting"""
    
    def __init__(self, crypto_verification: CryptographicVerification):
        self.crypto = crypto_verification
        self.pending_votes: Dict[str, Dict] = {}
        self.vote_history: List[Dict] = []
        
    async def initiate_security_vote(self, 
                                   subject_agent: str, 
                                   action: str, 
                                   evidence: Dict[str, Any], 
                                   proposer: str) -> str:
        """Initiate a cryptographically secured security vote"""
        vote_id = secrets.token_hex(16)
        
        vote_data = {
            'vote_id': vote_id,
            'subject_agent': subject_agent,
            'proposed_action': action,
            'evidence': evidence,
            'proposer': proposer,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'expires_at': (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
            'votes': {},
            'signatures': {}
        }
        
        # Sign the vote proposal
        signature = self.crypto.sign_message(vote_data, proposer)
        vote_data['proposal_signature'] = signature
        
        self.pending_votes[vote_id] = vote_data
        logger.info(f"Security vote initiated: {vote_id} for {subject_agent}")
        
        return vote_id
    
    async def cast_vote(self, vote_id: str, voter_id: str, vote: bool, justification: str) -> bool:
        """Cast a cryptographically signed vote"""
        if vote_id not in self.pending_votes:
            return False
        
        vote_data = self.pending_votes[vote_id]
        
        # Check if vote hasn't expired
        expires_at = datetime.fromisoformat(vote_data['expires_at'])
        if datetime.now(timezone.utc) > expires_at:
            return False
        
        # Create vote record
        vote_record = {
            'voter_id': voter_id,
            'vote': vote,
            'justification': justification,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Sign the vote
        signature = self.crypto.sign_message(vote_record, voter_id)
        
        vote_data['votes'][voter_id] = vote_record
        vote_data['signatures'][voter_id] = signature
        
        logger.info(f"Vote cast by {voter_id} on {vote_id}: {'APPROVE' if vote else 'REJECT'}")
        
        # Check if consensus reached
        await self._check_consensus(vote_id)
        
        return True
    
    async def _check_consensus(self, vote_id: str):
        """Check if consensus has been reached and execute if needed"""
        vote_data = self.pending_votes[vote_id]
        votes = vote_data['votes']
        
        if len(votes) < 3:  # Minimum voters
            return
        
        approve_votes = sum(1 for vote in votes.values() if vote['vote'])
        total_votes = len(votes)
        approval_ratio = approve_votes / total_votes
        
        if approval_ratio >= 0.67:  # 2/3 majority
            await self._execute_security_action(vote_data)
            del self.pending_votes[vote_id]
            
            # Archive in history
            vote_data['result'] = 'APPROVED'
            vote_data['executed_at'] = datetime.now(timezone.utc).isoformat()
            self.vote_history.append(vote_data)
    
    async def _execute_security_action(self, vote_data: Dict):
        """Execute the approved security action"""
        action = vote_data['proposed_action']
        subject = vote_data['subject_agent']
        
        logger.warning(f"Executing security action: {action} on {subject}")
        
        # This would integrate with the main security engine
        # For now, just log the action
        print(f" SECURITY ACTION EXECUTED: {action} on {subject}")


class IronCladSecurityManager:
    """Main security manager combining all security layers"""
    
    def __init__(self):
        self.crypto = CryptographicVerification()
        self.threat_detector = AdvancedThreatDetection()
        self.consensus_engine = SecurityConsensusEngine(self.crypto)
        
        # Security event log
        self.security_events: List[Dict] = []
        
    async def validate_agent_action(self, 
                                  agent_id: str, 
                                  action: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Comprehensive security validation"""
        
        # 1. Cryptographic verification
        if 'signature' in action:
            if not self.crypto.verify_message_integrity(action, agent_id, action['signature']):
                await self._log_security_event('INTEGRITY_VIOLATION', agent_id, action)
                return False, "Message integrity verification failed", 0.95
        
        # 2. Advanced threat detection
        recent_actions = context.get('recent_actions', [])
        threat_analysis = await self.threat_detector.analyze_agent_behavior(agent_id, recent_actions)
        
        if threat_analysis['overall_threat_score'] > 0.7:
            await self._log_security_event('HIGH_THREAT_DETECTED', agent_id, {
                'threat_score': threat_analysis['overall_threat_score'],
                'indicators': threat_analysis
            })
            
            # Initiate consensus vote for high-threat agents
            await self.consensus_engine.initiate_security_vote(
                subject_agent=agent_id,
                action='INVESTIGATE',
                evidence=threat_analysis,
                proposer='security_system'
            )
            
            return False, f"High threat score: {threat_analysis['overall_threat_score']:.2f}", threat_analysis['overall_threat_score']
        
        # 3. All checks passed
        return True, "Action approved", 1.0 - threat_analysis['overall_threat_score']
    
    async def _log_security_event(self, event_type: str, agent_id: str, details: Dict):
        """Log security events with encryption"""
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'agent_id': agent_id,
            'details': details,
            'event_id': secrets.token_hex(8)
        }
        
        # Encrypt sensitive details
        if 'sensitive' in details:
            encrypted_data = self.crypto.encrypt_sensitive_data(
                json.dumps(details['sensitive']).encode()
            )
            event['encrypted_details'] = base64.b64encode(encrypted_data).decode()
            del event['details']['sensitive']
        
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} for {agent_id}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        recent_events = [
            event for event in self.security_events
            if (datetime.now(timezone.utc) - datetime.fromisoformat(event['timestamp'])).total_seconds() < 3600
        ]
        
        return {
            'security_health': 'HEALTHY' if len(recent_events) < 10 else 'COMPROMISED',
            'recent_events': len(recent_events),
            'pending_votes': len(self.consensus_engine.pending_votes),
            'threat_detection_active': True,
            'cryptographic_verification_active': True,
            'consensus_engine_active': True
        } 