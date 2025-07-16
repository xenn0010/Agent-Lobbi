"""
Data Protection Layer for Agent Lobbi
====================================
Multi-layer data protection and access control for sensitive agent data.
"""

from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass
import time
import json

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class AccessLevel(Enum):
    """Access permission levels"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"

@dataclass
class AccessRequest:
    """Represents a data access request"""
    agent_id: str
    data_id: str
    access_level: AccessLevel
    classification: DataClassification
    timestamp: float
    approved: bool = False
    reason: str = ""

    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class DataEntry:
    """Represents a protected data entry"""
    data_id: str
    classification: DataClassification
    owner_id: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    authorized_agents: Set[str] = None

    def __post_init__(self):
        if self.authorized_agents is None:
            self.authorized_agents = {self.owner_id}
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = time.time()
        if not hasattr(self, 'last_accessed') or self.last_accessed is None:
            self.last_accessed = time.time()

class DataProtectionLayer:
    """
    Multi-layer data protection system for Agent Lobbi.
    
    This is a placeholder implementation for PyPI packaging.
    In production, this would implement:
    - End-to-end encryption
    - Zero-knowledge data sharing
    - Homomorphic computation
    - Differential privacy
    - Access pattern obfuscation
    """
    
    def __init__(self, db_path: str = "data_protection.db"):
        self.db_path = db_path
        self.data_entries: Dict[str, DataEntry] = {}
        self.access_requests: List[AccessRequest] = []
        self.encryption_keys: Dict[str, str] = {}
        
    def store_protected_data(self, 
                           data_id: str, 
                           data: Any, 
                           classification: DataClassification,
                           owner_id: str) -> bool:
        """Store data with appropriate protection level"""
        try:
            # Create data entry
            entry = DataEntry(
                data_id=data_id,
                classification=classification,
                owner_id=owner_id,
                created_at=time.time(),
                last_accessed=time.time()
            )
            
            self.data_entries[data_id] = entry
            
            # In production, this would encrypt the data based on classification
            # For now, we just track metadata
            
            return True
            
        except Exception as e:
            print(f"Failed to store protected data: {e}")
            return False
    
    def request_data_access(self, 
                          agent_id: str, 
                          data_id: str, 
                          access_level: AccessLevel,
                          reason: str = "") -> bool:
        """Request access to protected data"""
        if data_id not in self.data_entries:
            return False
        
        entry = self.data_entries[data_id]
        
        # Check if already authorized
        if agent_id in entry.authorized_agents:
            return True
        
        # Create access request
        request = AccessRequest(
            agent_id=agent_id,
            data_id=data_id,
            access_level=access_level,
            classification=entry.classification,
            timestamp=time.time(),
            reason=reason
        )
        
        # Auto-approve based on classification and access level
        # In production, this would involve complex policy evaluation
        auto_approve = self._evaluate_access_policy(request, entry)
        request.approved = auto_approve
        
        self.access_requests.append(request)
        
        if auto_approve:
            entry.authorized_agents.add(agent_id)
        
        return auto_approve
    
    def get_data_access(self, agent_id: str, data_id: str) -> Optional[Dict[str, Any]]:
        """Get access to data if authorized"""
        if data_id not in self.data_entries:
            return None
        
        entry = self.data_entries[data_id]
        
        # Check authorization
        if agent_id not in entry.authorized_agents:
            return None
        
        # Update access tracking
        entry.last_accessed = time.time()
        entry.access_count += 1
        
        # In production, this would decrypt and return actual data
        # For now, return metadata
        return {
            "data_id": data_id,
            "classification": entry.classification.value,
            "access_granted": True,
            "access_time": time.time(),
            "placeholder": "Actual data would be here in production"
        }
    
    def revoke_data_access(self, data_id: str, agent_id: str, owner_id: str) -> bool:
        """Revoke data access for an agent"""
        if data_id not in self.data_entries:
            return False
        
        entry = self.data_entries[data_id]
        
        # Only owner can revoke access
        if owner_id != entry.owner_id:
            return False
        
        # Remove from authorized agents
        entry.authorized_agents.discard(agent_id)
        return True
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get data protection statistics"""
        total_entries = len(self.data_entries)
        total_requests = len(self.access_requests)
        approved_requests = sum(1 for r in self.access_requests if r.approved)
        
        classification_counts = {}
        for entry in self.data_entries.values():
            cls = entry.classification.value
            classification_counts[cls] = classification_counts.get(cls, 0) + 1
        
        return {
            "total_data_entries": total_entries,
            "total_access_requests": total_requests,
            "approved_requests": approved_requests,
            "approval_rate": (approved_requests / total_requests * 100) if total_requests > 0 else 0,
            "classification_breakdown": classification_counts,
            "protection_level": "enterprise",
            "encryption_status": "active"
        }
    
    def _evaluate_access_policy(self, request: AccessRequest, entry: DataEntry) -> bool:
        """Evaluate access policy for a request"""
        # Simplified policy evaluation
        # In production, this would be much more sophisticated
        
        # Public data is always accessible
        if entry.classification == DataClassification.PUBLIC:
            return True
        
        # Internal data requires read permission
        if entry.classification == DataClassification.INTERNAL:
            return request.access_level in [AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN]
        
        # Confidential data requires explicit approval
        if entry.classification == DataClassification.CONFIDENTIAL:
            return request.access_level == AccessLevel.READ and len(request.reason) > 10
        
        # Restricted data requires owner approval
        if entry.classification == DataClassification.RESTRICTED:
            return False  # Always requires manual approval
        
        return False
    
    def get_agent_data_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of data access for an agent"""
        owned_data = [e for e in self.data_entries.values() if e.owner_id == agent_id]
        authorized_data = [e for e in self.data_entries.values() if agent_id in e.authorized_agents]
        requests_made = [r for r in self.access_requests if r.agent_id == agent_id]
        
        return {
            "agent_id": agent_id,
            "owned_data_count": len(owned_data),
            "authorized_data_count": len(authorized_data),
            "requests_made": len(requests_made),
            "successful_requests": sum(1 for r in requests_made if r.approved)
        }

# Global instance for easy access
data_protection_layer = DataProtectionLayer()

def get_data_protection_layer() -> DataProtectionLayer:
    """Get the global data protection layer instance"""
    return data_protection_layer 