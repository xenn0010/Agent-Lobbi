"""
Agent Learning Capabilities for Agent Lobbi
Defines learning capabilities and sessions for collaborative learning
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone


class LearningCapabilityType(Enum):
    """Types of learning capabilities"""
    SUPERVISED_LEARNING = "supervised_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TRANSFER_LEARNING = "transfer_learning"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    FEDERATED_LEARNING = "federated_learning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"


@dataclass
class LearningCapability:
    """Defines a learning capability of an agent"""
    type: LearningCapabilityType
    name: str
    description: str
    model_types: List[str] = field(default_factory=list)
    data_types: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "name": self.name,
            "description": self.description,
            "model_types": self.model_types,
            "data_types": self.data_types,
            "requirements": self.requirements
        }


class LearningSessionStatus(Enum):
    """Status of a learning session"""
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LearningSession:
    """Represents a collaborative learning session"""
    session_id: str
    name: str
    description: str
    participants: Set[str] = field(default_factory=set)
    status: LearningSessionStatus = LearningSessionStatus.CREATED
    shared_data: Dict[str, Any] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "description": self.description,
            "participants": list(self.participants),
            "status": self.status.value,
            "shared_data": self.shared_data,
            "model_parameters": self.model_parameters,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


# Add aliases for backward compatibility
LearningTaskSpec = Dict[str, Any]  # Simplified for now
TestEnvironment = Dict[str, Any]   # Simplified for now 