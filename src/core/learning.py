# Learning collaboration framework for agent ecosystem
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto
import uuid
import datetime
import json

class LearningCapabilityType(Enum):
    """Types of learning capabilities agents can have"""
    SUPERVISED_LEARNING = "supervised"
    REINFORCEMENT_LEARNING = "reinforcement" 
    FEDERATED_LEARNING = "federated"
    TRANSFER_LEARNING = "transfer"
    META_LEARNING = "meta"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    PARAMETER_SHARING = "parameter_sharing"

class LearningSessionStatus(Enum):
    """Status of learning sessions"""
    CREATED = "created"
    ACTIVE = "active" 
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class LearningCapability:
    """Enhanced capability specifically for learning tasks"""
    name: str
    type: LearningCapabilityType
    description: str
    input_modalities: List[str] = field(default_factory=list)  # ["text", "vision", "audio"]
    output_modalities: List[str] = field(default_factory=list)
    model_architecture: Optional[str] = None  # "transformer", "cnn", "lstm"
    training_data_requirements: Dict[str, Any] = field(default_factory=dict)
    compute_requirements: Dict[str, Any] = field(default_factory=dict)
    collaboration_protocols: List[str] = field(default_factory=list)  # ["parameter_sharing", "gradient_sharing"]
    keywords: List[str] = field(default_factory=list)
    authorized_requester_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "input_modalities": self.input_modalities,
            "output_modalities": self.output_modalities,
            "model_architecture": self.model_architecture,
            "training_data_requirements": self.training_data_requirements,
            "compute_requirements": self.compute_requirements,
            "collaboration_protocols": self.collaboration_protocols,
            "keywords": self.keywords,
            "authorized_requester_ids": self.authorized_requester_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningCapability':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            type=LearningCapabilityType(data["type"]),
            description=data["description"],
            input_modalities=data.get("input_modalities", []),
            output_modalities=data.get("output_modalities", []),
            model_architecture=data.get("model_architecture"),
            training_data_requirements=data.get("training_data_requirements", {}),
            compute_requirements=data.get("compute_requirements", {}),
            collaboration_protocols=data.get("collaboration_protocols", []),
            keywords=data.get("keywords", []),
            authorized_requester_ids=data.get("authorized_requester_ids")
        )

@dataclass
class LearningTaskSpec:
    """Specification for a learning task"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str = ""
    task_type: str = "supervised"  # "supervised", "reinforcement", "federated", etc.
    objective: str = ""
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    collaboration_preferences: List[str] = field(default_factory=list)
    computational_constraints: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime.datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "objective": self.objective,
            "data_requirements": self.data_requirements,
            "success_criteria": self.success_criteria,
            "collaboration_preferences": self.collaboration_preferences,
            "computational_constraints": self.computational_constraints,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "metadata": self.metadata
        }

@dataclass 
class LearningSession:
    """Represents an active learning collaboration session"""
    session_id: str = field(default_factory=lambda: f"learn_{uuid.uuid4().hex[:8]}")
    task_spec: LearningTaskSpec = field(default_factory=LearningTaskSpec)
    creator_id: str = ""
    participants: Set[str] = field(default_factory=set)
    status: LearningSessionStatus = LearningSessionStatus.CREATED
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    shared_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # agent_id -> parameters
    learning_progress: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # agent_id -> progress
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_participant(self, agent_id: str) -> bool:
        """Add a participant to the session"""
        if agent_id not in self.participants:
            self.participants.add(agent_id)
            self.updated_at = datetime.datetime.now()
            return True
        return False

    def remove_participant(self, agent_id: str) -> bool:
        """Remove a participant from the session"""
        if agent_id in self.participants:
            self.participants.remove(agent_id)
            # Clean up their data
            self.shared_parameters.pop(agent_id, None)
            self.learning_progress.pop(agent_id, None)
            self.updated_at = datetime.datetime.now()
            return True
        return False

    def update_parameters(self, agent_id: str, parameters: Dict[str, Any]) -> bool:
        """Update shared parameters from an agent"""
        if agent_id in self.participants:
            self.shared_parameters[agent_id] = {
                "parameters": parameters,
                "timestamp": datetime.datetime.now().isoformat(),
                "version": len(self.shared_parameters.get(agent_id, {})) + 1
            }
            self.updated_at = datetime.datetime.now()
            return True
        return False

    def update_progress(self, agent_id: str, progress: Dict[str, Any]) -> bool:
        """Update learning progress from an agent"""
        if agent_id in self.participants:
            self.learning_progress[agent_id] = {
                **progress,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.updated_at = datetime.datetime.now()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "task_spec": self.task_spec.to_dict(),
            "creator_id": self.creator_id,
            "participants": list(self.participants),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "shared_parameters": self.shared_parameters,
            "learning_progress": self.learning_progress,
            "results": self.results,
            "metadata": self.metadata
        }

@dataclass
class TestEnvironment:
    """Represents a test/sandbox environment"""
    env_id: str = field(default_factory=lambda: f"test_env_{uuid.uuid4().hex[:8]}")
    env_name: str = ""
    env_type: str = "basic"  # "basic", "simulation", "sandbox"
    creator_id: str = ""
    participants: Set[str] = field(default_factory=set)
    configuration: Dict[str, Any] = field(default_factory=dict)
    test_data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # agent_id -> test_results
    status: str = "created"  # "created", "running", "completed", "failed"
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_test_result(self, agent_id: str, results: Dict[str, Any]) -> bool:
        """Add test results for an agent"""
        self.results[agent_id] = {
            **results,
            "timestamp": datetime.datetime.now().isoformat()
        }
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "env_id": self.env_id,
            "env_name": self.env_name,
            "env_type": self.env_type,
            "creator_id": self.creator_id,
            "participants": list(self.participants),
            "configuration": self.configuration,
            "test_data": self.test_data,
            "results": self.results,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
