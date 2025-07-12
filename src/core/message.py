from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto, IntEnum
from typing import Any, Dict, Optional, List, Set
import uuid
import json
from json import JSONDecodeError

class MessagePriority(IntEnum):
    """Priority levels for message routing."""
    CRITICAL = 0    # System-critical messages (e.g., shutdown, security alerts)
    HIGH = 1        # Time-sensitive operations
    NORMAL = 2      # Standard operations
    LOW = 3         # Background tasks
    BULK = 4        # Batch operations, logging

class MessageType(Enum):
    REGISTER = auto()
    REGISTER_ACK = auto()
    REQUEST = auto()
    RESPONSE = auto()
    ACTION = auto() # Agent wants to perform an action in the world
    INFO = auto() # Agent shares information
    ERROR = auto()
    DISCOVER_SERVICES = auto()    # Agent asks lobby to find other agents/services
    SERVICES_AVAILABLE = auto()   # Lobby responds with available services/agents
    ADVERTISE_CAPABILITIES = auto() # Agent proactively tells lobby its capabilities
    BROADCAST = auto()        # New: For messages to all agents
    ACK = auto()             # New: For message acknowledgment
    HEALTH_CHECK = auto()     # New: For agent health monitoring
    CAPABILITY_UPDATE = auto() # New: For dynamic capability updates
    TASK_OUTCOME_REPORT = auto() # New: For agents to report task success/failure to Lobby for reputation
    TASK_COMPLETION = auto()     # For task completion responses from agents
    
    # Learning Collaboration Messages
    CREATE_LEARNING_SESSION = auto()    # Request to create a learning session
    JOIN_LEARNING_SESSION = auto()      # Request to join existing learning session
    LEAVE_LEARNING_SESSION = auto()     # Request to leave learning session
    SHARE_MODEL_PARAMETERS = auto()     # Share model parameters with session participants
    REQUEST_COLLABORATION = auto()      # Request collaboration on learning task
    REPORT_LEARNING_PROGRESS = auto()   # Report progress on learning task
    CREATE_TEST_ENVIRONMENT = auto()    # Request to create test/sandbox environment
    RUN_MODEL_TEST = auto()            # Run model test in sandbox
    GET_TEST_RESULTS = auto()          # Retrieve test results
    LEARNING_SESSION_UPDATE = auto()    # Notify about session changes

    # New message types for learning & collaboration
    REQUEST_TASK_ASSISTANCE = auto()
    TASK_ASSIGNMENT = auto()  # For task assignment to agents

class MessageValidationError(Exception):
    """Raised when message validation fails."""
    pass

@dataclass
class Message:
    sender_id: str
    receiver_id: str  # Can be "lobby", "broadcast", or specific agent_id
    message_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    conversation_id: Optional[str] = None # ADDED for multi-step conversations
    auth_token: Optional[str] = None # ADDED for authentication
    priority: MessagePriority = field(default=MessagePriority.NORMAL)
    requires_ack: bool = field(default=False)  # Whether message needs acknowledgment
    ack_timeout: Optional[float] = None  # Timeout in seconds for acknowledgment
    broadcast_scope: Optional[Set[str]] = None  # Specific agent types for broadcast
    
    def __post_init__(self):
        """Validate message after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate message contents."""
        if not self.sender_id or not isinstance(self.sender_id, str):
            raise MessageValidationError("Invalid sender_id")
        
        if not self.receiver_id or not isinstance(self.receiver_id, str):
            raise MessageValidationError("Invalid receiver_id")
            
        if self.receiver_id == "broadcast" and not self.broadcast_scope:
            # For broadcast messages, scope must be defined
            raise MessageValidationError("Broadcast message requires broadcast_scope")
            
        if self.requires_ack and not self.ack_timeout:
            # If acknowledgment is required, timeout must be set
            raise MessageValidationError("Acknowledgment timeout required when requires_ack is True")
            
        try:
            # Ensure payload is JSON serializable
            json.dumps(self.payload)
        except (TypeError, JSONDecodeError) as e:
            raise MessageValidationError(f"Payload must be JSON serializable: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.name,
            "payload": self.payload,
            "conversation_id": self.conversation_id, # ADDED
            "auth_token": self.auth_token, # ADDED
            "priority": self.priority.value,
            "requires_ack": self.requires_ack,
            "ack_timeout": self.ack_timeout,
            "broadcast_scope": list(self.broadcast_scope) if self.broadcast_scope else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary format."""
        # Convert broadcast_scope back to set if present
        broadcast_scope = set(data["broadcast_scope"]) if data.get("broadcast_scope") else None
        
        return cls(
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType[data["message_type"]],
            payload=data["payload"],
            message_id=data["message_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            conversation_id=data.get("conversation_id"), # ADDED .get() for backward compatibility if not present
            auth_token=data.get("auth_token"), # ADDED
            priority=MessagePriority(data.get("priority", MessagePriority.NORMAL.value)),
            requires_ack=data.get("requires_ack", False),
            ack_timeout=data.get("ack_timeout"),
            broadcast_scope=broadcast_scope
        )

    def create_ack(self) -> "Message":
        """Create an acknowledgment message for this message."""
        if not self.requires_ack:
            raise MessageValidationError("Cannot create acknowledgment for message that doesn't require it")
            
        return Message(
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            message_type=MessageType.ACK,
            payload={"ack_for": self.message_id},
            conversation_id=self.conversation_id,
            auth_token=self.auth_token,
            priority=MessagePriority.HIGH  # Acknowledgments are high priority
        )

if __name__ == "__main__":
    # Example usage with new features
    try:
        # Example 1: Standard message
        msg1 = Message(
            sender_id="agent_alpha",
            receiver_id="agent_beta",
            message_type=MessageType.REQUEST,
            payload={"task": "get_item", "item_name": "key"},
            priority=MessagePriority.HIGH,
            requires_ack=True,
            ack_timeout=5.0
        )
        print(f"High priority message with ack: {msg1.to_dict()}")

        # Example 2: Broadcast message
        msg2 = Message(
            sender_id="lobby",
            receiver_id="broadcast",
            message_type=MessageType.BROADCAST,
            payload={"announcement": "System update required"},
            broadcast_scope={"worker_agent", "supervisor_agent"},
            priority=MessagePriority.CRITICAL
        )
        print(f"Broadcast message: {msg2.to_dict()}")

        # Example 3: Create acknowledgment
        ack = msg1.create_ack()
        print(f"Acknowledgment message: {ack.to_dict()}")

        # Example 4: This should fail validation (broadcast without scope)
        try:
            invalid_msg = Message(
                sender_id="test",
                receiver_id="broadcast",
                message_type=MessageType.BROADCAST,
                payload={"test": "fail"}
            )
        except MessageValidationError as e:
            print(f"Validation caught error as expected: {e}")

    except MessageValidationError as e:
        print(f"Message validation error: {e}")

    # Example usage
    msg1 = Message(
        sender_id="agent_alpha",
        receiver_id="agent_beta",
        message_type=MessageType.REQUEST,
        payload={"task": "get_item", "item_name": "key"},
    )
    print(f"Message 1 (no conv_id, no auth_token): {msg1.to_dict()}")
    msg1_data = msg1.to_dict()
    recreated_msg1 = Message.from_dict(msg1_data)
    assert recreated_msg1.conversation_id is None
    assert recreated_msg1.auth_token is None

    conv_id = str(uuid.uuid4())
    token = "secret_token_example"
    msg3 = Message(
        sender_id="agent_gamma",
        receiver_id="lobby",
        message_type=MessageType.ACTION,
        payload={"action": "do_something"},
        conversation_id=conv_id,
        auth_token=token
    )
    print(f"Message 3 (with conv_id and auth_token): {msg3.to_dict()}")
    msg3_data = msg3.to_dict()
    recreated_msg3 = Message.from_dict(msg3_data)
    assert recreated_msg3.conversation_id == conv_id
    assert recreated_msg3.auth_token == token
    print("Serialization and deserialization with conversation_id and auth_token successful.") 