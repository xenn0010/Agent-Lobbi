"""
Task Encryption Layer for Agent Lobbi
Encrypts task data, collaboration payloads, and sensitive information during transit and storage
"""

import asyncio
import json
import logging
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import sqlite3

logger = logging.getLogger(__name__)


class EncryptionLevel(Enum):
    """Task encryption levels"""
    NONE = "none"
    BASIC = "basic"           # Symmetric encryption
    ENHANCED = "enhanced"     # Symmetric + integrity
    ENTERPRISE = "enterprise" # Asymmetric + symmetric hybrid
    CLASSIFIED = "classified" # Maximum security


class TaskSensitivity(Enum):
    """Task data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


@dataclass
class EncryptionKeys:
    """Encryption key management"""
    agent_id: str
    public_key: Optional[bytes] = None
    private_key: Optional[bytes] = None
    symmetric_key: Optional[bytes] = None
    created_at: str = ""
    expires_at: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class EncryptedTask:
    """Encrypted task data container"""
    task_id: str
    encrypted_payload: str
    encryption_level: EncryptionLevel
    sensitivity: TaskSensitivity
    sender_id: str
    recipient_id: str
    encryption_metadata: Dict[str, Any]
    integrity_hash: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class TaskEncryptionManager:
    """
    Comprehensive task encryption system
    Provides end-to-end encryption for task data, collaboration payloads, and sensitive information
    """
    
    def __init__(self, db_path: str = "task_encryption.db"):
        self.db_path = db_path
        self.agent_keys: Dict[str, EncryptionKeys] = {}
        self.encrypted_tasks: Dict[str, EncryptedTask] = {}
        
        # Master encryption key for system-level operations
        self.master_key = self._generate_master_key()
        self.master_cipher = Fernet(base64.urlsafe_b64encode(self.master_key))
        
        # Encryption configurations
        self.encryption_configs = {
            EncryptionLevel.NONE: {"enabled": False},
            EncryptionLevel.BASIC: {"algorithm": "Fernet", "key_size": 32},
            EncryptionLevel.ENHANCED: {"algorithm": "Fernet", "key_size": 32, "integrity": True},
            EncryptionLevel.ENTERPRISE: {"algorithm": "RSA+AES", "rsa_size": 2048, "aes_size": 32},
            EncryptionLevel.CLASSIFIED: {"algorithm": "RSA+AES", "rsa_size": 4096, "aes_size": 32, "multi_layer": True}
        }
        
        self._initialize_database()
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key"""
        return secrets.token_bytes(32)
    
    def _initialize_database(self):
        """Initialize encryption database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Agent keys table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_keys (
                    agent_id TEXT PRIMARY KEY,
                    public_key BLOB,
                    private_key BLOB,
                    symmetric_key BLOB,
                    created_at TEXT,
                    expires_at TEXT
                )
            ''')
            
            # Encrypted tasks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS encrypted_tasks (
                    task_id TEXT PRIMARY KEY,
                    encrypted_payload TEXT,
                    encryption_level TEXT,
                    sensitivity TEXT,
                    sender_id TEXT,
                    recipient_id TEXT,
                    encryption_metadata TEXT,
                    integrity_hash TEXT,
                    timestamp TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Task encryption database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption database: {e}")
    
    async def register_agent_encryption(self, agent_id: str, encryption_level: EncryptionLevel = EncryptionLevel.ENHANCED) -> EncryptionKeys:
        """Register agent for encryption and generate keys"""
        
        keys = EncryptionKeys(agent_id=agent_id)
        
        # Generate symmetric key for all encryption levels
        keys.symmetric_key = secrets.token_bytes(32)
        
        # Always generate RSA keypair for future compatibility
        key_size = 2048  # Default RSA key size
        if encryption_level == EncryptionLevel.CLASSIFIED:
            key_size = 4096  # Larger keys for classified
        
        if encryption_level in [EncryptionLevel.ENTERPRISE, EncryptionLevel.CLASSIFIED]:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            
            keys.private_key = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            keys.public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        
        # Store keys
        self.agent_keys[agent_id] = keys
        await self._save_agent_keys(keys)
        
        logger.info(f"Generated encryption keys for agent {agent_id} (level: {encryption_level.value})")
        return keys
    
    async def encrypt_task_data(self, 
                              task_data: Dict[str, Any],
                              sender_id: str,
                              recipient_id: str,
                              sensitivity: TaskSensitivity = TaskSensitivity.INTERNAL,
                              encryption_level: EncryptionLevel = EncryptionLevel.ENHANCED) -> EncryptedTask:
        """
        Encrypt task data with specified security level
        """
        
        task_id = task_data.get('task_id', secrets.token_hex(16))
        
        # Ensure agents have encryption keys
        if sender_id not in self.agent_keys:
            await self.register_agent_encryption(sender_id, encryption_level)
        if recipient_id not in self.agent_keys:
            await self.register_agent_encryption(recipient_id, encryption_level)
        
        # Upgrade keys if needed for higher encryption levels
        if encryption_level in [EncryptionLevel.ENTERPRISE, EncryptionLevel.CLASSIFIED]:
            sender_keys = self.agent_keys[sender_id]
            recipient_keys = self.agent_keys[recipient_id]
            
            if sender_keys.public_key is None or recipient_keys.public_key is None:
                # Re-register with proper encryption level
                await self.register_agent_encryption(sender_id, encryption_level)
                await self.register_agent_encryption(recipient_id, encryption_level)
        
        # Serialize task data
        task_json = json.dumps(task_data, sort_keys=True)
        task_bytes = task_json.encode('utf-8')
        
        # Encrypt based on level
        if encryption_level == EncryptionLevel.NONE:
            encrypted_payload = base64.b64encode(task_bytes).decode('utf-8')
            encryption_metadata = {"algorithm": "none"}
            
        elif encryption_level == EncryptionLevel.BASIC:
            encrypted_payload = await self._encrypt_symmetric(task_bytes, recipient_id)
            encryption_metadata = {"algorithm": "fernet", "key_source": "symmetric"}
            
        elif encryption_level == EncryptionLevel.ENHANCED:
            encrypted_payload = await self._encrypt_symmetric(task_bytes, recipient_id)
            encryption_metadata = {"algorithm": "fernet", "key_source": "symmetric", "integrity": True}
            
        elif encryption_level == EncryptionLevel.ENTERPRISE:
            encrypted_payload = await self._encrypt_hybrid(task_bytes, sender_id, recipient_id)
            encryption_metadata = {"algorithm": "rsa_aes", "rsa_size": 2048, "aes_size": 256}
            
        elif encryption_level == EncryptionLevel.CLASSIFIED:
            encrypted_payload = await self._encrypt_classified(task_bytes, sender_id, recipient_id)
            encryption_metadata = {"algorithm": "multi_layer", "layers": ["rsa_aes", "fernet"], "rsa_size": 4096}
        
        else:
            raise ValueError(f"Unsupported encryption level: {encryption_level}")
        
        # Generate integrity hash
        integrity_data = f"{sender_id}:{recipient_id}:{encrypted_payload}:{task_id}"
        integrity_hash = hmac.new(
            self.master_key,
            integrity_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Create encrypted task container
        encrypted_task = EncryptedTask(
            task_id=task_id,
            encrypted_payload=encrypted_payload,
            encryption_level=encryption_level,
            sensitivity=sensitivity,
            sender_id=sender_id,
            recipient_id=recipient_id,
            encryption_metadata=encryption_metadata,
            integrity_hash=integrity_hash
        )
        
        # Store encrypted task
        self.encrypted_tasks[task_id] = encrypted_task
        await self._save_encrypted_task(encrypted_task)
        
        logger.info(f"Encrypted task {task_id} (level: {encryption_level.value}, sensitivity: {sensitivity.value})")
        return encrypted_task
    
    async def decrypt_task_data(self, encrypted_task: EncryptedTask, requesting_agent: str) -> Dict[str, Any]:
        """
        Decrypt task data if agent is authorized
        """
        
        # Verify requesting agent is authorized
        if requesting_agent not in [encrypted_task.sender_id, encrypted_task.recipient_id]:
            raise PermissionError(f"Agent {requesting_agent} not authorized to decrypt task {encrypted_task.task_id}")
        
        # Verify integrity
        integrity_data = f"{encrypted_task.sender_id}:{encrypted_task.recipient_id}:{encrypted_task.encrypted_payload}:{encrypted_task.task_id}"
        expected_hash = hmac.new(
            self.master_key,
            integrity_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(encrypted_task.integrity_hash, expected_hash):
            raise ValueError("Task integrity verification failed - possible tampering detected")
        
        # Decrypt based on encryption level
        if encrypted_task.encryption_level == EncryptionLevel.NONE:
            decrypted_bytes = base64.b64decode(encrypted_task.encrypted_payload.encode('utf-8'))
            
        elif encrypted_task.encryption_level in [EncryptionLevel.BASIC, EncryptionLevel.ENHANCED]:
            decrypted_bytes = await self._decrypt_symmetric(encrypted_task.encrypted_payload, requesting_agent)
            
        elif encrypted_task.encryption_level == EncryptionLevel.ENTERPRISE:
            decrypted_bytes = await self._decrypt_hybrid(encrypted_task.encrypted_payload, requesting_agent)
            
        elif encrypted_task.encryption_level == EncryptionLevel.CLASSIFIED:
            decrypted_bytes = await self._decrypt_classified(encrypted_task.encrypted_payload, requesting_agent)
        
        else:
            raise ValueError(f"Unsupported encryption level: {encrypted_task.encryption_level}")
        
        # Parse task data
        task_json = decrypted_bytes.decode('utf-8')
        task_data = json.loads(task_json)
        
        logger.info(f"Decrypted task {encrypted_task.task_id} for agent {requesting_agent}")
        return task_data
    
    async def _encrypt_symmetric(self, data: bytes, recipient_id: str) -> str:
        """Encrypt with symmetric key"""
        recipient_keys = self.agent_keys[recipient_id]
        cipher = Fernet(base64.urlsafe_b64encode(recipient_keys.symmetric_key))
        encrypted = cipher.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    async def _decrypt_symmetric(self, encrypted_data: str, agent_id: str) -> bytes:
        """Decrypt with symmetric key"""
        agent_keys = self.agent_keys[agent_id]
        cipher = Fernet(base64.urlsafe_b64encode(agent_keys.symmetric_key))
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        return cipher.decrypt(encrypted_bytes)
    
    async def _encrypt_hybrid(self, data: bytes, sender_id: str, recipient_id: str) -> str:
        """Encrypt with RSA+AES hybrid"""
        # Generate random AES key
        aes_key = secrets.token_bytes(32)
        
        # Encrypt data with AES
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data for AES
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Encrypt AES key with recipient's RSA public key
        recipient_keys = self.agent_keys[recipient_id]
        public_key = serialization.load_pem_public_key(recipient_keys.public_key)
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted key, IV, and data
        combined = base64.b64encode(encrypted_aes_key).decode('utf-8') + ":" + \
                  base64.b64encode(iv).decode('utf-8') + ":" + \
                  base64.b64encode(encrypted_data).decode('utf-8')
        
        return combined
    
    async def _decrypt_hybrid(self, encrypted_data: str, agent_id: str) -> bytes:
        """Decrypt RSA+AES hybrid"""
        # Split combined data
        parts = encrypted_data.split(':')
        encrypted_aes_key = base64.b64decode(parts[0])
        iv = base64.b64decode(parts[1])
        encrypted_payload = base64.b64decode(parts[2])
        
        # Decrypt AES key with private RSA key
        agent_keys = self.agent_keys[agent_id]
        private_key = serialization.load_pem_private_key(agent_keys.private_key, password=None)
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt data with AES
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_payload) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    async def _encrypt_classified(self, data: bytes, sender_id: str, recipient_id: str) -> str:
        """Multi-layer encryption for classified data"""
        # Layer 1: Hybrid encryption
        layer1 = await self._encrypt_hybrid(data, sender_id, recipient_id)
        
        # Layer 2: Master key encryption
        layer2_bytes = layer1.encode('utf-8')
        layer2 = self.master_cipher.encrypt(layer2_bytes)
        
        return base64.b64encode(layer2).decode('utf-8')
    
    async def _decrypt_classified(self, encrypted_data: str, agent_id: str) -> bytes:
        """Multi-layer decryption for classified data"""
        # Layer 2: Master key decryption
        layer2 = base64.b64decode(encrypted_data.encode('utf-8'))
        layer1_bytes = self.master_cipher.decrypt(layer2)
        layer1 = layer1_bytes.decode('utf-8')
        
        # Layer 1: Hybrid decryption
        return await self._decrypt_hybrid(layer1, agent_id)
    
    async def _save_agent_keys(self, keys: EncryptionKeys):
        """Save agent keys to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO agent_keys 
                (agent_id, public_key, private_key, symmetric_key, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                keys.agent_id,
                keys.public_key,
                keys.private_key,
                keys.symmetric_key,
                keys.created_at,
                keys.expires_at
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save agent keys: {e}")
    
    async def _save_encrypted_task(self, encrypted_task: EncryptedTask):
        """Save encrypted task to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO encrypted_tasks 
                (task_id, encrypted_payload, encryption_level, sensitivity, 
                 sender_id, recipient_id, encryption_metadata, integrity_hash, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                encrypted_task.task_id,
                encrypted_task.encrypted_payload,
                encrypted_task.encryption_level.value,
                encrypted_task.sensitivity.value,
                encrypted_task.sender_id,
                encrypted_task.recipient_id,
                json.dumps(encrypted_task.encryption_metadata),
                encrypted_task.integrity_hash,
                encrypted_task.timestamp
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save encrypted task: {e}")
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get encryption system status"""
        return {
            'registered_agents': len(self.agent_keys),
            'encrypted_tasks': len(self.encrypted_tasks),
            'encryption_levels': [level.value for level in EncryptionLevel],
            'sensitivity_levels': [level.value for level in TaskSensitivity],
            'master_key_active': self.master_cipher is not None,
            'database_path': self.db_path
        } 