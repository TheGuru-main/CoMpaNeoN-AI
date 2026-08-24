import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# The four passkeys for rotation (hidden)
PASSKEYS = [
    "###@@@55019855199##@@@",
    "###@@@56019855199##@@@",
    "###@@@55019855197##@@@",
    "###@@@55019655199##@@@",
]

class BubbleJumboRules:
    """
    Security layer with BubbleJumbo K escalation and passkey rotation.
    - Four passkeys rotate upon attack detection.
    - K increases by +17 on each attack.
    - Passkeys are hashed before storage.
    """

    def __init__(self, K: int = 4, D: int = 1, initial_passkeys: Optional[List[str]] = None):
        self.K = K
        self.D = D
        self.passkey_index = 0          # current active passkey index
        self.passkeys = initial_passkeys if initial_passkeys else PASSKEYS.copy()
        self.hashed_passkeys = [self._hash_key(k) for k in self.passkeys]
        self.rotation_count = 0
        self.last_attack_time = None

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def get_active_passkey(self) -> str:
        """Return the current active passkey (plain text for internal use only)."""
        return self.passkeys[self.passkey_index % len(self.passkeys)]

    def verify_passkey(self, incoming: str) -> bool:
        """Check if incoming passkey matches any of the rotated keys."""
        hashed_incoming = self._hash_key(incoming)
        return hashed_incoming in self.hashed_passkeys

    def rotate_passkey(self):
        """Move to the next passkey in the rotation."""
        self.passkey_index = (self.passkey_index + 1) % len(self.passkeys)
        self.rotation_count += 1
        return self.get_active_passkey()

    def detect_attack(self, incoming: str) -> bool:
        """Return True if the incoming passkey is incorrect."""
        if not self.verify_passkey(incoming):
            self.last_attack_time = time.time()
            return True
        return False

    def escalate(self):
        """On attack, increase K and rotate passkey."""
        self.K += 17
        self.rotate_passkey()
        return self.K

    def get_status(self) -> dict:
        return {
            "K": self.K,
            "active_passkey_index": self.passkey_index,
            "rotation_count": self.rotation_count,
            "last_attack_time": self.last_attack_time,
        }