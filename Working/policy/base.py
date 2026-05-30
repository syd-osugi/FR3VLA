"""
Policy Interface
----------------
Defines the runtime interface for learned or scripted policies.

Training code can change model internals freely as long as the runtime object
implements this small interface: load a checkpoint and predict a PolicyAction
from a PolicyObservation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from policy.actions import PolicyAction, no_op_action
from policy.observation import PolicyObservation


class PolicyInterface(Protocol):
    """Protocol implemented by trainable policies and scripted baselines."""

    def load(self, checkpoint_path: str) -> None:
        """Load model weights or policy state from disk."""

    def predict(self, observation: PolicyObservation) -> PolicyAction:
        """Return one action for the current observation."""


class PlaceholderPolicy:
    """
    Import-safe policy stub.

    This class is useful while the controller and dataset plumbing are being
    built. It intentionally returns no-op actions so it cannot move the robot by
    accident.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.metadata = metadata or {}
        self.checkpoint_path: Optional[str] = None

    def load(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.metadata["load_status"] = (
            "placeholder_only; no weights were loaded because no model exists yet"
        )

    def predict(self, observation: PolicyObservation) -> PolicyAction:
        return no_op_action(
            "PlaceholderPolicy produced no motion. Replace it with a trained policy."
        )
