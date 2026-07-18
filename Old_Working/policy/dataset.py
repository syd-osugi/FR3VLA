"""
Policy Dataset Logging
----------------------
Utilities for recording demonstrations and policy rollouts.

The initial training loop will need synchronized observations, actions, safety
decisions, and success labels. This module writes a simple JSONL episode format
that can later be converted into a framework-specific dataset.

Expected future work:
- Store RGB/depth frames alongside steps and reference them by path.
- Add compression/chunking for long episodes.
- Add data validation scripts before training.
- Add versioned schema metadata so old datasets remain readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import config as cfg


def _serializable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass
class EpisodeStep:
    """One logged controller timestep."""

    step_index: int
    observation: Any
    action: Any
    safety_result: Optional[Any] = None
    execution_result: Optional[Dict[str, Any]] = None
    reward: Optional[float] = None
    success: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation": _serializable(self.observation),
            "action": _serializable(self.action),
            "safety_result": _serializable(self.safety_result),
            "execution_result": self.execution_result,
            "reward": self.reward,
            "success": self.success,
            "metadata": self.metadata,
        }


class EpisodeLogger:
    """
    Minimal JSONL episode logger.

    This is intentionally lightweight. Real image data should be saved by the
    observation/camera pipeline and referenced from each observation rather than
    embedded directly in JSON.
    """

    def __init__(self, dataset_root: Optional[str] = None) -> None:
        self.dataset_root = Path(dataset_root or getattr(cfg, "POLICY_DATASET_ROOT", "data/policy"))
        self.episode_dir: Optional[Path] = None
        self.steps_path: Optional[Path] = None
        self.step_index = 0

    def start_episode(
        self,
        task_instruction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        episode_id = f"{timestamp}_{uuid4().hex[:8]}"
        self.episode_dir = self.dataset_root / episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.steps_path = self.episode_dir / "steps.jsonl"
        self.step_index = 0

        episode_metadata = {
            "episode_id": episode_id,
            "created_utc": timestamp,
            "task_instruction": task_instruction,
            "schema": "policy_episode_v0_placeholder",
            "metadata": metadata or {},
            "todo": (
                "Add calibration snapshots, camera stream metadata, policy version, "
                "operator ID, and success criteria before collecting serious data."
            ),
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(episode_metadata, indent=2),
            encoding="utf-8",
        )
        return self.episode_dir

    def record_step(
        self,
        observation: Any,
        action: Any,
        safety_result: Optional[Any] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EpisodeStep:
        if self.steps_path is None:
            raise RuntimeError("start_episode() must be called before record_step()")

        step = EpisodeStep(
            step_index=self.step_index,
            observation=observation,
            action=action,
            safety_result=safety_result,
            execution_result=execution_result,
            reward=reward,
            success=success,
            metadata=metadata or {},
        )
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step.to_dict(), default=str) + "\n")
        self.step_index += 1
        return step

    def finish_episode(
        self,
        success: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.episode_dir is None:
            raise RuntimeError("start_episode() must be called before finish_episode()")

        summary = {
            "success": success,
            "step_count": self.step_index,
            "metadata": metadata or {},
            "todo": "Add task-specific metrics and failure labels.",
        }
        (self.episode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary
