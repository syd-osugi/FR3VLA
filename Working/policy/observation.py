"""
Policy Observation Contracts
----------------------------
Defines the data a trainable policy should receive at each control timestep.

The current LLM tools return text JSON and occasional images. Training needs a
more stable, timestamped, structured observation. This module provides that
structure without forcing a particular ML framework or camera storage format.

Expected future work:
- Add camera intrinsics/extrinsics snapshots to every episode.
- Decide whether images are stored as arrays, compressed files, or shared-memory
  handles during online inference.
- Add object masks, detections, or language embeddings if the trained policy uses
  higher-level perception features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CameraObservation:
    """One camera stream at one timestep."""

    camera_name: str
    timestamp_s: float = field(default_factory=time.time)
    rgb_image_path: Optional[str] = None
    depth_image_path: Optional[str] = None
    rgb_frame_id: Optional[str] = None
    depth_frame_id: Optional[str] = None
    intrinsics: Dict[str, Any] = field(default_factory=dict)
    extrinsics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "timestamp_s": self.timestamp_s,
            "rgb_image_path": self.rgb_image_path,
            "depth_image_path": self.depth_image_path,
            "rgb_frame_id": self.rgb_frame_id,
            "depth_frame_id": self.depth_frame_id,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "metadata": self.metadata,
        }


@dataclass
class RobotObservation:
    """Robot state used by a policy and by dataset logging."""

    timestamp_s: float = field(default_factory=time.time)
    ee_pose: Optional[List[List[float]]] = None
    joint_positions: Optional[List[float]] = None
    joint_velocities: Optional[List[float]] = None
    gripper_width_m: Optional[float] = None
    measured_force_n: Optional[List[float]] = None
    measured_torque_nm: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "ee_pose": self.ee_pose,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "gripper_width_m": self.gripper_width_m,
            "measured_force_n": self.measured_force_n,
            "measured_torque_nm": self.measured_torque_nm,
            "metadata": self.metadata,
        }


@dataclass
class PolicyObservation:
    """
    Complete policy input for one timestep.

    The policy can use raw camera observations, robot state, a task instruction,
    and optional target/object features. Keep this object serializable so it can
    be written directly into demonstration datasets.
    """

    timestamp_s: float = field(default_factory=time.time)
    task_instruction: Optional[str] = None
    robot: Optional[RobotObservation] = None
    cameras: Dict[str, CameraObservation] = field(default_factory=dict)
    target_xyz_m: Optional[List[float]] = None
    object_features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": self.timestamp_s,
            "task_instruction": self.task_instruction,
            "robot": self.robot.to_dict() if self.robot is not None else None,
            "cameras": {
                name: camera.to_dict()
                for name, camera in self.cameras.items()
            },
            "target_xyz_m": self.target_xyz_m,
            "object_features": self.object_features,
            "metadata": self.metadata,
        }


class ObservationBuilder:
    """
    Placeholder object that will build policy observations at runtime.

    In the working system this class should:
    - capture synchronized D435/D405 frames,
    - read the latest robot state,
    - attach calibration metadata,
    - optionally run object detection/tracking,
    - return a PolicyObservation at the controller rate.
    """

    def __init__(
        self,
        robot_state_provider: Optional[Callable[[], RobotObservation]] = None,
        camera_observation_provider: Optional[Callable[[], Dict[str, CameraObservation]]] = None,
        target_provider: Optional[Callable[[], Optional[List[float]]]] = None,
    ) -> None:
        self.robot_state_provider = robot_state_provider
        self.camera_observation_provider = camera_observation_provider
        self.target_provider = target_provider

    def build(self, task_instruction: Optional[str] = None) -> PolicyObservation:
        robot = self.robot_state_provider() if self.robot_state_provider else None
        cameras = self.camera_observation_provider() if self.camera_observation_provider else {}
        target_xyz_m = self.target_provider() if self.target_provider else None

        return PolicyObservation(
            task_instruction=task_instruction,
            robot=robot,
            cameras=cameras,
            target_xyz_m=target_xyz_m,
            metadata={
                "builder_status": (
                    "placeholder_observation"
                    if not (self.robot_state_provider and self.camera_observation_provider)
                    else "provider_backed_observation"
                ),
                "todo": (
                    "Replace placeholder providers with synchronized camera capture, "
                    "fresh robot state, and automatic target tracking before training."
                ),
            },
        )
