"""
Policy Package
--------------
Scaffolding for future robot policy training and inference.

The current runtime is still LLM tool-calling plus trajectory planning. The
modules in this package define the stable data contracts a trainable policy will
need later: observations, actions, model interface, dataset logging, and a small
runtime wrapper.
"""

from policy.actions import (
    ACTION_CARTESIAN_DELTA,
    ACTION_NO_OP,
    ACTION_WAYPOINT,
    PolicyAction,
)
from policy.base import PlaceholderPolicy, PolicyInterface
from policy.observation import CameraObservation, PolicyObservation, RobotObservation
from policy.scripted import ScriptedTrajectoryPolicy

__all__ = [
    "ACTION_CARTESIAN_DELTA",
    "ACTION_NO_OP",
    "ACTION_WAYPOINT",
    "CameraObservation",
    "PlaceholderPolicy",
    "PolicyAction",
    "PolicyInterface",
    "PolicyObservation",
    "RobotObservation",
    "ScriptedTrajectoryPolicy",
]
