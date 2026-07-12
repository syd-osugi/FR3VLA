"""
Shared Franka connection setup.

This module keeps robot-side control parameters in one place so main motion and
calibration scripts do not drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config as cfg


LOAD_PROFILE_RUNTIME = "runtime"
LOAD_PROFILE_D405_CALIBRATION = "d405_calibration"
LOAD_PROFILE_D435_CALIBRATION = "d435_calibration"

_LOAD_PROFILE_FIELDS = {
    LOAD_PROFILE_RUNTIME: {
        "mass": "FRANKA_LOAD_MASS_KG",
        "center_of_mass": "FRANKA_LOAD_CENTER_OF_MASS_IN_FLANGE_M",
        "inertia": "FRANKA_LOAD_INERTIA_KGM2",
    },
    LOAD_PROFILE_D405_CALIBRATION: {
        "mass": "D405_CALIBRATION_LOAD_MASS_KG",
        "center_of_mass": "D405_CALIBRATION_LOAD_CENTER_OF_MASS_IN_FLANGE_M",
        "inertia": "D405_CALIBRATION_LOAD_INERTIA_KGM2",
    },
    LOAD_PROFILE_D435_CALIBRATION: {
        "mass": "D435_CALIBRATION_LOAD_MASS_KG",
        "center_of_mass": "D435_CALIBRATION_LOAD_CENTER_OF_MASS_IN_FLANGE_M",
        "inertia": "D435_CALIBRATION_LOAD_INERTIA_KGM2",
    },
}


def _finite_float_list(values: Any, expected_len: int, name: str) -> list[float]:
    try:
        parsed = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {expected_len} numeric values") from exc

    if len(parsed) != expected_len or not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must contain {expected_len} finite values, got: {values}")
    return parsed


def _load_profile_fields(load_profile: str) -> dict[str, str]:
    try:
        return _LOAD_PROFILE_FIELDS[load_profile]
    except KeyError as exc:
        valid = ", ".join(sorted(_LOAD_PROFILE_FIELDS))
        raise ValueError(f"Unknown Franka load profile {load_profile!r}. Expected one of: {valid}") from exc


def get_configured_franka_load(load_profile: str = LOAD_PROFILE_RUNTIME) -> dict[str, Any]:
    """Return the configured gripper/tool load model."""
    fields = _load_profile_fields(load_profile)
    mass_name = fields["mass"]
    center_of_mass_name = fields["center_of_mass"]
    inertia_name = fields["inertia"]

    mass_kg = float(getattr(cfg, mass_name, 0.0))
    if mass_kg < 0.0 or not np.isfinite(mass_kg):
        raise ValueError(f"{mass_name} must be finite and non-negative, got: {mass_kg}")

    center_of_mass_m = _finite_float_list(
        getattr(cfg, center_of_mass_name, (0.0, 0.0, 0.0)),
        3,
        center_of_mass_name,
    )
    inertia_kgm2 = _finite_float_list(
        getattr(cfg, inertia_name, (0.0,) * 9),
        9,
        inertia_name,
    )

    return {
        "profile": load_profile,
        "config_keys": fields.copy(),
        "mass_kg": mass_kg,
        "center_of_mass_in_flange_m": center_of_mass_m,
        "inertia_kgm2": inertia_kgm2,
        "applied": False,
    }


def apply_franka_load(robot: Any, load_profile: str = LOAD_PROFILE_RUNTIME) -> dict[str, Any]:
    """
    Apply configured load parameters to a connected Franka robot.

    A zero-mass config intentionally skips the API call. If a positive payload
    mass is configured but the Python binding does not expose a load setter, we
    fail instead of silently moving with stale gravity compensation.
    """
    load = get_configured_franka_load(load_profile)
    if load["mass_kg"] <= 0.0:
        load["reason"] = f"{load['config_keys']['mass']} is 0.0; custom load model not applied."
        return load

    set_load = getattr(robot, "set_load", None)
    if not callable(set_load):
        set_load = getattr(robot, "setLoad", None)
    if not callable(set_load):
        raise RuntimeError(
            f"A positive {load['config_keys']['mass']} is configured, but this pylibfranka "
            "Robot object does not expose set_load/setLoad."
        )

    set_load(
        load["mass_kg"],
        load["center_of_mass_in_flange_m"],
        load["inertia_kgm2"],
    )
    load["applied"] = True
    return load


def apply_franka_control_config(
    robot: Any,
    load_profile: str = LOAD_PROFILE_RUNTIME,
) -> dict[str, Any]:
    """Apply collision thresholds, payload, and impedance settings to a robot."""
    torque = float(cfg.FRANKA_COLLISION_TORQUE_NM)
    force = float(cfg.FRANKA_COLLISION_FORCE_N)

    robot.set_collision_behavior(
        [torque] * 7,
        [torque] * 7,
        [force] * 6,
        [force] * 6,
    )

    load = apply_franka_load(robot, load_profile=load_profile)

    set_joint_impedance = getattr(robot, "set_joint_impedance", None)
    if callable(set_joint_impedance):
        set_joint_impedance([3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0])

    set_cartesian_impedance = getattr(robot, "set_cartesian_impedance", None)
    if callable(set_cartesian_impedance):
        set_cartesian_impedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])

    return {
        "collision_torque_nm": torque,
        "collision_force_n": force,
        "load_profile": load_profile,
        "load": load,
        "joint_impedance_configured": callable(set_joint_impedance),
        "cartesian_impedance_configured": callable(set_cartesian_impedance),
    }
