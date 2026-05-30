"""
Synchronized Camera Frame Helpers
---------------------------------
The LLM may ask for either camera by name, but every capture should come from a
timestamp-matched D435/D405 pair. Keeping that logic here prevents the image and
depth tools from drifting apart.
"""

from dataclasses import dataclass

import config as cfg


@dataclass(frozen=True)
class CameraFrames:
    """RGB, depth image, and RealSense depth frame for one camera."""

    rgb: object
    depth_array: object
    depth_rs: object


@dataclass(frozen=True)
class SyncedFrames:
    """One synchronized capture from both RealSense cameras."""

    d435: CameraFrames
    d405: CameraFrames


def capture_synced_frames(d435_cam, d405_cam):
    """
    Captures a synchronized D435/D405 snapshot.

    This function is called every time an image or depth-localization tool runs.
    That means a new tool call after robot motion gets a new camera state; the
    system does not reuse old images unless the caller reuses old pixel coords
    without asking for another image.

    Synchronization matters because D405 is mounted on the robot wrist. If D435
    and D405 images are captured at different times while the robot or object is
    moving, their robot-frame 3D estimates may describe different physical
    moments and fusion can become wrong.

    Returns:
        tuple: (SyncedFrames or None, error_message or None)
    """
    synced = d435_cam.grab_synced_snapshot(
        d405_cam,
        max_delta_ms=cfg.CAMERA_SYNC_TOLERANCE_MS,
    )
    if synced is None:
        return (
            None,
            f"Cameras failed to synchronize within {cfg.CAMERA_SYNC_TOLERANCE_MS}ms.",
        )

    return (
        SyncedFrames(
            d435=CameraFrames(synced[0], synced[1], synced[2]),
            d405=CameraFrames(synced[3], synced[4], synced[5]),
        ),
        None,
    )


def frames_for_camera(synced_frames, camera_name):
    """Selects the D435 or D405 frame bundle from a synchronized capture."""
    if camera_name == "d435":
        return synced_frames.d435
    if camera_name == "d405":
        return synced_frames.d405
    raise ValueError(f"Unknown camera: {camera_name}")
