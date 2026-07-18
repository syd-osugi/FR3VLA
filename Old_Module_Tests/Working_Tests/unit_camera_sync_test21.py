"""
Test 21: Synchronized camera frame helper behavior with fake cameras.
"""

from __future__ import annotations

from _working_test_utils import require, require_raises, run_tests

from vision.tools import camera_frames


class FakeCamera:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.requested_delta_ms = None

    def grab_synced_snapshot(self, other_camera, max_delta_ms=None):
        self.requested_delta_ms = max_delta_ms
        return self.snapshot


def test_capture_synced_frames_success():
    snapshot = ("d435-rgb", "d435-depth", "d435-rs", "d405-rgb", "d405-depth", "d405-rs")
    d435 = FakeCamera(snapshot)
    d405 = FakeCamera(None)

    synced, error = camera_frames.capture_synced_frames(d435, d405)
    require(error is None, f"unexpected sync error: {error}")
    require(synced.d435.rgb == "d435-rgb", "D435 rgb was unpacked incorrectly")
    require(synced.d435.depth_array == "d435-depth", "D435 depth array was unpacked incorrectly")
    require(synced.d435.depth_rs == "d435-rs", "D435 RealSense frame was unpacked incorrectly")
    require(synced.d405.rgb == "d405-rgb", "D405 rgb was unpacked incorrectly")
    require(d435.requested_delta_ms == camera_frames.cfg.CAMERA_SYNC_TOLERANCE_MS, "sync tolerance was not forwarded")

    require(camera_frames.frames_for_camera(synced, "d435") is synced.d435, "D435 selector failed")
    require(camera_frames.frames_for_camera(synced, "d405") is synced.d405, "D405 selector failed")


def test_capture_synced_frames_failure_and_unknown_camera():
    d435 = FakeCamera(None)
    d405 = FakeCamera(None)
    synced, error = camera_frames.capture_synced_frames(d435, d405)
    require(synced is None and "failed to synchronize" in error, "sync failure should return error")

    good_synced = camera_frames.SyncedFrames(
        d435=camera_frames.CameraFrames("a", "b", "c"),
        d405=camera_frames.CameraFrames("d", "e", "f"),
    )
    require_raises(
        ValueError,
        lambda: camera_frames.frames_for_camera(good_synced, "bad-camera"),
        "unknown camera name should raise",
    )


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("capture synced frames success", test_capture_synced_frames_success),
                ("capture synced frames failure and unknown camera", test_capture_synced_frames_failure_and_unknown_camera),
            ]
        )
    )
