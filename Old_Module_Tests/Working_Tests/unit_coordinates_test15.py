"""
Test 15: Pixel-to-XYZ math with fake RealSense frames.

No camera is opened. A fake pyrealsense2 module and fake depth frame isolate the
coordinate conversion behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

from _working_test_utils import require, require_close, run_tests

from utilities.coordinates import pixel_to_xyz


@dataclass
class FakeIntrinsics:
    width: int = 640
    height: int = 480
    fx: float = 600.0
    fy: float = 600.0
    ppx: float = 320.0
    ppy: float = 240.0
    coeffs: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)


class FakeProfile:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def as_video_stream_profile(self):
        return self


class FakeDepthFrame:
    def __init__(self, distances, intrinsics=None):
        self.distances = distances
        self.profile = FakeProfile(intrinsics or FakeIntrinsics())

    def get_distance(self, u, v):
        intrinsics = self.profile.intrinsics
        if not (0 <= u < intrinsics.width and 0 <= v < intrinsics.height):
            raise RuntimeError("pixel outside fake frame")
        return self.distances.get((u, v), 0.0)


class FakeRealSenseModule:
    @staticmethod
    def rs2_deproject_pixel_to_point(intrinsics, pixel, depth_m):
        u, v = pixel
        return [
            (u - intrinsics.ppx) / intrinsics.fx * depth_m,
            (v - intrinsics.ppy) / intrinsics.fy * depth_m,
            depth_m,
        ]


def with_fake_rs(test_fn):
    old_module = sys.modules.get("pyrealsense2")
    sys.modules["pyrealsense2"] = FakeRealSenseModule
    try:
        test_fn()
    finally:
        if old_module is None:
            sys.modules.pop("pyrealsense2", None)
        else:
            sys.modules["pyrealsense2"] = old_module


def test_none_depth_frame_is_invalid():
    result = pixel_to_xyz(0, 0, None, 0.001)
    require(result["valid"] is False, "None depth frame should be invalid")


def test_valid_center_pixel_deprojects():
    def body():
        frame = FakeDepthFrame({(320, 240): 1.25})
        result = pixel_to_xyz(320, 240, frame, 0.001)
        require(result["valid"] is True, "center pixel should be valid")
        require_close([result["x"], result["y"], result["z"]], [0.0, 0.0, 1.25], "center deprojection failed")

    with_fake_rs(body)


def test_valid_offset_pixel_deprojects():
    def body():
        frame = FakeDepthFrame({(620, 240): 2.0})
        result = pixel_to_xyz(620, 240, frame, 0.001)
        require(result["valid"] is True, "offset pixel should be valid")
        require_close([result["x"], result["y"], result["z"]], [1.0, 0.0, 2.0], "offset deprojection failed")

    with_fake_rs(body)


def test_zero_depth_and_bad_pixels_are_invalid():
    def body():
        frame = FakeDepthFrame({(320, 240): 0.0})
        require(pixel_to_xyz(320, 240, frame, 0.001)["valid"] is False, "zero depth should be invalid")
        require(pixel_to_xyz("not-int", 240, frame, 0.001)["valid"] is False, "bad u should be invalid")
        require(pixel_to_xyz(9999, 240, frame, 0.001)["valid"] is False, "out-of-bounds should be invalid")

    with_fake_rs(body)


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("none depth frame is invalid", test_none_depth_frame_is_invalid),
                ("valid center pixel deprojects", test_valid_center_pixel_deprojects),
                ("valid offset pixel deprojects", test_valid_offset_pixel_deprojects),
                ("zero depth and bad pixels are invalid", test_zero_depth_and_bad_pixels_are_invalid),
            ]
        )
    )
