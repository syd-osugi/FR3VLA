"""
Test 22: ChArUco helper behavior with fake OpenCV objects.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from _working_test_utils import require, require_close, require_raises, run_tests

from camera_calibration import charuco_utils


class FakeBoard:
    def __init__(self):
        self._corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )

    def getChessboardCorners(self):
        return self._corners


class FakeBoardConfig(dict):
    def setLegacyPattern(self, value):
        self["legacy_pattern"] = bool(value)


class FakeAruco:
    DICT_4X4_50 = 10
    DICT_4X4_100 = 11
    DICT_4X4_250 = 12
    DICT_5X5_50 = 20
    DICT_5X5_100 = 21
    DICT_6X6_50 = 30
    DICT_6X6_100 = 31

    def __init__(self, ids=None):
        self.ids = ids
        self.detected_corners = ["marker-corners"]
        self.rejected = ["rejected"]
        self.drawn_markers = False
        self.drawn_charuco = False

    def getPredefinedDictionary(self, dict_id):
        return {"dict_id": dict_id}

    def CharucoBoard_create(self, squares_x, squares_y, square_size, marker_size, aruco_dict):
        return FakeBoardConfig(
            {
                "api": "old",
                "squares_x": squares_x,
                "squares_y": squares_y,
                "square_size": square_size,
                "marker_size": marker_size,
                "dict": aruco_dict,
            }
        )

    def CharucoBoard(self, size, square_size, marker_size, aruco_dict):
        return FakeBoardConfig(
            {
                "api": "new",
                "size": size,
                "square_size": square_size,
                "marker_size": marker_size,
                "dict": aruco_dict,
            }
        )

    def detectMarkers(self, gray_image, aruco_dict):
        return self.detected_corners, self.ids, self.rejected

    def refineDetectedMarkers(self, gray_image, board, corners, ids, rejected):
        return corners, ids, rejected, "extra-value"

    def interpolateCornersCharuco(self, corners, ids, gray_image, board, *args):
        charuco_ids = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32)
        charuco_corners = np.array(
            [
                [[10.0, 10.0]],
                [[20.0, 10.0]],
                [[10.0, 20.0]],
                [[20.0, 20.0]],
                [[30.0, 20.0]],
                [[30.0, 30.0]],
            ],
            dtype=np.float32,
        )
        return len(charuco_ids), charuco_corners, charuco_ids

    def drawDetectedMarkers(self, image, marker_corners, marker_ids):
        self.drawn_markers = True

    def drawDetectedCornersCharuco(self, image, charuco_corners, charuco_ids, color):
        self.drawn_charuco = True


class FakeArucoNewOnly:
    def CharucoBoard(self, size, square_size, marker_size, aruco_dict):
        return FakeBoardConfig(
            {
                "api": "new",
                "size": size,
                "square_size": square_size,
                "marker_size": marker_size,
                "dict": aruco_dict,
            }
        )


class FakeCv2:
    def __init__(self, aruco):
        self.aruco = aruco
        self.axes_drawn = False

    def solvePnP(self, object_points, image_points, camera_matrix, dist_coeffs):
        return True, np.zeros((3, 1), dtype=float), np.array([[1.0], [2.0], [3.0]], dtype=float)

    def Rodrigues(self, value):
        array = np.array(value)
        if array.shape == (3, 1):
            return np.eye(3), None
        return np.zeros((3, 1)), None

    def drawFrameAxes(self, image, camera_matrix, dist_coeffs, rvec, tvec, axis_length):
        self.axes_drawn = True


def test_support_checks_and_dictionary_selection():
    require_raises(RuntimeError, lambda: charuco_utils.require_charuco_support(None), "None cv2 should fail")
    require_raises(
        RuntimeError,
        lambda: charuco_utils.require_charuco_support(SimpleNamespace()),
        "cv2 without aruco should fail",
    )
    require_raises(
        RuntimeError,
        lambda: charuco_utils.require_charuco_support(SimpleNamespace(aruco=SimpleNamespace())),
        "aruco without ChArUco helpers should fail",
    )

    aruco = FakeAruco()
    cv2 = FakeCv2(aruco)
    charuco_utils.require_charuco_support(cv2)
    require(charuco_utils.get_aruco_dictionary(cv2, "DICT_5X5_100")["dict_id"] == aruco.DICT_5X5_100, "known dict lookup failed")
    require(charuco_utils.get_aruco_dictionary(cv2, "UNKNOWN")["dict_id"] == aruco.DICT_4X4_50, "unknown dict should default")


def test_board_creation_and_object_points():
    aruco = FakeAruco()
    cv2 = FakeCv2(aruco)
    board = charuco_utils.create_charuco_board(cv2, (4, 5), 0.035, 0.025, {"dict_id": 10})
    require(board["api"] == "old" and board["squares_x"] == 5 and board["squares_y"] == 6, "old board API dimensions failed")

    cv2_new_only = FakeCv2(FakeArucoNewOnly())
    board = charuco_utils.create_charuco_board(cv2_new_only, (4, 5), 0.035, 0.025, {"dict_id": 10})
    require(board["api"] == "new" and board["size"] == (5, 6), "new board API dimensions failed")

    board = charuco_utils.create_charuco_board(
        cv2,
        (4, 5),
        0.035,
        0.025,
        {"dict_id": 10},
        legacy_pattern=True,
    )
    require(board["legacy_pattern"] is True, "legacy ChArUco pattern was not applied")

    object_points = charuco_utils.get_charuco_object_points(FakeBoard(), np.array([[2], [0]], dtype=np.int32))
    require_close(object_points, [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], "object point lookup failed")


def test_detection_short_circuit_and_success_path():
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros(5)
    board = FakeBoard()

    no_markers_cv2 = FakeCv2(FakeAruco(ids=None))
    result = charuco_utils.detect_charuco_board_pose(
        no_markers_cv2,
        gray_image="gray",
        aruco_dict={"dict_id": 10},
        charuco_board=board,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    require(result["success"] is False and result["T_board_to_cam"] is None, "missing markers should fail cleanly")
    require(result["marker_count"] == 0 and result["charuco_count"] == 0, "missing markers should report zero counts")

    partial_ids = np.array([[1], [2]], dtype=np.int32)
    partial_cv2 = FakeCv2(FakeAruco(ids=partial_ids))
    result = charuco_utils.detect_charuco_corners(
        partial_cv2,
        gray_image="gray",
        aruco_dict={"dict_id": 10},
        charuco_board=board,
        detector_params=object(),
    )
    require(result["success"] is False, "partial marker detection should not be capturable")
    require(result["marker_count"] == 2 and result["charuco_count"] == 0, "partial detection counts were not reported")

    ids = np.array([[1], [2], [3], [4]], dtype=np.int32)
    aruco = FakeAruco(ids=ids)
    cv2 = FakeCv2(aruco)
    corner_result = charuco_utils.detect_charuco_corners(
        cv2,
        gray_image="gray",
        aruco_dict={"dict_id": 10},
        charuco_board=board,
        detector_params=object(),
    )
    require(corner_result["success"] is True, f"expected successful fake corner detection, got {corner_result}")
    require(corner_result["marker_count"] == 4 and corner_result["charuco_count"] == 6, "successful detection counts failed")

    result = charuco_utils.detect_charuco_board_pose(
        cv2,
        gray_image="gray",
        aruco_dict={"dict_id": 10},
        charuco_board=board,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    require(result["success"] is True, f"expected successful fake detection, got {result}")
    require_close(result["T_board_to_cam"][:3, 3], [1.0, 2.0, 3.0], "solvePnP translation not copied")

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    charuco_utils.draw_charuco_detection(cv2, image, result)
    require(aruco.drawn_markers is True and aruco.drawn_charuco is True, "draw helpers were not called")
    charuco_utils.draw_pose_axes(cv2, image, result["T_board_to_cam"], camera_matrix, dist_coeffs, 0.1)
    require(cv2.axes_drawn is True, "pose axes helper was not called")


if __name__ == "__main__":
    raise SystemExit(
        run_tests(
            [
                ("support checks and dictionary selection", test_support_checks_and_dictionary_selection),
                ("board creation and object points", test_board_creation_and_object_points),
                ("detection short circuit and success path", test_detection_short_circuit_and_success_path),
            ]
        )
    )
