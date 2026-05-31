"""
Configuration File
------------------
Single source of truth for all hardware serials, resolutions, and LLM API settings.

Override values using environment variables:
    export D435_SERIAL="123456789"
    export D435_RESOLUTION="640,480"
"""

import os


def _parse_tuple(value, default):
    """
    Safely parse environment variables into tuples.
    
    Environment variables are strings like "640,480". This converts them to 
    Python tuples (640, 480) that OpenCV and cameras require.
    
    Args:
        value: The raw environment variable string (or None if not set).
        default: The fallback tuple to use if parsing fails.
        
    Returns:
        tuple: (width, height) integers.
    """
    if value is None:
        return default
    if not isinstance(value, str):
        return value

    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError:
        return default

    # Validate: must be exactly 2 positive integers
    if len(parts) != 2 or parts[0] <= 0 or parts[1] <= 0:
        return default
    return parts


def _parse_int(value, default, min_value=None, max_value=None):
    """
    Safely parse environment variable to integer with optional bounds clamping.
    
    Args:
        value: The raw environment variable string (or None).
        default: Fallback value if parsing fails.
        min_value: If set, output will never be lower than this.
        max_value: If set, output will never be higher than this.
        
    Returns:
        int: The parsed and clamped integer.
    """
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except ValueError:
            parsed = default

    # Clamp to bounds if specified
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _parse_float(value, default, min_value=None, max_value=None):
    """
    Safely parse environment variable to float with optional bounds clamping.
    
    Args:
        value: The raw environment variable string (or None).
        default: Fallback value if parsing fails.
        min_value: If set, output will never be lower than this.
        max_value: If set, output will never be higher than this.
        
    Returns:
        float: The parsed and clamped float.
    """
    if value is None:
        parsed = default
    else:
        try:
            parsed = float(value)
        except ValueError:
            parsed = default

    # Clamp to bounds if specified
    if min_value is not None:
        parsed = max(min_value, parsed)
    if max_value is not None:
        parsed = min(max_value, parsed)
    return parsed


def _parse_float_tuple(value, default, expected_len=None):
    """
    Safely parse a comma-separated environment variable into a tuple of floats.

    Use this for physical offsets/orientations where negative and decimal values
    are valid, for example: "0.01,-0.02,0.15".
    """
    if value is None:
        return default
    if not isinstance(value, str):
        return value

    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError:
        return default

    if expected_len is not None and len(parts) != expected_len:
        return default
    return parts


def _parse_bool(value, default):
    """Safely parse common environment variable strings into booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


# =============================================================================
# CAMERA HARDWARE SETTINGS
# =============================================================================
# Serial numbers uniquely identify each physical camera.
# If you have two D435s plugged in, Linux sees them as identical USB devices
# unless you specify the serial number to force the correct one open.

# to debug 
# conda run -n Sydney python -c "import pyrealsense2 as rs; ctx=rs.context(); [print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.usb_type_descriptor)) for d in ctx.query_devices()]"

D435_SERIAL = os.getenv("D435_SERIAL", "044322073387")
D405_SERIAL = os.getenv("D405_SERIAL", "130322273025")

# Image resolution in pixels (width, height).
# Higher resolution = more detail but slower processing and more RAM usage.
# D435 can do 640x480, 1280x720, or 1920x1080.
# D405 is designed for close-up work; 1280x720 is its sweet spot.
D435_RESOLUTION = _parse_tuple(os.getenv("D435_RESOLUTION"), (640, 480))
D405_RESOLUTION = _parse_tuple(os.getenv("D405_RESOLUTION"), (1280, 720))

# Frame rate in frames per second.
# 30fps is standard. Lower (15fps) saves USB bandwidth; higher (60fps) reduces motion blur.
CAMERA_FPS = _parse_int(os.getenv("CAMERA_FPS"), 30, min_value=1, max_value=90)

# Timeout for waiting on camera frames (milliseconds).
# If your USB 3.0 ports are slow and dropping frames, increase this to 2000-3000ms.
CAMERA_FRAME_TIMEOUT_MS = _parse_int(os.getenv("CAMERA_FRAME_TIMEOUT_MS"), 1000, min_value=100)

# Maximum allowed time difference for synchronized frame capture (milliseconds).
# At 30fps, frames arrive every ~33ms. 20ms tolerance ensures true synchronization.
CAMERA_SYNC_TOLERANCE_MS = _parse_int(os.getenv("CAMERA_SYNC_TOLERANCE_MS"), 20, min_value=1, max_value=50)

# Number of attempts made while waiting for a pair of synchronized camera frames.
CAMERA_SYNC_MAX_TRIES = _parse_int(os.getenv("CAMERA_SYNC_MAX_TRIES"), 10, min_value=1, max_value=100)

# Short sleep between sync attempts, in seconds.
CAMERA_SYNC_RETRY_SLEEP_S = _parse_float(
    os.getenv("CAMERA_SYNC_RETRY_SLEEP_S"),
    0.01,
    min_value=0.0,
    max_value=1.0,
)

# Warmup time for auto-exposure and laser settings to settle after a camera starts.
CAMERA_WARMUP_SECONDS = _parse_float(
    os.getenv("CAMERA_WARMUP_SECONDS"),
    2.0,
    min_value=0.0,
    max_value=30.0,
)

# Longer timeout used by interactive calibration scripts when the operator is waiting.
CALIBRATION_FRAME_TIMEOUT_MS = _parse_int(
    os.getenv("CALIBRATION_FRAME_TIMEOUT_MS"),
    5000,
    min_value=100,
)


# =============================================================================
# LLM / VISION MODEL SETTINGS
# =============================================================================
# Path to the local Qwen model file (.gguf format for llama.cpp)
QWEN_MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "models/Qwen3.5-4B-Q4_K_M.gguf")

# Settings for the local LLM server (llama.cpp, LM Studio, Ollama, etc.)
LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key-required")

# Temperature controls LLM "creativity":
#   0.0 = Deterministic, mathematical (best for exact pixel guessing)
#   1.0 = Highly creative, unpredictable
LLM_TEMPERATURE = _parse_float(os.getenv("LLM_TEMPERATURE"), 0.1, min_value=0.0, max_value=2.0)

# Maximum tokens the model can generate per response.
# Prevents runaway responses that freeze the system.
LLM_MAX_OUTPUT_TOKENS = _parse_int(os.getenv("LLM_MAX_OUTPUT_TOKENS"), 1024, min_value=1)

# Safety limit: prevents infinite tool-calling loops.
# If the LLM keeps calling tools without giving a final answer, we stop after this many rounds.
LLM_MAX_TOOL_ROUNDS = _parse_int(os.getenv("LLM_MAX_TOOL_ROUNDS"), 8, min_value=1, max_value=20)

# JPEG quality for images sent to the LLM (1-100).
# Base64-encoded images are HUGE strings (~500KB at quality 95, ~50KB at quality 30).
# Lower this if you have limited RAM or slow API responses.
# Raise this if the LLM struggles to see small objects.
LLM_IMAGE_JPEG_QUALITY = _parse_int(
    os.getenv("LLM_IMAGE_JPEG_QUALITY"),
    75,
    min_value=1,
    max_value=100,
)


# =============================================================================
# DEBUGGING / OUTPUT SETTINGS
# =============================================================================
# Directory for debug images (e.g., "last_ai_aim.jpg" showing where the LLM aimed).
DEBUG_IMAGE_DIR = os.getenv("DEBUG_IMAGE_DIR", "output_debug_images")


# =============================================================================
# CALIBRATION FILE PATHS
# =============================================================================
# All calibration data is stored as JSON files in this directory.
CALIBRATION_DIR = os.getenv("CALIBRATION_DIR", "calibration_data")

# Intrinsics: Lens distortion coefficients and focal length.
# These are camera-specific and don't change unless you physically swap the lens.
INTRINSICS_D435_PATH = os.path.join(CALIBRATION_DIR, "d435_intrinsics.json")
INTRINSICS_D405_PATH = os.path.join(CALIBRATION_DIR, "d405_intrinsics.json")

# Extrinsics: Where each camera is positioned relative to a reference frame.
# D405 extrinsic: Transform from D405 optical frame to robot end-effector (wrist) frame.
# This is the "hand-eye" calibration result.
HAND_EYE_D405_PATH = os.path.join(CALIBRATION_DIR, "d405_to_wrist.json")

# D435 extrinsic: Direct transform from D435 optical frame to robot base frame.
# This is saved by mounting a ChArUco board to the robot end-effector and
# capturing multiple robot poses with the fixed overhead D435.
BIRD_EYE_D435_PATH = os.path.join(CALIBRATION_DIR, "d435_to_robot_base.json")


# =============================================================================
# INTRINSIC CALIBRATION SETTINGS (Large ChArUco Board)
# =============================================================================
# The large ChArUco board is used for high-accuracy intrinsic calibration.
# It combines ArUco markers with a chessboard pattern for robust corner detection.

# Number of INNER chessboard corners (not squares, not markers).
# A 4x5 corner grid means 5x6 squares.
INTRINSIC_BOARD_CORNERS = _parse_tuple(os.getenv("INTRINSIC_BOARD_CORNERS"), (7, 10))

# Physical size of each chessboard square in meters.
INTRINSIC_SQUARE_SIZE = _parse_float(os.getenv("INTRINSIC_SQUARE_SIZE"), 0.020, min_value=0.001)

# Physical size of each ArUco marker in meters.
# Must be SMALLER than the square size so markers don't touch.
INTRINSIC_MARKER_SIZE = _parse_float(os.getenv("INTRINSIC_MARKER_SIZE"), 0.015, min_value=0.001)

# Number of images to capture for calibration.
# More images = better accuracy, but 15-20 is typically sufficient.
INTRINSIC_IMAGES_REQUIRED = _parse_int(os.getenv("INTRINSIC_IMAGES_REQUIRED"), 20, min_value=5, max_value=50)

# ArUco dictionary to use. DICT_4X4_50 has 50 unique markers with 4x4 bit patterns.
INTRINSIC_ARUCO_DICT_NAME = os.getenv("INTRINSIC_ARUCO_DICT_NAME", "DICT_4X4_50")


# =============================================================================
# D405 HAND-EYE EXTRINSIC CALIBRATION SETTINGS (ChArUco Board on Table)
# =============================================================================
# The D405 hand-eye calibration now uses a ChArUco board, matching the
# intrinsic calibration and D435 bird's-eye calibration target style.
#
# By default, use the same board geometry as intrinsic calibration. This keeps
# the setup simple: one printed ChArUco target type for all camera calibration.
# If you later use a physically different ChArUco board for hand-eye, add
# separate HAND_EYE_* env-backed values here.
HAND_EYE_BOARD_CORNERS = INTRINSIC_BOARD_CORNERS
HAND_EYE_SQUARE_SIZE = INTRINSIC_SQUARE_SIZE
HAND_EYE_MARKER_SIZE = INTRINSIC_MARKER_SIZE
HAND_EYE_ARUCO_DICT_NAME = INTRINSIC_ARUCO_DICT_NAME

# Number of robot poses to capture for robot-based extrinsic calibration.
# Used by D405 hand-eye and D435 mounted-board calibration.
# More poses = better accuracy, but each requires manually moving the robot.
# Minimum is typically 10-15; 20 is a safe choice.
HAND_EYE_POSES_REQUIRED = _parse_int(os.getenv("HAND_EYE_POSES_REQUIRED"), 20, min_value=5, max_value=50)


# =============================================================================
# BIRD'S EYE EXTRINSIC CALIBRATION SETTINGS (ChArUco Board on End-Effector)
# =============================================================================
# The D435 looks down at a ChArUco board mounted to the robot end-effector.
# Robot motion supplies the board's relationship to the robot base.

# Use the same board as intrinsics, but a different mounted board can be added
# here later if the physical calibration target changes.
BIRD_EYE_BOARD_CORNERS = INTRINSIC_BOARD_CORNERS
BIRD_EYE_SQUARE_SIZE = INTRINSIC_SQUARE_SIZE
BIRD_EYE_MARKER_SIZE = INTRINSIC_MARKER_SIZE
BIRD_EYE_ARUCO_DICT_NAME = INTRINSIC_ARUCO_DICT_NAME

# Rigid mount from ChArUco board frame to robot end-effector frame.
#
# Matrix name:
#     T_board_to_ee
#
# Matrix meaning:
#     p_ee = T_board_to_ee @ p_board
#
# This is a physical mounting measurement, not a camera calibration result.
# The D435 extrinsic script reads these values, builds T_board_to_ee, prints the
# matrix, and asks the operator to confirm it before any images are captured.
#
# If the board frame is intentionally identical to the end-effector frame, keep
# both tuples at all zeros. In most real mounts, the values will not be all zero
# because the ChArUco board origin is usually offset from the robot flange/tool
# frame origin.
#
# Translation is the board origin expressed in end-effector coordinates, meters.
# RPY is the board frame orientation relative to the end-effector frame, degrees.
# Update these values whenever the physical board mount changes.
BIRD_EYE_BOARD_TO_EE_TRANSLATION_M = _parse_float_tuple(
    os.getenv("BIRD_EYE_BOARD_TO_EE_TRANSLATION_M"),
    (0.0, 0.0, 0.0),
    expected_len=3,
)
BIRD_EYE_BOARD_TO_EE_RPY_DEG = _parse_float_tuple(
    os.getenv("BIRD_EYE_BOARD_TO_EE_RPY_DEG"),
    (0.0, 0.0, 0.0),
    expected_len=3,
)


# =============================================================================
# GRIPPER / TOOL CENTER POINT GEOMETRY
# =============================================================================
# The vision system localizes object positions in the robot base frame. To move
# the robot to an object, the motion target should usually be the gripper/tool
# point, not the raw Franka EE origin.
#
# Matrix name:
#     T_gripper_tcp_to_ee
#
# Matrix meaning:
#     p_ee = T_gripper_tcp_to_ee @ p_gripper_tcp
#
# The translation below is the gripper/TCP origin expressed in EE coordinates.
# In plain language: it is the offset from the Franka EE frame origin to the
# point on the gripper you want to place at the object, such as the midpoint
# between fingertips or another grasp/contact point.
#
# Current trajectory planning only outputs XYZ waypoints, so the translation is
# the part used now. The RPY is kept here so the same config can support full
# pose/orientation planning later.
GRIPPER_TCP_IN_EE_TRANSLATION_M = _parse_float_tuple(
    os.getenv("GRIPPER_TCP_IN_EE_TRANSLATION_M"),
    (0.0, 0.0, 0.0),
    expected_len=3,
)
GRIPPER_TCP_IN_EE_RPY_DEG = _parse_float_tuple(
    os.getenv("GRIPPER_TCP_IN_EE_RPY_DEG"),
    (0.0, 0.0, 0.0),
    expected_len=3,
)


# =============================================================================
# ROBOT NETWORK SETTINGS
# =============================================================================
# IP address of the Franka robot controller on your network.
FRANKA_IP = os.getenv("FRANKA_IP", "10.31.82.199")

# Conservative collision thresholds used while calibrating around the live robot.
FRANKA_COLLISION_TORQUE_NM = _parse_float(
    os.getenv("FRANKA_COLLISION_TORQUE_NM"),
    20.0,
    min_value=1.0,
)
FRANKA_COLLISION_FORCE_N = _parse_float(
    os.getenv("FRANKA_COLLISION_FORCE_N"),
    20.0,
    min_value=1.0,
)


# =============================================================================
# POLICY / FEEDBACK CONTROL SETTINGS
# =============================================================================
# These settings are used by the placeholder learning-policy scaffold. They are
# conservative defaults, not a guarantee of safety. Tighten them around your real
# workspace before enabling any hardware execution.

ROBOT_WORKSPACE_MIN_M = _parse_float_tuple(
    os.getenv("ROBOT_WORKSPACE_MIN_M"),
    (-0.8, -0.8, 0.0),
    expected_len=3,
)
ROBOT_WORKSPACE_MAX_M = _parse_float_tuple(
    os.getenv("ROBOT_WORKSPACE_MAX_M"),
    (0.8, 0.8, 0.8),
    expected_len=3,
)

ROBOT_MAX_CARTESIAN_SPEED_MPS = _parse_float(
    os.getenv("ROBOT_MAX_CARTESIAN_SPEED_MPS"),
    0.10,
    min_value=0.001,
)
ROBOT_MAX_CARTESIAN_ACCEL_MPS2 = _parse_float(
    os.getenv("ROBOT_MAX_CARTESIAN_ACCEL_MPS2"),
    0.25,
    min_value=0.001,
)

POLICY_CONTROL_RATE_HZ = _parse_float(
    os.getenv("POLICY_CONTROL_RATE_HZ"),
    10.0,
    min_value=1.0,
    max_value=100.0,
)
POLICY_MAX_STEP_TRANSLATION_M = _parse_float(
    os.getenv("POLICY_MAX_STEP_TRANSLATION_M"),
    0.02,
    min_value=0.0001,
)
POLICY_MAX_STEP_ROTATION_RAD = _parse_float(
    os.getenv("POLICY_MAX_STEP_ROTATION_RAD"),
    0.10,
    min_value=0.0001,
)
POLICY_REQUIRE_ROBOT_STATE = _parse_bool(
    os.getenv("POLICY_REQUIRE_ROBOT_STATE"),
    False,
)

POLICY_DATASET_ROOT = os.getenv("POLICY_DATASET_ROOT", "data/policy")
POLICY_CHECKPOINT_PATH = os.getenv("POLICY_CHECKPOINT_PATH", "checkpoints/policy/latest")
