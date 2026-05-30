# Tool JSON definitions (static data for LLM)

"""
Tool Schemas
------------
Defines the JSON schemas that describe available tools to the LLM.

WHY A SEPARATE FILE?
====================
Tool schemas are PURE DATA - they contain no logic. Separating them makes it
easy to:
1. Review what tools are available at a glance
2. Add/remove tools without touching handler code
3. Share schemas between different LLM interfaces

HOW THE LLM SEES TOOLS:
=======================
The LLM receives these schemas and uses them to decide:
1. WHAT tool to call (based on descriptions)
2. WHAT parameters to pass (based on parameter schemas)
3. WHEN to call (based on the task requirements)

SCHEMA FORMAT:
==============
Follows OpenAI's function calling format:
{
    "type": "function",
    "function": {
        "name": "tool_name",           # How the LLM calls it
        "description": "...",           # How the LLM decides when to use it
        "parameters": {                 # What arguments the LLM must provide
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
}
"""

import config as cfg


def _get_resolution_string(camera_name):
    """
    Helper to get resolution string for tool descriptions.
    
    We include the actual resolution in the description so the LLM knows
    the valid range for pixel coordinates. If we say "640x480" but the
    image is actually 1280x720, the LLM will guess wrong coordinates.
    
    Args:
        camera_name: "d435" or "d405"
        
    Returns:
        str: Resolution like "640x480"
    """
    if camera_name == "d435":
        return f"{cfg.D435_RESOLUTION[0]}x{cfg.D435_RESOLUTION[1]}"
    else:
        return f"{cfg.D405_RESOLUTION[0]}x{cfg.D405_RESOLUTION[1]}"


def _pixel_list_schema(description):
    return {
        "type": "array",
        "description": description,
        "items": {
            "type": "array",
            "description": "A single [u, v] pixel coordinate",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
    }


def _nullable_pixel_list_schema(description):
    schema = _pixel_list_schema(description)
    return {
        "description": description,
        "anyOf": [
            schema,
            {"type": "null"},
        ],
    }


# =============================================================================
# GET RESOLUTION STRINGS FOR DESCRIPTIONS
# =============================================================================
# Pre-compute these so they're only calculated once when the module loads
D435_RES_STRING = _get_resolution_string("d435")
D405_RES_STRING = _get_resolution_string("d405")


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

tool_json_list = [
    # =========================================================================
    # TOOL 1: BIRD'S EYE VIEW (D435)
    # =========================================================================
    # Purpose: Let the LLM see the entire workspace from above
    # When to use: Initial object detection, spatial awareness
    # Returns: RGB image only (no 3D data)
    {
        "type": "function",
        "function": {
            "name": "get_birds_eye_view",
            "description": (
                f"Captures a synchronized RGB image from the overhead D435 camera "
                f"({D435_RES_STRING} resolution). "
                "Use this to see the entire workspace from above and locate objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {},  # No parameters needed
                "required": []
            },
        },
    },

    # =========================================================================
    # TOOL 2: EYE-IN-HAND VIEW (D405)
    # =========================================================================
    # Purpose: Let the LLM see close-up details from the wrist camera
    # When to use: Detailed inspection, when overhead view is obstructed
    # Returns: RGB image only (no 3D data)
    {
        "type": "function",
        "function": {
            "name": "get_eye_in_hand_view",
            "description": (
                f"Captures a synchronized RGB image from the D405 camera mounted on the robot wrist "
                f"({D405_RES_STRING} resolution). "
                "Use this for close-up inspection when the overhead view is insufficient."
            ),
            "parameters": {
                "type": "object",
                "properties": {},  # No parameters needed
                "required": []
            },
        },
    },

    # =========================================================================
    # TOOL 3: SINGLE-CAMERA 3D COORDINATES (D435)
    # =========================================================================
    # Purpose: Get 3D position from overhead camera only
    # When to use: Object only visible in D435, or quick single-camera estimate
    # Returns: 3D XYZ coordinates in robot base frame
    {
        "type": "function",
        "function": {
            "name": "get_xyz_d435",
            "description": (
                f"Converts pixel [u, v] coordinates from a {D435_RES_STRING} D435 image "
                "into 3D XYZ coordinates in meters (robot base frame). "
                "If a point returns 'invalid', DO NOT retry the same pixel - the depth sensor "
                "cannot see that spot (transparent, shiny, or missing object)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": _pixel_list_schema(
                        (
                            f"List of [u, v] pixel coordinates from the D435 image. "
                            f"Image is {D435_RES_STRING}, top-left is [0, 0]."
                        )
                    ),
                },
                "required": ["coords"],
            },
        },
    },

    # =========================================================================
    # TOOL 4: SINGLE-CAMERA 3D COORDINATES (D405)
    # =========================================================================
    # Purpose: Get 3D position from wrist camera only
    # When to use: Object only visible in D405, close-range precision
    # Returns: 3D XYZ coordinates in robot base frame
    {
        "type": "function",
        "function": {
            "name": "get_xyz_d405",
            "description": (
                f"Converts pixel [u, v] coordinates from a {D405_RES_STRING} D405 image "
                "into 3D XYZ coordinates in meters (robot base frame). "
                "If a point returns 'invalid', DO NOT retry the same pixel - the depth sensor "
                "cannot see that spot (transparent, shiny, or missing object)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coords": _pixel_list_schema(
                        (
                            f"List of [u, v] pixel coordinates from the D405 image. "
                            f"Image is {D405_RES_STRING}, top-left is [0, 0]."
                        )
                    ),
                },
                "required": ["coords"],
            },
        },
    },

    # =========================================================================
    # TOOL 5: FUSED MULTI-CAMERA 3D COORDINATES
    # =========================================================================
    # Purpose: Get the BEST 3D estimate using both cameras
    # When to use: Object visible in both cameras, maximum accuracy needed
    # Returns: Fused 3D XYZ with confidence metrics
    #
    # WHY FUSION IS BETTER:
    # - D435 (top-down): Good XY accuracy, limited Z accuracy at distance
    # - D405 (close-up): Good Z accuracy, limited field of view
    # - Fused: Combines strengths, averages out noise
    {
        "type": "function",
        "function": {
            "name": "get_xyz_fused",
            "description": (
                "BEST ACCURACY: Uses BOTH cameras to calculate the 3D position of an object. "
                "Provide pixel coordinates [u, v] from whichever cameras can see the object. "
                "If the object is visible in BOTH cameras, provide coords from both for "
                "improved accuracy through sensor fusion. "
                "If only visible in one camera, provide that camera's coords and set the "
                "other to null. Returns position in robot base frame (meters). This tool "
                "captures a fresh synchronized D435/D405 frame pair every time it runs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "d435_coords": _nullable_pixel_list_schema(
                        (
                            f"Pixel coordinates [u, v] from D435 image ({D435_RES_STRING}). "
                            "Set to null if object not visible in D435."
                        )
                    ),
                    "d405_coords": _nullable_pixel_list_schema(
                        (
                            f"Pixel coordinates [u, v] from D405 image ({D405_RES_STRING}). "
                            "Set to null if object not visible in D405."
                        )
                    ),
                },
                "required": [],  # Neither is strictly required (object might be in only one)
            },
        },
    },

    # =========================================================================
    # TOOL 6: ROBOT TRAJECTORY PLAN
    # =========================================================================
    # Purpose: Convert a localized object position into robot-base waypoints.
    {
        "type": "function",
        "function": {
            "name": "plan_robot_trajectory",
            "description": (
                "Creates robot-base EE-origin waypoints that place the configured gripper/TCP "
                "at a target XYZ position returned by get_xyz_fused or a single-camera XYZ "
                "tool. This plans only; it does not execute robot motion. Re-capture synced "
                "images and re-plan after any movement; this tool does not auto-update while "
                "the robot is moving."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_xyz": {
                        "type": "array",
                        "description": (
                            "Desired gripper/TCP target [x, y, z] in robot base frame, meters."
                        ),
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    "approach_direction": {
                        "type": "string",
                        "description": "Axis used for the pre-approach waypoint.",
                        "enum": ["x", "y", "z"],
                    },
                    "approach_height_m": {
                        "type": "number",
                        "description": "Optional approach offset in meters.",
                        "exclusiveMinimum": 0,
                    },
                },
                "required": ["target_xyz"],
            },
        },
    },
]
