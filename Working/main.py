"""
Main Runtime Loop
-----------------
Starts the two RealSense cameras, starts the LLM tool-calling interface, and
keeps the terminal interaction alive.

At runtime this file does not perform calibration. It assumes calibration JSON
files already exist, then lets the LLM request synchronized camera images,
localize objects, and plan robot-base waypoints through the tool system.
"""

import numpy as np

import config as cfg  # Import our centralized configuration values

from hardware.camera import RealSense
from robot.robot_interface import FrankaRobotInterface
from vision.camera_viewer import CameraViewer
from vision.llm_interface import LLMinterface
import vision.tools as tools

def _configured_serial(value):
    """
    Safety check for serial numbers.
    If the user hasn't edited config.py and the value is still "YOUR_D435_SERIAL_HERE",
    we return None so the camera system can warn the user, rather than trying to find 
    a camera that literally has the serial number "YOUR_D435_SERIAL_HERE".
    """
    if not value or value.startswith("YOUR_"):
        return None
    return value

def _stop_camera(camera, name):
    """
    Safely shuts down a camera. 
    We wrap it in a try/except because if the camera was never initialized (e.g., it 
    was unplugged), calling .stop() would crash the script on exit.
    """
    if camera is None:
        return
    try:
        camera.stop()
    except Exception as exc:
        print(f"Warning: failed to stop {name}: {exc}")

# def _release_robot(robot):
#     """
#     Releases a Franka robot connection when the Python binding exposes an
#     explicit cleanup hook.

#     Some pylibfranka builds rely on Python object destruction instead of a
#     close() method, so cleanup must be best-effort.
#     """
#     close = getattr(robot, "close", None)
#     if callable(close):
#         close()


# def _close_robot_interface(robot_interface):
#     """Close the persistent RobotInterface used by main.py."""
#     if robot_interface is None:
#         return
#     try:
#         robot_interface.close()
#     except Exception as exc:
#         print(f"Warning: failed to close robot interface: {exc}")


def _stop_camera_viewer(camera_viewer):
    """Close the optional live OpenCV camera viewer."""
    if camera_viewer is None:
        return
    try:
        camera_viewer.stop()
    except Exception as exc:
        print(f"Warning: failed to stop camera viewer: {exc}")


# def get_robot_ee_pose(robot_interface=None):
#     """
#     Gets the current robot end-effector pose.

#     If main.py has a persistent FrankaRobotInterface, read from that connection.
#     The fallback path opens a short-lived pylibfranka Robot only for older call
#     sites or debugging utilities that call this helper directly.
    
#     Returns:
#         numpy.ndarray: 4x4 transform matrix, or None if robot not available
#     """
#     if robot_interface is not None:
#         try:
#             state = robot_interface.read_state()
#             if state.ee_pose is None:
#                 print("Warning: robot interface state did not include ee_pose")
#                 return None
#             return np.array(state.ee_pose, dtype=float)
#         except Exception as e:
#             print(f"Warning: could not read robot end-effector pose: {e}")
#             return None

#     robot = None
#     try:
#         from pylibfranka import Robot
#         robot = Robot(cfg.FRANKA_IP)
#         state = robot.read_once()
#         T_ee = np.array(state.O_T_EE).reshape((4, 4), order='F')
#         return T_ee
#     except Exception as e:
#         print(f"Warning: could not read robot end-effector pose: {e}")
#         return None
#     finally:
#         if robot is not None:
#             try:
#                 _release_robot(robot)
#             except Exception as exc:
#                 print(f"Warning: failed to release robot connection: {exc}")

def main():
    # Initialize these as None so the finally block can safely check them if we crash early
    d435 = None
    d405 = None
    camera_viewer = None
    robot_interface = None

    try:
        # 1. Initialize Hardware using values from config.py
        print("Starting Cameras...")
        d435_serial = _configured_serial(cfg.D435_SERIAL)
        d405_serial = _configured_serial(cfg.D405_SERIAL)

        # Warn the user if they forgot to set their serial numbers
        if d435_serial is None or d405_serial is None:
            print("Warning: Camera serial numbers are not fully configured. Check config.py.")

        # Initialize the RealSense objects. We pass the specific resolutions and FPS 
        # defined in config.py so they are never hardcoded here.
        d435 = RealSense(serial_number=d435_serial, resolution=cfg.D435_RESOLUTION, fps=cfg.CAMERA_FPS)
        d405 = RealSense(serial_number=d405_serial, resolution=cfg.D405_RESOLUTION, fps=cfg.CAMERA_FPS)

        camera_viewer = CameraViewer(d435, d405)
        camera_viewer.start()

        # print("Connecting to Franka robot...")
        # robot_interface = FrankaRobotInterface()
        # robot_connection = robot_interface.connect()
        # print(
        #     "Robot connected. "
        #     f"Motion confirmation required: {robot_connection.get('require_motion_confirmation')}"
        # )

        # 2. Initialize LLM
        # We pass the API URL and Key from config.py, along with the tool definitions
        llm = LLMinterface(
            model=cfg.QWEN_MODEL_PATH,
            tools_json=tools.tool_json_list,
            api_url=cfg.LLM_API_URL,
            api_key=cfg.LLM_API_KEY
        )

        # 3. The main interactive loop
        while True:
            # Ask the user what they want the robot to do
            command = llm.get_text()
            if camera_viewer is not None:
                camera_viewer.set_prompt(command)

            # The pose provider is called for each tool use so D405 transforms stay
            # current if the robot moves during a request.
            llm.send_message_with_tools(
                d435,
                d405,
                # robot_pose_provider=lambda: get_robot_ee_pose(robot_interface),
                # robot_interface=robot_interface,
                tool_result_callback=(
                    camera_viewer.handle_tool_result
                    if camera_viewer is not None
                    else None
                ),
            )

            # Print the final answer the LLM came up with
            llm.print_message()
            if camera_viewer is not None:
                camera_viewer.set_status("Ready for next command.")

            # CRITICAL: Remove the giant base64 images from the LLM's memory before 
            # the next loop, otherwise the API will crash from token limits.
            llm.prune_image_history()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # This block runs no matter what (even if the code crashes).
        # It ensures the USB cameras are properly released so they don't freeze up.
        print("Shutting down hardware safely...")
        _stop_camera_viewer(camera_viewer)
        # _close_robot_interface(robot_interface)
        _stop_camera(d435, "D435")
        _stop_camera(d405, "D405")

if __name__ == "__main__":
    main()
