"""
LLM Tool-Calling Interface
--------------------------
Wraps an OpenAI-compatible local LLM server and manages the message loop.

The model receives tool schemas, asks for synchronized camera images or 3D
localization tools as needed, and gets tool results fed back into the same
conversation until it can answer the user's instruction.
"""

import json
from textwrap import dedent
import config as cfg  # IMPORT CONFIG TO GET DYNAMIC RESOLUTION

from vision.base_classes import BaseLLM

import vision.tools as tools


FINAL_ANSWER_PROMPT = (
    "The tool-call safety limit has been reached. Stop requesting tools now. "
    "Use only the tool results already in the conversation and give the operator "
    "a concise final answer. If the available evidence is incomplete, say exactly "
    "what was learned and what failed instead of asking for another tool."
)


def _load_openai_client_class():
    """Lazy load the OpenAI client so the file can be imported without the package."""
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("The openai Python package is required for LLMinterface.") from exc
    return OpenAI

def _message_to_dict(message):
    """
    The OpenAI SDK returns Pydantic objects, but we need standard Python dictionaries 
    to append to our message history. This safely converts them regardless of version.
    """
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported chat message type: {type(message).__name__}")

class LLMinterface(BaseLLM):
    def __init__(self, model, tools_json, api_url=None, api_key=None):
        """
        Initializes the Qwen LLM client using an OpenAI-compatible API wrapper.
        
        Args:
            model (str): Path to the Qwen .gguf model file.
            tools_json (list): The JSON list of tools the LLM is allowed to call.
            api_url (str): URL of the local LLM server.
            api_key (str): API key for the local server.
        """
        OpenAI = _load_openai_client_class()
        self.openai_client = OpenAI(
            base_url=api_url or cfg.LLM_API_URL,
            api_key=api_key or cfg.LLM_API_KEY,
        )
        self.tools = tools_json
        self.model = model

        self.completion = None
        self.reply = None

        # DYNAMIC SYSTEM PROMPT
        # We use an f-string to automatically insert the D435 resolution from config.py.
        # e.g., if config.py says (640, 480), this becomes "640x480".
        # This is CRITICAL. If you change the camera resolution in config.py but forget 
        # to tell the LLM, the LLM will guess coordinates for a 640x480 image while 
        # looking at a 1280x720 image, and the whole system will fail.
        d435_res = f"{cfg.D435_RESOLUTION[0]}x{cfg.D435_RESOLUTION[1]}"
        d405_res = f"{cfg.D405_RESOLUTION[0]}x{cfg.D405_RESOLUTION[1]}"

        self.messages = [
            {
                "role": "system",
                "content": dedent(f"""
                    You are an expert robotic vision and manipulation system.
                    Your goal is to interact with the workspace using provided cameras and tools.

                    CAMERA SYSTEM:
                    - D435: Overhead bird's eye view ({d435_res}) - sees entire workspace
                    - D405: Eye-in-hand wrist camera ({d405_res}) - sees close-up details, moves with robot

                    OBJECT LOCALIZATION STRATEGY:
                    1. Use get_birds_eye_view to see the workspace and locate the target object
                    2. Optionally use get_eye_in_hand_view if you need closer inspection
                    3. For BEST ACCURACY: Use get_xyz_fused with coordinates from BOTH cameras
                    4. If object is only visible in one camera, use get_xyz_fused with that camera's coords and null for the other
                    5. Only use get_xyz_d435 or get_xyz_d405 if you specifically want single-camera data
                    6. Use plan_robot_trajectory after localization when the user asks for a robot trajectory
                    7. Use execute_robot_waypoints only after plan_robot_trajectory when the user explicitly wants robot motion

                    IMPORTANT: When an object is visible in BOTH cameras, ALWAYS use get_xyz_fused 
                    with coordinates from both cameras. This provides more accurate 3D positioning 
                    by combining depth data from two viewpoints.

                    MOTION EXECUTION:
                    - plan_robot_trajectory only plans; it never moves the robot.
                    - execute_robot_waypoints moves the physical Franka robot using waypoints from
                      plan_robot_trajectory.
                    - Do not invent waypoints. Localize first, plan second, execute third.
                    - After execute_robot_waypoints, capture fresh synchronized images, localize again,
                      and re-plan before making another correction.

                    FRESHNESS / RE-PLANNING:
                    - The image and localization tools always capture a fresh synchronized D435/D405 pair.
                    - If the robot or target may have moved, capture new images, localize again, and re-plan.
                    - plan_robot_trajectory is not a continuous controller. It creates waypoints only from
                      the latest target_xyz you provide.
                    - Never claim robot motion was executed unless an execution tool reports success.
                    - When you identify a target object, localize it with get_xyz_fused or a single-camera
                      XYZ tool so the operator camera viewer can annotate the selected pixel and XYZ.
                    - After a successful localization, trajectory plan, or execution result, produce a
                      final text answer for the operator. Do not call another tool unless the user
                      explicitly asks for another image, another localization, replanning, or execution.
                    - If a tool returns an error or invalid point twice for the same target, stop and
                      explain what failed instead of repeating the same tool call.

                    Coordinate format: [u, v] where top-left is [0,0]
                    """).strip(),
            }
        ]

    def get_text(self):
        """Helper to get user input from the terminal."""
        self.text = input("Enter command: ")
        return self.text

    def send_message_with_tools(
        self,
        d435_cam,
        d405_cam,
        robot_ee_pose=None,
        robot_pose_provider=None,
        robot_interface=None,
        tool_result_callback=None,
    ):
        """
        The core LLM loop. 
        Args:
        d435_cam: RealSense D435 camera object
        d405_cam: RealSense D405 camera object
        robot_ee_pose: Current robot end-effector pose (4x4 matrix) - fallback pose
        robot_pose_provider: Optional callable that returns a fresh EE pose per tool call
        robot_interface: Optional RobotInterface used by execute_robot_waypoints
        tool_result_callback: Optional callable that receives each executed tool
            result for UI/debug overlays.
        """
        # Add user's command to history
        self.messages.append({"role": "user", "content": self.text})
        self.reply = None
        tool_rounds = 0
        
        while True:
            # Ask LLM what it wants to do
            self.completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto", # Let the LLM decide if it needs a tool or just wants to talk
                temperature=cfg.LLM_TEMPERATURE,
                max_tokens=cfg.LLM_MAX_OUTPUT_TOKENS,
            )
            msg = self.completion.choices[0].message
            
            # If no tool calls, the LLM is done thinking and has a final answer
            if not msg.tool_calls:
                self.reply = msg.content
                self.messages.append({"role": "assistant", "content": self.reply})
                break
                
            # If the LLM DID request a tool, we must execute it
            # Save the LLM's tool request in history (convert to dict first for safety)
            self.messages.append(_message_to_dict(msg))
            tool_rounds += 1

            if tool_rounds > cfg.LLM_MAX_TOOL_ROUNDS:
                for tool_call in msg.tool_calls:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "ERROR: maximum tool-call rounds reached.",
                    })
                self.reply = self._force_final_answer_without_tools()
                self.messages.append({"role": "assistant", "content": self.reply})
                break
            
            for tool_call in msg.tool_calls:
                # Parse the arguments the LLM sent (e.g., {"coords": [[320, 240]]})
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"ERROR: tool arguments were not valid JSON: {exc}",
                    })
                    continue

                if not isinstance(args, dict):
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "ERROR: tool arguments must be a JSON object.",
                    })
                    continue
                
                # DISPATCH THE TOOL TO HARDWARE
                # This calls the function in tools.py, which talks to the cameras
                try:
                    current_robot_ee_pose = robot_ee_pose
                    if robot_pose_provider is not None:
                        current_robot_ee_pose = robot_pose_provider()

                    result_text, extra_image_msg = tools.dispatch(
                        tool_call.function.name,
                        args,
                        d435_cam,
                        d405_cam,
                        robot_ee_pose=current_robot_ee_pose,
                        robot_interface=robot_interface,
                    )
                except Exception as exc:
                    result_text = f"ERROR: tool execution failed: {exc}"
                    extra_image_msg = None

                if tool_result_callback is not None:
                    try:
                        tool_result_callback(
                            tool_call.function.name,
                            args,
                            result_text,
                            self.text,
                        )
                    except Exception as exc:
                        print(f"Warning: tool result callback failed: {exc}")
                
                # Feed the text result back to the LLM (e.g., "XYZ coords are: 0.5m...")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
                
                # Feed the image result back to the LLM (if the tool returned an image)
                if extra_image_msg:
                    self.messages.append(extra_image_msg)

    def prune_image_history(self):
        """
        CRITICAL FOR LONGEVITY: 
        Base64 images are massive strings (~500,000 characters each). If we leave them in 
        the LLM's memory, after 3-4 prompts the context window will fill up (usually 8k or 
        32k tokens) and the API will crash with an Out of Memory error.
        
        This function iterates through chat history and deletes the base64 strings, keeping 
        only the text descriptions like "Image captured successfully."
        """
        self.messages = [
            m for m in self.messages
            # Keep the message UNLESS the content is a list containing an image_url type
            if not (
                isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"])
            )
        ]

    def print_message(self):
        """Helper to print the final response cleanly."""
        print(self.reply or "")

    def _force_final_answer_without_tools(self):
        """
        Ask for one last operator-facing answer with tools removed from the API call.

        WHY THIS EXISTS:
        Local vision models sometimes get into a loop like:
            image -> localization -> image -> localization -> ...
        even after the tool results already contain enough information to answer.
        The normal loop must keep tools available so the model can use the cameras,
        but once the safety limit trips, continuing with tools enabled would invite
        the exact same loop again. This method appends a direct "stop using tools"
        instruction, then makes a completion request WITHOUT the tools argument.

        WHY OMIT TOOLS INSTEAD OF ONLY tool_choice="none":
        This project talks to OpenAI-compatible local servers, not only the official
        OpenAI API. Some local servers ignore or only partially implement
        tool_choice="none". Omitting the tool schemas is the most portable way to
        make the final request a plain text response.
        """
        final_request = {"role": "user", "content": FINAL_ANSWER_PROMPT}
        self.messages.append(final_request)

        try:
            final_completion = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=cfg.LLM_TEMPERATURE,
                max_tokens=cfg.LLM_MAX_OUTPUT_TOKENS,
            )
            final_msg = final_completion.choices[0].message
            final_text = getattr(final_msg, "content", None)
            if isinstance(final_msg, dict):
                final_text = final_msg.get("content")
            if final_text:
                return final_text
        except Exception as exc:
            return (
                "Stopped because the model kept requesting tools without producing "
                f"a final answer, and the forced final summary failed: {exc}"
            )

        return (
            "Stopped because the model kept requesting tools without producing "
            "a final answer, and the forced final summary was empty."
        )
