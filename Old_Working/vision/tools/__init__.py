"""
Vision Tools Package
--------------------
Public API for the LLM tool system.

This is the ONLY file that other modules should import from.
It provides:
  - tool_json_list: The JSON schemas sent to the LLM
  - dispatch(): The function that executes tool calls

USAGE:
    from vision.tools import tool_json_list, dispatch
    
    # When setting up the LLM:
    llm = LLMinterface(tools_json=tool_json_list, ...)
    
    # When the LLM calls a tool:
    result_text, image_msg = dispatch(tool_name, tool_args, d435_cam, d405_cam)
"""

# Import and re-export the public API
from .schemas import tool_json_list
from .dispatcher import dispatch

# Define what's available when someone does "from vision.tools import *"
__all__ = ['tool_json_list', 'dispatch']