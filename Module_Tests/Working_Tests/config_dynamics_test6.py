"""
Test 6: Configuration Dynamics
--------------------------------
Verifies that changing config.py actually changes the text sent to the LLM.
That if you change the resolution in config.py from 640x480 to 1280x720, the LLM tool 
descriptions and system prompts actually update themselves. If you forget an f-string somewhere, 
the LLM will think the image is 640x480 when it's actually 1280x720, and all coordinates will fail.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import config as cfg
import vision.tools as tools

def main():
    print("--- Testing Config Dynamics ---")
    
    # The tools.py file uses cfg.D435_RESOLUTION to build its JSON descriptions.
    # Let's read the description of the first tool and see if it contains the numbers.
    tool_desc = tools.tool_json_list[0]["function"]["description"]
    
    expected_w = cfg.D435_RESOLUTION[0]
    expected_h = cfg.D435_RESOLUTION[1]
    
    # Check if the dynamic string injection worked
    if f"{expected_w}x{expected_h}" in tool_desc:
        print(f"PASS: Tool JSON correctly updated with resolution {expected_w}x{expected_h}")
    else:
        print(f"FAIL: Tool JSON is static! Expected '{expected_w}x{expected_h}' but got: {tool_desc}")
        print("Check your f-strings in vision/tools.py")

if __name__ == "__main__":
    main()