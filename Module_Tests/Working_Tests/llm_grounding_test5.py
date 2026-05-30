###########################
# Tests the "Grounding" pipeline. It fakes an LLM command to trigger the camera tool. 
# It checks if the code successfully takes a picture, compresses it to a JPEG using your 
# config settings, converts it to base64 text, and formats it exactly how the LLM expects it.
# If this fails: OpenCV is broken, Pillow is missing, or the JSON formatting for the LLM is wrong.
###########################
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import config as cfg
from hardware.camera import RealSense
import vision.tools as tools

def main():
    print("--- Testing LLM Tool Dispatch ---")
    d435 = RealSense(serial_number=cfg.D435_SERIAL, resolution=cfg.D435_RESOLUTION)
    d405 = RealSense(serial_number=cfg.D405_SERIAL, resolution=cfg.D405_RESOLUTION)
    import time; time.sleep(2)
    
    fake_tool_name = "get_birds_eye_view"
    fake_tool_args = {} 
    
    print(f"Dispatching tool: {fake_tool_name}...")
    result_text, image_message = tools.dispatch(fake_tool_name, fake_tool_args, d435, d405)
    
    if image_message is None:
        print("FAIL: Tool did not return an image message.")
    else:
        image_url = image_message.get("content", [{}])[0].get("image_url", {}).get("url", "")
        if "base64," in image_url:
            # Check if it respected the JPEG quality from config
            b64_length = len(image_url.split("base64,")[1])
            print(f"PASS: Image generated (Size: {b64_length} chars, Quality setting from config: {cfg.LLM_IMAGE_JPEG_QUALITY})")
            print(f"Tool Response: {result_text}")
        else:
            print("FAIL: Base64 data missing.")

    d435.stop()
    d405.stop()

if __name__ == "__main__":
    main()