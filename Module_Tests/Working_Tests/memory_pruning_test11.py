"""
Test 11: Memory Pruning Verification
------------------------------------
Directly tests the LLM interface memory management without needing a real LLM server.
hat the prune_image_history() function in llm_interface.py actually deletes the giant 
500,000-character base64 strings from the LLM's memory, but leaves normal text alone. 
If this fails, your robot will work for 5 minutes and then crash with an "Out of Memory" error.
"""
import os

from _working_test_utils import add_working_to_path

add_working_to_path()

import base64
from io import BytesIO
from PIL import Image
import vision.llm_interface as llm_module

def generate_fake_b64_string(size_kb=500):
    """Generates a fake base64 string of a specific size in KB."""
    # Just create a random block of bytes to simulate an image
    fake_bytes = os.urandom(size_kb * 1024)
    return base64.b64encode(fake_bytes).decode("utf-8")

def main():
    print("--- Testing Memory Pruning ---")
    
    # Create a fake instance just to test the prune method (doesn't need API keys)
    # We bypass __init__ by directly defining the messages list
    fake_llm = type('FakeLLM', (), {})()
    fake_llm.messages = [
        # 1. Normal text (MUST BE KEPT)
        {"role": "user", "content": "Find the cup."},
        
        # 2. Normal assistant reply (MUST BE KEPT)
        {"role": "assistant", "content": "I will look for it."},
        
        # 3. A massive image message (MUST BE DELETED)
        {
            "role": "tool", 
            "tool_call_id": "123", 
            "content": [
                {"type": "text", "text": "Here is the image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{generate_fake_b64_string(500)}"}}
            ]
        },
        
        # 4. Another normal message after the image (MUST BE KEPT)
        {"role": "assistant", "content": "I see the cup at pixel 320, 240."}
    ]
    
    original_length = len(fake_llm.messages)
    print(f"Original message count: {original_length}")
    
    # Run the prune function
    llm_module.LLMinterface.prune_image_history(fake_llm)
    
    pruned_length = len(fake_llm.messages)
    print(f"Pruned message count: {pruned_length}")
    
    if pruned_length == 3:
        print("PASS: Image was deleted, text was preserved.")
    elif pruned_length == 4:
        print("FAIL: Image was NOT deleted! Pruning function is broken.")
    else:
        print(f"FAIL: Unexpected number of messages remaining ({pruned_length}).")

if __name__ == "__main__":
    main()
