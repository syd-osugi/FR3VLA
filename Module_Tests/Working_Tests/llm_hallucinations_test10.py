"""
Test 10: LLM Hallucination Handling
-----------------------------------
Simulates the LLM sending completely broken JSON to the tool dispatcher.
LLMs are chaotic. Sometimes they send [u, v] as strings "['320', '240']" instead of integers, 
or they send three numbers instead of two. This tests if tools.py safely rejects garbage 
data without crashing the whole robot program.
"""
from _working_test_utils import add_working_to_path

add_working_to_path()

import config as cfg
from hardware.camera import RealSense
import vision.tools as tools

def main():
    print("--- Testing Tool Dispatch Error Handling ---")
    
    # We don't even need real cameras for this, but we pass None to test safety
    # If tools.dispatch crashes on bad args BEFORE talking to the camera, it fails.
    
    # Scenario A: LLM sends strings instead of integers
    print("Scenario A: LLM sends coords as strings ['320', '240']...")
    bad_args_a = {"coords": [["320", "240"]]}
    result_a, _ = tools.dispatch("get_xyz_coords", bad_args_a, None, None)
    if "invalid" in result_a or "integer" in result_a:
        print(f"PASS: Correctly rejected strings. Response: {result_a}")
    else:
        print(f"FAIL: Accepted strings! Response: {result_a}")

    # Scenario B: LLM sends 3 numbers instead of 2
    print("\nScenario B: LLM sends 3 numbers [100, 200, 300]...")
    bad_args_b = {"coords": [[100, 200, 300]]}
    result_b, _ = tools.dispatch("get_xyz_coords", bad_args_b, None, None)
    if "invalid" in result_b or "expected" in result_b:
        print(f"PASS: Correctly rejected bad array length. Response: {result_b}")
    else:
        print(f"FAIL: Accepted bad length! Response: {result_b}")

    # Scenario C: LLM hallucinates a tool that doesn't exist
    print("\nScenario C: LLM hallucinates tool 'make_me_a_sandwich'...")
    result_c, _ = tools.dispatch("make_me_a_sandwich", {}, None, None)
    if "Unknown tool" in result_c:
        print(f"PASS: Correctly rejected fake tool. Response: {result_c}")
    else:
        print(f"FAIL: Accepted fake tool! Response: {result_c}")

if __name__ == "__main__":
    main()
