###########################
# Tests if your local LLM server (LM Studio or llama.cpp) is actually turned on 
# and accessible at the URL you put in config.py. It sends a tiny text message and waits for a response.
# If this fails: You forgot to open LM Studio, or the port number in config.py is wrong.
###########################
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'working'))

import config as cfg

def main():
    print("--- Testing LLM Server Connection ---")
    print(f"Target URL: {cfg.LLM_API_URL}")
    print(f"Target Model: {cfg.QWEN_MODEL_PATH}")
    
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.LLM_API_URL, api_key=cfg.LLM_API_KEY)
        
        response = client.chat.completions.create(
            model=cfg.QWEN_MODEL_PATH,
            messages=[{"role": "user", "content": "Say the word 'connected'."}],
            max_tokens=10
        )
        
        print(f"LLM Replied: {response.choices[0].message.content}")
        print("PASS: LLM connection successful.")
    except Exception as e:
        print(f"FAIL: Could not connect. Error: {e}")

if __name__ == "__main__":
    main()