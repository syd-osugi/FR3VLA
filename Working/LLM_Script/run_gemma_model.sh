#!/bin/bash

THINKING=false

for arg in "$@"; do
  if [ "$arg" = "--thinking" ]; then
    THINKING=true
  fi
done

if [ "$THINKING" = true ]; then
  TEMP=1.0
  TOP_P=0.95
  TOP_K=20
  MIN_P=0.0
  PRESENCE_PENALTY=1.5
  THINKING_FLAG='{"enable_thinking":true}'
else
  TEMP=0.7
  TOP_P=0.8
  TOP_K=20
  MIN_P=0.0
  PRESENCE_PENALTY=0.2 #A presence penalty of 1.5 is extremely aggressive for Gemma models. It heavily penalizes the model for repeating any words, which frequently breaks structural formatting, causes the model to lose track of markdown syntax, and produces incoherent gibberish during long-form "thinking" generation.
  THINKING_FLAG='{"enable_thinking":false}'
fi

ARGS=(
  -m models/gemma-4-E4B-it-UD-IQ3_XXS.gguf
  --mmproj models/mmproj-F16_gemma.gguf
  --ctx-size 131072 #Gemma 4 models have a maximum native context window of 131,072 tokens (128K). Setting --ctx-size 500000 (500K) forces the engine to allocate a massive, unnecessary KV cache, which will likely exhaust your RAM/VRAM and crash the server instantly. Furthermore, stretching RoPE embeddings beyond the model's trained limits without a proper scaling factor will corrupt the output.
  --temp "$TEMP"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --presence-penalty "$PRESENCE_PENALTY"
  --port 8080
  --jinja
  --chat-template-kwargs "$THINKING_FLAG"
)

llama-server "${ARGS[@]}"