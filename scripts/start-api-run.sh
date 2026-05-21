#!/bin/bash
cd /Users/mac/sloughGPT
export SLOUGHGPT_MODEL_PATH=/Users/mac/models/llama3.2-1b-q8_0.gguf
exec /usr/bin/python3 apps/api/server/main.py
