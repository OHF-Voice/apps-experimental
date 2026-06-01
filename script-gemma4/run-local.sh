#!/usr/bin/env bash
cd /usr/src

export HF_HOME=/data/cache
mkdir -p /data/state

exec .venv/bin/python3 app.py \
    --uri 'tcp://0.0.0.0:10500' \
    --http-host '0.0.0.0' \
    --http-port 5000 \
    --llama-state '/data/state/llama_state.bin' "$@"
