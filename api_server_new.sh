#!/bin/bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/s2-pro" \
    --decoder-checkpoint-path "checkpoints/s2-pro/codec.pth" \
    --api-key "lVu7HLfx8No2B8uz12IGyPW9jpLfvyGzGlxx3zwi0Yg=" \
    --idle-timeout 0 \
    --half \
    --compile \
    --device cuda
