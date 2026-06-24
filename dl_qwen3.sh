#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
HFCLI=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/huggingface-cli
DEST=/opt/data/wlcc/MER2026_Track2/my_affectgpt/models/Qwen3-8B
echo "[start] $(date)  endpoint=$HF_ENDPOINT  hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER"
$HFCLI download Qwen/Qwen3-8B --local-dir $DEST
echo "DOWNLOAD_EXIT=$?  $(date)"
