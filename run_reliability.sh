#!/bin/bash
# 一键：可靠性加权融合 + 评测 + 消融（方案 §8.3）
#   用法: bash run_reliability.sh [cv5|holdout] [npz路径]
#   cv5 = 协议B0(现成1200 OOF)；holdout = 协议A(需先扩集到1532)
set -u
PROJ=/opt/data/wlcc/MER2026_Track2; cd "$PROJ"
AFF=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
GT=/opt/data/wlcc/mer2026-data/track2_train_human.csv
SPLIT=${1:-cv5}
NPZ=${2:-output/results-emosync/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184/checkpoint_000035_loss_0.716_logits.npz}
echo "=== 可靠性加权 (split=$SPLIT) ==="
CUDA_VISIBLE_DEVICES="" $AFF -u fuse_logits.py --split "$SPLIT" --gt "$GT" --npz "$NPZ" --ablation
