#!/bin/bash
# 协议 B0：现成 1200 logits → 5 折 OOF 可靠性加权融合 + 消融 + 项目评估确认
set -u
PROJ=/opt/data/wlcc/MER2026_Track2; cd "$PROJ"
AFF=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
GT=/opt/data/wlcc/mer2026-data/track2_train_human.csv
CKPTDIR=output/results-emosync/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184
NPZ=$CKPTDIR/checkpoint_000035_loss_0.716_logits.npz
RESDIR=$CKPTDIR/mer2026-v3-result
mkdir -p "$RESDIR" output/_cal
ts(){ date '+%F %T'; }
echo "===B0_START $(ts)==="
CUDA_VISIBLE_DEVICES="" $AFF -u fuse_logits.py --split cv5 --gt "$GT" --npz "$NPZ" --ablation --outdir "$RESDIR"
rc=$?
if [ $rc -ne 0 ]; then echo "===B0_FAILED rc=$rc $(ts)==="; exit $rc; fi
# 项目正式入口 run_evaluation_emosync.py 确认（拷到 glob 可命中目录）
OS=$RESDIR/checkpoint_000035_loss_0.716_v3fused_b0-openset.npz
if [ -f "$OS" ]; then
  EVDIR=output/results-emosync/_v3eval_b0; rm -rf "$EVDIR"; mkdir -p "$EVDIR"; cp "$OS" "$EVDIR/"
  echo "--- run_evaluation_emosync.py 确认 (B0, =项目评估代码) ---"
  $AFF -u run_evaluation_emosync.py 2>/dev/null | grep -A3 "_v3eval_b0" || echo "(grep 未命中，完整输出见日志)"
  rm -rf "$EVDIR"
fi
touch "$RESDIR/B0_DONE"
echo "===B0_DONE $(ts)==="
