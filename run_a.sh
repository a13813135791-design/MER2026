#!/bin/bash
# 协议 A：扩 logits 到 1532（GPU 推理缺的 332）→ 合并 → holdout 留出标定 + 消融 + 项目评估确认
# 安全：先备份原始 1200 logits，绝不丢失；任一步失败即还原并退出。
set -u
PROJ=/opt/data/wlcc/MER2026_Track2; cd "$PROJ"
AFF=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
CFG=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
GT=/opt/data/wlcc/mer2026-data/track2_train_human.csv
CKPTDIR=output/results-emosync/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184
NPZ=$CKPTDIR/checkpoint_000035_loss_0.716_logits.npz
TOPK=$CKPTDIR/checkpoint_000035_loss_0.716_logits_topk.json
BAK=$CKPTDIR/checkpoint_000035_loss_0.716_logits_1200.npz
BAKJSON=$CKPTDIR/checkpoint_000035_loss_0.716_logits_topk_1200.json
ADD=$CKPTDIR/checkpoint_000035_loss_0.716_logits_332.npz
MERGED=$CKPTDIR/checkpoint_000035_loss_0.716_logits_1532.npz
RESDIR=$CKPTDIR/mer2026-v3-result
mkdir -p "$RESDIR" output/_cal
ts(){ date '+%F %T'; }
cnt(){ $AFF -c "import numpy as np;print(len(np.load('$1',allow_pickle=True)['name2logits'].item()))" 2>/dev/null; }
echo "===A_START $(ts)==="

# 0) 备份原始 1200（幂等，npz + topk json 都备份，扩集会覆盖默认路径）
[ -f "$BAK" ] || cp "$NPZ" "$BAK"
[ -f "$BAKJSON" ] || { [ -f "$TOPK" ] && cp "$TOPK" "$BAKJSON"; }
echo "[A] 1200 backup: $BAK  (n=$(cnt "$BAK"))"

if [ -f "$MERGED" ]; then
  echo "[A] 1532 merged 已存在($(cnt "$MERGED"))，跳过扩集/合并"
else
  # 1) GPU 扩集：human[1200:1532] 的 332 个样本 → logits（--logits_csv 读 human，会覆盖默认 NPZ 路径）
  echo "===A_EXTEND_START $(ts)==="
  CUDA_VISIBLE_DEVICES=0 $AFF -u inference_emosync.py --cfg-path=$CFG --dataset MER2026OV \
    --options inference.test_epoch=35 --score-mode logits --logits_csv "$GT" --start_idx 1200 --max_samples 332
  rc=$?
  if [ $rc -ne 0 ]; then echo "===A_EXTEND_FAILED rc=$rc $(ts)==="; cp "$BAK" "$NPZ"; [ -f "$BAKJSON" ] && cp "$BAKJSON" "$TOPK"; exit $rc; fi
  # 此刻 NPZ 已被新结果覆盖：移走为 ADD，再用 BAK 还原原始 1200（npz + topk json）
  mv "$NPZ" "$ADD"
  cp "$BAK" "$NPZ"
  [ -f "$BAKJSON" ] && cp "$BAKJSON" "$TOPK"
  NADD=$(cnt "$ADD")
  echo "[A] 扩集得到 add=$NADD 样本；原始 1200 已还原到 $NPZ (n=$(cnt "$NPZ"))"
  if [ "${NADD:-0}" -lt 50 ]; then
    echo "===A_EXTEND_BAD add=$NADD（疑似 MER2026OV 未枚举到 1200+，放弃 A，保留 B0）$(ts)==="
    exit 0
  fi
  # 2) 合并 1200 + add → 1532
  echo "===A_MERGE_START $(ts)==="
  $AFF -u merge_logits.py --base "$BAK" --add "$ADD" --out "$MERGED"
  rc=$?
  if [ $rc -ne 0 ]; then echo "===A_MERGE_FAILED rc=$rc $(ts)==="; exit $rc; fi
fi
echo "[A] 1532 logits: $MERGED (n=$(cnt "$MERGED"))"
echo "===A_EXTEND_DONE $(ts)==="

# 3) holdout 留出标定：train=human∖test(332) / eval=test(1200)
echo "===A_HOLDOUT_START $(ts)==="
CUDA_VISIBLE_DEVICES="" $AFF -u fuse_logits.py --split holdout --gt "$GT" --npz "$MERGED" --ablation --outdir "$RESDIR"
rc=$?
if [ $rc -ne 0 ]; then echo "===A_HOLDOUT_FAILED rc=$rc $(ts)==="; exit $rc; fi

OS=$RESDIR/checkpoint_000035_loss_0.716_v3fused_a-openset.npz
if [ -f "$OS" ]; then
  EVDIR=output/results-emosync/_v3eval_a; rm -rf "$EVDIR"; mkdir -p "$EVDIR"; cp "$OS" "$EVDIR/"
  echo "--- run_evaluation_emosync.py 确认 (A, =项目评估代码) ---"
  $AFF -u run_evaluation_emosync.py 2>/dev/null | grep -A3 "_v3eval_a" || echo "(grep 未命中，完整输出见日志)"
  rm -rf "$EVDIR"
fi
touch "$RESDIR/A_DONE"
echo "===A_DONE $(ts)==="
