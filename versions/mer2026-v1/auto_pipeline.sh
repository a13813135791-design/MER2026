#!/bin/bash
# EmoSync 全自动复现：① 校准测真分 → ② 全量1200出最终分
# 每阶段:产物校验 + 可断点续跑(已完成则SKIP) + 失败写 FAILED 标记并退出
PROJ=/opt/data/wlcc/MER2026_Track2
cd "$PROJ" || exit 1
AFF=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
Q3=/opt/data/wlcc/Software/Anaconda3/envs/qwen3eop/bin/python
CFG=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
CKPTDIR=mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184
RESDIR=output/results-emosync/$CKPTDIR
EPOCH=checkpoint_000035_loss_0.716
NPZ=$RESDIR/$EPOCH.npz
EOP=$RESDIR/$EPOCH-eop.npz
CALDIR=output/_cal
CALNPZ=$CALDIR/$EPOCH.npz
SCORES=$CALDIR/scores.json
mkdir -p "$CALDIR"

fail(){ echo "===$1_FAILED $(date '+%F %T')==="; exit 1; }
echo "===PIPELINE_START $(date '+%F %T')==="

# ===== Stage 1: 校准集推理 human[1200:1532]=332 (带caption,确定性) =====
if [ -f "$CALNPZ" ]; then
  echo "===STAGE1_CAL_INFER_SKIP==="
else
  mkdir -p output/_backup_verify
  mv "$RESDIR"/*.npz output/_backup_verify/ 2>/dev/null   # 清掉验证残留,防SKIP
  echo "===STAGE1_CAL_INFER_START $(date '+%F %T')==="
  CUDA_VISIBLE_DEVICES=0 $AFF -u inference_emosync.py --cfg-path=$CFG --dataset MER2026OV \
    --train_debug --options 'inference.test_epoch=35' --start_idx 1200 --max_samples 332 || fail STAGE1_CAL_INFER
  [ -f "$NPZ" ] || fail STAGE1_CAL_INFER_NOOUTPUT
  mv "$NPZ" "$CALNPZ"
  echo "===STAGE1_CAL_INFER_DONE $(date '+%F %T')==="
fi

# ===== Stage 2: 分视角测真分 -> scores.json =====
if [ -f "$SCORES" ]; then
  echo "===STAGE2_SCORES_SKIP==="
else
  echo "===STAGE2_SCORES_START $(date '+%F %T')==="
  CUDA_VISIBLE_DEVICES=0 $AFF -u run_perspective_scores.py "$CALNPZ" "$SCORES" || fail STAGE2_SCORES
  [ -f "$SCORES" ] || fail STAGE2_SCORES_NOOUTPUT
  echo "--- scores.json ---"; cat "$SCORES"; echo
  echo "===STAGE2_SCORES_DONE $(date '+%F %T')==="
fi

# ===== Stage 3: 全量1200推理 human[0:1200]=测试集 (带caption,确定性) =====
if [ -f "$NPZ" ]; then
  echo "===STAGE3_FULL_INFER_SKIP==="
else
  echo "===STAGE3_FULL_INFER_START $(date '+%F %T')==="
  CUDA_VISIBLE_DEVICES=0 $AFF -u inference_emosync.py --cfg-path=$CFG --dataset MER2026OV \
    --train_debug --options 'inference.test_epoch=35' --start_idx 0 --max_samples 1200 || fail STAGE3_FULL_INFER
  [ -f "$NPZ" ] || fail STAGE3_FULL_INFER_NOOUTPUT
  echo "===STAGE3_FULL_INFER_DONE $(date '+%F %T')==="
fi

# ===== Stage 4: Qwen3 EoP(用真分) -> -eop.npz =====
if [ -f "$EOP" ]; then
  echo "===STAGE4_EOP_SKIP==="
else
  echo "===STAGE4_EOP_START $(date '+%F %T')==="
  CUDA_VISIBLE_DEVICES=0 $Q3 run_eop_qwen3.py --npz "$NPZ" --score_json "$SCORES" || fail STAGE4_EOP
  [ -f "$EOP" ] || fail STAGE4_EOP_NOOUTPUT
  echo "===STAGE4_EOP_DONE $(date '+%F %T')==="
fi

# ===== Stage 5: numpy 跨版本格式转换(qwen3eop存的让affectgpt能读) =====
echo "===STAGE5_CONVERT_START $(date '+%F %T')==="
$Q3 -c "import numpy as np,json; d=np.load('$EOP',allow_pickle=True); json.dump({k:d[k].item() for k in d.files},open('/tmp/eop_data.json','w'),ensure_ascii=False)" || fail STAGE5_CONVERT
$AFF -c "import numpy as np,json; d=json.load(open('/tmp/eop_data.json')); np.savez_compressed('$EOP',**d)" || fail STAGE5_CONVERT
echo "===STAGE5_CONVERT_DONE $(date '+%F %T')==="

# ===== Stage 6: 抽openset标签(Qwen2.5) =====
echo "===STAGE6_OVLABEL_START $(date '+%F %T')==="
CUDA_VISIBLE_DEVICES=0 $AFF -u run_ovlabel_emosync.py || fail STAGE6_OVLABEL
echo "===STAGE6_OVLABEL_DONE $(date '+%F %T')==="

# ===== Stage 7: 评估 EW-F1 =====
echo "===STAGE7_EVAL_START $(date '+%F %T')==="
$AFF -u run_evaluation_emosync.py || fail STAGE7_EVAL
echo "===STAGE7_EVAL_DONE $(date '+%F %T')==="
echo "===PIPELINE_DONE $(date '+%F %T')==="
