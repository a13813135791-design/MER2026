#!/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
# -*- coding: utf-8 -*-
"""
分视角测真分 —— EmoSync EoP 的 performance score（论文 §2.3 的 "overall score"）。
在校准集(human 去掉 1200 测试集的那 332 个，有 GT)上，对 5 个视角各自抽 openset 标签、
和 GT 算 EW-F1，得到 5 个真实表现分，写成 scores.json 供 run_eop_qwen3.py 用。

无泄漏：校准集和最终评测的 1200 测试集零重叠。

用法: python run_perspective_scores.py <校准集npz> <输出scores.json>
"""
import sys, os, json
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2')
os.chdir('/opt/data/wlcc/MER2026_Track2')
import numpy as np
import pandas as pd
import config
from ovlabel_extraction import func_read_batch_calling_model, extract_openset_batchcalling
from my_affectgpt.evaluation.wheel import wheel_metric_calculation
from toolkit.utils.functions import string_to_list

CAL_NPZ  = sys.argv[1]
OUT_JSON = sys.argv[2]

# ---- GT ----
label_csv = config.PATH_TO_LABEL['Human']
df = pd.read_csv(label_csv)
name2gt = {row['name']: ', '.join(string_to_list(row['openset'])) for _, row in df.iterrows()}
print(f'[scores] GT 样本数: {len(name2gt)}', flush=True)

# ---- 读校准集中间结果 ----
d = np.load(CAL_NPZ, allow_pickle=True)
name2aur = d['name2aur'].item()
name2lsc = d['name2lsc'].item() if 'name2lsc' in d.files else {}
names = list(name2aur.keys())
print(f'[scores] 校准集样本数: {len(names)}', flush=True)

def lsc_text(n):
    v = name2lsc.get(n, {})
    if isinstance(v, dict):
        return v.get('fewshot') or v.get('cot1') or ''
    return ''

# 5 个视角的 {name: 文本}
perspectives = {
    'audio': {n: str(name2aur[n].get('audio', '')) for n in names},
    'face':  {n: str(name2aur[n].get('face',  '')) for n in names},
    'text':  {n: str(name2aur[n].get('text',  '')) for n in names},
    'multi': {n: str(name2aur[n].get('multi', '')) for n in names},
    'lsc':   {n: str(lsc_text(n)) for n in names},
}

# ---- 加载 Qwen2.5(抽标签模型)，每个视角抽一次 openset ----
print('[scores] 加载 Qwen25 抽标签模型 ...', flush=True)
llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname="Qwen25")

scores = {}
for persp, name2text in perspectives.items():
    # 过滤掉无效输出([SKIP]/[ERROR]/空)，避免污染抽取
    clean = {n: t for n, t in name2text.items() if t and not str(t).startswith('[')}
    if not clean:
        print(f'[scores] {persp}: 无有效文本，跳过(给低先验0.25)', flush=True)
        scores[persp] = 0.25
        continue
    tmp_in  = f'/tmp/_persp_{persp}_in.npz'
    tmp_out = f'/tmp/_persp_{persp}_out.npz'
    np.savez_compressed(tmp_in, name2reason=clean)
    extract_openset_batchcalling(reason_npz=tmp_in, store_npz=tmp_out,
                                 llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    od = np.load(tmp_out, allow_pickle=True)
    name2pred = {n: item for n, item in zip(od['filenames'], od['fileitems'])}
    common = [n for n in name2pred if n in name2gt]
    gt_sub   = {n: name2gt[n]   for n in common}
    pred_sub = {n: name2pred[n] for n in common}
    f1, _, _ = wheel_metric_calculation(name2gt=gt_sub, name2pred=pred_sub, inter_print=False)
    scores[persp] = round(float(f1), 4)
    print(f'[scores] {persp}: EW-F1 = {f1*100:.2f}  (n={len(common)})', flush=True)

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(scores, f, ensure_ascii=False, indent=2)
print(f'[scores] SAVED -> {OUT_JSON}  {scores}', flush=True)
