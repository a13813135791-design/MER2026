#!/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
# -*- coding: utf-8 -*-
"""方案2 logits 支持度结果评估：对 audio/face/text 三条单模态，
   按支持度取 Top-K(K=1/3/5 及与 GT 个数匹配) -> 经 emotion wheel 映射 ->
   用项目自带 wheel_metric_calculation 算 EW-F1(5 轮平均)+P/R。
   用法: python eval_logits.py [path_to_*_logits.npz]"""
import sys, os, glob, json
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2')
os.chdir('/opt/data/wlcc/MER2026_Track2')
import numpy as np
import pandas as pd
import config
from my_affectgpt.evaluation.wheel import wheel_metric_calculation
from toolkit.utils.functions import string_to_list

# ---- 定位 logits 结果 npz ----
if len(sys.argv) > 1:
    npz_path = sys.argv[1]
else:
    cands = sorted(glob.glob('output/results-emosync/*/*_logits.npz'))
    assert cands, 'no *_logits.npz found under output/results-emosync/'
    npz_path = cands[-1]
print('[eval] npz:', npz_path)
d = np.load(npz_path, allow_pickle=True)
name2logits = d['name2logits'].item()
labels = list(d['candidate_labels'])
tau = float(d['tau'])
print('[eval] samples in npz: %d | candidates: %d | tau: %s' % (len(name2logits), len(labels), tau))

# ---- GT: track2_test.csv 的 openset ----
test_csv = os.path.join(config.DATA_DIR['MER2026'], 'track2_test.csv')
gtdf = pd.read_csv(test_csv)
name2gt, name2gtn = {}, {}
for _, row in gtdf.iterrows():
    gl = string_to_list(row['openset'])
    name2gt[row['name']] = ', '.join(gl)
    name2gtn[row['name']] = max(1, len(gl))
print('[eval] GT samples:', len(name2gt))

MODALITIES = ['audio', 'face', 'text']
KS = [1, 3, 5, 'gt']   # 'gt' = 取与该样本 GT 标签数相同的 Top-K


def topk_labels(support, k):
    items = sorted(support.items(), key=lambda kv: kv[1], reverse=True)
    return [l for l, _ in items[:k]]


def modality_pred(name, modality, k):
    rec = name2logits.get(name, {})
    if not isinstance(rec, dict):
        return None
    md = rec.get(modality, {})
    if not isinstance(md, dict) or 'support' not in md:
        return None
    kk = name2gtn.get(name, 1) if k == 'gt' else k
    return ', '.join(topk_labels(md['support'], kk))


results = {}
print('\n==== 逐模态 EW-F1 (level1, 5 轮平均) ====')
for modality in MODALITIES:
    results[modality] = {}
    for k in KS:
        name2pred = {}
        for name in name2logits:
            if name not in name2gt:
                continue
            p = modality_pred(name, modality, k)
            if p is None:
                continue
            name2pred[name] = p
        common = list(name2pred.keys())
        if not common:
            print(f'[{modality:5s}] K={str(k):>3s}  no valid samples')
            continue
        gt_sub = {n: name2gt[n] for n in common}
        f, pr, rc = wheel_metric_calculation(name2gt=gt_sub, name2pred=name2pred, inter_print=False)
        results[modality][str(k)] = {'n': len(common), 'f1': f * 100, 'precision': pr * 100, 'recall': rc * 100}
        print(f'[{modality:5s}] K={str(k):>3s}  n={len(common):4d}  EW-F1={f*100:6.2f}  P={pr*100:6.2f}  R={rc*100:6.2f}')

out = {'npz': npz_path, 'tau': tau, 'n_candidates': len(labels), 'results': results}
summary_path = 'output/results-emosync/logits_eval_summary.json'
with open(summary_path, 'w', encoding='utf-8') as fj:
    json.dump(out, fj, ensure_ascii=False, indent=2)
print('\n[eval] summary saved to', summary_path)

# ---- 若干样本 case（预测 vs 真值）----
print('\n==== example cases (top-5 by support) ====')
shown = 0
for name in name2logits:
    if name not in name2gt:
        continue
    rec = name2logits[name]
    if not isinstance(rec, dict) or 'audio' not in rec:
        continue
    print(f'\n# {name}\n  GT   : {name2gt[name]}')
    for modality in MODALITIES:
        md = rec.get(modality, {})
        top = md.get('topk', [])[:5] if isinstance(md, dict) else []
        print(f'  {modality:5s}: ' + ', '.join(f'{l}:{float(p):.3f}' for l, p in top))
    shown += 1
    if shown >= 8:
        break
print('\n[eval] done.')
