# -*- coding: utf-8 -*-
"""评测：①快速逐样本 wheel-F1（标定软标签，复用 OUTSIDE_WHEEL_MAPPING）；
②批量 EW-F1（直接调项目 wheel_metric_calculation，= run_evaluation_emosync 同一函数）。"""
import numpy as np
import config
from my_affectgpt.evaluation.wheel import wheel_metric_calculation
from toolkit.utils.functions import string_to_list

WHEELS = ['wheel1', 'wheel2', 'wheel3', 'wheel4', 'wheel5']


def build_wheel_bank(labels, name2gt=None):
    """预映射每个词到 5 个 wheel 的 level1 桶，供 samp_f1 快速算单样本 F1。"""
    wm = np.load(config.OUTSIDE_WHEEL_MAPPING, allow_pickle=True)
    fmt = wm['format_mapping'].item()
    rawm = wm['raw_mapping'].item()
    wmw = wm['wheel_map_whole'].item()

    def bw3(label, wheel_map):
        if label not in fmt:
            return ""
        cand = [r for f in fmt[label] for r in rawm.get(f, [])]
        for lv1 in sorted(cand):
            if lv1 in wheel_map:
                return wheel_map[lv1]
        return ""

    allwords = set(str(l).lower().strip() for l in labels)
    if name2gt:
        for n in name2gt:
            for w in string_to_list(name2gt[n]):
                allwords.add(w.lower().strip())
    bank = {wh: {w: bw3(w, wmw[wh]['level1']) for w in allwords} for wh in WHEELS}
    return bank


def samp_f1(gt_words, pred_words, bank):
    """单样本 wheel-F1（5 轮平均），软标签 y_{n,m}。"""
    f = []
    for wh in WHEELS:
        bd = bank[wh]
        g = set(b for b in (bd.get(w.lower().strip(), "") for w in gt_words) if b)
        p = set(b for b in (bd.get(w.lower().strip(), "") for w in pred_words) if b)
        if not g:
            continue
        if not p:
            f.append(0.0)
            continue
        I = len(g & p)
        P = I / len(p)
        R = I / len(g)
        f.append(0.0 if P + R == 0 else 2 * P * R / (P + R))
    return float(np.mean(f)) if f else 0.0


def eval_fused(name2pred, name2gt):
    """批量 EW-F1（项目评估函数）→ (n_common, f%, p%, r%)"""
    common = [n for n in name2pred if n in name2gt]
    if not common:
        return 0, 0.0, 0.0, 0.0
    f, p, r = wheel_metric_calculation(
        name2gt={n: name2gt[n] for n in common},
        name2pred={n: name2pred[n] for n in common}, inter_print=False)
    return len(common), f * 100, p * 100, r * 100
