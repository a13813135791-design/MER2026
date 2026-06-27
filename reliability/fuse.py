# -*- coding: utf-8 -*-
"""加权线性池融合 + TopK 读出 + 总不确定性 U（仅记录）。"""
import numpy as np


def fuse(sup_by_m, w, labels):
    """sup_by_m:{m:support_vec}；w:{m:weight} → (融合分布 vec, U)。"""
    L = len(labels)
    fused = np.zeros(L)
    for m, wm in w.items():
        v = sup_by_m.get(m)
        if v is not None:
            fused += wm * v
    Z = fused.sum()
    if Z > 0:
        fused = fused / Z
    U = float(sum(wm * (1.0 - (sup_by_m[m].sum() if sup_by_m.get(m) is not None else 1.0))
                  for m, wm in w.items()))
    return fused, U


def read_topk(fused_vec, labels, K):
    idx = np.argsort(fused_vec)[::-1][:K]
    return [labels[i] for i in idx]
