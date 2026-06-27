# -*- coding: utf-8 -*-
"""逐样本逐模态 8 维 0 成本特征 φ_m（与 fuse_logits_cv.py::feats 完全一致）。"""
import numpy as np
from . import MODS

FEAT_NAMES = ['-Hn', 'top1', 'margin', 'gini', 'mass', 'shiftmax', 'klprior', 'agree']


def svec(res, m, labels):
    """该模态 support 向量（全 253 候选）；缺 support 或全 0 → None（硬门控）。"""
    md = res.get(m, {}) if isinstance(res, dict) else {}
    if not isinstance(md, dict) or 'support' not in md:
        return None
    v = np.array([md['support'].get(l, 0.0) for l in labels], dtype=float)
    return v if v.sum() > 0 else None


def rvec(res, m, labels, neutral_vec):
    """该模态 raw 向量；缺项以 neutral 兜底。"""
    md = res.get(m, {}) if isinstance(res, dict) else {}
    rd = md.get('raw', {}) if isinstance(md, dict) else {}
    return np.array([rd.get(l, neutral_vec[i]) for i, l in enumerate(labels)], dtype=float)


def neutral_p0(neutral_vec, tau):
    """中性 PMI 先验分布 p0 = softmax(neutral / tau)。"""
    p0 = np.exp((neutral_vec - neutral_vec.max()) / tau)
    return p0 / p0.sum()


def consensus_vecs(res, labels):
    """→ (sv:{m:vec|None}, av:可用模态列表, cons:{m:其它模态 support 均值|None})"""
    sv = {m: svec(res, m, labels) for m in MODS}
    av = [m for m in MODS if sv[m] is not None]
    cons = {}
    for m in MODS:
        others = [sv[k] for k in av if k != m]
        cons[m] = np.mean(others, axis=0) if others else None
    return sv, av, cons


def feats(sup, raw, neutral_vec, p0, cons):
    """8 维特征：[-Hn, top1, margin, gini, mass, shiftmax, klprior, agree]"""
    L = len(sup)
    p = np.sort(sup)[::-1]
    mass = sup.sum()
    H = -(sup * np.log(sup + 1e-12)).sum()
    Hn = H / np.log(L)
    p1, margin = p[0], p[0] - p[1]
    gini = 1 - (sup ** 2).sum() / (mass ** 2 + 1e-12)
    shiftmax = float((raw - neutral_vec).max())
    klprior = float((sup * np.log((sup + 1e-12) / (p0 + 1e-12))).sum())
    if cons is None or np.linalg.norm(cons) == 0:
        agree = 0.0
    else:
        agree = float(np.dot(sup, cons) / (np.linalg.norm(sup) * np.linalg.norm(cons) + 1e-12))
    return np.array([-Hn, p1, margin, gini, mass, shiftmax, klprior, agree])
