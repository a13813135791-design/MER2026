# -*- coding: utf-8 -*-
"""可靠性权重 w_m：equal / prior / tier0（闭式）/ lcg（学习门控）/ text（单模态）。
统一吃 RAW 8 维特征行 feat_row[3,8] + mask_row[3]；lcg 内部用 gp 的 (mu,sd) 做 z-score。"""
import numpy as np
from . import MODS, MI


def reliability_weights(feat_row, mask_row, mode, gp=None, pi=None, T_tier0=0.25):
    """→ {modality: weight}（缺失模态自动不入、权重和为 1）。"""
    av = [m for m in MODS if mask_row[MI[m]] == 1]
    if not av:
        return {}
    if mode == 'text':
        return {'text': 1.0} if 'text' in av else {m: 1.0 / len(av) for m in av}
    if mode == 'equal':
        return {m: 1.0 / len(av) for m in av}
    if mode == 'prior':
        Z = sum(pi[m] for m in av)
        return {m: pi[m] / Z for m in av}
    if mode == 'tier0':
        # 法子1x3x6：pi_m * exp(-Hn/T) * 硬门控；-Hn = feat[...,0]
        r = {m: pi[m] * np.exp(-(-feat_row[MI[m], 0]) / T_tier0) for m in av}
        Z = sum(r.values())
        return {m: r[m] / Z for m in av}
    if mode == 'lcg':
        w_, b_, mu, sd, T = gp['w'], gp['b'], gp['mu'], gp['sd'], gp['T']
        s = {m: float(((feat_row[MI[m]] - mu) / sd) @ w_ + b_[MI[m]]) for m in av}
        ex = {m: np.exp(s[m] / T) for m in av}
        Z = sum(ex.values())
        return {m: ex[m] / Z for m in av}
    raise ValueError('unknown mode: %s' % mode)
