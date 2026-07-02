# -*- coding: utf-8 -*-
"""可靠性权重 w_m：equal / prior / tier0（闭式）/ lcg（学习门控）/ text（单模态）/ mlpcap（caption 增强 MLP, v3.2）。
统一吃 RAW 8 维特征行 feat_row[3,8] + mask_row[3]；lcg/mlpcap 内部用 gp 的 (mu,sd) 做 z-score。
mlpcap 额外吃每样本 caption 特征 cap_row（内部用 gp['cap'] 的 (mu,sd) 归一）。纯 numpy，不 import torch。"""
import numpy as np
from . import MODS, MI


def reliability_weights(feat_row, mask_row, mode, gp=None, pi=None, T_tier0=0.25, cap_row=None):
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
    if mode == 'mlpcap':
        # v3.2：逐模态共享 MLP，输入 concat(z(φ_m)[8], z(cap)[D])。
        # 同一 caption 经 MLP 隐层非线性对三模态 s_m 产生不同调制（线性头会因 softmax 平移不变而抵消）。
        assert cap_row is not None, '[mlpcap] 需要 cap_row（caption 特征向量）'
        ml, cp, T = gp['mlp'], gp['cap'], gp['T']
        W, B, mu, sd, act = ml['W'], ml['B'], ml['mu'], ml['sd'], ml['act']
        zc = (np.asarray(cap_row, dtype=float) - cp['mu']) / cp['sd']

        def _fwd(zphi):
            x = np.concatenate([zphi, zc])           # 顺序必须与训练一致：φ 在前、cap 在后
            for li in range(len(W)):
                x = W[li] @ x + B[li]
                if li < len(W) - 1:
                    x = np.maximum(0.0, x) if act == 'relu' \
                        else 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
            return float(x[0])
        s = {m: _fwd((feat_row[MI[m]] - mu) / sd) for m in av}
        ex = {m: np.exp(s[m] / T) for m in av}
        Z = sum(ex.values())
        return {m: ex[m] / Z for m in av}
    raise ValueError('unknown mode: %s' % mode)
