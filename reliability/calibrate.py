# -*- coding: utf-8 -*-
"""标定：建 (样本,模态) 特征/软标签数据集 + 经验先验 π_m + z-score + 门控拟合（torch）。
软标签 y_{n,m} = 单样本 per-modality wheel-F1（快速 samp_f1）。"""
import numpy as np
from toolkit.utils.functions import string_to_list
from . import MODS, MI
from .features import consensus_vecs, rvec, feats
from .metrics import samp_f1


def build_arrays(names, name2logits, labels, neutral_vec, p0, name2gt, name2K, bank):
    """限定 names → X[N,3,8], Mask[N,3], Yt[N,3], SUP{name:{m:vec}}, keep:list, ni:{name:idx}"""
    keep = []
    for n in names:
        n = str(n)
        rec = name2logits.get(n)
        if isinstance(rec, dict) and 'status' not in rec and n in name2gt:
            keep.append(n)
    N = len(keep)
    ni = {n: i for i, n in enumerate(keep)}
    X = np.zeros((N, 3, 8))
    Mask = np.zeros((N, 3))
    Yt = np.zeros((N, 3))
    SUP = {}
    for n in keep:
        rec = name2logits[n]
        sv, av, cons = consensus_vecs(rec, labels)
        SUP[n] = sv
        gw = string_to_list(name2gt[n])
        K = name2K[n]
        i = ni[n]
        for m in av:
            X[i, MI[m]] = feats(sv[m], rvec(rec, m, labels, neutral_vec), neutral_vec, p0, cons[m])
            Mask[i, MI[m]] = 1
            pred = [labels[t] for t in np.argsort(sv[m])[::-1][:K]]
            Yt[i, MI[m]] = samp_f1(gw, pred, bank)
    return X, Mask, Yt, SUP, keep, ni


def empirical_prior(X, Mask, Yt, idx):
    """π_m = 各模态在标定集上的平均单模态 EW-F1，归一。"""
    pi = {}
    for m in MODS:
        vals = [Yt[i, MI[m]] for i in idx if Mask[i, MI[m]] == 1]
        pi[m] = float(np.mean(vals)) if vals else 1e-3
    Z = sum(pi.values())
    return {m: pi[m] / Z for m in MODS}


def zfit(X, Mask, idx):
    rows = X[np.asarray(idx)][Mask[np.asarray(idx)] == 1]
    mu = rows.mean(0)
    sd = rows.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def fit_gate(X, Mask, Yt, idx, mu, sd, T=1.0, Ty=0.1, lam=1e-3, steps=500, lr=0.05, seed=0):
    """listwise 门控：闭式特征头 s_m = w·z(φ)+b_m，softmax_m 逼近 softmax(y/Ty)。→ gp dict"""
    import torch
    torch.manual_seed(seed)
    tri = np.asarray(idx)
    Xt = torch.tensor((X[tri] - mu) / sd, dtype=torch.float32)
    Mt = torch.tensor(Mask[tri], dtype=torch.float32)
    Yk = torch.tensor(Yt[tri], dtype=torch.float32)
    w = torch.zeros(8, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        s = (Xt @ w + b) / T
        s = s.masked_fill(Mt == 0, -1e9)
        logp = torch.log_softmax(s, dim=1)
        q = torch.softmax((Yk / Ty).masked_fill(Mt == 0, -1e9), dim=1)
        loss = -(q * logp * Mt).sum(1).mean() + lam * (w * w).sum()
        loss.backward()
        opt.step()
    return {'w': w.detach().numpy(), 'b': b.detach().numpy(),
            'mu': np.asarray(mu), 'sd': np.asarray(sd), 'T': float(T)}
