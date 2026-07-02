# -*- coding: utf-8 -*-
"""train_mlp_gate_full.py — 全量训练 per-modality 共享 MLP 门控（torch）。
基于 train_mlp_gate.py 改造：
  * 用【全部】有 logits 且有 GT 的样本训练（human 真标注 + mcp 伪标注）；
  * 【不切】holdout、【不排除】test —— 无验证集 / 无测试集；
  * 无 holdout 早停，改为固定 epochs + 在【训练集自身】上按 EW-F1 存最优快照来定参（可复现）；
  * 默认保留全部样本（keep_noise=1，不丢 mcp 全错伪标签）；
  * 运行/输出均指向 _1027 副本，输出到不覆盖原文件的 gate_params_mlp_full.json。
训练侧独占 torch；推理侧(weights/io/fuse/features/apply_gate_candidate)绝不触达。"""
import sys, os, json, argparse, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJ = '/opt/data/wlcc/MER2026_Track2_1027'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd, torch, torch.nn as nn
import config
from toolkit.utils.functions import string_to_list
from reliability import MODS, MI
from reliability.io import load_logits
from reliability.features import neutral_p0
from reliability.metrics import build_wheel_bank, eval_fused
from reliability.calibrate import build_arrays, empirical_prior, zfit
from reliability.fuse import fuse, read_topk


def load_gt_merged(human_csv, mcp_csv):
    """MCP(src=0) 先读、human(src=1) 后读覆盖同名；跳过 MCP 空 openset 行(无 GT，无法监督)。"""
    name2gt, name2K, name2src = {}, {}, {}
    for csv_path, src in [(mcp_csv, 0), (human_csv, 1)]:
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            gl = string_to_list(r['openset'])
            if src == 0 and len(gl) == 0:
                continue
            nm = str(r['name'])
            name2gt[nm] = ', '.join(gl); name2K[nm] = max(1, len(gl)); name2src[nm] = src
    return name2gt, name2K, name2src


class GateMLP(nn.Module):
    def __init__(self, d=8, h=32, act='gelu', dropout=0.15):
        super().__init__()
        self.fc1 = nn.Linear(d, h); self.fc2 = nn.Linear(h, 1)
        self.drop = nn.Dropout(dropout); self.act = act
    def _a(self, x):
        return torch.nn.functional.gelu(x, approximate='tanh') if self.act == 'gelu' else torch.relu(x)
    def forward(self, x):
        return self.fc2(self.drop(self._a(self.fc1(x)))).squeeze(-1)


def eval_ewf1(model, Xz, Mask, SUP, names, ni, labels, name2K, name2gt, T):
    """在给定 names 上算融合 EW-F1（全量训练里 names=训练集自身，作为定参依据）。"""
    model.eval(); name2pred = {}
    with torch.no_grad():
        for n in names:
            if n not in ni: continue
            i = ni[n]; av = [m for m in MODS if Mask[i, MI[m]] == 1]
            if not av: continue
            s = {m: float(model(torch.tensor(Xz[i, MI[m]], dtype=torch.float32))) for m in av}
            ex = {m: np.exp(s[m] / T) for m in av}; Z = sum(ex.values())
            w = {m: ex[m] / Z for m in av}
            sup_by_m = {m: SUP[n][m] for m in av if SUP[n].get(m) is not None}
            fused, _ = fuse(sup_by_m, w, labels)
            name2pred[n] = ', '.join(read_topk(fused, labels, name2K[n]))
    if not name2pred: return 0.0
    _, f, _, _ = eval_fused(name2pred, name2gt)
    return f / 100.0


def build_rows(train_names, ni, Mask, Yt, name2src, w_mcp, keep_noise=True):
    """全量 rows。keep_noise=True 时保留全部（含 mcp 全错伪标签）；
    keep_noise=False 才丢弃【src=0 且所有可用模态 F1 全 0】的纯噪声样本。human(src=1) 恒保留。"""
    rows, n_drop = [], 0
    for n in train_names:
        if n not in ni: continue
        i = ni[n]; mrow = Mask[i].copy()
        av = [m for m in MODS if mrow[MI[m]] == 1]
        if (not keep_noise) and name2src.get(n, 1) == 0 and all(Yt[i, MI[m]] <= 0.0 for m in av):
            n_drop += 1; continue
        sw = 1.0 if name2src.get(n, 1) == 1 else w_mcp
        rows.append((i, mrow, sw))
    return rows, n_drop


def train_one(rows, Xz, Yt, eval_names, Mask, SUP, ni, labels, name2K, name2gt, a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GateMLP(8, a.hidden, a.act, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    Xz_t = torch.tensor(Xz, dtype=torch.float32); Yt_t = torch.tensor(Yt, dtype=torch.float32)
    best_f, best_state = -1.0, None
    idx_all = np.arange(len(rows))
    for ep in range(a.epochs):
        model.train(); np.random.shuffle(idx_all)
        t0 = time.time(); run_loss, nb = 0.0, 0
        for bs in range(0, len(idx_all), a.batch):
            bidx = idx_all[bs:bs + a.batch]; opt.zero_grad()
            loss = torch.zeros(()); wsum = 0.0
            for j in bidx:
                i, mrow, sw = rows[j]; m3 = torch.tensor(mrow, dtype=torch.float32)
                s = (model(Xz_t[i]) / a.T).masked_fill(m3 == 0, -1e9)
                logp = torch.log_softmax(s, dim=0)
                q = torch.softmax((Yt_t[i] / a.Ty).masked_fill(m3 == 0, -1e9), dim=0)
                loss = loss + sw * (-(q * logp * m3).sum()); wsum += sw
            nl = loss / max(wsum, 1e-6); nl.backward(); opt.step()
            run_loss += float(nl.detach()); nb += 1
        dt = time.time() - t0
        do_eval = ((ep + 1) % a.eval_every == 0) or (ep + 1 == a.epochs)
        if do_eval:
            te = time.time()
            f = eval_ewf1(model, Xz, Mask, SUP, eval_names, ni, labels, name2K, name2gt, a.T)
            mark = ''
            if f > best_f:
                best_f = f
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                mark = ' *best*'
            print('[seed %d][ep %3d/%3d] loss=%.4f (%.1fs train, %.1fs eval) trainEW-F1=%.4f%s'
                  % (seed, ep + 1, a.epochs, run_loss / max(nb, 1), dt, time.time() - te, f, mark), flush=True)
        else:
            print('[seed %d][ep %3d/%3d] loss=%.4f (%.1fs train)'
                  % (seed, ep + 1, a.epochs, run_loss / max(nb, 1), dt), flush=True)
    return best_f, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='合并后含 human+mcp 的 *_logits_merged_human_mcp.npz')
    ap.add_argument('--human', default=config.PATH_TO_LABEL['Human'])
    ap.add_argument('--mcp', default=config.PATH_TO_LABEL['MERCaptionPlus'])
    ap.add_argument('--out', default='output/_cal/gate_params_mlp_full.json')
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--act', choices=['gelu', 'relu'], default='gelu')
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--eval-every', dest='eval_every', type=int, default=5)
    ap.add_argument('--w-mcp', dest='w_mcp', type=float, default=0.25)
    ap.add_argument('--Ty', type=float, default=0.1)
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--keep-noise', dest='keep_noise', type=int, default=1,
                    help='1=保留全部(含mcp全错伪标签, 用户选定)；0=丢弃约418条纯噪声')
    ap.add_argument('--min-train', dest='min_train', type=int, default=300)
    a = ap.parse_args()

    print('[cfg] npz=%s' % a.npz)
    print('[cfg] out=%s  epochs=%d seeds=%d eval_every=%d keep_noise=%d w_mcp=%.2f hidden=%d act=%s'
          % (a.out, a.epochs, a.seeds, a.eval_every, a.keep_noise, a.w_mcp, a.hidden, a.act), flush=True)

    name2logits, labels, neutral_vec, tau, vhash, npzp = load_logits(a.npz)
    p0 = neutral_p0(neutral_vec, tau)
    name2gt, name2K, name2src = load_gt_merged(a.human, a.mcp)
    bank = build_wheel_bank(labels, name2gt)

    have = {str(k) for k, v in name2logits.items() if isinstance(v, dict) and 'status' not in v}
    all_gt = set(map(str, name2gt))
    full_names = sorted(have & all_gt)                     # 全量：有 logits 且有 GT，无任何切分
    n_h = sum(1 for n in full_names if name2src.get(n, 1) == 1)
    n_m = len(full_names) - n_h
    print('[FULL] have=%d all_gt=%d  => trainable(full)=%d  (NO holdout / NO test)'
          % (len(have), len(all_gt), len(full_names)))
    print('[FULL] source: human(real)=%d  mcp(pseudo)=%d' % (n_h, n_m), flush=True)
    assert len(full_names) >= a.min_train, '[ABORT] 全量样本不足(%d)' % len(full_names)

    X, Mask, Yt, SUP, keep, ni = build_arrays(full_names, name2logits, labels, neutral_vec, p0, name2gt, name2K, bank)
    tri = [ni[n] for n in full_names if n in ni]
    mu, sd = zfit(X, Mask, tri); pi = empirical_prior(X, Mask, Yt, tri)
    Xz = (X - mu) / sd

    rows, n_drop = build_rows(full_names, ni, Mask, Yt, name2src, a.w_mcp, keep_noise=bool(a.keep_noise))
    dist = {1: 0, 2: 0, 3: 0}
    for (i, mrow, sw) in rows: dist[int(mrow.sum())] = dist.get(int(mrow.sum()), 0) + 1
    print('[rows] 训练样本=%d  (噪声保留=%s, 丢弃=%d)  可用模态 1/2/3 = %d/%d/%d'
          % (len(rows), bool(a.keep_noise), n_drop, dist.get(1, 0), dist.get(2, 0), dist.get(3, 0)), flush=True)

    eval_names = full_names                                # 无验证集 → 用训练集自身 EW-F1 定参
    results = []
    for sd_i in range(a.seeds):
        bf, bs = train_one(rows, Xz, Yt, eval_names, Mask, SUP, ni, labels, name2K, name2gt, a, seed=sd_i)
        print('[seed %d] best trainset EW-F1=%.4f' % (sd_i, bf), flush=True)
        results.append((bf, bs))
    fs = [r[0] for r in results]
    print('[multi-seed] trainset EW-F1 mean=%.4f std=%.4f max=%.4f' % (np.mean(fs), np.std(fs), np.max(fs)))
    best_f, best_state = max(results, key=lambda r: r[0])

    W1 = best_state['fc1.weight'].numpy(); b1 = best_state['fc1.bias'].numpy()
    W2 = best_state['fc2.weight'].numpy(); b2 = best_state['fc2.bias'].numpy()
    obj = {'gate_type': 'mlp',
           'mlp': {'W': [W1.tolist(), W2.tolist()], 'B': [b1.tolist(), b2.tolist()],
                   'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist(), 'act': a.act},
           'T': float(a.T), 'pi': {m: float(pi[m]) for m in MODS},
           'tau': float(tau), 'vocab_hash': vhash,
           'calib': {'gt_csv': '%s+%s' % (os.path.basename(a.human), os.path.basename(a.mcp)),
                     'split': 'full_no_holdout_no_test',
                     'n_train': len(rows), 'n_human': n_h, 'n_mcp': n_m,
                     'noise': 'kept' if a.keep_noise else 'dropped', 'n_noise_dropped': n_drop,
                     'src_npz': os.path.basename(a.npz),
                     'best_trainset_ewf1': round(float(best_f), 4),
                     'ewf1_mean': round(float(np.mean(fs)), 4), 'ewf1_std': round(float(np.std(fs)), 4),
                     'seeds': a.seeds, 'epochs': a.epochs, 'hidden': a.hidden, 'act': a.act}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(obj, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[saved] %s  best_trainset_ewf1=%.4f mean=%.4f' % (a.out, best_f, np.mean(fs)), flush=True)


if __name__ == '__main__':
    main()
