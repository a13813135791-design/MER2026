# -*- coding: utf-8 -*-
"""train_mlp_gate_cap.py — v3.2 全量训练 caption 增强的逐模态共享 MLP 门控（torch）。
每模态输入 = concat(z(φ_m)[8], z(cap)[D])，cap 由本地 CLIP 文本塔编码（reliability.caption）。
用全部 human(1532) 训练（无 holdout / 无 test，用户指定全量），listwise KL：softmax_m(s/T) 逼近 softmax(y/Ty)。
多 seed 按训练集自身 EW-F1 存最优 → gate_params_mlp_cap_full.json（gate_type='mlpcap'）。
末尾做 numpy(reliability.weights.mlpcap) vs torch 前向一致性抽查。训练侧独占 torch。"""
import sys, os, json, argparse, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, torch, torch.nn as nn
import config
from reliability import MODS, MI
from reliability.io import load_logits, load_gt
from reliability.features import neutral_p0
from reliability.metrics import build_wheel_bank, eval_fused
from reliability.calibrate import build_arrays, empirical_prior, zfit
from reliability.fuse import fuse, read_topk
from reliability.caption import get_cap_features, CAP_METHOD, DEFAULT_CLIP

DEF_NPZ = ('/opt/data/wlcc/MER2026_Track2_1027/output/results-emosync/'
           'mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184/'
           'checkpoint_000035_loss_0.716_logits_1532.npz')
DEF_CAP = '/opt/data/wlcc/mer2026-data/track2_train_human_caption-AffectGPT.csv'


class GateMLPCap(nn.Module):
    """逐模态共享两层 MLP：din=8+D → h → 1。"""
    def __init__(self, din, h=64, act='gelu', dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(din, h); self.fc2 = nn.Linear(h, 1)
        self.drop = nn.Dropout(dropout); self.act = act
    def _a(self, x):
        return torch.nn.functional.gelu(x, approximate='tanh') if self.act == 'gelu' else torch.relu(x)
    def forward(self, x):                     # x:[...,din] → [...]
        return self.fc2(self.drop(self._a(self.fc1(x)))).squeeze(-1)


def make_inputs(Xz_i, capz_i):
    """[3,8] + [D] → [3,8+D]（cap 对三模态广播；φ 在前、cap 在后）。"""
    cap = capz_i.unsqueeze(0).expand(Xz_i.shape[0], -1)
    return torch.cat([Xz_i, cap], dim=1)


def eval_ewf1(model, Xz_t, Capz_t, Mask, SUP, names, ni, labels, name2K, name2gt, T):
    model.eval(); n2p = {}
    with torch.no_grad():
        for n in names:
            if n not in ni: continue
            i = ni[n]; av = [m for m in MODS if Mask[i, MI[m]] == 1]
            if not av: continue
            sfull = model(make_inputs(Xz_t[i], Capz_t[i]))
            s = {m: float(sfull[MI[m]]) for m in av}
            ex = {m: np.exp(s[m] / T) for m in av}; Z = sum(ex.values())
            w = {m: ex[m] / Z for m in av}
            sup_by_m = {m: SUP[n][m] for m in av if SUP[n].get(m) is not None}
            fused, _ = fuse(sup_by_m, w, labels)
            n2p[n] = ', '.join(read_topk(fused, labels, name2K[n]))
    if not n2p: return 0.0
    _, f, _, _ = eval_fused(n2p, name2gt); return f / 100.0


def train_one(rows, Xz_t, Capz_t, Yt_t, Mask, SUP, eval_names, ni, labels, name2K, name2gt, a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GateMLPCap(8 + a.cap_dim, a.hidden, a.act, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best_f, best_state = -1.0, None
    idx = np.arange(len(rows))
    for ep in range(a.epochs):
        model.train(); np.random.shuffle(idx); t0 = time.time(); rl, nb = 0.0, 0
        for bs in range(0, len(idx), a.batch):
            bidx = idx[bs:bs + a.batch]; opt.zero_grad()
            loss = torch.zeros(())
            for j in bidx:
                i, m3np = rows[j]; m3 = torch.tensor(m3np, dtype=torch.float32)
                s = (model(make_inputs(Xz_t[i], Capz_t[i])) / a.T).masked_fill(m3 == 0, -1e9)
                logp = torch.log_softmax(s, dim=0)
                q = torch.softmax((Yt_t[i] / a.Ty).masked_fill(m3 == 0, -1e9), dim=0)
                loss = loss + (-(q * logp * m3).sum())
            nl = loss / max(len(bidx), 1); nl.backward(); opt.step()
            rl += float(nl.detach()); nb += 1
        dt = time.time() - t0
        do_eval = ((ep + 1) % a.eval_every == 0) or (ep + 1 == a.epochs)
        if do_eval:
            te = time.time()
            f = eval_ewf1(model, Xz_t, Capz_t, Mask, SUP, eval_names, ni, labels, name2K, name2gt, a.T)
            mark = ''
            if f > best_f:
                best_f = f; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; mark = ' *best*'
            print('[seed %d][ep %3d/%3d] loss=%.4f (%.1fs train, %.1fs eval) trainEW-F1=%.4f%s'
                  % (seed, ep + 1, a.epochs, rl / max(nb, 1), dt, time.time() - te, f, mark), flush=True)
        else:
            print('[seed %d][ep %3d/%3d] loss=%.4f (%.1fs train)'
                  % (seed, ep + 1, a.epochs, rl / max(nb, 1), dt), flush=True)
    return best_f, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=DEF_NPZ)
    ap.add_argument('--gt', default=config.PATH_TO_LABEL['Human'])
    ap.add_argument('--caption-csv', dest='caption_csv', default=DEF_CAP)
    ap.add_argument('--out', default='output/_cal/gate_params_mlp_cap_full.json')
    ap.add_argument('--cap-device', dest='cap_device', default='cpu')
    ap.add_argument('--clip', default=DEFAULT_CLIP)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--act', choices=['gelu', 'relu'], default='gelu')
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--eval-every', dest='eval_every', type=int, default=5)
    ap.add_argument('--Ty', type=float, default=0.1)
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    print('[cfg] npz=%s' % a.npz)
    print('[cfg] gt=%s cap=%s out=%s' % (os.path.basename(a.gt), os.path.basename(a.caption_csv), a.out))
    print('[cfg] hidden=%d act=%s dropout=%.2f lr=%.1e wd=%.1e epochs=%d seeds=%d T=%.2f Ty=%.2f'
          % (a.hidden, a.act, a.dropout, a.lr, a.wd, a.epochs, a.seeds, a.T, a.Ty), flush=True)

    name2logits, labels, neutral_vec, tau, vhash, npzp = load_logits(a.npz)
    p0 = neutral_p0(neutral_vec, tau)
    name2gt, name2K = load_gt(a.gt)
    bank = build_wheel_bank(labels, name2gt)

    have = {str(k) for k, v in name2logits.items() if isinstance(v, dict) and 'status' not in v}
    full = sorted(have & set(map(str, name2gt)))
    print('[FULL] have=%d gt=%d => train(full)=%d  (NO holdout / NO test；用户指定全量, 532test⊆train)'
          % (len(have), len(name2gt), len(full)), flush=True)

    X, Mask, Yt, SUP, keep, ni = build_arrays(full, name2logits, labels, neutral_vec, p0, name2gt, name2K, bank)
    tri = [ni[n] for n in full if n in ni]
    mu, sd = zfit(X, Mask, tri); pi = empirical_prior(X, Mask, Yt, tri)
    Xz = (X - mu) / sd

    cache = os.path.join('output', '_cal',
                         'capfeat_clipL14_%s.npz' % os.path.splitext(os.path.basename(a.caption_csv))[0])
    print('[cap] encoding via CLIP (%s, device=%s) cache=%s' % (os.path.basename(a.clip), a.cap_device, cache), flush=True)
    Cap = get_cap_features(keep, a.caption_csv, cache_path=cache, model_path=a.clip, device=a.cap_device)
    a.cap_dim = Cap.shape[1]
    cap_mu = Cap.mean(0); cap_sd = Cap.std(0); cap_sd[cap_sd < 1e-6] = 1.0
    Capz = (Cap - cap_mu) / cap_sd
    print('[cap] dim=%d  → 门控输入维 = 8 + %d = %d' % (a.cap_dim, a.cap_dim, 8 + a.cap_dim), flush=True)

    Xz_t = torch.tensor(Xz, dtype=torch.float32)
    Capz_t = torch.tensor(Capz, dtype=torch.float32)
    Yt_t = torch.tensor(Yt, dtype=torch.float32)
    rows = [(ni[n], Mask[ni[n]].copy()) for n in full if n in ni]
    dist = {1: 0, 2: 0, 3: 0}
    for (i, mr) in rows: dist[int(mr.sum())] = dist.get(int(mr.sum()), 0) + 1
    print('[rows] 训练样本=%d 可用模态 1/2/3 = %d/%d/%d' % (len(rows), dist.get(1, 0), dist.get(2, 0), dist.get(3, 0)), flush=True)

    results = []
    for si in range(a.seeds):
        bf, bstate = train_one(rows, Xz_t, Capz_t, Yt_t, Mask, SUP, full, ni, labels, name2K, name2gt, a, seed=si)
        print('[seed %d] best trainset EW-F1=%.4f' % (si, bf), flush=True)
        results.append((bf, bstate))
    fs = [r[0] for r in results]
    print('[multi-seed] trainset EW-F1 mean=%.4f std=%.4f max=%.4f' % (np.mean(fs), np.std(fs), np.max(fs)))
    best_f, best_state = max(results, key=lambda r: r[0])

    W1 = best_state['fc1.weight'].numpy(); b1 = best_state['fc1.bias'].numpy()
    W2 = best_state['fc2.weight'].numpy(); b2 = best_state['fc2.bias'].numpy()
    obj = {'gate_type': 'mlpcap',
           'mlp': {'W': [W1.tolist(), W2.tolist()], 'B': [b1.tolist(), b2.tolist()],
                   'mu': np.asarray(mu).tolist(), 'sd': np.asarray(sd).tolist(), 'act': a.act},
           'cap': {'mu': cap_mu.tolist(), 'sd': cap_sd.tolist(), 'dim': int(a.cap_dim),
                   'method': CAP_METHOD, 'model': a.clip},
           'T': float(a.T), 'pi': {m: float(pi[m]) for m in MODS},
           'tau': float(tau), 'vocab_hash': vhash,
           'calib': {'gt_csv': os.path.basename(a.gt), 'caption_csv': os.path.basename(a.caption_csv),
                     'split': 'full_1532 (532 test⊆train, 有重叠)', 'n_train': len(rows),
                     'best_trainset_ewf1': round(float(best_f), 4),
                     'ewf1_mean': round(float(np.mean(fs)), 4), 'ewf1_std': round(float(np.std(fs)), 4),
                     'seeds': a.seeds, 'epochs': a.epochs, 'hidden': a.hidden, 'act': a.act,
                     'src_npz': os.path.basename(a.npz)}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(obj, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[saved] %s  best_trainset_ewf1=%.4f mean=%.4f' % (a.out, best_f, np.mean(fs)), flush=True)

    # numpy(reliability.weights.mlpcap) vs torch 前向一致性抽查（校验存/取/推理链）
    try:
        from reliability.io import load_gate
        from reliability.weights import reliability_weights
        gp, _ = load_gate(a.out, tau=tau, vocab_hash=vhash)
        model = GateMLPCap(8 + a.cap_dim, a.hidden, a.act, a.dropout); model.load_state_dict(best_state); model.eval()
        maxd = 0.0
        for n in full[:8]:
            i = ni[n]; av = [m for m in MODS if Mask[i, MI[m]] == 1]
            with torch.no_grad():
                sfull = model(make_inputs(Xz_t[i], Capz_t[i]))
            s = {m: float(sfull[MI[m]]) for m in av}; ex = {m: np.exp(s[m] / a.T) for m in av}; Z = sum(ex.values())
            w_torch = {m: ex[m] / Z for m in av}
            w_np = reliability_weights(X[i], Mask[i], 'mlpcap', gp=gp, cap_row=Cap[i])
            maxd = max(maxd, max(abs(w_torch[m] - w_np[m]) for m in av))
        print('[parity] numpy(weights.mlpcap) vs torch 最大权重差=%.2e (应<1e-5)' % maxd, flush=True)
    except Exception as e:
        print('[parity] 抽查失败:', repr(e), flush=True)


if __name__ == '__main__':
    main()
