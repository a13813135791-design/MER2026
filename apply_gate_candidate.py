# -*- coding: utf-8 -*-
"""应用训练好的可靠性门控(gate) 到无标签样本(candidate) → 出预测 name,openset。
区别于 fuse_logits.py：本脚本不标定、不需要 GT，只 load 已存的 gate 对新样本融合读出。
复用队友 reliability.io 的 load_logits / load_gate（正确处理 neutral dict、b 排序）。
  python apply_gate_candidate.py --npz <candidate_logits.npz> --gate <gate_params_1532.json> --out answer.csv [--K 5]
"""
import sys, os, csv, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
os.chdir(_PROJ)
import numpy as np
from reliability import MODS, MI
from reliability.io import load_logits, load_gate
from reliability.features import neutral_p0, consensus_vecs, rvec, feats
from reliability.weights import reliability_weights
from reliability.fuse import fuse, read_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='candidate 的 *_logits.npz')
    ap.add_argument('--gate', required=True, help='gate_params_*.json (训练好的门控)')
    ap.add_argument('--out', required=True, help='输出 answer.csv')
    ap.add_argument('--K', type=int, default=5, help='每样本预测情绪个数(candidate无GT,固定K)')
    ap.add_argument('--mode', default='lcg', choices=['lcg', 'tier0', 'prior', 'equal', 'text', 'mlp'])
    a = ap.parse_args()

    name2logits, labels, neutral_vec, tau, vhash, npzp = load_logits(a.npz)
    p0 = neutral_p0(neutral_vec, tau)
    gp, pi = load_gate(a.gate, tau=tau, vocab_hash=vhash)

    rows, miss, ok = [], 0, 0
    for name, rec in name2logits.items():
        if not isinstance(rec, dict) or 'status' in rec:
            rows.append({'name': name, 'openset': 'neutral'}); miss += 1; continue
        sv, av, cons = consensus_vecs(rec, labels)
        if not av:
            rows.append({'name': name, 'openset': 'neutral'}); miss += 1; continue
        feat_row = np.zeros((3, 8)); mask_row = np.zeros(3)
        for m in av:
            feat_row[MI[m]] = feats(sv[m], rvec(rec, m, labels, neutral_vec), neutral_vec, p0, cons[m])
            mask_row[MI[m]] = 1
        w = reliability_weights(feat_row, mask_row, a.mode, gp=gp, pi=pi)
        if not w:
            rows.append({'name': name, 'openset': 'neutral'}); miss += 1; continue
        sup_by_m = {m: sv[m] for m in av}
        fused, U = fuse(sup_by_m, w, labels)
        pred = read_topk(fused, labels, a.K)
        rows.append({'name': name, 'openset': ','.join(pred)}); ok += 1

    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        wri = csv.DictWriter(f, fieldnames=['name', 'openset'])
        wri.writeheader(); wri.writerows(rows)
    print('[apply] total=%d ok=%d fallback(neutral)=%d K=%d mode=%s vhash=%s -> %s'
          % (len(rows), ok, miss, a.K, a.mode, vhash, a.out))
    for r in rows[:5]:
        print('  ', r['name'], '->', r['openset'])


if __name__ == '__main__':
    main()
