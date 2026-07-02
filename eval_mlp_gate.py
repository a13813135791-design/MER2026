# -*- coding: utf-8 -*-
"""v3.1 测试脚本：加载已训练门控（默认 MLP full），在给定 GT csv 上做可靠性加权融合并评测 EW-F1(K=gt)。
纯复用 reliability 原语；从脚本自身目录定位工程根（自洽到 _1027，不依赖兄弟目录）。
  python eval_mlp_gate.py --npz <caption版 *_logits.npz> --gate output/_cal/gate_params_mlp_full.json \
                          --gt /opt/data/wlcc/mer2026-data/_ovmerdplus_ovlabel.csv --out answer_v31_ovmerdplus.csv
"""
import sys, os, csv, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ); os.chdir(_PROJ)
import numpy as np
from reliability import MODS
from reliability.io import load_logits, load_gt, load_gate, covered
from reliability.features import neutral_p0
from reliability.metrics import build_wheel_bank, eval_fused
from reliability.calibrate import build_arrays, empirical_prior
from reliability.run import predict_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='阶段一 caption 版 *_logits.npz')
    ap.add_argument('--gate', required=True, help='训练好的门控 json (gate_params_mlp_full.json)')
    ap.add_argument('--gt', required=True, help='测试 GT csv (name,openset)')
    ap.add_argument('--out', default='answer_v31_ovmerdplus.csv', help='mlp 预测输出 csv')
    ap.add_argument('--modes', default='text,equal,prior,tier0,mlp', help='对照模式(逗号分隔)')
    a = ap.parse_args()

    name2logits, labels, neutral_vec, tau, vhash, npzp = load_logits(a.npz)
    name2gt, name2K = load_gt(a.gt)
    p0 = neutral_p0(neutral_vec, tau)
    bank = build_wheel_bank(labels, name2gt)

    names = covered(name2gt, name2logits)          # 有GT且有非跳过 logits 的交集
    print('[eval] npz=%s  gt=%s  covered=%d  candidates=%d  tau=%.2f'
          % (os.path.basename(npzp), os.path.basename(a.gt), len(names), len(labels), tau))
    if not names:
        print('[eval][ABORT] GT 与 logits 交集为空'); return

    X, Mask, Yt, SUP, keep, ni = build_arrays(names, name2logits, labels, neutral_vec, p0,
                                              name2gt, name2K, bank)
    gp, pi = load_gate(a.gate, tau=tau, vocab_hash=vhash)
    tri = [ni[n] for n in names if n in ni]
    pi_use = pi if pi else empirical_prior(X, Mask, Yt, tri)   # prior/tier0 兜底

    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    print('\n==== [v3.1 %s] 融合 EW-F1 (K=gt) ====' % os.path.basename(a.gt))
    best = None
    for mode in modes:
        try:
            n2p, _ = predict_all(names, ni, X, Mask, SUP, labels, name2K, mode, gp=gp, pi=pi_use)
        except Exception as e:
            print('  %-7s [skip] %s' % (mode, e)); continue
        n, f, p, r = eval_fused(n2p, name2gt)
        star = '  <== 打分器(MLP)' if mode == 'mlp' else ''
        print('  %-7s n=%-4d EW-F1=%.2f  P=%.2f R=%.2f%s' % (mode, n, f, p, r, star))
        if mode == 'mlp':
            best = n2p
    if best is not None:
        with open(a.out, 'w', newline='', encoding='utf-8') as fo:
            wri = csv.DictWriter(fo, fieldnames=['name', 'openset']); wri.writeheader()
            for nm in best:
                wri.writerow({'name': nm, 'openset': best[nm]})
        print('[eval] mlp 预测已写 -> %s (n=%d)' % (a.out, len(best)))


if __name__ == '__main__':
    main()
