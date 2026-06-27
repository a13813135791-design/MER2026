# -*- coding: utf-8 -*-
"""入口②（v3 升级）：可靠性加权融合 + 评测 + 消融，并落盘融合预测。
吃 *_logits.npz + human GT → 逐样本可靠性权重 → 加权线性池 → TopK(K=|GT|) → EW-F1。
  python fuse_logits.py --split cv5     --ablation [--npz X] [--gt human.csv] [--outdir D]
  python fuse_logits.py --split holdout --ablation [--npz X] [--gt human.csv] [--outdir D]
消融同表打印 text/equal/prior/tier0/lcg；--split cv5 报 OOF，holdout 报留出 eval。
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2')
os.chdir('/opt/data/wlcc/MER2026_Track2')
from reliability.run import build_args, run, DEF_OUTDIR, DEF_BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=None)
    ap.add_argument('--gt', default=None, help='标定 GT csv，默认 human(1532)')
    ap.add_argument('--split', choices=['cv5', 'holdout'], default='cv5')
    ap.add_argument('--gate', default=None, help='gate_params.json 存档路径')
    ap.add_argument('--mode', default='lcg', help='保留位（落盘的融合预测固定取 lcg）')
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--Ty', type=float, default=0.1)
    ap.add_argument('--lam', type=float, default=1e-3)
    ap.add_argument('--ablation', action='store_true', help='打印 text/equal/prior/tier0/lcg 同表对比')
    ap.add_argument('--outdir', default=DEF_OUTDIR)
    ap.add_argument('--base', default=DEF_BASE)
    a = ap.parse_args()
    args = build_args(npz=a.npz, gt=a.gt, split=a.split, gate=a.gate,
                      T=a.T, Ty=a.Ty, lam=a.lam, emit=True, outdir=a.outdir, base=a.base)
    run(args)
    print('\n[fuse_logits] done.')


if __name__ == '__main__':
    main()
