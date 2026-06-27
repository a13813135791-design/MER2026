# -*- coding: utf-8 -*-
"""入口①：在 human(1532) 上标定可靠性门控 → 产出 gate_params.json（不落融合预测）。
用法:
  python calibrate_gate.py --split cv5     [--npz X] [--gt human.csv] [--out gate.json]
  python calibrate_gate.py --split holdout [--npz X] [--gt human.csv] [--out gate.json]
"""
import sys, os, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2')
os.chdir('/opt/data/wlcc/MER2026_Track2')
from reliability.run import build_args, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=None)
    ap.add_argument('--gt', default=None, help='标定 GT csv，默认 config.PATH_TO_LABEL["Human"]')
    ap.add_argument('--split', choices=['cv5', 'holdout'], default='cv5')
    ap.add_argument('--T', type=float, default=1.0)
    ap.add_argument('--Ty', type=float, default=0.1)
    ap.add_argument('--lam', type=float, default=1e-3)
    ap.add_argument('--out', default=None, help='gate_params.json 路径')
    a = ap.parse_args()
    args = build_args(npz=a.npz, gt=a.gt, split=a.split, gate=a.out,
                      T=a.T, Ty=a.Ty, lam=a.lam, emit=False)
    run(args)
    print('\n[calibrate_gate] done.')


if __name__ == '__main__':
    main()
