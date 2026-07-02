# -*- coding: utf-8 -*-
"""eval_mlp_gate_cap.py — v3.2 测试：加载 caption 增强 MLP 门控（mlpcap），在 GT csv 上做可靠性加权融合评测 EW-F1(K=gt)。
纯 numpy 推理（reliability.weights('mlpcap')）；caption 特征用 CLIP 文本塔（复用训练缓存，命中则不加载 CLIP）。
从脚本自身目录定位工程根（自洽到当前 worktree）。同时打印 text/equal/prior/tier0 对照。
  python eval_mlp_gate_cap.py --npz <1532 logits.npz> --gate output/_cal/gate_params_mlp_cap_full.json \
       --gt /opt/data/wlcc/mer2026-data/_ovmerdplus_ovlabel.csv --caption-csv <A> --out answer_v32_ovmerdplus.csv"""
import sys, os, csv, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
from reliability import MODS
from reliability.io import load_logits, load_gt, load_gate, covered
from reliability.features import neutral_p0
from reliability.metrics import build_wheel_bank, eval_fused
from reliability.calibrate import build_arrays, empirical_prior
from reliability.weights import reliability_weights
from reliability.fuse import fuse, read_topk
from reliability.caption import get_cap_features, DEFAULT_CLIP

DEF_NPZ = ('/opt/data/wlcc/MER2026_Track2_1027/output/results-emosync/'
           'mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184/'
           'checkpoint_000035_loss_0.716_logits_1532.npz')
DEF_CAP = '/opt/data/wlcc/mer2026-data/track2_train_human_caption-AffectGPT.csv'


def predict(names, ni, X, Mask, SUP, Cap, labels, name2K, mode, gp, pi):
    n2p = {}
    for n in names:
        if n not in ni: continue
        i = ni[n]
        kw = {'gp': gp, 'pi': pi}
        if mode == 'mlpcap': kw['cap_row'] = Cap[i]
        w = reliability_weights(X[i], Mask[i], mode, **kw)
        if not w: continue
        sup_by_m = {m: SUP[n][m] for m in MODS if SUP[n].get(m) is not None}
        fused, _ = fuse(sup_by_m, w, labels)
        n2p[n] = ', '.join(read_topk(fused, labels, name2K[n]))
    return n2p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=DEF_NPZ)
    ap.add_argument('--gate', default='output/_cal/gate_params_mlp_cap_full.json')
    ap.add_argument('--gt', default='/opt/data/wlcc/mer2026-data/_ovmerdplus_ovlabel.csv')
    ap.add_argument('--caption-csv', dest='caption_csv', default=DEF_CAP)
    ap.add_argument('--out', default='answer_v32_ovmerdplus.csv')
    ap.add_argument('--cap-device', dest='cap_device', default='cpu')
    ap.add_argument('--modes', default='text,equal,prior,tier0,mlpcap')
    a = ap.parse_args()

    name2logits, labels, neutral_vec, tau, vhash, npzp = load_logits(a.npz)
    name2gt, name2K = load_gt(a.gt)
    p0 = neutral_p0(neutral_vec, tau)
    bank = build_wheel_bank(labels, name2gt)
    names = covered(name2gt, name2logits)
    print('[eval] npz=%s gt=%s covered=%d candidates=%d tau=%.2f'
          % (os.path.basename(npzp), os.path.basename(a.gt), len(names), len(labels), tau))
    if not names:
        print('[eval][ABORT] GT 与 logits 交集为空'); return

    X, Mask, Yt, SUP, keep, ni = build_arrays(names, name2logits, labels, neutral_vec, p0, name2gt, name2K, bank)
    gp, pi = load_gate(a.gate, tau=tau, vocab_hash=vhash)
    tri = [ni[n] for n in names if n in ni]
    pi_use = pi if pi else empirical_prior(X, Mask, Yt, tri)

    clip_model = gp['cap'].get('model', DEFAULT_CLIP) if isinstance(gp, dict) and 'cap' in gp else DEFAULT_CLIP
    cache = os.path.join('output', '_cal',
                         'capfeat_clipL14_%s.npz' % os.path.splitext(os.path.basename(a.caption_csv))[0])
    Cap = get_cap_features(keep, a.caption_csv, cache_path=cache, model_path=clip_model, device=a.cap_device)
    print('[cap] dim=%d covered=%d (cache=%s)' % (Cap.shape[1], len(keep), cache))

    modes = [m.strip() for m in a.modes.split(',') if m.strip()]
    print('\n==== [v3.2 %s] 融合 EW-F1 (K=gt) ====' % os.path.basename(a.gt))
    best = None
    for mode in modes:
        try:
            n2p = predict(names, ni, X, Mask, SUP, Cap, labels, name2K, mode, gp, pi_use)
        except Exception as e:
            print('  %-8s [skip] %s' % (mode, repr(e))); continue
        n, f, p, r = eval_fused(n2p, name2gt)
        star = '  <== 打分器(caption-MLP)' if mode == 'mlpcap' else ''
        print('  %-8s n=%-4d EW-F1=%.2f  P=%.2f R=%.2f%s' % (mode, n, f, p, r, star))
        if mode == 'mlpcap': best = n2p
    if best is not None:
        with open(a.out, 'w', newline='', encoding='utf-8') as fo:
            wri = csv.DictWriter(fo, fieldnames=['name', 'openset']); wri.writeheader()
            for nm in best: wri.writerow({'name': nm, 'openset': best[nm]})
        print('[eval] mlpcap 预测已写 -> %s (n=%d)' % (a.out, len(best)))


if __name__ == '__main__':
    main()
