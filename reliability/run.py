# -*- coding: utf-8 -*-
"""编排：协议 B0(cv5 OOF) 与 协议 A(holdout) 的标定→融合→评测→消融→落盘。
被 calibrate_gate.py(入口①, emit=False) 与 fuse_logits.py(入口②, emit=True) 共用。"""
import os, json, argparse
import numpy as np
from . import MODS, MI
from .io import (load_logits, load_gt, covered, split_train_eval,
                 save_gate, HUMAN_GT_CSV)
from .features import neutral_p0, FEAT_NAMES
from .metrics import build_wheel_bank, eval_fused
from .calibrate import build_arrays, empirical_prior, zfit, fit_gate
from .weights import reliability_weights
from .fuse import fuse, read_topk

MODES = ['text', 'equal', 'prior', 'tier0', 'lcg']
DEF_OUTDIR = ('output/results-emosync/'
              'mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184/'
              'mer2026-v3-result')
DEF_BASE = 'checkpoint_000035_loss_0.716'


def build_args(**kw):
    a = argparse.Namespace(npz=None, gt=None, split='cv5', gate=None,
                           T=1.0, Ty=0.1, lam=1e-3, emit=True,
                           outdir=DEF_OUTDIR, base=DEF_BASE)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def predict_all(names, ni, X, Mask, SUP, labels, name2K, mode, gp=None, pi=None):
    name2pred, name2w = {}, {}
    for n in names:
        if n not in ni:
            continue
        i = ni[n]
        w = reliability_weights(X[i], Mask[i], mode, gp=gp, pi=pi)
        if not w:
            continue
        sup_by_m = {m: SUP[n][m] for m in MODS if SUP[n].get(m) is not None}
        fused, U = fuse(sup_by_m, w, labels)
        name2pred[n] = ', '.join(read_topk(fused, labels, name2K[n]))
        name2w[n] = {m: round(float(wm), 4) for m, wm in w.items()}
    return name2pred, name2w


def ablation(names, ni, X, Mask, SUP, labels, name2K, name2gt, gp, pi):
    table = {}
    for mode in MODES:
        pr, _ = predict_all(names, ni, X, Mask, SUP, labels, name2K, mode, gp=gp, pi=pi)
        table[mode] = eval_fused(pr, name2gt)
    return table


def print_table(tag, table):
    print('\n==== [%s] 融合 EW-F1 (K=gt, 5轮平均) ====' % tag)
    base = table.get('text', (0, 0, 0, 0))[1]
    for mode in MODES:
        n, f, p, r = table[mode]
        d = '' if mode == 'text' else '  (%+.2f vs text)' % (f - base)
        print('  %-7s n=%-4d EW-F1=%.2f  P=%.2f R=%.2f%s' % (mode, n, f, p, r, d))


def emit(args, name2pred, name2w, name2gt, tag, gate_path, table, calib):
    os.makedirs(args.outdir, exist_ok=True)
    jpath = os.path.join(args.outdir, '%s_v3fused_%s.json' % (args.base, tag))
    obj = {'_meta': {'tag': tag, 'gate': gate_path, 'calib': calib,
                     'ablation': {m: {'n': table[m][0], 'ew_f1': round(table[m][1], 2),
                                      'P': round(table[m][2], 2), 'R': round(table[m][3], 2)}
                                  for m in MODES}},
           'predictions': {n: {'pred': name2pred[n], 'weights': name2w.get(n, {})}
                           for n in name2pred}}
    json.dump(obj, open(jpath, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    npath = os.path.join(args.outdir, '%s_v3fused_%s-openset.npz' % (args.base, tag))
    fn = np.array(list(name2pred.keys()), dtype=object)
    fi = np.array([name2pred[n] for n in name2pred], dtype=object)
    np.savez(npath, filenames=fn, fileitems=fi)
    print('[emit:%s] preds=%d  json=%s' % (tag, len(name2pred), jpath))
    print('[emit:%s] openset(npz for run_evaluation_emosync)=%s' % (tag, npath))
    return jpath, npath


def _prep(data, gt):
    name2logits, labels, neutral_vec, tau, vhash, npz = data
    name2gt, name2K = load_gt(gt)
    p0 = neutral_p0(neutral_vec, tau)
    bank = build_wheel_bank(labels, name2gt)
    return name2logits, labels, neutral_vec, tau, vhash, npz, name2gt, name2K, p0, bank


def run_holdout(args, data):
    (name2logits, labels, neutral_vec, tau, vhash, npz,
     name2gt, name2K, p0, bank) = _prep(data, args.gt)
    train, eval_ = split_train_eval(name2gt, name2logits)
    print('[holdout] train(human∖test)=%d  eval(test)=%d' % (len(train), len(eval_)))
    if len(train) < 20 or len(eval_) < 20:
        print('[holdout][WARN] 划分样本过少，logits 可能未覆盖 1532（需先扩集）。')
    alln = sorted(set(train) | set(eval_))
    X, Mask, Yt, SUP, keep, ni = build_arrays(alln, name2logits, labels, neutral_vec, p0,
                                              name2gt, name2K, bank)
    tri = [ni[n] for n in train if n in ni]
    evn = [n for n in eval_ if n in ni]
    pi = empirical_prior(X, Mask, Yt, tri)
    mu, sd = zfit(X, Mask, tri)
    gp = fit_gate(X, Mask, Yt, tri, mu, sd, T=args.T, Ty=args.Ty, lam=args.lam)
    calib = {'gt_csv': args.gt or HUMAN_GT_CSV, 'split': 'holdout',
             'n_train': len(tri), 'n_eval': len(evn)}
    gate_path = args.gate or os.path.join('output', '_cal', 'gate_params_a.json')
    save_gate(gate_path, gp, pi, tau, vhash, calib)
    print('[holdout] gate saved: %s   pi=%s' % (gate_path, {m: round(pi[m], 4) for m in MODS}))
    table = ablation(evn, ni, X, Mask, SUP, labels, name2K, name2gt, gp, pi)
    print_table('A/holdout eval=%d' % len(evn), table)
    _feat_weights(gp['w'])
    if args.emit:
        n2p, n2w = predict_all(evn, ni, X, Mask, SUP, labels, name2K, 'lcg', gp=gp, pi=pi)
        emit(args, n2p, n2w, name2gt, 'a', gate_path, table, calib)
    return table


def run_cv5(args, data):
    (name2logits, labels, neutral_vec, tau, vhash, npz,
     name2gt, name2K, p0, bank) = _prep(data, args.gt)
    names = covered(name2gt, name2logits)
    print('[cv5] covered(有GT且有logits)=%d' % len(names))
    X, Mask, Yt, SUP, keep, ni = build_arrays(names, name2logits, labels, neutral_vec, p0,
                                              name2gt, name2K, bank)
    rng = np.random.RandomState(0)
    order = keep[:]
    rng.shuffle(order)
    folds = [order[i::5] for i in range(5)]
    oof = {mode: {} for mode in MODES}
    oof_w = {}
    wsum = np.zeros(8)
    pi_acc = {m: [] for m in MODS}
    gp_last = None
    for fi in range(5):
        te = folds[fi]
        tr = [n for j in range(5) if j != fi for n in folds[j]]
        tri = [ni[n] for n in tr]
        pi = empirical_prior(X, Mask, Yt, tri)
        for m in MODS:
            pi_acc[m].append(pi[m])
        mu, sd = zfit(X, Mask, tri)
        gp = fit_gate(X, Mask, Yt, tri, mu, sd, T=args.T, Ty=args.Ty, lam=args.lam)
        gp_last = gp
        wsum += gp['w']
        for mode in MODES:
            pr, w = predict_all(te, ni, X, Mask, SUP, labels, name2K, mode, gp=gp, pi=pi)
            oof[mode].update(pr)
            if mode == 'lcg':
                oof_w.update(w)
    table = {mode: eval_fused(oof[mode], name2gt) for mode in MODES}
    print_table('B0/cv5-OOF n=%d' % len(names), table)
    pi_avg = {m: float(np.mean(pi_acc[m])) for m in MODS}
    _feat_weights(wsum / 5.0)
    gp_avg = dict(gp_last)
    gp_avg['w'] = wsum / 5.0
    calib = {'gt_csv': args.gt or HUMAN_GT_CSV, 'split': 'cv5', 'n_train': 'OOF-5fold', 'n_eval': len(names)}
    gate_path = args.gate or os.path.join('output', '_cal', 'gate_params_b0.json')
    save_gate(gate_path, gp_avg, pi_avg, tau, vhash, calib)
    print('[cv5] gate(avg) saved: %s   pi=%s' % (gate_path, {m: round(pi_avg[m], 4) for m in MODS}))
    if args.emit:
        emit(args, oof['lcg'], oof_w, name2gt, 'b0', gate_path, table, calib)
    return table


def _feat_weights(wvec):
    print('[gate] 学到的特征权重(越大越被重视):')
    for nm, v in sorted(zip(FEAT_NAMES, wvec), key=lambda x: -abs(x[1])):
        print('    %-9s %+.3f' % (nm, v))


def run(args):
    data = load_logits(args.npz)
    print('[v3] npz=%s  samples=%d  candidates=%d  tau=%.2f  split=%s'
          % (os.path.basename(data[5]), len(data[0]), len(data[1]), data[3], args.split))
    if args.split == 'holdout':
        return run_holdout(args, data)
    return run_cv5(args, data)
