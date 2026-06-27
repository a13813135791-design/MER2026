# -*- coding: utf-8 -*-
"""合并两份 *_logits.npz（如 1200 + 332 → 1532）。校验词表/中性分一致、样本基本不重叠。
用法: python merge_logits.py --base <1200.npz> --add <332.npz> --out <1532.npz>
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--add', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    db = np.load(a.base, allow_pickle=True)
    da = np.load(a.add, allow_pickle=True)
    lb = list(db['candidate_labels'])
    la = list(da['candidate_labels'])
    assert lb == la, '[merge] candidate_labels 不一致，拒绝合并'
    nb = dict(db['name2logits'].item())
    na = dict(da['name2logits'].item())
    overlap = set(nb) & set(na)
    if overlap:
        print('[merge][WARN] %d 个样本重叠，以 add 覆盖 base' % len(overlap))
    merged = dict(nb)
    merged.update(na)
    np.savez(a.out,
             name2logits=np.array(merged, dtype=object),
             candidate_labels=db['candidate_labels'],
             neutral=db['neutral'],
             score_mode=db['score_mode'] if 'score_mode' in db.files else np.array('logits'),
             tau=db['tau'])
    print('[merge] base=%d  add=%d  -> out=%d  (%s)' % (len(nb), len(na), len(merged), a.out))


if __name__ == '__main__':
    main()
