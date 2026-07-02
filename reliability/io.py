# -*- coding: utf-8 -*-
"""读 *_logits.npz / human GT / 划分 / gate 参数存档与加载。"""
import os, glob, json, hashlib
import numpy as np
import pandas as pd
import config
from toolkit.utils.functions import string_to_list
from . import MODS

HUMAN_GT_CSV = config.PATH_TO_LABEL.get('Human', '/opt/data/wlcc/mer2026-data/track2_train_human.csv')
TEST_CSV = os.path.join(config.DATA_DIR['MER2026'], 'track2_test.csv')


def load_logits(npz_path=None):
    """→ (name2logits, labels:list, neutral_vec:np.ndarray, tau:float, vocab_hash:str, npz_path)"""
    if npz_path is None:
        npz_path = sorted(glob.glob('output/results-emosync/*/*_logits.npz'))[-1]
    d = np.load(npz_path, allow_pickle=True)
    name2logits = d['name2logits'].item()
    labels = list(d['candidate_labels'])
    neutral = d['neutral'].item()                       # 0 维 object 数组包 dict {label: score}
    neutral_vec = np.array([neutral.get(l, 0.0) for l in labels], dtype=float)
    tau = float(d['tau']) if 'tau' in d.files else 1.0
    vocab_hash = hashlib.md5(('|'.join(map(str, labels))).encode()).hexdigest()[:12]
    return name2logits, labels, neutral_vec, tau, vocab_hash, npz_path


def load_gt(csv_path=None):
    """读标定 GT（默认 human 1532）→ (name2gt:{name:'a, b'}, name2K:{name:K})"""
    if csv_path is None:
        csv_path = HUMAN_GT_CSV
    assert os.path.exists(csv_path), '[gt] 标定 GT 文件不存在: %s' % csv_path
    df = pd.read_csv(csv_path)
    name2gt, name2K = {}, {}
    for _, r in df.iterrows():
        gl = string_to_list(r['openset'])
        name2gt[str(r['name'])] = ', '.join(gl)
        name2K[str(r['name'])] = max(1, len(gl))
    return name2gt, name2K


def covered(name2gt, name2logits):
    """既有 GT 又有非跳过 logits 的样本（CV/B0 用）"""
    ok = {str(k) for k, v in name2logits.items() if isinstance(v, dict) and 'status' not in v}
    return sorted(set(map(str, name2gt)) & ok)


def split_train_eval(name2gt, name2logits, test_csv=None):
    """协议 A holdout：train = human∖test ∩ 有logits；eval = test ∩ 有logits ∩ 有GT。test⊆human 故零泄漏。"""
    if test_csv is None:
        test_csv = TEST_CSV
    have = {str(k) for k, v in name2logits.items() if isinstance(v, dict) and 'status' not in v}
    tn = set(pd.read_csv(test_csv)['name'].astype(str))
    gtn = set(map(str, name2gt))
    train = sorted((gtn - tn) & have)
    eval_ = sorted(tn & have & gtn)
    return train, eval_


def save_gate(path, gp, pi, tau, vocab_hash, calib_meta):
    b = gp['b']
    b_dict = {m: float(b[i]) for i, m in enumerate(MODS)} if not isinstance(b, dict) \
        else {k: float(v) for k, v in b.items()}
    obj = {
        'w': np.asarray(gp['w']).tolist(),
        'b': b_dict,
        'mu': np.asarray(gp['mu']).tolist(),
        'sd': np.asarray(gp['sd']).tolist(),
        'T': float(gp['T']),
        'pi': {m: float(pi[m]) for m in MODS},
        'tau': float(tau),
        'vocab_hash': vocab_hash,
        'feat_dim': int(len(gp['w'])),
        'calib': calib_meta,                  # {gt_csv, split, n_train, n_eval}
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    return path


def load_gate(path, tau=None, vocab_hash=None):
    g = json.load(open(path, encoding='utf-8'))
    if tau is not None and vocab_hash is not None:
        assert abs(g['tau'] - tau) < 1e-6 and g['vocab_hash'] == vocab_hash, \
            '[gate] tau/词表 hash 不匹配——配置漂移，请重新标定'
    if 'mlp' in g:
        ml = g['mlp']
        assert ml.get('act') in ('relu', 'gelu'), '[gate] 未知激活: %s' % ml.get('act')
        gp = {'mlp': {'W': [np.array(x, dtype=float) for x in ml['W']],
                      'B': [np.array(x, dtype=float) for x in ml['B']],
                      'mu': np.array(ml['mu']), 'sd': np.array(ml['sd']),
                      'act': ml['act']}, 'T': g['T']}
        return gp, g['pi']
    gp = {'w': np.array(g['w']),
          'b': np.array([g['b'][m] for m in MODS]),
          'mu': np.array(g['mu']),
          'sd': np.array(g['sd']),
          'T': g['T']}
    return gp, g['pi']
