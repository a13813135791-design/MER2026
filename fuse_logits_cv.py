# -*- coding: utf-8 -*-
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2'); os.chdir('/opt/data/wlcc/MER2026_Track2')
import numpy as np, pandas as pd, config
from my_affectgpt.evaluation.wheel import wheel_metric_calculation
from toolkit.utils.functions import string_to_list
import torch
np.random.seed(0); torch.manual_seed(0)

# ---------- load logits ----------
npz = sorted(glob.glob('output/results-emosync/*/*_logits.npz'))[-1]
print('[cv] npz:', os.path.basename(npz))
d = np.load(npz, allow_pickle=True)
name2logits = d['name2logits'].item()
labels = list(d['candidate_labels']); L = len(labels); tau = float(d['tau'])
neutral = d['neutral'].item()
neutral_vec = np.array([neutral.get(l, 0.0) for l in labels], dtype=float)

# ---------- GT ----------
gtdf = pd.read_csv(os.path.join(config.DATA_DIR['MER2026'], 'track2_test.csv'))
name2gt, name2gtn = {}, {}
for _, row in gtdf.iterrows():
    gl = string_to_list(row['openset'])
    name2gt[row['name']] = ', '.join(gl)
    name2gtn[row['name']] = max(1, len(gl))
MODS = ['audio','face','text']; MI = {'audio':0,'face':1,'text':2}

# ---------- wheel mapping (fast per-sample F1 labels) ----------
wm = np.load(config.OUTSIDE_WHEEL_MAPPING, allow_pickle=True)
fmt = wm['format_mapping'].item(); rawm = wm['raw_mapping'].item(); wmw = wm['wheel_map_whole'].item()
WHEELS = ['wheel1','wheel2','wheel3','wheel4','wheel5']
def bw3(label, wheel_map):
    if label not in fmt: return ""
    cand = [r for f in fmt[label] for r in rawm.get(f, [])]
    for lv1 in sorted(cand):
        if lv1 in wheel_map: return wheel_map[lv1]
    return ""
allwords = set(l.lower().strip() for l in labels)
for n in name2gt:
    for w in string_to_list(name2gt[n]): allwords.add(w.lower().strip())
wbk = {wh: {w: bw3(w, wmw[wh]['level1']) for w in allwords} for wh in WHEELS}
def samp_f1(gt_words, pred_words):
    f=[]
    for wh in WHEELS:
        bd = wbk[wh]
        g = set(b for b in (bd.get(w.lower().strip(),"") for w in gt_words) if b)
        p = set(b for b in (bd.get(w.lower().strip(),"") for w in pred_words) if b)
        if not g: continue
        if not p: f.append(0.0); continue
        I=len(g&p); P=I/len(p); R=I/len(g); f.append(0.0 if P+R==0 else 2*P*R/(P+R))
    return float(np.mean(f)) if f else 0.0

# ---------- per-sample support / features / labels ----------
def svec(rec,m):
    md = rec.get(m,{}) if isinstance(rec,dict) else {}
    if not isinstance(md,dict) or 'support' not in md: return None
    v = np.array([md['support'].get(l,0.0) for l in labels]); return v if v.sum()>0 else None
def rvec(rec,m):
    md = rec.get(m,{}) if isinstance(rec,dict) else {}
    rd = md.get('raw',{}) if isinstance(md,dict) else {}
    return np.array([rd.get(l, neutral_vec[i]) for i,l in enumerate(labels)])
p0 = np.exp((neutral_vec-neutral_vec.max())/tau); p0 /= p0.sum()
def feats(sup, raw, cons):
    p = np.sort(sup)[::-1]; mass = sup.sum()
    H = -(sup*np.log(sup+1e-12)).sum(); Hn = H/np.log(L)
    p1, margin = p[0], p[0]-p[1]; gini = 1-(sup**2).sum()/(mass**2+1e-12)
    shiftmax = (raw-neutral_vec).max()
    klprior = (sup*np.log((sup+1e-12)/(p0+1e-12))).sum()
    agree = 0.0 if (cons is None or np.linalg.norm(cons)==0) else float(np.dot(sup,cons)/((np.linalg.norm(sup)*np.linalg.norm(cons))+1e-12))
    return np.array([-Hn, p1, margin, gini, mass, shiftmax, klprior, agree])

names = [n for n in name2logits if n in name2gt]
N = len(names); ni = {n:i for i,n in enumerate(names)}
X = np.zeros((N,3,8)); Mask = np.zeros((N,3)); Yt = np.zeros((N,3)); SUP = {}
for n in names:
    rec = name2logits[n]; sv = {m: svec(rec,m) for m in MODS}
    av = [m for m in MODS if sv[m] is not None]; SUP[n] = sv
    gw = string_to_list(name2gt[n]); K = name2gtn[n]; i = ni[n]
    for m in av:
        others = [sv[k] for k in av if k!=m]
        cons = np.mean(others,axis=0) if others else None
        X[i,MI[m]] = feats(sv[m], rvec(rec,m), cons); Mask[i,MI[m]] = 1
        pred = [labels[t] for t in np.argsort(sv[m])[::-1][:K]]
        Yt[i,MI[m]] = samp_f1(gw, pred)
FEAT_NAMES = ['-熵','top1','margin','gini','mass','PMI位移','KL先验','跨模一致']

# ---------- gate ----------
def fit_gate(tr_idx, mu, sd, T=1.0, Ty=0.1, lam=1e-3, steps=500, lr=0.05):
    Xt = torch.tensor((X[tr_idx]-mu)/sd, dtype=torch.float32)
    Mt = torch.tensor(Mask[tr_idx], dtype=torch.float32)
    Yk = torch.tensor(Yt[tr_idx], dtype=torch.float32)
    w = torch.zeros(8, requires_grad=True); b = torch.zeros(3, requires_grad=True)
    opt = torch.optim.Adam([w,b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        s = (Xt@w + b)/T
        s = s.masked_fill(Mt==0, -1e9)
        logp = torch.log_softmax(s, dim=1)
        q = torch.softmax((Yk/Ty).masked_fill(Mt==0, -1e9), dim=1)
        loss = -(q*logp*Mt).sum(1).mean() + lam*(w*w).sum()
        loss.backward(); opt.step()
    return w.detach().numpy(), b.detach().numpy()

def predict(n, mode, gp=None, pin=None):
    av = [m for m in MODS if Mask[ni[n], MI[m]] == 1]
    if not av: return None
    i = ni[n]
    if mode == 'text':
        if 'text' not in av: return None
        w = {'text':1.0}
    elif mode == 'equal':
        w = {m: 1/len(av) for m in av}
    elif mode == 'tier0':
        r = {m: pin[m]*np.exp(-(-X[i,MI[m],0])/0.25) for m in av}; Z=sum(r.values()); w={m:r[m]/Z for m in av}
    elif mode == 'lcg':
        wv,bv,mu,sd = gp
        s = {m: float(((X[i,MI[m]]-mu)/sd)@wv + bv[MI[m]]) for m in av}
        ex = {m: np.exp(s[m]/1.0) for m in av}; Z=sum(ex.values()); w={m:ex[m]/Z for m in av}
    fused = np.zeros(L)
    for m,wmt in w.items(): fused += wmt*SUP[n][m]
    K = name2gtn[n]
    return ', '.join(labels[t] for t in np.argsort(fused)[::-1][:K])

# ---------- 5-fold ----------
rng = np.random.RandomState(0); order = names[:]; rng.shuffle(order)
folds = [order[i::5] for i in range(5)]
oof = {k:{} for k in ['text','equal','tier0','lcg']}
wsum = np.zeros(8)
for fi in range(5):
    te = folds[fi]; tr = [n for j in range(5) if j!=fi for n in folds[j]]
    tri = np.array([ni[n] for n in tr])
    pi = {m: float(np.mean([Yt[ni[n],MI[m]] for n in tr if Mask[ni[n],MI[m]]==1])) for m in MODS}
    Z = sum(pi.values()); pin = {m: pi[m]/Z for m in MODS}
    rows = X[tri][Mask[tri]==1]; mu = rows.mean(0); sd = rows.std(0); sd[sd<1e-6]=1.0
    wv,bv = fit_gate(tri, mu, sd); wsum += wv
    for n in te:
        for mode in ['text','equal','tier0']:
            r = predict(n, mode, pin=pin)
            if r is not None: oof[mode][n]=r
        r = predict(n, 'lcg', gp=(wv,bv,mu,sd))
        if r is not None: oof['lcg'][n]=r

print('\n==== 5折交叉验证 OOF EW-F1 (K=gt, 5轮平均) ====')
base=None
for meth,lab in [('text','单模态 text  '),('equal','等权平均     '),('tier0','Tier0 法子1x3x6'),('lcg','LCG 学习门控 ')]:
    pr = oof[meth]; common=[n for n in pr if n in name2gt]
    f,p,r = wheel_metric_calculation(name2gt={n:name2gt[n] for n in common},
                                     name2pred={n:pr[n] for n in common}, inter_print=False)
    tag = '' if base is None else ('  (%+.2f vs text)'%(f*100-base))
    if meth=='text': base=f*100
    print('  %s n=%d  EW-F1=%.2f  P=%.2f R=%.2f%s'%(lab,len(common),f*100,p*100,r*100,tag))
print('\n[gate] 学到的特征权重(5折平均, 越大越被重视):')
for nm,v in sorted(zip(FEAT_NAMES, wsum/5), key=lambda x:-abs(x[1])):
    print('    %-8s %+.3f'%(nm, v))
print('[cv] done.')
