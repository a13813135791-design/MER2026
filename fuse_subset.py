# -*- coding: utf-8 -*-
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0,'/opt/data/wlcc/MER2026_Track2'); os.chdir('/opt/data/wlcc/MER2026_Track2')
import numpy as np, pandas as pd, config
from my_affectgpt.evaluation.wheel import wheel_metric_calculation
from toolkit.utils.functions import string_to_list

npz = sorted(glob.glob('output/results-emosync/*/*_logits.npz'))[-1]
d = np.load(npz, allow_pickle=True)
name2logits = d['name2logits'].item(); labels = list(d['candidate_labels']); L=len(labels)
gtdf = pd.read_csv(os.path.join(config.DATA_DIR['MER2026'],'track2_test.csv'))
name2gt={}; name2gtn={}
for _,r in gtdf.iterrows():
    gl=string_to_list(r['openset']); name2gt[r['name']]=', '.join(gl); name2gtn[r['name']]=max(1,len(gl))
MODS=['audio','face','text']
Fpri={'audio':49.21,'face':47.32,'text':52.74}; pi={m:Fpri[m]/sum(Fpri.values()) for m in MODS}

def svec(rec,m):
    md=rec.get(m,{}) if isinstance(rec,dict) else {}
    if not isinstance(md,dict) or 'support' not in md: return None
    v=np.array([md['support'].get(l,0.0) for l in labels]); return v if v.sum()>0 else None
def Hn(v):
    s=v.sum(); p=v/s; H=-(p*np.log(p+1e-12)).sum(); return H/np.log(L)
def trunc(v, mode):
    if mode=='full': return v.copy()
    w=np.zeros(L)
    if mode.startswith('topk'):
        k=int(mode[4:]); idx=np.argsort(v)[::-1][:k]; w[idx]=v[idx]
    else:
        t=float(mode[3:]); mk=v>=t; w[mk]=v[mk]
    return w

def run(mode):
    name2pred={}; Us=[]
    for n in name2logits:
        if n not in name2gt: continue
        rec=name2logits[n]; sv={m:svec(rec,m) for m in MODS}; av=[m for m in MODS if sv[m] is not None]
        if not av: continue
        raw={m: pi[m]*np.exp(-Hn(sv[m])/0.25) for m in av}; Z=sum(raw.values()); w={m:raw[m]/Z for m in av}
        fused=np.zeros(L); U=0.0
        for m in av:
            tv=trunc(sv[m], mode); fused+=w[m]*tv; U+=w[m]*(1.0-tv.sum())
        Us.append(U); K=name2gtn[n]
        name2pred[n]=', '.join(labels[i] for i in np.argsort(fused)[::-1][:K])
    common=list(name2pred.keys())
    f,p,r=wheel_metric_calculation(name2gt={n:name2gt[n] for n in common},
                                   name2pred={n:name2pred[n] for n in common}, inter_print=False)
    return len(common), f*100, float(np.mean(Us))

print("按队友方案1严格实现: 子集截断(不重新归一)+残差, 权重同一Tier-0(T=0.25), 读出TopK(K=gt)")
print("对照基线: 单模态text=52.74")
for mode,lab in [('full','全分布(我们现有)   '),('topk10','子集 topk=10        '),
                 ('topk5','子集 topk=5         '),('topk3','子集 topk=3         '),
                 ('thr0.05','子集 阈值>=0.05     '),('thr0.02','子集 阈值>=0.02     ')]:
    n,f,u=run(mode); print("  %s n=%d  EW-F1=%.2f  平均总不确定性=%.3f"%(lab,n,f,u))
print("[done] 注: 第4步归一化不改变TopK排序, 故对评分无影响, 仅影响输出的概率值。")
