#!/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
import sys, os, glob
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2')
os.chdir('/opt/data/wlcc/MER2026_Track2')
import numpy as np
import pandas as pd
import config
from my_affectgpt.evaluation.wheel import wheel_metric_calculation
from toolkit.utils.functions import string_to_list

label_csv = config.PATH_TO_LABEL['Human']
name2gt = {}
df = pd.read_csv(label_csv)
for _, row in df.iterrows():
    name2gt[row['name']] = ', '.join(string_to_list(row['openset']))
print('[GT] total:', len(name2gt))

npz_files = glob.glob('output/results-emosync/*/*.npz')
openset_files = [p for p in npz_files if p.endswith('-openset.npz')]
print('[Pred] openset files:', len(openset_files))

for openset_npz in openset_files:
    modelname = openset_npz.split('/')[-2]
    d = np.load(openset_npz, allow_pickle=True)
    name2pred = {n: item for n, item in zip(d['filenames'], d['fileitems'])}
    common = [n for n in name2pred if n in name2gt]
    print('---')
    print('model:', modelname)
    print('pred samples:', len(name2pred), '  with GT:', len(common))
    if len(common) == 0:
        print('no matching samples, skip')
        continue
    gt_sub   = {n: name2gt[n]    for n in common}
    pred_sub = {n: name2pred[n]  for n in common}
    level1_fscore, _, _ = wheel_metric_calculation(name2gt=gt_sub, name2pred=pred_sub, inter_print=True)
    print('EW-F1 score: %.2f' % (level1_fscore * 100))
