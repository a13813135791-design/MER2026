#!/bin/bash
cd /opt/data/wlcc/MER2026_Track2
echo '====================================='
echo '正在加载库并评分，约需 20-40 秒，请耐心等待...'
echo '（这段时间没有输出是正常的，不要按 Ctrl+C）'
echo '====================================='
PY=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
$PY -u run_evaluation_emosync.py 2>/dev/null | grep -E 'GT|Pred|model:|samples|EW-F1'
echo '====================================='
echo '评分完成！上面的 EW-F1 score 就是分数'
echo '====================================='
