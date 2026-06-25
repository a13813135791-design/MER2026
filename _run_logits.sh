#!/bin/bash
cd /opt/data/wlcc/MER2026_Track2
AFF=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
CFG=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
CUDA_VISIBLE_DEVICES=0 $AFF -u inference_emosync.py --cfg-path=$CFG --dataset MER2026OV --options inference.test_epoch=35 --score-mode logits "$@"
