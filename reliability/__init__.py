# -*- coding: utf-8 -*-
"""mer2026-v3 可靠性加权融合：纯 numpy 离线后处理包（分层校准 LCG）。
吃阶段一已落盘的 *_logits.npz（三模态 support 表）→ 用学习门控把 0 成本信号
校准成权重 w_m → 加权线性池融合 → 读 TopK 作最终情绪。
"""
__version__ = "3.0.0"

MODS = ['audio', 'face', 'text']
MI = {'audio': 0, 'face': 1, 'text': 2}
