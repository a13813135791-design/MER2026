# mer2025-v1 (无置信度功能)

EmoSync 复现版本，产出全量 1200 的 58.68(Qwen3 EoP) / 60.46(自聚合)。
含 caption + 真分 + 确定性解码，但 AUR 只输出情绪、**不带置信度**。
(= inference_emosync.py.bak_preconf)

## 与 mer2026-v1 唯一区别
只有 `inference_emosync.py` 不同：本版无置信度功能。其余文件两版完全相同。

## 文件 / 还原: 同 mer2026-v1，区别仅 inference_emosync.py。
