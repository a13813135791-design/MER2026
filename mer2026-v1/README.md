# mer2026-v1 (含置信度功能)

EmoSync 复现 + 我们的增强：AUR 每个模态输出「情绪 + 置信度」。

## 与 mer2025-v1 唯一区别
只有 `inference_emosync.py` 一个文件不同：
- 本版含 `AUR_CONF_PROMPT` / `parse_emotion_confidence` / `name2aur_conf`（置信度）
- 其余文件(base_dataset.py / run_*.py / auto_pipeline.sh / config.py)两版完全相同

## 文件清单
- inference_emosync.py  主推理(AUR+LSC+caption+置信度)
- my_affectgpt/datasets/datasets/base_dataset.py  prompt模板(含<Caption>槽位)
- run_perspective_scores.py / run_eop_qwen3.py / run_ovlabel_emosync.py / run_evaluation_emosync.py / auto_pipeline.sh / config.py

## 还原: 把 inference_emosync.py 覆盖回项目根, base_dataset.py 覆盖回 my_affectgpt/datasets/datasets/
