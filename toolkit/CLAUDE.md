# CLAUDE.md — toolkit（数据处理与工具库）

源自 MERTools / AffectGPT 生态的通用工具库。**注意：并非所有文件都在 MER2026 Track2 流程中使用** —— 很多是历史数据集 / 传统模型的遗留代码。本赛道实际高频用到的，是 `utils/` 下少数读写与 LLM 调用工具。先读根 `CLAUDE.md`。

## 当前流程实际用到的

- `utils/read_files.py` — `func_read_key_from_csv`、`func_get_name2reason` 等读写。evaluation / ovlabel / inference 都在用。
- `utils/qwen.py` — `reason_to_openset_qwen(...)`：调用 Qwen2.5（vLLM）把 reason 文本转成 open-set 情绪标签列表。ovlabel 抽取的核心。
- `utils/functions.py` — `string_to_list`、`split_list_into_batch` 等通用函数。
- `utils/chatgpt.py` — GPT 系列调用封装（备用）。

## 其余（多为遗留 / 跨数据集复用，本赛道一般不碰）

- `dataloader/` — 各数据集 loader：`iemocap / meld / mer2023 / mer2024 / sims / simsv2 / cmudata / crossdim / crossdis`。
- `gptv/` — 各数据集的 GPT-V 提示 / 处理脚本（affectnet、rafdb、dfew、ferv39k、casme、ravdess… 共 20+）。
- `models/` — 传统多模态融合模型：`tfn / lmf / misa / mmim / mult / mfn / mfm / mctn / graph_mfn / ef_lstm / lf_dnn / e2e_model / attention(_topn) / videomae_pretrain`。
- `data/` — 特征级数据集封装（`feat_data`、`feat_data_topn`、`e2e_data`、`videomae_data`）。
- `preprocess/` — 各数据集预处理脚本。
- `utils/` 其余：`metric.py`、`loss.py`、`read_data.py`、`punctuation.py`、`ttsaudio_utils.py`、`videomae_utils.py`、`e2e_utils.py`。
- `globals.py`、`model-tune.yaml`。

## 约定

- 动这里前先确认是否在 MER2026 Track2 链路上：很多模块为其他数据集而写，改动可能无人测试。优先**只在 `utils/` 增量加函数**，别做全局重构。
- LLM 调用统一走 `utils/qwen.py` / `utils/chatgpt.py`，模型路径取自根 `config.py`（`PATH_TO_LLM`）。
- 新增工具函数放 `utils/functions.py` 或 `utils/read_files.py`，**保持向后兼容**（evaluation / ovlabel / inference 依赖其函数签名）。
