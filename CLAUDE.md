# CLAUDE.md — MER2026 Track2 (MER-FG)

本文件为仓库**全局规范**。各业务子目录另有专属 `CLAUDE.md`，遵循**就近优先**：在子目录工作时先读该目录的 CLAUDE.md，再回看本文件。

## 项目概述

MER2026 赛道二（MER-FG）：**开放词汇 / 细粒度多模态情感识别**。给定一段视频（人脸 + 音频 + 字幕），模型为每个样本输出**任意类别、任意数量**的情感标签。基于团队此前的 OV-MER / AffectGPT 工作（参考 https://github.com/zeroQiaoba/AffectGPT）。

最终**不直接对模型生成的文本评分**，而是走固定评估链：
`模型生成 reason → 抽取 open-set 情感标签 → emotion wheel 归一映射 → 逐样本 P/R → 5 个 wheel 的平均 F1`。

## 目录结构（业务 vs 第三方）

**业务代码（团队自研，可改 / 各有专属 CLAUDE.md）：**
- `my_affectgpt/` — AffectGPT 主框架（LAVIS 风格，registry 注册式）：模型 / 数据集 / 任务 / runner / 处理器。
- `toolkit/` — 数据处理与工具库：dataloader、LLM 调用、传统 MSA 模型、读写工具。
- `emotion_wheel/` — 情感轮标签体系与映射（评估归一的核心）。
- `train_configs/` — 训练 / 推理 YAML 配置。

**根目录脚本（业务入口）：**
- `config.py` — **全局路径中枢**：所有模型权重、数据集、标签路径都在这里。改路径只改这里，勿在脚本里硬编码。
- `train.py` — AffectGPT 训练入口。
- `inference_emosync.py` — **当前主推理脚本**：EmoSync 方法（AUR + LSC + EoP），产出 `output/results-emosync/`。
- `inference_hybird.py` — 旧版 baseline 推理（README 文档对应版本）。
- `ovlabel_extraction.py` — reason → open-set 标签（调用 Qwen2.5 / vLLM）。
- `evaluation.py` — 官方评估入口（遍历 `output/results-mer2026ov/`）。
- `run_ovlabel_emosync.py` / `run_evaluation_emosync.py` / `score.sh` — 针对 emosync 结果的便捷封装。

**第三方 vendored 代码（⚠️ 勿改 / 不生成 CLAUDE.md）：**
`Chat-UniVi/  LLaMA-VID/  Qwen-Audio/  SALMONN/  Video-ChatGPT/  Video-LLaVA/  mPLUG-Owl/`
这些是零样本 baseline 用到的外部多模态大模型仓库（直接拷贝，无 `.git`）。仅在跑零样本 baseline 时按 README 调用，**不要重构 / 修改其内部代码**。

**其他：** `output/`（结果产物，勿提交大文件）、`工具/`（连接远程服务器等文档）。

## 运行环境（本服务器实测）

- 项目根：`/opt/data/wlcc/MER2026_Track2`
- 数据根：`/opt/data/wlcc/mer2026-data`（= `config.DATA_DIR['MER2026']`）
- 主 conda 环境：**`affectgpt`** — python: `/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python`
  - 训练 / EmoSync 推理 / 评估 都用这个环境（含 torch、vllm、decord、transformers）。
  - README 里提到的 `vllm3` / `whisperx` 是原始 baseline 文档环境，本机不一定有；本机另有 `mse-env`。
- 跑前务必 `conda activate affectgpt`，且**在项目根目录执行**（脚本用相对路径 + `config.AFFECTGPT_ROOT='./'`）。

## 关键命令

训练（单卡示例，多卡见 train_configs/CLAUDE.md）：
```bash
conda activate affectgpt
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg-path=train_configs/human_outputhybird_bestsetup_bestfusion_face_lz.yaml
```

EmoSync 推理（当前主流程）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u inference_emosync.py \
  --dataset=MER2026OV \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"
# 冒烟测试：加 --max_samples 5；数据还在解压时可用 --train_debug（拿 track2_train_human.csv 顶替）
```

抽标签 + 评分（emosync 结果）：
```bash
python run_ovlabel_emosync.py     # reason → *-openset.npz（调用 Qwen2.5）
python run_evaluation_emosync.py  # 对 Human GT 算 EW-F1
bash score.sh                     # 上面评估的封装，只打印 EW-F1
```

官方评估（results-mer2026ov）：`python evaluation.py`

## 数据格式约定（npz）

- **reason npz**（推理产物，`output/results-*/<ckpt_dir>/<epoch>.npz`）：
  - `name2reason`: {样本名 → 最终情绪描述（EoP 输出）}
  - `name2aur`: {样本名 → 4 路单模态预测 dict}（EmoSync 专有）
  - `name2lsc`: {样本名 → {'cot1','cot2'}}（EmoSync 自纠错）
- **openset npz**（`*-openset.npz`）：`filenames`（样本名列表）、`fileitems`（每样本 open-set 标签）
- **GT csv**：列 `name`、`openset`（开放词汇标签，用 `toolkit.utils.functions.string_to_list` 解析）

## 评估指标（务必理解）

EW-based F1，5 个情感轮取平均（`my_affectgpt/evaluation/wheel.py::wheel_metric_calculation`）。
每样本：`P = |GT∩Pred|/|Pred|`，`R = |GT∩Pred|/|GT|`，空预测记 0；最终分 = 5 个 wheel 的 F1 均值 ×100。
→ **标签太多伤 P，太少伤 R**。优化重点：用 emotion-wheel 友好的规范情绪词、控制每样本标签数、去掉不可映射的词、做多模型 / 多 epoch / 多 prompt 的 ensemble。

## EmoSync 方法（inference_emosync.py 核心，当前研究主线）

三模块（对应代码注释）：
1. **AUR**（Asymmetric Unimodal Reasoning）：同一样本分别用 audio-only / face-only / text-only / multi 四种输入各推一次。
2. **LSC**（Logical Self-Correction）：用两个 CoT prompt（结构化 4 步 + “let's think step-by-step”）做零样本链式推理，抑制对立情绪共现（如 sad+happy）。
3. **EoP**（Ensemble of Predictions）：把 4 路 AUR + LSC 结果汇总，用模型自身做纯文本集成投票得最终 reason。
> EoP 仍在迭代（代码注释标注“下一步重写”）。改这块时同步更新本节。

## 全局约定

- **路径全部走 `config.py`**，不要在脚本里硬编码绝对路径；新增模型 / 数据集先在 config.py 注册。
- **registry 注册式**：模型 / 数据集 / 处理器 / 任务 / runner 通过 `@registry.register_*` 注册，靠字符串名在 YAML 引用（详见 my_affectgpt/CLAUDE.md）。
- 中文注释是项目惯例，可继续用中文写注释 / 文档。
- 改动控制在最小范围；**勿改 vendored 第三方目录**；大权重 / npz 结果不要进版本库。
- 跑任何脚本前确认 `conda activate affectgpt` 且 cwd = 项目根。
