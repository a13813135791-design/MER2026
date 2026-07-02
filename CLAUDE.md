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
- `reliability/` — **mer2026-v3 可靠性加权融合**（纯 numpy 离线后处理包，分层校准 LCG 门控）：吃阶段一 `*_logits.npz` 三模态 support 表 → 学习门控把 0 成本信号校准成权重 `w_m` → 加权线性池融合 → 读 TopK 出情绪。有专属 `reliability/CLAUDE.md`。训练侧 `train_mlp_gate.py` 独占 torch，推理侧（`weights/io/fuse/features` 等）绝不触达 torch。

**根目录脚本（业务入口）：**
- `config.py` — **全局路径中枢**：所有模型权重、数据集、标签路径都在这里。改路径只改这里，勿在脚本里硬编码。
- `train.py` — AffectGPT 训练入口。
- `inference_emosync.py` — **当前主推理脚本**：EmoSync 方法（AUR + LSC + EoP），产出 `output/results-emosync/`；加 `--score-mode logits`（方案2）时改产**闭集打分** `*_logits.npz`（audio/face/text 三模态对候选词表的 support 表）。
- `inference_hybird.py` — 旧版 baseline 推理（README 文档对应版本）。
- `ovlabel_extraction.py` — reason → open-set 标签（调用 Qwen2.5 / vLLM）。
- `evaluation.py` — 官方评估入口（遍历 `output/results-mer2026ov/`）。
- `run_ovlabel_emosync.py` / `run_evaluation_emosync.py` / `score.sh` — 针对 emosync 结果的便捷封装。
- `run_perspective_scores.py` — EmoSync EoP 的**分视角真分**：在校准集（human[1200:1532]，与 1200 测试集零重叠、无泄漏）上对 5 个视角各抽 openset、算 EW-F1，写 `scores.json` 供 `run_eop_qwen3.py` 用。
- `run_eop_qwen3.py` — 用 Qwen3（`qwen3eop` 环境）做 EoP 集成投票出最终 reason。
- `auto_pipeline.sh` — EmoSync 全自动复现（① 校准测真分 → ② 全量 1200 出最终分）：每阶段产物校验 + 断点续跑 + 失败写 FAILED 标记退出。
- `gen_caption_candidate.py` / `gen_caption_train_human.py` / `gen_caption_train_human_max.py` / `gen_caption_track1track2.py`（+ `run_caption_track1track2.sh`）— 各批数据的场景描述（caption）生成。

**根目录脚本（可靠性加权，mer2026-v3 支线）：**
- `calibrate_gate.py` — **入口①**：在 human(1532) 上标定可靠性门控 → 产出 `gate_params.json`（`emit=False`，不落融合预测）。支持 `--split cv5`（协议 B0，OOF）/ `--split holdout`（协议 A）。
- `fuse_logits.py` — **入口②（v3 升级）**：吃 `*_logits.npz` + human GT → 逐样本可靠性权重 → 加权线性池 → TopK(K=|GT|) → EW-F1，并落盘融合预测（固定取 lcg 模式）。`--ablation` 同表打印 `text/equal/prior/tier0/lcg` 消融。
- `apply_gate_candidate.py` — 把训练好的 gate 应用到**无标签 candidate** 样本（不标定、不需 GT）→ 出 `name,openset` 预测 csv；`--K` 固定每样本情绪个数。复用 `reliability.io` 的 `load_logits`/`load_gate`。
- `train_mlp_gate.py` — 离线训练 per-modality 共享 MLP 门控（torch）→ 存为纯 numpy 可读 JSON（`mlp` 模式）；训练侧独占 torch。
- `eval_logits.py` — 方案2 logits 支持度结果评估：三条单模态按 support 取 Top-K → wheel 映射 → 算 EW-F1（对照用）。
- `fuse_logits_cv.py` / `fuse_subset.py` — 融合的 CV / 子集变体（对照实验）。
- `merge_logits.py` — 合并两份 `*_logits.npz`（如 1200 + 332 → 1532），**校验 `candidate_labels` 一致**（`neutral`/`tau` 直接沿用 base、不校验），样本基本不重叠。
- `merge_candidate_logits.py` — 把 candidate 分片 `*_logits_cand_*.npz` 合并成 `_logits_candidate.npz`。
- `run_reliability.sh` — 一键：可靠性加权融合 + 评测 + 消融（默认 `cv5` = 现成 1200 OOF；`holdout` 需先扩集到 1532）。
- `run_mcp_logits.sh` / `run_candidate_logits.sh`（+ `watch_apply_candidate.sh`）— 全量 MCP(31327) / candidate logits 分片推理：多分片、失败即还原默认 NPZ、已存分片自动 skip 可续跑、固定 `score-tau 1.0`。
- 其余 `run_a.sh` / `run_b0.sh` / `dl_qwen3.py`/`.sh` / `_*.py`（`_check_coverage`/`_dump_lsc`/`_sweep_*`/`_verify` 等）为一次性辅助 / 调试脚本，非主链路。

**第三方 vendored 代码（⚠️ 勿改 / 不生成 CLAUDE.md）：**
`Chat-UniVi/  LLaMA-VID/  Qwen-Audio/  SALMONN/  Video-ChatGPT/  Video-LLaVA/  mPLUG-Owl/`
这些是零样本 baseline 用到的外部多模态大模型仓库（直接拷贝，无 `.git`）。仅在跑零样本 baseline 时按 README 调用，**不要重构 / 修改其内部代码**。

**其他：**
- `output/`（结果产物，勿提交大文件）、`工具/`（连接远程服务器等文档）。
- `versions/`（各版本代码快照 + 交接文档）：`mer2025-v1` / `mer2026-v1` / `mer2026-v3` / `mer2026-v4`，各含只读快照与方案记录（如 `versions/mer2026-v4/README.md` + `mer2026-v4.md` + `_代码说明_给队友.md`）。快照仅供浏览 / 对照，改代码请在对应 live worktree。
- `code_git/`（**并行独立仓库**，自带 `.git`，根 `.gitignore` 已忽略 `code_git/`）：mer2026-v1「含置信度功能」的独立版本库，还原时把内部文件覆盖回项目根。

## git worktree 布局（重要）

本仓用 git worktree 分版本并行开发（`git worktree list`）：
- `mer2026-v3` → `/opt/data/wlcc/MER2026_Track2_1027`（当前主 worktree）
- `mer2026-v4` → `/opt/data/wlcc/MER2026_Track2_v4`
- `mer2026-v5` → `/opt/data/wlcc/MER2026_Track2_v5`

`git branch` 里带 `+` 的分支（v4 / v5）已被别的 worktree 占用，**不能在本目录 `checkout`**。改哪个版本就进对应目录。

## 运行环境（本服务器实测）

- 项目根（当前 worktree）：`/opt/data/wlcc/MER2026_Track2_1027`（git 分支 `mer2026-v3`）。注：脚本内多以 `/opt/data/wlcc/MER2026_Track2`（指向本 worktree 的 symlink / 别名）为 `sys.path` 与 `chdir` 目标。
- 数据根：`/opt/data/wlcc/mer2026-data`（= `config.DATA_DIR['MER2026']`）
- 主 conda 环境：**`affectgpt`** — python: `/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python`
  - 训练 / EmoSync 推理 / logits 打分 / 可靠性加权 / 评估 都用这个环境（含 torch、vllm、decord、transformers）。
  - EoP 集成用 `qwen3eop` 环境（`run_eop_qwen3.py`）。
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

可靠性加权（reliability，先出 logits 再融合）：
```bash
# 1) 产闭集打分 logits（方案2）：inference_emosync.py 加 --score-mode logits → *_logits.npz
bash run_reliability.sh cv5   # 融合+评测+消融（cv5=1200 OOF；holdout 需先扩到 1532）
# 或分步：
python calibrate_gate.py --split cv5           # 入口① 标定门控 → gate_params.json
python fuse_logits.py --split cv5 --ablation    # 入口② 融合+落盘+消融对比
python apply_gate_candidate.py --npz <candidate_logits.npz> --gate <gate_1532.json> --out answer.csv --K 5
```

## 数据格式约定（npz）

- **reason npz**（推理产物，`output/results-*/<ckpt_dir>/<epoch>.npz`）：
  - `name2reason`: {样本名 → 最终情绪描述（EoP 输出）}
  - `name2aur`: {样本名 → 4 路单模态预测 dict}（EmoSync 专有）
  - `name2lsc`: {样本名 → {'cot1','cot2'}}（EmoSync 自纠错）
- **openset npz**（`*-openset.npz`）：`filenames`（样本名列表）、`fileitems`（每样本 open-set 标签）
- **logits npz**（`*_logits.npz`，方案2 闭集打分 / 可靠性加权输入）：`name2logits`（{样本名 → 三模态 audio/face/text 对候选词表的 support 表}）、`candidate_labels`（候选情绪词表）、`neutral`（中性分 dict）、`tau`（温度）、`score_mode='logits'`。
- **GT csv**：列 `name`、`openset`（开放词汇标签，用 `toolkit.utils.functions.string_to_list` 解析）

## 评估指标（务必理解）

EW-based F1，5 个情感轮取平均（`my_affectgpt/evaluation/wheel.py::wheel_metric_calculation`）。
每样本：`P = |GT∩Pred|/|Pred|`，`R = |GT∩Pred|/|GT|`，空预测记 0；最终分 = 5 个 wheel 的 F1 均值 ×100。
→ **标签太多伤 P，太少伤 R**。优化重点：用 emotion-wheel 友好的规范情绪词、控制每样本标签数、去掉不可映射的词、做多模型 / 多 epoch / 多 prompt 的 ensemble。

## EmoSync 方法（inference_emosync.py 核心，当前研究主线）

三模块（对应代码注释）：
1. **AUR**（Asymmetric Unimodal Reasoning）：同一样本分别用 audio-only / face-only / text-only / multi 四种输入各推一次。
2. **LSC**（Logical Self-Correction）：用两个 CoT prompt（结构化 4 步 + "let's think step-by-step"）做零样本链式推理，抑制对立情绪共现（如 sad+happy）。
3. **EoP**（Ensemble of Predictions）：把 4 路 AUR + LSC 结果汇总，用模型自身做纯文本集成投票得最终 reason。
> EoP 仍在迭代（代码注释标注"下一步重写"）。改这块时同步更新本节。

## 可靠性加权（reliability，mer2026-v3 研究支线）

链路：`inference_emosync.py --score-mode logits`（方案2）在候选词表上产 `*_logits.npz`（audio/face/text 三模态 support 表）→ `reliability/` 用**分层校准 LCG 门控**把 0 成本信号（模态置信 / 一致性等 8 维特征）校准成逐样本可靠性权重 `w_m` → 加权线性池融合 → 读 TopK(K=|GT| 或固定 K) 作最终情绪。
标定用 human(1532)，`cv5` 报 OOF、`holdout` 报留出 eval；消融对比 `text/equal/prior/tier0/lcg`。纯 numpy 离线后处理（推理侧不触 torch，训练 MLP 门控例外）。
细节 / 关键函数 / 契约见 `reliability/CLAUDE.md`。

## 全局约定

- **路径全部走 `config.py`**，不要在脚本里硬编码绝对路径；新增模型 / 数据集先在 config.py 注册。
- **registry 注册式**：模型 / 数据集 / 处理器 / 任务 / runner 通过 `@registry.register_*` 注册，靠字符串名在 YAML 引用（详见 my_affectgpt/CLAUDE.md）。
- **多 worktree**：改哪个版本进哪个目录；带 `+` 的分支不能在别处 checkout。`code_git/` 是并行独立仓库（已被根 `.gitignore` 忽略），不要与本仓混提交。
- 中文注释是项目惯例，可继续用中文写注释 / 文档。
- 改动控制在最小范围；**勿改 vendored 第三方目录**；大权重 / npz 结果不要进版本库。
- 跑任何脚本前确认 `conda activate affectgpt` 且 cwd = 项目根。

## mer2026-v3.1（caption 注入 + 全量 MLP 打分器）

基于 mer2026-v3。阶段一在三条单模态 logits 提示词中注入每样本场景描述 caption；阶段二功能不变，打分器改用全量数据训练。

**阶段一（caption 版 logits）**
- `inference_emosync.py` 新增 `--caption_csv`（默认 `track2_train_human_caption-AffectGPT.csv`，读 name,caption）与 `--logits_out`（自定义 NPZ 路径）；`run_aur_logits` 把 caption 透传给提示词，neutral 基线保持 caption-free。
- `my_affectgpt/datasets/datasets/base_dataset.py` 的 `audioonly/faceonly/textonly` 三分支补上 `<Caption>` 注入（原仅全模态分支有）。
- 运行（同 CSV 既作样本清单又作 caption 源）：
```
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u inference_emosync.py --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --dataset MER2026OV --options inference.test_epoch=35 --score-mode logits --score-tau 1.0 \
  --logits_csv <name列csv> --caption_csv /opt/data/wlcc/mer2026-data/track2_train_human_caption-AffectGPT.csv \
  --logits_out output/results-emosync/<ckpt>/checkpoint_000035_loss_0.716_logits_v31cap.npz
```

**阶段二（全量打分器，无验证/无测试）**
```
python train_mlp_gate_full.py --npz <v31cap.npz> \
  --human /opt/data/wlcc/mer2026-data/track2_train_human_caption.csv --mcp <空csv> \
  --out output/_cal/gate_params_mlp_full.json
```

**测试（_ovmerdplus，EW-F1 K=gt）**
```
python eval_mlp_gate.py --npz <v31cap.npz> --gate output/_cal/gate_params_mlp_full.json \
  --gt /opt/data/wlcc/mer2026-data/_ovmerdplus_ovlabel.csv --out answer_v31_ovmerdplus.csv
```
`apply_gate_candidate.py`/`eval_mlp_gate.py` 均按 `__file__` 自定位工程根（不再硬编码兄弟目录）。

**产物**：阶段一 `output/results-emosync/<ckpt>/checkpoint_000035_loss_0.716_logits_v31cap.npz`；阶段二 `output/_cal/gate_params_mlp_full.json`。
