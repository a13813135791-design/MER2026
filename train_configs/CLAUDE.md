# CLAUDE.md — train_configs（训练 / 推理 YAML 配置）

LAVIS 风格 YAML，被 `train.py` / `inference_*.py` 经 `--cfg-path` 加载，可用 `--options "a.b=c"` 覆盖任意字段。先读根 `CLAUDE.md`。

## 现有配置

- `human_outputhybird_bestsetup_bestfusion_face_lz.yaml` — 在 Human-OV（1,532，人工标注）上训练。
- `mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml` — 在 MER-Caption+（31,327，自动标注）上训练。

## 四大段

- `model:` — `arch: affectgpt`；编码器按**名字**选择（`llama_model: Qwen25` / `acoustic_encoder: HUBERT_LARGE` / `visual_encoder: CLIP_VIT_LARGE`，名字对应 `config.py` 的 `PATH_TO_*`）；各模态 `*_fusion_type`（attention / qformer / mean）、`frozen_*` 冻结开关、`num_*_query_token`、`vis/text/img_processor`（按名引用 my_affectgpt 处理器）。
- `datasets:` — `human` / `mercaptionplus`；`face_or_frame: multiface_audio_face_text`（用哪些模态），`label_type: ovlabel`（预测标签而非纯描述）。
- `run:` — 训练超参：`lr_sched`、`init_lr`（注释提示 3e-5 / 2e-5 会 NaN，故用 1e-5）、`max_epoch`、`iters_per_epoch`、`batch_size_train`（**单卡** batch，实际 = GPU 数 × 此值）、`amp`、`distributed` / `world_size`、`seed`。
- `inference:` — 推理专用：处理器（`*_eval`）、选 ckpt 的 `ckpt_root / ckpt_name / test_epoch / test_epochs / skip_epoch`、`gpu`。

## 新建配置

复制最接近的一份改名（命名惯例：`<数据集>_<结构标记>.yaml`）。注意：
- 训练输出落在 `output/<cfg文件名去后缀>/<cfg名_时间戳>/`（见 `train.py` 的 `job_id`）；inference 默认按 cfg 名去 `output/` 找 ckpt 目录。
- 多卡：`CUDA_VISIBLE_DEVICES=0,1,2` + 分布式启动；**实际 batch = 卡数 × `batch_size_train`**。
- `model.llama_model` / `acoustic_encoder` / `visual_encoder` 用的名字必须已在 `config.py` 注册路径，否则加载失败。
- 改 `init_lr` 留意 NaN 历史（见 YAML 注释），别贸然调大。
