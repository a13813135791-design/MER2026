# CLAUDE.md — my_affectgpt（AffectGPT 主框架）

LAVIS 风格的多模态情感大模型框架，**registry 注册式**。`train.py` / `inference_*.py` 通过 `from my_affectgpt.{tasks,models,runners,processors,datasets.builders} import *` 触发各模块注册，再由 YAML 里的字符串名实例化。先读根 `CLAUDE.md` 了解整体链路。

## 子模块

- `common/` — 基础设施：
  - `registry.py` — 全局注册表（模型 / 数据集 / 处理器 / 任务 / runner / lr_scheduler）。新增组件必经此处。
  - `config.py` — 解析 `--cfg-path` YAML + `--options a.b=c` 覆盖，暴露 `cfg.model_cfg / datasets_cfg / run_cfg / inference_cfg`。
  - `optims.py`（LinearWarmupCosineLR 等）、`logger.py`、`dist_utils.py`（分布式）、`utils.py`。
- `models/` — 模型：
  - `affectgpt.py` — **主模型**（`arch: affectgpt`，`@registry.register_model`）：多模态编码 + Q-Former 融合 + LLM。
  - `Qformer.py` / `blip2.py` / `blip2_outputs.py` / `base_model.py` — BLIP2 / Q-Former 骨架。
  - `encoder.py`、`eva_vit.py`、`modeling_llama.py`、`tokenizer.py`、`ImageBind/`。
  - ⚠️ 本目录内还内嵌了权重目录（`Qwen2.5-7B-Instruct/`、`clip-vit-large-patch14/`、`chinese-hubert-large/`）— 是模型权重，不是代码，**勿改 / 勿提交**。
- `datasets/`：
  - `builders/image_text_pair_builder.py` — 数据集 builder（`@registry.register_builder`）。
  - `datasets/base_dataset.py` — 基类，含 **`get_prompt_for_multimodal(mode, subtitle, user_message)`**（EmoSync 各模态 prompt 由此拼装），`read_frame_face_audio_text(...)`，`func_get_qa_ovlabel(...)`。
  - `datasets/{human,mercaptionplus,mer2026ov}_dataset.py` — 三个数据集类（训练用 human / mercaptionplus，测试用 `MER2026OV_Dataset`）。
- `processors/` — 视觉 / 视频处理器（`alpro_video_train/eval`、`blip2_image_train/eval` 等，YAML 按名引用）。
- `tasks/` — `video_text_pretrain.py`（`task: video_text_pretrain`）、`base_task.py`。
- `runners/runner_base.py` — 训练 / eval 主循环（epoch-based，`runner: runner_base`）。
- `conversation/conversation_video.py` — **`Chat`** 类：推理时封装模态编码（`postprocess_audio/face/frame/image/multi`）与生成（`answer_sample`）。`inference_emosync.py` 重度依赖。
- `evaluation/wheel.py` — **`wheel_metric_calculation(name2gt, name2pred, inter_print=...)`**：EW-based F1（5 wheel 平均），评估的真相源。

## 多模态 token

根 `config.py` 定义占位符：`<AudioHere> <FrameHere> <FaceHere> <ImageHere> <MultiHere>`。prompt 中这些 token 会被对应模态 embedding 替换；推理时 `img_list` dict 的 key 即 `audio/frame/face/image/multi`，不需要的模态置 `None`（EmoSync 的单模态推理正是靠置 `None` 实现）。

## 怎么改

- **加模型**：在 `models/` 写类 + `@registry.register_model("名字")`，在 `models/__init__.py` 导出，YAML `model.arch` 用该名。
- **加数据集**：写 dataset 类（`datasets/datasets/`）+ builder（`datasets/builders/`，`@registry.register_builder`），YAML `datasets:` 下引用。
- **加处理器**：`processors/` + `@registry.register_processor`。
- 改 prompt / 模态组合：改 `base_dataset.py::get_prompt_for_multimodal` 的 mode 分支。注意 `inference_emosync.py` 的 `AUR_MODES` 引用了这些 mode 名（`audioonly / faceonly / textonly / multiface_audio_face_text`），改名要同步推理脚本。

## 约定

- 组件一律走 registry，别直接 import 实例化绕过注册。
- 路径从根 `config.py` 取，不要在框架内写死。
- 与 `inference_emosync.py` 的耦合点：`Chat` 的 `postprocess_*` / `answer_sample` 签名、`get_prompt_for_multimodal` 的 mode 名、`img_list` 的 key — 改这些要回头改推理脚本。
