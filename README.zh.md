<div align="center">

# 赛道二 — MER-FG

### 细粒度情感识别基线方法

> 本项目基于我们此前在 OV-MER 和 AffectGPT 上的工作构建。
> 参考仓库：[AffectGPT](https://github.com/zeroQiaoba/AffectGPT)

</div>

---

## 目录

- [任务描述](#-任务描述)
- [数据集](#-数据集)
- [评测指标](#-评测指标)
- [零样本基线](#️-零样本基线)
- [AffectGPT（后训练）](#-affectgpt后训练)

---

## 📌 任务描述

MER-FG 最早在 MER2024 中提出，如今已是**第三届**。人类情感所涵盖的词汇极为丰富，远远超出传统任务中所使用的六类基本标签。将情感表达局限于固定的标签集合会导致表征不准确。在本赛道中，参赛者可以借助多模态大语言模型（MLLM）的词汇能力与多模态理解能力，预测**任意类别、任意数量的情感标签**——包括细粒度和细微的情感。

<div align="center">
  <img src="../icon/track2.png" alt="MER-FG 任务" width="85%">
  <p><em>MER-FG：不同于以往聚焦基本情感的 MER 任务，MER-FG 将识别范围扩展到任意情感类别。</em></p>
</div>


---

## 🚀 数据集

我们提供两个带标注的训练数据集：

| 数据集 | 标注类型 | 样本量 |
|---|---|---|
| `track2_train_human.csv`（Human-OV） | 人工标注 | 1,532 |
| `track2_train_mercaptionplus.csv`（MER-Caption+） | 自动标注 | 31,327 |


```
dataset/
└── mer2026-dataset/                      # https://huggingface.co/datasets/MERChallenge/MER2026
    ├── video/                            # 132,171 个样本
    ├── audio/                            # 132,171 个样本
    ├── openface_face/                    # 132,171 个样本
    ├── subtitle_chieng.csv               # 预提取的字幕，132,171 个样本
    ├── track2_train_human.csv            # 人工标注标签，1,532 个样本
    ├── track2_train_mercaptionplus.csv   # 自动标注标签，31,327 个样本
    └── track1_track2_candidate.csv       # 20,000 个候选样本
```

---

## 📏 评测指标

由于标签空间是开放的，评测采用基于**情感轮（Emotion Wheel，EW-based）**的指标，以处理同义词和标签变体。

**三级分组策略：**
- **L1：** 将情感词归一化为其基本形式（例如 *"angered"* → *"anger"*）
- **L2：** 将同义词映射到统一标签（例如 *"happy"* 和 *"joyful"* → *"happy"*）
- **L3：** 将情感轮外层标签映射到其最内层的基本类别（使用 5 个情感轮）

**最终得分：** 在全部 5 个情感轮上的平均 F1 分数（越高越好）。

---


## 🗝️ 零样本基线

### 准备工作

```
dataset/
└── mer2026-dataset/
    ├── ...
    ├── track2_train_human.csv
    ├── track2_train_mercaptionplus.csv
    └── track1_track2_candidate.csv

MER2026_Track2/
└── models/   # 预训练权重：https://pan.baidu.com/s/1KHL1oGCtvqr8IMNWDWxH3Q?pwd=djjw
    ├── bert-base-uncased/
    ├── Chat-UniVi/
    ├── clip-vit-large-patch14/
    ├── LanguageBind_Image/
    └── ...
```

### 推理

**第 1 步：** 生成描述并保存到 `./output/results-mer2026ov`

```bash
conda activate vllm3

cd Qwen-Audio
CUDA_VISIBLE_DEVICES=0 python main-audio.py --subtitle_flag='subtitle' --dataset='MER2026OV'

cd SALMONN
CUDA_VISIBLE_DEVICES=0 python main-audio.py --subtitle_flag='subtitle' --dataset='MER2026OV'

cd Video-ChatGPT
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle' --dataset='MER2026OV'

cd Chat-UniVi
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle' --dataset='MER2026OV'
```

```bash
conda activate whisperx

cd LLaMA-VID
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle' --dataset='MER2026OV'

cd Video-LLaVA
CUDA_VISIBLE_DEVICES=0 python main-video.py --subtitle_flag='subtitle' --dataset='MER2026OV'
```

**第 2 步：** 从生成的描述中提取开放词汇标签：
```bash
python ovlabel_extraction.py
```

---

## ✨ AffectGPT（后训练）

### 准备工作

```
MER2026_Track2/
└── models/                       
    ├── chinese-hubert-large/      # 音频编码器（HuggingFace: TencentGameMate/chinese-hubert-large）
    ├── clip-vit-large-patch14/    # 视频编码器（HuggingFace: openai/clip-vit-large-patch14）
    └── Qwen2.5-7B-Instruct/       # 大语言模型（HuggingFace: Qwen/Qwen2.5-7B-Instruct）
```

> 更新 `config.py` 中的路径占位符——将 `xxx` 替换为你自己的路径。

### 训练

```bash
# 模型 1：在 Human-OV 数据上训练
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg-path=train_configs/human_outputhybird_bestsetup_bestfusion_face_lz.yaml

# 模型 2：在 MERCaption+ 数据上训练
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
```

### 推理

结果保存到 `./output/results-mer2026ov`。

```bash
# 模型 1：在 Human-OV 上推理
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py \
  --zeroshot \
  --dataset='MER2026OV' \
  --cfg-path=train_configs/human_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"

# 模型 2：在 MERCaption+ 上推理
CUDA_VISIBLE_DEVICES=0 python -u inference_hybird.py \
  --zeroshot \
  --dataset='MER2026OV' \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --options "inference.test_epochs=10-60" "inference.skip_epoch=5"
```

---

## 👍 评测

```bash
# （描述 -> OV 标签）+ 真实标签 => 基于情感轮的得分
python evaluation.py
```
