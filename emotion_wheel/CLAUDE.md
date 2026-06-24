# CLAUDE.md — emotion_wheel（情感轮标签体系）

Track2 EW-based 评估的标签归一基础。把开放词汇的情绪词（同义 / 词形变体）映射到 5 个情感轮的标准类别，再算 F1。先读根 `CLAUDE.md` 的「评估指标」一节。

## 文件

- `wheel1.xlsx` … `wheel5.xlsx` — 5 个情感轮的层级定义（外层细粒度 → 内层基本类别）。
- `synonym.xlsx` — 同义词映射（happy / joyful → happy）。
- `format.csv` — 词形 / 格式归一（angered → anger）。
- `wheel_mapping.npz` — **编译产物**，由 `config.OUTSIDE_WHEEL_MAPPING` 指向，评估时直接加载。含三类映射：
  - `format_mapping`：词形 / 格式归一
  - `raw_mapping`：同义词映射
  - `wheel_map_whole`：情感轮层级映射
- `latest_version.md` — 版本说明。

## 归一流程（评估时对 GT 和 Pred 同样处理）

小写 → 去前后空格 → 词形归一(format) → 同义词映射(raw) → 映射到 wheel 层级 → 丢弃不可映射词 → set 去重。
不在映射体系内的预测词**不参与计分**。消费方：`my_affectgpt/evaluation/wheel.py::wheel_metric_calculation`，默认评 `case3_wheel{1..5}_level1`。

## 约定

- `wheel_mapping.npz` 是从 xlsx / csv 整理来的**编译产物**，**不要手改 npz**。需调整映射时改源文件（`wheel*.xlsx` / `synonym.xlsx` / `format.csv`）后再重新生成 npz；仓库内未见明显的生成脚本，动手前先确认生成方式，别凭空覆盖。
- 改映射会直接改变**所有历史结果**的分数 —— 改完务必重跑评估并说明影响。
- 优化模型输出时，尽量产出落在这些 wheel 标准词内的标签，提升可映射率（详见根 CLAUDE.md）。
