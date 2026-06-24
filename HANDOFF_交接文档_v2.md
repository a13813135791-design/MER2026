# MER2026 EmoSync 复现 + 自研方案 —— 交接文档 v2（自包含）

> 写于 2026-06-24。接手的新 Claude：**只读这一份 + 看代码**就能干活。
> 这份取代了旧的 `HANDOFF_交接文档.md`（旧版只到"环境搭好、80样本初步分"；本版含 caption / 真分 / 全量1200 / 置信度功能 全过程）。
> 本机项目记忆：`C:\Users\a1382\.claude\projects\C--Users-a1382-Desktop-misssion\memory\`（MEMORY.md 索引）。

---

## 0. 和用户协作的硬性约定（最重要，先读）
- 用户是**大一计算机学生**：看得懂 C/Java，**没学过 Python**，**机器学习/深度学习零基础**。术语要用大白话+类比，别默认他懂。
- **不要有幻觉**：不确定先连服务器查证再下结论，绝不编造路径/命令/参数/结论。
- **不可盲目按字面做**：他可能因不懂而说错；遇到他的要求和事实/最佳实践冲突，要指出问题、给更稳方案。**你对正确性负责。**
- 关键操作**先解释、小步执行、做完立即验证**；破坏性操作（删/覆盖/改配置）先征同意。
- 全局约定见 `C:\Users\a1382\.claude\CLAUDE.md`。
- 用户很较真、会反复追问细节——回答务必精确，分清"事实"和"推断"。

---

## 1. 任务背景
- **基础**：复现论文 **EmoSync (Tong 等, MRAC'25)**——三模块 **AUR + LSC + EoP**。论文全文：`C:\Users\a1382\Desktop\misssion\_emosync_paper.txt`。
- **目标演变（重要）**：最初是"忠实复现论文"，后来用户明确改为 **"在 MER2026 Track2 上做我们自己的方案，EmoSync 为基础"**。所以现在**不追求逐字复刻论文**，而是做一个好用的 2026 方案。
- **三模块说白了**：
  1. **AUR**（辅助单模态推理）：对同一样本，分 audio/face/text/multi 四个视角各做一次情绪判断。
  2. **LSC**（逻辑自纠错）：零样本 CoT（论文两个 prompt）→ 挑自洽范例 → 少样本 CoT，防对立情绪共现。
  3. **EoP**（视角集成）：用 Qwen3-8B 当裁判，综合各视角（带表现分）出最终情绪。
- **论文基准（Table 1，MER2025 官方测试集 1200，EW-F1）**：AffectGPT 基线 47.10 → +AUR+LSC+EoP 完整 **55.58**。

---

## 2. 服务器与环境
- **服务器**：`192.168.21.57` 端口 **1024**（wlcc 机），root / 123。
- **连法**：本机 `C:\新建文件夹\python.exe` + paramiko。**别用 Bash tool 直接 ssh**。脚本开头必加：
  ```python
  import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
  import paramiko
  ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
  ssh.connect("192.168.21.57",port=1024,username="root",password="123",timeout=30)
  ```
- **项目根**：`/opt/data/wlcc/MER2026_Track2`
- **GPU**：RTX 4090 24G。驱动 535 → CUDA≤12.2 → **torch 必须 cu121**。
- **两个 conda 环境**（`/opt/data/wlcc/Software/Anaconda3/envs/`，用绝对路径调 python）：
  | 环境 | 用途 | python 路径 |
  |---|---|---|
  | **affectgpt** | AffectGPT 推理(AUR/LSC/caption) + 抽标签(Qwen2.5) + 评估 | `.../envs/affectgpt/bin/python` |
  | **qwen3eop** | Qwen3-8B 的 EoP | `.../envs/qwen3eop/bin/python` |
- **显存约束**：AffectGPT(~18G) 和 Qwen3-8B(~16G) **不能同时驻留** → 必须分阶段、各跑各的、释放后再下一步。
- **配置/权重**：
  - CFG = `train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml`
  - 权重目录 CKPTDIR = `mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz_20250408184`
  - ckpt = `checkpoint_000035_loss_0.716`（**只有 epoch 35 一个**，推理必加 `--options 'inference.test_epoch=35'`）

---

## 3. 数据（关键，有坑）
- 根目录：`/opt/data/wlcc/mer2026-data/`
- CSV：
  | 文件 | 内容 |
  |---|---|
  | `track2_train_human.csv` | **1532 条，带 openset 标签**。我们实际就在它上面跑 |
  | `track2_test.csv` | 1200 条 = human 的**前 1200**（内容相同） |
  | `track2_train_mercaptionplus.csv` | 31327 条（AffectGPT 训练数据，**我们没用**） |
  | `track1_track2_candidate.csv` | 20000 条**无标签**（没用来评测） |
  | `subtitle_chieng.csv` | 中/英字幕（name,chinese,english） |
- 媒体：`audio/`(.wav)、`video/`(.mp4)、`openface_face/`(人脸 .npy)。
- **样本切分（我们的用法）**：human 的 **前 1200 = 测试集**（出最终分）、**后 332 = 校准集**（测真分，与测试集零重叠，无泄漏）。后 332 名字是 `sample_xxx` 前缀，前 1200 是 `samplenew3_xxx`。媒体都齐全（已验证）。
- **⚠️ 没有 caption 字段**：所有 CSV 只有 name/字幕/openset，**官方没提供画面描述 caption**（已反复验证）。
- **⚠️⚠️ 最大未决问题**：**这 1200 是不是论文用的"MER2025 官方测试集"——没核实！** 旧交接文档断言"track2_test=MER2025官方测试集"，但**查无官方出处**；README 只说"MER2026 is derived from MER2025"（派生 = 很可能重新标注过）。服务器上**没有真·2025 数据**可对照。**结论：我们的分不能确认和论文 55.58 同一把尺子。** 要严格对比论文，需去 HuggingFace `MERChallenge/MER2025` 下载真 2025 测试集核对/重跑。

---

## 4. 这次会话做了什么（按时间）

### 4.1 通读论文 + 核对忠实度
- AUR/LSC 结构忠实；EoP 用 Qwen3 + 论文 prompt。
- 发现两处主要偏差：① 数据没 caption；② EoP 的"表现分"原来是**写死的先验值**（不是实测）。

### 4.2 加 caption（用户选的 B 方案）
- 2026 没 caption → **用 AffectGPT 自己生成**视频描述当 caption（`run_caption`，确定性解码）。
- **重要性质**：① 是**自生成**的（非论文那种外部 caption）；② **会夹带情绪判断**（"experiencing anger"），属自我暗示。对"做自己的 2026 方案"这是**正当技巧**（CoT 式、不碰 GT、无作弊），但**和论文不是一回事，分数不能逐一硬比**。
- caption 注入位置：**LSC 两条 CoT prompt（与论文一致）+ AUR-multi（我们多放的一处，论文只在 LSC）**。
- 顺手把生成**改成确定性解码**（`do_sample=False`），结果可复现。

### 4.3 测真分（A 方案）—— `run_perspective_scores.py`（新写）
- 在**校准集 332** 上，对 5 个视角各自抽 openset 标签、和 GT 算 EW-F1 → 5 个真分 → `output/_cal/scores.json`。无泄漏。
- 结果：**multi 0.6073 / lsc 0.5495 / text 0.5467 / face 0.4013 / audio 0.3812**（排序合理：全模态>字幕≈自纠错>人脸>音频）。

### 4.4 全量 1200 跑通 —— `auto_pipeline.sh`（新写，7 阶段编排）
顺序：① 校准推理 → ② 测真分 → ③ 全量1200推理 → ④ Qwen3 EoP(用真分) → ⑤ numpy格式转换 → ⑥ 抽标签(Qwen2.5) → ⑦ 评估。挂 tmux 跑了一夜。
- **最终结果（全量1200，带caption+真分+确定性）**：
  - **Qwen3 EoP（论文方法版）= 58.68**
  - AffectGPT 自聚合（对照基线，非论文EoP）= 60.46
  - 论文 55.58 —— 两个都更高，但**见 §3 可比性警告**，别声称"赢了论文"。
- 落盘：`output/_cal/FINAL_RESULT.txt`。（注：当时 `/tmp/auto_pipeline.log` 被系统清掉，靠重跑 `run_evaluation_emosync.py` 恢复的分。）
- 各视角在**全量1200**上的分（`scores_full1200.json`）：multi 56.65 / text 49.15 / lsc 49.09 / audio 43.67 / face 40.36。

### 4.5 置信度功能（新功能，不是论文的）
- 需求：**AUR 每个模态不只输出情绪，还给每个情绪一个置信度**。方案 A：模型自报分（改提示词，1 次生成，最省）。
- 实现：`AUR_CONF_PROMPT`（要求"情绪: 0~1分数"逗号分隔）+ `parse_emotion_confidence()`（正则解析+去填充词）+ npz 增存 `name2aur_conf` + `--save_tag`（测试不覆盖正式结果）。
- 3 样本测试通过（写在 `..._conftest.npz`）。例：audio `anxious:0.9, tense:0.1, calm:0.0`；multi 给 8 个带分情绪。置信度有区分度。
- **小瑕疵**：偶尔混入非情绪词（如 `rumored:1.0`）；解析器不挑词。**未接入 EoP、未在全量跑。**

---

## 5. 代码改动清单 + 备份（接手必看）
所有改动在 `/opt/data/wlcc/MER2026_Track2/`：
| 文件 | 改了什么 | 备份 |
|---|---|---|
| `inference_emosync.py` | 加 `run_caption`/`CAPTION_PROMPT`；caption 只进 multi/LSC；`do_sample=False`；`--start_idx`；`--save_tag`；`AUR_CONF_PROMPT`+`parse_emotion_confidence`+`name2aur_conf` | `.bak_caption`(加caption前)、`.bak_preconf`(加置信度前) |
| `my_affectgpt/datasets/datasets/base_dataset.py` | `get_prompt_for_multimodal` 加 `caption=` 参数 + 多模态分支注入 `<Caption>` | `.bak_caption` |
| `run_perspective_scores.py` | **新增**：分视角测真分 → scores.json | — |
| `auto_pipeline.sh` | **新增**：7 阶段全自动编排 | — |
**回滚**：`cp xxx.bak_caption xxx`（或 `.bak_preconf`）。

---

## 6. 关键文件位置
**服务器产物**（`output/` 下）：
- `results-emosync/<CKPTDIR>/checkpoint_000035_loss_0.716.npz` —— **全量1200推理**（含 name2aur/lsc/caption/reason；旧版无 name2aur_conf）
- `..._loss_0.716-eop.npz` —— Qwen3 EoP 结果
- `..._loss_0.716{,-eop}-openset.npz` —— 抽出的 openset 标签
- `..._loss_0.716_conftest.npz` —— 3样本置信度测试（**有 name2aur_conf**）
- `_cal/scores.json` —— 校准集 5 个真分（EoP 用）
- `_cal/scores_full1200.json` —— 各视角在全量上的分
- `_cal/FINAL_RESULT.txt` —— 最终成绩落盘
- `_cal/置信度测试_3样本结果.md` —— 给队友看的可读结果
- `_backup_80nocap/` —— 旧80样本(无caption)结果(60.11/61.65)
- `_backup_verify/` —— 验证残留 npz

**本地**（`C:\Users\a1382\Desktop\misssion\`）：
- `_srv_*.py` —— 服务器代码的本地副本（改完上传用）
- `_emosync_paper.txt` —— 论文全文
- `STATUS_复现进度.md`、`FEATURE_置信度_开发记录.md`、`置信度测试_3样本结果.md`
- 一堆 `_*.py` —— 连服务器查/跑的一次性脚本（可参考写法）

---

## 7. 怎么跑（命令）
环境变量：
```bash
cd /opt/data/wlcc/MER2026_Track2
AFF=/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
Q3=/opt/data/wlcc/Software/Anaconda3/envs/qwen3eop/bin/python
CFG=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml
```
- **全自动跑全套**（推荐，挂 tmux）：`tmux new-session -d -s X "bash auto_pipeline.sh > /tmp/auto.log 2>&1"`。可断点续跑（产物在就 SKIP）。
- **只推理 N 个样本测试**（不覆盖正式结果）：
  ```bash
  CUDA_VISIBLE_DEVICES=0 $AFF -u inference_emosync.py --cfg-path=$CFG --dataset MER2026OV \
    --train_debug --options 'inference.test_epoch=35' --max_samples 3 --start_idx 0 --save_tag _test
  ```
  （`--train_debug` 读 human.csv 有GT；`--start_idx 1200 --max_samples 332` 取校准集；不加 train_debug 会读 2万无标签candidate）
- **单独测真分**：`$AFF run_perspective_scores.py <某个npz> <输出scores.json>`
- **单独评估**（从已有 -openset.npz 算分）：`$AFF run_evaluation_emosync.py`

---

## 8. 结果汇总
| 项 | 分(EW-F1) | 备注 |
|---|---|---|
| 论文 baseline / 完整 | 47.10 / 55.58 | MER2025，可比性见 §3 |
| 我们 80样本初步(无caption) | 自聚合60.11 / Qwen3-EoP 61.65 | 在 _backup_80nocap |
| 我们 全量1200 自聚合 | **60.46** | 对照基线(非论文EoP) |
| 我们 全量1200 Qwen3-EoP(真分) | **58.68** | 论文方法版，我们的主成绩 |
| 各视角真分(校准332) | multi.61/lsc.55/text.55/face.40/audio.38 | scores.json |
| 各视角(全量1200) | multi56.65/text49.15/lsc49.09/audio43.67/face40.36 | scores_full1200.json |

---

## 9. 未决问题 / 下一步（接手重点）
1. **数据出处（最重要）**：确认 1200 是不是论文的 MER2025 测试集（样本+标签）。要严格对比论文 → 下真 2025 数据核对/重跑；只做 2026 方案 → 现状即可。
2. **置信度功能续做**：① 接入 EoP（让 Qwen3 裁判用上每情绪置信度）；② 加情绪词表过滤去掉 `rumored` 这类非情绪词；③ 决定是否全量重跑带置信度版。
3. **caption 路线**：保持自生成（2026方案够用）/ 或去搞外部 caption（贴论文）。另：AUR-multi 的 caption 是否去掉（论文只在 LSC）。

---

## 10. 踩坑清单（务必记牢）
1. **paramiko 脚本必加** `sys.stdout.reconfigure(encoding='utf-8')`；本机终端是 GBK，输出乱码是显示问题不是数据错。
2. **别用 Bash tool 直接 ssh**（会改写 /opt 路径）；连服务器一律 paramiko。
3. **`/tmp` 会被系统定时清**：跑完的最终分**别只留在 /tmp 日志**里——产物 npz 在持久磁盘没事，但分数要重跑评估或落盘到 `output/`。
4. **numpy 跨版本不兼容**：qwen3eop(numpy2.x) 存的 -eop.npz，affectgpt(numpy1.x) 读会报 `No module named numpy._core` → auto_pipeline.sh 第⑤步做了 json 中转。
5. **npz 已存在会 SKIP**：重跑同名会跳过；测试用 `--save_tag` 写别的名，或先挪走旧 npz。
6. **环境历史坑**（旧交接遗留，现已解决，别重犯）：torch 装 cu121（非默认cu13）；`setuptools<81`；torchvision 缺 `functional_tensor` 要建 shim；EoP 用 transformers 不用 vLLM。
7. **GPU 显存**：AffectGPT 和 Qwen3 不能同时加载，阶段必须分开跑。
8. **评估 glob 顺序**：`run_evaluation_emosync.py` 用 `glob` 扫 `*-openset.npz`，顺序是文件系统序——`-openset`(自聚合) 在前、`-eop-openset`(Qwen3) 在后，两个 EW-F1 别对错版本。

---

## 11. 一句话现状
环境/全套流水线/caption/真分/全量1200 全部打通，最终分 **58.68(Qwen3-EoP) / 60.46(自聚合)**；刚加完"各模态情绪+置信度"功能并 3 样本验证通过（未接 EoP、未全量）。**最大悬而未决：1200 是否=论文 2025 测试集（影响能否和 55.58 硬比）。** 接手先读本文档，再看 `inference_emosync.py` / `auto_pipeline.sh` / `run_perspective_scores.py` 三个核心文件即可上手。
