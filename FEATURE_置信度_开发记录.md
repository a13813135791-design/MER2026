# 功能开发记录：AUR 各模态输出"情绪 + 置信度"

> 目标：在不改输入、不换模型的前提下，让 AUR 的每条模态流不只给情绪类别，还给**每个情绪的置信度**。
> 开始：2026-06-24（服务器时间约 04:5x / 本地 15:0x）

## 1. 可行性判断（已确认：可行）
- AffectGPT 是生成式模型，提示词+输出解析都可控 → 让它按"情绪: 置信度"输出可行。
- 真正难点是"置信度准不准"，不是"能不能吐数字"。

## 2. 方案选择
- 候选：A 模型自报分(1次生成,省,弱校准) / B token概率(难,不选) / C 多次采样数频率(贵,较准)。
- **用户拍板：先做 A**（最高效，先看效果，必要时再升 C）。

## 3. 实现内容（A 方案）
- 新增 `AUR_CONF_PROMPT`：要求"识别所有情绪 + 各给 0~1 置信度，按 `情绪: 分数` 逗号分隔输出"。
- 新增 `parse_emotion_confidence()`：正则把输出解析成 `{情绪: 置信度}`。
- `run_aur` 改用 `AUR_CONF_PROMPT` 提问；模态输出仍存为文本（含置信度），另存解析后的 `{情绪:置信度}`。
- npz 增存 `name2aur_conf`。
- 新增 `--save_tag`：测试写到独立文件，不覆盖已跑好的全量结果。
- 原文件备份为 `.bak_preconf`。

## 4. 样本测试结果（3 样本，写到独立 _conftest.npz，未碰全量结果）

### 使用的模型
只用 AffectGPT（`checkpoint_000035_loss_0.716.pth`，LLaMA+LoRA ~7.66B）。4 模态全是同一个它。本次未用 Qwen2.5/Qwen3。

### 提示词（4 模态指令相同 = AUR_CONF_PROMPT；仅输入描述段不同）
"Please identify all possible emotional states of the character, and assign each one a confidence score between 0.0 and 1.0 (higher means more certain). Output ONLY a comma-separated list in the exact format 'emotion: score', and nothing else. Example: anxious: 0.8, tense: 0.5, calm: 0.2"
- audio 段: `The audio content is: <Audio><AudioHere></Audio>.`
- face 段:  `...extract faces...: <Video><FaceHere></Video>.`
- text 段:  `The subtitle is: <Subtitle>{字幕}</Subtitle>.`
- multi 段: `<Multi>…<Audio>…<Face>…<Subtitle>…<Caption>…`（带 caption）

### 运行命令（产出 _conftest.npz；只推理、未跑 EW-F1 评分）
```
CUDA_VISIBLE_DEVICES=0 /opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python -u inference_emosync.py \
  --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_lz.yaml \
  --dataset MER2026OV --train_debug --options 'inference.test_epoch=35' \
  --max_samples 3 --save_tag _conftest
```
日志: /tmp/conftest.log

### 三个样本各模态真实输出
**00000062** (GT: shocked,care,serious,surprised,worry)
- audio: `anxious:0.9, tense:0.1, calm:0.0`
- face:  `tense:0.9, anxious:0.8, calm:0.1`
- text:  `rumored:1.0, tense:0.5, anxious:0.5, calm:0.1`
- multi: `anxious:0.9, tense:0.7, calm:0.1, shocked:0.8, overwhelmed:0.6, confused:0.5, sad:0.4, angry:0.3`

**00000181** (GT: serious,disapprove,accuse,complain)
- audio: `anxious:0.9, tense:0.1, calm:0.0`
- face:  `calm:0.9, neutral:0.1`
- text:  `concern:0.9, tension:0.4, disappointment:0.3, frustration:0.2, anger:0.1, sadness:0.1`
- multi: `anger:0.9, frustration:0.7, stress:0.6, concern:0.4, dissatisfaction:0.3`

**00001616** (GT: anxious,afraid,worried,concerned,uneasy,nervous)
- audio: `anxious:0.8, tense:0.1, calm:0.1`
- face:  `anger:0.9, sadness:0.1`
- text:  `ouch:1.0`
- multi: `anxious:0.8, tense:0.5, calm:0.2`

3 样本 × 4 模态全部解析成功（每模态非空）。

**好的方面**：格式干净、解析稳、置信度有区分度（0.9~0.0，不是一刀切），主情绪高/备选低很合理。
**待优化**：偶尔混入非情绪词（如 `rumored: 1.0`）——解析器不挑词，照单全收。可加"情绪词表过滤"清掉噪声。
**没动到的**：全量 58.68/60.46 结果完好（测试写在独立 _conftest 文件）。

## 5. 结论
A 方案（模型自报置信度，1 次生成）**可行、已跑通**。代码：`AUR_CONF_PROMPT` + `parse_emotion_confidence` + `name2aur_conf`(存进 npz) + `--save_tag`。原文件备份 `.bak_preconf`。
**下一步（待用户定）**：① 把置信度接入 EoP（让 Qwen3 裁判用上每情绪的置信度）；② 加情绪词表过滤去噪；③ 决定是否在全量上重跑。
