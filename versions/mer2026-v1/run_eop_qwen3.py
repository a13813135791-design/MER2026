#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmoSync · EoP 独立阶段 —— 用 Qwen3-8B 当「裁判 LLM」聚合多视角预测
严格按论文 EmoSync(MRAC'25) §2.3 Ensemble-of-Perspective Strategy + page4 的 EoP prompt 实现。

为什么单独成一个脚本(而不是塞回 inference_emosync.py 主循环)：
  4090 只有 24G 显存，AffectGPT(~18G) 和 Qwen3-8B(vLLM 默认~0.9×24G) 装不下同一张卡。
  所以流程拆两段：
    第1段  inference_emosync.py  → 用 AffectGPT 跑 AUR+LSC，把中间结果存进 npz，跑完释放显存；
    第2段  本脚本                → 单独加载 Qwen3-8B，读那个 npz，做 EoP，写出最终情绪。
  两段不同时占显存，互不干扰。

输入：inference_emosync.py 产出的 npz（output/results-emosync/<ckpt_dir>/<epoch>.npz）
      其中含 name2aur = {name: {audio/face/text/multi: reason文本}}
            name2lsc = {name: {cot1, cot2, fewshot}}
输出：同名 + "-eop.npz"，把 name2reason 替换为 Qwen3-8B 的 EoP 最终结论（情绪 list 字符串）。
      原 npz 不动，便于和「AffectGPT 自聚合版 EoP」对照。

运行(在项目根 /opt/data/wlcc/MER2026_Track2，且已激活带新版 vllm/transformers 的 EoP 环境)：
  python run_eop_qwen3.py --npz output/results-emosync/<ckpt_dir>/<epoch>.npz
可选：--score_json scores.json 用 human 验证集实测的各视角 EW-F1 覆盖默认先验分(见阶段3)。
"""
import os
import re
import json
import argparse
import numpy as np

import config


# ─────────────────────────────────────────────────────────────────────────────
# 各视角的 performance score（论文 §2.3："incorporates each model's performance score"）
#   这里「模型」= 同一个 AffectGPT 的不同视角(AUR 4 路 + LSC)。
#   默认先用「固定先验分」让全链路先跑通：可靠性 multi(全模态) > text > face > audio，
#   LSC(自纠错后)给最高先验。阶段3会用 human 验证集分视角实测 EW-F1 来替换(--score_json)。
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PERSPECTIVE_SCORES = {
    'multi': 0.50,
    'text':  0.40,
    'face':  0.35,
    'audio': 0.30,
    'lsc':   0.55,
}

# ── 论文 page4 EoP prompt 原版(指令部分逐句照抄) ─────────────────────────────
EOP_INSTRUCTION = (
    "You are a professional emotion analysis expert. "
    "We collected multiple emotion recognition results for the same character, "
    "coming from different perspectives with different result scores. "
    "Your task is to determine the final emotional states of the character, "
    "by considering all provided predictions.\n"
    "Please follow these guidelines:\n"
    "(1) Prioritize higher scored predictions.\n"
    "(2) Merge consistent emotions across sources.\n"
    "(3) Discard conflicting or vague emotions.\n"
    "(4) Only include clear, psychological emotional terms like anxious, happy, frustrated.\n"
    "Return the final emotion list in Python list format. "
    "Do not include explanations or extra words. "
    "If no valid emotion is found, return: []\n"
)

# AUR 视角在 prompt 里的呈现顺序
AUR_ORDER = ['audio', 'face', 'text', 'multi']


def _is_bad(resp):
    """判断某路视角输出是否无效(无数据/报错/空)"""
    if resp is None:
        return True
    s = str(resp).strip()
    return (s == '') or s.startswith('[SKIP') or s.startswith('[ERROR') or s.startswith('[LSC')


def build_eop_prompt(sample_id, aur, lsc_fewshot, scores):
    """按论文格式拼 candidate predictions —— 每条带 score，对应 page4 的
       '- Prediction1 (score=...): {emotional states1}'。"""
    cands = []
    idx = 1
    for key in AUR_ORDER:
        resp = aur.get(key) if isinstance(aur, dict) else None
        if _is_bad(resp):
            continue
        cands.append(f"- Prediction{idx} (score={scores.get(key, 0.3):.2f}): {str(resp).strip()}")
        idx += 1
    # LSC 自纠错(few-shot CoT)结果，作为最后一个高可靠视角
    if not _is_bad(lsc_fewshot):
        cands.append(f"- Prediction{idx} (score={scores.get('lsc', 0.5):.2f}): {str(lsc_fewshot).strip()}")

    cand_block = "\n".join(cands) if cands else "(no valid prediction)"
    return (
        EOP_INSTRUCTION +
        f"CharacterID: {sample_id}\n"
        f"Candidate predictions:\n{cand_block}\n"
    )


def clean_qwen3_output(text):
    """Qwen3 即使关了 thinking，偶尔仍带 <think>…</think>；并清掉多余空白。"""
    if text is None:
        return '[]'
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.replace('<think>', '').replace('</think>', '')
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description='EmoSync EoP stage — Qwen3-8B judge (paper-faithful)')
    ap.add_argument('--npz', required=True,
                    help='inference_emosync.py 产出的 npz(含 name2aur/name2lsc)')
    ap.add_argument('--qwen3', default=config.PATH_TO_LLM.get('Qwen3', 'my_affectgpt/models/Qwen3-8B'),
                    help='Qwen3-8B 路径(默认读 config.PATH_TO_LLM["Qwen3"])')
    ap.add_argument('--score_json', default=None,
                    help='可选：各视角实测 score 的 json，覆盖默认先验分')
    ap.add_argument('--gpu_mem', type=float, default=0.85, help='vLLM gpu_memory_utilization')
    ap.add_argument('--max_model_len', type=int, default=4096, help='vLLM 上下文上限(省显存)')
    ap.add_argument('--max_samples', type=int, default=0, help='调试：只跑前 N 条')
    args = ap.parse_args()

    # ── 读 score ──
    scores = dict(DEFAULT_PERSPECTIVE_SCORES)
    if args.score_json and os.path.exists(args.score_json):
        with open(args.score_json, 'r', encoding='utf-8') as f:
            scores.update(json.load(f))
        print(f'[EoP] perspective scores (from {args.score_json}): {scores}')
    else:
        print(f'[EoP] perspective scores (default prior): {scores}')

    # ── 读 npz(中间结果) ──
    if not os.path.exists(args.npz):
        raise FileNotFoundError(f'npz 不存在: {args.npz}')
    data = np.load(args.npz, allow_pickle=True)
    name2aur = data['name2aur'].item()
    name2lsc = data['name2lsc'].item() if 'name2lsc' in data.files else {}
    name2reason_old = data['name2reason'].item() if 'name2reason' in data.files else {}

    names = list(name2aur.keys())
    if args.max_samples:
        names = names[:args.max_samples]
    print(f'[EoP] {len(names)} samples to aggregate from {args.npz}')

    # ── 构造所有 EoP prompt ──
    prompts, used_names = [], []
    for name in names:
        aur = name2aur.get(name, {})
        lsc = name2lsc.get(name, {})
        lsc_fewshot = None
        if isinstance(lsc, dict):
            lsc_fewshot = lsc.get('fewshot') or lsc.get('cot1')
        prompts.append(build_eop_prompt(name, aur, lsc_fewshot, scores))
        used_names.append(name)

    # ── 加载 Qwen3-8B(transformers)，逐样本生成 ──
    #   注：本机驱动(535, CUDA≤12.2)带不动新版 vLLM 的 cu13 依赖，故 EoP 改用 transformers。
    #   EoP 是一次性聚合(每样本一次)，逐条 generate 足够，不需要 vLLM 的高吞吐。
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'[EoP] loading Qwen3-8B from {args.qwen3} (transformers) ...')
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen3, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    # ── 逐样本生成，写回 name2reason ──
    name2reason = dict(name2reason_old)
    for i, (name, prompt) in enumerate(zip(used_names, prompts)):
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 专属：关掉 thinking 模式(老版 transformers 无此参数则回退)
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                 temperature=0.7, top_p=0.8, repetition_penalty=1.05,
                                 pad_token_id=tokenizer.eos_token_id)
        out_ids = gen[0][inputs['input_ids'].shape[1]:]   # 只取新生成部分
        resp = clean_qwen3_output(tokenizer.decode(out_ids, skip_special_tokens=True))
        name2reason[name] = resp
        print(f'  [EoP {i+1}/{len(used_names)}] {name}: {resp[:90]}')

    out_path = args.npz[:-4] + '-eop.npz'
    np.savez_compressed(out_path,
                        name2reason=name2reason,
                        name2aur=name2aur,
                        name2lsc=name2lsc)
    print(f'[EoP] done. saved -> {out_path}  ({len(used_names)} samples)')


if __name__ == '__main__':
    main()
