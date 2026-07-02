#!/usr/bin/env python3
"""
EmoSync inference for MER2026 Track2
三个核心模块:
  AUR (Asymmetric Unimodal Reasoning) - 4路并行单模态推理
  LSC (Logical Self-Correction)        - 逻辑自我纠错
  EoP (Ensemble of Predictions)        - 集成投票
"""
import os, sys, glob, argparse, copy, json, contextlib
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import *
from my_affectgpt.processors import BaseProcessor

import config
from toolkit.utils.read_files import func_read_key_from_csv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (同 inference_hybird.py)
# ─────────────────────────────────────────────────────────────────────────────

def search_for_ckpt_root(root_candidates):
    if not root_candidates:
        return ''
    maxcount, targetroot = 0, ''
    for root in root_candidates:
        count = len([p for p in os.listdir(root) if p.startswith('checkpoint_')])
        if count > maxcount:
            maxcount, targetroot = count, root
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    print(f'Targetroot: {targetroot}  epoch range: 0-{maxcount-1}')
    print(f'Last ckpt time: {datetime.fromtimestamp(Path(last_file).stat().st_ctime)}')
    return targetroot


def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    if inference_cfg.test_epoch != 'xxx':
        ckpts = glob.glob('%s/*%06d*.pth' % (ckpt3_root, int(inference_cfg.test_epoch)))
        assert len(ckpts) == 1
        return [ckpts[0]]
    elif inference_cfg.test_epochs == 'xxx-xxx':
        return [sorted(glob.glob('%s/*.pth' % ckpt3_root))[-1]]
    else:
        start, end = inference_cfg.test_epochs.split('-')
        skip = int(inference_cfg.skip_epoch)
        whole = []
        for epoch in range(int(start), int(end)+1):
            if epoch % skip == 0:
                ckpts = glob.glob('%s/*%06d*.pth' % (ckpt3_root, epoch))
                assert len(ckpts) == 1
                whole.append(ckpts[0])
        return whole


def get_face_or_frame(datasets_cfg, outside):
    if outside is not None:
        return outside
    candidates = []
    if 'mercaptionplus' in datasets_cfg:
        candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'human' in datasets_cfg:
        candidates.append(datasets_cfg['human'].face_or_frame)
    assert len(set(candidates)) == 1
    return list(set(candidates))[0]


# ─────────────────────────────────────────────────────────────────────────────
# Module 1: AUR — Asymmetric Unimodal Reasoning
# 对同一个样本分别用 4 种输入组合跑推理，各自得到一个情感预测
# ─────────────────────────────────────────────────────────────────────────────

# 每条路对应 base_dataset.py 里 get_prompt_for_multimodal 支持的 mode 名
AUR_MODES = {
    'audio': 'audioonly',                    # 只听声音
    'face':  'faceonly',                     # 只看人脸
    'text':  'textonly',                     # 只看字幕文本
    'multi': 'multiface_audio_face_text',    # 三路合并 (等同 baseline)
}

GEN_KWARGS = dict(num_beams=1, temperature=1, do_sample=False, top_p=0.9,
                  max_new_tokens=800, max_length=2000)

# caption 生成专用：确定性、短一点(描述不需要 800 token)
CAPTION_GEN_KWARGS = dict(num_beams=1, temperature=1, do_sample=False, top_p=0.9,
                          max_new_tokens=256, max_length=2000)

# 论文 prompt 有 <Caption> 字段(视频画面/场景描述)。官方数据没提供 caption，
# 故按论文造法(模型生成描述)由 AffectGPT 自己描述视频，再喂进多模态 prompt。
# 只描述客观内容、先不下情绪判断，避免和后续 AUR/LSC 的情绪推理重复。
CAPTION_PROMPT = (
    "Please objectively describe this video clip in detail: the scene and setting, "
    "the people present, their actions and body language, facial expressions, and the "
    "tone of the voice. Describe only what is observed; do not judge emotions yet."
)


def run_caption(chat, img_list, dataset_cls, subtitle):
    """让 AffectGPT 给该样本生成一段客观视频描述(caption)，供多模态 prompt 使用。
       失败则返回空串，下游 get_prompt_for_multimodal 会自动省略 <Caption> 字段。"""
    prompt = dataset_cls.get_prompt_for_multimodal(
        'multiface_audio_face_text', subtitle, CAPTION_PROMPT)
    try:
        cap = chat.answer_sample(prompt=prompt, img_list=img_list, **CAPTION_GEN_KWARGS)
        return str(cap).strip()
    except Exception as e:
        return ''


def _make_uni_img_list(img_list_full, modality):
    """只保留指定模态的 embedding，其余置 None"""
    uni = {k: None for k in img_list_full}
    if modality == 'audio':
        uni['audio'] = img_list_full['audio']
    elif modality == 'face':
        uni['face']  = img_list_full['face']
        uni['multi'] = img_list_full['multi']
    elif modality == 'text':
        pass  # 全 None，字幕通过 prompt 文本传入
    elif modality == 'multi':
        return dict(img_list_full)
    return uni


def run_aur(chat, img_list, dataset_cls, subtitle, user_message, caption=''):
    """跑 4 路 AUR，返回 {modality: response} 字典。
       caption 只给全模态(multi)那路；单模态路保持纯净，不喂视频描述。"""
    responses = {}
    for modality, mode in AUR_MODES.items():
        if modality in ('audio', 'face') and img_list.get(modality) is None:
            responses[modality] = '[SKIP:no_data]'
            continue
        uni = _make_uni_img_list(img_list, modality)
        cap = caption if modality == 'multi' else ''
        prompt = dataset_cls.get_prompt_for_multimodal(mode, subtitle, user_message, caption=cap)
        try:
            responses[modality] = chat.answer_sample(
                prompt=prompt, img_list=uni, **GEN_KWARGS)
        except Exception as e:
            responses[modality] = f'[ERROR:{e}]'
    return responses


# ─────────────────────────────────────────────────────────────────────────────
# Module 2: LSC — Logical Self-Correction (Logical 自我纠错)
# 严格按论文 Section 2.2 实现：
#   第一阶段(本步)：零样本 CoT，用两个不同的 CoT prompt 各跑一次多模态推理，
#                    强制模型"先分步推理、再给情绪"，从而避免对立情绪共现(sad+happy)。
#   第二阶段(下一步)：少样本 CoT，挑高质量案例当 in-context 范例再精炼。
# 注：论文 prompt 里的 <Caption> 字段我们数据没有，故省略(不影响 CoT 机制)。
# ─────────────────────────────────────────────────────────────────────────────

# 论文 PromptType1：结构化 4 步推理(Subtitle→Audio→Video→Synthesis)
LSC_COT_PROMPT1 = (
    "Please answer the question based on all provided information by reasoning "
    "step-by-step in the following format: "
    "(1) Subtitle Analysis: Examine the literal meaning and emotional undertone "
    "of the subtitle. Consider the speaker's role and possible context. "
    "(2) Audio Analysis: Analyze tone, pitch, intensity, pace, and other vocal "
    "cues to infer emotion. "
    "(3) Video Analysis: Use facial expressions, gestures, and scene context to "
    "deduce visual emotional signals. "
    "(4) Synthesis: Integrate findings across modalities. Identify consistent or "
    "conflicting signals. Infer a complete set of likely emotional states. "
    "After reasoning, please recognize all possible emotional states of the character."
)

# 论文 PromptType2：简单 "Let's think step-by-step"
LSC_COT_PROMPT2 = (
    "Please recognize all possible emotional states of the character. "
    "Let's think step-by-step."
)


def run_lsc(chat, img_list, dataset_cls, subtitle, caption=''):
    """
    LSC 第一阶段：零样本 CoT。
    用论文的两个 CoT prompt(结构化4步 + 简单分步)各跑一次多模态推理，
    返回 (cot1, cot2) 两条"推理过程 + 情绪预测"。
    论文的这两个 CoT prompt 本就含 <Caption> 字段，故把 caption 喂进去。
    """
    p1 = dataset_cls.get_prompt_for_multimodal(
        'multiface_audio_face_text', subtitle, LSC_COT_PROMPT1, caption=caption)
    p2 = dataset_cls.get_prompt_for_multimodal(
        'multiface_audio_face_text', subtitle, LSC_COT_PROMPT2, caption=caption)
    try:
        cot1 = chat.answer_sample(prompt=p1, img_list=img_list, **GEN_KWARGS)
    except Exception as e:
        cot1 = f'[LSC-CoT1 error: {e}]'
    try:
        cot2 = chat.answer_sample(prompt=p2, img_list=img_list, **GEN_KWARGS)
    except Exception as e:
        cot2 = f'[LSC-CoT2 error: {e}]'
    return cot1, cot2


# ── LSC 第二阶段：少样本 CoT（self-consistency 挑范例，论文 Section 2.2 后半）──────
# 常见情绪词表：用于从输出里识别"真正的情绪词"，过滤掉 the/is 之类
EMOTION_LEXICON = {
    'happy','happiness','joy','joyful','glad','pleased','delighted','cheerful',
    'excited','enthusiastic','eager','hopeful','hope','proud','pride','grateful',
    'sad','sadness','sorrow','unhappy','depressed','grief','miserable','disappointed',
    'disappointment','angry','anger','furious','rage','irritated','annoyed','annoyance',
    'frustrated','frustration','mad','resentful','resentment','reproach','blame',
    'anxious','anxiety','nervous','tense','tension','worried','worry','stressed','stress',
    'uneasy','unease','agitated','overwhelmed','helpless','desperate','distress','distressed',
    'fear','afraid','scared','terrified','fearful','panic',
    'surprised','surprise','shocked','shock','astonished','amazed',
    'calm','relaxed','peaceful','content','serene','satisfied','satisfaction',
    'disgust','disgusted','contempt','disdain','dissatisfied','dissatisfaction',
    'confused','confusion','doubtful','hesitant','concerned','concern','sympathetic',
    'embarrassed','ashamed','guilty','guilt','bored','boredom','indifferent','apathetic',
}

# 对立情绪组：同一条最终预测里不该同时出现（self-consistency 判据）
OPPOSITE_PAIRS = [
    ({'happy','happiness','joy','joyful','glad','pleased','delighted','cheerful'},
     {'sad','sadness','sorrow','unhappy','depressed','grief','miserable'}),
    ({'calm','relaxed','peaceful','content','serene','satisfied'},
     {'anxious','anxiety','nervous','tense','tension','agitated','stressed','stress'}),
    ({'calm','relaxed','peaceful','content','serene','satisfied'},
     {'angry','anger','furious','rage','irritated','annoyed','frustrated','frustration'}),
    ({'excited','enthusiastic','eager'},
     {'bored','boredom','indifferent','apathetic'}),
    ({'content','satisfied','satisfaction','pleased'},
     {'dissatisfied','dissatisfaction','disappointed','disappointment'}),
]


def extract_emotion_words(text):
    """从文本里抽出属于情绪词表的词（小写）"""
    import re
    words = set(re.findall(r"[a-zA-Z]+", str(text).lower()))
    return words & EMOTION_LEXICON


def get_conclusion(text):
    """取输出里"最终情绪结论"那一段（最后一次出现 emotional state(s) 之后）"""
    low = str(text).lower()
    best = -1
    for marker in ['emotional states of the character', 'emotional state of the character',
                   'emotional states are', 'emotional state is', 'states of the character']:
        idx = low.rfind(marker)
        if idx > best:
            best = idx
    return str(text)[best:] if best != -1 else str(text)[-200:]


def is_self_consistent(emotion_words):
    """给定情绪词集合，若有任一对立情绪组同时出现则判为不自洽"""
    for pos, neg in OPPOSITE_PAIRS:
        if (emotion_words & pos) and (emotion_words & neg):
            return False
    return True


def is_mostly_english(text):
    """判断文本是否以英文为主（用于挑英文 CoT 当范例，避免中文混进英文 prompt）"""
    import re
    n_en  = len(re.findall(r'[a-zA-Z]', str(text)))
    n_cjk = len(re.findall(r'[一-鿿]', str(text)))
    return n_en >= n_cjk


def select_lsc_exemplars(pass1_results, k=2):
    """
    从第一阶段结果里挑"高质量自洽案例"当少样本范例（self-consistency 挑选）：
      - 综合 3 个来源(AUR多模态 / CoT1 / CoT2)的情绪词；
      - "至少在 2 个来源出现"的情绪 = 高置信一致集合 agree；
      - agree 必须自洽（无对立情绪共现）；
      - 取一条以英文为主、足够长的 CoT 当范例推理文本。
    按 agree 大小排序取前 k 个。
    """
    from collections import Counter
    scored = []
    for name, d in pass1_results.items():
        cot1, cot2 = str(d['cot1']), str(d['cot2'])
        aur_multi  = str(d['aur'].get('multi', ''))
        cnt = Counter()
        for src in (extract_emotion_words(cot1),
                    extract_emotion_words(cot2),
                    extract_emotion_words(aur_multi)):
            for w in src:
                cnt[w] += 1
        agree = {w for w, c in cnt.items() if c >= 2}   # 至少 2 个来源认同
        if not agree or not is_self_consistent(agree):
            continue
        # 选一条英文为主、非垃圾的 CoT 当范例推理文本
        reasoning = cot1 if (len(cot1) >= 60 and is_mostly_english(cot1)) else cot2
        if len(reasoning) < 60 or not is_mostly_english(reasoning):
            continue
        scored.append((len(agree), name, reasoning, agree))
    scored.sort(key=lambda x: x[0], reverse=True)
    exemplars = []
    for _, name, reasoning, agree in scored[:k]:
        exemplars.append({
            'reasoning': reasoning.strip()[:400],
            'emotions':  ', '.join(sorted(agree)),
        })
    return exemplars


def build_lsc_fewshot_message(exemplars):
    """把自洽范例拼成少样本 CoT 的 user_message（范例 + 论文4步指令）"""
    blocks = []
    for i, ex in enumerate(exemplars, 1):
        blocks.append(f"Example {i}:\n{ex['reasoning']}\n"
                      f"Final emotional states: {ex['emotions']}")
    exemplar_text = "\n\n".join(blocks)
    return (
        "Here are some high-quality examples of step-by-step emotion reasoning whose "
        "final emotions are self-consistent (no contradictory emotions co-occur):\n\n"
        + exemplar_text
        + "\n\nNow, following the same step-by-step reasoning style as the examples "
          "above, and making sure your final emotions are NOT mutually contradictory, "
          "analyze the current video. " + LSC_COT_PROMPT1
    )


def run_lsc_fewshot(chat, img_list, dataset_cls, subtitle, fewshot_message, caption=''):
    """LSC 第二阶段：少样本 CoT，单次推理（带自洽范例）"""
    prompt = dataset_cls.get_prompt_for_multimodal(
        'multiface_audio_face_text', subtitle, fewshot_message, caption=caption)
    try:
        return chat.answer_sample(prompt=prompt, img_list=img_list, **GEN_KWARGS)
    except Exception as e:
        return f'[LSC-fewshot error: {e}]'


def encode_sample(chat, dataset_cls, name):
    """读取并编码一个样本的所有模态，返回 (img_list, status)；失败时 img_list=None"""
    sample     = {'name': name}
    video_path = dataset_cls._get_video_path(sample)
    audio_path = dataset_cls._get_audio_path(sample)
    face_npy   = dataset_cls._get_face_path(sample)
    if not os.path.exists(audio_path):
        return None, 'audio missing'
    if not os.path.exists(face_npy):
        return None, 'face missing'
    try:
        sample_data = dataset_cls.read_frame_face_audio_text(
            video_path, face_npy, audio_path, None)
    except Exception as e:
        return None, f'data load error: {e}'
    audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
    face_hiddens,  face_llms  = chat.postprocess_face(sample_data)
    frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
    _,             image_llms = chat.postprocess_image(sample_data)
    _,             multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
    img_list = {'audio': audio_llms, 'frame': frame_llms, 'face': face_llms,
                'image': image_llms, 'multi': multi_llms}
    return img_list, 'ok'


# ─────────────────────────────────────────────────────────────────────────────
# Module 3: EoP — Ensemble of Predictions
# 把 4 路 AUR + 1 路 LSC 的结果汇总，让模型给出最终答案
# (用 AffectGPT 自身做纯文本推理，无需额外加载第二个模型)
# ─────────────────────────────────────────────────────────────────────────────

def run_eop(chat, aur_responses, lsc_response, subtitle):
    """把所有预测汇总后让模型集成投票，得到最终结果"""
    summary = (
        f'1. Audio-only prediction:  {aur_responses.get("audio", "N/A")}\n'
        f'2. Face-only prediction:   {aur_responses.get("face", "N/A")}\n'
        f'3. Text-only prediction:   {aur_responses.get("text", "N/A")}\n'
        f'4. Multimodal prediction:  {aur_responses.get("multi", "N/A")}\n'
        f'5. Self-corrected result:  {lsc_response}'
    )
    eop_prompt = (
        f'###Human: You are an expert emotion analyst. '
        f'A video clip was analyzed from five different perspectives:\n'
        f'{summary}\n'
        f'The subtitle is: <Subtitle>{subtitle}</Subtitle>. '
        f'Based on all perspectives above, synthesize and recognize all possible '
        f'emotional states of the character. '
        f'Please recognize all possible emotional states of the character. '
        f'###Assistant: '
    )
    empty_img = {k: None for k in ['audio', 'frame', 'face', 'image', 'multi']}
    try:
        return chat.answer_sample(prompt=eop_prompt, img_list=empty_img, **GEN_KWARGS)
    except Exception as e:
        print(f'  [EoP error: {e}] fallback to LSC response')
        return lsc_response


# ─────────────────────────────────────────────────────────────────────────────
# 方案2: 基于模型 logits 的闭集情绪支持度打分 (B1 closed-set / verbalizer scoring)
# 不让模型自报 0~1 置信度(方案1)，而是对候选情绪词做 teacher-forcing 序列对数似然打分，
# 经长度归一 + PMI 校准 + 候选 softmax 得到每个单模态的 {情绪: 支持度}。
# 按需求只跑 audio/face/text 三条单模态(不含 multi)。
# ─────────────────────────────────────────────────────────────────────────────

# 三条单模态 → base_dataset.py 的 mode 名
AUR_LOGITS_MODES = {
    'audio': 'audioonly',
    'face':  'faceonly',
    'text':  'textonly',
}

# 打分锚点：复刻微调答案模板 "The character's emotional state is {label}."。
# get_prompt_for_multimodal 末尾是 "###Assistant: "，故把 lead-in 接上后 prefix 收尾在 "...is"，
# 候选词作为 " {label}" 的自然续写被打分(token 位置与训练时一致)。
SCORE_QUESTION = "Please recognize all possible emotional states of the character."
SCORE_LEAD_IN  = "The character's emotional state is"


@torch.no_grad()
def build_prefix_embeds(chat, prompt, img_list, max_length=2000):
    """复刻 Chat.answer_sample 的 step1-2：把多模态 patch token 替换成各模态特征 embedding，
       拼成 inputs_embeds，但不调用 generate。返回 (prefix_embeds[1,L,H], attn[1,L])。"""
    tok = chat.tokenizer
    IDS = {
        'frame': tok.get_vocab()[config.DEFAULT_FRAME_PATCH_TOKEN],
        'face':  tok.get_vocab()[config.DEFAULT_FACE_PATCH_TOKEN],
        'audio': tok.get_vocab()[config.DEFAULT_AUDIO_PATCH_TOKEN],
        'multi': tok.get_vocab()[config.DEFAULT_MULTI_PATCH_TOKEN],
        'image': tok.get_vocab()[config.DEFAULT_IMAGE_PATCH_TOKEN],
    }
    prompt = chat.replace_token_for_multimodal(prompt)
    input_id = chat.to_token_ids(prompt, max_length)
    attention_mask = input_id.ne(tok.pad_token_id).to(chat.device)

    temp_input_id = copy.deepcopy(input_id).to(chat.device)
    for pid in IDS.values():
        temp_input_id[temp_input_id == pid] = 0
    cur = chat.model.llama_model.model.model.embed_tokens(temp_input_id)
    cur_ids = input_id
    for (modality, pid, n) in [('frame', IDS['frame'], chat.num_video_query_token),
                               ('face',  IDS['face'],  chat.num_video_query_token),
                               ('audio', IDS['audio'], chat.num_audio_query_token),
                               ('multi', IDS['multi'], chat.num_multi_query_token),
                               ('image', IDS['image'], chat.num_image_query_token)]:
        if (cur_ids == pid).sum() != 0:
            embeds = img_list[modality]
            assert embeds is not None, f'[build_prefix_embeds] missing embeds for {modality}'
            start = torch.where(cur_ids == pid)[0][0]
            cur = torch.cat((cur[:start], embeds[0], cur[start + n:]), dim=0)
    return cur.unsqueeze(0), attention_mask.unsqueeze(0)


def build_candidate_vocab(chat, wheel_path='emotion_wheel/wheel_mapping.npz'):
    """候选情绪词表 E = 5 个情感轮 level1 的 keys 并集去重(与评测 EW-F1 同源)。
       预分词缓存 {label: token_ids}，候选以 ' '+label 分词(对齐 '...is {label}' 的训练 token)。"""
    d = np.load(wheel_path, allow_pickle=True)
    wmw = d['wheel_map_whole'].item()
    labels = set()
    for wk in wmw:
        labels |= set(wmw[wk]['level1'].keys())
    labels = sorted(labels)
    label2ids = {lab: chat.tokenizer(' ' + lab, add_special_tokens=False)['input_ids']
                 for lab in labels}
    return labels, label2ids


def _autocast_ctx(model):
    fn = getattr(model, 'maybe_autocast', None)
    return fn() if callable(fn) else contextlib.nullcontext()


@torch.no_grad()
def score_candidates_batched(chat, prefix_embeds, attn, labels, label2ids, chunk=32):
    """对候选标签算长度归一对数似然 s_raw（teacher forcing）。返回 {label: s_raw}。
       优化(与逐候选打分数学等价)：候选首 token 的 logp 只由 prefix 最后位 logits 决定，
       故先做一次 prefix 前向给所有"单 token 候选"一次性打分；仅多 token 候选才需要
       额外的 teacher-forcing 增量前向。避免对每个候选重算 prefix。"""
    model = chat.model
    emb = model.llama_model.model.model.embed_tokens
    L, H = prefix_embeds.size(1), prefix_embeds.size(2)
    scores = {}

    # (1) 一次 prefix 前向 → 最后位 log_softmax，对所有候选的"首 token"通用
    with _autocast_ctx(model):
        out = model.llama_model(inputs_embeds=prefix_embeds, attention_mask=attn,
                                use_cache=False, return_dict=True)
        last_logp = torch.log_softmax(out.logits[0, L - 1, :].float(), dim=-1)  # [V]

    multi = []
    for lab in labels:
        ids = label2ids[lab]
        if len(ids) == 1:                       # 单 token：s_raw = logp(首 token)
            scores[lab] = float(last_logp[ids[0]].item())
        else:
            multi.append(lab)

    # (2) 多 token 候选：teacher forcing 批量前向(数量很少)
    for s in range(0, len(multi), chunk):
        block = multi[s:s + chunk]
        seqs = [label2ids[l] for l in block]
        T = max(len(x) for x in seqs)
        B = len(block)
        cand_ids = torch.full((B, T), chat.tokenizer.pad_token_id,
                              device=chat.device, dtype=torch.long)
        cand_len = torch.tensor([len(x) for x in seqs], device=chat.device)
        for i, x in enumerate(seqs):
            cand_ids[i, :len(x)] = torch.tensor(x, device=chat.device)
        with _autocast_ctx(model):
            cand_emb = emb(cand_ids)                                   # [B,T,H]
            pre = prefix_embeds.expand(B, L, H)
            full_emb = torch.cat([pre, cand_emb], dim=1)              # [B,L+T,H]
            cand_attn = (torch.arange(T, device=chat.device)[None, :] < cand_len[:, None]).long()
            full_attn = torch.cat([attn.expand(B, L).long(), cand_attn], dim=1)
            out = model.llama_model(inputs_embeds=full_emb,
                                    attention_mask=full_attn,
                                    use_cache=False, return_dict=True)
            step = out.logits[:, L - 1:L - 1 + T, :].float()          # [B,T,V]
        logp = torch.log_softmax(step, dim=-1)
        tok_lp = logp.gather(-1, cand_ids.unsqueeze(-1)).squeeze(-1)   # [B,T]
        mask = (torch.arange(T, device=chat.device)[None, :] < cand_len[:, None])
        sums = (tok_lp * mask).sum(dim=1)
        norm = sums / cand_len.clamp(min=1)                            # 长度归一
        for i, l in enumerate(block):
            scores[l] = float(norm[i].item())
    return scores


@torch.no_grad()
def precompute_neutral_scores(chat, dataset_cls, labels, label2ids):
    """§4.6 PMI 中性分：无任何模态证据(textonly + 空字幕 + 空 img_list)下每个候选词的
       纯语言先验 s_raw(e|∅)。样本无关，全局算一次缓存。"""
    empty = {k: None for k in ['audio', 'frame', 'face', 'image', 'multi']}
    prompt = dataset_cls.get_prompt_for_multimodal('textonly', '', SCORE_QUESTION) + SCORE_LEAD_IN
    pre, attn = build_prefix_embeds(chat, prompt, empty)
    return score_candidates_batched(chat, pre, attn, labels, label2ids)


@torch.no_grad()
def run_aur_logits(chat, img_list, dataset_cls, subtitle, labels, label2ids,
                   neutral, tau=1.0, topk=15, caption=''):
    """对 audio/face/text 三条单模态用 logits 打分候选情绪词。
       返回 {modality: {'raw':{label:s_raw}, 'support':{label:prob}, 'topk':[[label,prob],...]}}。
       support = softmax((s_raw - neutral)/tau)（长度归一 + PMI + 温度 softmax）。"""
    out = {}
    for modality, mode in AUR_LOGITS_MODES.items():
        if modality in ('audio', 'face') and img_list.get(modality) is None:
            out[modality] = {}
            continue
        uni = _make_uni_img_list(img_list, modality)
        prompt = dataset_cls.get_prompt_for_multimodal(mode, subtitle, SCORE_QUESTION, caption=caption) + SCORE_LEAD_IN
        pre, attn = build_prefix_embeds(chat, prompt, uni)
        raw = score_candidates_batched(chat, pre, attn, labels, label2ids)
        pmi = torch.tensor([raw[l] - neutral[l] for l in labels]) / tau
        prob = torch.softmax(pmi, dim=0)
        support = {labels[i]: float(prob[i]) for i in range(len(labels))}
        top = sorted(support.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        out[modality] = {'raw': raw, 'support': support, 'topk': [[l, p] for l, p in top]}
    return out


def run_logits_scoring(chat, dataset_cls, name2subtitle, args, ckpt3_root, cfg):
    """方案2 运行入口：在 track2_test.csv 上对三条单模态做 logits 闭集打分并落盘。
       只跑修改的代码(不含 caption/LSC/EoP)。"""
    test_csv   = args.logits_csv if getattr(args, 'logits_csv', None) else os.path.join(config.DATA_DIR['MER2026'], 'track2_test.csv')
    test_names = func_read_key_from_csv(test_csv, 'name')
    print(f'[logits] {len(test_names)} samples from {os.path.basename(test_csv)}')
    import pandas as _pd
    cap_csv = getattr(args, 'caption_csv', None)
    name2caption = {}
    if cap_csv and os.path.exists(cap_csv):
        _df = _pd.read_csv(cap_csv)
        if 'caption' in _df.columns:
            name2caption = {str(r['name']): ('' if _pd.isna(r['caption']) else str(r['caption']))
                            for _, r in _df.iterrows()}
        _hit = sum(1 for n in test_names if name2caption.get(n))
        print(f'[logits] captions loaded from {os.path.basename(cap_csv)}; hit {_hit}/{len(test_names)}')
    else:
        print('[logits] no caption_csv -> captions disabled')
    if args.start_idx:
        test_names = test_names[args.start_idx:]
    if args.max_samples:
        test_names = test_names[:args.max_samples]
        print(f'[logits] limited to {len(test_names)} samples')

    print('[logits] building candidate vocab + neutral (PMI) scores ...')
    labels, label2ids = build_candidate_vocab(chat)
    neutral = precompute_neutral_scores(chat, dataset_cls, labels, label2ids)
    print(f'[logits] candidate labels: {len(labels)}; tau={args.score_tau}')

    save_root = os.path.join('output/results-emosync', os.path.basename(ckpt3_root))
    os.makedirs(save_root, exist_ok=True)
    epoch = os.path.basename(cfg.model_cfg.ckpt_3)[:-4]
    save_path = f'{save_root}/{epoch}_logits.npz'
    json_path = f'{save_root}/{epoch}_logits_topk.json'
    if getattr(args, 'logits_out', None):
        save_path = args.logits_out
        json_path = os.path.splitext(save_path)[0] + '_topk.json'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        print(f'[logits] custom output -> {save_path}')

    name2logits = {}
    readable = {}

    def _save():
        np.savez_compressed(save_path,
                            name2logits=name2logits,
                            candidate_labels=np.array(labels, dtype=object),
                            neutral=neutral,
                            score_mode='logits',
                            tau=args.score_tau)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(readable, f, ensure_ascii=False, indent=1)

    for idx, name in enumerate(test_names):
        subtitle = name2subtitle.get(name, '')
        print(f'\n-- [LOGITS {idx+1}/{len(test_names)}] {name}')
        try:
            img_list, status = encode_sample(chat, dataset_cls, name)
            if img_list is None:
                print(f'  [SKIP] {status}')
                name2logits[name] = {'status': status}
                continue
            res = run_aur_logits(chat, img_list, dataset_cls, subtitle,
                                 labels, label2ids, neutral, tau=args.score_tau,
                                 caption=name2caption.get(name, ''))
            name2logits[name] = res
            readable[name] = {m: d.get('topk', []) for m, d in res.items() if isinstance(d, dict)}
            for m in AUR_LOGITS_MODES:
                top = res.get(m, {}).get('topk', [])
                shown = ', '.join(f'{l}:{p:.3f}' for l, p in top[:5])
                print(f'  [{m:5s}] {shown}')
        except Exception as e:
            print(f'  [ERROR] {name}: {e}')
            name2logits[name] = {'status': f'error: {e}'}
        if (idx + 1) % 200 == 0:               # 增量 checkpoint，防长跑中断丢结果
            _save()
            print(f'[logits] checkpoint saved at {idx+1}/{len(test_names)}')

    print(f'\n[logits] saving to {save_path} ...')
    _save()
    print(f'[logits] human-readable top-k saved to {json_path}')
    print('[logits] All done.')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EmoSync Inference (AUR+LSC+EoP)')
    parser.add_argument('--cfg-path',              default='xxx')
    parser.add_argument('--options', nargs='+')
    parser.add_argument('--dataset',               default='MER2026OV')
    parser.add_argument('--zeroshot',              action='store_true', default=False)
    parser.add_argument('--outside_user_message',  default=None)
    parser.add_argument('--outside_face_or_frame', default=None)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='只处理前 N 个样本，用于冒烟测试')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='从第几个样本开始(配合 max_samples 切片，用于取校准集 human[1200:])')
    parser.add_argument('--train_debug', action='store_true', default=False,
                        help='用 track2_train_human.csv 代替 candidate CSV（数据还在解压时用）')
    parser.add_argument('--score-mode', dest='score_mode', choices=['selfreport', 'logits'],
                        default='selfreport',
                        help='selfreport=现状自由生成(AUR+LSC+EoP); '
                             'logits=方案2闭集logits支持度打分(只跑audio/face/text)')
    parser.add_argument('--score-tau', dest='score_tau', type=float, default=1.0,
                        help='方案2候选 softmax 温度(默认1.0)')
    parser.add_argument('--logits_csv', default=None,
                        help='v3: logits 打分改读此 csv 的 name 列(默认 track2_test.csv); 配 --start_idx/--max_samples 可扩到 human[1200:1532]')
    parser.add_argument('--caption_csv',
                        default='/opt/data/wlcc/mer2026-data/track2_train_human_caption-AffectGPT.csv',
                        help='v3.1: 逐样本场景描述 caption 注入 logits 提示词(读该 csv 的 name,caption 列)')
    parser.add_argument('--logits_out', default=None,
                        help='v3.1: logits NPZ output path (default <ckpt>_logits.npz)')
    args = parser.parse_args()

    cfg           = Config(args)
    model_cfg     = cfg.model_cfg
    datasets_cfg  = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device        = 'cuda:{}'.format(inference_cfg.gpu)

    # ── Step1: 确定 checkpoint 路径 ──────────────────────────────────────
    print('======== Step1: ckpt resolution ========')
    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name   = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
    else:
        cfg_name        = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name+'*'))
        ckpt3_root      = search_for_ckpt_root(root_candidates)
    print(f'ckpt3_root: {ckpt3_root}')

    whole_ckpt3s  = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print(f'face_or_frame: {face_or_frame}')

    for ii, ckpt_3 in enumerate(whole_ckpt3s):

        # ── Step2: 加载模型 ───────────────────────────────────────────────
        print(f'======== Step2: load model ({os.path.basename(ckpt_3)}) ========')
        model_cfg.ckpt_3 = ckpt_3
        if ii == 0:
            model_cls = registry.get_model_class(model_cfg.arch)
            model     = model_cls.from_config(model_cfg)
        else:
            ckpt = torch.load(model_cfg.ckpt_3, map_location='cpu', weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval()
        chat  = Chat(model, model_cfg, device=device)

        # ── Step3: 数据集 & 样本列表 ──────────────────────────────────────
        print('======== Step3: dataset setup ========')
        dataset_cls = MER2026OV_Dataset()
        dataset_cls.needed_data   = dataset_cls.get_needed_data(face_or_frame)
        dataset_cls.vis_processor = BaseProcessor()
        dataset_cls.img_processor = BaseProcessor()
        vis_cfg = inference_cfg.get('vis_processor')
        img_cfg = inference_cfg.get('img_processor')
        if vis_cfg:
            dataset_cls.vis_processor = registry.get_processor_class(
                vis_cfg.train.name).from_config(vis_cfg.train)
        if img_cfg:
            dataset_cls.img_processor = registry.get_processor_class(
                img_cfg.train.name).from_config(img_cfg.train)
        dataset_cls.n_frms    = model_cfg.vis_processor.train.n_frms
        name2subtitle         = dataset_cls.name2subtitle

        if args.train_debug:
            label_csv  = os.path.join(config.DATA_DIR['MER2026'], 'track2_train_human.csv')
            test_names = func_read_key_from_csv(label_csv, 'name')
            print(f'[train_debug] {len(test_names)} samples from track2_train_human.csv')
        else:
            test_names = dataset_cls.read_test_names()

        if args.start_idx:
            test_names = test_names[args.start_idx:]
            print(f'[start_idx] skip first {args.start_idx}, {len(test_names)} samples remain')
        if args.max_samples:
            test_names = test_names[:args.max_samples]
            print(f'[max_samples] limited to {len(test_names)} samples')

        user_message = (args.outside_user_message if args.outside_user_message
                        else dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True))

        # ===== 方案2: logits 闭集支持度打分（只跑 audio/face/text，跳过 caption/LSC/EoP）=====
        if args.score_mode == 'logits':
            run_logits_scoring(chat, dataset_cls, name2subtitle, args, ckpt3_root, cfg)
            continue

        # ── 输出路径 ──────────────────────────────────────────────────────
        save_root = os.path.join('output/results-emosync', os.path.basename(ckpt3_root))
        os.makedirs(save_root, exist_ok=True)
        epoch     = os.path.basename(cfg.model_cfg.ckpt_3)[:-4]
        save_path = f'{save_root}/{epoch}.npz'
        if os.path.exists(save_path):
            print(f'[SKIP] {save_path} already exists')
            continue

        # ── Step4: EmoSync 推理主循环（两遍：先零样本CoT挑范例，再少样本CoT+EoP）──
        print('======== Step4: EmoSync inference (AUR + LSC + EoP) ========')
        name2reason  = {}   # EoP 最终结果 (供 evaluation pipeline 直接使用)
        name2aur     = {}   # 4路 AUR 原始结果
        name2lsc     = {}   # LSC 结果 (cot1/cot2/fewshot)
        name2caption = {}   # AffectGPT 为每个样本生成的视频描述(caption)

        # ===== Pass 1: AUR + LSC 零样本 CoT（两个 prompt）=====
        print('---- Pass 1: AUR + LSC zero-shot CoT (two prompts) ----')
        pass1 = {}
        for idx, name in enumerate(test_names):
            subtitle = name2subtitle.get(name, '')
            print(f'\n── [P1 {idx+1}/{len(test_names)}] {name}')
            img_list, status = encode_sample(chat, dataset_cls, name)
            if img_list is None:
                print(f'  [SKIP] {status}')
                continue
            # Module 0: caption — 让 AffectGPT 先生成一段客观视频描述(论文 <Caption> 字段)
            caption = run_caption(chat, img_list, dataset_cls, subtitle)
            # Module 1: AUR — 4路并行单模态推理(caption 只进 multi 路)
            aur = run_aur(chat, img_list, dataset_cls, subtitle, user_message, caption=caption)
            # Module 2(a): LSC 零样本 CoT — 论文的两个 prompt(含 caption)
            cot1, cot2 = run_lsc(chat, img_list, dataset_cls, subtitle, caption=caption)
            pass1[name] = {'aur': aur, 'cot1': cot1, 'cot2': cot2,
                           'subtitle': subtitle, 'caption': caption}
            name2aur[name] = aur
            name2lsc[name] = {'cot1': cot1, 'cot2': cot2}
            name2caption[name] = caption
            print(f'  [Caption]   {str(caption)[:80]}')
            print(f'  [AUR-multi] {str(aur.get("multi",""))[:80]}')
            print(f'  [LSC-CoT1]  {str(cot1)[:80]}')
            print(f'  [LSC-CoT2]  {str(cot2)[:80]}')

        # ===== 自洽性挑选少样本范例 =====
        exemplars = select_lsc_exemplars(pass1, k=2)
        print(f'\n---- selected {len(exemplars)} LSC exemplars (self-consistent) ----')
        for i, ex in enumerate(exemplars, 1):
            print(f'  exemplar{i} emotions: {ex["emotions"]}')
        fewshot_message = build_lsc_fewshot_message(exemplars) if exemplars else None

        # ===== Pass 2: LSC 少样本 CoT + EoP =====
        print('\n---- Pass 2: LSC few-shot CoT + EoP ----')
        for idx, name in enumerate(test_names):
            if name not in pass1:
                continue
            subtitle = pass1[name]['subtitle']
            caption  = pass1[name].get('caption', '')   # 复用 Pass1 生成的 caption，不重算
            print(f'\n── [P2 {idx+1}/{len(test_names)}] {name}')
            img_list, status = encode_sample(chat, dataset_cls, name)
            if img_list is None:
                print(f'  [SKIP] {status}')
                continue
            # Module 2(b): LSC 少样本 CoT — 带自洽范例(含 caption)
            if fewshot_message is not None:
                lsc_final = run_lsc_fewshot(chat, img_list, dataset_cls, subtitle, fewshot_message, caption=caption)
            else:
                lsc_final = pass1[name]['cot1']   # 没挑到范例就退回零样本结果
            name2lsc[name]['fewshot'] = lsc_final
            print(f'  [LSC-fewshot] {str(lsc_final)[:90]}')
            # Module 3: EoP — 集成（EoP 下一步重写；这里先用 few-shot LSC 作纠错结果）
            final = run_eop(chat, pass1[name]['aur'], lsc_final, subtitle)
            name2reason[name] = final
            print(f'  [EoP-final]   {str(final)[:90]}')

        print(f'\nSaving to {save_path} ...')
        np.savez_compressed(save_path,
                            name2reason=name2reason,
                            name2aur=name2aur,
                            name2lsc=name2lsc,
                            name2caption=name2caption)
        print('All done.')
