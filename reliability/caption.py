# -*- coding: utf-8 -*-
"""caption 特征提取(v3.2)：本地 CLIP 文本塔把每样本情境描述(caption)编成 768 维语义向量。
caption 多为长段落(>77 token)，用滑窗(75 token)分块编码 + 均值池化 + L2 归一覆盖全文。
纯特征提取，仅训练/测试脚本调用；核心推理链(weights/io/fuse/features)不 import 本模块。
torch/transformers 仅在真正需要编码时(CaptionEncoder.__init__)才 import，缓存命中则不触达。"""
import os
import numpy as np
import pandas as pd

DEFAULT_CLIP = 'my_affectgpt/models/clip-vit-large-patch14'
CAP_METHOD = 'clip-vitL14-winmean'   # 滑窗均值池化 + L2


def load_captions(csv_path):
    """读 name,caption → {name: caption_str}（缺/NaN → ''）。"""
    df = pd.read_csv(csv_path)
    assert 'name' in df.columns and 'caption' in df.columns, \
        '[caption] csv 需含 name,caption 列: %s' % csv_path
    out = {}
    for _, r in df.iterrows():
        c = r['caption']
        out[str(r['name'])] = '' if (c is None or (isinstance(c, float) and pd.isna(c))) else str(c)
    return out


class CaptionEncoder:
    """本地 CLIP 文本塔编码器。encode(texts:list[str]) → np.ndarray[N,dim]（L2 归一）。"""
    def __init__(self, model_path=DEFAULT_CLIP, device='cpu', max_len=77, batch=64):
        import torch
        from transformers import CLIPTokenizerFast, CLIPTextModelWithProjection
        self.torch = torch
        self.device = device
        self.max_len = max_len          # CLIP 文本上限(含 bos/eos)
        self.win = max_len - 2          # 每窗内容 token 数
        self.batch = batch
        self.tok = CLIPTokenizerFast.from_pretrained(model_path)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_path).eval().to(device)
        self.bos = self.tok.bos_token_id
        self.eos = self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos
        self.dim = int(self.model.config.projection_dim)

    def _windows(self, text):
        """一条 caption → 若干 [bos]+(<=win 内容)+[eos] 窗口 id（至少 1 个）。"""
        ids = self.tok(text or '', add_special_tokens=False)['input_ids']
        if len(ids) == 0:
            return [[self.bos, self.eos]]
        return [[self.bos] + ids[i:i + self.win] + [self.eos] for i in range(0, len(ids), self.win)]

    def _encode_windows(self, win_ids):
        """一批窗口 id → text_embeds[len,dim]（未归一）。定长 pad + attention_mask 屏蔽 pad。"""
        torch = self.torch
        L = max(len(w) for w in win_ids)
        inp = np.full((len(win_ids), L), self.pad, dtype=np.int64)
        att = np.zeros((len(win_ids), L), dtype=np.int64)
        for i, w in enumerate(win_ids):
            inp[i, :len(w)] = w
            att[i, :len(w)] = 1
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor(inp, device=self.device),
                             attention_mask=torch.tensor(att, device=self.device)).text_embeds
        return out.detach().cpu().float().numpy()

    def encode(self, texts):
        """texts:list[str] → [N,dim]。逐样本滑窗均值池化后 L2 归一。空文本得零向量。"""
        flat, owner = [], []
        for n, t in enumerate(texts):
            for w in self._windows(t):
                flat.append(w); owner.append(n)
        embs = np.zeros((len(flat), self.dim), dtype=np.float32)
        for s in range(0, len(flat), self.batch):
            embs[s:s + self.batch] = self._encode_windows(flat[s:s + self.batch])
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        cnt = np.zeros(len(texts), dtype=np.float32)
        for j, n in enumerate(owner):
            out[n] += embs[j]; cnt[n] += 1.0
        cnt[cnt == 0] = 1.0
        out /= cnt[:, None]
        for i, t in enumerate(texts):                      # 空 caption → 零向量
            if not (t and str(t).strip()):
                out[i] = 0.0
        norm = np.linalg.norm(out, axis=1, keepdims=True); norm[norm == 0] = 1.0
        return (out / norm).astype(np.float32)


def build_cap_matrix(names, name2cap, encoder):
    """按 names 顺序编码 → [len(names), dim]（缺 caption → 零向量）。"""
    texts = [name2cap.get(str(n), '') for n in names]
    return encoder.encode(texts)


def get_cap_features(names, cap_csv, cache_path=None, model_path=DEFAULT_CLIP, device='cpu'):
    """按 names 出 [N,dim] caption 特征。cache_path 命中(覆盖全部 names)则复用、不加载 CLIP；否则编码缺失项并写回缓存。"""
    name2cap = load_captions(cap_csv)
    cache = {}
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        cache = {str(k): np.asarray(v, dtype=np.float32) for k, v in d['name2vec'].item().items()}
    miss = [str(n) for n in names if str(n) not in cache]
    if miss:
        enc = CaptionEncoder(model_path=model_path, device=device)
        mat = build_cap_matrix(miss, name2cap, enc)
        for i, n in enumerate(miss):
            cache[n] = mat[i]
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            np.savez(cache_path, name2vec=np.array(cache, dtype=object))
    dim = len(next(iter(cache.values())))
    return np.stack([cache.get(str(n), np.zeros(dim, dtype=np.float32)) for n in names]).astype(np.float32)
