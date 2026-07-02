# CLAUDE.md — reliability（可靠性加权融合 / 分层校准门控 LCG）

mer2026-v3 新增的**纯 numpy 离线后处理包**。吃阶段一已落盘的逐模态 `*_logits.npz`（audio/face/text 三模态 support 表），用一层**学习门控**把 8 维 0 成本信号校准成每模态可靠性权重 `w_m`，做**加权线性池融合**后读 TopK 出最终开放词情绪。**推理链绝不 import torch**；torch 只在标定/训练侧出现。先读根 `CLAUDE.md` 的「评估指标」（EW-F1、emotion wheel 归一）一节，再回本文件。

## 数据流（一句话）

`*_logits.npz`（逐模态 `support`/`raw` 分布）→ 逐样本逐模态 8 维特征 φ → 门控 `reliability_weights` 出 `w_m`（缺失模态硬门控不入、和归一）→ `fuse` 加权线性池 → `read_topk(K)` 读出 → `wheel_metric_calculation` 算 EW-F1。软标签 `y_{n,m}` = 该模态单样本 wheel-F1（listwise 逼近目标）。

## 文件清单

- `__init__.py` — 包常量。`MODS=['audio','face','text']`、`MI={'audio':0,'face':1,'text':2}`（模态→行号）。`__version__='3.0.0'`。
- `io.py` — 落盘 I/O 与划分。
  - `load_logits(npz_path=None)` → `(name2logits, labels, neutral_vec, tau, vocab_hash, npz_path)`；npz 里 `neutral` 是 0 维 object 数组包 `{label:score}` dict，`vocab_hash=md5('|'.join(labels))[:12]`；缺省取 `output/results-emosync/*/*_logits.npz` 最新一个。
  - `load_gt(csv_path=None)` → `(name2gt:{name:'a, b'}, name2K:{name:K})`；读 `openset` 列，`K=max(1,len(gl))`；默认 `HUMAN_GT_CSV`。
  - `covered(name2gt, name2logits)` → 既有 GT 又有非 skip logits（`'status' not in v`）的样本，CV/B0 用。
  - `split_train_eval(name2gt, name2logits, test_csv=None)` → 协议 A holdout：`train = (human GT − test) ∩ have`、`eval = test ∩ have ∩ GT`。**因 test⊆human 故零泄漏**。
  - `save_gate(path, gp, pi, tau, vocab_hash, calib_meta)` / `load_gate(path, tau=None, vocab_hash=None)` — gate_params.json 存/取；load 时若传 tau+hash 会**断言配置未漂移**。`load_gate` 已支持 `'mlp'` 分支（见下）。
  - 常量：`HUMAN_GT_CSV = config.PATH_TO_LABEL['Human']`（默认 `/opt/data/wlcc/mer2026-data/track2_train_human.csv`）、`TEST_CSV = <DATA_DIR/MER2026>/track2_test.csv`。
- `features.py` — 逐样本逐模态 8 维特征 φ（8 维语义与 `fuse_logits_cv.py::feats` 对齐；注意后者签名为 `feats(sup,raw,cons)`、neutral/p0 走模块全局，FEAT_NAMES 用中文名，属等价实现而非逐字一致）。`FEAT_NAMES=['-Hn','top1','margin','gini','mass','shiftmax','klprior','agree']`。
  - `svec(res,m,labels)` → 该模态全候选 support 向量；缺 `support` 或全 0 → `None`（触发硬门控）。
  - `rvec(res,m,labels,neutral_vec)` → raw 向量，缺项以 neutral 兜底。
  - `neutral_p0(neutral_vec,tau)` → 中性 PMI 先验 `p0=softmax(neutral/tau)`。
  - `consensus_vecs(res,labels)` → `(sv:{m:vec|None}, av:可用模态列表, cons:{m:其它模态 support 均值|None})`。
  - `feats(sup,raw,neutral_vec,p0,cons)` → 8 维：负归一熵、top1、margin、gini、mass(=sup 和)、shiftmax(=max(raw−neutral))、klprior(=KL(sup‖p0))、agree(=sup 与 cons 余弦)。
- `calibrate.py` — 建数据集 + 先验 + z-score + **门控拟合（唯一 import torch 的推理外文件）**。
  - `build_arrays(names, name2logits, labels, neutral_vec, p0, name2gt, name2K, bank)` → `X[N,3,8], Mask[N,3], Yt[N,3](软标签), SUP{name:{m:vec}}, keep:list, ni:{name:idx}`；`Yt[i,m]=samp_f1(gt, top-K(sup_m))`。
  - `empirical_prior(X,Mask,Yt,idx)` → `π_m` = 各模态标定集平均单模态 EW-F1 归一。
  - `zfit(X,Mask,idx)` → `(mu,sd)`（只用可用模态行）。
  - `fit_gate(X,Mask,Yt,idx,mu,sd,T=1.0,Ty=0.1,lam=1e-3,steps=500,lr=0.05,seed=0)` → **闭式线性头** `s_m=w·z(φ)+b_m`，listwise 让 `softmax_m(s/T)` 逼近 `softmax(y/Ty)`，Adam 500 步；→ `gp={'w'[8],'b'[3],'mu','sd','T'}`。
- `weights.py` — `reliability_weights(feat_row[3,8], mask_row[3], mode, gp=None, pi=None, T_tier0=0.25)` → `{modality:weight}`（缺失模态不入、和为 1）。见下「门控模式」。
- `fuse.py` — `fuse(sup_by_m, w, labels)` → `(融合分布 vec 归一, U)`；`U` = 加权“不确定质量”仅记录不参与读出。`read_topk(fused_vec, labels, K)` → 前 K 个标签。
- `metrics.py` — 评测双轨。
  - `build_wheel_bank(labels, name2gt=None)` — 预映射每词到 5 wheel level1 桶（复用 `config.OUTSIDE_WHEEL_MAPPING`）。
  - `samp_f1(gt_words, pred_words, bank)` — **快速单样本 wheel-F1（5 轮平均）**，作软标签 `y_{n,m}`。
  - `eval_fused(name2pred, name2gt)` → `(n, f%, p%, r%)`，**直接调项目 `my_affectgpt.evaluation.wheel.wheel_metric_calculation`**（与 `run_evaluation_emosync` 同一函数，返回百分制）。
- `run.py` — 编排。被 `calibrate_gate.py`(emit=False) 与 `fuse_logits.py`(emit=True) 共用。
  - `MODES=['text','equal','prior','tier0','lcg']`（消融表固定这 5 个；落盘预测固定取 `lcg`）。
  - `run(args)` → 按 `args.split` 分流 `run_cv5`(协议 B0，5 折 OOF) / `run_holdout`(协议 A，留出)。
  - `build_args(**kw)`、`predict_all(...)`、`ablation(...)`、`emit(...)`；`DEF_OUTDIR`/`DEF_BASE` 为缺省输出目录与 ckpt 前缀。

## 门控模式（`weights.reliability_weights` 的 `mode`）

| mode | 公式 | 需要 |
|---|---|---|
| `text` | text 存则独占 1.0，否则等权 | — |
| `equal` | 均分 `1/|av|` | — |
| `prior` | `π_m` 归一 | `pi` |
| `tier0` | `π_m·exp(-Hn/T_tier0)` 归一（闭式，`-Hn=feat[...,0]`） | `pi` |
| `lcg` | z-score 后 `s_m=w·z+b_m`，`softmax_m(s/T)`（学习线性门控） | `gp`（`w,b,mu,sd,T`） |
| `mlp` | `s_m=MLP(z(φ_m))`（共享两层 MLP，relu/gelu），`softmax_m(s/T)` | `gp['mlp']`（`W,B,mu,sd,act`）+ `gp['T']` |

`mlp` 是**当前工作树未提交改动**（`git diff -- reliability/weights.py reliability/io.py`，分支 `mer2026-v3`）：`weights.py` 加了纯 numpy 前向 `_fwd`（`W[li]@x+B[li]`，末层前 relu/gelu，gelu 用 tanh 近似常数 0.7978845608028654），`io.py::load_gate` 加了 `if 'mlp' in g` 分支还原 `{W,B,mu,sd,act}`（断言 `act∈{relu,gelu}`）。`.bak_mlp` 是改前备份，勿当活文件。

## 契约

- **张量形状**：`X[N,3,8]`（模态维顺序按 `MI`：audio,face,text）、`Mask[N,3]∈{0,1}`（模态是否可用=硬门控）、`Yt[N,3]` 软标签∈[0,1]（逐模态单样本 wheel-F1）。
- **support 向量**：全 `labels`(≈253 候选)长度、非负；缺 support 或全 0 → 该模态硬门控出局。
- **gate_params.json 字段**（`save_gate`）：`w[8]`、`b{audio,face,text}`、`mu[8]`、`sd[8]`、`T`、`pi{m}`、`tau`、`vocab_hash`、`feat_dim`、`calib{gt_csv,split,n_train,n_eval}`。mlp 变体额外含 `gate_type:'mlp'`、`mlp{W:[W1,W2],B:[b1,b2],mu,sd,act}`。**load 时 tau/vocab_hash 不匹配即报「配置漂移，请重新标定」**。
- **零泄漏三集**：`test ⊆ human`；`train=human GT−test`、`eval=test`（协议 A）；mlp 训练侧另分 `holdout=(human−test)∩have`、`train=(all_gt−test)∩have−holdout` 且断言 `train∩test=∅`、`train∩holdout=∅`。

## 入口脚本（在项目根，驱动本包）

> 三个入口都 `sys.path.insert(0,'/opt/data/wlcc/MER2026_Track2')` 且 `os.chdir(...)` 到 `/opt/data/wlcc/MER2026_Track2`（**指向本 worktree 的 symlink/别名，并非另一份独立 checkout**）；用 affectgpt conda python，融合链 `CUDA_VISIBLE_DEVICES=""`。

- `calibrate_gate.py`（入口①，仅标定不落预测）：
  `python calibrate_gate.py --split {cv5|holdout} [--npz X] [--gt human.csv] [--out gate.json] [--T --Ty --lam]`
- `fuse_logits.py`（入口②，标定+融合+消融+落盘）：
  `python fuse_logits.py --split {cv5|holdout} --ablation [--npz X] [--gt human.csv] [--gate g.json] [--outdir D]`；消融同表打印 text/equal/prior/tier0/lcg，落盘预测固定 `lcg`。
- `apply_gate_candidate.py`（应用到无标签候选集，不需 GT）：
  `python apply_gate_candidate.py --npz <candidate_logits.npz> --gate <gate.json> --out answer.csv [--K 5] [--mode {lcg|tier0|prior|equal|text|mlp}]`；缺模态/无权重 → 兜底 `neutral`。
- `train_mlp_gate.py`（离线训练 mlp 门控，独占 torch）：吃 human(1532)+MCP 合并 npz，多 seed 早停选 holdout EW-F1 最优，存 `output/_cal/gate_params_mlp.json`。MCP 样本源权重 `--w-mcp 0.25`；噪声过滤**仅对 MCP 源样本**、当其所有可用模态 F1 全 0 时整样本丢弃（human 源样本一律保留）。
- `merge_logits.py`：合并两份 npz（如 1200+332→1532），**校验 `candidate_labels` 一致**（`neutral`/`tau` 直接沿用 base、不校验），重叠以 add 覆盖 base。
- `run_reliability.sh [cv5|holdout] [npz]`：一键跑 `fuse_logits.py --ablation`。

## 怎么改 / 约定

- **纯 numpy 后处理**：`weights/io/fuse/features/apply_gate_candidate` 推理链**绝不 import torch**；训练/标定（`calibrate.fit_gate`、`train_mlp_gate.py`）才用 torch，产物落成纯 numpy 可读 JSON。加新门控模式：在 `reliability_weights` 加分支 + 让 `load_gate` 能还原其参数 + 更新本表。
- **路径走 config**：GT/npz/wheel 映射统一经 `config.PATH_TO_LABEL` / `config.DATA_DIR` / `config.OUTSIDE_WHEEL_MAPPING`，勿硬编码绝对路径。
- **零泄漏不可破**：任何新划分都要保证 `train∩test=∅`；改 split 逻辑后重跑断言。
- **特征一致性**：`features.feats` 的 8 维需与 `fuse_logits_cv.py::feats` 语义对齐（两者签名不同、属等价实现）；改 8 维会使旧 `gate_params.json` 的 `mu/sd/w` 失配（`vocab_hash` 不变但语义漂移），改完须重标定。
- **勿改 vendored / 编译产物**：wheel 映射走 `emotion_wheel/`（见其 CLAUDE.md），本包只消费不改。`.bak_*` 是备份文件，别当活代码编辑或删。

## v3.2 追加：caption 增强门控 `mlpcap`（分支 mer2026-v3.2）

在逐模态 8 维分布特征之外，为打分器再拼接每样本**情境描述 caption 的语义向量**（本地 CLIP 文本塔 `clip-vit-large-patch14` 编码，768 维；长 caption 用 75-token 滑窗均值池化 + L2 归一覆盖全文）。

- 新增 `caption.py`：`CaptionEncoder`（滑窗均值池化编码）、`load_captions`、`get_cap_features`（按 name 缓存到 `output/_cal/capfeat_clipL14_*.npz`，命中则不加载 CLIP）。torch/transformers 仅编码时惰性 import；缓存命中后推理链仍纯 numpy。
- 门控模式表新增：

  | mode | 公式 | 需要 |
  |---|---|---|
  | `mlpcap` | 每模态 `s_m=MLP(concat(z(φ_m)[8], z(cap)[768]))`，`softmax_m(s/T)`；**MLP 隐层非线性使同一 caption 对三模态产生不同调制**（纯线性头会因 softmax 跨模态平移不变而抵消 caption） | `gp['mlp']{W,B,mu,sd,act}` + `gp['cap']{mu,sd,dim,method,model}` + `gp['T']` + 传入 `cap_row` |

- `reliability_weights(..., cap_row=None)` 加 `mlpcap` 分支（纯 numpy 前向，φ 在前 cap 在后，与训练逐位对齐）；`io.load_gate` 加 `gate_type=='mlpcap'` 还原。
- 入口脚本（项目根）：`train_mlp_gate_cap.py`（全量 1532 human 训练，多 seed 按训练集 EW-F1 选优 → `output/_cal/gate_params_mlp_cap_full.json`，末尾 numpy/torch 前向一致性抽查）、`eval_mlp_gate_cap.py`（在 GT csv 上评 text/equal/prior/tier0/**mlpcap** 对照 EW-F1，写 answer csv）。
- gate json 新增字段：`gate_type:'mlpcap'`、`mlp{W:[W1,W2],B:[b1,b2],mu[8],sd[8],act}`、`cap{mu[768],sd[768],dim,method:'clip-vitL14-winmean',model}`。
- ⚠️ 本次评测 `_ovmerdplus_ovlabel.csv`(532) ⊆ 训练集 human(1532)，为**含重叠的全量训练**（用户指定），mlpcap 在 532 上的 EW-F1 偏乐观、非无泄漏泛化分。
