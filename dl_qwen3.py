import os, sys, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
from huggingface_hub import HfApi, hf_hub_download

REPO = "Qwen/Qwen3-8B"
DEST = "/opt/data/wlcc/MER2026_Track2/my_affectgpt/models/Qwen3-8B"
api = HfApi()

files = None
for a in range(1, 16):
    try:
        info = api.model_info(REPO, files_metadata=True)
        files = [(s.rfilename, s.size or 0) for s in info.siblings]
        print(f"[list] {len(files)} files, total {sum(sz for _,sz in files)/1e9:.2f}GB", flush=True)
        break
    except Exception as ex:
        print(f"[list {a} failed] {type(ex).__name__}: {str(ex)[:120]}", flush=True)
        time.sleep(8)
if files is None:
    print("LIST_FAILED", flush=True); sys.exit(1)

def lsize(fn):
    p = os.path.join(DEST, fn)
    return os.path.getsize(p) if os.path.exists(p) else -1
def done(fn, sz):
    return (sz > 0 and lsize(fn) == sz) or (sz == 0 and lsize(fn) >= 0)

for rnd in range(1, 81):
    missing = [(fn, sz) for fn, sz in files if not done(fn, sz)]
    if not missing:
        print("ALL_DONE", flush=True); break
    done_gb = sum(sz for fn, sz in files if done(fn, sz))/1e9
    print(f"=== round {rnd} {time.strftime('%H:%M:%S')}: 剩 {len(missing)} 个文件, 已完成 {done_gb:.2f}GB ===", flush=True)
    for fn, sz in missing:
        try:
            hf_hub_download(repo_id=REPO, filename=fn, local_dir=DEST)
            got = lsize(fn)
            tag = "OK" if (sz == 0 or got == sz) else f"SIZE_MISMATCH {got}/{sz}"
            print(f"  [{tag}] {fn}", flush=True)
        except Exception as ex:
            print(f"  [retry] {fn}: {type(ex).__name__} {str(ex)[:100]}", flush=True)
    time.sleep(5)

bad = [(fn, sz, lsize(fn)) for fn, sz in files if sz > 0 and lsize(fn) != sz]
print("FINAL:", "SUCCESS" if not bad else f"INCOMPLETE({len(bad)})", flush=True)
for fn, sz, got in bad[:10]:
    print(f"  partial: {fn} {got}/{sz}", flush=True)
