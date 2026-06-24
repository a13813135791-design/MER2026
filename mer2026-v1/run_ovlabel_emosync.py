#!/opt/data/wlcc/Software/Anaconda3/envs/affectgpt/bin/python
import sys, os
sys.path.insert(0, '/opt/data/wlcc/MER2026_Track2')
os.chdir('/opt/data/wlcc/MER2026_Track2')

from ovlabel_extraction import func_read_batch_calling_model, extract_openset_batchcalling
import glob

print("[1/3] 加载 Qwen25 模型 ...")
llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname="Qwen25")
print("[2/3] 模型加载完毕，扫描 results-emosync/ ...")

npz_files = glob.glob("output/results-emosync/*/*.npz")
todo = [p for p in npz_files if not p.endswith("-openset.npz")]
print(f"  找到 {len(todo)} 个待处理 npz 文件")

for result_path in todo:
    openset_npz = result_path[:-4] + "-openset.npz"
    if os.path.exists(openset_npz):
        print(f"  SKIP (已存在): {openset_npz}")
        continue
    print(f"  处理: {result_path}")
    extract_openset_batchcalling(
        reason_npz=result_path,
        store_npz=openset_npz,
        llm=llm,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
    )
    print(f"  -> 写出: {openset_npz}")

print("[3/3] 全部完成！")
