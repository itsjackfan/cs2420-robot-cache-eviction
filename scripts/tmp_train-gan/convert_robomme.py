"""Download (if missing) + extract + convert every RoboMME task: <archives>/record_dataset_<task>.h5.tar.xz ->
<out>/<task>.h5 (src.robomme.convert), extracting to node-local <tmp> and deleting the raw .h5 after each task.
Skips tasks already converted. Usage: python convert_robomme.py <archives> <out> <tmp> [workers]."""
from multiprocessing import Pool
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import hf_hub_download

from src.robomme import TASKS, convert


def one(task, archives, out, tmp):
    if (out / f"{task}.h5").exists():
        return f"{task}: exists"
    start = time.time()
    archive = hf_hub_download("Yinpei/robomme_data_h5", f"record_dataset_{task}.h5.tar.xz", repo_type="dataset",
                              local_dir=archives)
    (tmp / task).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xJf", archive, "-C", str(tmp / task)], check=True)
    raw = tmp / task / f"record_dataset_{task}.h5"
    convert(raw, out / f"{task}.h5.partial")
    (out / f"{task}.h5.partial").rename(out / f"{task}.h5")
    raw.unlink()
    return f"{task}: {(out / f'{task}.h5').stat().st_size / 1e9:.1f} GB in {(time.time() - start) / 60:.0f} min"


if __name__ == "__main__":
    archives, out, tmp = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as pool:
        for msg in pool.starmap(one, [(t, archives, out, tmp) for t in TASKS]):
            print(msg, flush=True)
