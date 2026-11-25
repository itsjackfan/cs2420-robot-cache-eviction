"""Cache Universal Sentence Encoder Large v5 embeddings (RT-1-X language input) for every RoboMME instruction.
Strings = all variants of RoboMME's goal templates over generous parameter grids, plus every `lang` attr in the
converted H5s. Templates are run from robomme's own task_goal.py with fake env objects.
Usage: uv run --no-project --with tensorflow-cpu --with tensorflow-hub --with h5py python use_cache.py \
         <robomme task_goal.py> <converted h5 dir> <out npz>"""
from itertools import permutations, product
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util
import sys

import h5py
import numpy as np

COLORS = ["red", "blue", "green", "purple", "gray"]
N = range(1, 21)


def fake(pick=1, **attrs):
    return NS(env=NS(unwrapped=NS(configs={"d": {"pick": pick}}, **attrs)), difficulty="d", pick_times=pick, **attrs)


def strings(goal):
    g = lambda env, **kw: goal(fake(**kw), env)
    out = []
    for r, b, gr in product(range(11), repeat=3):
        out += g("BinFill", red_cubes_target_number=r, blue_cubes_target_number=b, green_cubes_target_number=gr)
    for c, n in product(COLORS, N):
        out += g("PickXtimes", num_repeats=n, target_color_name=c) + g("SwingXtimes", num_repeats=n, target_color_name=c)
        out += g("VideoPlaceOrder", target_color_name=c, which_in_subset=n)
    for cs, p in product(permutations(COLORS, 2), (1, 2)):
        for env in ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"):
            out += g(env, pick=p, color_names=list(cs))
    for c, when in product(COLORS, ("before", "after")):
        out += g("VideoPlaceButton", target_color_name=c, target_target_language=when)
    for n in N:
        out += g("VideoRepick", num_repeats=n) + g("StopCube", stop_time=n)
    for env in ("PickHighlight", "InsertPeg", "MoveCube", "PatternLock", "RouteStick"):
        out += g(env)
    return out


def main(task_goal_py, h5_dir, out):
    spec = importlib.util.spec_from_file_location("task_goal", task_goal_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lang = set()
    for p in sorted(Path(h5_dir).glob("*.h5")):
        with h5py.File(p) as f:
            lang |= {str(f[k].attrs["lang"]) for k in f}
    s = sorted(set(strings(mod.get_language_goal)) | lang)
    print(f"{len(s)} strings ({len(lang)} from H5, {len(lang - set(strings(mod.get_language_goal)))} not in templates)")
    import tensorflow_hub as hub
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    emb = np.concatenate([use(s[i:i + 256]).numpy() for i in range(0, len(s), 256)])
    np.savez(out, strings=np.array(s), emb=emb.astype(np.float32))
    print(out, emb.shape)


if __name__ == "__main__":
    main(*sys.argv[1:])
