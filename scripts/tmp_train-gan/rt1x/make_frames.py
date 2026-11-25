"""Parity inputs for rt1x_ref.py: 15 consecutive front frames of one RoboMME episode (first execution steps), resized
256 -> 300 px, f32 in [0, 1], plus the cached USE embedding of the episode instruction.
Usage: python make_frames.py <task h5> <use cache npz> <out frames.npz>"""
from io import BytesIO
import sys

import h5py
import numpy as np
from PIL import Image

h5, cache, out = sys.argv[1:]
with h5py.File(h5) as f:
    g = f["episode_0"]
    t = int(np.flatnonzero(~g["video"][:])[0])
    image = np.stack([np.asarray(Image.open(BytesIO(bytes(j))).convert("RGB").resize((300, 300), Image.BILINEAR))
                      for j in g["front"][t:t + 15]]).astype(np.float32)[None] / 255
    lang = str(g.attrs["lang"])
c = np.load(cache)
emb = c["emb"][list(c["strings"]).index(lang)]
np.savez(out, image=image, lang=np.repeat(emb[None, None], 15, 1))
print(lang, image.shape)
