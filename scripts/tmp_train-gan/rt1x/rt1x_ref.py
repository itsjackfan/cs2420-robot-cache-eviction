"""Run in the throwaway JAX env: official RT-1-X (OXE models/rt1.py, released config) on fixed inputs -> params + outputs.
Float32 everywhere, as the official model runs (and was trained).
usage: python rt1x_ref.py <oxe_src> <checkpoint> <out_dir> [frames.npz]
frames.npz: {image [1, 15, 300, 300, 3] f32 in [0, 1], lang [1, 15, 512]}; default: uniform noise images (out of
distribution: saturate TokenLearner's softmax, so expect float32 rounding to flip some token selections)."""
import sys

import flax.serialization
import jax
import numpy as np
from flax.traverse_util import flatten_dict

oxe_src, ckpt_path, out, *frames = sys.argv[1:]
sys.path.insert(0, f"{oxe_src}/models")
import rt1  # noqa: E402

ckpt = flax.serialization.msgpack_restore(open(ckpt_path, "rb").read())
variables = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}
model = rt1.RT1(num_image_tokens=81, num_action_tokens=11, layer_size=256, vocab_size=512, use_token_learner=True,
                world_vector_range=(-2.0, 2.0))
flat = {f"params/{k}": v for k, v in flatten_dict(ckpt["params"], sep="/").items()}
flat |= {f"batch_stats/{k}": v for k, v in flatten_dict(ckpt["batch_stats"], sep="/").items()}
np.savez(f"{out}/rt1x_params.npz", **{k: np.asarray(v) for k, v in flat.items()})

rng = np.random.default_rng(0)
O = 15
if frames:
    image, lang = (np.load(frames[0])[k] for k in ("image", "lang"))
else:
    image = rng.random((1, O, 300, 300, 3), dtype=np.float32)
    lang = np.repeat(rng.standard_normal((1, 1, 512)).astype(np.float32), O, 1)
    lang /= np.linalg.norm(lang, axis=-1, keepdims=True)  # USE embeddings are unit norm
kw = dict(train=False, rngs={"random": jax.random.PRNGKey(0)})
image_tokens = model.apply(variables, image, lang, method=rt1.RT1.tokenize_image, **kw)
logits = model.apply(variables, {"image": image, "natural_language_embedding": lang}, act=None,
                     act_tokens=np.zeros((1, O, 11)), **kw)
np.savez(f"{out}/rt1x_ref{'_frames' * bool(frames)}.npz", image=image, lang=lang, image_tokens=np.asarray(image_tokens), logits=np.asarray(logits))
print(image_tokens.dtype, image_tokens.shape, logits.dtype, logits.shape)
