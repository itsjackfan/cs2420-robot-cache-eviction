"""Inspect one RoboMME record_dataset_<task>.h5: layout, episode lengths, video-demo frames, action/state ranges,
rpy histograms (wrap check), gripper values. Usage: python inspect_h5.py <file.h5> [n_episodes]."""
import sys

import h5py
import numpy as np


def steps(ep):
    return sorted((k for k in ep if k.startswith("timestep_")), key=lambda k: int(k.split("_")[1]))


def main(path, n_eps=20):
    f = h5py.File(path)
    eps = sorted(f, key=lambda k: int(k.split("_")[1]))
    print(len(eps), "episodes; first:", eps[0])
    ep = f[eps[0]]
    ep.visit(lambda name: print(" ", name, getattr(ep[name], "shape", ""), getattr(ep[name], "dtype", ""),
                                getattr(ep[name], "compression", "")) if not name.startswith("timestep_") or name.startswith("timestep_1/") else None)
    acts, states, grips, lens, videos, bounds = [], [], [], [], [], []
    for e in eps[:n_eps]:
        ts = [f[e][t] for t in steps(f[e])]
        video = np.array([bool(t["info/is_video_demo"][()]) for t in ts])
        lens.append(len(ts)), videos.append(video.sum())
        bounds.append(sum(bool(t["info/is_subgoal_boundary"][()]) for t in ts))
        acts += [t["action/eef_action"][()] for t, v in zip(ts, video) if not v]
        states += [t["obs/eef_state"][()] for t in ts]
        grips += [t["obs/gripper_state"][()] for t in ts]
    acts, states, grips = np.array(acts, np.float64), np.array(states, np.float64), np.array(grips)
    print(f"\nfirst {n_eps} eps: steps min/mean/max {min(lens)}/{np.mean(lens):.0f}/{max(lens)}; "
          f"video-demo frames {sorted(set(videos))}; subgoal boundaries {sorted(set(bounds))}")
    print("goal:", f[eps[0]]["setup/task_goal"][()])
    for name, x in [("eef_action (exec steps)", acts), ("eef_state", states), ("gripper_state", grips)]:
        print(f"\n{name} {x.shape}")
        for i in range(x.shape[1]):
            print(f"  dim {i}: min {x[:, i].min():+.3f} max {x[:, i].max():+.3f} mean {x[:, i].mean():+.3f} std {x[:, i].std():.3f}")
    print("\naction gripper values:", np.unique(acts[:, 6].round(3))[:10])
    for i, a in zip((3, 4, 5), "rpy"):
        h, edges = np.histogram(acts[:, i], bins=12, range=(-np.pi, np.pi))
        print(f"action {a} hist [-pi, pi] x12:", h.tolist())
        mu = np.angle(np.exp(1j * acts[:, i]).mean())
        print(f"   circular mean {mu:+.3f}; std after wrap(x - mu) {np.angle(np.exp(1j * (acts[:, i] - mu))).std():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
