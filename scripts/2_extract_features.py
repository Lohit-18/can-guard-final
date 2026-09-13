"""
Step 2 - turn raw frames into windowed feature vectors.

Each row of the output describes 250 ms of bus traffic. Rows overlap by half
a window, so an attack that lasts a second produces several rows.
"""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from canguard import config as C
from canguard import dataset, features

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def run(name):
    t0 = time.time()
    frames = dataset.load_trace(DATA / f"{name}.csv")
    X, names, y, acode, starts = features.build_windows(frames)
    np.savez_compressed(DATA / f"{name}_features.npz",
                        X=X, y=y, acode=acode, starts=starts,
                        names=np.array(names))
    print(f"  {name:9s} {X.shape[0]:>6,} windows x {X.shape[1]} features   "
          f"attack windows: {int(y.sum()):>5,} ({100*y.mean():5.2f}%)   "
          f"[{time.time()-t0:.1f}s]")
    return names


if __name__ == "__main__":
    print(f"Extracting features (window {C.WINDOW_MS:.0f} ms, "
          f"stride {C.STRIDE_MS:.0f} ms)...")
    names = None
    for n in ["baseline", "train", "test"]:
        names = run(n)
    print(f"\n{len(names)} features:")
    for i in range(0, len(names), 4):
        print("   " + "".join(f"{x:<26s}" for x in names[i:i + 4]))
