"""Saving and loading traces in a candump-like CSV form."""

import numpy as np
import pandas as pd


def save_trace(frames, path):
    pd.DataFrame({
        "t": np.round([f[0] for f in frames], 6),
        "can_id": [f"{f[1]:03X}" for f in frames],
        "dlc": [len(f[2]) for f in frames],
        "data": [f[2].hex().upper() for f in frames],
        "label": [f[3] for f in frames],
        "attack": [f[4] for f in frames],
    }).to_csv(path, index=False)


def load_trace(path):
    df = pd.read_csv(path, dtype={"can_id": str, "data": str})
    return [
        (t, int(cid, 16), bytes.fromhex(d), int(l), int(a))
        for t, cid, d, l, a in zip(df.t, df.can_id, df.data, df.label, df.attack)
    ]


def save_episode_log(log, path):
    pd.DataFrame(log, columns=["attack", "start_s", "end_s"]).to_csv(path, index=False)
