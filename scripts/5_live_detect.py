"""
Step 5 - the live monitor.

Replays a trace one frame at a time, exactly as a real gateway would see it,
keeps a rolling 250 ms buffer, and re-scores every 125 ms. This is the part
that shows the detector works *online*, not just on a saved matrix.

    python scripts/5_live_detect.py                 # replay the test trace
    python scripts/5_live_detect.py --speed 0       # as fast as possible
    python scripts/5_live_detect.py --trace data/train.csv
"""

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import joblib

from canguard import config as C
from canguard import dataset, features, models

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW = "\033[31m", "\033[32m", "\033[33m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(DATA / "test.csv"))
    ap.add_argument("--speed", type=float, default=8.0,
                    help="replay speed multiplier; 0 = no delay")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after N seconds of trace time (0 = all)")
    ap.add_argument("--confirm", type=int, default=2,
                    help="raise the alarm only after N of the last 3 windows "
                         "are flagged (a real IDS debounces; 1 disables)")
    a = ap.parse_args()

    bundle = joblib.load(MODELS / "scaler.joblib")
    meta = json.load(open(MODELS / "meta.json"))
    rf = joblib.load(MODELS / "random_forest.joblib")

    from tensorflow import keras
    ae = keras.models.load_model(MODELS / "autoencoder.keras")
    clf = keras.models.load_model(MODELS / "mlp_multiclass.keras")

    frames = dataset.load_trace(a.trace)
    if a.seconds:
        frames = [f for f in frames if f[0] <= frames[0][0] + a.seconds]

    w, stride = C.WINDOW_MS / 1000.0, C.STRIDE_MS / 1000.0
    print(f"{BOLD}CAN-Guard live monitor{RESET}  -  {len(frames):,} frames "
          f"from {pathlib.Path(a.trace).name}")
    print(f"{DIM}window {C.WINDOW_MS:.0f} ms, re-scored every {C.STRIDE_MS:.0f} ms. "
          f"Ctrl-C to stop.{RESET}\n")

    buf, recent = [], []
    next_score = frames[0][0] + w
    alarms = tp = fp = fn = checks = 0
    t_wall = time.time()

    for f in frames:
        buf.append(f)
        if f[0] < next_score:
            continue

        if a.speed > 0:
            target = t_wall + (f[0] - frames[0][0]) / a.speed
            if target > time.time():
                time.sleep(target - time.time())

        buf = [b for b in buf if b[0] >= f[0] - w]
        next_score += stride
        if len(buf) < 8:
            continue

        X, y0, _ = features.build_one(buf)
        if X is None:
            continue
        Xs = models.scale(bundle["scaler"], X)
        checks += 1

        p_rf = float(rf.predict_proba(Xs)[0, 1])
        err = float(models.recon_error(ae, Xs)[0])
        novel = err > meta["ae_threshold"]
        kind = C.ATTACK_TYPES[int(clf.predict(Xs, verbose=0)[0].argmax())]
        flagged = p_rf > 0.5 or novel
        recent = (recent + [flagged])[-3:]
        # debounce: a single odd window is not an attack, a run of them is
        alert = sum(recent) >= a.confirm
        truth = bool(y0 == 1)

        tp += alert and truth
        fp += alert and not truth
        fn += (not alert) and truth
        if alert:
            alarms += 1
            tag = f"{RED}{BOLD}ALERT{RESET}"
            why = []
            if p_rf > 0.5:
                why.append(f"classifier {p_rf:.2f} -> {kind}")
            if novel:
                why.append(f"anomaly {err:.1f} (> {meta['ae_threshold']:.1f})")
            if not why:
                why.append("confirming the previous window")
            mark = f"{GREEN}ok{RESET}" if truth else f"{YELLOW}false alarm{RESET}"
            print(f"  t={f[0]:7.2f}s  {tag}  {'; '.join(why):<52} [{mark}]")
        elif truth:
            print(f"  t={f[0]:7.2f}s  {YELLOW}missed{RESET} "
                  f"{DIM}(classifier {p_rf:.2f}, anomaly {err:.1f}){RESET}")

    print(f"\n{BOLD}{checks:,} windows scored{RESET}   "
          f"alarms {alarms:,}   correct {tp:,}   false {fp:,}   missed {fn:,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
