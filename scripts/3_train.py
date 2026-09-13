"""
Step 3 - fit the detectors.

Training discipline:
  * the scaler and both unsupervised models see ONLY attack-free traffic;
  * the supervised models see labelled attacks from `train.csv`;
  * `test.csv` is a separate simulation run and is never touched here.
"""

import json
import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import joblib

from canguard import models

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"


def load(name):
    z = np.load(DATA / f"{name}_features.npz", allow_pickle=True)
    return z["X"], z["y"], z["acode"], list(z["names"])


def main():
    MODELS.mkdir(exist_ok=True)
    Xb, yb, _, names = load("baseline")
    Xtr, ytr, atr, _ = load("train")

    # --- normalisation, fitted on healthy traffic only -------------------
    sc = models.make_scaler(Xb)
    joblib.dump({"scaler": sc, "names": names}, MODELS / "scaler.joblib")
    Bs, Ts = models.scale(sc, Xb), models.scale(sc, Xtr)

    # Hold out the last 20% of the clean run to set the alarm threshold on
    # data the autoencoder never fitted.
    cut = int(len(Bs) * 0.8)
    Bfit, Bcal = Bs[:cut], Bs[cut:]

    # --- 1. TensorFlow autoencoder (the behavioural baseline) ------------
    import tensorflow as tf
    from tensorflow import keras

    # Make the TensorFlow runs reproducible too, not just the sklearn ones.
    keras.utils.set_random_seed(0)
    tf.config.experimental.enable_op_determinism()

    print("[1/4] TensorFlow autoencoder on attack-free traffic...")
    ae = models.build_autoencoder(Bs.shape[1])
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    t0 = time.time()
    ae.fit(Bfit, Bfit, epochs=120, batch_size=64, verbose=0,
           validation_split=0.15,
           callbacks=[keras.callbacks.EarlyStopping(patience=15,
                                                    restore_best_weights=True)])
    cal_err = models.recon_error(ae, Bcal)
    thr = float(np.percentile(cal_err, 99.0))
    ae.save(MODELS / "autoencoder.keras")
    print(f"      trained in {time.time()-t0:.1f}s   "
          f"alarm threshold (99th pct of clean error) = {thr:.4f}")

    # --- 2. Isolation Forest (classical unsupervised baseline) -----------
    print("[2/4] Scikit-learn Isolation Forest on attack-free traffic...")
    iso = models.make_isolation_forest()
    iso.fit(Bfit)
    joblib.dump(iso, MODELS / "isolation_forest.joblib")

    # --- 3. Random Forest (supervised, binary) ---------------------------
    print("[3/4] Scikit-learn Random Forest on labelled traffic...")
    rf = models.make_random_forest()
    rf.fit(Ts, ytr)
    joblib.dump(rf, MODELS / "random_forest.joblib")

    # --- 4. TensorFlow classifiers ---------------------------------------
    print("[4/4] TensorFlow MLP (binary) + attack-type classifier...")
    mlp = models.build_mlp(Ts.shape[1], 1)
    w = {0: 1.0, 1: float((ytr == 0).sum() / max(1, (ytr == 1).sum()))}
    mlp.fit(Ts, ytr, epochs=80, batch_size=64, verbose=0, validation_split=0.2,
            class_weight=w,
            callbacks=[keras.callbacks.EarlyStopping(patience=12,
                                                     restore_best_weights=True)])
    mlp.save(MODELS / "mlp_binary.keras")

    clf = models.build_mlp(Ts.shape[1], 6)
    clf.fit(Ts, atr, epochs=80, batch_size=64, verbose=0, validation_split=0.2,
            callbacks=[keras.callbacks.EarlyStopping(patience=12,
                                                     restore_best_weights=True)])
    clf.save(MODELS / "mlp_multiclass.keras")

    json.dump({"ae_threshold": thr, "n_features": int(Ts.shape[1])},
              open(MODELS / "meta.json", "w"), indent=2)
    print(f"\nModels saved to {MODELS}")


if __name__ == "__main__":
    main()
