"""
Step 4 - evaluate on the held-out test trace and write the report.

Reported for every model: accuracy, precision, recall, F1 and ROC-AUC, plus a
breakdown of recall per attack family. The `replay` family never appears in
training, so its column is the honest test of whether the unsupervised model
earns its keep.
"""

import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from canguard import config as C
from canguard import models, viz

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, MODELS, REPORTS = ROOT / "data", ROOT / "models", ROOT / "reports"


def metrics(y, pred, score):
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, score),
        false_alarm_rate=float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum())),
    )


def main():
    REPORTS.mkdir(exist_ok=True)
    viz.style()

    z = np.load(DATA / "test_features.npz", allow_pickle=True)
    X, y, acode, starts = z["X"], z["y"], z["acode"], z["starts"]
    names = list(z["names"])

    bundle = joblib.load(MODELS / "scaler.joblib")
    Xs = models.scale(bundle["scaler"], X)
    meta = json.load(open(MODELS / "meta.json"))

    from tensorflow import keras

    ae = keras.models.load_model(MODELS / "autoencoder.keras")
    mlp = keras.models.load_model(MODELS / "mlp_binary.keras")
    clf = keras.models.load_model(MODELS / "mlp_multiclass.keras")
    rf = joblib.load(MODELS / "random_forest.joblib")
    iso = joblib.load(MODELS / "isolation_forest.joblib")

    # ---------------- scores -------------------------------------------
    err = models.recon_error(ae, Xs)
    thr = meta["ae_threshold"]
    res = {
        "Random Forest (supervised)":   (rf.predict(Xs), rf.predict_proba(Xs)[:, 1]),
        "TF Neural Net (supervised)":   (None, mlp.predict(Xs, verbose=0).ravel()),
        "TF Autoencoder (unsupervised)": ((err > thr).astype(int), err),
        "Isolation Forest (unsupervised)": ((iso.predict(Xs) == -1).astype(int),
                                            -iso.score_samples(Xs)),
    }
    res["TF Neural Net (supervised)"] = (
        (res["TF Neural Net (supervised)"][1] > 0.5).astype(int),
        res["TF Neural Net (supervised)"][1])

    table, curves = {}, []
    for name, (pred, score) in res.items():
        table[name] = metrics(y, pred, score)
        fpr, tpr, _ = roc_curve(y, score)
        curves.append((name, fpr, tpr, table[name]["roc_auc"]))

    # ---------------- per-attack recall --------------------------------
    fams = [c for c in sorted(set(acode.tolist())) if c != 0]
    fam_names = [C.ATTACK_TYPES[c] for c in fams]
    per_attack = {}
    for name, (pred, _) in res.items():
        per_attack[name] = []
        for c in fams:
            m = (acode == c) & (y == 1)
            per_attack[name].append(float(pred[m].mean()) if m.any() else 0.0)

    # ---------------- injection vs masquerade --------------------------
    # Which threat model each attack window belongs to, from the episode log.
    ep_path = DATA / "test_episodes.csv"
    eps = []
    if ep_path.exists():
        e = pd.read_csv(ep_path)
        eps = list(zip(e.attack, e.start_s, e.end_s))
    w = C.WINDOW_MS / 1000.0
    subtype = np.array(["-"] * len(y), dtype=object)
    for nm, s, en in eps:
        hit = (starts < en) & (starts + w > s)
        subtype[hit & (y == 1)] = nm
    subs = [s for s in sorted(set(subtype.tolist()))
            if "/" in s and int(((subtype == s)).sum()) >= 10]
    per_sub = {name: [float(pred[subtype == s].mean()) for s in subs]
               for name, (pred, _) in res.items()}

    # ---------------- attack-type classifier ---------------------------
    yc = clf.predict(Xs, verbose=0).argmax(axis=1)
    true_c = np.where(y == 1, acode, 0)
    cls_acc = float((yc == true_c).mean())

    # ---------------- print --------------------------------------------
    print("\n" + "=" * 78)
    print("CAN-Guard  -  results on the held-out test trace")
    print("=" * 78)
    print(f"{len(y):,} windows  |  {int(y.sum()):,} contain an attack "
          f"({100*y.mean():.1f}%)\n")
    hdr = (f"{'model':<34}{'acc':>7}{'prec':>8}{'rec':>8}"
           f"{'F1':>8}{'AUC':>8}{'FA rate':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k, v in table.items():
        print(f"{k:<34}{v['accuracy']:>7.3f}{v['precision']:>8.3f}"
              f"{v['recall']:>8.3f}{v['f1']:>8.3f}{v['roc_auc']:>8.3f}"
              f"{v['false_alarm_rate']:>9.3f}")

    print("\nRecall by attack family  (* = never seen during training)")
    cols = "".join(f"{n.replace('_','-')[:14]:>16}" for n in fam_names)
    print(f"{'model':<34}{cols}")
    print("-" * (34 + 16 * len(fams)))
    for k, v in per_attack.items():
        print(f"{k:<34}" + "".join(f"{100*x:>15.1f}%" for x in v))
    print(f"{'':<34}" + "".join(
        f"{('*' if n == 'replay' else ''):>16}" for n in fam_names))

    if subs:
        print("\nRecall by threat model  (masquerade = attacker replaced the "
              "real ECU; nothing about the timing is wrong)")
        cols = "".join(
            f"{s.split('/')[0][:9] + '/' + s.split('/')[1][:4]:>22}" for s in subs)
        print(f"{'model':<34}{cols}")
        print("-" * (34 + 22 * len(subs)))
        for k, v in per_sub.items():
            print(f"{k:<34}" + "".join(f"{100*x:>21.1f}%" for x in v))

    print(f"\nAttack-type classifier accuracy (6 classes): {cls_acc:.3f}")

    # ---------------- figures ------------------------------------------
    best = max(table, key=lambda k: table[k]["f1"])
    pred_best = res[best][0]
    cm = confusion_matrix(y, pred_best)

    viz.roc_curves(curves, REPORTS / "roc_curves.png",
                   "Detection performance on unseen traffic",
                   "ROC across 3,358 windows of a held-out 7-minute drive")

    viz.confusion(cm, ["normal", "attack"], REPORTS / "confusion_matrix.png",
                  f"Confusion matrix - {best}",
                  "counts and row-normalised percentages")

    viz.grouped_bars(
        [n.replace("_", "\n") for n in fam_names],
        [(k, v) for k, v in per_attack.items()],
        REPORTS / "recall_by_attack.png",
        "Which attacks does each model actually catch?",
        "replay was held out of training entirely - it is the zero-day test",
        "windows detected")

    if subs:
        viz.grouped_bars(
            [s.replace("_", " ").replace("/", "\n") for s in subs],
            [(k, v) for k, v in per_sub.items()],
            REPORTS / "recall_by_threat_model.png",
            "Loud attacks are easy. Stealthy ones are the real test.",
            "masquerade = the attacker replaced the real ECU, so timing and "
            "checksums stay perfect",
            "windows detected")

    imp = np.argsort(rf.feature_importances_)[::-1][:15]
    viz.importance([names[i] for i in imp], rf.feature_importances_[imp],
                   REPORTS / "feature_importance.png",
                   "What the Random Forest relies on",
                   "top 15 of 67 features")

    viz.anomaly_timeline(starts, err, thr, eps,
                         REPORTS / "anomaly_timeline.png",
                         "The autoencoder's view of the drive",
                         "trained only on attack-free traffic; "
                         "shaded bands are real attacks")

    json.dump({"overall": table, "per_attack_recall": per_attack,
               "attack_families": fam_names,
               "per_threat_model_recall": per_sub, "threat_models": subs,
               "multiclass_accuracy": cls_acc,
               "n_windows": int(len(y)), "n_attack_windows": int(y.sum())},
              open(REPORTS / "metrics.json", "w"), indent=2)

    pd.DataFrame(table).T.to_csv(REPORTS / "metrics.csv")
    print(f"\nCharts + metrics written to {REPORTS}")


if __name__ == "__main__":
    main()
