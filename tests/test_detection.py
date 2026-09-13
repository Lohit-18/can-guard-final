"""
An end-to-end smoke test: does the thing actually detect anything?

Deliberately small and Scikit-learn only, so CI can run it in seconds without
waiting for TensorFlow to train. The full TensorFlow evaluation lives in
`python run_all.py`.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from canguard import attacks, features, models, simulator
from canguard import config as C

KINDS = ["throttle_spoof", "fake_brake", "dos_flood", "fuzzing"]


def build(duration, seed, kinds=None, episodes=2):
    frames, states = simulator.simulate(duration_s=duration, seed=seed)
    if kinds:
        frames, _ = attacks.apply_attacks(frames, states, seed=seed + 100,
                                          duration=duration, kinds=kinds,
                                          episodes_per_kind=episodes)
    return features.build_windows(frames)


@pytest.fixture(scope="module")
def split():
    # Enough episodes that both threat models - injection and masquerade - land
    # in each split. With two or three, a coin flip can put every stealthy
    # episode in the test set and the result says nothing.
    Xc, _, _, _, _ = build(90.0, 21)                       # clean, for the scaler
    Xtr, _, ytr, _, _ = build(300.0, 22, KINDS, 6)
    Xte, names, yte, ate, _ = build(210.0, 23, KINDS, 5)
    return Xc, Xtr, ytr, Xte, yte, ate, names


def test_there_is_something_to_detect(split):
    _, _, ytr, _, yte, _, _ = split
    assert ytr.sum() > 100 and yte.sum() > 80
    assert ytr.mean() < 0.5 and yte.mean() < 0.5, "attacks should be the minority"


def test_supervised_model_detects_attacks(split):
    Xc, Xtr, ytr, Xte, yte, _, _ = split
    sc = models.make_scaler(Xc)
    rf = models.make_random_forest()
    rf.fit(models.scale(sc, Xtr), ytr)
    score = rf.predict_proba(models.scale(sc, Xte))[:, 1]
    auc = roc_auc_score(yte, score)
    assert auc > 0.96, f"ROC-AUC dropped to {auc:.3f}"


def test_unsupervised_model_flags_attacks_without_ever_seeing_one(split):
    Xc, _, _, Xte, yte, _, _ = split
    sc = models.make_scaler(Xc)
    iso = models.make_isolation_forest()
    iso.fit(models.scale(sc, Xc))
    auc = roc_auc_score(yte, -iso.score_samples(models.scale(sc, Xte)))
    assert auc > 0.75, f"ROC-AUC dropped to {auc:.3f}"


def test_false_alarms_on_clean_traffic_stay_low(split):
    Xc, Xtr, ytr, _, _, _, _ = split
    sc = models.make_scaler(Xc)
    rf = models.make_random_forest()
    rf.fit(models.scale(sc, Xtr), ytr)
    fresh, _, y_fresh, _, _ = build(90.0, 24)               # clean, unseen
    assert y_fresh.sum() == 0
    rate = rf.predict(models.scale(sc, fresh)).mean()
    assert rate < 0.05, f"{100*rate:.1f}% of clean windows raised an alarm"


def test_every_attack_family_is_caught_at_least_sometimes(split):
    Xc, Xtr, ytr, Xte, yte, ate, _ = split
    sc = models.make_scaler(Xc)
    rf = models.make_random_forest()
    rf.fit(models.scale(sc, Xtr), ytr)
    pred = rf.predict(models.scale(sc, Xte))
    for name in KINDS:
        code = C.NAME_TO_CODE[name]
        m = (ate == code) & (yte == 1)
        if m.sum() < 5:
            continue
        # a floor, not a target - the published per-family numbers come from
        # the full run, on far more traffic than this smoke test uses
        assert pred[m].mean() > 0.5, f"{name} recall {pred[m].mean():.2f}"


def test_scaling_is_bounded(split):
    """Clipping keeps a DoS flood from producing values that break training."""
    Xc, _, _, Xte, _, _, _ = split
    Z = models.scale(models.make_scaler(Xc), Xte)
    assert np.isfinite(Z).all()
    assert np.abs(Z).max() <= models.CLIP + 1e-6
