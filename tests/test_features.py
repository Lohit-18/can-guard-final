"""Feature extraction: shape, sanity, determinism, and no label leakage."""

import numpy as np
import pytest

from canguard import attacks, features, simulator
from canguard import config as C


@pytest.fixture(scope="module")
def clean_windows():
    frames, _ = simulator.simulate(duration_s=25.0, seed=5)
    return features.build_windows(frames)


@pytest.fixture(scope="module")
def attacked():
    frames, states = simulator.simulate(duration_s=90.0, seed=9)
    frames, _ = attacks.apply_attacks(
        frames, states, seed=9, duration=90.0,
        kinds=["throttle_spoof", "fake_brake", "dos_flood", "fuzzing", "replay"],
        episodes_per_kind=2)
    return features.build_windows(frames)


def test_shape_matches_the_feature_names(clean_windows):
    X, names, y, acode, starts = clean_windows
    assert X.shape[1] == len(names) == 67
    assert len(y) == len(acode) == len(starts) == X.shape[0]
    assert len(set(names)) == len(names), "duplicate feature name"


def test_no_nan_or_infinity(clean_windows, attacked):
    for X, *_ in (clean_windows, attacked):
        assert np.isfinite(X).all()


def test_clean_traffic_is_entirely_unlabelled(clean_windows):
    _, _, y, acode, _ = clean_windows
    assert y.sum() == 0 and acode.sum() == 0


def test_extraction_is_deterministic():
    frames, _ = simulator.simulate(duration_s=15.0, seed=11)
    a, _, ya, _, _ = features.build_windows(frames)
    b, _, yb, _, _ = features.build_windows(frames)
    assert np.array_equal(a, b) and np.array_equal(ya, yb)


def test_clean_bus_sits_near_its_nominal_cycle(clean_windows):
    """cnt_ratio and iat_ratio are defined so that healthy traffic scores ~1.0.

    Averaged, not per window: a 100 ms message lands 2 or 3 times in a 250 ms
    window, so any single window reads 0.8 or 1.2 by arithmetic alone.
    """
    X, names, *_ = clean_windows
    for c in C.KNOWN_IDS:
        col = X[:, names.index(f"cnt_ratio_{c:03X}")]
        assert col.mean() == pytest.approx(1.0, abs=0.05), f"cnt_ratio_{c:03X}"
        col = X[:, names.index(f"iat_ratio_{c:03X}")]
        assert col.mean() == pytest.approx(1.0, abs=0.05), f"iat_ratio_{c:03X}"


def test_clean_traffic_has_intact_counters_and_checksums(clean_windows):
    X, names, *_ = clean_windows
    assert X[:, names.index("ctr_anom_rate")].max() == 0.0
    assert X[:, names.index("cksum_bad_rate")].max() == 0.0


def test_physics_mismatch_features_separate_attacks(attacked):
    """The whole argument for the physics family rests on this holding."""
    X, names, y, *_ = attacked
    assert y.sum() > 20, "not enough attack windows to judge"
    for col in ("mismatch_thr_accel", "mismatch_brake_decel", "both_pedals"):
        i = names.index(col)
        assert X[y == 1, i].mean() > X[y == 0, i].mean(), col


def test_attack_codes_are_valid(attacked):
    _, _, y, acode, _ = attacked
    assert set(np.unique(acode)) <= set(C.ATTACK_TYPES)
    assert (acode[y == 0] == 0).all()


def test_live_buffer_matches_a_batch_window():
    """`build_one` is what the online detector uses; it must agree with the
    batch path, or the live demo would be scoring different numbers."""
    frames, _ = simulator.simulate(duration_s=12.0, seed=13)
    w = C.WINDOW_MS / 1000.0
    t0 = 5.0
    buf = [f for f in frames if t0 <= f[0] < t0 + w]
    X_one, _, _ = features.build_one(buf)
    X_batch, _, _, _, starts = features.build_windows(
        frames, window_ms=C.WINDOW_MS, stride_ms=C.STRIDE_MS)
    k = int(np.argmin(np.abs(starts - buf[0][0])))
    # the same traffic, so the timing and integrity columns must line up closely
    assert X_one.shape == (1, X_batch.shape[1])
    assert np.allclose(X_one[0, :9], X_batch[k, :9], rtol=0.25, atol=0.25)
