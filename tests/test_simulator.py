"""The simulated bus has to behave like a bus, or none of the features mean anything."""

import numpy as np
import pytest

from canguard import attacks, simulator
from canguard import config as C

DURATION = 20.0


@pytest.fixture(scope="module")
def clean():
    return simulator.simulate(duration_s=DURATION, seed=7)


def test_clean_trace_carries_no_attack_labels(clean):
    frames, _ = clean
    assert frames, "simulator produced nothing"
    assert all(f[3] == 0 and f[4] == 0 for f in frames)


def test_frames_are_ordered_in_time(clean):
    frames, _ = clean
    t = np.array([f[0] for f in frames])
    assert np.all(np.diff(t) >= 0)


def test_only_known_ids_appear(clean):
    frames, _ = clean
    assert set(f[1] for f in frames) <= set(C.KNOWN_IDS)


@pytest.mark.parametrize("spec", C.BUS, ids=[m.name for m in C.BUS])
def test_each_ecu_holds_its_cycle_time(clean, spec):
    frames, _ = clean
    t = np.array([f[0] for f in frames if f[1] == spec.can_id])
    assert len(t) > 10, f"{spec.name} barely transmitted"
    iat = np.diff(t) * 1000.0
    nominal = spec.cycle_ms
    # mean within 2% of nominal, jitter within three times the ECU's spec
    assert iat.mean() == pytest.approx(nominal, rel=0.02)
    assert iat.std() < nominal * spec.jitter_pct * 3


def test_vehicle_state_stays_physical(clean):
    _, states = clean
    t, kmh, rpm, thr, brk, gear = states.T
    assert (kmh >= -1e-9).all() and kmh.max() < 200
    assert (rpm >= C.IDLE_RPM - 1).all() and rpm.max() <= C.MAX_RPM + 1
    assert (thr >= 0).all() and (thr <= 100).all()
    assert (brk >= 0).all() and (brk <= 100).all()
    assert set(np.unique(gear)) <= set(range(1, len(C.GEAR_RATIOS) + 1))


def test_driver_never_holds_both_pedals_down(clean):
    """The physics feature `both_pedals` is only meaningful if honest traffic
    never does this."""
    _, states = clean
    thr, brk = states[:, 3], states[:, 4]
    assert np.minimum(thr, brk).max() < 5.0


def test_injection_raises_the_message_rate_masquerade_does_not(clean):
    frames, states = clean
    rng = np.random.default_rng(0)
    eps = [(6.0, 10.0)]
    base = sum(1 for f in frames if f[1] == 0x0D0)

    inj = attacks.throttle_spoof(frames, states, rng, eps, "injection")
    mas = attacks.throttle_spoof(frames, states, rng, eps, "masquerade")

    assert sum(1 for f in inj if f[1] == 0x0D0) > base * 1.1
    assert sum(1 for f in mas if f[1] == 0x0D0) == base
    assert any(f[3] == 1 for f in inj) and any(f[3] == 1 for f in mas)


def test_masquerade_keeps_counters_and_checksums_valid(clean):
    """That is what makes it the hard case - only the physics gives it away."""
    from canguard import signals
    frames, states = clean
    mas = attacks.throttle_spoof(frames, states, np.random.default_rng(1),
                                 [(6.0, 10.0)], "masquerade")
    spoofed = [f for f in mas if f[3] == 1]
    assert spoofed
    assert all(signals.checksum_ok(f[1], f[2]) for f in spoofed)


def test_apply_attacks_labels_every_family():
    frames, states = simulator.simulate(duration_s=60.0, seed=3)
    kinds = ["throttle_spoof", "fake_brake", "dos_flood", "fuzzing", "replay"]
    out, log = attacks.apply_attacks(frames, states, seed=3, duration=60.0,
                                     kinds=kinds, episodes_per_kind=1)
    seen = {f[4] for f in out if f[3] == 1}
    assert seen == {C.NAME_TO_CODE[k] for k in kinds}
    assert len(log) == len(kinds)
