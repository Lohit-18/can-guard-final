"""Payload encoding must be lossless enough for the physics features to work."""

import numpy as np
import pytest

from canguard import config as C
from canguard import signals
from canguard.vehicle import Vehicle


def make_state(speed_kmh=72.0, rpm=2400.0, gear=4):
    v = Vehicle(np.random.default_rng(0))
    v.speed = speed_kmh / 3.6
    v.rpm = rpm
    v.gear = gear
    v.accel = -1.5
    return v


@pytest.mark.parametrize("throttle", [0.0, 17.5, 63.0, 100.0])
def test_throttle_roundtrip(throttle):
    veh = make_state()
    data = signals.encode(0x0D0, veh, throttle, 0.0, 3)
    got = signals.decode(0x0D0, data)["throttle"]
    assert got == pytest.approx(throttle, abs=0.5)


@pytest.mark.parametrize("rpm", [750.0, 2400.0, 6100.0])
def test_rpm_roundtrip(rpm):
    veh = make_state(rpm=rpm)
    data = signals.encode(0x0C0, veh, 20.0, 0.0, 1)
    assert signals.decode(0x0C0, data)["rpm"] == pytest.approx(rpm, abs=0.3)


@pytest.mark.parametrize("kmh", [0.0, 31.4, 118.0])
def test_speed_roundtrip(kmh):
    veh = make_state(speed_kmh=kmh)
    data = signals.encode(0x1D0, veh, 0.0, 0.0, 0, np.random.default_rng(0))
    assert signals.decode(0x1D0, data)["speed"] == pytest.approx(kmh, abs=0.05)


def test_brake_signals_agree_with_each_other():
    veh = make_state()
    data = signals.encode(0x1A0, veh, 0.0, 55.0, 7)
    d = signals.decode(0x1A0, data)
    assert d["brake"] == pytest.approx(55.0 * 0.9, abs=0.5)
    assert d["brake_sw"] == 1
    assert d["decel"] == pytest.approx(veh.accel, abs=0.02)


def test_brake_switch_clears_when_pedal_is_up():
    veh = make_state()
    assert signals.decode(0x1A0, signals.encode(0x1A0, veh, 0.0, 0.0, 0))["brake_sw"] == 0


@pytest.mark.parametrize("can_id", [m.can_id for m in C.BUS])
def test_encoded_frames_are_eight_bytes(can_id):
    veh = make_state()
    data = signals.encode(can_id, veh, 30.0, 0.0, 5, np.random.default_rng(0))
    assert len(data) == 8


@pytest.mark.parametrize("can_id", [m.can_id for m in C.BUS if m.counter])
def test_checksum_and_counter_are_valid(can_id):
    veh = make_state()
    for ctr in range(16):
        data = signals.encode(can_id, veh, 40.0, 10.0, ctr, np.random.default_rng(0))
        assert signals.checksum_ok(can_id, data), f"{can_id:03X} counter {ctr}"
        assert signals.counter_of(can_id, data) == ctr


def test_a_tampered_byte_breaks_the_checksum():
    veh = make_state()
    data = bytearray(signals.encode(0x0D0, veh, 40.0, 0.0, 2))
    data[0] ^= 0x20
    assert not signals.checksum_ok(0x0D0, bytes(data))


def test_unknown_id_decodes_to_nothing():
    assert signals.decode(0x123, bytes(8)) == {}
    assert signals.counter_of(0x123, bytes(8)) is None
