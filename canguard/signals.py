"""
Payload encoding / decoding - our miniature DBC file.

Real CAN payloads are packed binary, not JSON. Encoding the vehicle state the
way an ECU actually would matters for two reasons:

  1. byte-level features (entropy, bit-flip distance) become meaningful, and
  2. the rolling counter + checksum give us a cheap integrity signal that
     catches sloppy attackers - while a *careful* attacker gets them right,
     which is exactly why we also need the machine-learning layer.
"""

import numpy as np

from . import config as C


def _u16(v):
    v = int(max(0, min(65535, round(v))))
    return [(v >> 8) & 0xFF, v & 0xFF]


def _i16(v):
    v = int(max(-32768, min(32767, round(v))))
    if v < 0:
        v += 65536
    return [(v >> 8) & 0xFF, v & 0xFF]


def _u8(v):
    return int(max(0, min(255, round(v))))


def checksum(data, can_id):
    """XOR-fold of the first 6 bytes plus the ID - the sort of thing a real
    ECU uses to spot a corrupted frame."""
    x = can_id & 0xFF
    for b in data[:6]:
        x ^= b
    return ((x + 0x5A) ^ 0xA5) & 0xFF


def encode(can_id, veh, throttle_pct, brake_pct, counter, rng=None):
    """Build the 8-byte payload for `can_id` from the current vehicle state."""
    spd_kmh = veh.speed * 3.6
    n = (counter & 0x0F)

    if can_id == 0x0C0:                               # ENGINE_RPM
        load = np.clip(throttle_pct * 0.9 + 8, 0, 100)
        d = (_u16(veh.rpm / 0.25)
             + [_u8(load * 2.55), _u8(90 + (veh.rpm - 750) / 400), 0x00, 0x00])
    elif can_id == 0x0D0:                             # THROTTLE
        # the plate lags the pedal slightly
        plate = np.clip(throttle_pct * 0.97 + 1.5, 0, 100)
        torque = throttle_pct * 3.2 + 40
        d = [_u8(throttle_pct * 2.55), _u8(plate * 2.55)] + _u16(torque) + [0x01, 0x00]
    elif can_id == 0x1A0:                             # BRAKE
        press = brake_pct * 0.9                       # bar-ish
        flags = 0x01 if brake_pct > 1.5 else 0x00
        d = [_u8(press * 2.55), flags] + _i16(veh.accel * 100) + [0x00, 0x00]
    elif can_id == 0x1D0:                             # WHEEL_SPEED
        wob = rng.normal(0, 0.05, 2) if rng is not None else np.zeros(2)
        d = (_u16(spd_kmh * 100)
             + _u16(max(0, spd_kmh + wob[0]) * 100)
             + _u16(max(0, spd_kmh + wob[1]) * 100))
    elif can_id == 0x2C0:                             # STEERING
        d = (_i16(veh.steering * 10) + _i16(veh.yaw_rate * 100)
             + [0x00, 0x00, 0x00, 0x00])
        return bytes(d[:8])
    elif can_id == 0x3B0:                             # GEAR
        d = [veh.gear, 0x02] + _u16(veh.rpm / 0.5) + [0x00, 0x00, 0x00, 0x00]
        return bytes(d[:8])
    elif can_id == 0x4B1:                             # ABS_ESP
        lat = veh.yaw_rate * veh.speed / 57.3
        d = (_i16(veh.yaw_rate * 100) + _i16(lat * 1000)
             + [0x01 if brake_pct > 60 else 0x00, 0x00, 0x00, 0x00])
        return bytes(d[:8])
    elif can_id == 0x5A0:                             # BODY_STATUS
        lights = 0x04 if brake_pct > 1.5 else 0x00
        d = [lights, 0x00, 0x00, _u8(spd_kmh / 2), 0x00, 0x00, 0x00, 0x00]
        return bytes(d[:8])
    elif can_id == 0x7E8:                             # OBD-II mode 01 response
        pid = [0x0C, 0x0D, 0x11, 0x05][counter % 4]
        if pid == 0x0C:
            val = _u16(veh.rpm * 4)
        elif pid == 0x0D:
            val = [_u8(spd_kmh), 0x00]
        elif pid == 0x11:
            val = [_u8(throttle_pct * 2.55), 0x00]
        else:
            val = [_u8(88 + 40), 0x00]
        return bytes([0x04, 0x41, pid] + val + [0x55, 0x55, 0x55][:3])
    else:
        return bytes(8)

    d = list(d)[:6]
    d.append(n)
    d.append(checksum(d, can_id))
    return bytes(d[:8])


# --- decoding: the IDS is allowed to know the DBC, same as the OEM ---------

def decode(can_id, data):
    """Return a dict of physical values, or {} for an unknown/garbage frame."""
    if len(data) < 8:
        return {}
    if can_id == 0x0C0:
        return {"rpm": ((data[0] << 8) | data[1]) * 0.25}
    if can_id == 0x0D0:
        return {"throttle": data[0] / 2.55, "plate": data[1] / 2.55}
    if can_id == 0x1A0:
        raw = (data[2] << 8) | data[3]
        return {"brake": data[0] / 2.55,
                "brake_sw": data[1] & 0x01,
                "decel": (raw - 65536 if raw > 32767 else raw) / 100.0}
    if can_id == 0x1D0:
        return {"speed": ((data[0] << 8) | data[1]) / 100.0}
    return {}


def counter_of(can_id, data):
    """Rolling counter for IDs that carry one, else None."""
    spec = C.BY_ID.get(can_id)
    if spec is None or not spec.counter or len(data) < 8:
        return None
    return data[6] & 0x0F


def checksum_ok(can_id, data):
    spec = C.BY_ID.get(can_id)
    if spec is None or not spec.counter or len(data) < 8:
        return True
    return checksum(list(data[:6]) + [data[6]], can_id) == data[7]
