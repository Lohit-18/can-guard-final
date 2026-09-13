"""
Bus simulator: turns a driven vehicle into a stream of CAN frames.

Each ECU has its own clock, so frames for one ID arrive at `cycle_ms` plus a
little jitter, and the streams from different ECUs interleave the way they do
on a real bus. The result is a trace with the same shape as a `candump` log:

    timestamp, can_id, dlc, data(hex)
"""

import numpy as np

from . import config as C
from . import signals
from .vehicle import Driver, Vehicle


def simulate(duration_s=600.0, seed=0):
    """
    Run the vehicle for `duration_s` and return:
      frames : list of (t, can_id, data(bytes), label, attack_code)
      states : (N,6) array of [t, speed_kmh, rpm, throttle, brake, gear]
    All frames are attack-free; attacks are layered on afterwards.
    """
    rng = np.random.default_rng(seed)
    veh = Vehicle(rng)
    drv = Driver(rng, duration_s)

    dt = C.TICK_MS / 1000.0
    n_ticks = int(duration_s / dt)

    # Each ECU starts its cycle at a random phase.
    next_tx = {m.can_id: rng.uniform(0, m.cycle_ms) / 1000.0 for m in C.BUS}
    counters = {m.can_id: int(rng.integers(0, 16)) for m in C.BUS}

    frames = []
    states = []
    throttle = brake = steer = 0.0

    for k in range(n_ticks):
        t = k * dt
        throttle, brake, steer = drv.control(veh, t, dt)
        veh.step(throttle, brake, steer, dt)

        if k % 10 == 0:
            states.append((t, veh.speed * 3.6, veh.rpm, throttle, brake, veh.gear))

        for m in C.BUS:
            if t >= next_tx[m.can_id]:
                # Stamp the frame with the time the ECU *meant* to send it, and
                # schedule the next one from that same instant. Using the tick
                # time instead would round every cycle up to the next
                # millisecond, biasing every period by half a tick - which the
                # per-ID timing features would then read as normal.
                due = next_tx[m.can_id]
                data = signals.encode(m.can_id, veh, throttle, brake,
                                      counters[m.can_id], rng)
                frames.append((due, m.can_id, data, 0, 0))
                counters[m.can_id] = (counters[m.can_id] + 1) & 0x0F
                jitter = rng.normal(0, m.cycle_ms * m.jitter_pct)
                next_tx[m.can_id] = due + (m.cycle_ms + jitter) / 1000.0

    frames.sort(key=lambda f: f[0])
    return frames, np.array(states, dtype=float)


def to_dataframe(frames):
    import pandas as pd
    return pd.DataFrame({
        "t": [f[0] for f in frames],
        "can_id": [f[1] for f in frames],
        "dlc": [len(f[2]) for f in frames],
        "data": [f[2].hex() for f in frames],
        "label": [f[3] for f in frames],
        "attack": [f[4] for f in frames],
    })
