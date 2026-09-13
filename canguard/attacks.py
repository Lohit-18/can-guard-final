"""
Attack injection.

Two threat models are simulated, because they are very different problems:

* **injection** - the attacker simply transmits extra frames. The genuine ECU
  keeps talking too, so the bus now carries that ID roughly twice as often and
  the rolling counters stop making sense. Loud, but this is what most real
  CAN attacks look like.

* **masquerade** - the attacker first silences the genuine ECU (a bus-off
  attack) and then transmits in its place, at the correct cycle time, with a
  correct counter and checksum. Nothing about the *traffic* is wrong. The only
  thing that gives it away is that the car does not behave the way the frames
  claim it does.

A detector that only counts messages will catch the first and miss the second.
That is the whole reason CAN-Guard learns the physics as well as the timing.
"""

import numpy as np

from . import config as C
from . import signals


class _State:
    """Minimal stand-in for Vehicle, so we can reuse the payload encoder."""
    __slots__ = ("speed", "rpm", "accel", "steering", "yaw_rate", "gear")


def _state_at(states, t):
    i = int(np.clip(np.searchsorted(states[:, 0], t), 0, len(states) - 1))
    s = _State()
    s.speed = states[i, 1] / 3.6
    s.rpm = states[i, 2]
    s.accel = 0.0
    s.steering = 0.0
    s.yaw_rate = 0.0
    s.gear = int(states[i, 5])
    return s, states[i, 3], states[i, 4]   # state, real throttle, real brake


def _pick_episodes(rng, duration, n, dur_range, busy):
    """Choose `n` non-overlapping [start, end) windows that avoid `busy`."""
    out = []
    for _ in range(n * 40):
        if len(out) == n:
            break
        d = rng.uniform(*dur_range)
        s = rng.uniform(5.0, max(6.0, duration - d - 5.0))
        e = s + d
        if any(not (e < b0 or s > b1) for b0, b1 in busy + out):
            continue
        out.append((s, e))
    return sorted(out)


# --------------------------------------------------------------------------

def _ramp(t, s, e, rise=0.30):
    """Attackers do not step a pedal from 0 to 80 in one frame - they ramp.
    Without this the spoofed signal would be detectable from its jitter alone,
    which would make the whole problem artificially easy."""
    if t <= s or t >= e:
        return 0.0
    up = min(1.0, (t - s) / rise)
    down = min(1.0, (e - t) / rise)
    return float(min(up, down))


def _episode_params(rng, episodes, lo, hi):
    """One target magnitude per episode, not per frame."""
    return {ep: float(rng.uniform(lo, hi)) for ep in episodes}


def throttle_spoof(frames, states, rng, episodes, mode):
    """Unauthorized acceleration: THROTTLE (0x0D0) frames claim a pedal
    position the driver never asked for."""
    code = C.NAME_TO_CODE["throttle_spoof"]
    # a mix of blatant and restrained attackers
    targets = _episode_params(rng, episodes, 18.0, 65.0)
    out, extra = [], []
    for t, cid, data, lab, at in frames:
        ep = next((e for e in episodes if e[0] <= t < e[1]), None)
        if ep is None or cid != 0x0D0:
            out.append((t, cid, data, lab, at))
            continue
        st, real_thr, real_brk = _state_at(states, t)
        bump = targets[ep] * _ramp(t, ep[0], ep[1]) + rng.normal(0, 0.4)
        fake = float(np.clip(real_thr + bump, 0, 100))
        ctr = signals.counter_of(cid, data)
        if mode == "masquerade":
            # genuine ECU is silenced - its frame is replaced in place
            out.append((t, cid, signals.encode(cid, st, fake, 0.0, ctr, rng), 1, code))
        else:
            out.append((t, cid, data, 0, 0))
            off = rng.uniform(0.004, 0.014)
            if t + off < ep[1]:
                extra.append((t + off, cid,
                              signals.encode(cid, st, fake, 0.0, ctr, rng), 1, code))
    return sorted(out + extra, key=lambda f: f[0])


def fake_brake(frames, states, rng, episodes, mode):
    """Forged brake-pressure frames. The car never slows down, and the body
    controller never turns the brake lights on - two things that should
    always happen together."""
    code = C.NAME_TO_CODE["fake_brake"]
    targets = _episode_params(rng, episodes, 25.0, 90.0)
    # Half of the attackers are thorough enough to also fake the brake-light
    # bit in BODY_STATUS, closing the obvious cross-ECU giveaway.
    also_lights = {ep: bool(rng.random() < 0.5) for ep in episodes}
    out, extra = [], []
    for t, cid, data, lab, at in frames:
        ep = next((e for e in episodes if e[0] <= t < e[1]), None)
        if ep is not None and cid == 0x5A0 and also_lights[ep]:
            d = bytearray(data)
            d[0] |= 0x04
            out.append((t, cid, bytes(d), 1, code))
            continue
        if ep is None or cid != 0x1A0:
            out.append((t, cid, data, lab, at))
            continue
        st, real_thr, real_brk = _state_at(states, t)
        fake = float(np.clip(targets[ep] * _ramp(t, ep[0], ep[1])
                             + rng.normal(0, 0.3), 0, 100))
        # a careful attacker keeps the payload internally consistent: the
        # decel it reports matches the pressure it claims.
        st.accel = -(fake / 100.0) * 5.5
        ctr = signals.counter_of(cid, data)
        if mode == "masquerade":
            out.append((t, cid, signals.encode(cid, st, 0.0, fake, ctr, rng), 1, code))
        else:
            out.append((t, cid, data, 0, 0))
            off = rng.uniform(0.004, 0.014)
            if t + off < ep[1]:
                extra.append((t + off, cid,
                              signals.encode(cid, st, 0.0, fake, ctr, rng), 1, code))
    return sorted(out + extra, key=lambda f: f[0])


def dos_flood(frames, states, rng, episodes):
    """0x000 is the highest-priority ID on the bus; flooding it wins every
    arbitration and starves every real ECU."""
    code = C.NAME_TO_CODE["dos_flood"]
    extra = []
    for s, e in episodes:
        t = s
        while t < e:
            extra.append((t, 0x000, bytes([0] * 8), 1, code))
            t += rng.uniform(0.00025, 0.00045)
    # genuine frames lose arbitration and are pushed back
    out = []
    for t, cid, data, lab, at in frames:
        if any(s <= t < e for s, e in episodes):
            t = t + rng.uniform(0.0005, 0.004)
        out.append((t, cid, data, lab, at))
    return sorted(out + extra, key=lambda f: f[0])


def fuzzing(frames, states, rng, episodes):
    """Blind fuzzing: random IDs, random payloads. Cheap for the attacker,
    and how most CAN exploits are discovered in the first place."""
    code = C.NAME_TO_CODE["fuzzing"]
    extra = []
    for s, e in episodes:
        t = s
        while t < e:
            if rng.random() < 0.45:
                cid = int(rng.choice(C.KNOWN_IDS))
            else:
                cid = int(rng.integers(0, 0x7FF))
            extra.append((t, cid, bytes(rng.integers(0, 256, 8).tolist()), 1, code))
            t += rng.uniform(0.0008, 0.012)
    return sorted(frames + extra, key=lambda f: f[0])


def replay(frames, states, rng, episodes):
    """Record a slice of perfectly valid traffic and play it back later. Every
    frame is well-formed; only the context is a lie."""
    code = C.NAME_TO_CODE["replay"]
    arr_t = np.array([f[0] for f in frames])
    extra = []
    for s, e in episodes:
        dur = e - s
        src = rng.uniform(2.0, max(3.0, s - dur - 2.0))
        i0, i1 = np.searchsorted(arr_t, [src, src + dur])
        for j in range(i0, i1):
            ft, cid, data, _, _ = frames[j]
            extra.append((s + (ft - src), cid, data, 1, code))
    return sorted(frames + extra, key=lambda f: f[0])


# --------------------------------------------------------------------------

def apply_attacks(frames, states, seed=0, duration=600.0, kinds=None,
                  episodes_per_kind=6):
    """Layer a mix of attack episodes onto a clean trace."""
    rng = np.random.default_rng(seed)
    kinds = kinds or ["throttle_spoof", "fake_brake", "dos_flood", "fuzzing", "replay"]
    busy, log = [], []

    for kind in kinds:
        if kind in ("dos_flood",):
            dur_range = (0.4, 1.6)
        elif kind == "fuzzing":
            dur_range = (0.6, 2.5)
        else:
            dur_range = (1.5, 5.0)
        eps = _pick_episodes(rng, duration, episodes_per_kind, dur_range, busy)
        busy += eps
        if not eps:
            continue

        if kind in ("throttle_spoof", "fake_brake"):
            # split the episodes between the loud and the stealthy threat model
            inj = [e for e in eps if rng.random() < 0.5]
            mas = [e for e in eps if e not in inj]
            fn = throttle_spoof if kind == "throttle_spoof" else fake_brake
            if inj:
                frames = fn(frames, states, rng, inj, "injection")
            if mas:
                frames = fn(frames, states, rng, mas, "masquerade")
            log += [(kind + ("/injection" if e in inj else "/masquerade"), *e)
                    for e in eps]
        else:
            fn = {"dos_flood": dos_flood, "fuzzing": fuzzing, "replay": replay}[kind]
            frames = fn(frames, states, rng, eps)
            log += [(kind, *e) for e in eps]

    return frames, sorted(log, key=lambda r: r[1])
