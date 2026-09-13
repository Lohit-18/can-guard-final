"""
Feature engineering: 250 ms of raw bus traffic -> one row of numbers.

The features fall into four families, and each one exists because a specific
class of attack breaks it:

  timing     - how often each ID appears, and how regular it is.
               Broken by: injection, DoS, fuzzing, replay.
  integrity  - rolling counters, checksums, duplicate payloads, bit churn.
               Broken by: injection, fuzzing, replay.
  physics    - do throttle, RPM, wheel speed and brake pressure tell the
               same story?  Broken by: masquerade spoofing, replay.
  cross-ECU  - do independent ECUs agree (brake pressure vs brake lights)?
               Broken by: any single-ECU spoof.

Nothing here peeks at the label. `mismatch_*` columns are hand-built physics
residuals, which is the part a plain frequency-based IDS does not have.
"""

import numpy as np

from . import config as C

POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
ID_INDEX = {cid: i for i, cid in enumerate(C.KNOWN_IDS)}
N_IDS = len(C.KNOWN_IDS)


# ---------------------------------------------------------------- raw arrays

def frames_to_arrays(frames):
    n = len(frames)
    t = np.empty(n)
    cid = np.empty(n, dtype=np.int32)
    data = np.zeros((n, 8), dtype=np.uint8)
    label = np.empty(n, dtype=np.int8)
    atype = np.empty(n, dtype=np.int8)
    for i, (ft, fid, fd, fl, fa) in enumerate(frames):
        t[i] = ft
        cid[i] = fid
        b = np.frombuffer(fd[:8], dtype=np.uint8)
        data[i, : len(b)] = b
        label[i] = fl
        atype[i] = fa
    return t, cid, data, label, atype


def per_frame_flags(t, cid, data):
    """One vectorised pass that gives every frame its integrity / timing flags
    relative to the previous frame carrying the same ID."""
    n = len(t)
    o = np.lexsort((t, cid))                  # group by ID, ordered in time
    same = np.zeros(n, dtype=bool)
    same[1:] = cid[o][1:] == cid[o][:-1]

    cur, prv = o, np.roll(o, 1)

    iat_id = np.full(n, np.nan)
    iat_id[cur[same]] = t[cur[same]] - t[prv[same]]

    xor = np.bitwise_xor(data[cur], data[prv])
    bits = POPCNT[xor].sum(axis=1).astype(float)
    bitdiff = np.full(n, np.nan)
    bitdiff[cur[same]] = bits[same]

    dup = np.zeros(n, dtype=bool)
    dup[cur[same]] = (xor[same].sum(axis=1) == 0)

    # rolling-counter continuity (only for IDs that carry one)
    has_ctr = np.array([C.BY_ID[c].counter if c in C.BY_ID else False
                        for c in C.KNOWN_IDS])
    ctr_capable = np.zeros(n, dtype=bool)
    for c in C.KNOWN_IDS:
        if C.BY_ID[c].counter:
            ctr_capable |= (cid == c)
    ctr = (data[:, 6] & 0x0F).astype(np.int16)
    expected = (ctr[prv] + 1) & 0x0F
    ctr_bad = np.zeros(n, dtype=bool)
    m = same & ctr_capable[cur]
    ctr_bad[cur[m]] = ctr[cur[m]] != expected[m]

    # checksum
    x = (cid & 0xFF).astype(np.uint16)
    for k in range(6):
        x = np.bitwise_xor(x, data[:, k].astype(np.uint16))
    calc = (((x + 0x5A) ^ 0xA5) & 0xFF).astype(np.uint8)
    cks_bad = ctr_capable & (calc != data[:, 7])

    # payload byte entropy (8 samples -> max 3 bits)
    s = np.sort(data, axis=1)
    new = np.ones_like(s, dtype=bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    g = np.cumsum(new, axis=1) - 1
    key = (np.arange(n)[:, None] * 8 + g).ravel()
    counts = np.bincount(key, minlength=n * 8).reshape(n, 8) / 8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(counts > 0, counts * np.log2(counts), 0.0), axis=1)

    return dict(iat_id=iat_id, bitdiff=bitdiff, dup=dup,
                ctr_bad=ctr_bad, cks_bad=cks_bad, entropy=ent,
                ctr_capable=ctr_capable, unused=has_ctr)


def decode_columns(cid, data):
    """Vectorised DBC decode. NaN wherever the frame does not carry a signal."""
    d = data.astype(np.float64)
    nan = np.full(len(cid), np.nan)

    def where(mask, vals):
        out = nan.copy()
        out[mask] = vals[mask]
        return out

    u16_01 = d[:, 0] * 256 + d[:, 1]
    raw23 = d[:, 2] * 256 + d[:, 3]
    i23 = np.where(raw23 > 32767, raw23 - 65536, raw23)

    return dict(
        rpm=where(cid == 0x0C0, u16_01 * 0.25),
        thr=where(cid == 0x0D0, d[:, 0] / 2.55),
        plate=where(cid == 0x0D0, d[:, 1] / 2.55),
        brk=where(cid == 0x1A0, d[:, 0] / 2.55),
        brk_sw=where(cid == 0x1A0, np.mod(d[:, 1], 2)),
        rep_decel=where(cid == 0x1A0, i23 / 100.0),
        spd=where(cid == 0x1D0, u16_01 / 100.0),
        gear=where(cid == 0x3B0, d[:, 0]),
        blight=where(cid == 0x5A0, np.floor(np.mod(d[:, 0], 8) / 4)),
    )


# ------------------------------------------------------------------ helpers

def _slope(x, y):
    """Least-squares slope of y on x; 0 when it is not defined."""
    ok = ~np.isnan(y)
    if ok.sum() < 3:
        return 0.0
    x, y = x[ok], y[ok]
    vx = x.var()
    if vx < 1e-12:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / vx)


def _m(a):
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else 0.0


def _s(a):
    return float(np.nanstd(a)) if np.sum(~np.isnan(a)) > 1 else 0.0


def _expected_rpm(speed_kmh, gear):
    g = int(np.clip(gear, 1, len(C.GEAR_RATIOS)))
    v = speed_kmh / 3.6
    wheel_rps = v / (2 * np.pi * C.WHEEL_RADIUS_M)
    return max(C.IDLE_RPM, wheel_rps * C.FINAL_DRIVE * C.GEAR_RATIOS[g - 1] * 60.0)


# --------------------------------------------------------------- main entry

def build_windows(frames, window_ms=None, stride_ms=None):
    """Return (X, feature_names, y, attack_code, window_start_times)."""
    window_ms = window_ms or C.WINDOW_MS
    stride_ms = stride_ms or C.STRIDE_MS
    w = window_ms / 1000.0
    stride = stride_ms / 1000.0

    t, cid, data, label, atype = frames_to_arrays(frames)
    f = per_frame_flags(t, cid, data)
    sig = decode_columns(cid, data)

    idx = np.full(0x800, -1, dtype=np.int32)
    for c, i in ID_INDEX.items():
        idx[c] = i
    id_i = np.where(cid < 0x800, idx[np.clip(cid, 0, 0x7FF)], -1)
    known = id_i >= 0

    expected_per_window = np.array(
        [w / (C.BY_ID[c].cycle_ms / 1000.0) for c in C.KNOWN_IDS])
    cycle_s = np.array([C.BY_ID[c].cycle_ms / 1000.0 for c in C.KNOWN_IDS])

    names = (["n_frames", "msg_rate", "iat_mean", "iat_std", "iat_min", "iat_p95",
              "n_unique_ids", "frac_unknown_id", "n_unknown_frames"]
             + [f"cnt_ratio_{c:03X}" for c in C.KNOWN_IDS]
             + [f"iat_ratio_{c:03X}" for c in C.KNOWN_IDS]
             + [f"iat_cv_{c:03X}" for c in C.KNOWN_IDS]
             + ["ctr_anom_rate", "cksum_bad_rate", "dup_payload_rate",
                "bitdiff_mean", "bitdiff_std", "entropy_mean", "entropy_max",
                "thr_mean", "thr_std", "thr_max", "thr_jump_max",
                "rpm_mean", "rpm_slope", "rpm_jump_max",
                "spd_mean", "spd_slope", "spd_jump_max",
                "brk_mean", "brk_std", "brk_sw_rate", "rep_decel_mean",
                "gear_mean", "blight_rate",
                "mismatch_thr_accel", "mismatch_thr_rpm", "mismatch_brake_decel",
                "brake_light_mismatch", "decel_report_err", "both_pedals",
                "rpm_speed_err", "plate_pedal_err"])

    t0, t_end = t[0], t[-1]
    starts = np.arange(t0, t_end - w, stride)
    if len(starts) == 0:                    # buffer shorter than one window
        starts = np.array([t0])
    lo = np.searchsorted(t, starts)
    hi = np.searchsorted(t, starts + w)

    X = np.zeros((len(starts), len(names)), dtype=np.float32)
    y = np.zeros(len(starts), dtype=np.int8)
    acode = np.zeros(len(starts), dtype=np.int8)

    for k in range(len(starts)):
        a, b = lo[k], hi[k]
        if b - a < 8:
            y[k] = -1          # too little traffic to judge; dropped later
            continue
        tt = t[a:b]
        ii = id_i[a:b]
        kn = known[a:b]
        row = []

        # ---- global timing
        dt = np.diff(tt)
        row += [b - a, (b - a) / w,
                float(dt.mean()) if len(dt) else 0.0,
                float(dt.std()) if len(dt) > 1 else 0.0,
                float(dt.min()) if len(dt) else 0.0,
                float(np.percentile(dt, 95)) if len(dt) else 0.0,
                float(len(np.unique(cid[a:b]))),
                float((~kn).mean()), float((~kn).sum())]

        # ---- per-ID counts and regularity
        cnt = np.bincount(ii[kn], minlength=N_IDS).astype(float)
        row += list(cnt / expected_per_window)

        iat = f["iat_id"][a:b]
        ok = kn & ~np.isnan(iat)
        sm = np.bincount(ii[ok], weights=iat[ok], minlength=N_IDS)
        sq = np.bincount(ii[ok], weights=iat[ok] ** 2, minlength=N_IDS)
        nn = np.bincount(ii[ok], minlength=N_IDS).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_iat = np.where(nn > 0, sm / np.maximum(nn, 1), cycle_s)
            var = np.where(nn > 1, sq / np.maximum(nn, 1) - mean_iat ** 2, 0.0)
        std_iat = np.sqrt(np.maximum(var, 0.0))
        row += list(mean_iat / cycle_s)
        row += list(std_iat / cycle_s)

        # ---- integrity
        row += [float(f["ctr_bad"][a:b].mean()), float(f["cks_bad"][a:b].mean()),
                float(f["dup"][a:b].mean()),
                _m(f["bitdiff"][a:b]), _s(f["bitdiff"][a:b]),
                float(f["entropy"][a:b].mean()), float(f["entropy"][a:b].max())]

        # ---- physics
        thr, rpm, spd = sig["thr"][a:b], sig["rpm"][a:b], sig["spd"][a:b]
        brk, bsw = sig["brk"][a:b], sig["brk_sw"][a:b]
        rdec, gear, bl = sig["rep_decel"][a:b], sig["gear"][a:b], sig["blight"][a:b]
        plate = sig["plate"][a:b]

        def jump(v):
            vv = v[~np.isnan(v)]
            return float(np.abs(np.diff(vv)).max()) if len(vv) > 1 else 0.0

        thr_mean = _m(thr)
        thr_max = float(np.nanmax(thr)) if np.any(~np.isnan(thr)) else 0.0
        rpm_mean = _m(rpm)
        rpm_slope = _slope(tt, rpm)
        spd_mean = _m(spd)
        spd_slope = _slope(tt, spd)                      # km/h per second
        brk_mean = _m(brk)
        bsw_rate = _m(bsw)
        bl_rate = _m(bl)
        gear_mean = _m(gear) or 1.0
        rdec_mean = _m(rdec)

        row += [thr_mean, _s(thr), thr_max, jump(thr),
                rpm_mean, rpm_slope, jump(rpm),
                spd_mean, spd_slope, jump(spd),
                brk_mean, _s(brk), bsw_rate, rdec_mean,
                gear_mean, bl_rate]

        accel = spd_slope / 3.6                          # m/s^2
        row += [
            thr_mean / 100.0 - float(np.clip(accel / 3.5, 0, 1)),
            thr_mean / 100.0 - float(np.clip(rpm_slope / 2500.0, 0, 1)),
            brk_mean / 100.0 + float(np.clip(accel / 3.5, -1, 1)),
            bsw_rate - bl_rate,
            abs(rdec_mean - accel),
            min(thr_mean, brk_mean) / 100.0,
            abs(rpm_mean - _expected_rpm(spd_mean, gear_mean)) / 1000.0,
            _m(np.abs(plate - thr)),
        ]

        X[k] = row
        na = int((label[a:b] == 1).sum())
        y[k] = 1 if na >= C.MIN_ATTACK_FRAMES else 0
        if y[k]:
            # only label the family once the window actually counts as an
            # attack, so acode and y can never disagree downstream
            vals = atype[a:b][label[a:b] == 1]
            acode[k] = int(np.bincount(vals).argmax())

    keep = y >= 0
    X = np.nan_to_num(X[keep], nan=0.0, posinf=0.0, neginf=0.0)
    return X, names, y[keep], acode[keep], starts[keep]


def build_one(frames):
    """Score a single live buffer: returns (X_row, y_true, attack_code)."""
    span = (frames[-1][0] - frames[0][0]) * 1000.0 + 1.0
    X, _, y, acode, _ = build_windows(frames, window_ms=span, stride_ms=span)
    if len(X) == 0:
        return None, 0, 0
    return X[-1:], int(y[-1]), int(acode[-1])
