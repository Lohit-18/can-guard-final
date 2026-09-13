"""
CAN-Guard bus definition.

This file is the "DBC-lite" for our simulated vehicle: which ECU broadcasts
which message, how often, and whether the payload carries a rolling counter
and checksum (many real powertrain messages do).

Cycle times and ID priorities are modelled on a typical 500 kbit/s powertrain
CAN bus. Lower CAN ID == higher arbitration priority.
"""

from dataclasses import dataclass

# Physics/driver simulation tick, in milliseconds.
TICK_MS = 1.0


@dataclass(frozen=True)
class MsgSpec:
    can_id: int
    name: str
    cycle_ms: float      # nominal broadcast period
    dlc: int             # payload length in bytes
    jitter_pct: float    # sending ECU clock jitter (std-dev as fraction of cycle)
    counter: bool        # payload ends with rolling counter nibble + checksum byte


BUS = [
    MsgSpec(0x0C0, "ENGINE_RPM",    10.0, 8, 0.015, True),
    MsgSpec(0x0D0, "THROTTLE",      20.0, 8, 0.020, True),
    MsgSpec(0x1A0, "BRAKE",         20.0, 8, 0.020, True),
    MsgSpec(0x1D0, "WHEEL_SPEED",   10.0, 8, 0.015, True),
    MsgSpec(0x2C0, "STEERING",      20.0, 8, 0.030, False),
    MsgSpec(0x3B0, "GEAR",          50.0, 8, 0.030, False),
    MsgSpec(0x4B1, "ABS_ESP",       20.0, 8, 0.025, False),
    MsgSpec(0x5A0, "BODY_STATUS",  100.0, 8, 0.050, False),
    MsgSpec(0x7E8, "OBD2_RESP",    200.0, 8, 0.050, False),
]

BY_ID = {m.can_id: m for m in BUS}
KNOWN_IDS = sorted(BY_ID)

# Attack labels. 0 is always "normal".
ATTACK_TYPES = {
    0: "normal",
    1: "throttle_spoof",     # unauthorized acceleration commands
    2: "fake_brake",         # forged brake-pressure frames
    3: "dos_flood",          # 0x000 flood, starves the bus
    4: "fuzzing",            # random IDs / random payloads
    5: "replay",             # valid frames replayed out of context
}
NAME_TO_CODE = {v: k for k, v in ATTACK_TYPES.items()}

# --- Windowing -------------------------------------------------------------
# The IDS does not judge single frames; it judges a short slice of bus traffic.
# Time-based rather than frame-count-based, so a flood cannot shrink the
# physical time a window covers.
WINDOW_MS = 250.0
STRIDE_MS = 125.0
MIN_ATTACK_FRAMES = 2   # frames needed before a window counts as "attack"

# --- Vehicle constants (used by the physics model and by feature checks) ---
VEHICLE_MASS_KG = 1500.0
WHEEL_RADIUS_M = 0.32
FINAL_DRIVE = 3.9
GEAR_RATIOS = [3.5, 2.1, 1.4, 1.0, 0.8, 0.65]
IDLE_RPM = 750.0
MAX_RPM = 6500.0
MAX_ENGINE_FORCE_N = 4200.0
MAX_BRAKE_FORCE_N = 9000.0
