"""
A small longitudinal vehicle model plus a driver.

Why bother with physics? Because the whole point of CAN-Guard is that a
spoofed frame is not just "statistically odd" - it is *physically impossible*.
A throttle message claiming 80% pedal while the engine RPM is flat and the
car is not accelerating is a lie that physics exposes. To learn that, the
model needs traffic where the signals genuinely agree with each other.
"""

import math

import numpy as np

from . import config as C


class Vehicle:
    """Longitudinal dynamics: pedal in, speed / RPM / gear out."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.speed = 0.0          # m/s
        self.rpm = C.IDLE_RPM
        self.gear = 1
        self.accel = 0.0          # m/s^2
        self.yaw_rate = 0.0       # deg/s
        self.steering = 0.0       # deg
        self.odo = 0.0

    def _gear_for_speed(self, v_kmh):
        for g, upper in enumerate([25, 45, 70, 100, 135], start=1):
            if v_kmh < upper:
                return g
        return 6

    def step(self, throttle_pct, brake_pct, steer_deg, dt):
        """Advance one tick. throttle/brake in percent, dt in seconds."""
        v = max(self.speed, 0.0)

        drive = ((throttle_pct / 100.0) * C.MAX_ENGINE_FORCE_N
                 / max(self.gear, 1) ** 0.45)
        brake = (brake_pct / 100.0) * C.MAX_BRAKE_FORCE_N
        drag = 0.42 * v * v + 180.0          # aero + rolling resistance
        engine_brake = 90.0 if throttle_pct < 2 else 0.0

        net = drive - brake - drag - engine_brake
        self.accel = net / C.VEHICLE_MASS_KG
        self.speed = max(0.0, v + self.accel * dt)
        self.odo += self.speed * dt

        v_kmh = self.speed * 3.6
        self.gear = self._gear_for_speed(v_kmh)

        # RPM follows wheel speed through the driveline, floored at idle.
        wheel_rps = self.speed / (2 * math.pi * C.WHEEL_RADIUS_M)
        driven = wheel_rps * C.FINAL_DRIVE * C.GEAR_RATIOS[self.gear - 1] * 60.0
        target_rpm = max(C.IDLE_RPM + throttle_pct * 6.0, driven)
        # engine inertia: RPM cannot jump instantly
        self.rpm += (min(target_rpm, C.MAX_RPM) - self.rpm) * min(1.0, dt * 9.0)

        self.steering = steer_deg
        self.yaw_rate = (steer_deg / 14.0) * (self.speed / max(1.0, 1.0)) * 0.06


class Driver:
    """
    Follows a randomly generated speed profile with a simple PI controller.

    The profile is built from urban-style phases so the traffic contains
    stops, pull-aways, cruising and hard braking - all of which the IDS has
    to accept as normal.
    """

    def __init__(self, rng: np.random.Generator, duration_s: float):
        self.rng = rng
        self.t = 0.0
        self.integral = 0.0
        self.profile = self._build_profile(duration_s)
        self.steer_phase = rng.uniform(0, 6.28)

    def _build_profile(self, duration_s):
        """Returns (times, target_speeds_kmh) knots for linear interpolation."""
        times, speeds = [0.0], [0.0]
        t = 0.0
        while t < duration_s:
            phase = self.rng.choice(
                ["idle", "accel", "cruise", "decel", "stop"],
                p=[0.10, 0.25, 0.35, 0.20, 0.10],
            )
            if phase == "idle":
                dur, target = self.rng.uniform(3, 8), 0.0
            elif phase == "accel":
                dur, target = self.rng.uniform(6, 14), self.rng.uniform(35, 110)
            elif phase == "cruise":
                dur = self.rng.uniform(10, 30)
                target = max(speeds[-1], self.rng.uniform(30, 95))
            elif phase == "decel":
                dur = self.rng.uniform(5, 12)
                target = max(0.0, speeds[-1] * self.rng.uniform(0.2, 0.7))
            else:
                dur, target = self.rng.uniform(4, 10), 0.0
            t += dur
            times.append(t)
            speeds.append(float(target))
        return np.array(times), np.array(speeds)

    def target_kmh(self, t):
        return float(np.interp(t, self.profile[0], self.profile[1]))

    def control(self, vehicle: Vehicle, t, dt):
        """Return (throttle_pct, brake_pct, steer_deg)."""
        err = self.target_kmh(t) - vehicle.speed * 3.6
        self.integral = float(np.clip(self.integral + err * dt, -60, 60))
        u = 1.9 * err + 0.25 * self.integral

        if u >= 0:
            throttle, brake = min(100.0, u), 0.0
        else:
            throttle, brake = 0.0, min(100.0, -u * 1.4)

        # Humans are not perfectly smooth.
        throttle = float(np.clip(throttle + self.rng.normal(0, 0.6), 0, 100))
        brake = float(np.clip(brake + self.rng.normal(0, 0.4), 0, 100))
        steer = 9.0 * math.sin(0.22 * t + self.steer_phase) + self.rng.normal(0, 0.7)
        return throttle, brake, steer
