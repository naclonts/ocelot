"""
Motion pattern classes for scenario generator.

All patterns share the interface:
    reset(x0, y0, z0)  — called at episode start
    step(t) -> (x, y, z)  — absolute world position at time t seconds

World bounds enforced by all patterns:
    x: unchanged (constant depth)
    y: [-1.0, 1.0]
    z: [ 0.3, 1.7]
"""

import math
import random


class MotionPattern:
    def reset(self, x0: float, y0: float, z0: float) -> None:
        raise NotImplementedError

    def step(self, t: float) -> tuple:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StaticMotion
# ---------------------------------------------------------------------------

class StaticMotion(MotionPattern):
    """Face stays at its initial position throughout the episode."""

    def __init__(self):
        self._pos = (0.0, 0.0, 0.0)

    def reset(self, x0: float, y0: float, z0: float) -> None:
        self._pos = (x0, y0, z0)

    def step(self, t: float) -> tuple:
        return self._pos


# ---------------------------------------------------------------------------
# SinusoidalMotion
# ---------------------------------------------------------------------------

class SinusoidalMotion(MotionPattern):
    """
    Sinusoidal oscillation in y and z with incommensurate periods.

    amp_y   = speed * period_y / (2π)
    period_z = period_y * 1.7   (incommensurate → no exact path repeat)
    amp_z   = amp_y * 0.4       (tilt swing smaller than pan swing)
    """

    def __init__(self, speed: float, period_y: float):
        self.speed = speed
        self.period_y = period_y
        self.amp_y = speed * period_y / (2.0 * math.pi)
        self.period_z = period_y * 1.7
        self.amp_z = self.amp_y * 0.4
        self._x0 = self._y0 = self._z0 = 0.0

    def reset(self, x0: float, y0: float, z0: float) -> None:
        self._x0, self._y0, self._z0 = x0, y0, z0

    def step(self, t: float) -> tuple:
        y = self._y0 + self.amp_y * math.sin(2.0 * math.pi * t / self.period_y)
        z = self._z0 + self.amp_z * math.sin(2.0 * math.pi * t / self.period_z)
        y = max(-1.0, min(1.0, y))
        z = max(0.3, min(1.7, z))
        return (self._x0, y, z)


# ---------------------------------------------------------------------------
# LinearDriftMotion
# ---------------------------------------------------------------------------

def _reflect_1d(x0: float, v: float, t: float, lo: float, hi: float) -> float:
    """
    Analytically compute bouncing position: start at x0 with velocity v,
    reflect off walls at lo and hi after elapsed time t.
    """
    span = hi - lo
    if span <= 0:
        return lo
    # Displacement from lo
    pos_rel = x0 - lo
    raw = pos_rel + v * t
    period = 2.0 * span
    # Fold into [0, 2*span)
    raw_mod = raw % period
    if raw_mod < 0:
        raw_mod += period
    if raw_mod <= span:
        return lo + raw_mod
    else:
        return hi - (raw_mod - span)


class LinearDriftMotion(MotionPattern):
    """
    Constant-velocity drift with wall reflection.

    vy and vz are sampled as speed * random_sign at construction.
    Position is computed analytically so step(t) is consistent regardless of
    call order.
    """

    def __init__(self, vy: float, vz: float):
        self.vy = vy
        self.vz = vz
        self._x0 = self._y0 = self._z0 = 0.0

    def reset(self, x0: float, y0: float, z0: float) -> None:
        self._x0, self._y0, self._z0 = x0, y0, z0

    def step(self, t: float) -> tuple:
        y = _reflect_1d(self._y0, self.vy, t, -1.0, 1.0)
        z = _reflect_1d(self._z0, self.vz, t,  0.3, 1.7)
        return (self._x0, y, z)


# ---------------------------------------------------------------------------
# RandomWalkMotion
# ---------------------------------------------------------------------------

class RandomWalkMotion(MotionPattern):
    """
    Ornstein-Uhlenbeck random walk on velocity.

    dv = -v/τ * dt + speed * sqrt(2/τ * dt) * N(0,1)

    The steady-state velocity standard deviation ≈ speed, so the face
    meanders at roughly `speed` m/s on average.

    Seeded for reproducibility: same (seed, reset position) always gives the
    same trajectory.  step(t) integrates forward from the last call, so calls
    must be made with non-decreasing t within an episode.
    """

    DT = 0.05  # internal integration step (20 Hz)

    def __init__(self, speed: float, tau: float = 3.0, seed: int = 0):
        self.speed = speed
        self.tau = tau
        self._seed = seed
        self._rng: random.Random = random.Random(seed)
        self._x0 = self._y = self._z = 0.0
        self._vy = self._vz = 0.0
        self._t_integrated = 0.0

    def reset(self, x0: float, y0: float, z0: float) -> None:
        self._x0 = x0
        self._y = y0
        self._z = z0
        self._vy = 0.0
        self._vz = 0.0
        self._rng = random.Random(self._seed)
        self._t_integrated = 0.0

    def step(self, t: float) -> tuple:
        while self._t_integrated < t - 1e-9:
            dt = min(self.DT, t - self._t_integrated)
            noise = self.speed * math.sqrt(2.0 / self.tau * dt)
            self._vy += (-self._vy / self.tau) * dt + noise * self._rng.gauss(0.0, 1.0)
            self._vz += (-self._vz / self.tau) * dt + noise * self._rng.gauss(0.0, 1.0)
            # Clamp velocity magnitude to 3× speed to prevent runaway
            cap = self.speed * 3.0
            self._vy = max(-cap, min(cap, self._vy))
            self._vz = max(-cap, min(cap, self._vz))

            ny = self._y + self._vy * dt
            nz = self._z + self._vz * dt

            # Reflect at world boundaries
            if ny > 1.0:
                ny = 2.0 - ny
                self._vy = -abs(self._vy)
            elif ny < -1.0:
                ny = -2.0 - ny
                self._vy = abs(self._vy)

            if nz > 1.7:
                nz = 3.4 - nz
                self._vz = -abs(self._vz)
            elif nz < 0.3:
                nz = 0.6 - nz
                self._vz = abs(self._vz)

            self._y = max(-1.0, min(1.0, ny))
            self._z = max(0.3, min(1.7, nz))
            self._t_integrated += dt

        return (self._x0, self._y, self._z)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_motion(motion_type: str, speed: float, period: float,
                rng: random.Random) -> MotionPattern:
    """
    Construct the appropriate MotionPattern from a ScenarioConfig face entry.

    Args:
        motion_type: one of "static", "linear_drift", "sinusoidal", "random_walk"
        speed:       peak velocity in m/s
        period:      sinusoidal period in seconds (ignored for other types)
        rng:         seeded RNG used to draw direction/seed for the pattern
    """
    if motion_type == "static":
        return StaticMotion()
    elif motion_type == "sinusoidal":
        return SinusoidalMotion(speed=speed, period_y=period)
    elif motion_type == "linear_drift":
        sign_y = rng.choice([-1.0, 1.0])
        sign_z = rng.choice([-1.0, 1.0])
        return LinearDriftMotion(vy=speed * sign_y, vz=speed * 0.3 * sign_z)
    elif motion_type == "random_walk":
        return RandomWalkMotion(speed=speed, seed=rng.randint(0, 2**31))
    else:
        raise ValueError(f"Unknown motion type: {motion_type!r}")
