r"""Euler–Maruyama integration of overdamped Langevin dynamics.

The equation of motion is

.. math::

    \zeta\,\dot x = F(x) + \sqrt{2 \zeta k_B T}\,\eta(t),

with friction coefficient :math:`\zeta` (``friction``; equal to
:math:`m\gamma` in the notation of the original notebook) and
:math:`\langle \eta(t)\eta(t')\rangle = \delta(t-t')`. Discretised:

.. math::

    x_{j+1} = x_j + \frac{\Delta t}{\zeta} F(x_j) + \sqrt{2 D \Delta t}\,\xi_j,
    \qquad \xi_j \sim \mathcal N(0, 1),\quad D = k_B T / \zeta.

All routines are vectorised over many independent walkers, which is far
cheaper in NumPy than one long trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .potentials import Potential

_NOISE_CHUNK = 2048  # steps of Gaussian noise drawn per RNG call


@dataclass
class Trajectory:
    """Positions saved every ``stride`` steps.

    Attributes
    ----------
    time:
        Shape ``(n_frames,)``.
    x:
        Shape ``(n_frames, n_walkers)``.
    """

    time: NDArray[np.float64]
    x: NDArray[np.float64]
    dt: float
    stride: int
    kT: float
    friction: float
    params: dict = field(default_factory=dict)

    @property
    def n_walkers(self) -> int:
        return self.x.shape[1]

    @property
    def save_interval(self) -> float:
        return self.dt * self.stride

    def save(self, path: str) -> None:
        """Write to a compressed ``.npz`` file."""
        np.savez_compressed(
            path,
            time=self.time,
            x=self.x,
            dt=self.dt,
            stride=self.stride,
            kT=self.kT,
            friction=self.friction,
            **{f"param_{k}": v for k, v in self.params.items()},
        )

    @classmethod
    def load(cls, path: str) -> Trajectory:
        with np.load(path) as f:
            params = {k[len("param_"):]: f[k].item() for k in f.files if k.startswith("param_")}
            return cls(
                time=f["time"],
                x=f["x"],
                dt=float(f["dt"]),
                stride=int(f["stride"]),
                kT=float(f["kT"]),
                friction=float(f["friction"]),
                params=params,
            )


def _check(dt: float, kT: float, friction: float) -> None:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if kT < 0:
        raise ValueError("kT must be non-negative")
    if friction <= 0:
        raise ValueError("friction must be positive")


def simulate(
    potential: Potential,
    x0: ArrayLike,
    n_steps: int,
    dt: float = 1e-3,
    kT: float = 1.0,
    friction: float = 1.0,
    n_walkers: int | None = None,
    stride: int = 1,
    seed: int | np.random.Generator | None = None,
) -> Trajectory:
    """Integrate overdamped Langevin dynamics for ``n_steps`` steps.

    Parameters
    ----------
    potential:
        Object with a vectorised ``force(x)`` method.
    x0:
        Initial position(s). A scalar is broadcast to ``n_walkers`` walkers.
    n_steps:
        Number of integration steps (the initial frame is stored in addition).
    dt, kT, friction:
        Time step, thermal energy and friction coefficient :math:`\\zeta`.
    n_walkers:
        Number of independent walkers. Defaults to ``len(x0)`` or 1.
    stride:
        Store every ``stride``-th step.
    seed:
        Seed or :class:`numpy.random.Generator` for reproducibility.
    """
    _check(dt, kT, friction)
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    rng = np.random.default_rng(seed)
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    if n_walkers is None:
        n_walkers = x0.size
    x = np.broadcast_to(x0, (n_walkers,)).copy()

    n_frames = n_steps // stride + 1
    out = np.empty((n_frames, n_walkers))
    out[0] = x

    mobility_dt = dt / friction
    sigma = np.sqrt(2.0 * kT * dt / friction)

    if n_walkers == 1 and hasattr(potential, "scalar_force"):
        _simulate_scalar(potential.scalar_force(), float(x[0]), n_steps, stride, mobility_dt, sigma, rng, out[:, 0])
    else:
        _simulate_vector(potential, x, n_steps, stride, mobility_dt, sigma, rng, out)

    time = np.arange(n_frames) * dt * stride
    params = {k: v for k, v in vars(potential).items() if np.isscalar(v)} if hasattr(potential, "__dict__") else {}
    params["potential"] = type(potential).__name__
    return Trajectory(time=time, x=out, dt=dt, stride=stride, kT=kT, friction=friction, params=params)


def _simulate_scalar(force, x, n_steps, stride, mobility_dt, sigma, rng, out) -> None:
    """Single walker with plain floats: ~10x faster than NumPy for one walker.

    Draws the same random numbers in the same order as the vector path.
    """
    frame = 1
    step = 0
    while step < n_steps:
        chunk = min(_NOISE_CHUNK, n_steps - step)
        noise = (rng.standard_normal((chunk, 1))[:, 0] * sigma).tolist()
        for xi in noise:
            x += mobility_dt * force(x) + xi
            step += 1
            if step % stride == 0:
                out[frame] = x
                frame += 1


def _simulate_vector(potential, x, n_steps, stride, mobility_dt, sigma, rng, out) -> None:
    n_walkers = x.size
    frame = 1
    step = 0
    while step < n_steps:
        chunk = min(_NOISE_CHUNK, n_steps - step)
        noise = rng.standard_normal((chunk, n_walkers))
        noise *= sigma
        for k in range(chunk):
            x += mobility_dt * potential.force(x) + noise[k]
            step += 1
            if step % stride == 0:
                out[frame] = x
                frame += 1


def first_passage_times(
    potential: Potential,
    x0: float,
    target: float,
    n_walkers: int = 1000,
    dt: float = 1e-3,
    kT: float = 1.0,
    friction: float = 1.0,
    max_time: float = np.inf,
    seed: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """First time each walker started at ``x0`` reaches ``target``.

    Walkers are removed from the integration as soon as they cross, so the
    cost is dominated by the slowest walkers. Walkers that have not crossed by
    ``max_time`` are reported as ``nan``.
    """
    _check(dt, kT, friction)
    if x0 == target:
        return np.zeros(n_walkers)

    rng = np.random.default_rng(seed)
    direction = 1.0 if target > x0 else -1.0

    x = np.full(n_walkers, float(x0))
    idx = np.arange(n_walkers)
    fpt = np.full(n_walkers, np.nan)

    mobility_dt = dt / friction
    sigma = np.sqrt(2.0 * kT * dt / friction)
    max_steps = np.inf if np.isinf(max_time) else int(np.ceil(max_time / dt))

    step = 0
    while x.size and step < max_steps:
        chunk = int(min(_NOISE_CHUNK, max_steps - step))
        noise = rng.standard_normal((chunk, x.size)) * sigma
        for k in range(chunk):
            x += mobility_dt * potential.force(x) + noise[k]
            step += 1
            crossed = direction * (x - target) >= 0
            if crossed.any():
                fpt[idx[crossed]] = step * dt
                keep = ~crossed
                x, idx = x[keep], idx[keep]
                noise = noise[:, keep]
                if not x.size:
                    break
    return fpt
