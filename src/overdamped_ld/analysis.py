r"""Trajectory analysis and analytical reference results.

Analytical results let every simulation be checked against theory:

* :func:`boltzmann_density` — the stationary distribution
  :math:`p(x) \propto e^{-U(x)/k_BT}`.
* :func:`mfpt_exact` — the exact 1D mean first-passage time (quadrature).
* :func:`kramers_rate` — the high-barrier (Kramers/Smoluchowski) asymptote.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .potentials import Potential


def _cumtrapz(y: NDArray, x: NDArray) -> NDArray:
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


# --------------------------------------------------------------------------
# Analytical references
# --------------------------------------------------------------------------


def boltzmann_density(potential: Potential, x: ArrayLike, kT: float = 1.0) -> NDArray[np.float64]:
    """Normalised equilibrium density on the grid ``x`` (must be sorted)."""
    x = np.asarray(x, dtype=float)
    u = potential.energy(x)
    w = np.exp(-(u - u.min()) / kT)
    return w / np.trapezoid(w, x)


def mfpt_exact(
    potential: Potential,
    start: float,
    target: float,
    kT: float = 1.0,
    friction: float = 1.0,
    boundary: float | None = None,
    n_grid: int = 20_001,
) -> float:
    r"""Exact mean first-passage time from ``start`` to ``target`` in 1D.

    With a reflecting wall at ``boundary`` (behind ``start``) and
    :math:`D = k_BT/\zeta`,

    .. math::

        \tau = \frac{1}{D}\int_{\text{start}}^{\text{target}} \mathrm{d}y\;
               e^{\beta U(y)} \int_{\text{boundary}}^{y} \mathrm{d}z\; e^{-\beta U(z)} .

    For a confining potential the default ``boundary`` (far behind ``start``)
    reproduces a natural boundary at :math:`\mp\infty`.
    """
    if start == target:
        return 0.0
    sign = 1.0 if target > start else -1.0
    if boundary is None:
        boundary = start - sign * max(5.0 * abs(target - start), 5.0)

    beta = 1.0 / kT
    D = kT / friction
    y = np.linspace(boundary, target, n_grid)  # boundary -> target (may be decreasing)
    u = potential.energy(y)
    u0 = u.min()
    inner = np.abs(_cumtrapz(np.exp(-beta * (u - u0)), y))

    # outer integral only over [start, target]
    mask = sign * (y - start) >= 0
    integrand = np.exp(beta * (u[mask] - u0)) * inner[mask]
    return float(np.abs(np.trapezoid(integrand, y[mask])) / D)


def kramers_rate(
    potential: Potential,
    kT: float = 1.0,
    friction: float = 1.0,
    minimum: float | None = None,
    saddle: float | None = None,
) -> float:
    r"""Overdamped Kramers escape rate out of the well at ``minimum``.

    .. math::

        k = \frac{\sqrt{U''(x_{\min})\,|U''(x_{\ddagger})|}}{2\pi\zeta}\,
            e^{-\Delta U/k_BT}

    Valid when :math:`\Delta U \gg k_BT`.
    """
    if minimum is None:
        minimum = potential.minima[0]  # type: ignore[attr-defined]
    if saddle is None:
        saddle = potential.saddle  # type: ignore[attr-defined]
    w_min = float(potential.curvature(minimum))
    w_sad = abs(float(potential.curvature(saddle)))
    dU = float(potential.energy(saddle) - potential.energy(minimum))
    return np.sqrt(w_min * w_sad) / (2.0 * np.pi * friction) * np.exp(-dU / kT)


# --------------------------------------------------------------------------
# Trajectory analysis
# --------------------------------------------------------------------------


def assign_states(x: ArrayLike, core: float = 0.5) -> NDArray[np.int8]:
    """Core-set (hysteresis) state assignment.

    A frame is in state ``+1`` once the walker has entered ``x >= core`` and
    stays there until it enters ``x <= -core`` (state ``-1``), and vice versa.
    Frames before the first core visit are ``0``. This removes the spurious
    "transitions" that a single threshold at the barrier top produces when
    the walker jitters around it.

    ``x`` may be 1D ``(n_frames,)`` or 2D ``(n_frames, n_walkers)``.
    """
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    raw = np.zeros(x.shape, dtype=np.int8)
    raw[x >= core] = 1
    raw[x <= -core] = -1
    frames = np.arange(x.shape[0])[:, None]
    last = np.where(raw != 0, frames, 0)
    np.maximum.accumulate(last, axis=0, out=last)
    states = np.take_along_axis(raw, last, axis=0)
    return states[:, 0] if squeeze else states


def count_transitions(states: ArrayLike) -> NDArray[np.int64] | int:
    """Number of ``-1 <-> +1`` switches along axis 0 (per walker if 2D)."""
    s = np.asarray(states)
    flips = (s[1:] != s[:-1]) & (s[:-1] != 0)
    return flips.sum(axis=0)


def transition_rate(states: ArrayLike, save_interval: float) -> float:
    """Pooled switching rate: total transitions / total assigned time.

    For a symmetric double well this estimates the one-way escape rate ``k``.
    """
    s = np.asarray(states)
    n = np.sum(count_transitions(s))
    t = np.count_nonzero(s[1:] != 0) * save_interval
    return float(n / t) if t > 0 else np.nan


def dwell_times(states: ArrayLike, save_interval: float) -> dict[int, NDArray[np.float64]]:
    """Durations of completed visits to each state.

    The first and last visit of every walker are censored (their true
    duration is unknown) and are discarded.
    """
    s = np.asarray(states)
    if s.ndim == 1:
        s = s[:, None]
    out: dict[int, list[float]] = {-1: [], 1: []}
    for col in s.T:
        col = col[col != 0]
        if col.size == 0:
            continue
        edges = np.flatnonzero(np.diff(col)) + 1
        bounds = np.concatenate(([0], edges, [col.size]))
        for i in range(1, len(bounds) - 2):  # skip censored first/last
            out[int(col[bounds[i]])].append((bounds[i + 1] - bounds[i]) * save_interval)
    return {k: np.asarray(v) for k, v in out.items()}


def populations(states: ArrayLike) -> dict[int, float]:
    """Fraction of assigned frames spent in each state."""
    s = np.asarray(states)
    assigned = s[s != 0]
    if assigned.size == 0:
        return {-1: np.nan, 1: np.nan}
    return {-1: float(np.mean(assigned == -1)), 1: float(np.mean(assigned == 1))}
