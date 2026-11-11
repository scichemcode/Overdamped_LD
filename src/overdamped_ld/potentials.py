"""One-dimensional potential energy surfaces.

Any object exposing ``energy(x)``, ``force(x)`` and ``curvature(x)`` (all
vectorised over NumPy arrays) can be used with the integrators and analysis
tools in this package. :class:`DoubleWell` is the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class Potential(Protocol):
    """Interface expected by :func:`overdamped_ld.simulate` and friends."""

    def energy(self, x: ArrayLike) -> NDArray[np.float64]: ...

    def force(self, x: ArrayLike) -> NDArray[np.float64]: ...

    def curvature(self, x: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class DoubleWell:
    r"""Symmetric quartic double well.

    .. math::

        U(x) = E_b \left[ \left(\frac{x}{d}\right)^2 - 1 \right]^2

    Minima sit at :math:`x = \pm d` with :math:`U = 0`; the barrier top is at
    :math:`x = 0` with :math:`U = E_b`.

    Parameters
    ----------
    barrier:
        Barrier height :math:`E_b` in absolute energy units (the same units as
        ``kT`` passed to the integrator).
    d:
        Half the distance between the two minima.
    """

    barrier: float = 1.0
    d: float = 1.0

    def __post_init__(self) -> None:
        if self.barrier < 0:
            raise ValueError("barrier must be non-negative")
        if self.d <= 0:
            raise ValueError("d must be positive")

    @classmethod
    def in_kT(cls, barrier_kT: float, kT: float = 1.0, d: float = 1.0) -> DoubleWell:
        """Build a double well whose barrier is ``barrier_kT`` multiples of ``kT``."""
        return cls(barrier=barrier_kT * kT, d=d)

    def energy(self, x: ArrayLike) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=float)
        return self.barrier * ((x / self.d) ** 2 - 1.0) ** 2

    def force(self, x: ArrayLike) -> NDArray[np.float64]:
        """:math:`F(x) = -\\mathrm{d}U/\\mathrm{d}x`."""
        x = np.asarray(x, dtype=float)
        return -4.0 * self.barrier * x * ((x / self.d) ** 2 - 1.0) / self.d**2

    def curvature(self, x: ArrayLike) -> NDArray[np.float64]:
        """:math:`U''(x)`."""
        x = np.asarray(x, dtype=float)
        return self.barrier * (12.0 * x**2 / self.d**4 - 4.0 / self.d**2)

    def scalar_force(self):
        """Return a plain-float ``F(x)`` closure.

        :func:`overdamped_ld.simulate` uses it for single-walker runs, where
        per-step NumPy overhead would dominate. Optional for other potentials.
        """
        a = -4.0 * self.barrier / self.d**2
        inv_d2 = 1.0 / self.d**2

        def force(x: float) -> float:
            return a * x * (x * x * inv_d2 - 1.0)

        return force

    @property
    def minima(self) -> tuple[float, float]:
        return (-self.d, self.d)

    @property
    def saddle(self) -> float:
        return 0.0
