"""Publication-style figures. Requires ``matplotlib``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from .analysis import boltzmann_density
from .potentials import DoubleWell

# Ordinal blue ramp (barrier heights are ordered -> one hue, light to dark).
_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
_CMAP = LinearSegmentedColormap.from_list("barrier", _RAMP)
# Fixed categorical slots for identity (simulation / exact / Kramers).
SERIES = {"sim": "#2a78d6", "exact": "#eb6834", "kramers": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "lines.linewidth": 2.0,
    "font.size": 11,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def use_style() -> None:
    """Apply the package's matplotlib style globally."""
    plt.rcParams.update(STYLE)


def barrier_colors(n: int) -> list:
    """``n`` ordered colours from the barrier ramp."""
    return [_CMAP(v) for v in np.linspace(0.0, 1.0, n)] if n > 1 else [_CMAP(0.6)]


def _ax(ax: Axes | None, figsize=(6.5, 4.5)) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_potentials(
    barriers_kT: Sequence[float], kT: float = 1.0, d: float = 1.0, ax: Axes | None = None
) -> Axes:
    """Double-well profiles for several barrier heights, energy in units of kT."""
    ax = _ax(ax)
    x = np.linspace(-1.8 * d, 1.8 * d, 600)
    for eb, c in zip(barriers_kT, barrier_colors(len(barriers_kT))):
        u = DoubleWell.in_kT(eb, kT, d).energy(x) / kT
        ax.plot(x, u, color=c, label=f"{eb:g} $k_BT$")
    top = max(barriers_kT)
    ax.set_ylim(0, 1.35 * top)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("$U(x)$  [$k_BT$]")
    ax.set_title("Double-well potential")
    ax.legend(title="Barrier", title_fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return ax


def plot_trajectories(
    trajectories: Mapping[float, tuple[np.ndarray, np.ndarray]], d: float = 1.0
) -> plt.Figure:
    """Small multiples: one time series per barrier height.

    ``trajectories`` maps barrier (kT) -> ``(time, x)`` for a single walker.
    """
    n = len(trajectories)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.5 * n + 0.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (eb, (t, x)), c in zip(axes, trajectories.items(), barrier_colors(n)):
        ax.plot(t, x, color=c, lw=0.8)
        ax.axhline(d, color=MUTED, lw=0.8, ls="--")
        ax.axhline(-d, color=MUTED, lw=0.8, ls="--")
        ax.text(1.01, 0.5, f"{eb:g} $k_BT$", transform=ax.transAxes, va="center", color=INK2)
        ax.set_ylim(-2, 2)
        ax.set_yticks([-d, 0, d])
    axes[0].set_title("Trajectories: higher barriers mean rarer hops")
    axes[-1].set_xlabel("Time")
    fig.supylabel("Position $x$", color=INK2, fontsize=11)
    fig.tight_layout()
    return fig


def plot_distributions(
    samples: Mapping[float, np.ndarray], kT: float = 1.0, d: float = 1.0
) -> plt.Figure:
    """Histogram of sampled positions vs the exact Boltzmann density."""
    n = len(samples)
    ncols = n if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), sharex=True, squeeze=False)
    grid = np.linspace(-2 * d, 2 * d, 600)
    bins = np.linspace(-2 * d, 2 * d, 61)
    for ax, (eb, xs), c in zip(axes.flat, samples.items(), barrier_colors(n)):
        ax.hist(np.ravel(xs), bins=bins, density=True, color=c, alpha=0.55, edgecolor="#fcfcfb", lw=0.5,
                label="Simulation")
        ax.plot(grid, boltzmann_density(DoubleWell.in_kT(eb, kT, d), grid, kT), color=INK, lw=1.5,
                label=r"$e^{-U/k_BT}/Z$")
        ax.set_title(f"{eb:g} $k_BT$", fontsize=11)
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper center", fontsize=9)
    fig.supxlabel("Position $x$", color=INK2, fontsize=11)
    fig.suptitle("Sampled distribution vs Boltzmann", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_rates(
    barriers_kT: Sequence[float],
    tau_sim: Sequence[float],
    tau_sim_err: Sequence[float],
    tau_exact: Sequence[float],
    tau_kramers: Sequence[float],
    ax: Axes | None = None,
) -> Axes:
    """Mean first-passage time vs barrier: simulation, exact quadrature, Kramers."""
    ax = _ax(ax)
    eb = np.asarray(barriers_kT, dtype=float)
    ax.plot(eb, tau_exact, color=SERIES["exact"], label="Exact MFPT (quadrature)")
    ax.plot(eb, tau_kramers, color=SERIES["kramers"], ls="--", label="Kramers  $1/k$")
    ax.errorbar(eb, tau_sim, yerr=tau_sim_err, fmt="o", ms=7, color=SERIES["sim"],
                mec="#fcfcfb", mew=1.5, capsize=0, elinewidth=1.5, label="Langevin simulation", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Barrier height  $E_b$  [$k_BT$]")
    ax.set_ylabel(r"Mean first-passage time  $\tau_{-1\to+1}$")
    ax.set_title("Arrhenius slowdown: simulation vs theory")
    ax.legend(loc="upper left")
    return ax
