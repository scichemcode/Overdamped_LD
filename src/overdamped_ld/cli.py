"""Command-line interface: ``overdamped-ld {simulate,rates,figures}``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import assign_states, count_transitions, kramers_rate, mfpt_exact, populations, transition_rate
from .integrators import first_passage_times, simulate
from .potentials import DoubleWell


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kT", type=float, default=1.0, help="thermal energy (default: 1)")
    p.add_argument("--friction", type=float, default=1.0, help="friction coefficient zeta = m*gamma (default: 1)")
    p.add_argument("--d", type=float, default=1.0, help="half-distance between minima (default: 1)")
    p.add_argument("--dt", type=float, default=1e-3, help="time step (default: 1e-3)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")


def cmd_simulate(a: argparse.Namespace) -> int:
    pot = DoubleWell.in_kT(a.barrier, a.kT, a.d)
    t0 = time.perf_counter()
    traj = simulate(pot, a.x0, a.steps, a.dt, a.kT, a.friction, a.walkers, a.stride, a.seed)
    elapsed = time.perf_counter() - t0
    traj.params["barrier_kT"] = a.barrier
    states = assign_states(traj.x, core=a.core * a.d)
    k_sim = transition_rate(states, traj.save_interval)
    summary = {
        "barrier_kT": a.barrier,
        "walkers": traj.n_walkers,
        "simulated_time_per_walker": float(traj.time[-1]),
        "transitions": int(np.sum(count_transitions(states))),
        "rate_simulation": k_sim,
        "rate_kramers": kramers_rate(pot, a.kT, a.friction),
        "rate_exact_1_over_mfpt": 1.0 / mfpt_exact(pot, -a.d, a.d, a.kT, a.friction),
        "populations": populations(states),
        "wall_time_s": round(elapsed, 2),
    }
    if a.out:
        traj.save(a.out)
        summary["saved"] = str(a.out)
    print(json.dumps(summary, indent=2))
    return 0


def _rates(a: argparse.Namespace) -> dict:
    rows = []
    for i, eb in enumerate(a.barriers):
        pot = DoubleWell.in_kT(eb, a.kT, a.d)
        fpt = first_passage_times(pot, -a.d, a.d, a.walkers, a.dt, a.kT, a.friction, seed=a.seed + i)
        fpt = fpt[np.isfinite(fpt)]
        rows.append({
            "barrier_kT": eb,
            "tau_sim": float(fpt.mean()),
            "tau_sim_err": float(fpt.std(ddof=1) / np.sqrt(fpt.size)),
            "tau_exact": mfpt_exact(pot, -a.d, a.d, a.kT, a.friction),
            "tau_kramers": 1.0 / kramers_rate(pot, a.kT, a.friction),
        })
        r = rows[-1]
        print(f"  Eb={eb:>5g} kT  tau_sim={r['tau_sim']:10.3f} ± {r['tau_sim_err']:<8.3f}"
              f" exact={r['tau_exact']:10.3f}  kramers={r['tau_kramers']:10.3f}", file=sys.stderr)
    return {"kT": a.kT, "friction": a.friction, "d": a.d, "dt": a.dt, "walkers": a.walkers, "rows": rows}


def cmd_rates(a: argparse.Namespace) -> int:
    result = _rates(a)
    text = json.dumps(result, indent=2)
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


def cmd_figures(a: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import plotting as P

    P.use_style()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    barriers = a.barriers

    ax = P.plot_potentials(barriers, a.kT, a.d)
    ax.figure.savefig(out / "potentials.png")
    plt.close(ax.figure)

    trajs = {}
    for i, eb in enumerate(barriers):
        pot = DoubleWell.in_kT(eb, a.kT, a.d)
        tr = simulate(pot, -a.d, a.steps, a.dt, a.kT, a.friction, stride=a.stride, seed=a.seed + i)
        trajs[eb] = (tr.time, tr.x[:, 0])
    fig = P.plot_trajectories(trajs, a.d)
    fig.savefig(out / "trajectories.png")
    plt.close(fig)

    eq_samples = {}
    for i, eb in enumerate(barriers):
        pot = DoubleWell.in_kT(eb, a.kT, a.d)
        # equilibrium sample: half the walkers in each well, first 10% dropped as burn-in
        x0 = np.where(np.arange(a.walkers) % 2 == 0, -a.d, a.d)
        tr = simulate(pot, x0, a.steps, a.dt, a.kT, a.friction, stride=a.stride, seed=1000 + a.seed + i)
        eq_samples[eb] = tr.x[len(tr.time) // 10 :]
    fig = P.plot_distributions(eq_samples, a.kT, a.d)
    fig.savefig(out / "distributions.png")
    plt.close(fig)

    print("Computing first-passage times...", file=sys.stderr)
    a.barriers = a.rate_barriers
    a.walkers = a.fpt_walkers
    res = _rates(a)
    rows = res["rows"]
    (out / "rates.json").write_text(json.dumps(res, indent=2))
    ax = P.plot_rates([r["barrier_kT"] for r in rows], [r["tau_sim"] for r in rows],
                      [r["tau_sim_err"] for r in rows], [r["tau_exact"] for r in rows],
                      [r["tau_kramers"] for r in rows])
    ax.figure.savefig(out / "rates.png")
    plt.close(ax.figure)
    print(f"Figures written to {out}/", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="overdamped-ld", description="Overdamped Langevin dynamics in a 1D double well."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="run trajectories at one barrier height and summarise kinetics")
    p.add_argument("--barrier", type=float, default=4.0, help="barrier height in kT (default: 4)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--walkers", type=int, default=16)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--x0", type=float, default=-1.0)
    p.add_argument("--core", type=float, default=0.5, help="core-set boundary as a fraction of d (default: 0.5)")
    p.add_argument("--out", type=Path, help="save trajectory to .npz")
    _common(p)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("rates", help="mean first-passage times vs barrier: simulation, exact, Kramers")
    p.add_argument("--barriers", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--walkers", type=int, default=500)
    p.add_argument("--out", type=Path, help="write JSON results")
    _common(p)
    p.set_defaults(func=cmd_rates)

    p = sub.add_parser("figures", help="regenerate all README figures")
    p.add_argument("--outdir", default="docs/figures")
    p.add_argument("--barriers", type=float, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--rate-barriers", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--fpt-walkers", type=int, default=400)
    p.add_argument("--stride", type=int, default=50)
    _common(p)
    p.set_defaults(func=cmd_figures)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
