"""Animated trajectories (matplotlib ``FuncAnimation``) and the interactive web page.

Each animation shows the particle sliding on the potential (top) while its
trajectory x(t) is drawn underneath. Every barrier height is driven by the
*same* noise sequence, so the only thing that changes between animations is
the barrier.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from .analysis import assign_states, count_transitions, kramers_rate, mfpt_exact, populations
from .integrators import simulate
from .potentials import DoubleWell

THEMES = {
    "light": {
        "bg": "#fbfbf9", "ink": "#1b1d22", "ink2": "#5a5d66", "muted": "#9a9ca3", "grid": "#e4e4df",
        "curve": "#c9cbd1", "ramp": ["#8db8ea", "#2a78d6", "#0d366b"], "ball": "#e8590c",
    },
    "dark": {
        "bg": "#15171b", "ink": "#eceae4", "ink2": "#b4b2aa", "muted": "#74767c", "grid": "#26282d",
        "curve": "#3b3e45", "ramp": ["#9ec5f4", "#5598e7", "#3987e5"], "ball": "#ff8a4c",
    },
}


@dataclass
class AnimationConfig:
    t_total: float = 150.0      # simulated time shown in one loop
    dt: float = 1e-3
    n_frames: int = 240
    fps: int = 24
    trace_points_per_frame: int = 10  # x(t) resolution between frames
    kT: float = 1.0
    friction: float = 1.0
    seed: int = 7
    quality: int = 60          # WebP quality
    width_in: float = 6.4
    height_in: float = 4.6
    dpi: int = 100


def _ramp_color(theme: dict, frac: float) -> str:
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    return to_hex(LinearSegmentedColormap.from_list("r", theme["ramp"])(frac))


def simulate_clip(barrier_kT: float, cfg: AnimationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory used for the animation, sampled ``trace_points_per_frame`` times per frame."""
    n_points = cfg.n_frames * cfg.trace_points_per_frame
    n_steps = int(round(cfg.t_total / cfg.dt))
    stride = max(1, n_steps // n_points)
    traj = simulate(DoubleWell.in_kT(barrier_kT, cfg.kT), -1.0, stride * n_points, cfg.dt, cfg.kT,
                    cfg.friction, stride=stride, seed=cfg.seed)
    return traj.time, traj.x[:, 0]


def render_animation(
    barrier_kT: float,
    out_path: str | Path,
    theme: str = "light",
    cfg: AnimationConfig | None = None,
    color_frac: float = 0.5,
    clip: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Render one barrier height to an animated WebP (or GIF, by extension)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cfg = cfg or AnimationConfig()
    th = THEMES[theme]
    color = _ramp_color(th, color_frac)
    t, x = clip if clip is not None else simulate_clip(barrier_kT, cfg)
    pot = DoubleWell.in_kT(barrier_kT, cfg.kT)
    k = cfg.trace_points_per_frame

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": th["grid"], "axes.labelcolor": th["ink2"],
        "xtick.color": th["muted"], "ytick.color": th["muted"],
        "xtick.labelcolor": th["ink2"], "ytick.labelcolor": th["ink2"],
    })
    fig, (ax_u, ax_x) = plt.subplots(
        2, 1, figsize=(cfg.width_in, cfg.height_in), dpi=cfg.dpi,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.45},
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.1)
    fig.patch.set_facecolor(th["bg"])
    for ax in (ax_u, ax_x):
        ax.set_facecolor(th["bg"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # --- potential panel -------------------------------------------------
    xs = np.linspace(-1.75, 1.75, 400)
    u_max = max(barrier_kT, 1.0) * 1.6 + 0.5
    ax_u.plot(xs, pot.energy(xs) / cfg.kT, color=color, lw=2.2, zorder=1)
    ax_u.set_xlim(-1.75, 1.75)
    ax_u.set_ylim(-0.08 * u_max, u_max)
    ax_u.set_xlabel("Position $x$")
    ax_u.set_ylabel("$U\\,/\\,k_BT$")
    ax_u.grid(True, color=th["grid"], lw=0.6)
    trail = ax_u.scatter(np.zeros(12), np.full(12, np.nan), zorder=2, linewidths=0)
    ball, = ax_u.plot([], [], "o", ms=11, color=th["ball"], mec=th["bg"], mew=2, zorder=3)
    clock = ax_u.text(1.0, 1.04, "", transform=ax_u.transAxes, ha="right", va="bottom",
                      color=th["ink2"], fontsize=10, family="monospace")
    hops_txt = ax_u.text(0.0, 1.04, "", transform=ax_u.transAxes, ha="left", va="bottom",
                         color=th["ink2"], fontsize=10, family="monospace")

    # --- trajectory panel ------------------------------------------------
    ax_x.set_xlim(0, t[-1])
    ax_x.set_ylim(-1.9, 1.9)
    ax_x.set_yticks([-1, 0, 1])
    ax_x.set_xlabel("Time")
    ax_x.set_ylabel("$x(t)$")
    for y in (-1, 1):
        ax_x.axhline(y, color=th["muted"], lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax_x.axhline(0, color=th["grid"], lw=0.8, zorder=0)
    line, = ax_x.plot([], [], color=color, lw=1.1)
    head, = ax_x.plot([], [], "o", ms=6, color=th["ball"], mec=th["bg"], mew=1.5)

    states = assign_states(x, core=0.5)
    flips = np.concatenate(([0], np.cumsum((states[1:] != states[:-1]) & (states[:-1] != 0))))

    # fading trail: the last few positions, older ones smaller and fainter
    n_trail, trail_step = 12, max(1, k // 2)
    trail_rgba = np.tile(matplotlib.colors.to_rgba(th["ball"]), (n_trail, 1))
    trail_rgba[:, 3] = np.linspace(0.05, 0.45, n_trail)
    trail.set_facecolors(trail_rgba)
    trail.set_sizes(np.linspace(8, 60, n_trail))

    def update(f: int):
        i = min((f + 1) * k, len(x) - 1)
        xi = x[i]
        ball.set_data([xi], [pot.energy(xi) / cfg.kT])
        tx = x[np.clip(np.arange(i - n_trail * trail_step, i, trail_step), 0, None)]
        trail.set_offsets(np.c_[tx, pot.energy(tx) / cfg.kT])
        line.set_data(t[: i + 1], x[: i + 1])
        head.set_data([t[i]], [xi])
        clock.set_text(f"t = {t[i]:6.1f}")
        hops_txt.set_text(f"hops: {flips[i]}")
        return ball, trail, line, head, clock, hops_txt

    anim = FuncAnimation(fig, update, frames=cfg.n_frames, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".webp":
        writer = _webp_writer_class()(fps=cfg.fps, quality=cfg.quality)
    else:
        writer = PillowWriter(fps=cfg.fps)
    anim.save(out_path, writer=writer, savefig_kwargs={"facecolor": th["bg"]})
    plt.close(fig)
    return out_path


def _webp_writer_class():
    """PillowWriter subclass that saves an animated WebP with a quality setting."""
    from matplotlib.animation import PillowWriter

    class WebPWriter(PillowWriter):
        def __init__(self, fps: int, quality: int = 75):
            super().__init__(fps=fps)
            self.quality = quality

        def finish(self):
            self._frames[0].save(
                self.outfile, save_all=True, append_images=self._frames[1:],
                duration=int(1000 / self.fps), loop=0, quality=self.quality, method=4,
            )

    return WebPWriter



# --------------------------------------------------------------------------
# Web page
# --------------------------------------------------------------------------


def _clip_stats(barrier_kT: float, cfg: AnimationConfig, clip) -> dict:
    t, x = clip
    pot = DoubleWell.in_kT(barrier_kT, cfg.kT)
    s = assign_states(x, core=0.5)
    return {
        "barrier": barrier_kT,
        "hops": int(count_transitions(s)),
        "frac_right": round(populations(s)[1], 3),
        "tau_exact": mfpt_exact(pot, -1, 1, cfg.kT, cfg.friction),
        "tau_kramers": 1.0 / kramers_rate(pot, cfg.kT, cfg.friction),
    }


def _render_one(args) -> dict:
    barrier, frac, outdir, cfg = args
    clip = simulate_clip(barrier, cfg)
    from PIL import Image

    for theme in THEMES:
        webp = render_animation(barrier, Path(outdir) / f"eb{barrier:g}-{theme}.webp", theme, cfg, frac, clip)
        # last frame as a still, shown to visitors who prefer reduced motion
        with Image.open(webp) as im:
            im.seek(im.n_frames - 1)
            im.convert("RGB").save(webp.with_suffix(".png"), optimize=True)
    return _clip_stats(barrier, cfg, clip)


def build_site(
    outdir: str | Path = "docs",
    barriers: tuple[float, ...] = tuple(range(1, 11)),
    cfg: AnimationConfig | None = None,
    workers: int | None = None,
) -> Path:
    """Render all animations and write ``<outdir>/index.html``.

    Animations go to ``<outdir>/anim/``. The page is self-contained apart from
    those images, so it works from GitHub Pages or straight from disk.
    """
    cfg = cfg or AnimationConfig()
    outdir = Path(outdir)
    anim_dir = outdir / "anim"
    anim_dir.mkdir(parents=True, exist_ok=True)
    fracs = np.linspace(0, 1, len(barriers)) if len(barriers) > 1 else [0.5]
    jobs = [(b, float(f), anim_dir, cfg) for b, f in zip(barriers, fracs)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(_render_one, jobs))

    curve_eb = np.linspace(min(barriers), max(barriers), 60)
    curve = [{"barrier": float(e),
              "tau_exact": mfpt_exact(DoubleWell.in_kT(e, cfg.kT), -1, 1, cfg.kT, cfg.friction),
              "tau_kramers": 1 / kramers_rate(DoubleWell.in_kT(e, cfg.kT), cfg.kT, cfg.friction)}
             for e in curve_eb]
    data = {
        "clip_time": cfg.t_total, "seed": cfg.seed, "kT": cfg.kT, "friction": cfg.friction,
        "dt": cfg.dt, "loop_seconds": round(cfg.n_frames / cfg.fps, 1),
        "heights": stats, "curve": curve,
    }
    template = files("overdamped_ld").joinpath("web_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out = outdir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
