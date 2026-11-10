"""Physics validation: the integrator must reproduce known statistical mechanics."""

import numpy as np
import pytest

import overdamped_ld as old


class Harmonic:
    """U = k x^2 / 2 — has closed-form Ornstein–Uhlenbeck statistics."""

    def __init__(self, k):
        self.k = k

    def energy(self, x):
        return 0.5 * self.k * np.asarray(x) ** 2

    def force(self, x):
        return -self.k * np.asarray(x)

    def curvature(self, x):
        return np.full_like(np.asarray(x, dtype=float), self.k)


def test_free_diffusion_msd():
    """<x^2> = 2 D t with D = kT / zeta."""
    kT, zeta, dt, n = 0.7, 2.0, 1e-3, 1000
    traj = old.simulate(Harmonic(0.0), 0.0, n, dt, kT, zeta, n_walkers=4000, stride=n, seed=0)
    msd = np.mean(traj.x[-1] ** 2)
    expected = 2 * kT / zeta * n * dt
    assert msd == pytest.approx(expected, rel=0.05)


def test_harmonic_equipartition():
    """Stationary variance kT/k (up to O(dt) Euler–Maruyama bias)."""
    k, kT = 4.0, 0.5
    traj = old.simulate(Harmonic(k), 0.0, 20_000, 1e-3, kT, 1.0, n_walkers=500, stride=100, seed=1)
    var = np.var(traj.x[50:])
    assert var == pytest.approx(kT / k, rel=0.05)


def test_double_well_samples_boltzmann():
    pot = old.DoubleWell.in_kT(2.0)
    x0 = np.tile([-1.0, 1.0], 200)
    traj = old.simulate(pot, x0, 40_000, 2e-3, stride=40, seed=2)
    xs = traj.x[100:].ravel()
    bins = np.linspace(-2, 2, 41)
    hist, edges = np.histogram(xs, bins=bins, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    fine = np.linspace(-2.5, 2.5, 5001)
    ref = np.interp(centres, fine, old.boltzmann_density(pot, fine))
    assert np.max(np.abs(hist - ref)) < 0.06


def test_seed_reproducible():
    pot = old.DoubleWell.in_kT(3)
    a = old.simulate(pot, -1, 500, seed=123).x
    b = old.simulate(pot, -1, 500, seed=123).x
    np.testing.assert_array_equal(a, b)


def test_stride_and_shapes():
    traj = old.simulate(old.DoubleWell(), [-1.0, 0.0, 1.0], 100, stride=10, seed=0)
    assert traj.x.shape == (11, 3)
    np.testing.assert_allclose(traj.time[-1], 100 * 1e-3)
    np.testing.assert_array_equal(traj.x[0], [-1, 0, 1])


def test_trajectory_roundtrip(tmp_path):
    traj = old.simulate(old.DoubleWell.in_kT(2), -1, 200, n_walkers=3, stride=20, seed=0)
    path = tmp_path / "t.npz"
    traj.save(path)
    back = old.Trajectory.load(path)
    np.testing.assert_array_equal(back.x, traj.x)
    assert back.dt == traj.dt and back.stride == traj.stride
    assert back.params["barrier"] == 2.0


def test_mfpt_simulation_matches_exact():
    """Brownian-dynamics MFPT agrees with the exact quadrature."""
    pot = old.DoubleWell.in_kT(3.0)
    fpt = old.first_passage_times(pot, -1.0, 1.0, n_walkers=600, dt=1e-3, seed=7)
    assert np.all(np.isfinite(fpt))
    exact = old.mfpt_exact(pot, -1.0, 1.0)
    sem = fpt.std(ddof=1) / np.sqrt(fpt.size)
    assert abs(fpt.mean() - exact) < 4 * sem


def test_first_passage_max_time():
    fpt = old.first_passage_times(old.DoubleWell.in_kT(20), -1, 1, n_walkers=10, max_time=0.5, seed=0)
    assert np.all(np.isnan(fpt))
