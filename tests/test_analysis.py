import numpy as np
import pytest

import overdamped_ld as old


def test_boltzmann_normalised():
    x = np.linspace(-3, 3, 2001)
    p = old.boltzmann_density(old.DoubleWell.in_kT(4), x)
    assert np.trapezoid(p, x) == pytest.approx(1.0)
    assert p[np.argmin(abs(x - 1))] > p[np.argmin(abs(x))]


def test_mfpt_free_diffusion_reflecting_wall():
    """Flat potential, wall at 0, start at 0: tau(L) = L^2 / (2D)."""

    class Flat:
        def energy(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

    L, kT, zeta = 2.0, 0.5, 3.0
    tau = old.mfpt_exact(Flat(), 0.0, L, kT, zeta, boundary=0.0)
    assert tau == pytest.approx(L**2 / (2 * kT / zeta), rel=1e-6)


def test_mfpt_symmetric():
    pot = old.DoubleWell.in_kT(3)
    assert old.mfpt_exact(pot, -1, 1) == pytest.approx(old.mfpt_exact(pot, 1, -1), rel=1e-6)


@pytest.mark.parametrize("eb", [8, 12, 16])
def test_kramers_is_high_barrier_limit(eb):
    """Kramers tau = 1/k converges to the exact MFPT as the barrier grows (O(kT/Eb))."""
    pot = old.DoubleWell.in_kT(eb)
    ratio = old.mfpt_exact(pot, -1, 1) * old.kramers_rate(pot)
    assert ratio == pytest.approx(1.0, abs=1.5 / eb)


def test_kramers_scaling():
    k1 = old.kramers_rate(old.DoubleWell.in_kT(5), friction=1.0)
    k2 = old.kramers_rate(old.DoubleWell.in_kT(5), friction=2.0)
    assert k1 / k2 == pytest.approx(2.0)


def test_assign_states_hysteresis():
    x = np.array([0.0, 0.2, 0.6, 0.3, -0.3, 0.4, -0.7, -0.1, 0.1, 0.8])
    s = old.assign_states(x, core=0.5)
    np.testing.assert_array_equal(s, [0, 0, 1, 1, 1, 1, -1, -1, -1, 1])
    assert old.count_transitions(s) == 2


def test_assign_states_2d():
    x = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
    s = old.assign_states(x)
    np.testing.assert_array_equal(s, [[-1, 1], [-1, 1], [1, -1]])
    np.testing.assert_array_equal(old.count_transitions(s), [1, 1])


def test_dwell_times_drop_censored():
    s = np.array([-1, -1, 1, 1, 1, -1, -1, 1])
    d = old.dwell_times(s, 0.5)
    np.testing.assert_allclose(d[1], [1.5])
    np.testing.assert_allclose(d[-1], [1.0])


def test_transition_rate_and_populations():
    s = np.array([0, 1, 1, -1, -1, 1])
    assert old.transition_rate(s, 1.0) == pytest.approx(2 / 5)
    assert old.populations(s) == {-1: 0.4, 1: 0.6}


def test_cli_simulate(capsys):
    from overdamped_ld.cli import main

    assert main(["simulate", "--barrier", "2", "--steps", "2000", "--walkers", "4"]) == 0
    assert '"rate_kramers"' in capsys.readouterr().out
