import numpy as np
import pytest

from overdamped_ld import DoubleWell


@pytest.mark.parametrize("barrier,d", [(1.0, 1.0), (3.5, 0.7), (10.0, 2.0)])
def test_force_is_minus_gradient(barrier, d):
    pot = DoubleWell(barrier, d)
    x = np.linspace(-2.5 * d, 2.5 * d, 101)
    h = 1e-6
    numeric = -(pot.energy(x + h) - pot.energy(x - h)) / (2 * h)
    np.testing.assert_allclose(pot.force(x), numeric, rtol=1e-6, atol=1e-6)


def test_curvature_is_second_derivative():
    pot = DoubleWell(2.0, 1.3)
    x = np.linspace(-2, 2, 51)
    h = 1e-5
    numeric = (pot.force(x - h) - pot.force(x + h)) / (2 * h)
    np.testing.assert_allclose(pot.curvature(x), numeric, rtol=1e-6, atol=1e-6)


def test_landmarks():
    pot = DoubleWell.in_kT(5, kT=0.5, d=2.0)
    assert pot.barrier == 2.5
    np.testing.assert_allclose(pot.energy(pot.minima), 0.0)
    np.testing.assert_allclose(pot.energy(pot.saddle), pot.barrier)
    np.testing.assert_allclose(pot.force(pot.minima), 0.0)
    assert pot.curvature(pot.d) == pytest.approx(8 * pot.barrier / pot.d**2)
    assert pot.curvature(0.0) == pytest.approx(-4 * pot.barrier / pot.d**2)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        DoubleWell(-1.0)
    with pytest.raises(ValueError):
        DoubleWell(1.0, d=0.0)
