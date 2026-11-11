import json
import re

import numpy as np

import overdamped_ld as old
from overdamped_ld.integrators import _simulate_vector
from overdamped_ld.web import BROWSER_MODULES, build_web


def test_scalar_fast_path_matches_vector_path():
    pot = old.DoubleWell.in_kT(3, d=1.3)
    fast = old.simulate(pot, -1.0, 5000, dt=2e-3, kT=0.8, friction=1.5, stride=7, seed=11).x
    ref = np.empty_like(fast)
    ref[0] = -1.0
    rng = np.random.default_rng(11)
    _simulate_vector(pot, np.array([-1.0]), 5000, 7, 2e-3 / 1.5, np.sqrt(2 * 0.8 * 2e-3 / 1.5), rng, ref)
    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12)  # same RNG stream; differs only in rounding


def test_build_web_embeds_package(tmp_path):
    html = build_web(tmp_path).read_text()
    m = re.search(r"const PY_SOURCES = (\{.*?\});\n", html)
    sources = json.loads(m.group(1).replace("<\\/", "</"))
    assert sorted(sources) == sorted(BROWSER_MODULES)
    assert "def simulate(" in sources["integrators.py"]
    assert "__VERSION__" not in html and old.__version__ in html
    # the browser bundle must not need matplotlib
    assert not any("matplotlib" in src for src in sources.values())
