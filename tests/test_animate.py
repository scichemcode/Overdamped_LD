import json
import re

import pytest

pytest.importorskip("PIL")

from overdamped_ld.animate import AnimationConfig, build_site  # noqa: E402


def test_build_site_small(tmp_path):
    cfg = AnimationConfig(t_total=2.0, n_frames=4, trace_points_per_frame=2, width_in=3, height_in=2.2, dpi=50)
    out = build_site(tmp_path, barriers=(1, 3), cfg=cfg, workers=1)
    html = out.read_text()
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html).group(1))
    assert [h["barrier"] for h in data["heights"]] == [1, 3]
    assert data["heights"][1]["tau_exact"] > data["heights"][0]["tau_exact"]
    for b in (1, 3):
        for theme in ("light", "dark"):
            assert (tmp_path / "anim" / f"eb{b}-{theme}.webp").stat().st_size > 0
            assert (tmp_path / "anim" / f"eb{b}-{theme}.png").exists()
