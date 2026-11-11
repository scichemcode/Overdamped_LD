"""Build the live, in-browser explorer (``docs/index.html``).

The page runs this package itself in the browser through Pyodide (CPython
compiled to WebAssembly). At build time the package's simulation and
analysis modules are embedded in the page, so the browser runs exactly the
code that is tested here. Every slider change calls
:func:`overdamped_ld.simulate` in a Web Worker and the result is animated on
a canvas.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from . import __version__

# Modules the browser needs (numpy only; no matplotlib).
BROWSER_MODULES = ("__init__.py", "potentials.py", "integrators.py", "analysis.py")


def bundle_sources() -> dict[str, str]:
    pkg = files("overdamped_ld")
    return {name: pkg.joinpath(name).read_text(encoding="utf-8") for name in BROWSER_MODULES}


def build_web(outdir: str | Path = "docs") -> Path:
    """Write ``<outdir>/index.html``: a self-contained page apart from the Pyodide CDN."""
    template = files("overdamped_ld").joinpath("web_template.html").read_text(encoding="utf-8")
    html = (
        # "<\/" keeps a literal "</script>" inside the sources from closing the tag
        template.replace("/*__SOURCES__*/{}", json.dumps(bundle_sources()).replace("</", "<\\/"))
        .replace("__VERSION__", __version__)
    )
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out
