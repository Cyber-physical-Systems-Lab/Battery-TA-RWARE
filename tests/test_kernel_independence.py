"""battery_tarware is a standalone kernel: no imports of the legacy `tarware` package, of RendeCoord
(`planner`, `sh`, `analysis`), or of any native coordinator; it runs with only its own directory on
the path."""
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "battery_tarware"
FORBIDDEN_TOP = {"tarware", "planner", "sh", "analysis", "torch", "pettingzoo"}


def test_kernel_sources_have_no_forbidden_imports():
    bad = []
    for py in KERNEL.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__" and node.args \
                    and isinstance(node.args[0], ast.Constant):
                names = [node.args[0].value]
            bad += ["%s:%d %s" % (py.relative_to(ROOT), node.lineno, n) for n in names
                    if n.split(".")[0] in FORBIDDEN_TOP]
    assert not bad, bad


def test_kernel_runs_in_isolation(tmp_path):
    (tmp_path / "battery_tarware").symlink_to(KERNEL, target_is_directory=True)
    site = [p for p in sys.path if p.rstrip("/").endswith(("site-packages", "dist-packages"))]
    code = textwrap.dedent("""
        import importlib, pkgutil, sys
        import battery_tarware as bt
        for m in pkgutil.walk_packages(bt.__path__, "battery_tarware."):
            if m.name != "battery_tarware.rendering":      # needs a display stack (pyglet)
                importlib.import_module(m.name)
        from battery_tarware.protocol_v4 import make_tiled_warehouse_v4
        env = bt.make_tiled_warehouse(tile_rows=1, tile_cols=2, num_agvs=3, num_pickers=1, max_steps=50)
        env.reset(seed=3)
        for _ in range(20):
            done, info = env.step([0] * 4)
        env4 = make_tiled_warehouse_v4(tile_rows=1, tile_cols=2, num_agvs=3, num_pickers=1, max_steps=50)
        env4.reset(seed=3); env4.step([0] * 4)
        leaked = sorted(n for n in sys.modules if n.split(".")[0] in {"tarware", "planner"})
        assert not leaked, leaked
        print("OK", bt.__version__)
    """)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path)] + site)
    # -S: no site processing, so no editable-install hook can put a repository back on the path.
    p = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    assert p.stdout.startswith("OK")
