"""Every example runs (shortened)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "examples").glob("*.py")))
def test_example_runs(script):
    if script == "pettingzoo_and_primitive.py":
        pytest.importorskip("pettingzoo")
    if script == "custom_scenario.py":
        pytest.importorskip("yaml")
    env = {**os.environ, "EXAMPLE_STEPS": "120", "PYTHONPATH": str(ROOT)}
    p = subprocess.run([sys.executable, str(ROOT / "examples" / script)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
