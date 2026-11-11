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


def test_tutorial_scenario_serves_every_task_type():
    pytest.importorskip("yaml")
    import battery_tarware as bt
    env = bt.make(str(ROOT / "examples" / "tutorial.yaml"), max_steps=1500)
    m = bt.run_episode(env, bt.GreedyCoalitionCoordinator("fifo"), seed=0)
    done = {t.type.name for t in env.sim.tasks.values() if t.state.value == "completed"}
    assert done == {"tote", "pallet", "return_tote"} and m["stranded"] == 0


def test_ascii_map_accepts_row_list():
    import battery_tarware as bt
    from battery_tarware.config import MapSpec
    rows = ["C..", ".S.", "G.."]
    assert MapSpec(ascii=rows).ascii == MapSpec(ascii="\n".join(rows)).ascii
    bt.ScenarioConfig  # noqa: B018 (import check)
