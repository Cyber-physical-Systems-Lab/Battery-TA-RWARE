"""Automated browser tests of the replay dashboard (headless Chrome/Chromium; skipped without one).

The page is driven through its own handlers (playback, kanban, filters, dependency graph, robot,
package, decision and dock selection, exports, deep links) and everything it shows is compared
with the authoritative event log and frames (experiments/checks/dashboard_browser.py)."""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "checks"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "orders_v2"))
import dashboard_browser as D  # noqa: E402

pytestmark = pytest.mark.skipif(D.find_chrome() is None, reason="no Chrome/Chromium (set BT_CHROME)")


def _trace(cfg, rule, seed, ticks):
    from battery_tarware.orders.run import run_episode
    return run_episode(cfg, rule, seed, ticks, 0)["trace"]


def test_dashboard_example_orders_with_dependencies():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "examples", "orders.yaml")))
    assert D.verify(_trace(cfg, "fifo", 0, None)) == []


@pytest.mark.parametrize("base,model", [("intermediate", "2.2"), ("advanced", "2.2"), ("basic", "2.1")])
def test_dashboard_generated_compound_orders(base, model):
    import workloads
    tr = _trace(workloads.build(base, model, "standard", max_steps=1500), "nsp", 1, 1500)
    assert D.verify(tr) == []


def test_dashboard_long_episode():
    import workloads
    tr = _trace(workloads.build("basic", "2.2", "standard", battery="tight", max_steps=6000), "wsp_eq", 2, 6000)
    assert D.verify(tr, n_ticks=20) == []
