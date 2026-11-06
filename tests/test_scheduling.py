"""Order scheduler: rule semantics on hand-built cases, common candidate set, decision log,
identical workloads across rules, trace reconstruction, metrics on a synthetic trace, diagnosis,
dashboard and evaluation harness."""
import copy
import json

import numpy as np
import pytest

import battery_tarware as bt
from battery_tarware.orders.dashboard import write_dashboard
from battery_tarware.orders.evaluation import bootstrap_ci, paired, run_grid
from battery_tarware.orders.metrics import evaluate
from battery_tarware.orders.reports import replay_inventory
from battery_tarware.orders.run import run_episode
from battery_tarware.orders.scenarios import order_scenario
from battery_tarware.orders.scheduler import RULE_NAMES, OrderScheduler, _staffable, rack_zones
from battery_tarware.robots import RobotStatus

# basic map: L0000 (3,2) ... storage blocks rows 3-6 and 9-12; goals (15,3), (15,10); docks on row 0
INITIAL = {"L0000": 1, "L0011": 1, "L0040": 1, "L0047": 1, "L0012": 0, "L0013": 0}


def stage(orders, free_agv_cell=(13, 3), weight=4.0, relocation_free=True):
    """Simulator with one free AGV (others broken down), the given orders accepted one tick apart."""
    cfg = order_scenario(source={"kind": "none"}, inventory={"initial": INITIAL, "package_weight": weight})
    env = bt.make(cfg, max_steps=500)
    sim = env.unwrapped.sim
    env.reset(seed=0)
    free = sim.robots[0]
    for r in sim.robots[1:]:
        r.status, r.repair_left, r.resume_status = RobotStatus.FAILED, 10 ** 6, RobotStatus.IDLE
    free.cell = free_agv_cell
    for o in orders:
        sim.orders.submit(o)
        env.step([0] * len(sim.robots))
    return env, sim


def first_assigned(env, sim, rule):
    co = OrderScheduler(rule)
    co.act(env)
    t = sim.tasks[sim.robots[0].task_id] if sim.robots[0].task_id is not None else None
    return sim.orders.task_for_sim(t.id).order_id if t else None


# packages: P00000 on L0000 (3,2), P00001 on L0011 (4,11), P00002 on L0040 (12,2), P00003 on L0047 (12,11)
A = {"op": "outbound", "packages": ["P00000"], "priority": 1, "due": 400}       # far from goal (15,3)
B = {"op": "outbound", "packages": ["P00002"], "priority": 3, "due": 60}        # near goal (15,3)


@pytest.mark.parametrize("rule,expected", [("fifo", "O00000"), ("lwf", "O00000"), ("edd", "O00001"),
                                           ("wsp", "O00001"), ("spt", "O00001"), ("cr", "O00001"),
                                           ("nsp", "O00001"), ("zlp_2d", "O00001")])
def test_rule_choice_on_hand_built_case(rule, expected):
    env, sim = stage([A, B])
    assert first_assigned(env, sim, rule) == expected


def test_nsp_prefers_nearest_even_against_fifo():
    env, sim = stage([B, A], free_agv_cell=(2, 2))           # the AGV stands next to P00000's rack
    assert first_assigned(env, sim, "fifo") == "O00000"
    env, sim = stage([B, A], free_agv_cell=(2, 2))
    assert first_assigned(env, sim, "nsp") == "O00001"


def test_wsp_type_weight_vs_equal_type_ablation():
    reloc = {"op": "relocation", "shelves": ["S0000"], "priority": 2}
    out = {"op": "outbound", "packages": ["P00001"], "priority": 2}
    env, sim = stage([reloc, out])
    assert first_assigned(env, sim, "wsp") == "O00001"         # outbound type score 0.8 > relocation 0.4
    env, sim = stage([reloc, out])
    assert first_assigned(env, sim, "wsp_eq") == "O00000"      # equal type scores: the longer wait wins


def test_plb_components_and_scores_in_unit_interval():
    env, sim = stage([A, B, {"op": "outbound", "packages": ["P00003"]}])
    co = OrderScheduler("plb")
    co.reset(env)
    free = co._free(sim)
    pairs = [(t, c) for t in sim.open_tasks() for c in co._feasible(sim, t, free)]
    occ = co._occupied(sim)
    for t, c in pairs:
        assert 0.0 <= co._conflict(sim, t, c, occ) <= 1.0
        assert co._path_length(sim, t, c) > 0
    from battery_tarware.orders.scheduler import task_info
    sc = co._scores(sim, pairs, {t.id: task_info(sim, t) for t, _ in pairs})
    assert all(-1.0 - 1e-9 <= s[0] <= 0.0 for s in sc)


def test_staffable_needs_distinct_robots():
    class R:
        def __init__(self, c):
            self.c = c
    fits = lambda r, role: r.c == ("transport" if role == "carrier" else "pick")
    assert not _staffable([R("transport"), R("pick")], {"carrier": 1, "picker": 1, "picker_2": 1}, fits)
    assert _staffable([R("transport"), R("pick"), R("pick")], {"carrier": 1, "picker": 1, "picker_2": 1}, fits)


def test_rack_zones_partition_map():
    sim = bt.make(order_scenario()).unwrapped.sim
    z = rack_zones(sim)
    assert (z >= 0).all() and len(np.unique(z)) == 6          # basic: 2 x 3 rack blocks


SRC = {"kind": "poisson", "rate": 0.2, "max_orders": 10 ** 6, "mix": {"outbound": 0.5, "inbound": 0.35, "relocation": 0.15},
       "priority": {"kind": "choice", "values": [1, 2, 3]}, "due_after": {"kind": "uniform", "low": 150, "high": 400}}


def test_common_candidate_set_and_feasibility_across_rules():
    env = bt.make(order_scenario(source=SRC), max_steps=400)
    sim = env.unwrapped.sim
    env.reset(seed=1)
    co = OrderScheduler("fifo")
    for _ in range(150):
        env.step(co.act(env))
    seen = None
    for rule in RULE_NAMES:
        e2 = copy.deepcopy(env)
        OrderScheduler(rule).act(e2)
        d = [e for e in e2.unwrapped.sim.events.events if e.type == "schedule_decision"][-1].data
        cand = sorted((row[0], row[3], row[5] if row[3] == 0 else "feasible") for row in d["tasks"])
        if seen is None:
            seen = cand
        assert cand == seen, rule                           # same eligible tasks, feasibility and rejections
        for tid, robots in d["assigned"].items():           # every committed crew is a feasible one
            t = e2.unwrapped.sim.tasks[int(tid)]
            for i in robots:
                r = e2.unwrapped.sim.robots[i]
                assert t.type.role(r.role).accepts(r) and r.task_id == t.id


def test_identical_arrivals_across_rules():
    arrivals = {}
    for rule in ("fifo", "plb", "spt"):
        env = bt.make(order_scenario(source=SRC), max_steps=600)
        sim = env.unwrapped.sim
        env.reset(seed=3)
        co = OrderScheduler(rule)
        for _ in range(600):
            env.step(co.act(env))
        arrivals[rule] = [(e.tick, e.data["kind"], e.data["priority"], e.data["due"])
                          for e in sim.events.events if e.type == "order_arrived"]
    assert arrivals["fifo"] == arrivals["plb"] == arrivals["spt"]


def test_decision_log_and_runtime():
    res = run_episode(order_scenario(source=SRC), "wsp", 0, 300)
    sim, co = res["sim"], res["scheduler"]
    ds = [e for e in sim.events.events if e.type == "schedule_decision"]
    assert ds and all(e.data["rule"] == "wsp" for e in ds)
    outcomes = {row[5] for e in ds for row in e.data["tasks"]}
    assert "assigned" in outcomes and outcomes <= {"assigned", "robot_taken", "budget", "infeasible:capability",
                                                  "infeasible:payload", "unavailable:robots_busy", "unavailable:charging",
                                                  "unavailable:breakdown", "infeasible:battery", "infeasible:unreachable"}
    assert len(co.runtime) == 300 and all(r[1] >= 0 for r in co.runtime)
    for e in ds:                                             # assigned crews match the simulator's joins
        for tid, robots in e.data["assigned"].items():
            joins = {x.robots[0] for x in sim.events.events if x.type == "robot_joined" and x.task == int(tid)
                     and x.tick == e.tick}
            assert set(robots) <= joins


def test_scheduler_runs_task_only_scenarios():
    for name in ("basic", "intermediate"):
        env = bt.make(name, max_steps=300)
        env.reset(seed=0)
        co = OrderScheduler("nsp")
        for _ in range(300):
            env.step(co.act(env))
        assert env.unwrapped.sim.metrics()["tasks_completed"] > 0


# ─────────────────────────────────────────────────────────────── trace, metrics, diagnosis
def test_trace_reconstructs_history():
    res = run_episode(order_scenario(source=SRC), "plb", 2, 500)
    tr, sim = res["trace"], res["sim"]
    last = tr["frames"][-1]
    for r, f in zip(sim.robots, last):
        assert (f[0], f[1]) == r.cell and abs(f[2] - r.soc) < 1e-3
    per = {row["robot"]: row["distance"] for row in res["metrics"]["robot"]["per_robot"]}
    assert per == {r.name: r.distance for r in sim.robots}
    replay = replay_inventory(tr["events"])
    inv = sim.orders.inventory
    for pid, p in inv.packages.items():
        rp = replay[pid]
        assert (rp["status"], rp["shelf"], rp["loc"], rp["reserved"]) == \
            (p.status, p.shelf_id, inv.rack_location_of(p), p.reserved_by), pid
    js = json.loads(json.dumps(tr, default=str))
    assert js["meta"]["ticks"] == sim.tick and len(js["frames"]) == sim.tick


def test_metrics_on_synthetic_trace():
    # 2 robots, 4 ticks; order O0 arrives 1, completes 3 (due 5); O1 arrives 2, open at T=4
    ev = [[1, "order_arrived", [], None, {"order_id": "O0", "kind": "outbound", "priority": 1, "due": 5}],
          [1, "robot_joined", [0], 7, {"role": "carrier"}], [2, "stage_started", [0], 7, {"stage": "pick"}],
          [2, "order_arrived", [], None, {"order_id": "O1", "kind": "inbound", "priority": 2, "due": 3}],
          [3, "order_task", [], 7, {"task_id": "T0", "order_id": "O0", "to": "completed"}],
          [3, "order_completed", [], None, {"order_id": "O0"}],
          [3, "inventory", [], None, {"op": "package_shipped"}]]
    # frame row: [y, x, soc, status, task, role, activity]; activities: 1 travel_empty, 3 working, 4 waiting_partner, 0 idle
    fr = [[[0, 1, 1.0, 1, 7, 0, 1], [5, 5, 1.0, 0, -1, -1, 0]], [[0, 1, 1.0, 1, 7, 0, 3], [5, 5, 1.0, 0, -1, -1, 0]],
          [[0, 1, 1.0, 1, 7, 0, 4], [5, 5, 1.0, 0, -1, -1, 0]], [[0, 2, 1.0, 0, -1, -1, 1], [5, 5, 1.0, 0, -1, -1, 0]]]
    tr = {"meta": {"scenario": "x", "seed": 0, "first_tick": 1, "ticks": 4, "robots": [{"name": "a", "type": "agv"},
                                                                                          {"name": "b", "type": "agv"}]},
          "initial": [[0, 0, 1.0], [5, 5, 1.0]], "frames": fr, "events": ev, "series": {"open_orders": [1, 2, 1, 1]},
          "spans": {"T0": [[1, 1, "travel"], [2, 2, "execution"], [3, 3, "coalition_waiting"]]},
          "energy_wh": {"a": 10.0, "b": 2.0},
          "tables": {"orders": [{"order_id": "O0", "state": "completed", "arrival": 1, "tasks": "T0"},
                                {"order_id": "O1", "state": "open", "arrival": 2, "tasks": ""}], "tasks": []}}
    m = evaluate(tr)
    assert m["order"]["completed"] == 1 and m["order"]["unfinished"] == 1
    assert m["order"]["lead_time_mean"] == 2 and m["order"]["throughput_per_1000"] == 250.0
    assert m["order"]["on_time_rate"] == 0.5                  # O0 on time; O1 open past its due (3 <= 4) = miss
    assert m["robot"]["distance_total"] == 2 and m["robot"]["per_robot"][0]["utilization"] == 1.0
    assert m["coalition"]["waiting_ratio"] == 0.25 and m["coalition"]["formation_time"]["mean"] == 1
    assert m["system"]["energy_wh_per_order"] == 12.0 and m["system"]["deliveries"] == 1
    assert m["bottlenecks"]["task_ticks_by_cause"] == {"coalition_waiting": 1, "execution": 1, "travel": 1}
    assert m["bottlenecks"]["unsuccessful_orders"][0]["order_id"] == "O1"


def test_diagnosis_capability_and_charging_contention():
    cfg = order_scenario(source={"kind": "trace", "orders": [
        {"arrival": 0, "op": "inbound", "packages": [{"weight_kg": 80.0}]},          # no crew can lift 80 kg
        {"arrival": 0, "op": "outbound", "packages": ["P00000"]}]},                   # 8 kg: needs a picking robot
        inventory={"initial": {"L0000": 1, "L0010": 0}, "package_weight": 8.0})
    env = bt.make(cfg, max_steps=150)
    sim = env.unwrapped.sim
    env.reset(seed=0)
    for r in sim.robots:
        if r.spec.name == "picking_robot":
            r.energy_wh = 0.2 * r.spec.energy.capacity_wh         # below the 30% charging rule
    co = OrderScheduler("fifo")
    from battery_tarware.orders.trace import TraceRecorder
    rec = TraceRecorder(sim, co)
    co.reset(env)
    for _ in range(150):
        env.step(co.act(env))
        rec.update()
    from battery_tarware.orders.run import tables
    m = evaluate(rec.to_dict(tables=tables(sim)))
    causes = m["bottlenecks"]["task_ticks_by_cause"]
    assert causes.get("charging_contention", 0) > 0
    diag = {u["order_id"]: u for u in m["bottlenecks"]["unsuccessful_orders"]}
    assert diag["O00000"]["category"] == "insufficient capabilities"


# ─────────────────────────────────────────────────────────────── dashboard and evaluation
def test_dashboard_is_self_contained(tmp_path):
    res = run_episode(order_scenario(source=SRC), "fifo", 0, 120)
    p = write_dashboard([res["trace"]], str(tmp_path / "d.html"))
    html = open(p).read()
    assert "<script src" not in html and "http" not in html.split("<script id=\"data\"")[0].replace("http-equiv", "")
    data = html.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(data.replace("<\\/", "</"))["runs"][0]["meta"]["ticks"] == 120


def test_paired_bootstrap_and_grid():
    rows = run_grid({"c": order_scenario(source=SRC)}, ["fifo", "nsp"], [0, 1], 150, 0, jobs=1)
    assert all(r["error"] is None for r in rows)
    pr = paired(rows, "fifo", ("throughput",))
    nsp = [p for p in pr if p["rule"] == "nsp"][0]
    assert nsp["n"] == 2 and nsp["ci_lo"] <= nsp["diff"] <= nsp["ci_hi"]
    lo, hi = bootstrap_ci(np.array([1.0, 1.0, 1.0]))
    assert lo == hi == 1.0
    bad = run_grid({"broken": {"name": "x"}}, ["fifo"], [0], 10, 0)
    assert bad[0]["error"] and "ConfigError" in bad[0]["error"]
