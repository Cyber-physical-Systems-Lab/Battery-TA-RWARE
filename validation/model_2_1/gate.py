"""Model 2.1 behavioural gate (PREREGISTRATION.md in this folder): external, per-tick validator.

The simulator package is imported from ``$BT_ROOT`` (default: this worktree), so the same script
records the model 2.0 reference digests on the base commit and validates model 2.1 here.

  python validation/model_2_1/gate.py episode <scenario> <seed> <config> <mode> <out.json>
  python validation/model_2_1/gate.py probe <out.json>
  python validation/model_2_1/gate.py report <out_dir>

config: A (2.0, inline) | B (2.1, inline) | C (2.0, bays) | D (2.1, bays) | REF (base commit, no new keys)
mode:   macro (greedy FIFO, full episode) | primitive (masked random, 500 ticks)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BT_ROOT = os.environ.get("BT_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, BT_ROOT)
sys.path.insert(0, os.path.join(BT_ROOT, "docs"))

import battery_tarware as bt                       # noqa: E402
import battery_tarware.traffic as T               # noqa: E402
from battery_tarware.robots import RobotStatus     # noqa: E402

CONFIGS = {"A": ("2.0", "inline"), "B": ("2.1", "inline"), "C": ("2.0", "bays"), "D": ("2.1", "bays"), "REF": None}
SCENARIOS = ("basic", "intermediate", "advanced", "basic_large", "intermediate_large", "advanced_large")
STREAK, OSC = 100, 10                              # L1 and L2 thresholds (frozen)
SEPARATE = 10      # PROPOSED S1 amendment (not registered; defined before run 6): former partners must part within this many ticks
PRIMITIVE_TICKS = 500


def scenario(name: str, config: str) -> dict:
    import make_gifs
    if CONFIGS[config] is None:                    # REF: the base commit's own definitions
        if name.endswith("_large"):
            return copy.deepcopy(make_gifs.large_scenarios()[name[:-len("_large")]])
        return copy.deepcopy(bt.get_scenario(name))
    model, layout = CONFIGS[config]
    if name.endswith("_large"):
        cfg = copy.deepcopy(make_gifs.large_scenarios(model=model)[name[:-len("_large")]])
    else:
        cfg = copy.deepcopy(bt.get_scenario(name, model=model))
    if True:
        cfg["model"] = model
        cfg["map"] = dict(cfg["map"], dock_layout=layout)
    return cfg


def digest(sim) -> str:
    rob = tuple((r.cell, round(r.energy_wh, 9), r.status.value, r.task_id, r.role, r.charger, r.charging)
                for r in sim.robots)
    tasks = tuple((t.id, t.state.value, t.stage, round(t.progress, 9), t.payload_cell, t.carried,
                   tuple(sorted((k, tuple(v)) for k, v in t.assigned.items())))
                  for t in sorted(sim.tasks.values(), key=lambda t: t.id))
    return hashlib.sha1(repr((sim.tick, rob, tasks)).encode()).hexdigest()[:16]


class Monitor:
    """Wraps PIBT.step to see every unit's goal and move; checks invariants after every tick."""

    def __init__(self, sim):
        self.sim = sim
        self.c = Counter()
        self.streak = {}            # unit id -> [goal, best static distance, ticks without a new minimum]
        self.osc = {}               # unit id -> [cell two moves ago, previous cell, alternating moves]
        self.max_streak = 0
        self.max_osc = 0
        self.violations = defaultdict(list)
        self.units = []
        self.pairs = {}             # PROPOSED S1: (frozenset(robot ids), cell) -> ticks shared after their task let them go
        orig = T.PIBT.step
        mon = self

        def step(pibt, units):
            orig(pibt, units)
            mon.units = [(u.id, u.cell, u.goal, u.next, u.zones, u.fixed, getattr(u, "lasting", False)) for u in units]
            mon._movement(pibt, units)
        T.PIBT.step = step

    def _movement(self, pibt, units):
        sim = self.sim
        lasting = {u.cell for u in units if u.fixed and (not all(m.operational for m in u.members)
                                                          or any(m.charging for m in u.members))}
        for u in units:
            moved = u.next is not None and u.next != u.cell
            # L2 oscillation: consecutive moves alternating between the same two cells
            o = self.osc.setdefault(u.id, [None, u.cell, 0])
            if moved:
                if o[0] == u.next and o[1] == u.cell:
                    o[2] += 1
                else:
                    o[2] = 1
                o[0], o[1] = u.cell, u.next
                self.max_osc = max(self.max_osc, o[2])
                if o[2] == OSC:
                    self.violations["L2"].append((sim.tick, u.id, u.cell))
            if u.goal is None or u.goal == u.cell:
                self.streak.pop(u.id, None)
                continue
            self.c["intent_ticks"] += 1
            d = int(pibt.distances(u.zones, u.goal)[u.cell])
            s = self.streak.get(u.id)
            if s is None or s[0] != u.goal:
                s = self.streak[u.id] = [u.goal, d, 0]
            elif d < s[1]:
                s[1], s[2] = d, 0
            else:
                s[2] += 1
                self.max_streak = max(self.max_streak, s[2])
                if s[2] == STREAK:
                    self.violations["L1"].append((sim.tick, u.id, u.cell, u.goal))
            if not u.fixed and not moved:
                st = u.members[0].status.value
                self.c["stalled"] += 1
                self.c["stalled_" + st] += 1
                dist = pibt.distances(u.zones, u.goal)
                closer = [(u.cell[0] + dy, u.cell[1] + dx) for dy, dx in T.MOVES]
                closer = [c for c in closer if sim.layout.in_bounds(c) and dist[c] < dist[u.cell]]
                if closer and all(c in lasting for c in closer):
                    self.c["stalled_behind_lasting"] += 1

    def after_tick(self, prev_cells):
        sim = self.sim
        # S1 vertex conflicts (registered rule), and the PROPOSED amendment S1p alongside
        by_cell = defaultdict(list)
        for r in sim.robots:
            by_cell[r.cell].append(r)
        pairs_now = {}
        for cell, rs in by_cell.items():
            if len(rs) > 1:
                tids = {r.task_id for r in rs}
                ok = (len(tids) == 1 and None not in tids
                      and all(r.role == sim.tasks[r.task_id].type.carrier_role for r in rs))
                if not ok:
                    self.violations["S1"].append((sim.tick, cell, [r.id for r in rs]))
                # S1p: (ii) a task's carrier and its crew on that task's work cell are allowed
                t = sim.tasks.get(next(iter(tids))) if len(tids) == 1 and None not in tids else None
                crew_ok = t is not None and cell in (t.stage_cell, t.payload_cell)
                if ok or crew_ok:
                    pairs_now[(frozenset(r.id for r in rs), cell)] = 0
                    continue
                # (iii) former partners (allowed last tick on this cell) still parting: at most SEPARATE
                # ticks. Revised after run 6: a group counts if it is a subset of a group allowed on
                # the same cell the tick before (a partner that leaves first does not reset it).
                key = frozenset(r.id for r in rs)
                prev = [v for (g, c), v in self.pairs.items() if c == cell and key <= g]
                if prev and min(prev) + 1 <= SEPARATE:
                    pairs_now[(key, cell)] = min(prev) + 1
                    continue
                self.violations["S1p"].append((sim.tick, cell, [r.id for r in rs]))
        self.pairs = pairs_now
        # S2 swaps
        cur = {r.id: r.cell for r in sim.robots}
        at_prev = {c: i for i, c in prev_cells.items()}
        for i, c in cur.items():
            j = at_prev.get(c)
            if j is not None and j != i and cur[j] == prev_cells[i] and c != prev_cells[i]:
                same = sim.robots[i].task_id is not None and sim.robots[i].task_id == sim.robots[j].task_id
                if not same and i < j:
                    self.violations["S2"].append((sim.tick, i, j))
        # S3 dock exclusivity
        holders = defaultdict(list)
        charging = defaultdict(list)
        for r in sim.robots:
            if r.charger is not None:
                holders[r.charger].append(r.id)
            if r.charging:
                charging[r.charger].append(r.id)
                if r.cell != sim.chargers[r.charger].cell:
                    self.violations["S3"].append((sim.tick, "charging_off_dock", r.id))
        for cid, ids in holders.items():
            if len(ids) > 1:
                self.violations["S3"].append((sim.tick, "dock_held_by_several", cid, ids))
        for cid, ids in charging.items():
            if len(ids) > 1:
                self.violations["S3"].append((sim.tick, "several_charging", cid, ids))
        for c in sim.chargers:
            rb = getattr(c, "reserved_by", None)
            if rb is not None and sim.robots[rb].charger != c.id:
                self.violations["S3"].append((sim.tick, "stale_reservation", c.id, rb))
        # reported: charging approach (status CHARGE, not yet charging)
        for r in sim.robots:
            if r.status == RobotStatus.CHARGE and not r.charging:
                self.c["charge_approach_ticks"] += 1


def episode(name, seed, config, mode, out):
    cfg = scenario(name, config)
    env = bt.make(cfg, action_mode=mode) if mode == "primitive" else bt.make(cfg)
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    mon = Monitor(sim)
    coord = bt.GreedyCoalitionCoordinator("fifo") if mode == "macro" else bt.RandomCoordinator(seed=seed)
    env.reset(seed=seed)
    coord.reset(env)
    digests, charge_starts = [], 0
    limit = PRIMITIVE_TICKS if mode == "primitive" else None
    info = {}
    while True:
        prev = {r.id: r.cell for r in sim.robots}
        _, _, term, trunc, info = env.step(coord.act(env))
        if mode == "primitive":
            # primitive moves bypass PIBT: L1/L2 are macro-only, S1-S3 apply
            pass
        mon.after_tick(prev)
        charge_starts += sum(1 for e in info.get("events", [])
                             if (e.get("type") if isinstance(e, dict) else getattr(e, "type", None)) == "charge_started")
        digests.append(digest(sim))
        if term or trunc or (limit is not None and sim.tick >= limit):
            break
    m = {k: v for k, v in dict(info.get("metrics", {})).items() if isinstance(v, (int, float))}
    res = {"scenario": name, "seed": seed, "config": config, "mode": mode, "ticks": sim.tick,
           "metrics": m, "counts": dict(mon.c), "charge_starts": charge_starts,
           "max_no_progress": mon.max_streak, "max_oscillation": mon.max_osc,
           "violations": {k: v[:20] for k, v in mon.violations.items()},
           "n_violations": {k: len(v) for k, v in mon.violations.items()},
           "digests": digests}
    with open(out, "w") as f:
        json.dump(res, f)


def probe(out):
    """API probe: a second robot sent to a reserved dock, then to an occupied dock."""
    res = {}
    for model in ("2.0", "2.1"):
        cfg = copy.deepcopy(bt.get_scenario("basic"))
        cfg["model"] = model
        env = bt.make(cfg)
        env.reset(seed=0)
        sim = env.unwrapped.sim
        r0, r1, r2 = sim.robots[0], sim.robots[1], sim.robots[2]
        c = sim.chargers[0]
        a = sim.send_to_charger(r0.id, c.id)
        b = sim.send_to_charger(r1.id, c.id)                       # dock reserved by r0
        mask_b = int(sim.macro_action_mask(r1)[1 + sim.K + c.id])
        sim.release(r1.id) if r1.charger is not None else None
        r0.energy_wh = 0.5 * r0.spec.energy.capacity_wh
        occupied = [o for o in sim.robots if o.cell == c.cell and o.id != r0.id]
        for o in occupied:                                         # clear the dock cell for the probe
            o.cell = (o.cell[0] + 1, o.cell[1])
        r0.cell = c.cell
        env.step([0] * len(sim.robots))
        d = sim.send_to_charger(r2.id, c.id)                       # dock occupied by r0
        res[model] = {"first_send": list(a), "second_send_reserved": list(b), "mask_reserved": mask_b,
                      "r0_charging": r0.charging, "third_send_occupied": list(d)}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


def report(out_dir):
    import glob
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "ep_*.json")))]
    ref = {(r["scenario"], r["seed"], r["mode"]): r for r in rows if r["config"] == "REF"}
    gate = {"R0": [], "S1": [], "S1p": [], "S2": [], "S3": [], "L1": [], "L2": [], "L3": []}
    for r in rows:
        key = (r["scenario"], r["seed"], r["mode"])
        if r["config"] == "A" and key in ref:
            gate["R0"].append((key, r["digests"] == ref[key]["digests"]))
        twoone = r["config"] in ("B", "D")
        if twoone:
            for s in ("S1", "S1p", "S2", "S3"):
                gate[s].append((key + (r["config"],), r["n_violations"].get(s, 0)))
        if twoone and r["mode"] == "macro":
            for s in ("L1", "L2"):
                gate[s].append((key + (r["config"],), r["n_violations"].get(s, 0)))
    by = {(r["scenario"], r["seed"], r["config"]): r for r in rows if r["mode"] == "macro"}
    for (sc, sd, cf), r in by.items():
        base = {"B": "A", "D": "C"}.get(cf)
        if base and (sc, sd, base) in by:
            gate["L3"].append(((sc, sd, cf), r["metrics"].get("stranded", 0) - by[(sc, sd, base)]["metrics"].get("stranded", 0)))
    verdict = {}
    for k, v in gate.items():
        if k == "R0":
            verdict[k] = (all(ok for _, ok in v) and len(v) > 0, "%d/%d identical" % (sum(ok for _, ok in v), len(v)))
        elif k == "L3":
            bad = [x for x in v if x[1] > 0]
            verdict[k] = (not bad and len(v) > 0, "%d pairs, %d with more stranding" % (len(v), len(bad)))
        else:
            tot = sum(n for _, n in v)
            verdict[k] = (tot == 0 and len(v) > 0, "%d episodes, %d violations" % (len(v), tot))
    print("=== GATE ===")
    for k, (ok, msg) in verdict.items():
        note = "   (= S1a/S1b, POST HOC amendment frozen 2026-11-06)" if k == "S1p" else ""
        print("  %-3s %s  %s%s" % (k, "PASS" if ok else "FAIL", msg, note))
    print("\n=== Descriptive (macro, mean over seeds) ===")
    print("%-19s %-3s %7s %8s %8s %9s %9s %8s %8s %7s %6s" % ("scenario", "cfg", "tasks", "latency", "kWh", "stalled",
          "behind_L", "appr/chg", "moves/t", "maxNP", "strand"))
    for sc in SCENARIOS:
        for cf in ("A", "B", "C", "D"):
            rs = [by[(sc, s, cf)] for s in range(100) if (sc, s, cf) in by]
            if not rs:
                continue
            mean = lambda f: sum(f(r) for r in rs) / len(rs)
            print("%-19s %-3s %7.1f %8.1f %8.2f %9.0f %9.0f %8.1f %8.2f %7.0f %6.1f" % (
                sc, cf, mean(lambda r: r["metrics"].get("tasks_completed", 0)),
                mean(lambda r: r["metrics"].get("mean_latency", 0)), mean(lambda r: r["metrics"].get("energy_used_kwh", 0)),
                mean(lambda r: r["counts"].get("stalled", 0)), mean(lambda r: r["counts"].get("stalled_behind_lasting", 0)),
                mean(lambda r: r["counts"].get("charge_approach_ticks", 0) / max(1, r["charge_starts"])),
                mean(lambda r: r["metrics"].get("distance", 0) / max(1, r["metrics"].get("tasks_completed", 0))),
                max(r["max_no_progress"] for r in rs), mean(lambda r: r["metrics"].get("stranded", 0))))
    for k in ("S1", "S2", "S3", "L1", "L2"):
        bad = [r for r in rows if r["n_violations"].get(k)]
        for r in bad[:6]:
            print("  %s %s/%s/%s/%s: %s" % (k, r["scenario"], r["seed"], r["config"], r["mode"], r["violations"][k][:3]))
    return all(ok for k, (ok, _) in verdict.items() if k != "S1p")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "episode":
        episode(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])
    elif cmd == "probe":
        probe(sys.argv[2])
    elif cmd == "report":
        sys.exit(0 if report(sys.argv[2]) else 1)
