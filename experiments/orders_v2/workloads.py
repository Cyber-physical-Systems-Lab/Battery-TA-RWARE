"""Workloads of the second order-scheduling evaluation (model 2.2 and the expanded warehouses).

    build(base="intermediate", model="2.2", profile="standard", rate=None, battery="default")

base      basic | intermediate | advanced (battery_tarware.orders.scenarios.order_scenario)
model     "2.1" (the frozen model of the first study) or "2.2"
profile   simple   single-line inbound / outbound / relocation orders (the first study's structure)
          standard adds cross-dock (inbound -> outbound of that package) and replenishment
                   (relocation -> outbound from that shelf) orders, 1-3 lines, 30% sequenced
          dag      larger dependency graphs: 2-4 lines, 70% sequenced, half the orders compound
rate      orders per tick (default: RATE[base], the calibrated "loaded" level; x0.4 for dag)
battery   default | tight (every battery x0.12, half the docks: charging is scarce and contended)
substitution  intermediate only: role alternatives for the case support role (default on)
"""
import copy

from battery_tarware.orders.scenarios import order_scenario

RATE = {"basic": 0.09, "intermediate": 0.06, "advanced": 0.035}
MIX = {
    "simple": {"outbound": 0.45, "inbound": 0.45, "relocation": 0.10},
    "standard": {"outbound": 0.35, "inbound": 0.35, "relocation": 0.10, "cross_dock": 0.10, "replenish": 0.10},
    "dag": {"outbound": 0.25, "inbound": 0.25, "relocation": 0.0, "cross_dock": 0.30, "replenish": 0.20},
}
LINES = {"simple": {"kind": "fixed", "value": 1},
         "standard": {"kind": "choice", "values": [1, 2, 3], "weights": [0.6, 0.3, 0.1]},
         "dag": {"kind": "choice", "values": [2, 3, 4], "weights": [0.4, 0.4, 0.2]}}
SEQUENCE = {"simple": 0.0, "standard": 0.3, "dag": 0.7}
DAG_RATE_FACTOR = 0.4          # a dag order holds about 2.5x the tasks of a standard one


def build(base="basic", model="2.1", profile="standard", rate=None, battery="default", max_steps=3000,
          substitution=None):
    rate = RATE[base] * DAG_RATE_FACTOR if (rate is None and profile == "dag") else (RATE[base] if rate is None else rate)
    mix = {k: v for k, v in MIX[profile].items() if v > 0}
    src = {"kind": "poisson", "rate": float(rate), "max_orders": 10 ** 6, "mix": mix, "lines": LINES[profile],
           "priority": {"kind": "choice", "values": [1, 2, 3]},
           "due_after": {"kind": "uniform", "low": 400, "high": 1200}}
    if SEQUENCE[profile] > 0:
        src["sequence"] = SEQUENCE[profile]
    cfg = order_scenario(source=src, base=base, model=model, **({} if substitution is None else {"substitution": substitution}))
    cfg["max_steps"] = max_steps
    if battery == "tight":
        for rt in cfg["robot_types"]:
            rt["energy"] = {**rt["energy"], "capacity_wh": round(0.12 * rt["energy"]["capacity_wh"], 1)}
        for g in cfg["fleet"]:                       # per-group battery overrides scale too
            e = g.get("overrides", {}).get("energy")
            if e and "capacity_wh" in e:
                e["capacity_wh"] = round(0.12 * e["capacity_wh"], 1)
        m = copy.deepcopy(cfg["map"])
        if "generator" in m:
            m["generator"]["docks"] = max(2, m["generator"]["docks"] // 2)
        else:                                        # ascii map: keep every other dock
            rows, k = [], 0
            for row in m["ascii"].strip("\n").split("\n"):
                out = ""
                for ch in row:
                    if ch == "C":
                        out += "C" if k % 2 == 0 else "."
                        k += 1
                    else:
                        out += ch
                rows.append(out)
            m["ascii"] = "\n".join(rows)
        cfg["map"] = m
    cfg["name"] = "%s_%s_%s_%s_r%g%s" % (cfg["name"], profile, battery, model.replace(".", ""), rate,
                                       "_nosubst" if substitution is False else "")
    return cfg
