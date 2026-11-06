"""Order layer quick start: run an order scenario with one of the nine scheduling rules, print the
primary metrics and the diagnosis, and write a trace, CSV ledgers and the offline replay dashboard.

    python examples/orders_quickstart.py                      # generated orders (Poisson), PLB
    python examples/orders_quickstart.py examples/orders.yaml fifo
    python examples/orders_quickstart.py - nsp 3              # scenario '-' = generated, rule, seed

Outputs go to ./orders_out/: trace.json, orders.csv, tasks.csv, packages.csv, shelves.csv, robots.csv,
dashboard.html (open it in a browser).
"""
import json
import os
import sys

import battery_tarware as bt
from battery_tarware.orders.dashboard import write_dashboard
from battery_tarware.orders.run import run_episode
from battery_tarware.orders.scenarios import order_scenario
from battery_tarware.orders.trace import _json_default

args = sys.argv[1:]
path = args[0] if args else "-"
rule = args[1] if len(args) > 1 else "plb"
seed = int(args[2]) if len(args) > 2 else 0
scenario = order_scenario(source={"kind": "poisson", "rate": 0.09, "max_orders": 10 ** 6,
                                  "mix": {"outbound": 0.45, "inbound": 0.45, "relocation": 0.1},
                                  "priority": {"kind": "choice", "values": [1, 2, 3]},
                                  "due_after": {"kind": "uniform", "low": 300, "high": 800}},
                          inventory={"slot_capacity": 2, "package_fill": 0.5}) if path == "-" else path
res = run_episode(scenario, rule, seed, max_steps=1500 if path == "-" else None, warmup=150 if path == "-" else 0)
m = res["metrics"]
print("rule %s, seed %d, %d ticks" % (rule, seed, m["meta"]["horizon"]))
for k, v in m["primary"].items():
    print("  %-28s %s" % (k, v if not isinstance(v, float) else round(v, 3)))
print("task time by cause:", {k: round(v, 3) for k, v in m["bottlenecks"]["task_share_by_cause"].items()})
print("unsuccessful orders by category:", m["bottlenecks"]["unsuccessful_by_category"])
out = "orders_out"
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "trace.json"), "w") as f:
    json.dump(res["trace"], f, default=_json_default)
for name, rows in res["trace"]["tables"].items():
    bt.write_csv(rows, os.path.join(out, name + ".csv"))
print(write_dashboard([res["trace"]], os.path.join(out, "dashboard.html")))
