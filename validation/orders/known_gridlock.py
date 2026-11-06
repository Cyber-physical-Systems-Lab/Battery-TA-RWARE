"""Reproduce the model 2.1 hold-and-wait gridlock found in the order-scheduling evaluation.

    python validation/orders/known_gridlock.py

Base order scenario (experiments/orders/factors.py), rule PLB, seed 1017: around tick 150 a
`large` task's return stage needs two picking robots beside its shelf. One picking robot works
from an adjacent storage cell that is another task's work cell, whose loaded carrier then waits for
that cell; that crew and other loaded AGVs block the 2-wide aisle the second picking robot needs.
No robot of these crews moves again. Model 2.1's admission rule keeps *carriers* off shared work
cells but not helpers working from an adjacent cell. Also seen: rate=0.05, PLB, seed 6. No other
rule produced it in 1480 evaluation runs. Model 2.1 is frozen, so this is documented, not changed;
a model fix should make this script report no stall.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "orders"))
import factors  # noqa: E402
from battery_tarware.orders.run import run_episode  # noqa: E402

res = run_episode(factors.build(), "plb", 1017, 600, 300)
stalls = [e for e in res["sim"].events.events if e.type == "stall_suspected"]
for e in stalls:
    d = e.data
    print("tick %d  %s  %s  %s  last progress %d" % (e.tick, d["task_id"], d["wait_reason"], d["blocking_resource"],
                                                     d["last_progress_tick"]))
    for c in d["snapshot"]["crew"]:
        print("     %-16s %-9s at %s -> %s" % (c["name"], c["role"], tuple(c["cell"]), tuple(c["target"])))
print("GRIDLOCK REPRODUCED" if stalls else "no stall (gridlock not reproduced)")
