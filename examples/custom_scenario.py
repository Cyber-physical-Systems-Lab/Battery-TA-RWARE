"""Load a custom scenario (YAML), inspect feasible coalitions, and run the greedy baseline."""
import os
from pathlib import Path

import battery_tarware as bt

STEPS = int(os.environ.get("EXAMPLE_STEPS", 2000))
path = Path(__file__).with_name("custom_scenario.yaml")

env = bt.make(str(path), max_steps=STEPS)
obs, info = env.reset(seed=1)
sim = env.sim
print(sim.layout.to_ascii())
for r in sim.robots:
    print("%-9s %-24s speed %.1f payload %5.0f kg battery %5.0f Wh soc %.2f" % (
        r.name, ",".join(r.capabilities), r.spec.speed, r.spec.payload_kg, r.spec.energy.capacity_wh, r.soc))

# advance until a joint-transport task appears, then ask the environment who can do it
for _ in range(STEPS):
    env.step([0] * len(env.robot_names))
    carts = [t for t in sim.open_tasks() if t.type.name == "sterile_cart"]
    if carts:
        t = carts[0]
        print("\ntask %d (%s, %.0f kg) feasible coalitions:" % (t.id, t.type.name, t.weight_kg))
        for c in sim.feasible_coalitions(t.id, limit=3):
            names = {role: [sim.robots[i].name for i in ids] for role, ids in c.assignment.items()}
            print("  %s  ~%.0f ticks, carriers lift %.0f kg" % (names, c.est_ticks, c.payload_capacity_kg))
        break

metrics = bt.run_episode(bt.make(str(path), max_steps=STEPS), bt.GreedyCoalitionCoordinator("priority"), seed=1)
print("\ngreedy baseline:", {k: metrics[k] for k in ("tasks_completed", "late_tasks", "stranded", "energy_used_kwh")})
