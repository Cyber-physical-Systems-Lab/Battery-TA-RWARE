"""Plug in your own coordinator through read-only state views (e.g. to drive it from C++).

``env.state_view()`` is a frozen snapshot of robots, tasks and chargers; ``.to_arrays()`` gives the
same content as contiguous fixed-dtype numpy arrays (schema version 1) that native code can read.
Decisions go back through actions or the validated simulator API.
"""
import os

import numpy as np

import battery_tarware as bt

STEPS = int(os.environ.get("EXAMPLE_STEPS", 800))


class NearestIdleCoordinator:
    """Toy policy: for each ready task, pick the nearest feasible crew; recharge below 25 %."""

    def reset(self, env):
        pass

    def act(self, env):
        view = env.state_view()                     # read-only; building it changes nothing
        sim = env.sim
        for r in view.robots:
            if r.status == "idle" and r.soc < 0.25:
                free = [c for c in view.chargers if c.occupant is None]
                if free:
                    c = min(free, key=lambda c: abs(c.cell[0] - r.cell[0]) + abs(c.cell[1] - r.cell[1]))
                    sim.send_to_charger(r.id, c.id)
        for t in view.tasks:
            if t.open_roles and t.state in ("ready", "assigned", "active", "suspended"):
                options = sim.feasible_coalitions(t.id, limit=1)
                if options:
                    sim.assign_coalition(t.id, dict(options[0].assignment))
        return np.zeros(len(view.robots), dtype=np.int64)


env = bt.make("advanced", max_steps=STEPS)
arrays = (env.reset(seed=0), env.state_view().to_arrays())[1]
print({k: (v.dtype.name, v.shape) for k, v in arrays.items()})
metrics = bt.run_episode(env, NearestIdleCoordinator(), seed=0)
print({k: metrics[k] for k in ("tasks_completed", "tasks_open", "stranded", "energy_used_kwh")})
