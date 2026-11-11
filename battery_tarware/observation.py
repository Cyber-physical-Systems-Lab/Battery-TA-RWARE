"""Per-robot observations with heterogeneous information.

Robot i sees robot j if j is within i's ``sensing_range`` (Chebyshev), or if some robot k that
sees j is within communication range of i (both robots' ``comm_range`` must cover the distance;
one relay hop). Positions a robot observes (its own included) carry Gaussian noise with that
robot's ``localization_noise`` (cells). The task board is global unless
``observation.task_visibility == "sensing"``.

All features are float32 in [-1, 1] (positions by map size, energy as state of charge, mass by the
largest payload in the scenario, time by ``max_steps``), except per-capability role counts, which are
bounded by the largest crew in the scenario.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from gymnasium import spaces

from battery_tarware.robots import RobotStatus

STATUSES = list(RobotStatus)


class ObservationBuilder:
    def __init__(self, sim):
        self.sim = sim
        cfg = sim.config
        self.H, self.W = sim.layout.shape
        self.caps = list(sim.capabilities)
        self.types = list(sim.type_names)
        self.ttypes = list(sim.task_type_names)
        self.N, self.K, self.C = len(sim.robots), sim.K, sim.C
        specs = [r.spec for r in sim.robots]
        self.max_payload = max([s.payload_kg for s in specs] + [t.weight.max_value for t in cfg.task_types] + [1.0])
        self.max_capacity = max(s.energy.capacity_wh for s in specs)
        self.max_power = max([c.power_w for c in sim.chargers_template] + [1.0])
        self.max_priority = max([t.priority for t in cfg.task_types] + [1.0])
        self.max_handling = max(s.handling_speed for s in specs)
        self.high = float(max([1] + [r.count for t in cfg.task_types for r in t.roles]
                              + [sum(r.count for r in t.roles) for t in cfg.task_types]))
        self.T = float(cfg.max_steps)
        self.n_actions = sim.n_macro_actions if cfg.action_mode == "macro" else sim.n_primitive_actions
        nt, nc, ns, ntt = len(self.types), len(self.caps), len(STATUSES), len(self.ttypes)
        self.F_SELF = 2 + 6 + ns + 2 + nt + nc
        self.F_ROBOT = 1 + 2 + 2 + ns + nt
        self.F_TASK = 2 + 1 + 6 + 3 + 2 + ntt + 2 * nc
        self.F_CHARGER = 6

    # ---- spaces
    def robot_space(self) -> spaces.Dict:
        # every feature is normalized to [-1, 1] except role counts (up to the largest crew)
        f = lambda *shape: spaces.Box(-1.0, self.high, shape=shape, dtype=np.float32)
        return spaces.Dict({
            "self": f(self.F_SELF),
            "robots": f(self.N, self.F_ROBOT),
            "tasks": f(self.K, self.F_TASK),
            "chargers": f(max(1, self.C), self.F_CHARGER),
            "action_mask": spaces.MultiBinary(self.n_actions),
        })

    # ---- visibility
    def visibility(self) -> np.ndarray:
        """vis[i, j] = robot i currently observes robot j."""
        sim = self.sim
        P = np.array([r.cell for r in sim.robots])
        D = np.maximum(np.abs(P[:, None, 0] - P[None, :, 0]), np.abs(P[:, None, 1] - P[None, :, 1]))
        sense = np.array([np.inf if r.spec.sensing_range is None else r.spec.sensing_range for r in sim.robots])
        comm = np.array([np.inf if r.spec.comm_range is None else r.spec.comm_range for r in sim.robots])
        direct = D <= sense[:, None]
        talk = D <= np.minimum(comm[:, None], comm[None, :])
        alive = np.array([r.status != RobotStatus.STRANDED for r in sim.robots])
        talk &= alive[:, None] & alive[None, :]
        vis = direct | ((talk.astype(np.int32) @ direct.astype(np.int32)) > 0)
        np.fill_diagonal(vis, True)
        return vis

    # ---- build
    def build(self, masks: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        sim = self.sim
        rng = sim.rng["obs"]
        vis = self.visibility()
        slots = sim.task_slots()
        out = {}
        tick = sim.tick
        for i, r in enumerate(sim.robots):
            sigma = r.spec.localization_noise

            def pos(cell):
                y, x = float(cell[0]), float(cell[1])
                if sigma > 0:
                    y, x = y + rng.normal(0, sigma), x + rng.normal(0, sigma)
                return (min(1.0, max(0.0, y / max(1, self.H - 1))), min(1.0, max(0.0, x / max(1, self.W - 1))))

            st = np.zeros(len(STATUSES), np.float32)
            st[STATUSES.index(r.status)] = 1
            y, x = pos(r.cell)
            t = sim.tasks.get(r.task_id) if r.task_id is not None else None
            selfv = np.concatenate([
                [y, x],
                [r.soc, r.spec.speed, r.spec.payload_kg / self.max_payload, r.spec.handling_speed / self.max_handling,
                 r.spec.energy.capacity_wh / self.max_capacity, float(r.charging)],
                st,
                [float(sim.carrying(r)), float(t is not None)],
                np.eye(len(self.types), dtype=np.float32)[r.type_index],
                [float(c in r.spec.capabilities) for c in self.caps],
            ]).astype(np.float32)

            robots = np.zeros((self.N, self.F_ROBOT), np.float32)
            for j, o in enumerate(sim.robots):
                if not vis[i, j]:
                    continue
                oy, ox = pos(o.cell)
                s2 = np.zeros(len(STATUSES), np.float32)
                s2[STATUSES.index(o.status)] = 1
                robots[j] = np.concatenate([[1.0, oy, ox, o.soc, float(sim.carrying(o))], s2,
                                            np.eye(len(self.types), dtype=np.float32)[o.type_index]])

            tasks = np.zeros((self.K, self.F_TASK), np.float32)
            for k, tk in enumerate(slots):
                if tk is None:
                    continue
                visible = sim.task_visible(r, tk)
                if not visible:
                    tasks[k, 0] = 1.0
                    continue
                oy, ox = tk.origin[0] / max(1, self.H - 1), tk.origin[1] / max(1, self.W - 1)
                dy, dx = tk.destination[0] / max(1, self.H - 1), tk.destination[1] / max(1, self.W - 1)
                sy, sx = tk.stage_cell[0] / max(1, self.H - 1), tk.stage_cell[1] / max(1, self.W - 1)
                left = 1.0 if tk.deadline is None else max(-1.0, min(1.0, (tk.deadline - tick) / self.T))
                open_r = tk.joinable_roles()
                open_caps = [sum(n for role, n in open_r.items() if c in tk.type.role(role).capabilities) for c in self.caps]
                req = tk.required
                tasks[k] = np.concatenate([
                    [1.0, 1.0, min(1.0, tk.weight_kg / self.max_payload), oy, ox, dy, dx, sy, sx,
                     tk.stage / len(tk.type.stages), min(1.0, tk.age(tick) / self.T), left,
                     tk.type.priority / self.max_priority, float(r.task_id == tk.id)],
                    np.eye(len(self.ttypes), dtype=np.float32)[self.ttypes.index(tk.type.name)],
                    open_caps, [req.get(c, 0) for c in self.caps],
                ])

            chargers = np.zeros((max(1, self.C), self.F_CHARGER), np.float32)
            for c in sim.chargers:
                chargers[c.id] = [c.cell[0] / max(1, self.H - 1), c.cell[1] / max(1, self.W - 1),
                                  c.power_w / self.max_power,
                                  float(c.occupant is not None if sim.config.model == "2.0"
                                        else c.occupant not in (None, r.id) or c.reserved_by not in (None, r.id)),
                                  float(r.compatible_with(c.type)), float(r.charger == c.id)]
            out[r.name] = {"self": selfv, "robots": robots, "tasks": tasks, "chargers": chargers,
                           "action_mask": masks[i].astype(np.int8)}
        return out
