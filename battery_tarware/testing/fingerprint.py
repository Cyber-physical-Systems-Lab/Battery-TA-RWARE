"""Per-tick trajectory fingerprints (determinism / regression tool).

Identical in definition to RendeCoord's G0 tool (``sh/bt_g0_trajectories.py`` at tag
``bt-g0-baseline``), so digests are comparable across the two repositories. After every
``env.step`` the recorder hashes an explicit canonical projection of simulator state, the raw occupancy
grid, ``(done, info)``, and the state of both global RNG streams (``np.random`` and ``random``), so an
extra or reordered random draw is detected. Hashing draws no random numbers and mutates nothing.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import List, Optional

import numpy as np

FINGERPRINT_VERSION = "bt-g0-v1"


def _f(x):
    return None if x is None else repr(float(x))


def _e(x):
    if x is None:
        return None
    return int(getattr(x, "value", x))


def projection(env) -> dict:
    agents = [(_e(a.id), _e(a.type), _e(a.x), _e(a.y), _e(a.dir), _f(a.battery), bool(a.busy), bool(a.charging),
               _e(a.carrying_shelf.id) if a.carrying_shelf else 0,
               _e(a.req_action), _e(a.target), _e(a.fixing_clash),
               None if a.path is None else [tuple(map(int, p)) for p in a.path],
               bool(a.has_delivered), int(a.total_lifts)) for a in env.agents]
    shelves = [(_e(s.id), _e(s.x), _e(s.y), _e(s.package_type), _f(s.weight), _e(s.required_agvs),
                _e(s.required_pickers)) for s in env.shelfs]
    st = {"t": int(env._cur_steps), "inactive": int(env._cur_inactive_steps), "agents": agents,
          "shelves": shelves, "queue": [_e(s.id) for s in env.request_queue],
          "chargers": [(_e(c.x), _e(c.y)) for c in env.charging_stations]}
    log = getattr(env, "arrival_log", None)
    if log:
        st["arrival"] = {k: (len(v) if isinstance(v, list) else v) for k, v in sorted(log.items())}
    return st


def rng_digest() -> str:
    s = np.random.get_state()
    h = hashlib.sha256()
    h.update(s[1].tobytes())
    h.update(repr((s[0], s[2], s[3], s[4])).encode())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()[:16]


class TrajectoryRecorder:
    """Wraps ``env.step`` on the instance (the class is untouched) and records per-tick digests."""

    def __init__(self, env):
        self.env = env
        self.chain = hashlib.sha256(FINGERPRINT_VERSION.encode())
        self.ticks: List[str] = []
        self.reset_digest = self._digest(None)

    def _digest(self, step_out) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(projection(self.env), sort_keys=True).encode())
        h.update(self.env.grid.tobytes())
        if step_out is not None:
            done, info = step_out
            h.update(repr(bool(done)).encode())
            h.update(json.dumps(info, sort_keys=True, default=lambda o: repr(o)).encode())
        h.update(rng_digest().encode())
        return h.hexdigest()[:16]

    def wrap(self) -> "TrajectoryRecorder":
        orig = self.env.step

        def step(action):
            out = orig(action)
            d = self._digest(out)
            self.ticks.append(d)
            self.chain.update(d.encode())
            return out
        self.env.step = step
        return self

    def hexdigest(self) -> str:
        return self.chain.hexdigest()


def random_target_driver(env, seed: int, steps: int, recorder: Optional[TrajectoryRecorder] = None) -> int:
    """The G0 env-only driver: a private ``default_rng([seed, 9090])`` picks macro actions (requested
    shelves for empty AGVs and pickers, goals for loaded robots, otherwise any location). It never
    touches the global RNG streams. Returns the number of deliveries."""
    drv = np.random.default_rng([seed, 9090])
    coord_to_id = {tuple(map(int, v)): int(k) for k, v in env.action_id_to_coords_map.items()}
    goal_ids = [coord_to_id[(int(g[1]), int(g[0]))] for g in env.goals if (int(g[1]), int(g[0])) in coord_to_id]
    all_ids = sorted(coord_to_id.values())
    deliveries = 0
    for _ in range(steps):
        req_ids = [coord_to_id[(s.y, s.x)] for s in env.request_queue if (s.y, s.x) in coord_to_id]
        acts = []
        for a in env.agents:
            u = drv.random()
            if u < 0.2 or a.busy:
                acts.append(0)
            elif a.type.name == "AGV" and a.carrying_shelf is None and req_ids and u < 0.8:
                acts.append(req_ids[int(drv.integers(len(req_ids)))])
            elif a.carrying_shelf is not None and goal_ids and u < 0.8:
                acts.append(goal_ids[int(drv.integers(len(goal_ids)))])
            elif a.type.name == "PICKER" and req_ids and u < 0.7:
                acts.append(req_ids[int(drv.integers(len(req_ids)))])
            else:
                acts.append(all_ids[int(drv.integers(len(all_ids)))])
        done, info = env.step(acts)
        deliveries += int(info["shelf_deliveries"])
        if done:
            break
    return deliveries
