"""Protocol v4 extension of the tiled warehouse (RAL, 2026-10-22).

A versioned EXTENSION, not a change to the frozen benchmark. `WarehouseV4` with its defaults
(charger_ratio=1.0, arrivals=None) is the protocol-v2/v3 tiled warehouse exactly: no override acts.
Two optional factors:

charger_ratio < 1
    Every dock keeps its cell, so geometry, inventory, action ids and parking capacity are unchanged.
    Only a spatially balanced subset of docks CHARGES (`select_functional_chargers`).
    After every reset:
      * `charging_stations` / `_charging_station_positions` = the functional subset (what the
        coordinator routes charging robots to, and the only cells where CHARGE adds energy);
      * `parking_stations` = the non-charging docks;
      * `_dock_positions` = every dock.
    The coordinator's parking layer reads the last two (default-off branch in planner/coordinator.py),
    so idle robots park on non-charging docks first, and charger scarcity is not confounded with
    parking scarcity. The full dock list is restored BEFORE the base reset, because reset turns every
    non-dock, non-goal rack cell into a shelf.

charger_layout / charger_power (R07, 2026-10-29; default "balanced" / "uniform" = unchanged)
    Act only with charger_ratio < 1. "concentrated" takes the same number of functional chargers
    from ONE dock band (`select_concentrated_chargers`). "split" keeps the total charging power
    but gives the half nearest a goal 1.5x and the rest 0.5x the base rate (`split_charge_rates`,
    applied in `_execute_charge`). The coordinator sees only `charging_stations`, so policies are
    speed-unaware.

arrivals = ArrivalSpec(...)
    Replaces the replenished (saturated) queue with an external, seeded arrival schedule; delivered
    requests are not replenished. The schedule is a pure function of (spec, steps, seed):
    `make_arrival_schedule`.
      * N = round(rate x steps) arrivals in total, for every profile.
      * The k-th arrival has the same task type in every profile, and its time is the same uniform
        draw mapped through each profile's cumulative intensity.
    "constant" and "shift" therefore offer the same total demand, the same type sequence and the
    same order. Only the timing differs (common random numbers).

    "shift" alternates peak (1 + a) x rate and trough (1 - a) x rate in half-periods. The episode
    runs to max_steps even when the queue empties (the base env would stop then).

    Arrivals use their own RNGs (types, times, shelf choice, package weight), never the global numpy
    RNG. If no shelf is free, the arrival is DROPPED and counted.

    Recorded (env.arrival_log):
      * offered, arrived and dropped, by type;
      * the lead time (arrival -> delivery) of every delivered request;
      * the backlog, sampled every `backlog_every` steps;
      * the final backlog.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from battery_tarware.definitions import PACKAGE_REQUIREMENTS, PackageType
import battery_tarware.warehouse as _W
from battery_tarware.warehouse import Warehouse, _PKG_WEIGHT_MAX, _PKG_WEIGHT_MIN, _PKG_WEIGHT_MU, _PKG_WEIGHT_SIGMA

TYPES = (PackageType.SOLO, PackageType.STANDARD, PackageType.LARGE)
_WEIGHT_BIAS = {PackageType.SOLO: 0.5, PackageType.STANDARD: 1.0, PackageType.LARGE: 1.75}  # as in Warehouse


@dataclass(frozen=True)
class ArrivalSpec:
    profile: str = "constant"          # "constant" | "shift"
    rate: float = 0.2                  # mean arrivals per step over the episode
    period: int = 2880                 # shift: steps per peak+trough cycle (4 h at 5 s/step)
    amplitude: float = 0.5             # shift: peak (1+a)*rate, trough (1-a)*rate
    mix: Tuple[float, float, float] = (0.3, 0.4, 0.3)   # SOLO, STANDARD, LARGE
    backlog_every: int = 60

    def __post_init__(self):
        if self.profile not in ("constant", "shift"):
            raise ValueError(f"unknown arrival profile {self.profile!r}")
        if not 0.0 <= self.amplitude < 1.0:
            raise ValueError("amplitude must be in [0, 1)")
        if self.rate <= 0 or self.period < 2:
            raise ValueError("rate must be > 0 and period >= 2")
        if abs(sum(self.mix) - 1.0) > 1e-9:
            raise ValueError("mix must sum to 1")


def _cumulative_intensity(spec: ArrivalSpec, steps: int) -> np.ndarray:
    """Lambda(t) at t = 0..steps (relative units), piecewise linear."""
    t = np.arange(steps + 1, dtype=float)
    if spec.profile == "constant":
        return t
    half = spec.period / 2.0
    hi, lo = 1.0 + spec.amplitude, 1.0 - spec.amplitude
    full = np.floor(t / spec.period)
    r = t - full * spec.period
    return full * half * (hi + lo) + np.where(r < half, r * hi, half * hi + (r - half) * lo)


def make_arrival_schedule(spec: ArrivalSpec, steps: int, seed: int) -> List[Tuple[int, PackageType]]:
    """[(step, type)] sorted by step. A pure function of (spec, steps, seed). The type sequence and
    the uniform draws do not depend on the profile, so profiles differ in timing only."""
    n = int(round(spec.rate * steps))
    rng = np.random.default_rng([int(seed), 4004])
    u = np.sort(rng.random(n))
    types = rng.choice(len(TYPES), size=n, p=np.asarray(spec.mix, dtype=float))
    lam = _cumulative_intensity(spec, steps)
    target = u * lam[-1]
    idx = np.searchsorted(lam, target, side="right") - 1          # Lambda(idx) <= target < Lambda(idx+1)
    idx = np.clip(idx, 0, steps - 1)
    return [(int(s), TYPES[int(k)]) for s, k in zip(idx, types)]


def select_functional_chargers(docks: List, ratio: float, grid_columns: Optional[int] = None) -> List:
    """Spatially balanced subset of docks that charge.

    k = max(1, round(ratio x len(docks))). Docks are grouped by rack column (x, plus the band of rows
    for the tile row). Each column gets floor(k/B) of its docks, and the k mod B extra docks go to
    columns evenly spaced over the column order. Within a column the top-most docks are chosen
    (nearest the cross-aisle). Deterministic; ratio >= 1 returns every dock."""
    if ratio >= 1.0:
        return list(docks)
    k = max(1, int(round(ratio * len(docks))))
    blocks: Dict[Tuple[int, int], List] = {}
    ys = sorted({d.y for d in docks})
    # tile-row band: docks of one block are contiguous rows; split bands at row gaps > 1
    band_of, band = {}, 0
    for i, y in enumerate(ys):
        if i and y - ys[i - 1] > 1:
            band += 1
        band_of[y] = band
    for d in docks:
        blocks.setdefault((band_of[d.y], d.x), []).append(d)
    keys = sorted(blocks)
    for key in keys:
        blocks[key].sort(key=lambda d: d.y)
    B = len(keys)
    base, extra = divmod(k, B)
    extra_idx = {int(math.floor((i + 0.5) * B / extra)) for i in range(extra)} if extra else set()
    chosen = []
    for i, key in enumerate(keys):
        q = min(len(blocks[key]), base + (1 if i in extra_idx else 0))
        chosen.extend(blocks[key][:q])
    return chosen


def select_concentrated_chargers(docks: List, ratio: float) -> List:
    """R07 'concentrated' layout: the same number of functional chargers as select_functional_chargers,
    k = max(1, round(ratio x len(docks))), all taken from ONE dock band, the top one (smallest y),
    in (y, x) order. Deterministic. A band holds 24 docks on the 3x5 map, so any k <= 24 fits."""
    k = max(1, int(round(ratio * len(docks))))
    ys = sorted({d.y for d in docks})
    band, b = {}, 0
    for i, y in enumerate(ys):
        if i and y - ys[i - 1] > 1:
            b += 1
        band[y] = b
    top = sorted((d for d in docks if band[d.y] == 0), key=lambda d: (d.y, d.x))
    if k > len(top):
        raise ValueError(f"concentrated layout needs k <= {len(top)} docks in one band, got {k}")
    return top[:k]


def split_charge_rates(stations: List, goals) -> Dict:
    """R07 'split' power: the same TOTAL charging power as uniform. The half of the functional chargers
    nearest a delivery goal (Manhattan; ties by (y, x)) charge at 1.5 x the base rate, the other half at
    0.5 x. Fixed a priori: deliveries end at goals."""
    base = _W._BATTERY_CHARGE_RATE
    order = sorted(stations, key=lambda s: (min(abs(s.x - gx) + abs(s.y - gy) for gx, gy in goals), s.y, s.x))
    fast = len(order) // 2
    return {(s.x, s.y): (1.5 if i < fast else 0.5) * base for i, s in enumerate(order)}


class WarehouseV4(Warehouse):
    """Tiled warehouse with the protocol-v4 factors; identical to Warehouse at its defaults."""

    def __init__(self, *args, charger_ratio: float = 1.0, arrivals: Optional[ArrivalSpec] = None,
                 arrival_seed_offset: int = 0, charger_layout: str = "balanced",
                 charger_power: str = "uniform", **kwargs):
        if arrivals is not None:
            kwargs["replenish_requests"] = False
        super().__init__(*args, **kwargs)
        if not 0.0 < float(charger_ratio) <= 1.0:
            raise ValueError("charger_ratio must be in (0, 1]")
        self.charger_ratio = float(charger_ratio)
        if charger_layout not in ("balanced", "concentrated") or charger_power not in ("uniform", "split"):
            raise ValueError("charger_layout in {balanced, concentrated}; charger_power in {uniform, split}")
        if (charger_layout, charger_power) != ("balanced", "uniform") and self.charger_ratio >= 1.0:
            raise ValueError("charger_layout / charger_power act only on a functional subset (charger_ratio < 1)")
        self.charger_layout, self.charger_power = charger_layout, charger_power
        self._charge_rate_at: Dict = {}
        self.arrivals = arrivals
        self.arrival_seed_offset = int(arrival_seed_offset)
        self._all_docks = list(self.charging_stations)
        self._all_dock_positions = {(s.x, s.y) for s in self._all_docks}
        self.arrival_log: Dict = {}

    # ── charger subset ────────────────────────────────────────────────────────
    def _apply_charger_ratio(self) -> None:
        # With infinite energy, chargers have no function, so every ratio is the same warehouse.
        # T_inf is then one ceiling per (composition, demand), independent of rho, by construction.
        if self.charger_ratio >= 1.0 or not self.battery_drain:
            return
        if self.charger_layout == "concentrated":
            functional = select_concentrated_chargers(self._all_docks, self.charger_ratio)
        else:
            functional = select_functional_chargers(self._all_docks, self.charger_ratio)
        fset = {(s.x, s.y) for s in functional}
        self._charge_rate_at = split_charge_rates(functional, self.goals) if self.charger_power == "split" else {}
        self.charging_stations = functional
        self._charging_station_positions = set(fset)
        self.parking_stations = [d for d in self._all_docks if (d.x, d.y) not in fset]
        self._dock_positions = set(self._all_dock_positions)

    # ── per-charger power (R07; default uniform = the base rule) ─────────────
    def _execute_charge(self, agent) -> None:
        if not self._charge_rate_at:
            return super()._execute_charge(agent)
        rate = self._charge_rate_at.get((agent.x, agent.y))
        if rate is not None:
            agent.battery = min(_W._BATTERY_FULL, agent.battery + rate)

    # ── arrivals ──────────────────────────────────────────────────────────────
    def _start_arrivals(self, seed: Optional[int]) -> None:
        s = (0 if seed is None else int(seed)) + self.arrival_seed_offset
        steps = int(self.max_steps)
        self._schedule = make_arrival_schedule(self.arrivals, steps, s)
        self._sched_pos = 0
        self._shelf_rng = np.random.default_rng([s, 4005])
        self._arrival_step: Dict[int, int] = {}
        self.request_queue = []
        z = {t.name: 0 for t in TYPES}
        self.arrival_log = {"offered": dict(z), "arrived": dict(z), "dropped": dict(z),
                            "lead_times": [], "lead_types": [], "backlog": [], "final_backlog": None,
                            "spec": self.arrivals.__dict__ | {"mix": list(self.arrivals.mix)}, "seed": s}
        self._enqueue_due(0)

    def _assign_package_v4(self, shelf, package_type: PackageType) -> None:
        shelf.package_type = package_type
        shelf.required_agvs, shelf.required_pickers = PACKAGE_REQUIREMENTS[package_type]
        w = self._shelf_rng.normal(_PKG_WEIGHT_MU * _WEIGHT_BIAS[package_type], _PKG_WEIGHT_SIGMA)
        shelf.weight = float(np.clip(w, _PKG_WEIGHT_MIN, _PKG_WEIGHT_MAX))

    def _enqueue_due(self, t: int) -> None:
        sched, log = self._schedule, self.arrival_log
        while self._sched_pos < len(sched) and sched[self._sched_pos][0] <= t:
            _, ptype = sched[self._sched_pos]
            self._sched_pos += 1
            log["offered"][ptype.name] += 1
            requested = {s.id for s in self.request_queue}
            carried = {a.carrying_shelf.id for a in self.agents if a.carrying_shelf}
            free = [s for s in self.shelfs if s.id not in requested and s.id not in carried]
            if not free:
                log["dropped"][ptype.name] += 1
                continue
            shelf = free[int(self._shelf_rng.integers(len(free)))]
            self._assign_package_v4(shelf, ptype)
            self.request_queue.append(shelf)
            self._arrival_step[shelf.id] = t
            log["arrived"][ptype.name] += 1

    # ── gym API ───────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.charger_ratio < 1.0 and self.battery_drain:
            # restore every dock first: the base reset makes every non-dock rack cell a shelf
            self.charging_stations = list(self._all_docks)
            self._charging_station_positions = set(self._all_dock_positions)
        out = super().reset(seed=seed, options=options)
        self._apply_charger_ratio()
        if self.arrivals is not None:
            self._start_arrivals(seed)
        return out

    def step(self, action):
        if self.arrivals is None:
            return super().step(action)
        before = {s.id: s for s in self.request_queue}
        done, info = super().step(action)
        t = self._cur_steps
        log = self.arrival_log
        left = set(before) - {s.id for s in self.request_queue}       # requests leave only by delivery
        for sid in left:
            log["lead_times"].append(t - self._arrival_step.pop(sid, t))
            log["lead_types"].append(before[sid].package_type.name)
        self._enqueue_due(t)
        if t % self.arrivals.backlog_every == 0:
            log["backlog"].append(len(self.request_queue))
        done = bool((self.max_steps and t >= self.max_steps)
                    or (self.max_inactivity_steps and self._cur_inactive_steps >= self.max_inactivity_steps))
        if done:
            log["final_backlog"] = len(self.request_queue)
        return done, info


def make_tiled_warehouse_v4(*, charger_ratio: float = 1.0, arrivals: Optional[ArrivalSpec] = None,
                            **kwargs) -> WarehouseV4:
    """make_tiled_warehouse with the protocol-v4 factors (same defaults and argument handling)."""
    from battery_tarware import DEFAULT_PACKAGE_MIX
    tile_rows, tile_cols = kwargs.pop("tile_rows"), kwargs.pop("tile_cols")
    queue = kwargs.pop("request_queue_size", None)
    pd = kwargs.pop("package_distribution", None)
    return WarehouseV4(
        shelf_columns=tile_cols, column_height=kwargs.pop("column_height", 16), shelf_rows=tile_rows,
        num_agvs=kwargs.pop("num_agvs"), num_pickers=kwargs.pop("num_pickers"),
        request_queue_size=queue if queue is not None else 4 * tile_rows * tile_cols,
        max_inactivity_steps=kwargs.pop("max_inactivity_steps", 100), max_steps=kwargs.pop("max_steps", 5000),
        package_distribution=pd if pd is not None else dict(DEFAULT_PACKAGE_MIX), layout="tiled",
        rack_width=kwargs.pop("rack_width", 1), lane_width=kwargs.pop("lane_width", 2),
        replenish_requests=kwargs.pop("replenish_requests", True),
        initial_battery_profile=kwargs.pop("initial_battery_profile", "legacy"),
        initial_battery_low=kwargs.pop("initial_battery_low", 20.0),
        initial_battery_high=kwargs.pop("initial_battery_high", 100.0),
        battery_drain=kwargs.pop("battery_drain", True),
        picker_energy_factor=kwargs.pop("picker_energy_factor", 1.25),
        charger_ratio=charger_ratio, arrivals=arrivals, **kwargs)
