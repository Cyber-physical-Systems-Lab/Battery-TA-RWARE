"""Order layer: orders, task groups, order-layer tasks, inventory, readiness, retries.

The manager runs inside ``Simulator.step`` (after the simulator's own phases) when the scenario has
an ``orders:`` key, and never otherwise. Each tick it

1. commits inventory changes from this tick's simulator events (pick-up, drop, re-pick, hand-off
   at the station, placement);
2. mirrors the simulator state of every dispatched attempt (ready, assigned, executing, suspended,
   completed) into the order-layer task;
3. runs the progress monitor; a suspected stall is recorded and, with ``auto_retry``, fails the
   attempt (only while the payload has not left its origin; otherwise the simulator's own recovery
   continues);
4. accepts arriving orders (trace, Poisson generator, or ``submit``), creating their tasks and
   taking reservations;
5. checks readiness of pending, blocked and retrying tasks (dependencies, inventory, payload and
   access) and dispatches ready ones as simulator tasks. A blocked task is never dispatched, so no
   robot can be assigned to it.

Ids: orders ``O00000``, task groups ``G00000``, order-layer tasks ``T00000`` (persist across
dispatch attempts; each attempt is one simulator task id), packages ``P00000``, shelves ``S0000``,
rack locations ``L0000``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from battery_tarware.config import ConfigError
from battery_tarware.navigation import astar
from battery_tarware.orders.inventory import Inventory, InventoryError
from battery_tarware.orders.lifecycle import FROM_SIM, OTaskState, check_transition
from battery_tarware.orders.monitor import ProgressMonitor
from battery_tarware.orders.spec import COMPOUND, OrderSpec, handoff_stage
from battery_tarware.tasks.graph import TaskGraph, TaskGraphError

Cell = Tuple[int, int]
S = OTaskState


@dataclass
class Order:
    order_id: str
    kind: str
    arrival: int
    priority: float
    due: Optional[int]
    source: str
    task_ids: List[str] = field(default_factory=list)
    group_ids: List[str] = field(default_factory=list)
    state: str = "open"                          # open | completed | failed | cancelled
    reason: Optional[str] = None
    closed: Optional[int] = None


@dataclass
class TaskGroup:
    group_id: str
    order_id: str
    key: str
    task_ids: List[str] = field(default_factory=list)


@dataclass
class FailureRecord:
    tick: int
    attempt: int
    sim_task: Optional[int]
    failure_reason: str
    blocking_resource: Optional[str]
    last_progress_tick: Optional[int]
    snapshot: Optional[Dict]


@dataclass
class OrderTask:
    task_id: str
    order_id: str
    group_id: str
    key: str
    op: str
    created: int
    depends_on: Tuple[str, ...] = ()
    package_id: Optional[str] = None
    shelf_id: Optional[str] = None
    station: Optional[Cell] = None
    dest_location: Optional[str] = None          # relocation target
    explicit_dest: bool = False
    shelf_ref: Optional[str] = None              # relocation of "the shelf holding task X's package"
    state: OTaskState = S.PENDING
    reason: Optional[str] = None
    attempt: int = 0                             # dispatches so far
    sim_task: Optional[int] = None               # the attached (current) attempt
    attempts: List[int] = field(default_factory=list)
    history: List[Tuple[int, str, str, Optional[str]]] = field(default_factory=list)
    failures: List[FailureRecord] = field(default_factory=list)
    retry_at: Optional[int] = None
    final: bool = False                          # failed with no retry left
    task_type: Optional[str] = None
    weight_kg: Optional[float] = None
    first_ready: Optional[int] = None
    ready_since: Optional[int] = None            # last time it (or one of its roles) became open for recruitment
    completed_at: Optional[int] = None
    stalls: int = 0
    waits: Dict[str, int] = field(default_factory=dict)
    _hp: int = 0                                 # simulator history already mirrored
    _stages_done: int = 0

    @property
    def terminal(self) -> bool:
        return self.state in (S.COMPLETED, S.CANCELLED) or (self.state == S.FAILED and self.final)


class OrderManager:
    def __init__(self, sim):
        self.sim = sim
        self.spec: OrderSpec = OrderSpec.from_dict(sim.config.orders, sim.config)
        self.storage = sorted(sim.layout.cells("storage"))
        if not self.storage:
            raise ConfigError("orders: the map has no storage cells (rack locations)")
        self.stations: Dict[str, List[Cell]] = {}
        for op, o in self.spec.operations.items():
            if o.station is not None:
                cells = sorted(sim.layout.cells(o.station))
                if not cells:
                    raise ConfigError("orders.operations.%s: the map has no %r cells" % (op, o.station))
                self.stations[op] = cells
        self._reach: Dict[Tuple, bool] = {}

    # ════════════════════════════════════════════════════════════ episode
    def reset(self, seed: Optional[int]) -> None:
        sim = self.sim
        self.rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(6)[5])
        self.orders: Dict[str, Order] = {}
        self.groups: Dict[str, TaskGroup] = {}
        self.tasks: Dict[str, OrderTask] = {}
        self.by_sim: Dict[int, str] = {}
        self._queue: List[Tuple[int, int, Dict, str]] = []
        self._seq = 0
        self._generated = 0
        self.monitor = ProgressMonitor(sim, self.spec.lifecycle.stall_ticks)
        self.inventory = Inventory(self.storage, self.spec.inventory.slot_capacity, self.spec.inventory.shelf_kg,
                                   lambda op, **d: self._emit("inventory", op=op, **d))
        self.inventory.populate(self.rng, self.spec.inventory, sim.tick)
        for o in self.spec.source.orders:
            self._enqueue(dict(o), "trace")
        self._accept_arrivals()
        self._advance()
        self._cursor = len(sim.events)

    def step(self) -> None:
        sim = self.sim
        for e in sim.events.since(self._cursor):
            self._on_sim_event(e)
        self._mirror()
        attached = [t.sim_task for t in self.tasks.values() if t.sim_task is not None]
        for rep in self.monitor.update(attached):
            self._on_stall(rep)
        self._accept_arrivals()
        self._advance()
        if self.spec.lifecycle.check_inventory:
            self.inventory.check()
        self._cursor = len(sim.events)

    def finished(self) -> bool:
        src = self.spec.source
        if src.kind == "poisson" and (src.max_orders is None or self._generated < src.max_orders):
            return False
        if self._queue:
            return False
        return all(o.state != "open" for o in self.orders.values()) and \
            all(t.sim_task is None for t in self.tasks.values())

    # ════════════════════════════════════════════════════════════ public API
    def submit(self, order: Mapping[str, Any]) -> None:
        """Queue an order (dict as in a trace); ``arrival`` defaults to the current tick. Orders
        arriving now are accepted at the end of the current (or, between steps, the next) tick."""
        self._enqueue(dict(order), "submitted")

    def fail_task(self, task_id: str, reason: str, blocking_resource: Optional[str] = None) -> Tuple[bool, str]:
        """Fail the current attempt of a task (e.g. a coordinator reports repeated path-planning
        failure). Allowed while its payload has not left the origin."""
        ot = self.tasks[task_id]
        if ot.sim_task is None:
            return False, "task %s has no dispatched attempt (state %s)" % (task_id, ot.state.value)
        st = self.monitor.state.get(ot.sim_task)
        return self._fail_attempt(ot, reason, blocking_resource, st.last if st else None)

    def cancel_order(self, order_id: str, reason: str) -> List[str]:
        """Cancel an order. Tasks whose payload has left its origin finish physically (inventory stays
        consistent); their ids are returned."""
        o = self.orders[order_id]
        if o.state != "open":
            return []
        self._close_order(o, "cancelled", reason)
        return [tid for tid in o.task_ids if not self._cancel(self.tasks[tid], "order cancelled: %s" % reason)]

    def task_for_sim(self, sim_task_id: int) -> Optional[OrderTask]:
        tid = self.by_sim.get(sim_task_id)
        return self.tasks[tid] if tid else None

    def group_status(self, group_id: str) -> str:
        states = [self.tasks[t] for t in self.groups[group_id].task_ids]
        if all(t.state == S.COMPLETED for t in states):
            return "completed"
        if any(t.state == S.FAILED and t.final for t in states):
            return "failed"
        if all(t.state == S.CANCELLED for t in states):
            return "cancelled"
        if any(t.state in (S.ASSIGNED, S.EXECUTING, S.SUSPENDED, S.COMPLETED) for t in states):
            return "in_progress"
        return "open"

    # ════════════════════════════════════════════════════════════ arrivals
    def _enqueue(self, d: Dict, source: str) -> None:
        arrival = int(d.get("arrival", self.sim.tick))
        self._queue.append((arrival, self._seq, d, source))
        self._seq += 1
        self._queue.sort(key=lambda x: (x[0], x[1]))

    def _accept_arrivals(self) -> None:
        tick = self.sim.tick
        while self._queue and self._queue[0][0] <= tick:
            _, _, d, source = self._queue.pop(0)
            self._accept(d, source)
        src = self.spec.source
        if src.kind == "poisson":
            n = int(self.rng.poisson(src.rate))
            for _ in range(n):
                if src.max_orders is not None and self._generated >= src.max_orders:
                    break
                self._generated += 1
                self._accept(self._generate(), "generated")

    def _generate(self) -> Dict:
        """One generated order. The number of random draws is fixed per order (kind, lines,
        priority, due date, and per line a selection draw, a station draw and a weight draw), so the
        arrival process is identical for every policy on the same seed; only *which* stored package
        or shelf a line names depends on the inventory at arrival (by the line's selection draw)."""
        src, rng, inv = self.spec.source, self.rng, self.inventory
        names = sorted(src.mix)
        w = np.array([src.mix[k] for k in names], dtype=float)
        kind = names[int(rng.choice(len(names), p=w / w.sum()))]
        lines = max(1, int(round(src.lines.sample(rng))))
        d: Dict[str, Any] = {"kind": kind, "priority": src.priority.sample(rng)}
        due = None if src.due_after is None else self.sim.tick + max(1, int(round(src.due_after.sample(rng))))
        if due is not None:
            d["due"] = due
        draws = [(float(rng.random()), float(rng.random()), self.spec.inventory.package_weight.sample(rng))
                 for _ in range(lines)]
        chain = src.sequence > 0 and float(rng.random()) < src.sequence     # drawn only if configured
        out_cells = self.stations.get("outbound", [])
        cells = self.stations.get(COMPOUND[kind][0] if kind in COMPOUND else kind, [])
        if kind == "outbound":
            cands = sorted((p.package_id for p in inv.stored_packages()))
        elif kind in ("relocation", "replenish"):
            taken = {t.shelf_id for t in self.tasks.values() if t.op == "relocation" and not t.terminal}
            cands = sorted(s.shelf_id for s in inv.shelves.values()
                           if s.status == "stored" and s.busy_task is None and s.shelf_id not in taken
                           and (kind == "relocation" or any(inv.packages[p].status == "stored"
                                                            and inv.packages[p].reserved_by is None
                                                            for p in s.packages)))
        tasks, ends = [], []                                 # ends: (first key, last key) of each line
        for i, (u, v, wt) in enumerate(draws):
            st = list(cells[int(v * len(cells))]) if cells else None
            if kind == "inbound":
                k = "t%d" % len(tasks)                     # the key an unkeyed task would get
                tasks.append({"key": k, "op": "inbound", "weight_kg": wt, "station": st})
                ends.append((k, k))
                continue
            if kind == "cross_dock":
                so = list(out_cells[int(v * len(out_cells))]) if out_cells else None
                tasks += [{"key": "in%d" % i, "op": "inbound", "weight_kg": wt, "station": st},
                          {"key": "out%d" % i, "op": "outbound", "package": "@in%d" % i, "depends_on": ["in%d" % i],
                           **({"station": so} if so else {})}]
                ends.append(("in%d" % i, "out%d" % i))
                continue
            if not cands:
                continue
            pick = cands.pop(int(u * len(cands)))
            if kind == "replenish":
                so = list(out_cells[int(v * len(out_cells))]) if out_cells else None
                pkg = sorted(p for p in inv.shelves[pick].packages
                             if inv.packages[p].status == "stored" and inv.packages[p].reserved_by is None)[0]
                tasks += [{"key": "rel%d" % i, "op": "relocation", "shelf": pick},
                          {"key": "out%d" % i, "op": "outbound", "package": pkg, "depends_on": ["rel%d" % i],
                           **({"station": so} if so else {})}]
                ends.append(("rel%d" % i, "out%d" % i))
                continue
            k = "t%d" % len(tasks)
            tasks.append({"key": k, "op": kind, ("package" if kind == "outbound" else "shelf"): pick,
                          **({"station": st} if st else {})})
            ends.append((k, k))
        if chain:                                            # line i starts after line i-1 has finished
            first = {t["key"]: t for t in tasks}
            for (_, prev_last), (nxt_first, _) in zip(ends, ends[1:]):
                first[nxt_first]["depends_on"] = list(first[nxt_first].get("depends_on", [])) + [prev_last]
        d["tasks"] = tasks
        return d

    def _expand(self, d: Dict) -> List[Dict]:
        """Task dicts of an order (explicit ``tasks`` or the per-operation shorthand)."""
        if "tasks" in d:
            return [dict(t) for t in d["tasks"]]
        op = d.get("op") or d.get("kind")
        if op == "outbound":
            return [{"op": "outbound", "package": p} for p in d.get("packages", ())]
        if op == "inbound":
            pk = d.get("packages")
            if pk is None:
                pk = [{} for _ in range(int(d.get("count", 1)))]
            return [{"op": "inbound", **({"weight_kg": p["weight_kg"]} if isinstance(p, Mapping) and "weight_kg" in p
                                         else {})} for p in pk]
        if op == "relocation":
            to = list(d.get("to") or [])
            return [{"op": "relocation", "shelf": s, **({"to": to[i]} if i < len(to) else {})}
                    for i, s in enumerate(d.get("shelves", ()))]
        raise ConfigError("order %r: unknown operation %r" % (d, op))

    def _accept(self, d: Dict, source: str) -> None:
        sim, inv = self.sim, self.inventory
        tick = sim.tick
        lines = self._expand(d)
        kinds = sorted({t.get("op") for t in lines})
        kind = d.get("kind") or d.get("op") or (kinds[0] if len(kinds) == 1 else "mixed")
        o = Order("O%05d" % len(self.orders), kind, tick, float(d.get("priority", 1.0)),
                  None if d.get("due") is None else int(d["due"]), source)
        self.orders[o.order_id] = o
        # validate and build the tasks (system errors for malformed orders)
        keys = [str(t.get("key", "t%d" % i)) for i, t in enumerate(lines)]
        if len(set(keys)) != len(keys):
            raise ConfigError("order %s: duplicate task keys %s" % (o.order_id, keys))
        for t in lines:
            op = t.get("op")
            if op not in self.spec.operations:
                raise ConfigError("order %s: operation %r is not configured" % (o.order_id, op))
            for dep in t.get("depends_on", ()):
                if dep not in keys:
                    raise ConfigError("order %s: task depends on unknown key %r" % (o.order_id, dep))
        idx = {k: i for i, k in enumerate(keys)}

        class _N:
            def __init__(self, i, deps):
                self.id, self.depends_on, self.required = i, deps, {}
        try:
            TaskGraph([_N(i, tuple(idx[x] for x in t.get("depends_on", ()))) for i, t in enumerate(lines)])
        except TaskGraphError as e:
            raise ConfigError("order %s: %s" % (o.order_id, e)) from e
        station = d.get("station")
        self._emit("order_arrived", order_id=o.order_id, kind=o.kind, source=source, priority=o.priority,
                   due=o.due, tasks=len(lines))
        if not lines:
            self._close_order(o, "cancelled", "rejected: no package or shelf available for a %s order" % kind)
            return
        gids: Dict[str, str] = {}
        tids = {k: "T%05d" % (len(self.tasks) + i) for i, k in enumerate(keys)}
        for k, t in zip(keys, lines):
            gkey = str(t.get("group", k))
            if gkey not in gids:
                g = TaskGroup("G%05d" % len(self.groups), o.order_id, gkey)
                self.groups[g.group_id] = g
                o.group_ids.append(g.group_id)
                gids[gkey] = g.group_id
            ot = OrderTask(tids[k], o.order_id, gids[gkey], k, t["op"], tick,
                           tuple(tids[x] for x in t.get("depends_on", ())))
            self.groups[gids[gkey]].task_ids.append(ot.task_id)
            st = t.get("station", station)
            if ot.op in self.stations:
                cells = self.stations[ot.op]
                ot.station = tuple(st) if st is not None else cells[int(self.rng.integers(len(cells)))]
            if ot.op == "outbound":
                ref = str(t.get("package", ""))
                if ref.startswith("@"):                  # the package of an earlier inbound task of this order
                    rk = ref[1:]
                    if rk not in idx or lines[idx[rk]].get("op") != "inbound" or idx[rk] >= idx[k]:
                        raise ConfigError("order %s: package reference %r must name an earlier inbound task"
                                          % (o.order_id, ref))
                    ot.package_id = self.tasks[tids[rk]].package_id
                else:
                    p = inv.packages.get(ref)
                    if p is None:
                        raise ConfigError("order %s: unknown package %r" % (o.order_id, ref))
                    ot.package_id = p.package_id
            elif ot.op == "inbound":
                w = t.get("weight_kg")
                w = self.spec.inventory.package_weight.sample(self.rng) if w is None else float(w)
                p = inv.new_package(max(0.0, w), tick, ot.station)
                ot.package_id = p.package_id
            else:
                ref = str(t.get("shelf", ""))
                if ref.startswith("@"):                  # the shelf holding another task's package
                    if ref[1:] not in t.get("depends_on", ()):
                        raise ConfigError("order %s: shelf reference %r must also be a dependency" % (o.order_id, ref))
                    ot.shelf_ref = tids[ref[1:]]
                else:
                    s = inv.shelves.get(ref)
                    if s is None:
                        raise ConfigError("order %s: unknown shelf %r" % (o.order_id, ref))
                    ot.shelf_id = s.shelf_id
                if t.get("to") is not None:
                    if t["to"] not in inv.locations:
                        raise ConfigError("order %s: unknown rack location %r" % (o.order_id, t["to"]))
                    ot.dest_location, ot.explicit_dest = str(t["to"]), True
            if ot.package_id:
                p = inv.packages[ot.package_id]
                p.tasks.append(ot.task_id)
                p.orders.append(o.order_id)
            self.tasks[ot.task_id] = ot
            o.task_ids.append(ot.task_id)
            self._set(ot, S.PENDING, None, initial=True)
            self._reserve(ot)

    # ════════════════════════════════════════════════════════════ readiness and dispatch
    def _reserve(self, ot: OrderTask) -> Optional[str]:
        """Take this task's reservations if possible; returns a blocking reason or None."""
        inv = self.inventory
        if ot.op == "outbound":
            p = inv.packages[ot.package_id]
            if p.reserved_by not in (None, ot.task_id):
                return "inventory:package_reserved_by_%s" % p.reserved_by
            if p.status in ("stored", "in_transit", "dropped"):
                inv.reserve_package(p, ot.task_id, self.sim.tick)
            return None
        if ot.op == "inbound":
            p = inv.packages[ot.package_id]
            if ot.shelf_id is None:
                s = self._slot_for(ot)
                if s is None:
                    return "inventory:no_free_slot"
                inv.reserve_slot(s, p, ot.task_id)
                ot.shelf_id = s.shelf_id
            return None
        if ot.shelf_id is None:                          # shelf reference not resolved yet
            return None
        s = inv.shelves[ot.shelf_id]
        if s.status != "stored" and not (ot.dest_location and inv.locations[ot.dest_location].reserved_by == ot.task_id):
            return "inventory:shelf_%s" % s.status       # destination chosen once the shelf is back
        if ot.dest_location is None or inv.locations[ot.dest_location].reserved_by != ot.task_id:
            if ot.explicit_dest:
                l = inv.locations[ot.dest_location]
                if l.shelf_id is not None or l.reserved_by is not None:
                    return "inventory:destination_%s_occupied" % l.location_id
            else:
                l = self._free_location_for(s)
                if l is None:
                    return "inventory:no_free_location"
                ot.dest_location = l.location_id
            inv.reserve_location(inv.locations[ot.dest_location], ot.task_id, s.shelf_id)
        return None

    def _slot_for(self, ot: OrderTask):
        cands = [s for s in self.inventory.shelves.values() if s.status == "stored" and s.free_slots > 0]
        if not cands:
            return None
        if self.spec.slotting == "first_free":
            return min(cands, key=lambda s: s.shelf_id)
        return min(cands, key=lambda s: (self.sim.manhattan(s.cell, ot.station), s.shelf_id))

    def _free_location_for(self, s):
        inv = self.inventory
        frees = sorted(inv.free_locations(), key=lambda l: (self.sim.manhattan(l.cell, s.cell), l.location_id))
        for l in frees:
            if self._reachable(self.spec.operations["relocation"].task_types[-1][1], s.cell, l.cell, False):
                return l
        return None

    def _readiness(self, ot: OrderTask) -> Tuple[str, Any]:
        """('ready', plan) | ('blocked', reason) | ('cancel', reason) | ('fail', reason)."""
        inv = self.inventory
        for d in ot.depends_on:
            dt = self.tasks[d]
            if dt.state == S.CANCELLED or (dt.state == S.FAILED and dt.final):
                return "cancel", "dependency %s %s" % (d, dt.state.value)
            if dt.state != S.COMPLETED:
                return "blocked", "dependency:%s" % d
        if ot.shelf_ref is not None and ot.shelf_id is None:
            p = inv.packages.get(self.tasks[ot.shelf_ref].package_id or "")
            if p is None or p.shelf_id is None:
                return "fail", "reference:%s_has_no_shelved_package" % ot.shelf_ref
            ot.shelf_id = p.shelf_id
        why = self._reserve(ot)
        if why:
            return "blocked", why
        if ot.op == "outbound":
            p = inv.packages[ot.package_id]
            if p.status in ("shipped", "cancelled"):
                return "fail", "inventory:package_%s_%s" % (p.package_id, p.status)
            if p.status != "stored":
                return "blocked", "inventory:package_%s" % p.status
            s = inv.shelves[p.shelf_id]
            extra: Tuple[str, ...] = ()
        else:
            s = inv.shelves[ot.shelf_id]
            if s.status != "stored":
                return "blocked", "inventory:shelf_%s" % s.status
            extra = (ot.package_id,) if ot.op == "inbound" else ()
        if s.busy_task not in (None, ot.task_id):
            return "blocked", "inventory:shelf_busy_with_%s" % s.busy_task
        weight = inv.shelf_weight(s, extra)
        ttype = self.spec.operations[ot.op].task_type_for(weight)
        if ttype is None or not self._can_lift(ttype, weight):
            return "blocked", "infeasible:payload_%.1fkg" % weight
        if ot.op == "relocation":
            dest = inv.locations[ot.dest_location].cell
            ok = self._reachable(ttype, s.cell, dest, False)
        else:
            dest = ot.station
            ok = self._reachable(ttype, s.cell, dest, True)
        if not ok:
            return "blocked", "access:unreachable_%s_to_%s" % (s.cell, dest)
        return "ready", (s, ttype, s.cell, dest, weight)

    def _advance(self) -> None:
        tick = self.sim.tick
        for ot in list(self.tasks.values()):
            if ot.state in (S.PENDING, S.BLOCKED) or (ot.state == S.RETRYING and tick >= ot.retry_at):
                if self.orders[ot.order_id].state != "open":
                    continue
                verdict, x = self._readiness(ot)
                if verdict == "ready":
                    self._dispatch(ot, *x)
                elif verdict == "blocked":
                    if ot.state != S.BLOCKED or ot.reason != x:
                        self._set(ot, S.BLOCKED, x)
                elif verdict == "cancel":
                    self._cancel(ot, x)
                else:
                    self._final_failure(ot, x, None, None, None)

    def _dispatch(self, ot: OrderTask, shelf, ttype: str, origin: Cell, dest: Cell, weight: float) -> None:
        o = self.orders[ot.order_id]
        self.inventory.acquire_shelf(shelf, ot.task_id)
        ot.shelf_id = shelf.shelf_id
        t = self.sim.create_task(ttype, origin, dest, weight_kg=weight, deadline=o.due)
        ot.attempt += 1
        ot.sim_task = t.id
        ot.attempts.append(t.id)
        ot.task_type, ot.weight_kg = ttype, weight
        ot._hp, ot._stages_done = len(t.record.history), 0
        self.by_sim[t.id] = ot.task_id
        if ot.first_ready is None:
            ot.first_ready = self.sim.tick
        ot.ready_since = self.sim.tick
        self._set(ot, S.READY, None, sim_task=t.id, task_type=ttype, origin=origin, destination=dest,
                  weight_kg=round(weight, 3))

    def _can_lift(self, ttype: str, weight: float) -> bool:
        t = self.sim.config.task_type(ttype)
        role = t.role(t.carrier_role)
        caps = sorted((s.payload_kg for _, s, _ in self.sim._robot_specs
                       if set(role.capabilities) & set(s.capabilities) and s.payload_kg > 0), reverse=True)
        return sum(caps[:role.count]) + 1e-9 >= weight and len(caps) >= role.count

    def _reachable(self, ttype: str, a: Cell, b: Cell, both_ways: bool) -> bool:
        """A loaded carrier of ``ttype`` can drive from ``a`` to ``b`` (and back) on the static map."""
        t = self.sim.config.task_type(ttype)
        role = t.role(t.carrier_role)
        zones = sorted({tuple(sorted(s.zones_loaded or s.zones)) for _, s, _ in self.sim._robot_specs
                        if set(role.capabilities) & set(s.capabilities) and s.payload_kg > 0})
        for z in zones:
            key = (z, a, b, both_ways)
            if key not in self._reach:
                pas = self.sim.passable(z)
                d = self.sim.layout.directions
                ok = astar(pas, a, b, None, d) is not None and (not both_ways or astar(pas, b, a, None, d) is not None)
                self._reach[key] = ok
            if self._reach[key]:
                return True
        return False

    # ════════════════════════════════════════════════════════════ execution feedback
    def _on_sim_event(self, e) -> None:
        if e.task is None or e.task not in self.by_sim:
            return
        ot = self.tasks[self.by_sim[e.task]]
        if ot.sim_task != e.task:
            return
        inv, tick = self.inventory, self.sim.tick
        t = self.sim.tasks[e.task]
        if e.type == "robot_released" and e.data.get("reason") not in ("task cancelled",):
            ot.ready_since = tick                        # a role is open again (stage-by-stage recruitment)
        elif e.type == "payload_picked":
            s = inv.shelves[ot.shelf_id]
            if e.data.get("reload"):
                inv.reload(s, ot.task_id, tick)
            else:
                inv.lift(s, ot.task_id, t.stage_cells[0], tick, keep_home=ot.op != "relocation")
        elif e.type == "payload_dropped":
            inv.drop(inv.shelves[ot.shelf_id], e.data["cell"], ot.task_id, tick)
        elif e.type == "stage_completed":
            ot._stages_done += 1
            if ot.op != "relocation" and ot._stages_done - 1 == handoff_stage(t.type):
                s, p = inv.shelves[ot.shelf_id], inv.packages[ot.package_id]
                if ot.op == "outbound":
                    inv.ship(p, s, t.destination, ot.task_id, tick)
                else:
                    inv.put(p, s, ot.task_id, tick)
        elif e.type == "payload_placed":
            s = inv.shelves[ot.shelf_id]
            lid = ot.dest_location if ot.op == "relocation" else s.home_id
            inv.place(s, inv.locations[lid], e.data["cell"], ot.task_id, tick)

    def _mirror(self) -> None:
        for ot in list(self.tasks.values()):
            if ot.sim_task is None:
                continue
            rec = self.sim.tasks[ot.sim_task].record
            for _, _, to, reason in rec.history[ot._hp:]:
                new = FROM_SIM.get(to.value)
                if new is None or new == ot.state:
                    continue
                if new in (S.READY, S.SUSPENDED):
                    ot.ready_since = self.sim.tick
                self._set(ot, new, reason)
            ot._hp = len(rec.history)
            if ot.state == S.COMPLETED:
                self._on_completed(ot)

    def _on_completed(self, ot: OrderTask) -> None:
        sid = ot.sim_task
        self._collect_waits(ot)
        ot.sim_task = None
        ot.completed_at = self.sim.tasks[sid].completed_at
        self.inventory.release_shelf(self.inventory.shelves[ot.shelf_id], ot.task_id)
        o = self.orders[ot.order_id]
        if o.state == "open" and all(self.tasks[t].state == S.COMPLETED for t in o.task_ids):
            self._close_order(o, "completed", None)

    def _collect_waits(self, ot: OrderTask) -> None:
        for k, v in self.monitor.waits(ot.sim_task).items():
            ot.waits[k] = ot.waits.get(k, 0) + v

    def _on_stall(self, rep) -> None:
        ot = self.task_for_sim(rep.sim_task)
        if ot is None or ot.sim_task != rep.sim_task:
            return
        ot.stalls += 1
        lc = self.spec.lifecycle
        can, why = self.sim.can_cancel(rep.sim_task)
        action = ("fail_attempt" if can else "existing_recovery: %s" % why) if lc.auto_retry else "diagnostic_only"
        self._emit("stall_suspected", task=rep.sim_task, robots=tuple(c["robot"] for c in rep.snapshot["crew"]),
                   order_id=ot.order_id, task_id=ot.task_id, attempt=ot.attempt, wait_reason=rep.wait_reason,
                   blocking_resource=rep.blocking_resource, last_progress_tick=rep.last_progress_tick,
                   stalled_ticks=rep.stalled_ticks, blocked_moves=rep.blocked_moves, action=action,
                   snapshot=rep.snapshot)
        if lc.auto_retry and can:
            self._fail_attempt(ot, "stall: no progress for %d ticks (%s)" % (rep.stalled_ticks, rep.wait_reason),
                               rep.blocking_resource, rep.last_progress_tick, rep.snapshot)

    # ════════════════════════════════════════════════════════════ failure, retry, cancellation
    def _fail_attempt(self, ot: OrderTask, reason: str, blocking: Optional[str], last_progress: Optional[int],
                      snap: Optional[Dict] = None) -> Tuple[bool, str]:
        sim = self.sim
        sid = ot.sim_task
        ok, why = sim.can_cancel(sid)
        if not ok:
            return False, why
        if snap is None:
            from battery_tarware.orders.monitor import snapshot
            snap = snapshot(sim, sim.tasks[sid])
        self._collect_waits(ot)
        ot.sim_task = None                                   # detach before the simulator releases the crew
        sim.cancel_task(sid, "order layer: attempt %d of %s failed: %s" % (ot.attempt, ot.task_id, reason))
        self.inventory.release_shelf(self.inventory.shelves[ot.shelf_id], ot.task_id)
        rec = FailureRecord(sim.tick, ot.attempt, sid, reason, blocking, last_progress, snap)
        ot.failures.append(rec)
        self._set(ot, S.FAILED, reason, sim_task=sid, failure_reason=reason, blocking_resource=blocking,
                  last_progress_tick=last_progress, snapshot=snap)
        lc = self.spec.lifecycle
        if ot.attempt < 1 + lc.max_dispatch_retries and self.orders[ot.order_id].state == "open":
            ot.retry_at = sim.tick + lc.retry_delay
            self._set(ot, S.RETRYING, "retry %d of %d after: %s" % (ot.attempt, lc.max_dispatch_retries, reason),
                      retry_at=ot.retry_at)
        else:
            ot.final = True
            self._release_reservations(ot, "task failed")
            self._close_order(self.orders[ot.order_id], "failed",
                              "task %s failed after %d attempt(s): %s" % (ot.task_id, ot.attempt, reason))
        return True, "ok"

    def _final_failure(self, ot: OrderTask, reason: str, blocking, last_progress, snap) -> None:
        ot.failures.append(FailureRecord(self.sim.tick, ot.attempt, None, reason, blocking, last_progress, snap))
        self._set(ot, S.FAILED, reason, failure_reason=reason, blocking_resource=blocking,
                  last_progress_tick=last_progress, snapshot=snap)
        ot.final = True
        self._release_reservations(ot, "task failed")
        self._close_order(self.orders[ot.order_id], "failed", "task %s failed: %s" % (ot.task_id, reason))

    def _cancel(self, ot: OrderTask, reason: str) -> bool:
        """Cancel a task unless its payload has left the origin. Returns False if it keeps running."""
        if ot.terminal:
            return True
        if ot.sim_task is not None:
            ok, _ = self.sim.can_cancel(ot.sim_task)
            if not ok:
                return False
            sid = ot.sim_task
            self._collect_waits(ot)
            ot.sim_task = None
            self.sim.cancel_task(sid, "order layer: %s" % reason)
        if ot.state == S.FAILED:                             # failed, retry not yet scheduled
            ot.final = True
        self._release_reservations(ot, reason)
        self._set(ot, S.CANCELLED, reason)
        for other in self.tasks.values():
            if ot.task_id in other.depends_on and not other.terminal:
                self._cancel(other, "dependency %s cancelled" % ot.task_id)
        return True

    def _release_reservations(self, ot: OrderTask, reason: str) -> None:
        inv = self.inventory
        if ot.op == "outbound":
            inv.unreserve_package(inv.packages[ot.package_id], ot.task_id)
        elif ot.op == "inbound":
            p = inv.packages[ot.package_id]
            if p.status == "arrived":
                if ot.shelf_id is not None:
                    inv.release_slot(inv.shelves[ot.shelf_id], p, ot.task_id)
                inv.cancel_package(p, "%s %s: %s" % (ot.task_id, "not stored", reason), self.sim.tick)
        elif ot.dest_location is not None:
            inv.release_location(inv.locations[ot.dest_location], ot.task_id)
        if ot.shelf_id is not None:
            inv.release_shelf(inv.shelves[ot.shelf_id], ot.task_id)

    def _close_order(self, o: Order, state: str, reason: Optional[str]) -> None:
        if o.state != "open":
            return
        o.state, o.reason, o.closed = state, reason, self.sim.tick
        self._emit("order_%s" % state, order_id=o.order_id, reason=reason,
                   lead_time=self.sim.tick - o.arrival, due=o.due)
        if state == "failed":
            for tid in o.task_ids:
                self._cancel(self.tasks[tid], "order %s failed" % o.order_id)   # started ones finish

    # ════════════════════════════════════════════════════════════ bookkeeping
    def _set(self, ot: OrderTask, new: OTaskState, reason: Optional[str], initial: bool = False, **extra) -> None:
        old = ot.state
        if not initial:
            check_transition(ot.task_id, old, new, reason)
        ot.history.append((self.sim.tick, old.value if not initial else "", new.value, reason))
        ot.state, ot.reason = new, reason
        self._emit("order_task", task=ot.sim_task if ot.sim_task is not None else extra.pop("sim_task", None),
                   order_id=ot.order_id, task_id=ot.task_id, group_id=ot.group_id, op=ot.op,
                   frm=None if initial else old.value, to=new.value, reason=reason, attempt=ot.attempt,
                   package_id=ot.package_id, shelf_id=ot.shelf_id, **extra)

    def _emit(self, type_: str, task: Optional[int] = None, robots=(), **data) -> None:
        self.sim.events.emit(self.sim.tick, type_, robots, task, **data)

    def summary(self) -> Dict[str, Any]:
        orders: Dict[str, int] = {}
        for o in self.orders.values():
            orders[o.state] = orders.get(o.state, 0) + 1
        tasks: Dict[str, int] = {}
        for t in self.tasks.values():
            tasks[t.state.value] = tasks.get(t.state.value, 0) + 1
        return {"orders": orders, "tasks": tasks, "packages": self.inventory.counts(),
                "retries": sum(max(0, t.attempt - 1) for t in self.tasks.values()),
                "stalls_suspected": sum(t.stalls for t in self.tasks.values())}


__all__ = ["OrderManager", "Order", "OrderTask", "TaskGroup", "FailureRecord", "InventoryError"]
