"""The ``orders:`` scenario key: operations, inventory, order source and lifecycle settings.

    orders:
      operations:
        outbound:   {task_types: [{max_kg: 6, task_type: solo}, {task_type: standard}], station: goal}
        inbound:    {task_types: [{task_type: standard}], station: goal}
        relocation: {task_types: [{task_type: relocate}]}
      inventory:  {shelf_fill: 0.85, package_fill: 0.6, slot_capacity: 1, package_weight: {...}, shelf_kg: 0}
      source:     {kind: poisson, rate: 0.02, mix: {outbound: 0.6, inbound: 0.3, relocation: 0.1},
                   max_orders: 200, lines: 1, priority: 1, due_after: {kind: uniform, low: 300, high: 600},
                  sequence: 0.0}
                  # mix may also name compound orders: cross_dock (per line: inbound, then outbound of
                  # that package) and replenish (per line: relocate a shelf, then outbound of one of
                  # its packages); with probability ``sequence`` a multi-line order's lines form a chain
                  # or {kind: trace, orders: [...]} or {kind: none} (orders only through OrderManager.submit)
      lifecycle:  {stall_ticks: 100, max_dispatch_retries: 3, retry_delay: 10, auto_retry: false}
      slotting: nearest

Order trace entries: ``{arrival, kind, priority, due, tasks: [...]}`` with tasks
``{key, op, package | weight_kg | shelf, to, station, depends_on: [keys], group}``, or the shorthand
``{op: outbound, packages: [...]}``, ``{op: inbound, count: n}``, ``{op: relocation, shelves: [...],
to: [...]}``. Within an order, ``package: "@key"`` names the package of an earlier inbound task and
``shelf: "@key"`` the shelf holding that task's package (``key`` must then be a dependency).

Task types per operation are tried in order; the first whose ``max_kg`` (omitted = no limit) is at
least the shelf weight is used. Outbound and inbound task types carry the shelf from its rack
location to a station cell and back (first stage at ``origin``, a stage at ``destination``, last
stage at ``origin``). Relocation task types carry it to another rack location (first stage at
``origin``, last at ``destination``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from battery_tarware.config import ConfigError, Distribution, _check

OPERATIONS = ("outbound", "inbound", "relocation")
# generated compound orders (poisson ``mix`` keys): a dependency pair per line
COMPOUND = {"cross_dock": ("inbound", "outbound"),        # receive a package, then ship that package
            "replenish": ("relocation", "outbound")}      # move a shelf, then pick a package from it
SOURCE_KINDS = ("poisson", "trace", "none")
SLOTTING = ("nearest", "first_free")


def _keys(d: Mapping, allowed, where: str) -> None:
    bad = sorted(set(d) - set(allowed))
    _check(not bad, "%s: unknown key(s) %s (allowed: %s)" % (where, bad, sorted(allowed)))


@dataclass(frozen=True)
class OperationSpec:
    op: str
    task_types: Tuple[Tuple[float, str], ...]      # (max_kg, task type), tried in order
    station: Optional[str] = None                  # zone of the station cells (outbound, inbound)

    def task_type_for(self, weight_kg: float) -> Optional[str]:
        for max_kg, name in self.task_types:
            if weight_kg <= max_kg + 1e-9:
                return name
        return None


@dataclass(frozen=True)
class InventorySpec:
    shelf_fill: float = 0.85                       # share of rack locations that hold a shelf at reset
    package_fill: float = 0.6                      # share of shelf slots that hold a package at reset
    slot_capacity: int = 1                         # packages per shelf
    package_weight: Distribution = field(default_factory=lambda: Distribution("uniform", low=0.5, high=10.0))
    shelf_kg: float = 0.0                          # tare weight carried with every shelf
    initial: Optional[Mapping[str, int]] = None    # explicit {location_id: packages}; other locations empty


@dataclass(frozen=True)
class SourceSpec:
    kind: str = "none"
    rate: float = 0.0                              # poisson: orders per tick
    mix: Mapping[str, float] = field(default_factory=lambda: {"outbound": 1.0})
    max_orders: Optional[int] = None
    lines: Distribution = field(default_factory=lambda: Distribution("fixed", value=1.0))
    priority: Distribution = field(default_factory=lambda: Distribution("fixed", value=1.0))
    due_after: Optional[Distribution] = None       # ticks after arrival; None = no due date
    orders: Tuple[Mapping[str, Any], ...] = ()     # trace
    sequence: float = 0.0                          # poisson: probability a multi-line order is a chain


@dataclass(frozen=True)
class LifecycleSpec:
    stall_ticks: int = 100                         # eligible ticks without progress before a stall is suspected
    max_dispatch_retries: int = 3                  # retries after the first dispatch
    retry_delay: int = 10                          # ticks in ``retrying`` before readiness is rechecked
    auto_retry: bool = False                       # a suspected stall fails the attempt (else diagnostic only)
    check_inventory: bool = True                   # check the inventory invariants every tick


ORDER_KEYS = ("arrival", "kind", "priority", "due", "tasks", "op", "packages", "count", "shelves", "to", "station")
TASK_KEYS = ("key", "op", "package", "weight_kg", "shelf", "to", "station", "depends_on", "group")


@dataclass(frozen=True)
class OrderSpec:
    operations: Mapping[str, OperationSpec]
    inventory: InventorySpec
    source: SourceSpec
    lifecycle: LifecycleSpec
    slotting: str = "nearest"

    @staticmethod
    def from_dict(d: Mapping[str, Any], cfg=None) -> "OrderSpec":
        w = "orders"
        _keys(d, ("operations", "inventory", "source", "lifecycle", "slotting"), w)
        ops = {}
        for op, od in dict(d.get("operations", {})).items():
            _check(op in OPERATIONS, "%s.operations: unknown operation %r (known: %s)" % (w, op, OPERATIONS))
            od = dict(od)
            _keys(od, ("task_types", "task_type", "station"), "%s.operations.%s" % (w, op))
            raw = od.get("task_types") or ([{"task_type": od["task_type"]}] if od.get("task_type") else [])
            _check(raw, "%s.operations.%s: give task_types (or task_type)" % (w, op))
            tts = []
            for i, e in enumerate(raw):
                e = {"task_type": e} if isinstance(e, str) else dict(e)
                _keys(e, ("task_type", "max_kg"), "%s.operations.%s.task_types[%d]" % (w, op, i))
                tts.append((float(e.get("max_kg", math.inf)), str(e["task_type"])))
            station = od.get("station", "goal" if op != "relocation" else None)
            _check(op == "relocation" or station, "%s.operations.%s: needs a station zone" % (w, op))
            ops[op] = OperationSpec(op, tuple(tts), station)
        _check(ops, "%s: needs at least one operation" % w)

        inv = dict(d.get("inventory", {}))
        _keys(inv, InventorySpec.__dataclass_fields__, "%s.inventory" % w)
        if "package_weight" in inv:
            inv["package_weight"] = Distribution.from_any(inv["package_weight"], "%s.inventory.package_weight" % w)
            inv["package_weight"].validate("%s.inventory.package_weight" % w)
        if inv.get("initial") is not None:
            inv["initial"] = {str(k): int(v) for k, v in dict(inv["initial"]).items()}
        inventory = InventorySpec(**inv)
        _check(0 <= inventory.shelf_fill <= 1 and 0 <= inventory.package_fill <= 1,
               "%s.inventory: shelf_fill and package_fill must be in [0, 1]" % w)
        _check(inventory.slot_capacity >= 1, "%s.inventory.slot_capacity must be >= 1" % w)
        _check(inventory.shelf_kg >= 0, "%s.inventory.shelf_kg must be >= 0" % w)
        if inventory.initial is not None:
            _check(all(0 <= n <= inventory.slot_capacity for n in inventory.initial.values()),
                   "%s.inventory.initial: package counts must be in [0, slot_capacity]" % w)

        src = dict(d.get("source", {}))
        _keys(src, SourceSpec.__dataclass_fields__, "%s.source" % w)
        for k in ("lines", "priority", "due_after"):
            if src.get(k) is not None:
                src[k] = Distribution.from_any(src[k], "%s.source.%s" % (w, k))
                src[k].validate("%s.source.%s" % (w, k))
        if "mix" in src:
            src["mix"] = {str(k): float(v) for k, v in dict(src["mix"]).items()}
        if "orders" in src:
            src["orders"] = tuple(dict(o) for o in src["orders"])
        source = SourceSpec(**src)
        _check(source.kind in SOURCE_KINDS, "%s.source.kind must be one of %s" % (w, SOURCE_KINDS))
        if source.kind == "poisson":
            _check(source.rate >= 0, "%s.source.rate must be >= 0" % w)
            _check(source.mix and sum(source.mix.values()) > 0, "%s.source.mix needs positive weights" % w)
            need = set()
            for k in source.mix:
                need.update(COMPOUND.get(k, (k,)))
            bad = sorted(need - set(ops))
            _check(not bad, "%s.source.mix: operation(s) %s are not configured" % (w, bad))
            _check(0.0 <= source.sequence <= 1.0, "%s.source.sequence must be in [0, 1]" % w)
        for i, o in enumerate(source.orders):
            _keys(o, ORDER_KEYS, "%s.source.orders[%d]" % (w, i))
            for j, t in enumerate(o.get("tasks", ())):
                _keys(t, TASK_KEYS, "%s.source.orders[%d].tasks[%d]" % (w, i, j))
        _check(source.max_orders is None or source.max_orders >= 0, "%s.source.max_orders must be >= 0" % w)

        lc = dict(d.get("lifecycle", {}))
        _keys(lc, LifecycleSpec.__dataclass_fields__, "%s.lifecycle" % w)
        lifecycle = LifecycleSpec(**lc)
        _check(lifecycle.stall_ticks >= 1, "%s.lifecycle.stall_ticks must be >= 1" % w)
        _check(lifecycle.max_dispatch_retries >= 0 and lifecycle.retry_delay >= 0,
               "%s.lifecycle: max_dispatch_retries and retry_delay must be >= 0" % w)
        slotting = d.get("slotting", "nearest")
        _check(slotting in SLOTTING, "%s.slotting must be one of %s" % (w, SLOTTING))
        spec = OrderSpec(ops, inventory, source, lifecycle, slotting)
        if cfg is not None:
            spec.validate_against(cfg)
        return spec

    def validate_against(self, cfg) -> None:
        tnames = {t.name for t in cfg.task_types}
        for op, o in self.operations.items():
            w = "orders.operations.%s" % op
            if o.station is not None:
                _check(o.station in cfg.zones, "%s: unknown station zone %r" % (w, o.station))
            for _, name in o.task_types:
                _check(name in tnames, "%s: unknown task type %r" % (w, name))
                t = cfg.task_type(name)
                _check(t.carrier_role is not None, "%s: task type %r has no carrier role" % (w, name))
                at = [s.at for s in t.stages]
                _check(at[0] == "origin", "%s: task type %r must start at its origin (the rack location)" % (w, name))
                if op == "relocation":
                    _check(at[-1] == "destination", "%s: task type %r must end at its destination" % (w, name))
                else:
                    _check("destination" in at and at[-1] == "origin",
                           "%s: task type %r must visit its destination (the station) and end at its origin"
                           % (w, name))


def handoff_stage(ttype) -> int:
    """Index of the stage at which an outbound package leaves (or an inbound package joins) the
    shelf: the first stage at the destination."""
    return [s.at for s in ttype.stages].index("destination")


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)
