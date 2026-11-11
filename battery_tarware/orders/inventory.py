"""Persistent, location-aware inventory (logical packages on the existing shelves).

* **Rack location** (``L0000``): a storage cell; holds at most one shelf. Ids follow the cells in
  row-major order.
* **Shelf** (``S0000``): the unit carriers move (TA-RWARE convention); it stands on a rack location,
  is carried, or rests where a carrier dropped it. ``slot_capacity`` packages fit on one shelf.
* **Package** (``P00000``): a logical unit on a shelf. Its id is kept across inbound, storage,
  relocation and outbound. No package geometry or handling physics is added.

Reservations are taken when an operation is accepted (a package for an outbound task, a shelf slot
for an inbound package, a free location for a relocation, the home location of a shelf that is
away) and released on cancellation or failure. Physical changes are committed only from simulator
events: a shelf leaves its location when the payload is picked up, a package leaves or joins the
shelf when the destination stage completes, and the shelf is placed when the task completes.

Package status: ``arrived`` (inbound, waiting at its station), ``stored``, ``in_transit`` (its shelf
is carried), ``dropped`` (its shelf rests off its location after a carrier failure), ``shipped``
(left at an outbound station; terminal), ``cancelled`` (an inbound package that was never stored).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Cell = Tuple[int, int]


class InventoryError(RuntimeError):
    """The inventory became inconsistent (a system error, never a task failure)."""


@dataclass
class RackLocation:
    location_id: str
    cell: Cell
    shelf_id: Optional[str] = None               # shelf standing here
    reserved_by: Optional[str] = None            # task holding it (a shelf returning or arriving)
    reserved_for: Optional[str] = None           # shelf expected here


@dataclass
class Shelf:
    shelf_id: str
    capacity: int
    location_id: Optional[str] = None            # rack location while stored, else None
    home_id: Optional[str] = None                # location it stands on or will return to
    cell: Optional[Cell] = None                  # physical cell (None while carried)
    status: str = "stored"                       # stored | in_transit | dropped
    packages: List[str] = field(default_factory=list)
    incoming: List[str] = field(default_factory=list)      # inbound packages holding a slot
    busy_task: Optional[str] = None              # task currently dispatched on this shelf

    @property
    def free_slots(self) -> int:
        return self.capacity - len(self.packages) - len(self.incoming)


@dataclass
class Package:
    package_id: str
    weight_kg: float
    created: int
    status: str = "stored"
    shelf_id: Optional[str] = None
    cell: Optional[Cell] = None                  # last known physical cell (None while carried)
    reserved_by: Optional[str] = None            # task that will move it off its shelf / onto a shelf
    tasks: List[str] = field(default_factory=list)
    orders: List[str] = field(default_factory=list)
    history: List[Tuple[int, str, str]] = field(default_factory=list)   # (tick, status, detail)


class Inventory:
    def __init__(self, storage_cells: List[Cell], capacity: int, shelf_kg: float, emit: Callable):
        self.capacity = capacity
        self.shelf_kg = shelf_kg
        self._emit = emit                        # emit(op, **data) -> event in the simulator log
        self.locations: Dict[str, RackLocation] = {}
        self.by_cell: Dict[Cell, str] = {}
        for i, c in enumerate(sorted(storage_cells)):
            lid = "L%04d" % i
            self.locations[lid] = RackLocation(lid, c)
            self.by_cell[c] = lid
        self.shelves: Dict[str, Shelf] = {}
        self.packages: Dict[str, Package] = {}
        self.created = 0                         # packages ever created (initial + inbound)
        self.shipped = 0
        self.cancelled = 0

    # ---------------------------------------------------------------- construction
    def populate(self, rng, spec, tick: int) -> None:
        """Initial shelves and packages: explicit (``spec.initial``) or drawn with the order stream."""
        lids = list(self.locations)
        if spec.initial is not None:
            unknown = sorted(set(spec.initial) - set(lids))
            if unknown:
                raise InventoryError("inventory.initial: unknown location(s) %s" % unknown)
            fill = [(lid, spec.initial[lid]) for lid in lids if lid in spec.initial]
        else:
            n = int(round(spec.shelf_fill * len(lids)))
            chosen = sorted(rng.choice(len(lids), size=n, replace=False)) if n else []
            fill = [(lids[i], int(rng.binomial(self.capacity, spec.package_fill))) for i in chosen]
        for lid, npk in fill:
            s = Shelf("S%04d" % len(self.shelves), self.capacity, lid, lid, self.locations[lid].cell)
            self.shelves[s.shelf_id] = s
            self.locations[lid].shelf_id = s.shelf_id
            for _ in range(npk):
                p = self.new_package(max(0.0, spec.package_weight.sample(rng)), tick, s.cell, emit=False)
                p.status, p.shelf_id = "stored", s.shelf_id
                s.packages.append(p.package_id)
                p.history.append((tick, "stored", "initial inventory on %s at %s" % (s.shelf_id, lid)))
        self._emit("initialised", locations={l.location_id: list(l.cell) for l in self.locations.values()},
                   shelves={s.shelf_id: [s.location_id, list(s.packages)] for s in self.shelves.values()},
                   packages={p.package_id: round(p.weight_kg, 3) for p in self.packages.values()})

    def new_package(self, weight_kg: float, tick: int, cell: Optional[Cell], emit: bool = True) -> Package:
        p = Package("P%05d" % self.created, weight_kg, tick, status="arrived", cell=cell)
        self.created += 1
        self.packages[p.package_id] = p
        if emit:
            p.history.append((tick, "arrived", "at %s" % (cell,)))
            self._emit("package_arrived", package_id=p.package_id, cell=cell, weight_kg=round(weight_kg, 3))
        return p

    # ---------------------------------------------------------------- queries
    def shelf_weight(self, s: Shelf, extra: Tuple[str, ...] = ()) -> float:
        return self.shelf_kg + sum(self.packages[p].weight_kg for p in list(s.packages) + list(extra))

    def free_locations(self) -> List[RackLocation]:
        return [l for l in self.locations.values() if l.shelf_id is None and l.reserved_by is None]

    def stored_packages(self, unreserved: bool = True) -> List[Package]:
        return [p for p in self.packages.values() if p.status == "stored"
                and (not unreserved or p.reserved_by is None)]

    def rack_location_of(self, p: Package) -> Optional[str]:
        return self.shelves[p.shelf_id].location_id if p.shelf_id else None

    # ---------------------------------------------------------------- reservations
    def reserve_package(self, p: Package, task_id: str, tick: int) -> None:
        assert p.reserved_by in (None, task_id), (p.package_id, p.reserved_by)
        if p.reserved_by != task_id:
            p.reserved_by = task_id
            self._emit("package_reserved", package_id=p.package_id, task_id=task_id)

    def unreserve_package(self, p: Package, task_id: str) -> None:
        if p.reserved_by == task_id:
            p.reserved_by = None
            self._emit("package_unreserved", package_id=p.package_id, task_id=task_id)

    def reserve_slot(self, s: Shelf, p: Package, task_id: str) -> None:
        assert s.free_slots > 0 and p.package_id not in s.incoming
        s.incoming.append(p.package_id)
        p.reserved_by = task_id
        self._emit("slot_reserved", shelf_id=s.shelf_id, package_id=p.package_id, task_id=task_id)

    def release_slot(self, s: Shelf, p: Package, task_id: str) -> None:
        if p.package_id in s.incoming:
            s.incoming.remove(p.package_id)
            self._emit("slot_released", shelf_id=s.shelf_id, package_id=p.package_id, task_id=task_id)
        if p.reserved_by == task_id:
            p.reserved_by = None

    def reserve_location(self, l: RackLocation, task_id: str, shelf_id: str) -> None:
        assert l.shelf_id is None and l.reserved_by is None, l
        l.reserved_by, l.reserved_for = task_id, shelf_id
        self._emit("location_reserved", location_id=l.location_id, task_id=task_id, shelf_id=shelf_id)

    def release_location(self, l: RackLocation, task_id: str) -> None:
        if l.reserved_by == task_id and l.shelf_id is None:
            l.reserved_by = l.reserved_for = None
            self._emit("location_released", location_id=l.location_id, task_id=task_id)

    def acquire_shelf(self, s: Shelf, task_id: str) -> None:
        assert s.busy_task in (None, task_id), (s.shelf_id, s.busy_task)
        s.busy_task = task_id

    def release_shelf(self, s: Shelf, task_id: str) -> None:
        if s.busy_task == task_id:
            s.busy_task = None

    # ---------------------------------------------------------------- physical commits
    def lift(self, s: Shelf, task_id: str, cell: Cell, tick: int, keep_home: bool) -> None:
        """The carrier picked the shelf up at ``cell`` (its rack location)."""
        if s.status != "stored" or s.location_id is None or self.locations[s.location_id].cell != tuple(cell):
            raise InventoryError("%s lifted at %s but it is %s at %s" % (s.shelf_id, cell, s.status, s.location_id))
        l = self.locations[s.location_id]
        l.shelf_id = None
        if keep_home:                            # outbound / inbound: the shelf comes back here
            l.reserved_by, l.reserved_for = task_id, s.shelf_id
        s.location_id, s.cell, s.status = None, None, "in_transit"
        self._set_packages(s, "in_transit", None, tick, "%s lifted from %s" % (s.shelf_id, l.location_id))
        self._emit("shelf_lifted", shelf_id=s.shelf_id, location_id=l.location_id, task_id=task_id)

    def drop(self, s: Shelf, cell: Cell, task_id: str, tick: int) -> None:
        s.cell, s.status = tuple(cell), "dropped"
        self._set_packages(s, "dropped", tuple(cell), tick, "%s dropped at %s" % (s.shelf_id, cell))
        self._emit("shelf_dropped", shelf_id=s.shelf_id, cell=cell, task_id=task_id)

    def reload(self, s: Shelf, task_id: str, tick: int) -> None:
        if s.status != "dropped":
            raise InventoryError("%s re-picked but it is %s" % (s.shelf_id, s.status))
        s.cell, s.status = None, "in_transit"
        self._set_packages(s, "in_transit", None, tick, "%s re-picked" % s.shelf_id)
        self._emit("shelf_reloaded", shelf_id=s.shelf_id, task_id=task_id)

    def ship(self, p: Package, s: Shelf, cell: Cell, task_id: str, tick: int) -> None:
        if p.package_id not in s.packages:
            raise InventoryError("%s shipped from %s but it is not on it" % (p.package_id, s.shelf_id))
        s.packages.remove(p.package_id)
        p.status, p.shelf_id, p.cell, p.reserved_by = "shipped", None, tuple(cell), None
        self.shipped += 1
        p.history.append((tick, "shipped", "at %s by %s" % (cell, task_id)))
        self._emit("package_shipped", package_id=p.package_id, shelf_id=s.shelf_id, cell=cell, task_id=task_id)

    def put(self, p: Package, s: Shelf, task_id: str, tick: int) -> None:
        if p.package_id not in s.incoming or p.status != "arrived":
            raise InventoryError("%s put on %s without a reserved slot" % (p.package_id, s.shelf_id))
        s.incoming.remove(p.package_id)
        s.packages.append(p.package_id)
        p.status, p.shelf_id, p.cell, p.reserved_by = s.status, s.shelf_id, s.cell, None
        p.history.append((tick, p.status, "put on %s by %s" % (s.shelf_id, task_id)))
        self._emit("package_put", package_id=p.package_id, shelf_id=s.shelf_id, task_id=task_id)

    def place(self, s: Shelf, l: RackLocation, cell: Cell, task_id: str, tick: int) -> None:
        """The shelf was set down at the end of its task, on rack location ``l``."""
        if tuple(cell) != l.cell or l.shelf_id is not None or l.reserved_by != task_id:
            raise InventoryError("%s placed at %s, expected free location %s %s reserved by %s"
                                 % (s.shelf_id, cell, l.location_id, l.cell, task_id))
        l.shelf_id, l.reserved_by, l.reserved_for = s.shelf_id, None, None
        s.location_id, s.home_id, s.cell, s.status = l.location_id, l.location_id, l.cell, "stored"
        self._set_packages(s, "stored", l.cell, tick, "%s placed on %s" % (s.shelf_id, l.location_id))
        self._emit("shelf_placed", shelf_id=s.shelf_id, location_id=l.location_id, task_id=task_id)

    def cancel_package(self, p: Package, reason: str, tick: int) -> None:
        if p.status != "arrived":
            raise InventoryError("%s cancelled but it is %s" % (p.package_id, p.status))
        p.status, p.reserved_by = "cancelled", None
        self.cancelled += 1
        p.history.append((tick, "cancelled", reason))
        self._emit("package_cancelled", package_id=p.package_id, reason=reason)

    def _set_packages(self, s: Shelf, status: str, cell, tick: int, detail: str) -> None:
        for pid in s.packages:
            p = self.packages[pid]
            p.status, p.cell = status, cell
            p.history.append((tick, status, detail))

    # ---------------------------------------------------------------- invariants
    def check(self) -> None:
        """Raise InventoryError if any invariant is violated (conservation, single placement)."""
        seen: Dict[str, str] = {}
        for l in self.locations.values():
            if l.shelf_id is not None:
                s = self.shelves.get(l.shelf_id)
                if s is None or s.location_id != l.location_id or s.status != "stored":
                    raise InventoryError("location %s holds %s, which is not stored there" % (l.location_id, l.shelf_id))
                if l.reserved_by is not None:
                    raise InventoryError("location %s holds %s and is reserved by %s"
                                         % (l.location_id, l.shelf_id, l.reserved_by))
        for s in self.shelves.values():
            if s.status == "stored":
                if s.location_id is None or self.locations[s.location_id].shelf_id != s.shelf_id:
                    raise InventoryError("%s is stored but no location holds it" % s.shelf_id)
            elif s.location_id is not None:
                raise InventoryError("%s is %s but still on location %s" % (s.shelf_id, s.status, s.location_id))
            if len(s.packages) + len(s.incoming) > s.capacity:
                raise InventoryError("%s holds %d + %d incoming > capacity %d"
                                     % (s.shelf_id, len(s.packages), len(s.incoming), s.capacity))
            for pid in s.packages:
                if pid in seen:
                    raise InventoryError("%s is on %s and %s" % (pid, seen[pid], s.shelf_id))
                seen[pid] = s.shelf_id
                p = self.packages[pid]
                want = {"stored": "stored", "in_transit": "in_transit", "dropped": "dropped"}[s.status]
                if p.shelf_id != s.shelf_id or p.status != want:
                    raise InventoryError("%s on %s has status %s / shelf %s" % (pid, s.shelf_id, p.status, p.shelf_id))
            for pid in s.incoming:
                if self.packages[pid].status != "arrived":
                    raise InventoryError("%s holds a slot on %s but is %s" % (pid, s.shelf_id, self.packages[pid].status))
        counts = {"arrived": 0, "on_shelf": 0, "shipped": 0, "cancelled": 0}
        for p in self.packages.values():
            if p.status in ("stored", "in_transit", "dropped"):
                if seen.get(p.package_id) != p.shelf_id:
                    raise InventoryError("%s says it is on %s but that shelf does not hold it" % (p.package_id, p.shelf_id))
                counts["on_shelf"] += 1
            else:
                counts[p.status] += 1
        if (counts["shipped"], counts["cancelled"]) != (self.shipped, self.cancelled) or \
                sum(counts.values()) != self.created:
            raise InventoryError("package conservation violated: %r, created %d, shipped %d, cancelled %d"
                                 % (counts, self.created, self.shipped, self.cancelled))

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for p in self.packages.values():
            out[p.status] = out.get(p.status, 0) + 1
        return out
