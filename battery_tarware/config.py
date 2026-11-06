"""Scenario configuration: every robot, task, charger and demand property is data.

A scenario is a :class:`ScenarioConfig`, built from a plain dict (or JSON / YAML file) with
:func:`ScenarioConfig.from_dict`. All validation happens here, so the simulator can assume a
consistent configuration.

Units: time in ticks (``seconds_per_tick`` converts to wall time), distance in grid cells, energy in
watt-hours, power in watts, mass in kilograms.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

CELL_ZONES = ("aisle", "storage", "goal", "dock")   # built-in zone names; maps may add more


class ConfigError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _build(cls, data: Mapping[str, Any], where: str):
    """Construct dataclass ``cls`` from ``data``, rejecting unknown keys (typos must not pass)."""
    if isinstance(data, cls):
        return data
    _check(isinstance(data, Mapping), "%s: expected a mapping, got %r" % (where, type(data).__name__))
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    _check(not unknown, "%s: unknown key(s) %s (allowed: %s)" % (where, unknown, sorted(names)))
    return cls(**dict(data))


# ───────────────────────────────────────────────────────────── robots

@dataclass(frozen=True)
class EnergySpec:
    """Battery and power model of one robot type (defaults: OMRON LD-250-class AMR)."""
    capacity_wh: float = 1843.0          # 25.6 V x 72 Ah
    idle_w: float = 20.0                 # drawn every tick while operational (incl. waiting)
    move_w: float = 142.0                # driving power when not carrying
    payload_w_per_kg: float = 1.7        # extra driving power per kg carried (own share of the load)
    handling_w: float = 60.0             # extra power while working on a task stage
    max_charge_w: float = 1500.0         # robot-side charging limit
    charge_efficiency: float = 0.95

    def validate(self, where: str) -> None:
        _check(self.capacity_wh > 0, "%s: capacity_wh must be > 0" % where)
        for k in ("idle_w", "move_w", "payload_w_per_kg", "handling_w", "max_charge_w"):
            _check(getattr(self, k) >= 0, "%s: %s must be >= 0" % (where, k))
        _check(0 < self.charge_efficiency <= 1, "%s: charge_efficiency must be in (0, 1]" % where)


@dataclass(frozen=True)
class FailureSpec:
    """Random breakdowns: each operational tick a robot fails with probability ``rate``; it is
    then immobile for ``repair_ticks`` ticks (it keeps its commitments unless released)."""
    rate: float = 0.0
    repair_ticks: int = 0

    def validate(self, where: str) -> None:
        _check(0 <= self.rate <= 1, "%s: failure rate must be in [0, 1]" % where)
        _check(self.repair_ticks >= 0, "%s: repair_ticks must be >= 0" % where)
        _check(self.rate == 0 or self.repair_ticks > 0, "%s: failures need repair_ticks > 0" % where)


@dataclass(frozen=True)
class RobotTypeSpec:
    """One robot model. Capabilities are free-form strings (e.g. transport, pick, lift, inspect,
    manipulate) that task roles refer to."""
    name: str
    capabilities: Tuple[str, ...]
    speed: float = 1.0                         # cells per tick, (0, 1]; fractional speeds accumulate
    payload_kg: float = 0.0                    # max load this robot can carry (0 = cannot carry)
    handling_speed: float = 1.0                # stage work rate multiplier (2.0 = twice as fast)
    zones: Tuple[str, ...] = ("aisle", "storage", "goal", "dock")   # zones it may enter
    zones_loaded: Optional[Tuple[str, ...]] = None                  # zones while carrying (None = zones)
    charger_types: Optional[Tuple[str, ...]] = None                 # compatible chargers (None = all)
    energy: EnergySpec = field(default_factory=EnergySpec)
    sensing_range: Optional[int] = None        # Chebyshev cells it observes other robots in (None = all)
    comm_range: Optional[int] = None           # it shares observations with robots this close (None = all, 0 = none)
    localization_noise: float = 0.0            # std (cells) of noise on positions it observes
    failure: FailureSpec = field(default_factory=FailureSpec)

    @staticmethod
    def from_dict(d: Mapping[str, Any], where: str = "robot_type") -> "RobotTypeSpec":
        d = dict(d)
        for k in ("capabilities", "zones", "zones_loaded", "charger_types"):
            if d.get(k) is not None:
                d[k] = tuple(d[k]) if not isinstance(d[k], str) else (d[k],)
        if "energy" in d:
            d["energy"] = _build(EnergySpec, d["energy"], where + ".energy")
        if "failure" in d:
            d["failure"] = _build(FailureSpec, d["failure"], where + ".failure")
        return _build(RobotTypeSpec, d, where)

    def validate(self, zones: Sequence[str]) -> None:
        w = "robot type %r" % self.name
        _check(bool(self.name), "robot type needs a name")
        _check(len(self.capabilities) > 0, "%s: needs at least one capability" % w)
        _check(0 < self.speed <= 1, "%s: speed must be in (0, 1] cells/tick" % w)
        _check(self.payload_kg >= 0, "%s: payload_kg must be >= 0" % w)
        _check(self.handling_speed > 0, "%s: handling_speed must be > 0" % w)
        for zs in (self.zones, self.zones_loaded or ()):
            bad = sorted(set(zs) - set(zones))
            _check(not bad, "%s: unknown zone(s) %s (map has %s)" % (w, bad, sorted(zones)))
        _check("aisle" in self.zones, "%s: must be allowed in 'aisle'" % w)
        for k in ("sensing_range", "comm_range"):
            v = getattr(self, k)
            _check(v is None or v >= 0, "%s: %s must be >= 0 or null" % (w, k))
        _check(self.localization_noise >= 0, "%s: localization_noise must be >= 0" % w)
        self.energy.validate(w)
        self.failure.validate(w)


@dataclass(frozen=True)
class RobotGroupSpec:
    """``count`` robots of ``type``; ``overrides`` changes fields for this group only (heterogeneity
    within a type, e.g. an older battery), and ``initial_soc`` is a value or a [low, high] range."""
    type: str
    count: int = 1
    name: Optional[str] = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    initial_soc: Any = 1.0

    def soc_range(self) -> Tuple[float, float]:
        v = self.initial_soc
        lo, hi = (float(v), float(v)) if isinstance(v, (int, float)) else (float(v[0]), float(v[1]))
        return lo, hi


# ───────────────────────────────────────────────────────────── chargers

@dataclass(frozen=True)
class ChargerTypeSpec:
    name: str
    power_w: float
    share: float = 1.0          # fraction of docks of this type (normalized over all types)

    def validate(self) -> None:
        _check(self.power_w > 0, "charger type %r: power_w must be > 0" % self.name)
        _check(self.share > 0, "charger type %r: share must be > 0" % self.name)


# ───────────────────────────────────────────────────────────── tasks

@dataclass(frozen=True)
class RoleSpec:
    """A slot in a task crew: ``count`` robots with ``capability``. ``presence`` says where they must
    be during a stage: ``on_cell`` (the stage cell itself) or ``adjacent`` (within ``radius``).
    ``stay``: keep the same robots committed between the stages this role takes part in (otherwise
    they are released after a stage and the role is filled again for its next stage). The carrier
    role always stays: it holds the payload."""
    name: str
    capability: str
    count: int = 1
    presence: str = "adjacent"
    radius: int = 1
    stay: bool = False

    def validate(self, where: str) -> None:
        _check(self.count >= 1, "%s role %r: count must be >= 1" % (where, self.name))
        _check(self.presence in ("on_cell", "adjacent"), "%s role %r: presence must be on_cell|adjacent" % (where, self.name))
        _check(self.radius >= 0, "%s role %r: radius must be >= 0" % (where, self.name))


@dataclass(frozen=True)
class StageSpec:
    """One synchronized step of a task: all ``roles`` present at ``at`` for ``duration`` ticks of
    work (scaled by the slowest participant's handling_speed). ``at`` is ``origin``,
    ``destination`` or ``zone:<name>``."""
    name: str
    at: str
    roles: Tuple[str, ...]
    duration: int = 1

    def validate(self, where: str, role_names: Sequence[str], zones: Sequence[str]) -> None:
        w = "%s stage %r" % (where, self.name)
        _check(self.duration >= 1, "%s: duration must be >= 1" % w)
        _check(len(self.roles) > 0, "%s: needs at least one role" % w)
        bad = sorted(set(self.roles) - set(role_names))
        _check(not bad, "%s: unknown role(s) %s" % (w, bad))
        if self.at.startswith("zone:"):
            _check(self.at[5:] in zones, "%s: unknown zone %r" % (w, self.at[5:]))
        else:
            _check(self.at in ("origin", "destination"), "%s: at must be origin|destination|zone:<name>" % w)


@dataclass(frozen=True)
class Distribution:
    """``fixed`` (value), ``uniform`` (low, high), ``normal`` (mean, std, clipped to [low, high]),
    or ``choice`` (values, optional weights)."""
    kind: str = "fixed"
    value: float = 0.0
    low: float = 0.0
    high: float = math.inf
    mean: float = 0.0
    std: float = 0.0
    values: Tuple[float, ...] = ()
    weights: Optional[Tuple[float, ...]] = None

    @staticmethod
    def from_any(v: Any, where: str) -> "Distribution":
        if isinstance(v, Distribution):
            return v
        if isinstance(v, (int, float)):
            return Distribution(kind="fixed", value=float(v))
        d = dict(v)
        for k in ("values", "weights"):
            if d.get(k) is not None:
                d[k] = tuple(float(x) for x in d[k])
        return _build(Distribution, d, where)

    def validate(self, where: str) -> None:
        _check(self.kind in ("fixed", "uniform", "normal", "choice"), "%s: unknown distribution %r" % (where, self.kind))
        if self.kind == "uniform":
            _check(math.isfinite(self.high) and self.low <= self.high, "%s: uniform needs low <= high" % where)
        if self.kind == "normal":
            _check(self.std >= 0 and self.low <= self.high, "%s: normal needs std >= 0, low <= high" % where)
        if self.kind == "choice":
            _check(len(self.values) > 0, "%s: choice needs values" % where)
            _check(self.weights is None or (len(self.weights) == len(self.values) and sum(self.weights) > 0),
                   "%s: weights must match values" % where)

    def sample(self, rng) -> float:
        if self.kind == "fixed":
            return float(self.value)
        if self.kind == "uniform":
            return float(rng.uniform(self.low, self.high))
        if self.kind == "normal":
            return float(min(self.high, max(self.low, rng.normal(self.mean, self.std))))
        p = None if self.weights is None else [w / sum(self.weights) for w in self.weights]
        return float(self.values[int(rng.choice(len(self.values), p=p))])

    @property
    def max_value(self) -> float:
        if self.kind == "fixed":
            return self.value
        if self.kind == "choice":
            return max(self.values)
        return self.high if math.isfinite(self.high) else self.mean + 6 * self.std


@dataclass(frozen=True)
class TaskTypeSpec:
    """A task family. The crew is a set of roles; stages run in order; if ``carrier_role`` is set,
    the payload (weight from ``weight``) is picked up at the first stage and carried by that role
    between stage locations. A carrier role with count k > 1 is joint transport: the k carriers move
    as one rigid unit and their payload capacities add up."""
    name: str
    roles: Tuple[RoleSpec, ...]
    stages: Tuple[StageSpec, ...]
    carrier_role: Optional[str] = None
    weight: Distribution = field(default_factory=Distribution)
    priority: float = 1.0
    deadline: Optional[int] = None          # ticks after release; None = no deadline
    origin: str = "storage"                 # zone the origin cell is drawn from
    destination: str = "goal"               # zone the destination cell is drawn from

    @staticmethod
    def from_dict(d: Mapping[str, Any], where: str = "task_type") -> "TaskTypeSpec":
        d = dict(d)
        w = where + " %r" % d.get("name")
        d["roles"] = tuple(_build(RoleSpec, r, w + ".roles") for r in d.get("roles", ()))
        stages = []
        for s in d.get("stages", ()):
            s = dict(s)
            s["roles"] = tuple(s.get("roles", ())) if not isinstance(s.get("roles"), str) else (s["roles"],)
            stages.append(_build(StageSpec, s, w + ".stages"))
        d["stages"] = tuple(stages)
        if "weight" in d:
            d["weight"] = Distribution.from_any(d["weight"], w + ".weight")
        return _build(TaskTypeSpec, d, where)

    def role(self, name: str) -> RoleSpec:
        for r in self.roles:
            if r.name == name:
                return r
        raise KeyError(name)

    @property
    def crew_size(self) -> int:
        return sum(r.count for r in self.roles)

    def validate(self, zones: Sequence[str]) -> None:
        w = "task type %r" % self.name
        _check(len(self.roles) > 0 and len(self.stages) > 0, "%s: needs roles and stages" % w)
        names = [r.name for r in self.roles]
        _check(len(set(names)) == len(names), "%s: duplicate role names" % w)
        for r in self.roles:
            r.validate(w)
        for s in self.stages:
            s.validate(w, names, zones)
        used = {r for s in self.stages for r in s.roles}
        _check(used == set(names), "%s: role(s) %s never appear in a stage" % (w, sorted(set(names) - used)))
        self.weight.validate(w + ".weight")
        _check(self.weight.kind != "fixed" or self.weight.value >= 0, "%s: weight must be >= 0" % w)
        if self.carrier_role is not None:
            _check(self.carrier_role in names, "%s: carrier_role %r is not a role" % (w, self.carrier_role))
            for s in self.stages:
                _check(self.carrier_role in s.roles,
                       "%s: the carrier role must take part in every stage (it holds the payload)" % w)
        _check(self.origin in zones and self.destination in zones, "%s: origin/destination zone unknown" % w)
        _check(self.deadline is None or self.deadline > 0, "%s: deadline must be > 0" % w)


@dataclass(frozen=True)
class JobTaskSpec:
    key: str
    type: str
    depends_on: Tuple[str, ...] = ()
    origin_from: Optional[str] = None        # "<key>.origin" or "<key>.destination"
    destination_from: Optional[str] = None


@dataclass(frozen=True)
class JobSpec:
    """A multi-stage job: several tasks with a dependency DAG; locations can be chained, e.g. a
    storage task whose origin is the receiving task's destination."""
    name: str
    tasks: Tuple[JobTaskSpec, ...]

    @staticmethod
    def from_dict(d: Mapping[str, Any], where: str = "job") -> "JobSpec":
        d = dict(d)
        ts = []
        for t in d.get("tasks", ()):
            t = dict(t)
            t["depends_on"] = tuple(t.get("depends_on", ()))
            ts.append(_build(JobTaskSpec, t, where + ".tasks"))
        d["tasks"] = tuple(ts)
        return _build(JobSpec, d, where)

    def validate(self, task_types: Sequence[str]) -> None:
        w = "job %r" % self.name
        keys = [t.key for t in self.tasks]
        _check(len(keys) > 0 and len(set(keys)) == len(keys), "%s: needs unique task keys" % w)
        for t in self.tasks:
            _check(t.type in task_types, "%s: unknown task type %r" % (w, t.type))
            _check(set(t.depends_on) <= set(keys) and t.key not in t.depends_on, "%s: bad depends_on in %r" % (w, t.key))
            for ref in (t.origin_from, t.destination_from):
                if ref is not None:
                    k, _, part = ref.partition(".")
                    _check(k in keys and k != t.key and part in ("origin", "destination"),
                           "%s: %r must reference '<other key>.origin|destination'" % (w, ref))
        # acyclic
        deps = {t.key: set(t.depends_on) for t in self.tasks}
        done: set = set()
        while len(done) < len(deps):
            ready = [k for k, d in deps.items() if k not in done and d <= done]
            _check(bool(ready), "%s: dependency cycle" % w)
            done.update(ready)


@dataclass(frozen=True)
class DemandStream:
    """Where tasks come from. ``kind``:
    ``backlog``  keep ``open`` released items available (replenished on completion);
    ``poisson``  Poisson arrivals at ``rate`` per tick (``rates`` = [[tick, rate], ...] for a
                 piecewise schedule, e.g. peaks and troughs);
    ``batch``    ``count`` items at tick 0;
    ``trace``    explicit ``[tick, ...]`` release times.
    An item is one task of ``task_type`` or one ``job``; ``mix`` = {name: weight} draws the
    item among several types."""
    kind: str = "backlog"
    task_type: Optional[str] = None
    job: Optional[str] = None
    mix: Optional[Mapping[str, float]] = None
    open: int = 10
    rate: float = 0.0
    rates: Optional[Tuple[Tuple[int, float], ...]] = None
    count: int = 0
    trace: Tuple[int, ...] = ()
    max_items: Optional[int] = None

    def items(self) -> Dict[str, float]:
        if self.mix:
            return dict(self.mix)
        return {(self.task_type or self.job): 1.0}

    def validate(self, task_types: Sequence[str], jobs: Sequence[str]) -> None:
        w = "demand stream %r" % (self.task_type or self.job or self.mix)
        _check(self.kind in ("backlog", "poisson", "batch", "trace"), "%s: unknown kind %r" % (w, self.kind))
        given = sum(x is not None for x in (self.task_type, self.job, self.mix))
        _check(given == 1, "%s: give exactly one of task_type, job, mix" % w)
        for name, wt in self.items().items():
            _check(name in task_types or name in jobs, "%s: unknown task type / job %r" % (w, name))
            _check(wt > 0, "%s: mix weights must be > 0" % w)
        if self.kind == "backlog":
            _check(self.open >= 1, "%s: backlog needs open >= 1" % w)
        if self.kind == "poisson":
            _check(self.rate > 0 or self.rates, "%s: poisson needs rate or rates" % w)
            if self.rates:
                ticks = [t for t, _ in self.rates]
                _check(ticks == sorted(ticks) and ticks[0] == 0, "%s: rates must start at tick 0, sorted" % w)
                _check(all(r >= 0 for _, r in self.rates), "%s: rates must be >= 0" % w)
        if self.kind == "batch":
            _check(self.count >= 1, "%s: batch needs count >= 1" % w)


# ───────────────────────────────────────────────────────────── map, rewards, observations

DOCK_LAYOUTS = ("inline", "bays")
MODELS = ("2.0", "2.1")                 # simulation model versions (docs/semantics.md)


@dataclass(frozen=True)
class MapSpec:
    """``ascii`` rows using the legend ``.`` aisle, ``S`` storage, ``G`` goal, ``C`` charger dock,
    ``#`` wall, plus ``zones`` = {symbol: zone name} for custom zones (e.g. receiving, inspection).
    ``ascii`` is one string or a list of row strings.
    Or a generated layout: ``generator`` = {rack_rows, rack_cols, rack_height, goals, docks}.
    ``dock_layout``: ``inline`` (docks sit in a row robots drive along, as drawn) or ``bays`` (every
    row made only of docks and aisle has its aisle cells walled off, so each dock is a one-cell bay
    off the corridor next to it; model 2.1)."""
    ascii: Optional[str] = None
    zones: Mapping[str, str] = field(default_factory=dict)
    generator: Optional[Mapping[str, int]] = None
    dock_layout: str = "inline"

    def __post_init__(self):
        if isinstance(self.ascii, (list, tuple)):
            object.__setattr__(self, "ascii", "\n".join(str(row) for row in self.ascii))

    def validate(self) -> None:
        _check((self.ascii is None) != (self.generator is None), "map: give exactly one of ascii, generator")
        _check(self.dock_layout in DOCK_LAYOUTS, "map: dock_layout must be one of %s" % (DOCK_LAYOUTS,))
        for sym, zone in self.zones.items():
            _check(len(sym) == 1 and sym not in ".SGC#", "map: custom zone symbol %r must be one new character" % sym)
            _check(zone not in CELL_ZONES and zone != "wall", "map: custom zone name %r clashes with a built-in" % zone)


@dataclass(frozen=True)
class RewardSpec:
    task_completed: float = 1.0          # x task priority
    stage_completed: float = 0.0
    energy_per_kwh: float = 0.0          # usually negative
    late_per_tick: float = 0.0           # per tick of lateness at completion, usually negative
    stranded: float = -1.0               # once, when a robot's battery hits 0
    per_tick: float = 0.0


@dataclass(frozen=True)
class ObservationSpec:
    task_slots: int = 16                 # open tasks shown (by release order); also macro task actions
    task_visibility: str = "global"      # global task board, or "sensing" (only tasks within sensing range)

    def validate(self) -> None:
        _check(self.task_slots >= 1, "observation.task_slots must be >= 1")
        _check(self.task_visibility in ("global", "sensing"), "observation.task_visibility must be global|sensing")


# ───────────────────────────────────────────────────────────── scenario

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    map: MapSpec
    robot_types: Tuple[RobotTypeSpec, ...]
    fleet: Tuple[RobotGroupSpec, ...]
    task_types: Tuple[TaskTypeSpec, ...]
    demand: Tuple[DemandStream, ...]
    charger_types: Tuple[ChargerTypeSpec, ...] = (ChargerTypeSpec("standard", 1000.0),)
    jobs: Tuple[JobSpec, ...] = ()
    action_mode: str = "macro"           # macro | primitive
    max_steps: int = 5000
    seconds_per_tick: float = 5.0
    rewards: RewardSpec = field(default_factory=RewardSpec)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    description: str = ""
    model: str = "2.1"                   # simulation model version; "2.0" reproduces model 2.0 exactly

    # ---- construction
    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ScenarioConfig":
        d = dict(d)
        _check("name" in d, "scenario needs a name")
        m = dict(d.get("map", {}))
        if "zones" in m:
            m["zones"] = dict(m["zones"])
        d["map"] = _build(MapSpec, m, "map")
        d["robot_types"] = tuple(RobotTypeSpec.from_dict(r, "robot_types[%d]" % i)
                                 for i, r in enumerate(d.get("robot_types", ())))
        fleet = []
        for i, g in enumerate(d.get("fleet", ())):
            g = dict(g)
            g["overrides"] = dict(g.get("overrides", {}))
            fleet.append(_build(RobotGroupSpec, g, "fleet[%d]" % i))
        d["fleet"] = tuple(fleet)
        d["task_types"] = tuple(TaskTypeSpec.from_dict(t, "task_types[%d]" % i)
                                for i, t in enumerate(d.get("task_types", ())))
        d["jobs"] = tuple(JobSpec.from_dict(j, "jobs[%d]" % i) for i, j in enumerate(d.get("jobs", ())))
        streams = []
        for i, s in enumerate(d.get("demand", ())):
            s = dict(s)
            if s.get("rates") is not None:
                s["rates"] = tuple((int(t), float(r)) for t, r in s["rates"])
            if s.get("trace") is not None:
                s["trace"] = tuple(int(t) for t in s["trace"])
            if s.get("mix") is not None:
                s["mix"] = dict(s["mix"])
            streams.append(_build(DemandStream, s, "demand[%d]" % i))
        d["demand"] = tuple(streams)
        if "charger_types" in d:
            d["charger_types"] = tuple(_build(ChargerTypeSpec, c, "charger_types") for c in d["charger_types"])
        if "rewards" in d:
            d["rewards"] = _build(RewardSpec, d["rewards"], "rewards")
        if "observation" in d:
            d["observation"] = _build(ObservationSpec, d["observation"], "observation")
        cfg = _build(ScenarioConfig, d, "scenario")
        cfg.validate()
        return cfg

    @staticmethod
    def load(path: str) -> "ScenarioConfig":
        p = Path(path)
        text = p.read_text()
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover
                raise ConfigError("reading YAML needs pyyaml: pip install 'battery-ta-rware[yaml]'") from e
            return ScenarioConfig.from_dict(yaml.safe_load(text))
        return ScenarioConfig.from_dict(json.loads(text))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **changes) -> "ScenarioConfig":
        cfg = dataclasses.replace(self, **changes)
        cfg.validate()
        return cfg

    # ---- lookups
    def robot_type(self, name: str) -> RobotTypeSpec:
        for r in self.robot_types:
            if r.name == name:
                return r
        raise KeyError(name)

    def task_type(self, name: str) -> TaskTypeSpec:
        for t in self.task_types:
            if t.name == name:
                return t
        raise KeyError(name)

    def job(self, name: str) -> JobSpec:
        for j in self.jobs:
            if j.name == name:
                return j
        raise KeyError(name)

    @property
    def zones(self) -> Tuple[str, ...]:
        return CELL_ZONES + tuple(self.map.zones.values())

    @property
    def capabilities(self) -> Tuple[str, ...]:
        caps = []
        for r in self.robot_types:
            for c in r.capabilities:
                if c not in caps:
                    caps.append(c)
        for t in self.task_types:
            for role in t.roles:
                if role.capability not in caps:
                    caps.append(role.capability)
        return tuple(caps)

    def resolved_robot_specs(self):
        """[(robot name, RobotTypeSpec with group overrides applied, (soc low, soc high))]"""
        out = []
        for g in self.fleet:
            base = self.robot_type(g.type)
            spec = base
            if g.overrides:
                merged = dataclasses.asdict(base)
                for k, v in g.overrides.items():
                    _check(k in merged and k != "name", "fleet group %r: cannot override %r" % (g.name or g.type, k))
                    if k in ("energy", "failure"):
                        merged[k] = {**merged[k], **dict(v)}
                    else:
                        merged[k] = v
                spec = RobotTypeSpec.from_dict(merged, "fleet group %r" % (g.name or g.type))
            gname = g.name or g.type
            for i in range(g.count):
                out.append(("%s_%d" % (gname, i), spec, g.soc_range()))
        return out

    # ---- validation
    def validate(self) -> None:
        self.map.validate()
        _check(self.model in MODELS, "model must be one of %s" % (MODELS,))
        zones = self.zones
        _check(len(self.robot_types) > 0, "scenario needs robot_types")
        names = [r.name for r in self.robot_types]
        _check(len(set(names)) == len(names), "duplicate robot type names")
        chargers = [c.name for c in self.charger_types]
        _check(len(chargers) > 0 and len(set(chargers)) == len(chargers), "charger types must be non-empty, unique")
        for c in self.charger_types:
            c.validate()
        for r in self.robot_types:
            r.validate(zones)
            if r.charger_types is not None:
                bad = sorted(set(r.charger_types) - set(chargers))
                _check(not bad, "robot type %r: unknown charger type(s) %s" % (r.name, bad))
        _check(len(self.fleet) > 0, "scenario needs a fleet")
        for g in self.fleet:
            _check(g.type in names, "fleet: unknown robot type %r" % g.type)
            _check(g.count >= 1, "fleet group %r: count must be >= 1" % (g.name or g.type))
            lo, hi = g.soc_range()
            _check(0 < lo <= hi <= 1, "fleet group %r: initial_soc must be in (0, 1], low <= high" % (g.name or g.type))
        robot_names = [n for n, _, _ in self.resolved_robot_specs()]
        _check(len(set(robot_names)) == len(robot_names), "fleet: robot names collide; give groups distinct names")
        _check(len(self.task_types) > 0, "scenario needs task_types")
        tnames = [t.name for t in self.task_types]
        _check(len(set(tnames)) == len(tnames), "duplicate task type names")
        specs = [s for _, s, _ in self.resolved_robot_specs()]
        for t in self.task_types:
            t.validate(zones)
            for role in t.roles:
                capable = [s for s in specs if role.capability in s.capabilities]
                _check(len(capable) >= role.count, "task type %r: role %r needs %d robot(s) with %r, fleet has %d"
                       % (t.name, role.name, role.count, role.capability, len(capable)))
            if t.carrier_role is not None:
                role = t.role(t.carrier_role)
                caps = sorted((s.payload_kg for s in specs if role.capability in s.capabilities), reverse=True)
                best = sum(caps[:role.count])
                _check(best >= t.weight.max_value or t.weight.kind == "normal",
                       "task type %r: the strongest %d carrier(s) lift %.1f kg < max weight %.1f kg"
                       % (t.name, role.count, best, t.weight.max_value))
        jnames = [j.name for j in self.jobs]
        _check(len(set(jnames)) == len(jnames) and not (set(jnames) & set(tnames)), "job names must be unique and differ from task types")
        for j in self.jobs:
            j.validate(tnames)
        _check(len(self.demand) > 0, "scenario needs at least one demand stream")
        for s in self.demand:
            s.validate(tnames, jnames)
        _check(self.action_mode in ("macro", "primitive"), "action_mode must be macro|primitive")
        _check(self.max_steps >= 1 and self.seconds_per_tick > 0, "max_steps must be >= 1 and seconds_per_tick > 0")
        self.observation.validate()
