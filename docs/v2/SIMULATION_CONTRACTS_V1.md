# Battery-TA-RWARE simulation contracts v1

Status: **draft v1** (2026-11-05), shared by the three v2 tracks (see [ROADMAP.md](ROADMAP.md)).
Written while extracting the kernel from RendeCoord: line references to `planner/...` point into
RendeCoord at tag `bt-g0-baseline` and describe how its coordinator uses the kernel today. Under
the v2 positioning, sections 2-4 and 6 describe the **frozen profiles** exactly; v2 profiles may change
them, with a schema-version bump and a release note.

Change rule: any field added, removed or re-typed bumps `SCHEMA_VERSION`; the old version stays
readable.

## 0. Package and naming

| Name | Meaning |
|---|---|
| `battery-tarware` | distribution name |
| `battery_tarware` | the kernel; must not import `tarware`, RendeCoord code, or any native coordinator |
| `tarware` (this repo) | the frozen public TA-RWARE v1.1.0 Gymnasium environment |
| `tarware` (RendeCoord) | a compatibility alias there (`tarware.X is battery_tarware.X`) for its frozen research scripts |

## 1. Ownership

| State | Owner | Today | Contract |
|---|---|---|---|
| grid layers, layout, goals, chargers, docks | kernel | `Warehouse` fields | read via `StateView` |
| robot pose/dir, battery, carried shelf, `busy`, `path`, `req_action`, `charging` flag, clash/stuck counters | kernel | `Agent` fields | read via `StateView`; written **only** through kernel commands (§4) |
| shelves, package type/weight/requirements, request queue, arrivals | kernel | `Shelf`, `request_queue`, `WarehouseV4` | read via `StateView`; mutation only by the kernel's task system |
| global RNG streams (`random`, `np.random` legacy MT19937) + v4 `default_rng([seed,4004/4005])` | kernel | module globals | kernel-owned `RngStreams`; draw order frozen (§6) |
| missions, assignments, coalitions, return crews, reservations, charger reservations, path-length caches, EdgeState, adaptive stats | coordinator | partly **stored on the env object** (`env._return_crews`, `_calvo_state`, `_coupling_stats`, `_adaptive_*`, `_matching_stats`, `_arbitration_stats`, `_joint_resilience_reserve`, `_cpp_step_ctx`) | move to a coordinator context; keep attribute shims on the env until W2d/CF1/G0 replays pass |
| C++ `EnvSnapshot`, `ReservationTable`, `EnergyModel` copies | coordinator (native) | `tarware._cpp_planner` | built from `StateView` by an adapter (§5) |

## 2. Step API (unchanged in P0)

```python
env.reset(seed: int | None = None, options: dict | None = None) -> dict   # returns {} today
env.step(macro_actions: Sequence[int]) -> tuple[bool, dict]              # (done, info)
```
- `macro_actions[i]` is for `env.agents[i]`; AGVs first, then pickers. `0` = NOOP; `1..len(goals)` = goals;
  then shelf / charger locations via `action_id_to_coords_map` (values are `(y, x)`).
- A NOOP to a `busy` robot is **ignored**: the robot keeps walking its `path`. This is why the
  coordinator needs commands (§4).
- `info` keys (frozen): `vehicles_busy, shelf_deliveries, clashes, stucks, agvs_distance_travelled,
  pickers_distance_travelled, agvs_idle_time, pickers_idle_time, battery_levels, actions, pkg_weights,
  total_lifts, charging_states, deliveries_by_pkg_type, carrying_pkg_type`.
- `done`: legacy = inactivity | max_steps | empty queue; v4 with arrivals = inactivity | max_steps.
- Gymnasium / PettingZoo / SMDP signatures are **adapters** (§7), never the kernel API.
- `compute_valid_action_masks()` is broken at the baseline (reads a missing `action_size`): not part of the
  contract until fixed in a dedicated, regression-neutral change.

Frozen step order (do not reorder): `attribute_macro_actions → resolve_move_conflict →
resolve_stuck_agents → execute_micro_actions (agent list order) → process_shelf_deliveries
(→ _replenish_request RNG) → _recalc_grid → _cur_steps += 1 → done → _build_info`; v4 then
records lead times, enqueues due arrivals, samples backlog, recomputes `done`.

## 3. StateView (read-only projection, schema 1)

Not a restorable snapshot. Built on demand; holds no live references. Source of truth:
`battery_tarware/interfaces/state_view.py`, which extends the sketch below with `grid_shape` and
`occupancy` (one little-endian int32 entity-id layer per `CollisionLayers`, row-major [y][x]) and a
static per-episode `LayoutView` (`grid_shape`, `highways`, `goals`, `chargers`,
`action_coords` = (action_id, y, x)). Together they carry every input of the C++ `EnvSnapshot`.

```python
@dataclass(frozen=True)
class RobotView:      id: int; kind: int  # 0 AGV, 1 PICKER
                      y: int; x: int; dir: int; battery: float      # SoC percent, [0, 100]
                      busy: bool; charging: bool; carrying_shelf: int  # 0 = none
                      target: int; path: tuple[tuple[int, int], ...]  # (x, y) cells, as stored today
@dataclass(frozen=True)
class ShelfView:      id: int; y: int; x: int; package_type: int     # PackageType: SOLO 0, STANDARD 1, LARGE 2
                      weight_kg: float; required_agvs: int; required_pickers: int
@dataclass(frozen=True)
class StateView:      schema_version: int; tick: int
                      robots: tuple[RobotView, ...]; shelves: tuple[ShelfView, ...]
                      request_queue: tuple[int, ...]    # shelf ids, queue order
                      goals: tuple[tuple[int, int], ...]; chargers: tuple[tuple[int, int], ...]
```
Conventions: IDs are 1-based and equal `index + 1` **today**; consumers must index by ID, never by
position, so the kernel may change this later. Coordinates are stated per field ((x, y) for `path`
and `goals` as stored; (y, x) elsewhere) and never silently swapped. Enum ints are `.value` (`AgentType`, `Action`, `Direction` are plain `Enum`; `PackageType`,
`CollisionLayers` are `IntEnum`). Directions: UP 0, DOWN 1, LEFT 2, RIGHT 3.

## 4. Commands (kernel-validated state changes)

Exactly the writes the coordinator performs today (inventory at `bt-g0-baseline`: 29 sites in
`planner/coordinator.py`, 4 in `safety.py`, 2 in `joint_om_lagrangian.py`, 1 comment in `rules.py`):

| Command | Effect (bit-identical to today) | Sites |
|---|---|---|
| `cancel_motion(robot_id)` | `busy=False; path=[]` | coordinator 1154,1354,1441,1452,1475,1541,1593,1644,1652,1716,1787; safety 181,302; joint_om 218 |
| `cancel_motion(robot_id, clear_charge=True)` | `charging=False; busy=False; path=[]` | coordinator 1399 |
| `clear_charge_intent(robot_id)` | `charging=False` | coordinator 1693 |
| `arm_charging(robot_id)` | `path=[]; charging=True` (robot must be on a charger cell; next step issues CHARGE) | coordinator 1766 |

Migration rule: phase 1 commands **take effect immediately** (method call replacing the attribute
write, same position in the coordinator's code) and emit an event. Deferring them into a batch
applied inside `step()` changes what the coordinator reads later in the same tick, so it is a
behavior change and needs its own regression gate. `ExecutionResult(ok, reason)` is returned;
in phase 1 every call is accepted (no new rejections = no behavior change).

## 5. C++ boundary

The existing `_cpp_planner.EnvSnapshot` (`agents`, `request_queue`, `goals`,
`charging_station_xy`, `charging_station_action_ids`, `set_layers`, `set_coord_to_action`,
`rebuild_indices`), built by `planner/cpp_bridge.py::StepContext` and `grid_assembler.cpp`, is
**DTO v1**. Work item: document each field's dtype, units, coordinate order and lifetime, and build it
from `StateView` instead of live `Warehouse` objects. Native duplicates of kernel semantics are a
parity surface and must be listed with their Python counterpart: `astar_path`/`find_path_len` ↔
`tarware.astar` + `Warehouse.find_path`; `EnergyModel` ↔ `energy_coupling.EnergyModel` +
`_BATTERY_*` constants (hand-synced header); `ReservationTable` ↔ `planner/reservations.py`.
Renaming `tarware._cpp_planner` and splitting the wheel happen after this documentation.

## 6. Determinism

- Global streams are seeded in `reset()` (`np.random.seed(s); random.seed(s)`). The kernel is the only
  consumer of those streams at baseline (planner uses private `random.Random`). A refactor must keep
  legacy `RandomState`/`Random` objects with identical call order; switching generator changes
  every trajectory.
- Track B/C features draw only from their own `np.random.default_rng([seed, <unique tag>])` streams.
- G0 fingerprints hash both stream states each tick, so an extra or reordered draw is detected.

## 7. Event (schema 1)

```python
@dataclass(frozen=True)
class Event:  episode_id: str; tick: int        # tick = env._cur_steps AFTER the step that caused it
              seq: int                          # order within the tick, deterministic
              type: str; robot_ids: tuple[int, ...]; shelf_id: int | None
              task_id: int | None; payload: Mapping[str, object]
```
Kernel event types (phase 1): `request_enqueued, shelf_loaded, shelf_unloaded, delivered,
charge_started, charge_stopped, clash, stuck_reset, command_applied`. `delivered` (request leaves
queue) and `shelf_unloaded` (shelf back on a rack cell) are distinct. Coordinator-owned events
(assignment, coalition, return crew) stay in RendeCoord until a coordinator contract exists; the
kernel never infers them from positions. Event recording draws no RNG and changes no state;
G0 fingerprints must be identical with recording on and off.

## 8. Track B / C interfaces (develop against fixtures)

- `TaskSpec(id, kind, required: Mapping[capability, int], location(s), arrival_tick, deadline=None,
  depends_on=())` with DAG validation (cycles, infeasible capability demand vs fleet).
  Existing package types map as SOLO=(1 AGV, 0 pickers), STANDARD=(1 AGV, 1 picker),
  LARGE=(1 AGV, 2 pickers) (`PACKAGE_REQUIREMENTS`). New task kinds are opt-in profiles; `legacy` and `protocol_v4` profiles
  must keep producing their G0 fingerprints.
- Coordinator protocol: `decide(view: StateView) -> (macro_actions, commands)`; no mutation of kernel
  objects.
- Adapters (Gymnasium, PettingZoo Parallel, SMDP option wrapper, C++ DTO) live in
  `battery_tarware.adapters` behind optional extras; base install imports none of torch / pettingzoo /
  pybind11.

## 9. Gates (summary; details in the G0 reports)

G1 independence (`battery_tarware` imports without `planner/` on path) · G2 per-profile env-only
fingerprint equality · G3 per-backend planner fingerprint equality (Python vs C++ not required to match)
· G4 events reconcile with state · G5 adapters change no default trajectory.
