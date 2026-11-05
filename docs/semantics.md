# Simulation semantics (model version 2.0)

This page is part of the public contract of Battery-TA-RWARE. Results obtained with the simulator
depend on these rules, so **report the package version (and this model version) with every
result**. A change to any rule below is released as a new minor model version (2.x) and listed in
[CHANGELOG.md](../CHANGELOG.md). A change that breaks existing configurations or trajectories gets a
new major version.

Version 2 is a new engine. It is **not trajectory-compatible** with TA-RWARE / Battery-TA-RWARE v1
(git tag `legacy-v1.1.0`, branch `v1`), or with any study that ran on the v1 kernel. Its movement,
task, energy and observation models are all different.

## Time and units

- Time advances in discrete ticks. `seconds_per_tick` (default 5 s) converts power to energy only.
  It does not change the dynamics.
- Distances are in grid cells. Robots move with 4-connectivity, at most one cell per tick.
- Power is in W, energy in Wh, payload in kg. Coordinates are `(row, col)`.

## Order of one tick

`step(actions)` advances the tick counter by one and then runs these phases in a fixed order:

1. **Actions.** Robots are processed in ascending id order. In macro mode the task board is
   recomputed after every join, so an earlier robot's join changes what a later robot's slot
   index refers to. A rejected action counts in `info["invalid_actions"]` and has no effect.
2. **Breakdowns.** A failed robot counts down its repair time. An operational robot fails with
   probability `failure.rate`. A failure stops charging, keeps the robot's commitment and freezes
   the robot in place.
3. **Movement**: PIBT in macro mode, simultaneous moves in primitive mode (see below).
4. **Stage work.** Tasks are processed in id order. Completing a task frees its crew and readies
   its dependents.
5. **Charging and energy.** Energy is integrated over the tick from what happened in phases 3 and 4.
   Robots can become full or stranded here.
6. **Demand.** New tasks and jobs are released. Tasks released in tick *t* can be acted on from tick
   *t + 1*.
7. **Rewards and episode end.**

Events in `info["events"]` carry the tick in which they happened. Within a tick they appear in the
order above.

## Movement

**Macro mode.** Each robot has an implicit target derived from its commitment: a stage cell (or a
standing cell within `radius` for `adjacent` roles), a charger, or a parking/queueing cell. The
robot moves towards it with PIBT (Okumura et al., AIJ 2022):

- Priority is the number of ticks spent travelling towards the current target, with ties broken by
  robot id.
- Candidate cells are ordered by exact remaining distance. This is a reverse BFS within the robot's
  allowed zones (`zones`, or `zones_loaded` while carrying) that respects one-way cells. Ties prefer
  unoccupied cells.
- The occupant of a wanted cell inherits the priority and must move away first, recursively. A
  pushed robot may not step into its pusher's cell.
- A restricted swap operation (after Okumura, IJCAI 2023) is applied only when the wanted cell
  holds a robot trapped in a dead end. The implementation was written for this package.
- The following units are fixed obstacles for the tick: failed robots, charging robots, robots
  working a stage, and robots without enough speed credit.
- Joint carriers move as one unit at the speed of their slowest member.

**Primitive mode.** Every robot's intended move is resolved simultaneously, in the same tick:

- A stationary occupant blocks a move into its cell.
- Among robots moving into the same cell, the lowest id wins.
- Swaps (two robots exchanging cells) are rejected.
- Carriers of the same task may share a cell. No other robots may.

**Speed.** Each tick a unit gains `speed` credit, capped at 1, and moves only with at least 1
credit. A speed of 0.5 therefore means one move every two ticks.

## Tasks

**Lifecycle**: `pending → ready → assigned → active → completed`, plus `suspended`, `failed` and
`cancelled`. The allowed transitions are in `battery_tarware.tasks.ALLOWED_TRANSITIONS`. Every
transition is recorded with its tick, and suspended, failed or cancelled transitions also record a
reason.

- `pending`: released, but its dependencies are not complete.
- `ready`: it can be assigned.
- `assigned`: at least one robot holds one of its roles.
- `active`: the first stage has started.
- `suspended`: the whole crew left after work had started. The task keeps its progress and resumes
  when a new crew arrives.
- A crew released before any work started returns the task to `ready`. This is a reassignment, not
  a failure.

**Stages**:

- Stages run in order. A stage advances only in ticks in which every role it needs is filled, and
  every member is present (`on_cell`: on the stage cell; `adjacent`: within `radius`, Chebyshev
  distance) and operational.
- Progress per tick is the smallest `handling_speed` in the crew. A stage completes when progress
  reaches `duration`.
- After a stage, roles that are not needed in the next stage are released (unless `stay: true`), as
  are roles that no later stage needs.

**Payload**:

- The payload is picked up when the first stage completes and placed when the last one completes.
- A carrier that strands or is released while it has failed drops the payload on its current cell.
- A new carrier crew must gather on that cell before work continues. The `payload_picked` event then
  has `reload=True`.
- An operational robot cannot be released while it is carrying.

**Rewards**:

- A completed task pays `task_completed × priority + late_per_tick × lateness`, split equally across
  the crew at completion time. Lateness is the number of ticks past the deadline.
- A completed stage pays `stage_completed`, split across that stage's crew.

## Energy

The power drawn in a tick is the sum of the following terms, multiplied by
`seconds_per_tick / 3600` to give energy:

| Term | When |
|---|---|
| `idle_w` | every tick in which the robot is not stranded, including while failed or charging |
| `move_w` | the robot moved this tick |
| `payload_w_per_kg × weight / n_carriers` | the robot moved while carrying; each carrier pays for its share |
| `handling_w` | the robot worked a stage this tick |

**Charging**:

- A robot charges only while it occupies a compatible dock. One robot fits per dock.
- It gains `min(charger power_w, max_charge_w) × charge_efficiency`, and the battery is capped at
  `capacity_wh`.
- Charging stops automatically when the battery is full, and the robot becomes idle. It also stops
  when the robot is released, reassigned, fails or strands.

**Stranding**: a robot whose energy reaches 0 is stranded permanently. It releases its role and
drops its payload, if it has one.

## Episode end

- `terminated`: the demand is finite and every released item is done, or every robot is stranded.
- `truncated`: `max_steps` has been reached and the episode did not terminate.

## Randomness

`reset(seed)` derives five independent streams with `numpy.random.SeedSequence(seed).spawn(5)`:
spawn positions, demand arrivals, task sampling (cells, weights, mixes), failures and observation
noise.

For a given package version, the same seed and the same actions reproduce a run exactly. Changing
one random process (for example the failure rate) does not change the draws of the others.
