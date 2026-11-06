# Design: order management, dependency-aware scheduling, allocation and evaluation

Status: **draft for review** (branch `feat/orders-scheduling`, based on `feat/capability-alternatives`).
Nothing here changes Battery-TA-RWARE 2.1.0 or model 2.1 behaviour: every new feature is opt-in, and
a configuration without the new keys runs tick for tick as in 2.1.0 (checked by the existing
trajectory-identity test, extended to the new code paths).

Decisions that need the maintainer's confirmation are marked **[D1]–[D6]**.

## 0. What exists and what is new

| Area | In 2.1.0 | New |
|---|---|---|
| Work units | `Task` (typed roles, stages, payload weight); `Job` = DAG of tasks with chained cells | `Order`, `Package`, `TaskGroup` with persistent ids; inventory of storage locations |
| Lifecycle | pending, ready, assigned, active, suspended, failed, completed, cancelled; reasons for suspend/fail/cancel; retry budget (never used) | `blocked`, `retrying`, failure and retry paths actually exercised; reasons for every suspension, failure, retry, cancellation |
| Scheduling | `RULES` (FIFO, EDD, SPT, CR, priority) + `feasible_coalitions` + greedy coordinator | explicit five-step pipeline; nine rules (FIFO, EDD, LWF, WSP, NSP, ZLP, PLB, SPT, CR) on one eligible set; decision log |
| Metrics | 10 episode totals in `sim.metrics()` | order, scheduling, robot/coalition and system levels with precise definitions; censoring of unfinished work |
| Visualisation | per-frame renderer (`rgb_array`), README GIFs | offline dashboard with filters, linked to events, decisions and frames |
| Evaluation | `run_episode` | paired, seeded harness with workload traces and bootstrap CIs |

The rule set follows the open-source *warehouse-multi-robot-scheduler* (MIT, also derived from
TA-RWARE), which registers these nine rules plus RANDOM, ROUND_ROBIN, GA and SA. Several of its
rules are project-specific heuristics (3D layers, fixed weights), so every rule below is defined
for the 2D Battery-TA-RWARE grid and implemented independently; the docs cite that project as the
source of the rule set.

## 1. Entities

All ids are assigned at release in release order, are unique within an episode and never reused.

- **Order** (`order_id`): an external request of one operation type, with `arrival` tick,
  `priority`, optional `due` tick, and one or more lines. It is complete when all its tasks are
  completed; it fails when a task fails beyond its retry budget, or it is cancelled.
- **Package** (`package_id`): a physical unit (tote, case or pallet) with weight and a current
  location: a storage location, a robot (while carried), a goal/inbound cell, or "delivered".
  Every location change is an event.
- **Storage location**: a storage cell of the map that holds at most one package (**[D1]**: one
  package per storage cell; TA-RWARE's shelf = location). Initial inventory is generated from the
  seed (fill ratio configurable).
- **Task** (`task_id`): as today (typed roles, stages, coalition), plus `order_id`, `package_id`,
  `group_id`, `kind` (`inbound` / `outbound` / `relocation`), `depends_on`.
- **TaskGroup** (`group_id`): the tasks one order line generates (e.g. a relocation that frees
  access, then the retrieval). Group status is derived from its tasks, never set by hand.

### Operation types (2D definitions)

- **Inbound**: a package appears at an inbound cell (zone `inbound`, or goals if the map has none).
  Task: carry it to a free storage location chosen at assignment time by the configured slotting
  rule (nearest free location by default) and reserved until the task ends.
- **Outbound**: a package in storage is ordered. Task: carry it to a goal (with the task type's
  crew, e.g. AGV plus picker); the location becomes free when the payload is picked up. Optional
  return leg as in the built-in task types.
- **Relocation**: move a package to a free storage location, for one of three reasons:
  (a) **access**: an outbound or inbound task's target location cannot be reached by a loaded
  robot or is blocked by stored packages (the order generator creates the relocation as a
  prerequisite of that task);
  (b) **consolidation/slotting**: requested by the order stream;
  (c) **prerequisite**: an inbound order whose target zone is full.
  Access in 2D (**[D2]**): a location is accessible if a loaded carrier can reach an aisle cell
  from it through cells that are passable for loaded robots and hold no package. In the
  generated maps storage cells border an aisle, so access relocations only occur in maps (or
  inventory states) where that is not the case; the definition makes them explicit and testable.

## 2. Task states

Opt-in profile `lifecycle: extended` (the default profile keeps today's state machine exactly).

| State | Meaning | Enters from |
|---|---|---|
| pending | known, not yet released to scheduling (order arrival, or release delay) | – |
| blocked | released, but a prerequisite task, inventory condition (no free location, package not yet stored) or access condition is unmet; reason recorded | pending, retrying |
| ready | eligible for assignment | pending, blocked, assigned (reassignment) |
| assigned | crew committed, work not started | ready |
| executing | first stage started (today's `active`) | assigned, suspended |
| suspended | crew released after work started; keeps progress; reason recorded | assigned, executing |
| failed | attempt failed (crew stranded, payload unreachable, timeout); reason recorded | assigned, executing, suspended |
| retrying | failed with budget left: waits `retry_delay` ticks, then rechecks prerequisites | failed |
| completed / cancelled | terminal; cancellation reason recorded | – |

Retry limits: `retry_budget` per task type (default 0), `retry_delay` ticks. Failure detectors
(**[D3]**, all configurable, all off by default): stranded crew member; no progress for
`stall_timeout` ticks while assigned; payload location unreachable.

## 3. Scheduling architecture

Every tick, a `Scheduler` runs five separate steps; each is a replaceable component.

1. **Release and prioritisation**: orders arrive (Poisson, trace, batch); tasks are generated and
   enter `pending` → `blocked`/`ready`.
2. **Readiness**: dependency DAG, inventory and access conditions; produces the **eligible set**.
3. **Feasibility**: for each eligible task, the feasible coalitions (capabilities, role
   alternatives and work positions, payload, zones, battery with the common reserve, robot
   availability). Infeasible tasks stay in the set with their rejection reasons.
4. **Assignment**: the rule ranks (task, robot or coalition) pairs **only within the feasible
   pairs** and commits assignments atomically.
5. **Execution and recovery**: the simulator executes; suspensions, failures, retries and
   reassignments flow back into steps 2–4.

Charging is a separate, shared component (the common rule used by the reference coordinators, or
the minimum-cost dock matching), so rules differ only in step 4.

### Rule definitions

Notation: now `t`; task `j` with release `a_j` (time it became ready for the last time: `r_j`),
due `d_j` (∞ if none), priority `w_j`, processing estimate `p_j` (remaining stage work plus carried
legs, as `spt` today), and for a robot/coalition `c` the arrival estimate `e_jc` (travel time of the
crew to the first work cell). Ties always break by task id, then robot id.

| Rule | Ranks | Key (smallest first unless stated) |
|---|---|---|
| FIFO | tasks | `(a_j)` |
| EDD | tasks | `(d_j, a_j)` |
| LWF | tasks | largest `t - r_j` (waiting since it last became ready: differs from FIFO after blocking, suspension, reassignment or retry) |
| WSP | tasks | largest `α·wait_norm + β·type_weight + γ·w_norm`; wait and priority min-max normalised over the eligible set; α, β, γ and type weights configurable (defaults **[D4]**) |
| SPT | tasks | `(p_j, a_j)` |
| CR | tasks | `((d_j - t) / max(p_j, 1), a_j)`; tasks without due date last |
| NSP | pairs | `(e_jc, a_j)`: for each free robot (id order), the nearest feasible task; for coalitions, the crew with the smallest max arrival |
| ZLP | tasks | 2D zone priority (**[D5]**): zones are named map zones and rack blocks (storage cells grouped by the rack block they belong to, numbered by row then column); the config gives a zone order (e.g. inbound before outbound docks, or blocks nearest the goals first); key `(zone_rank(j), d_j - t, a_j)` |
| PLB | pairs | 2D load balancing (**[D5]**): the map is partitioned into regions (rack-block columns plus the goal row); region load = robots committed to tasks whose current work cell is in the region; key `(load(region_j) / capacity(region_j), e_jc, a_j)`; optional congestion term from PIBT stall counts |

Rules that rank tasks (FIFO … CR) choose the task; the crew is then the feasible coalition with
the earliest estimated completion (the greedy's choice), so they differ only in sequencing. NSP and
PLB rank pairs and so also choose the crew.

### Decision log

One record per assignment attempt: tick, rule, eligible task ids with their keys, feasible
coalitions per task with estimated completion time and energy, the selected assignment, and for
every rejected task or coalition a reason code (`not_ready:<cause>`, `infeasible:<capability|
payload|battery|zone|unreachable>`, `ranked_lower`, `robot_taken`, `budget`). Stored in memory as
compact arrays; written to the episode trace (section 5).

## 4. Metrics

Every metric states its unit, denominator, window and censoring. Window = the episode
`[0, T]` unless noted; time in ticks (convert with `seconds_per_tick`). **Censoring**: work
unfinished at `T` is excluded from flow-time and tardiness averages and reported separately as
count and age (`T - arrival`) distribution; rates use all arrivals as denominator.

**Primary (five)**

| Metric | Definition |
|---|---|
| Order throughput | completed orders per 1,000 ticks over `[warmup, T]` |
| Order lead time | completion − arrival, completed orders; mean and P95; unfinished reported separately |
| On-time completion rate | orders with a due date completed by it ÷ orders with a due date that arrived by `T − horizon_guard`; unfinished past due count as misses |
| Energy per completed order | total robot energy (Wh) ÷ completed orders (also per delivery) |
| Coalition waiting ratio | robot-ticks committed to a task and present but not working because the crew is incomplete ÷ robot-ticks committed to tasks |

**Secondary** (all recorded, none primary): order completion and deadline-miss rates, mean/P95
tardiness, waiting/blocked/suspended durations, unfinished orders and WIP over time, dependency
delay (ready time − max prerequisite completion), retry frequency; assignment latency (ready →
assigned), scheduler compute time per tick (mean, P95, max), completion rate by type and priority,
task response time, reassignment and rescheduling counts, starvation (tasks ready longer than
`k ×` median), workload fairness (Jain index over robot busy time), share of feasible tasks left
unassigned per tick, allocation efficiency (estimated vs realised completion); per-robot and total
distance, empty vs loaded distance, utilisation/idle/charging time, energy per delivery,
coalition formation time (first member assigned → crew complete), partner waiting, capability and
role-alternative usage, workload imbalance (coefficient of variation), dock waiting, dock
conflicts, deadlock episodes (no progress ≥ 100 ticks); deliveries and completed orders per
episode, throughput by type, end-to-end lead time, backlog growth (slope of open orders), and a
**bottleneck decomposition** that attributes every robot-tick and every order-waiting tick to one
cause: sequencing (eligible, feasible, not chosen), coalition (no feasible crew), travel, resource
(dock wait, charging), execution (working), blocked (dependency/inventory/access).

## 5. Traceability and dashboard

- **Episode trace** (opt-in): events (existing log, extended with order/package/inventory/
  decision events), decision log, a state snapshot every `k` ticks, and a frame index (tick →
  frame number of the recorded video). Every failure or delay record carries the tick, so it
  links to the snapshot, the events of that tick and the frame.
- **Dashboard** (**[D6]**): a self-contained offline HTML report generated from a trace (no
  server): order queue and dependency graph per tick, package/shelf/destination tracking,
  assignments and coalition members, policy decisions with rejection reasons, robot position,
  battery, activity and distance, task durations and failure causes, cumulative metrics; filters
  by order, package, task, robot, coalition and policy; a frame strip linked to the selected tick.
  A live in-loop view can come later.

## 6. Reproducible evaluation

- **Workloads** are order traces (generated once from a seed, then fixed), so every rule sees the
  same arrivals, packages, inventory, robot configuration and failure draws (the simulator's
  independent random streams already make this exact).
- **Harness**: rules × workloads × seeds; paired differences vs a reference rule with bootstrap
  95% CIs; factors: demand rate, battery capacity/dock count (energy ratio), capability
  heterogeneity (role alternatives on/off), dependency complexity (share of orders with
  relocation prerequisites, DAG depth).
- **Separation of effects**: report sequencing rules (FIFO … CR, crew fixed by the same
  coalition choice) separately from coordination methods (NSP, PLB, CBBA, Calvo, matched docks),
  and use the bottleneck decomposition to locate where time is lost.

## 7. Compatibility and validation

- Opt-in keys: `orders:` (operation types, generator or trace), `inventory:`, `lifecycle:
  extended`, `scheduler:` (rule and parameters), `trace:`. Without them nothing changes.
- Tests per phase; the 2.1.0 trajectory-identity check on all built-in scenarios; a behavioural
  regression (gate thresholds) on order scenarios; property tests for the state machine and the
  inventory (no package in two places, no location holding two packages, every event consistent).

## 8. Phases

1. Entities, inventory, operation types, order generator/trace, extended lifecycle (no new
   scheduling): tests + identity check.
2. Scheduler pipeline, nine rules, decision log: tests on hand-built cases per rule.
3. Metrics (all four levels, censoring, bottleneck decomposition): tests on synthetic traces with
   known answers.
4. Trace + offline dashboard.
5. Evaluation harness and a first paired comparison of the nine rules on order scenarios.

## Open decisions

- **[D1]** One package per storage cell, initial fill ratio from the seed.
- **[D2]** Access = a loaded carrier can reach an aisle without crossing occupied storage cells.
- **[D3]** Failure detectors and defaults (all off by default).
- **[D4]** WSP default weights (proposal: α 0.4 wait, β 0.2 type, γ 0.4 priority; type weights
  outbound 1.0, inbound 0.7, relocation 0.5), all configurable.
- **[D5]** ZLP zones = named zones plus rack blocks with a configured order; PLB regions =
  rack-block columns plus the goal row.
- **[D6]** Offline HTML dashboard first; live view later.
- Publication: develop on `feat/orders-scheduling`, publish as a minor package release (2.2.0)
  only after validation; RA-L code stays in RendeCoord.
