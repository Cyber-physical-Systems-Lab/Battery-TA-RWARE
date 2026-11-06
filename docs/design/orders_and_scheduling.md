# Design: order management, dependency-aware scheduling, allocation and evaluation

Status (branch `feat/orders-scheduling`, based on `feat/capability-alternatives`):
**Phase 1 implemented** (entities, inventory, operations, extended lifecycle, failure detection and
retries, order metrics from the event log). Phases 2–5 are specified below and not implemented.
Decisions D1–D6 were confirmed by the maintainer on 2026-11-06.

Every feature is opt-in: a scenario without the `orders:` key runs tick for tick as in 2.1.0
(`validation/orders/identity.py`: 22 trajectory fingerprints identical to commit 93658ea).

## 0. What exists and what is new

| Area | In 2.1.0 | New |
|---|---|---|
| Work units | `Task` (typed roles, stages, payload weight); `Job` = DAG of tasks | `Order`, `TaskGroup`, order-layer `OrderTask`, `Package`, `Shelf`, `RackLocation` with persistent ids |
| Lifecycle | pending … completed, cancelled; retry budget never used | blocked, failed, retrying actually exercised, with recorded reasons |
| Scheduling | `RULES` (FIFO, EDD, SPT, CR, priority), `feasible_coalitions`, greedy | Phase 2: five-step pipeline, nine rules on one candidate set, decision log |
| Metrics | 10 episode totals | Phase 1: order metrics from the event log; Phase 3: four levels |
| Visualisation | renderer, GIFs, package/robot tracker | Phase 4: offline HTML replay dashboard |
| Evaluation | `run_episode` | Phase 5: paired, seeded harness with bootstrap CIs |

The rule set follows the open-source *warehouse-multi-robot-scheduler* (MIT, a TA-RWARE
derivative). Rules are implemented independently for the 2D grid and the docs cite that project;
WSP and PLB use its coefficients as baseline choices (not optimal weights); ZLP is adapted to 2D.

## 1. Entities and inventory (Phase 1, implemented; D1)

`battery_tarware/orders/`. Ids are assigned in creation order, unique per episode, never reused.

- **Order** `O00000`: arrival tick, priority, optional due tick, source (trace, generated,
  submitted), tasks and groups; state `open` → `completed` | `failed` | `cancelled` (reason
  recorded). An order completes only when **all** its tasks completed, never when one package
  reaches its goal.
- **TaskGroup** `G00000`: the tasks of one order line; status derived from its tasks.
- **OrderTask** `T00000`: one operation on one package or shelf, with `depends_on` (task ids of the
  same order). It persists across dispatch attempts; each attempt is one simulator task.
- **Package** `P00000`: logical unit on a shelf; keeps its id across inbound, storage, relocation
  and outbound. Status: arrived, stored, in_transit, dropped, shipped, cancelled. A package can take
  part in several tasks and orders; its status is not monotone across them (task state and package
  state are separate).
- **Shelf** `S0000`: the unit carriers move (TA-RWARE convention); `slot_capacity` packages
  (default 1).
- **RackLocation** `L0000`: a storage cell (row-major order); holds at most one shelf.

No package geometry or manipulation physics is added. Reservations are taken when an operation is
accepted (outbound: the package; inbound: a shelf slot, nearest to the station; relocation: the
destination location; a shelf that is away keeps its home location). Physical changes are committed
only from simulator events: a shelf leaves its location at pick-up, a package leaves (outbound) or
joins (inbound) the shelf when the destination stage completes, the shelf is placed when the task
completes. `Inventory.check()` (every tick by default) raises `InventoryError`, a system error, on
any inconsistency: a package on two shelves, a location holding two shelves, a stored shelf no
location holds, capacity exceeded, or broken conservation (created = on shelves + arrived + shipped
+ cancelled).

### Operations

- **Outbound**: carry the package's shelf to a station cell (`goal` by default) and back; the
  package ships at the station.
- **Inbound**: the package arrives at a station; the shelf with the reserved slot comes to the
  station and goes back with it.
- **Relocation**: carry a shelf to a free rack location (explicit `to`, or the nearest reachable free
  location).

The task type per operation is chosen by shelf weight (first `max_kg` that fits), so crews follow
the existing task types (e.g. basic's solo / standard / large).

## 2. Access and readiness (D2)

The 2D grid, A* navigation, PIBT collision handling and capability rules are unchanged. A task is
released to the simulator only when it is **ready**:

1. every dependency completed (a cancelled or finally failed dependency cancels it);
2. its reservations are held and the inventory condition holds (package stored, shelf stored and
   not in use by another task, free slot or location);
3. the payload fits the strongest carrier crew of the chosen task type;
4. a loaded carrier can reach the destination from the shelf (and back, for outbound and inbound)
   on the static map.

Otherwise it is **blocked** with a reason (`dependency:T…`, `inventory:…`, `infeasible:payload…`,
`access:unreachable…`) and does not exist in the simulator, so no robot can be assigned to it.
Coalition feasibility at assignment (capabilities, role alternatives and work positions, battery
reserve, robot availability) stays with `can_join` / `feasible_coalitions`.

If access requires moving another shelf, the order states that relocation explicitly as a task with
a dependency (`{"key": "mv", "op": "relocation", …}`, `"depends_on": ["mv"]`). Nothing is moved
automatically, teleported, or allowed through collision rules. Automatic generation of access
relocations is a later extension. Within an order, `package: "@key"` names the package of an
earlier inbound task and `shelf: "@key"` the shelf now holding it.

## 3. Task lifecycle, failures and retries (Phase 1, implemented; D3)

| State | Meaning |
|---|---|
| pending | accepted with its order, readiness not evaluated |
| blocked | not released (reason recorded) |
| ready | released as a simulator task (one dispatch attempt) |
| assigned / executing / suspended | mirrored from the simulator attempt |
| failed | attempt failed: `failure_reason`, `attempt_number`, `blocking_resource`, `last_progress_tick` and a state snapshot recorded; final when no retry is left |
| retrying | waits `retry_delay` ticks, then readiness is rechecked |
| completed / cancelled | terminal |

Waiting is distinguished from failure (`ProgressMonitor`, read-only, usable on any simulator):

| Condition | Handling |
|---|---|
| waiting for a task dependency | blocked, not failed |
| waiting for a partner (stage crew incomplete) | `waiting_partner`, stall clock does not run |
| crew member broken down / out of energy | `robot_repair` / `robot_stranded`, clock does not run |
| waiting to be assigned | not monitored (sequencing delay) |
| no movement or work progress with a complete crew | clock runs (`waiting_cell` with the holder as blocking resource, or `no_progress`); `stall_ticks` (default 100) → `stall_suspected` event |
| repeated path-planning failure | reported by the coordinator through `OrderManager.fail_task` → retry |
| collision deadlock | evidence recorded in `stall_suspected`; the simulator's own recovery continues |
| invalid configuration, malformed order, inconsistent inventory | `ConfigError` / `InventoryError` |
| episode ends with incomplete work | unfinished (censored), never failed |

`auto_retry` is **off by default**: a suspected stall is then diagnostic only. With it on, the
attempt fails and is retried up to `max_dispatch_retries` (default 3) times. An attempt can be
failed or cancelled only while its payload has not left the origin. Afterwards the task finishes
physically, or the simulator re-recruits a crew after a drop, so the inventory stays consistent.

Threshold check on successful task-only traces (`validation/orders/stall_threshold.py`, greedy
FIFO, 3 built-ins × 5 seeds × 3,000 ticks): longest eligible no-progress streak 9 / 16 / 39 ticks
(basic / intermediate / advanced), no task reached 50, no report at 100.

## 4. Scheduling architecture (Phase 2, specified)

Every tick a `Scheduler` runs five separate, replaceable steps: (1) release and prioritisation of
orders; (2) readiness: the order layer above produces the **eligible set**; (3) feasibility: feasible
coalitions per eligible task (capabilities, alternatives, work positions, payload, battery with the
common reserve, availability); infeasible tasks keep rejection reasons; (4) assignment: a rule ranks
**only feasible (task, coalition) pairs** and commits atomically; (5) execution and recovery feed back
into 2–4. Charging is a shared component (the common rule, or minimum-cost dock matching), so rules
differ only in step 4. All time-based terms use simulation ticks, never wall-clock time.

### Rule definitions

Notation: now `t`; task `j`: arrival `a_j`, last time it became ready `r_j`, due `d_j`, order
priority `w_j`, processing estimate `p_j` (remaining stage work plus carried legs); coalition `c`.
Normalisation `norm(x)` = min-max over the current candidate set (0 if all equal). Ties break by
task id, then robot id.

| Rule | Ranks | Definition |
|---|---|---|
| FIFO | tasks | smallest `a_j` |
| EDD | tasks | smallest `d_j` (none last), then `a_j` |
| LWF | tasks | largest `t − r_j` (differs from FIFO after blocking, suspension, reassignment, retry) |
| WSP | tasks | largest `S_j = 0.45·P_j + 0.30·W_j + 0.25·T_j`, `P_j = norm(w_j)`, `W_j = min((t − r_j)/100, 1)`, `T_j` = 0.8 outbound, 0.6 inbound, 0.4 relocation (D4; all values configurable) |
| WSP-eq | tasks | WSP with equal type scores, to test whether a gain comes from sequencing or from favouring outbound |
| SPT | tasks | smallest `p_j` |
| CR | tasks | smallest `(d_j − t) / max(p_j, 1)`; no due date last |
| NSP | pairs | smallest crew arrival estimate `e_jc` (the crew's largest travel time to its work cell) |
| ZLP_2D | pairs | largest `0.8·A_jc + 0.2·P_j` (D5): zones = configurable 2D operating zones, by default the rack blocks (aligned with aisles) plus the station zone; zone affinity `A_jc` = 1.0 if the task's source and destination are both in the carrier's current zone, 0.7 if one is, 0.3 if neither. An adaptation of the 3D original, hence the name |
| PLB | pairs | largest `0.4·(1 − C) + 0.3·B + 0.3·(1 − L)` (D5): `L` = norm(estimated path length of the complete crew: each member's leg to its work cell plus the carried legs); `C` = share of the carrier's planned cells (approach and carried legs, by the existing A*) that lie on the current planned paths or cells of committed robots; `B` = 1 − norm(the crew's mean busy share over the last 100 ticks). All in [0, 1], from current and past state only: PLB gets no information other rules lack. Computation time and number of candidate pairs are logged |

Task-ranking rules (FIFO … CR) choose the task; the crew is then the feasible coalition with the
earliest estimated completion (as the greedy does), so they differ only in sequencing. NSP, ZLP_2D
and PLB rank pairs and so also choose the crew.

### Decision log

One record per assignment attempt: tick, rule, eligible tasks with keys/scores, feasible coalitions
per task (estimated completion, energy), selection, and per rejected task or coalition a reason
(`not_ready:<cause>`, `infeasible:<capability|payload|battery|zone|unreachable>`, `ranked_lower`,
`robot_taken`, `budget`), plus computation time and candidates evaluated. Written to the event log.

## 5. Metrics

Phase 1 (implemented, `orders/reports.py`, computed from the event log only): orders arrived /
completed / failed / cancelled / unfinished; throughput per 1,000 ticks over `(warmup, T]`; lead time
mean and P95 (completed only); on-time rate over orders whose outcome is known by `T` (open orders
not yet due are censored and counted separately); energy per completed order; unfinished count and
age; tasks completed, retries, stalls. Tests reconcile these with the manager's state and with
package and task event histories.

Phase 3 adds the remaining primary metric (coalition waiting ratio: robot-ticks present but not
working because the stage crew is incomplete ÷ robot-ticks committed), the secondary metrics listed
in the earlier draft (scheduling, robot/coalition and system levels), and the bottleneck
decomposition (sequencing, coalition, travel, resource, execution, blocked).

## 6. Traceability and dashboard (Phase 4; D6)

Keep the map renderer, live boards, GIF generator and CSV exports. Add a lightweight offline HTML
replay generated from recorded logs (no new simulation frontend, no server): select a tick, package,
order, task or robot; dependency view; event timeline with assignment changes, coalition formation,
charging, suspensions, retries and failure reasons. The single event log is the authoritative
source for dashboard, CSVs and metrics. The full-resolution 4.9 MB tracking GIF stays out of Git; a
shorter optimised demo GIF may be committed.

## 7. Reproducible evaluation (Phase 5)

Workloads are order traces fixed by seed; rules × workloads × seeds; paired differences against a
reference rule with bootstrap 95% CIs; factors: demand rate, energy ratio (battery, docks),
capability heterogeneity, dependency complexity. Sequencing rules are reported separately from
coordination methods (NSP, ZLP_2D, PLB, CBBA, Calvo, matched docks).

## 8. Phases

1. **Done.** Entities, inventory, operations, extended lifecycle, failure detection and retries,
   order metrics. Tests `tests/test_orders.py`; milestone `validation/orders/phase1.py` (inbound,
   outbound, relocation, dependencies, failure recovery; 8 gates; seeds 0–9 pass); identity check.
2. Scheduler pipeline, nine rules (+ WSP-eq), decision log: hand-built tests per rule.
3. Metrics (four levels, censoring, bottleneck decomposition): synthetic traces with known answers.
4. Offline replay dashboard.
5. Evaluation harness and a first paired comparison on order scenarios.

Publication: develop on `feat/orders-scheduling`; release as 2.2.0 only after validation; RA-L code
stays in RendeCoord.
