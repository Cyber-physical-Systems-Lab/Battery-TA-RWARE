# Order-layer metrics

All metrics are computed by `battery_tarware.orders.metrics.evaluate(trace)` from one recorded trace
(the simulator's event log plus the per-tick frames of `TraceRecorder`). Every algorithm is
measured by this code, from this record. Implementation tests: `tests/test_scheduling.py`
(`test_metrics_on_synthetic_trace` checks the definitions on a hand-made trace with known answers).

## Conventions

- **Time** is in simulation ticks (`seconds_per_tick` converts; basic: 5 s). Wall-clock time appears
  only in the scheduler-runtime metrics.
- **Horizon** `T` is the last recorded tick. **Window** for rates is `(warmup, T]`.
- **Censoring.** Work unfinished at `T` is never counted as failed. It is excluded from lead time and
  tardiness, and reported separately by count and age (`T − arrival`).
- **Robot-tick**: one robot for one tick. Activity shares are over all robot-ticks.
- **Paired comparisons** (`evaluation.paired`): the per-seed difference rule − reference (same seed,
  hence identical arrivals, inventory and robot placement). The CI is a percentile bootstrap over
  seeds (10,000 resamples, 95%).

## Primary metrics

| Metric | Unit | Definition |
|---|---|---|
| Order throughput | orders / 1,000 ticks | orders completed in the window ÷ window length × 1,000 |
| Order lead time | ticks | completion − arrival, completed orders (mean; P95 secondary) |
| On-time completion rate | share | over orders with a due date whose outcome is known by `T` (completed, failed, or open past due; cancelled excluded): completed at or before due. Open orders not yet due are censored (`on_time_censored`) |
| Energy per order | Wh / order | total robot energy drawn (driving, payload, handling, idle) ÷ completed orders |
| Coalition waiting ratio | share | robot-ticks *waiting for partners* (at its work position, not working) ÷ robot-ticks committed to tasks (travelling, working or waiting) |
| Unfinished orders | count, ticks | open orders at `T` and their mean age |

## Order level

`orders_arrived`, `completed`, `failed`, `cancelled` (including rejected: no stored package or shelf at
arrival), `unfinished`; `completion_rate` = completed ÷ arrived; `lead_time_p95`; `tardiness`
(`max(0, completion − due)` over completed orders with a due date: mean, P95, max);
`late_completed`; `deadline_miss_rate` = 1 − on-time rate; `wip_mean` (mean open orders per tick);
`backlog_slope_per_1000` (least-squares slope of open orders over time, per 1,000 ticks; > 0 = backlog
growing); throughput and lead time by order kind; mean lead time by priority.

## Scheduling level

| Metric | Definition |
|---|---|
| assignment latency | first `ready` → first `assigned` of each order-layer task (mean, P50, P95, max) |
| reassignments / suspensions | `assigned → ready` transitions / transitions into `suspended` |
| attempt failures, retries, stalls | failed dispatch attempts, retries started, suspected stalls |
| dependency delay | first `ready` − completion of the last prerequisite |
| starving tasks | tasks whose wait for a first assignment exceeds 5 × the median latency (or is still running at `T` beyond that) |
| decision outcomes | count of eligible-task-ticks per outcome of the decision log: `assigned`, `robot_taken` (its coalitions lost robots to higher-ranked tasks), `budget`, `infeasible:capability / payload / battery / unreachable`, `unavailable:robots_busy / charging / breakdown` |
| feasible-unassigned share | eligible-task-ticks with ≥ 1 feasible coalition that were not assigned ÷ eligible-task-ticks |
| runtime | scheduler wall-clock time per tick, ms (mean, P50, P95, P99, max), over all ticks and over decision ticks (`runtime_ms_decision_ticks`: ticks with at least one eligible task); also the shared charging rule alone (`charging_rule_ms`) |
| deadline compliance | `deadline_ms` (scheduler argument, default 20 ms; a *reporting target*: the scheduler never stops early), `deadline_violations` (ticks over it), `deadline_violation_rate` (÷ all ticks), `deadline_violation_rate_decision_ticks` (÷ decision ticks), `deadline_worst_ms`. The algorithmic budget is count-based (`max_assign_per_tick` 8 commits, `coalitions_per_task` 4, `per_role_candidates` 6, one re-query), deterministic and machine-independent; wall-clock figures depend on the machine and its load |
| eligible per tick | eligible tasks the scheduler considered per tick |
| candidates per tick | (task, coalition) pairs evaluated per tick |

## Robot level

Per robot and fleet-wide: `distance` (cells moved), `loaded_distance` / `empty_distance`,
`energy_wh`, `utilization` = (travel + working + waiting) robot-ticks ÷ robot-ticks, share of each
activity (idle, travel_empty, travel_loaded, working, waiting_partner, waiting_blocked, to_charger,
dock_queue, charging, failed, stranded). Fleet: `utilization_jain` (Jain's fairness index over robots'
busy ticks: 1 = perfectly even), `workload_cv` (coefficient of variation of busy ticks),
`charge_sessions`, `charging_ticks`, `dock_queue_ticks`, `min_soc`, `stranded`, `breakdowns`.

## Coalition level

`formation_time` (first robot joined a simulator task → its first stage started, i.e. crew complete
and present), `waiting_ratio` (primary), `partner_waiting_ticks`, coalition `size` (distinct robots
per simulator task), `joint_transport_tasks` (more than one carrier), `coalitions_formed`.

## System level

`deliveries` = packages shipped (outbound) + packages stored (inbound); `energy_wh`;
`energy_wh_per_order`; `energy_wh_per_delivery`; `distance_per_delivery`;
`congestion_blocked_moves_per_1000_robot_ticks` (PIBT `collision_blocked` events);
`congestion_waiting_share` (robot-ticks `waiting_blocked`). Scalability: scheduler runtime and wall
time per tick by warehouse and fleet size (`experiments/orders/scalability.py`).

## Resource level (stations and charging docks)

`evaluate(trace)["resources"]` (`metrics.resource_metrics`), shown in the dashboard's *Stations and
charging docks* panel and exported as `stations.csv` / `chargers.csv`.

| Resource | Metric | Definition |
|---|---|---|
| Station (each goal cell) | `shipped`, `received` | `package_shipped` / `package_arrived` events at the cell |
| | `deliveries_per_1000` | (shipped + received) ÷ window × 1,000 |
| | `stage_visits`, `service_ticks_mean` | task stages whose cell is the station, completed; mean stage start → completion |
| | `occupancy`, `occupied_ticks` | ticks with a robot on the cell ÷ ticks |
| | `working_ticks`, `waiting_partner_ticks`, `blocked_ticks` | robot-ticks on the cell by activity |
| Charging dock | `sessions`, `open_sessions`, `robots` | `charge_started` events; sessions still open at `T`; distinct robots |
| | `charging_ticks`, `occupancy` | ticks with a robot charging on the dock cell; ÷ ticks |
| | `energy_wh` | Σ (SoC at stop − SoC at start) × the robot's capacity; an open session counts to the last frame |
| | `session_ticks_mean`, `soc_start_mean`, `soc_end_mean`, `stop_reasons` | per session |
| Fleet | `dock_queue` | robots in the dock queue per tick (robot-ticks, mean, max); a queuing robot is not tied to one dock |

## Bottleneck decomposition and diagnosis

Every tick of every open order-layer task gets exactly one **cause** (`TraceRecorder`, see the table
in `battery_tarware/orders/trace.py`): blocked (dependency, inventory, access), capability,
sequencing, coalition_unavailable, charging_contention, breakdown, battery, unreachable, travel,
execution, coalition_waiting, congestion, failure_recovery, unassigned. The cause of a task that
is waiting for a crew comes from the scheduler's decision log of that tick (the same reason codes
for every rule). Reported: ticks and share by cause (`task_share_by_cause`) and the share of
non-progress causes.

**Unsuccessful orders** (failed, rejected or unfinished at `T`) are diagnosed by their dominant
non-progress cause, or by the failure reason for failed orders. Each diagnosis is mapped to a
category (scheduling decision, coalition infeasibility, insufficient capabilities, battery
constraints, charging contention, navigation congestion or no route, task dependencies, inventory
condition, robot breakdown, failure and retry). It carries an **evidence span** (first tick, last
tick, task): the longest span of that cause, which the dashboard opens at that tick, on the map
and in the event log. Simulator or configuration errors raised during a run are recorded by the
evaluation harness as rows with `error` set.
