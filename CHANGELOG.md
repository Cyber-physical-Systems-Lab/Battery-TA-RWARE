# Changelog

## Unreleased (development: role alternatives; package tracking; order layer)

- **Model 2.3 (opt-in): picker-assisted loading, two-cell convoys, goal lines.** See
  [docs/semantics.md](docs/semantics.md#model-23-opt-in-pickers-on-every-pick-two-cell-convoys-goals-and-waiting).
  - Every pick and return involves a picker (`solo` in basic and `tote` in intermediate get one; the
    `oversize` spotter stands on the shelf cell).
  - Joint carriers form a rigid two-cell convoy (leader on the shelf slot or goal, follower on the
    free-way cell) with rigid-pair PIBT moves; macro mode only, at most two carriers.
  - Rendering: a goal line for every moving robot, delivery goals framed apart from return slots, and a
    `blocked` state apart from `waiting`; `Simulator.goal_of` and `Simulator.wait_reason`.
  - 2.0-2.2 are unchanged: 31/31 identity fingerprints, and six model-2.2 runs hash-identical to
    the commit before this change. Tests: `tests/test_model_2_3.py`.

- **Model 2.2 (opt-in) and the second order-scheduling evaluation.**
  - Model 2.2: a robot working from beside a stage cell never chooses, keeps or waits on another
    open, staffed task's work cell (removes the model 2.1 hold-and-wait gridlock). Default stays
    2.1; 2.0/2.1 trajectories are unchanged (`experiments/checks/identity.py`, 31 fingerprints).
    Gate (`experiments/model_2_2/RESULT.md`): all criteria pass except L2 (7 alternation events,
    5 also under 2.1), reported as failed.
  - Order scenarios on the intermediate and advanced warehouses (role alternatives, joint
    transport, three-type crews with a QC stage, forklifts, breakdowns); compound orders
    (`cross_dock`, `replenish`) and chained lines (`source.sequence`).
  - Scheduler: P99 runtime, a wall-clock `deadline_ms` reporting target (default 20 ms; never
    stops the scheduler), violations over all and decision ticks; diagnosis label fix (busy heavy
    carriers were reported as `infeasible:unreachable`).
  - Dashboard: automated headless-browser tests against the event log (`tests/test_dashboard_browser.py`);
    model version in the run selector.
  - Tracked checks replace the ignored `validation/` scripts (`experiments/checks/`).
  - Second evaluation (`experiments/orders_v2/README.md`): exploratory seeds 2000-2009 (2900 runs),
    pre-registered protocol (6e2a2d4), confirmatory seeds 3000-3019: 11 of 13 hypotheses
    supported; 36 stress runs of 18,000 ticks pass all checks; runtime S-XL.

- **Order scheduling, metrics, dashboard and evaluation** (Phases 2–5; `battery_tarware.orders`,
  `experiments/orders/`).
  - `OrderScheduler(rule)`: one pipeline for every rule.
    - Steps: shared charging rule; eligible set; feasibility through `feasible_coalitions` with the
      common 15% reserve, with rejection reasons; ranking; atomic commits.
    - Rules: FIFO, EDD, LWF, WSP (0.45/0.30/0.25; type scores 0.8/0.6/0.4), the WSP_eq ablation,
      SPT, CR, NSP, ZLP_2D (rack-block zones; 0.8 affinity + 0.2 priority) and PLB
      (0.4(1−C) + 0.3B + 0.3(1−L) over whole coalitions; current information only).
    - A `schedule_decision` event per tick.
  - The order generator draws a fixed number of random numbers per order, so the arrival process is
    identical for all rules on a seed.
  - `TraceRecorder`: per-tick robot frames and activities, and per-tick cause attribution for every
    waiting task.
  - `evaluate`: order, scheduling, robot, coalition and system metrics (see `docs/metrics.md`), and
    a diagnosis of every unsuccessful order with an evidence span.
  - `replay_inventory` rebuilds the package ledger from the event log.
  - `write_dashboard`: an offline, self-contained HTML replay.
    - Panels: map, kanban, order queue, dependency graph, decisions, robots, packages, charts, cause
      timeline, metrics, diagnosis, events.
    - Filters, CSV/JSON export and deep links.
  - `docs/img/orders_kanban.gif` (1.9 MB) and `docs/img/dashboard.png`.
  - Dashboard follow-up (user feedback):
    - Selection works during playback. Clicks were lost because panels are re-rendered every frame
      (press landed on a destroyed element); now one delegated `pointerdown` handler.
    - Stations, charging docks and shelves are selectable (map, with click-again cycling on shared
      cells, or the new *Stations and charging docks* table). Per-resource metrics
      (`metrics.resource_metrics`, `evaluate(...)["resources"]`): station deliveries, stage service
      time, occupancy by activity; dock sessions, energy, occupancy, SoC in/out, stop reasons;
      fleet dock queue. Exports `stations.csv` / `chargers.csv`.
    - Order selection shows the order's tasks with their current states, its events and the
      dependency graph; its cards stay visible and highlighted on the kanban, its crews on the map.
    - Charts redone: order flow (cumulative arrived / completed, band = in progress), throughput per
      bucket (orders completed, packages shipped / stored, arrivals), battery with fleet mean and the
      selected robot; axes, legends, hover read-out, click to jump. The old backlog chart showed
      only a cumulative staircase without axes.
    - Filters accept short IDs (`12` → `O00012`) and mark unknown IDs; *clear selection* button;
      deep links for every selection kind.
  - `evaluation`: parallel run grid (errors kept as rows) and paired bootstrap CIs.
  - Experiments: exploratory one factor at a time; confirmatory with a pre-registered protocol
    (`76193bd`) on fresh seeds; scalability. Results are in `experiments/orders/README.md`.
  - Examples: `examples/orders.yaml`, `examples/orders_quickstart.py`.
  - Fix (`725582f`): a relocation whose shelf is away now waits instead of raising `TypeError`.
  - Results (`experiments/orders/README.md`): 7 of 9 pre-registered hypotheses are supported.
    - NSP beats FIFO on throughput and lead time at base load, with heavy coalitions and with tight
      batteries, and beats PLB. LWF is slower than FIFO. NSP and FIFO are equivalent when
      demand-limited.
    - Not supported: PLB > FIFO (one gridlocked seed), and WSP type weights > the equal-type
      ablation.
  - Known limitation: model 2.1 can deadlock through hold-and-wait.
    - A picking robot working from a cell next to its own task's shelf can stand on another task's
      work cell. That task's loaded AGV then waits for the cell, and the waiting crews block the
      aisle the first task needs.
    - This happened in 2 of 1,480 evaluation runs, both under PLB.
    - The stall detector reports it, and the diagnosis names it.
    - Reproduce: `python experiments/checks/known_gridlock.py`. Model 2.1 is not changed (fixed
      in the opt-in model 2.2, below).

- **Order layer, Phase 1** (`battery_tarware.orders`; opt-in through the new `orders:` scenario
  key). Design and confirmed decisions: `docs/design/orders_and_scheduling.md`.
  - Entities with persistent ids: orders, task groups, order-layer tasks (persist across dispatch
    attempts), packages, shelves and rack locations. A package keeps its id across inbound,
    storage, relocation and outbound.
  - Inventory ledger of logical packages on the existing shelves (`slot_capacity`, default 1):
    reservations at acceptance, commits only from simulator events (pick-up, drop, re-pick, station
    hand-off, placement), invariants checked every tick (`InventoryError`).
  - Operations: outbound, inbound, relocation, mapped to task types by shelf weight. Dependencies
    within an order, cycles rejected. Blocked tasks never reach the simulator. Explicit relocation
    prerequisites; `@key` references to an earlier task's package or shelf.
  - Extended task lifecycle: pending, blocked, ready, assigned, executing, suspended, failed,
    retrying, completed, cancelled, with reasons. Failure records hold reason, attempt,
    blocking resource, last progress tick and a snapshot. Order completion is separate from task
    completion.
  - `ProgressMonitor`: event-based stall detection that tells legitimate waiting (partner, repair)
    from no progress. Defaults: 100 ticks, 3 dispatch retries, `auto_retry` off (diagnostic only).
  - `OrderManager.submit`, `fail_task` and `cancel_order`. A task can be failed or cancelled only
    while its payload has not left the origin.
  - Order metrics computed from the event log only (`orders.reports`), with unfinished orders
    censored. CSV ledgers of orders, tasks, packages and shelves.
  - Example scenario `orders.scenarios.order_scenario()` (basic map plus a relocation task type).
  - Simulator additions, inert without `orders:`: `create_task`, `can_cancel`, `cancel_task`, and
    order event types in the log.
  - In order mode the package tracker numbers trips `M00000` and links each trip to its order, task
    and shelf.
  - Validation:
    - Identity check (now `experiments/checks/identity.py`): task-only runs are identical to
      93658ea (22 fingerprints at the time).
    - `experiments/checks/phase1.py`: inbound, outbound, relocation, dependencies and failure
      recovery; 8 gates pass on seeds 0–9.
    - `experiments/checks/stall_threshold.py`: the longest no-progress streak in successful runs
      is 39 ticks against the 100-tick threshold.
    - `tests/test_orders.py`: 27 tests.

- **Package and robot tracking** (`battery_tarware.tracking`, read-only; tracking never changes a
  run).
  - `PackageTracker(sim).update()` after each step keeps a ledger per package and per robot.
  - Per package: id, task, type, job, weight, rack, goal, status, current location, carriers,
    every handler with role and join/leave ticks and reason, loaded distance, and the released,
    first-assigned, picked, delivered and completed ticks plus drops.
  - Package statuses: waiting, assigned, loading, in_transit, unloading, returning, dropped,
    delivered, stored, cancelled.
  - Per robot: total, empty and loaded distance, packages handled, current package and role,
    battery, activity, and ticks per activity.
  - Helpers: `package_table`, `robot_table`, `status_counts`, `write_csv`.
  - `docs/make_tracking_demo.py` renders the map with a live package and robot board, and writes the
    two ledgers as CSV.

Not part of the 2.1.0 release. Model 2.1 rules are unchanged; configurations without
`alternatives` (all built-in scenarios by default) run tick for tick as in 2.1.0.

- **Role alternatives.** A task role may list `alternatives: [{capability, presence, radius}, ...]`
  instead of one `capability`. Any listed capability can fill the slot, and each has its own work
  position. A robot fills the role through the first alternative it matches and must be present at
  that alternative's position. The carrier role cannot have alternatives.
  - The rule applies in configuration validation, joining (`can_join`, action masks), coalition
    feasibility and estimates, stage execution, standing-cell choice and crew co-location.
  - Capability-indexed observation features count an alternative slot under each capability that
    can fill it.
- **`get_scenario("intermediate", substitution=True)`.** The `case` support role is filled by a
  picking robot on the shelf cell, or by a tote AGV with the new `load_assist` capability from the
  next cell (radius 1).
  - Only `tote_agv` gets `load_assist`.
  - Demand, fleet, crew size, stage durations and all other task types are unchanged.
  - Effective handling speed is unchanged: stage progress follows the slowest crew member, here the
    carrier tote AGV at 1.0.
  - `load_assist` is an abstract benchmark capability, not a validated physical loading behaviour.
- **Semantics of capability counts.**
  - Capability-indexed features are eligibility counts: an either/or slot appears under each
    capability that can fill it, so they must not be summed.
  - The exact requirement is in the new state-view arrays `task_open_slots`, `task_role_open` and
    `task_role_eligible` (OR within a role, AND across roles).
  - The observation vector is unchanged.
- **No order-dependent work positions.** A robot type matching two alternatives of one role with
  different work positions is rejected at configuration time.
- **Validation.** A regression check on the variant, standard and large maps, inline docks and
  bays, seeds 0-9: PASS (the script is kept outside the published repository; the properties are
  covered by `tests/test_capability_alternatives.py`).
  - Under the model 2.1 gate thresholds: no swaps, dock violations, no-progress, oscillation or
    stranding.
  - No assisting AGV on the shelf cell or outside radius 1, and no robot in two roles of one task.
  - Tote AGVs took 30-43% of support stages.
  - With the variant off, trajectories match 2.1.0 exactly (9 greedy episodes: positions, energy,
    rewards, observations).

## 2.1.0 (model 2.1)

Model 2.1 is the new default (`model: "2.1"`). It adds explicit occupancy, docking and waiting rules
so that robots wait, move, cooperate and charge for reasons that follow from the task and resource
semantics. `model: "2.0"` (with `get_scenario(name, model="2.0")` for built-ins) reproduces model 2.0
tick for tick. See [docs/semantics.md](docs/semantics.md#model-21-occupancy-docking-and-waiting).

- **Shelf loading as in TA-RWARE.** The picking robot steps onto the shelf cell beside the AGV to
  load and unload (built-in picker roles are `on_cell`); a second picker works from the next cell.
  Occupancy is task-dependent: a carrier and its `on_cell` crew may share their work cell, co-carriers
  share the payload cell, and unrelated robots never share a cell.
- **Docks.** Explicit dock states (available, reserved, occupied, unavailable;
  `Simulator.charger_state`). `send_to_charger` reserves; a dock held by another robot is refused.
  Docks are destinations only. New `map.dock_layout: bays` for off-corridor docks.
- **Movement.** Robots route around long-lived obstacles (charging, broken-down, stranded robots and
  robots on their own work cell) instead of waiting behind them; a pushed robot steps aside instead of
  retreating ahead of its pusher.
- **Waiting.** Waiting and parking spots are neutral cells, kept by the robot that chose them and not
  shared. A robot does not occupy a work cell another task also needs until its own crew is complete
  (no hold-and-wait on shared work cells).
- **Behavioural changes.** In basic, pickers now have to reach the shelf cell, which can lower
  throughput slightly against model 2.0 (by design). Baselines in the README are model 2.1 numbers.
- **Validation.** A pre-registered behavioural gate (scripts and record kept outside the published
  repository; outcome summarised in `docs/semantics.md`): exact reproduction of
  model 2.0; no vertex or swap conflicts, dock exclusivity, no persistent no-progress or oscillation
  and no new stranding under model 2.1, on all built-in scenarios and their large versions. The
  record of every run, revision and the post hoc occupancy amendment was kept with the gate scripts.
  The same checks are re-run for 2.1 and 2.2 by `experiments/model_2_2/gate.py`.
- README animations regenerated from model 2.1.

## 2.0.0

A new engine replaces the v1 `tarware` package. The distribution is `battery-ta-rware`; the
import name is `battery_tarware`. The simulation rules are model version 2.0, documented in
[docs/semantics.md](docs/semantics.md).

v1 is preserved on branch `v1` (the last v1 `main`, including the PRISM environment) and at tag
`legacy-v1.1.0`. Version 2 is not trajectory-compatible with v1.

- **Heterogeneity as configuration**:
  - robot types with free-form capabilities, speed, payload, handling speed, allowed zones (empty
    and loaded), a battery and power model, charger compatibility, sensing, communication and
    localization noise, and breakdowns;
  - per-group overrides for heterogeneity within a type.
- **Cooperative tasks**:
  - roles with capability, count and presence rules;
  - ordered synchronized stages at origin, destination or custom zones;
  - carrier roles with joint transport;
  - stage-by-stage recruitment;
  - payload drop and recovery;
  - jobs with dependency DAGs and location chaining.
- **Demand**: backlog, Poisson (piecewise rates), batch and trace streams, with task mixes.
- **Feasible coalitions** computed by the environment: capability, carrier capacity, reachability
  and an energy reserve.
- **Interfaces**:
  - Gymnasium 1.x environment (`battery_tarware/{Basic,Intermediate,Advanced,Warehouse}-v2`),
    macro and primitive action modes, action masks, typed event log;
  - PettingZoo ParallelEnv;
  - frozen `StateView` with a primitive-array DTO for native coordinators.
- **Movement**:
  - PIBT with a swap operation restricted to dead ends in macro mode, and conflict-free simultaneous moves in
    primitive mode;
  - maps from ASCII (custom zones, optional one-way cells) or a generator (aisle width, one-way grids).
- **Scenarios**: three tiers (basic, intermediate, advanced). Their robot types are named after
  warehouse roles: `agv` / `tote_agv` / `pallet_agv`, `picking_robot` / `fast_picking_robot`
  (mobile piece-picking robots), `forklift`, `mobile_manipulator` and `inventory_drone`.
- **Coordinators**: greedy coalition baseline, masked random baseline, and documented dispatching
  rules (FIFO, EDD, SPT, CR, priority).
- **Rendering** in the TA-RWARE style (numpy only):
  - polygon shape per robot type and colour per activity (idle, travelling, waiting, working,
    carrying, heading to a charger, charging, failed, stranded);
  - requested shelves, delivery targets, heading tick and battery badge.
- **Documentation**:
  - README with an annotated explanation figure, four large-scale animations, the action and
    observation spaces, dynamics and rewards, and a step-by-step scenario tutorial
    ([examples/tutorial.yaml](examples/tutorial.yaml));
  - configuration reference;
  - simulation semantics;
  - acknowledgements.
- **Dependencies**: base install needs only numpy and gymnasium.
