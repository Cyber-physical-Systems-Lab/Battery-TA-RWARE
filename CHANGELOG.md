# Changelog

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
- **Validation.** A pre-registered behavioural gate (`validation/model_2_1/`): exact reproduction of
  model 2.0; no vertex or swap conflicts, dock exclusivity, no persistent no-progress or oscillation
  and no new stranding under model 2.1, on all built-in scenarios and their large versions. The
  record of every run, revision and the post hoc occupancy amendment is in
  `validation/model_2_1/PREREGISTRATION.md`.
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
- **Scenarios**: three tiers (basic, intermediate, advanced).
- **Coordinators**: greedy coalition baseline, masked random baseline, and documented dispatching
  rules (FIFO, EDD, SPT, CR, priority).
- **Documentation**:
  - README with animations and a step-by-step scenario tutorial
    ([examples/tutorial.yaml](examples/tutorial.yaml));
  - configuration reference;
  - simulation semantics;
  - acknowledgements.
- **Dependencies**: base install needs only numpy and gymnasium.
