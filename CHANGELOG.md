# Changelog

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
