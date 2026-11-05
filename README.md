# Battery-TA-RWARE

**Heterogeneous multi-robot cooperation under battery constraints**, as a
[Gymnasium](https://gymnasium.farama.org) environment with a
[PettingZoo](https://pettingzoo.farama.org) wrapper.

Robots differ in what they can do, how fast they move, what they can lift, how long their battery
lasts, which chargers they can use and how much they can see. Tasks say which capabilities they
need, how many robots of each, where the crew must meet and for how long. The environment works out
which teams *can* do a task; who actually does it is up to your coordinator or learning agent.

```bash
pip install -e .                 # numpy + gymnasium only
pip install -e ".[all]"          # + pettingzoo, pyyaml, matplotlib
```

```python
import gymnasium as gym
import battery_tarware as bt     # registers the environments

env = gym.make("battery_tarware/Intermediate-v2")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

bt.run_episode(bt.make("advanced"), bt.GreedyCoalitionCoordinator("edd"), seed=0)
```

## Contents

- [What is configurable](#what-is-configurable)
- [Built-in scenarios](#built-in-scenarios)
- [Custom scenarios](#custom-scenarios)
- [Interfaces](#interfaces)
- [How a tick works](#how-a-tick-works)
- [Baselines](#baselines)
- [Tests and reproducibility](#tests-and-reproducibility)

## What is configurable

Everything below is plain configuration (dict, JSON or YAML) and validated on load. The complete
field reference is in [docs/configuration.md](docs/configuration.md).

| Dimension | Per robot type (and per fleet group via `overrides`) | Effect |
|---|---|---|
| **Capability** | any set of strings: `transport`, `pick`, `lift`, `inspect`, `manipulate`, ... | which task roles a robot can fill |
| **Physical performance** | `speed` (cells/tick, fractional), `payload_kg`, `handling_speed`, `zones` / `zones_loaded` | travel time, what it may carry, how fast it works, where it may drive, empty or loaded |
| **Resources** | battery `capacity_wh`, `idle_w`, `move_w`, `payload_w_per_kg`, `handling_w`, `max_charge_w`, `charge_efficiency`; compatible `charger_types`; `initial_soc` range; random breakdowns (`failure.rate`, `repair_ticks`) | how long it can work, where and how fast it recharges, when it is out of service |
| **Information** | `sensing_range`, `comm_range` (one-hop relay), `localization_noise`; task board `global` or `sensing` | what each robot observes |
| **Cooperation** | per task type: roles `{capability, count, presence: on_cell or adjacent, radius, stay}`, ordered stages `{at, roles, duration}`, a `carrier_role` | crew composition, rendezvous and synchronized work; joint transport when the carrier role has `count > 1` |

The same structure covers tasks:

| | Options |
|---|---|
| **Task types** | any number; roles, stages, payload `weight` distribution (`fixed`, `uniform`, `normal`, `choice`), `priority`, `deadline`, `origin` / `destination` zones, stages at `origin`, `destination` or any custom `zone:<name>` |
| **Jobs** | dependency DAGs of tasks with location chaining (`origin_from: "receive.destination"`) |
| **Demand** | `backlog` (keep N open), `poisson` (constant, or piecewise `rates` for peaks), `batch`, `trace`; one task type, one job or a weighted `mix` per stream; any number of streams |
| **Chargers** | several types with their own `power_w` and `share` of the docks |
| **Map** | ASCII art (`.` aisle, `S` storage, `G` goal, `C` charger, `#` wall, custom zone symbols, optional one-way cells `^ v < >` and crossings `+`) or a generator (`rack_rows`, `rack_cols`, `rack_height`, `goals`, `docks`, `aisle_width`, `one_way`) |
| **Rewards** | task completion (× priority), stage completion, energy (per kWh), lateness (per tick), stranding, per tick |

A coalition is *feasible* only if every role is filled by distinct, operational robots with the
right capability, the carriers' combined payload capacity covers the load, every member can reach
its cells within its zones, and every member's estimated energy for its part leaves a reserve.
`env.sim.feasible_coalitions(task_id)` enumerates them, ranked by estimated completion time.

## Built-in scenarios

Each tier adds cooperation constraints that mean something, not just more robots.

| Scenario | Gym id | Robots | Tasks | What makes it hard |
|---|---|---|---|---|
| `basic` | `battery_tarware/Basic-v2` | 6 AGVs, 3 pickers | SOLO (1 AGV), STANDARD (AGV + picker), LARGE (AGV + 2 pickers), each picked, delivered and returned | role coupling: pickers are needed at pickup and at return |
| `intermediate` | `battery_tarware/Intermediate-v2` | light/fast and heavy/slow AGVs (one with an aged battery), fast and standard pickers | totes, cases, pallets (heavy AGV only), oversize loads (two AGVs carry jointly) | heterogeneity within roles, joint transport, peak/trough demand, fast chargers that heavy AGVs cannot use, pickers sense 6 cells |
| `advanced` | `battery_tarware/Advanced-v2` | AGVs, forklifts, mobile manipulators, inspection drones | orders through a quality-check zone, bulky joint orders, inbound jobs (receive → inspect, linked by location), relocations | 4 capability types, custom zones, dependency DAGs, deadlines, breakdowns, noisy drones with small batteries |

`battery_tarware/Warehouse-v2` takes any scenario: `gym.make("battery_tarware/Warehouse-v2", scenario=my_dict)`.
Top-level fields can be overridden, for example `bt.make("basic", action_mode="primitive", max_steps=500)`.

## Custom scenarios

A complete example (tug robots, a strong but slow lifter, a manipulator arm with limited sensing;
supply runs and a two-carrier sterile cart) is in [examples/custom_scenario.yaml](examples/custom_scenario.yaml):

```yaml
robot_types:
  - name: lifter
    capabilities: [transport, lift]
    speed: 0.5                 # moves every other tick
    payload_kg: 150
    zones: [aisle, storage, goal, dock]
    zones_loaded: [aisle, goal, dock]
    charger_types: [slow]
    energy: {capacity_wh: 3000, move_w: 300, payload_w_per_kg: 1.5, handling_w: 200}
    failure: {rate: 0.0005, repair_ticks: 40}
fleet:
  - {type: tug, count: 4, initial_soc: [0.4, 1.0]}
  - {type: tug, name: tug_old, count: 1, overrides: {speed: 0.8, energy: {capacity_wh: 500}}}
task_types:
  - name: sterile_cart
    carrier_role: carriers
    weight: {kind: normal, mean: 60, std: 10, low: 40, high: 80}
    roles:
      - {name: carriers, capability: transport, count: 2, presence: on_cell}   # joint transport
      - {name: handler, capability: manipulate, stay: true}
    stages:
      - {name: secure, at: origin, roles: [carriers, handler], duration: 4}
      - {name: hand_over, at: destination, roles: [carriers, handler], duration: 5}
```

```python
env = bt.make("examples/custom_scenario.yaml")            # or a dict, or bt.ScenarioConfig
cfg = bt.get_scenario("advanced"); cfg["fleet"][0]["count"] = 10; env = bt.make(cfg)   # edit a built-in
```

Configuration mistakes are rejected with a message: unknown keys, capabilities no robot has,
loads heavier than the strongest possible carrier crew, dependency cycles, unknown zones and so on.

## Interfaces

**Gymnasium** (`BatteryTarwareEnv`)

- `observation_space`: `Dict({robot_name: Dict(self, robots, tasks, chargers, action_mask)})`, all
  float32 and bounded. `robots` holds only the robots this robot can see; positions carry its
  localization noise.
- `action_space`: `MultiDiscrete([n_actions] * n_robots)`. `action_mode` selects the action set:
  - `macro`: `0` no-op · `1..K` join the task in board slot k · `K+1..K+C` charge at charger c · last = release.
  - `primitive`: `wait, up, down, left, right, interact, release`. `interact` docks at a charger, or joins the task the robot is positioned for.
- reward: team reward. Per-robot credit is in `info["robot_rewards"]`.
- `info`: `action_mask` (n_robots × n_actions), `events` (typed transition log for this tick),
  `metrics`, `invalid_actions`.
- `terminated`: finite demand is done, or every robot is stranded. `truncated`: `max_steps` reached.
- render modes: `rgb_array`, `ansi`, `human` (matplotlib).

**PettingZoo**: `battery_tarware.pettingzoo_env.BatteryTarwareParallelEnv(scenario, reward_mode="individual"|"team")`
passes `pettingzoo.test.parallel_api_test` and adds `state()` for centralized critics.

**Coordinators**: anything with `reset(env)` and `act(env) -> actions`.
- Centralized coordinators may also use the validated simulator API: `assign`, `assign_coalition`,
  `release`, `send_to_charger` and `feasible_coalitions`.
- Built in: `GreedyCoalitionCoordinator(rule)` and `RandomCoordinator`.
- Dispatching rules, each with full definition in `bt.RULES`: `fifo`, `edd`, `spt`, `cr`, `priority`.

**State for external code**: `env.state_view()` is a frozen snapshot of robots, tasks and
chargers. `.to_arrays()` returns contiguous, fixed-dtype numpy arrays (schema version 1, `(y, x)`
int32 coordinates) that C++ or other native coordinators can read directly, for example through
pybind11 `py::array_t`. See [examples/external_coordinator.py](examples/external_coordinator.py).

## How a tick works

1. **Actions**: commit to a task role, go and charge, release; or a primitive move or interact.
2. **Breakdowns**: random failures and repairs. A failed robot stops but keeps its commitment.
3. **Movement**:
   - *Macro mode*: robots move with [PIBT](https://doi.org/10.1016/j.artint.2022.103752) (Priority
     Inheritance with Backtracking), including the dead-end swap operation. Robots that are in the
     way are pushed aside, joint carriers move as one unit, and distances respect zones and one-way
     cells.
   - *Primitive mode*: simultaneous moves with deterministic conflict resolution.
4. **Stage work**: a stage advances only while its whole crew is present and operational, at the
   slowest member's handling speed. Crews are recruited stage by stage; helpers not needed in the
   next stage are released unless their role has `stay: true`.
5. **Energy**: idle, driving, payload (each carrier pays for its share of the load), handling and
   charging power, all in watts and integrated over `seconds_per_tick`. A battery at 0 % strands the
   robot permanently. If it was carrying, it drops the payload where it stands for another crew to
   recover.
6. **Demand**: new tasks and jobs are released. Dependent tasks become ready when their prerequisites complete.

Every random draw (spawn, demand, task sampling, failures, observation noise) has its own seeded
stream, so `reset(seed)` reproduces a run exactly.

## Baselines

Full-length episodes, seeds 0–4, from `python benchmarks/baselines.py`:

| scenario | policy | tasks completed | mean latency (ticks) | late | energy (kWh) | stranded |
|---|---|---|---|---|---|---|
| basic | greedy (FIFO) | 203.0 ± 11.0 | 77 | 0.0 | 2.51 | 0.0 |
| basic | random (masked) | 3.6 ± 2.2 | 302 | 0.0 | 3.06 | 0.0 |
| intermediate | greedy (FIFO) | 134.4 ± 8.9 | 68 | 0.0 | 3.00 | 0.0 |
| intermediate | random (masked) | 11.2 ± 2.8 | 671 | 0.0 | 7.13 | 2.0 |
| advanced | greedy (FIFO) | 293.0 ± 33.2 | 217 | 3.8 | 9.45 | 0.0 |
| advanced | greedy (priority) | 258.6 ± 54.4 | 211 | 14.0 | 8.82 | 0.4 |
| advanced | random (masked) | 0.6 ± 0.8 | 1394 | 0.0 | 15.43 | 0.8 |

On one CPU core the simulator runs about 400–1100 ticks per second with the greedy baseline.

## Tests and reproducibility

```bash
python -m pytest -q
```

The suite covers:
- configuration validation;
- strong connectivity of every built-in map for every robot type;
- exact energy accounting, charging, compatibility and stranding;
- fractional speeds, handling speed, synchronized crews, joint transport, payload drop and recovery,
  breakdowns;
- job dependencies and all demand processes;
- sensing, communication relay and noise;
- primitive mode and seed determinism;
- PIBT properties (no vertex or swap collisions, push chains, the dead-end swap);
- sustained throughput in every tier;
- Gymnasium `check_env` in both action modes, PettingZoo `parallel_api_test`;
- the examples.

## Origins and citation

Battery-TA-RWARE grew out of [TA-RWARE](https://github.com/raulsteleac/task-assignment-robotic-warehouse)
(Krnjaic et al., 2022), itself based on [RWARE](https://github.com/semitable/robotic-warehouse).
Version 2 is a new engine. The v1 Gymnasium environment is kept at git tag `legacy-v1.1.0`. See
[CITATION.cff](CITATION.cff) and [CHANGELOG.md](CHANGELOG.md). MIT licence.
