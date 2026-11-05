# Battery-TA-RWARE

**Heterogeneous multi-robot cooperation under battery constraints**: a configurable warehouse
simulator for task allocation, coalition formation and charging. It is a
[Gymnasium](https://gymnasium.farama.org) environment with a
[PettingZoo](https://pettingzoo.farama.org) wrapper.

<table>
  <tr>
    <td align="center"><img src="docs/img/basic.gif" width="400" alt="basic scenario"><br><b>basic</b>: AGVs and pickers, pick + return</td>
    <td align="center"><img src="docs/img/intermediate.gif" width="400" alt="intermediate scenario"><br><b>intermediate</b>: light/heavy AGVs, joint transport</td>
  </tr>
  <tr>
    <td align="center"><img src="docs/img/advanced.gif" width="400" alt="advanced scenario"><br><b>advanced</b>: AGVs, forklifts, arms, drones, custom zones</td>
    <td align="center"><img src="docs/img/tutorial.gif" width="400" alt="tutorial scenario"><br><b>tutorial</b>: the scenario built below, in YAML</td>
  </tr>
</table>

<sub>Rendered in the TA-RWARE style:
**shape** = robot type (hexagon, diamond, triangle, octagon, ...);
**colour** = activity: orange idle, blue travelling, violet waiting for the crew, red working a
stage, maroon carrying, green charging, grey broken down, black stranded.
Shelves are dark blue, a requested shelf (payload waiting) is teal, and a purple frame marks where
a carried payload is going. The badge is the battery (green > 80 %, yellow > 20 %, red), and the
tick shows the last move direction.
All runs use the greedy baseline, seed 0. Regenerate them with `python docs/make_gifs.py`; the
same image comes from `env.render()` with `render_mode="rgb_array"`.</sub>

The simulator rests on three ideas:

- Robots differ in what they can do, how fast they move, what they can lift, how long their
  battery lasts, which chargers they can use and how much they can see.
- Tasks say which capabilities they need, how many robots of each, where the crew must meet and
  for how long.
- The environment works out which teams *can* do a task. Your coordinator or learning agent
  decides who actually does it.

> **Version 2 is a new engine.** It is not trajectory-compatible with Battery-TA-RWARE v1 (branch
> [`v1`](../../tree/v1), tag `legacy-v1.1.0`), or with results obtained on the v1 kernel. The rules
> are fixed in [docs/semantics.md](docs/semantics.md) (model version 2.0). Report the package
> version with your results.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Build your own scenario](#build-your-own-scenario)
- [What is configurable](#what-is-configurable)
- [Built-in scenarios](#built-in-scenarios)
- [Interfaces](#interfaces)
- [How a tick works](#how-a-tick-works)
- [Baselines](#baselines)
- [Citing, licence and acknowledgements](#citing-licence-and-acknowledgements)

## Install

Python ≥ 3.10.

```bash
pip install "battery-ta-rware @ git+https://github.com/Cyber-physical-Systems-Lab/Battery-TA-RWARE@v2.0.0"
# with the optional extras (PettingZoo, YAML scenarios, matplotlib window):
pip install "battery-ta-rware[all] @ git+https://github.com/Cyber-physical-Systems-Lab/Battery-TA-RWARE@v2.0.0"
```

Or install from a clone:

```bash
git clone https://github.com/Cyber-physical-Systems-Lab/Battery-TA-RWARE.git
cd Battery-TA-RWARE
pip install -e ".[all]"
```

The base install needs only `numpy` and `gymnasium`. The optional extras are `pettingzoo`, `yaml`,
`render` (matplotlib) and `gifs` (Pillow), or `all`. The distribution is called
`battery-ta-rware`; the Python package is imported as `battery_tarware`.

## Quick start

```python
import gymnasium as gym
import numpy as np
import battery_tarware as bt        # registers battery_tarware/{Basic,Intermediate,Advanced,Warehouse}-v2

env = gym.make("battery_tarware/Intermediate-v2")
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
for _ in range(100):
    mask = info["action_mask"]                                   # (n_robots, n_actions), 1 = allowed
    action = [rng.choice(np.flatnonzero(m)) for m in mask]       # a random valid action per robot
    obs, reward, terminated, truncated, info = env.step(action)
print(info["metrics"])

# or let a coordinator drive a whole episode
metrics = bt.run_episode(bt.make("advanced"), bt.GreedyCoalitionCoordinator("edd"), seed=0)
```

More examples are in [examples/](examples):

- [quickstart.py](examples/quickstart.py): a masked random policy and the greedy baseline.
- [pettingzoo_and_primitive.py](examples/pettingzoo_and_primitive.py): the PettingZoo API and
  primitive (move-level) actions.
- [external_coordinator.py](examples/external_coordinator.py): your own coordinator on read-only
  state views and numpy arrays.
- [custom_scenario.py](examples/custom_scenario.py) and
  [custom_scenario.yaml](examples/custom_scenario.yaml): a hospital-logistics fleet.
- [tutorial.yaml](examples/tutorial.yaml): the scenario built in the next section.

## Build your own scenario

A scenario is a plain mapping (a Python dict, a JSON file or a YAML file) that is validated on
load. The full file of this walkthrough is [examples/tutorial.yaml](examples/tutorial.yaml), and
every field is documented in [docs/configuration.md](docs/configuration.md).

**1. Map.** Draw it in ASCII or use the generator. Custom zones get their own symbol.

```yaml
map:
  ascii:
    - "C..C..C..C...."        # C charger dock
    - ".............."        # . aisle
    - ".SS..SS..SS..."        # S storage
    - ".SS..SS..SS..."
    - ".............."
    - "GG...PPP...GG."        # G goal, P custom zone
  zones: {P: packing}
# or: map: {generator: {rack_rows: 2, rack_cols: 3, rack_height: 4, goals: 2, docks: 4, aisle_width: 2}}
```

One-way cells (`^ v < >`) and crossings (`+`) are optional, as are walls (`#`).

**2. Robot types.** A robot type sets what a robot can do (capabilities), how it moves and works,
its battery, what it can see and how often it breaks down.

```yaml
robot_types:
  - name: agv
    capabilities: [transport]
    speed: 1.0                                   # cells per tick, (0, 1]
    payload_kg: 80
    zones: [aisle, storage, goal, dock, packing]
    zones_loaded: [aisle, goal, dock, packing]   # not under racks while loaded
    energy: {capacity_wh: 1000, idle_w: 20, move_w: 140, payload_w_per_kg: 1.5, handling_w: 40}
  - name: picker
    capabilities: [pick, pack]
    speed: 0.5                                   # one cell every two ticks
    handling_speed: 1.5                          # works stages 1.5x faster
    zones: [aisle, dock, packing]
    charger_types: [standard]                    # cannot use fast chargers
    sensing_range: 5                             # partial observability
    energy: {capacity_wh: 600, idle_w: 15, move_w: 90, handling_w: 120}
    failure: {rate: 0.0002, repair_ticks: 30}    # random breakdowns
```

**3. Fleet.** Set how many robots of each type there are and their initial charge. A group can
override fields of its type, for example to model an aged battery.

```yaml
fleet:
  - {type: agv, count: 5, initial_soc: [0.5, 1.0]}
  - {type: agv, name: agv_old, count: 1, overrides: {energy: {capacity_wh: 500}}}
  - {type: picker, count: 3}
```

**4. Chargers.** Charger types differ in power, and each type takes a share of the docks.

```yaml
charger_types:
  - {name: standard, power_w: 800, share: 1}
  - {name: fast, power_w: 2500, share: 1}
```

**5. Task types.** A task type defines its roles (who is needed, how many, and how close they must
be) and its stages (where, how long and with which roles). The `carrier_role` holds the payload. A
carrier role with `count > 1` means joint transport.

```yaml
task_types:
  - name: tote
    carrier_role: carrier
    weight: {kind: uniform, low: 1, high: 15}
    deadline: 300
    roles:
      - {name: carrier, capability: transport, presence: on_cell}
      - {name: picker, capability: pick, presence: adjacent, radius: 1}
    stages:
      - {name: pick, at: origin, roles: [carrier, picker], duration: 3}
      - {name: pack, at: "zone:packing", roles: [carrier, picker], duration: 2}
      - {name: drop, at: destination, roles: [carrier], duration: 1}
  - name: pallet                                 # 90-150 kg: two 80 kg AGVs must carry it together
    carrier_role: carriers
    priority: 2
    weight: {kind: normal, mean: 110, std: 15, low: 90, high: 150}
    roles:
      - {name: carriers, capability: transport, count: 2, presence: on_cell}
    stages:
      - {name: lift, at: origin, roles: [carriers], duration: 4}
      - {name: drop, at: destination, roles: [carriers], duration: 2}
```

**6. Jobs (optional).** A job links tasks by dependencies and by location.

```yaml
jobs:
  - name: order_and_return
    tasks:
      - {key: order, type: tote}
      - {key: back, type: return_tote, depends_on: [order], origin_from: order.destination}
```

**7. Demand.** Demand comes from any number of streams. Each stream is a backlog, Poisson arrivals
(constant or piecewise rates), a batch, or a trace of release times.

```yaml
demand:
  - {kind: backlog, open: 3, task_type: tote}                               # always 3 totes waiting
  - {kind: poisson, task_type: pallet, rates: [[0, 0.005], [1000, 0.02]]}   # a peak after tick 1000
  - {kind: poisson, job: order_and_return, rate: 0.004}
```

**8. Rewards, observations and episode length.**

```yaml
rewards: {task_completed: 1.0, late_per_tick: -0.01, energy_per_kwh: -0.2, stranded: -10.0}
observation: {task_slots: 8, task_visibility: global}
max_steps: 2000
action_mode: macro            # or primitive
```

**9. Run it.**

```python
env = bt.make("examples/tutorial.yaml")              # a path, a dict or a bt.ScenarioConfig
env = gym.make("battery_tarware/Warehouse-v2", scenario="examples/tutorial.yaml")

cfg = bt.get_scenario("advanced")                    # or start from a built-in and edit it
cfg["fleet"][0]["count"] = 10
env = bt.make(cfg, max_steps=3000, action_mode="primitive")   # top-level overrides
```

Configuration mistakes are rejected with an explanation. Examples are unknown keys, a capability
no robot has, a load heavier than the strongest possible carrier crew, a dependency cycle or an
unknown zone:

```
ConfigError: task type 'solo': role 'carrier' needs 1 robot(s) with 'fly', fleet has 0
```

## What is configurable

Each robot type sets the fields below; each fleet group can override them.

| Dimension | Fields | Effect |
|---|---|---|
| **Capability** | any set of strings: `transport`, `pick`, `lift`, `inspect`, `manipulate`, ... | which task roles a robot can fill |
| **Physical performance** | `speed` (cells per tick, fractional), `payload_kg`, `handling_speed`, `zones`, `zones_loaded` | travel time; what it may carry; how fast it works; where it may drive, empty or loaded |
| **Resources** | battery: `capacity_wh`, `idle_w`, `move_w`, `payload_w_per_kg`, `handling_w`, `max_charge_w`, `charge_efficiency`; compatible `charger_types`; `initial_soc` range; breakdowns: `failure.rate`, `repair_ticks` | how long it can work; where and how fast it recharges; when it is out of service |
| **Information** | `sensing_range`, `comm_range` (one-hop relay), `localization_noise`; task board `global` or `sensing` | what each robot observes |
| **Cooperation** | per task type: roles `{capability, count, presence (on_cell or adjacent), radius, stay}`, ordered stages `{at, roles, duration}`, a `carrier_role` | crew composition, rendezvous and synchronized work; joint transport when the carrier role has `count > 1` |

Tasks, demand and the environment are configured the same way:

| | Options |
|---|---|
| **Task types** | any number of them; roles; stages at `origin`, `destination` or any `zone:<name>`; payload `weight` (`fixed`, `uniform`, `normal`, `choice`); `priority`; `deadline`; `origin` and `destination` zones |
| **Jobs** | dependency DAGs of tasks, with location chaining (`origin_from: "order.destination"`) |
| **Demand** | `backlog`, `poisson` (constant or piecewise `rates`), `batch`, `trace`; one task type, one job or a weighted `mix` per stream; any number of streams |
| **Chargers** | several types, each with its own `power_w` and `share` of the docks |
| **Map** | ASCII (custom zone symbols, one-way cells, walls) or a generator (`rack_rows`, `rack_cols`, `rack_height`, `goals`, `docks`, `aisle_width`, `one_way`) |
| **Rewards** | task completion (× priority), stage completion, energy (per kWh), lateness (per tick), stranding, per tick |

A coalition is *feasible* only if all of the following hold:

- every role is filled by distinct, operational robots with the right capability;
- the carriers' combined payload capacity covers the load;
- every member can reach its cells within its allowed zones;
- every member's estimated energy for its part leaves a reserve.

`env.unwrapped.sim.feasible_coalitions(task_id)` enumerates the feasible coalitions, ranked by
estimated completion time.

## Built-in scenarios

Each tier adds new cooperation constraints, not just more robots.

| Scenario | Gym id | Robots | Tasks | Main challenges |
|---|---|---|---|---|
| `basic` | `battery_tarware/Basic-v2` | 6 AGVs, 3 pickers | SOLO (1 AGV), STANDARD (AGV + picker) and LARGE (AGV + 2 pickers), each picked, delivered and returned | role coupling: pickers are needed at pickup and at return |
| `intermediate` | `battery_tarware/Intermediate-v2` | light/fast and heavy/slow AGVs (one group with an aged battery), fast and standard pickers | totes, cases, pallets (heavy AGVs only), oversize loads carried jointly by two AGVs | heterogeneity within roles, joint transport, peak and trough demand, fast chargers that heavy AGVs cannot use, pickers that sense only 6 cells |
| `advanced` | `battery_tarware/Advanced-v2` | AGVs, forklifts, mobile manipulators, inspection drones | orders through a quality-check zone, bulky joint orders, inbound jobs (receive → inspect, linked by location), relocations | 4 capability types, custom zones, dependency DAGs, deadlines, breakdowns, noisy drones with small batteries |

`battery_tarware/Warehouse-v2` runs any scenario. `bt.get_scenario(name)` returns a built-in as an
editable dict.

## Interfaces

**Gymnasium** (`BatteryTarwareEnv`)

- `observation_space` is `Dict({robot_name: Dict(self, robots, tasks, chargers, action_mask)})`;
  all entries are float32 and bounded. `robots` holds only the robots that this robot can see, at
  positions that include its localization noise.
- `action_space` is `MultiDiscrete([n_actions] * n_robots)`. `action_mode` selects the action set:
  - `macro` (the default): `0` no-op; `1..K` join the task in board slot k; `K+1..K+C` charge at
    charger c; the last action releases the robot. Movement is automatic.
  - `primitive`: `wait`, `up`, `down`, `left`, `right`, `interact`, `release`. `interact` docks at
    a charger, or joins the task the robot is positioned for.
- The reward is the team reward. Per-robot credit is in `info["robot_rewards"]`.
- `info` holds:
  - `action_mask`: n_robots × n_actions;
  - `events`: the typed transition log for this tick;
  - `metrics`;
  - `invalid_actions`.
- `terminated`: the finite demand is done, or every robot is stranded. `truncated`: `max_steps` has
  been reached.
- Render modes: `rgb_array`, `ansi`, `human` (matplotlib).

**PettingZoo**: `battery_tarware.pettingzoo_env.BatteryTarwareParallelEnv(scenario, reward_mode="individual" | "team")`
passes `pettingzoo.test.parallel_api_test`. It also provides `state()` for centralized critics.

**Coordinators** are objects with `reset(env)` and `act(env) -> actions`.

- A centralized coordinator can also call the validated simulator API on `env.unwrapped.sim`:
  `assign`, `assign_coalition`, `release`, `send_to_charger` and `feasible_coalitions`. Every call
  returns `(ok, reason)`.
- Built in: `GreedyCoalitionCoordinator(rule)` and `RandomCoordinator`.
- Dispatching rules: `fifo`, `edd`, `spt`, `cr`, `priority`. Each is defined in full in `bt.RULES`.

**State for external code**: `env.unwrapped.state_view()` is a frozen snapshot of the robots, tasks
and chargers. `.to_arrays()` returns contiguous, fixed-dtype numpy arrays (schema version 1, int32
`(row, col)` coordinates). C++ or other native coordinators can read them directly, for example
through pybind11 `py::array_t`.

## How a tick works

Each tick runs these phases in order (details in [docs/semantics.md](docs/semantics.md)):

1. **Actions**: join a task role, go and charge, or release; in primitive mode, move or interact.
2. **Breakdowns**: random failures and repairs. A failed robot stops but keeps its commitment.
3. **Movement**:
   - *Macro mode*: robots move with [PIBT](https://doi.org/10.1016/j.artint.2022.103752). Robots
     in the way are pushed aside, joint carriers move as one unit, and distances respect zones and
     one-way cells.
   - *Primitive mode*: simultaneous moves with deterministic conflict resolution.
4. **Stage work**: a stage advances only while its whole crew is present and operational, at the
   slowest member's handling speed. Crews are recruited stage by stage.
5. **Energy**: idle, driving, payload (each carrier pays for its share), handling and charging
   power, in watts, integrated over `seconds_per_tick`.
   - A battery at 0 % strands the robot permanently.
   - A stranded carrier drops its payload for another crew to recover.
6. **Demand**: new tasks and jobs are released. Dependent tasks become ready once their
   prerequisites complete.

Every random process (spawn positions, demand, task sampling, failures and observation noise) has
its own seeded stream, so `reset(seed)` reproduces a run exactly.

## Baselines

These are full-length episodes over seeds 0–4, produced by `python benchmarks/baselines.py`:

| Scenario | Policy | Tasks completed | Mean latency (ticks) | Late | Energy (kWh) | Stranded |
|---|---|---|---|---|---|---|
| basic | greedy (FIFO) | 203.0 ± 11.0 | 77 | 0.0 | 2.51 | 0.0 |
| basic | random (masked) | 3.6 ± 2.2 | 302 | 0.0 | 3.06 | 0.0 |
| intermediate | greedy (FIFO) | 134.4 ± 8.9 | 68 | 0.0 | 3.00 | 0.0 |
| intermediate | random (masked) | 11.2 ± 2.8 | 671 | 0.0 | 7.13 | 2.0 |
| advanced | greedy (FIFO) | 293.0 ± 33.2 | 217 | 3.8 | 9.45 | 0.0 |
| advanced | greedy (priority) | 258.6 ± 54.4 | 211 | 14.0 | 8.82 | 0.4 |
| advanced | random (masked) | 0.6 ± 0.8 | 1394 | 0.0 | 15.43 | 0.8 |

Compare these numbers only within a scenario. The tiers differ in map size, fleet, demand process,
task mix and episode length. Raw task counts are therefore not normalized for difficulty and do not
rank the tiers: advanced completing more tasks than basic does not make it easier. Intermediate is
demand-limited; its Poisson arrivals, not the fleet, cap throughput.

On one CPU core the simulator runs about 400–1100 ticks per second with the greedy baseline.

## Citing, licence and acknowledgements

If you use Battery-TA-RWARE, please cite it using [CITATION.cff](CITATION.cff) and state the
package version:

```bibtex
@software{battery_ta_rware,
  author  = {Niu, Xuezhi},
  title   = {Battery-TA-RWARE: heterogeneous multi-robot cooperation under battery constraints},
  version = {2.0.0},
  year    = {2026},
  url     = {https://github.com/Cyber-physical-Systems-Lab/Battery-TA-RWARE}
}
```

Battery-TA-RWARE grew out of [TA-RWARE](https://github.com/raulsteleac/task-assignment-robotic-warehouse)
(Krnjaic et al., 2022), which builds on [RWARE](https://github.com/semitable/robotic-warehouse)
(Papoudakis et al., 2021). Version 2 is a new engine. The macro-mode movement is an independent
implementation of PIBT (Okumura et al., 2022) with a restricted swap operation (Okumura, 2023). See
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for full credits.

The v1 environment (package `tarware`) remains available on the [`v1`](../../tree/v1) branch and at
tag `legacy-v1.1.0`.

The software is released under the MIT licence; see [LICENSE](LICENSE). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).
