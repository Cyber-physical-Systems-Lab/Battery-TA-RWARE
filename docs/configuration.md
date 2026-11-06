# Configuration reference

A scenario is a mapping (dict / JSON / YAML) validated by `battery_tarware.ScenarioConfig.from_dict`.
Unknown keys are errors. Units: ticks, grid cells, watts, watt-hours, kilograms.
Coordinates are `(row, col)`.

## Top level

| key | type | default | meaning |
|---|---|---|---|
| `name` | str | required | scenario name |
| `description` | str | `""` | |
| `map` | [map](#map) | required | layout |
| `robot_types` | list of [robot type](#robot-type) | required | robot models |
| `fleet` | list of [fleet group](#fleet-group) | required | how many robots of which model |
| `charger_types` | list of [charger type](#charger-type) | one `standard` 1000 W | charger models, spread over the docks |
| `task_types` | list of [task type](#task-type) | required | task families |
| `jobs` | list of [job](#job) | `[]` | dependency-linked task groups |
| `demand` | list of [demand stream](#demand-stream) | required | where tasks come from |
| `action_mode` | `macro` \| `primitive` | `macro` | action set |
| `max_steps` | int | 5000 | truncation |
| `model` | `"2.1"` \| `"2.0"` | `"2.1"` | simulation model version ([semantics](semantics.md)); `"2.0"` reproduces model 2.0 |
| `seconds_per_tick` | float | 5.0 | converts power (W) to energy per tick |
| `rewards` | [rewards](#rewards) | | |
| `observation` | [observation](#observation) | | |

## Map

Exactly one of `ascii` or `generator`.

- `ascii`: one string, or a list of row strings, using `.` aisle, `S` storage, `G` goal, `C` charger dock, `#` wall. One-way aisle
  cells `^ v < >` and crossings `+` are optional. Robots leave a one-way cell in its direction, or
  turn off sideways into a non-street cell; they never enter one against its direction.
- `zones`: `{symbol: name}` for custom zones (for example `I: receiving`). A custom zone can be
  used in robot `zones`, task `origin` / `destination` and stage `at: zone:<name>`.
- `generator`:

  | key | default | meaning |
  |---|---|---|
  | `rack_rows` | 2 | blocks of racks, vertically |
  | `rack_cols` | 3 | blocks of racks, horizontally |
  | `rack_height` | 4 | storage cells per block, vertically |
  | `goals` | 2 | goal cells |
  | `docks` | 4 | charger docks |
  | `aisle_width` | 2 | aisle width in cells |
  | `one_way` | false | one-way street grid; only with `aisle_width: 1` |

- `dock_layout`: `inline` (default) or `bays`. With `bays`, every row made only of docks and aisle
  cells has its aisle cells walled off, so each dock is a one-cell bay that opens onto the corridor
  next to it and a charging robot blocks no traffic. Works with `ascii` and `generator` maps; not with
  one-way streets.

## Robot type

| key | default | meaning |
|---|---|---|
| `name` | required | |
| `capabilities` | required | list of strings |
| `speed` | 1.0 | cells per tick in (0, 1]; fractional speeds accumulate credit |
| `payload_kg` | 0 | maximum load (0 = cannot carry) |
| `handling_speed` | 1.0 | stage work rate; a stage runs at the slowest crew member's rate |
| `zones` | aisle, storage, goal, dock | zones it may enter (must include `aisle`) |
| `zones_loaded` | = `zones` | zones while carrying; the destination cell itself is always allowed |
| `charger_types` | all | compatible charger types |
| `energy.capacity_wh` | 1843 | battery |
| `energy.idle_w` | 20 | always drawn while operational |
| `energy.move_w` | 142 | extra while driving |
| `energy.payload_w_per_kg` | 1.7 | extra while driving, per kg of its share of the load |
| `energy.handling_w` | 60 | extra while working a stage |
| `energy.max_charge_w` | 1500 | robot-side charging limit |
| `energy.charge_efficiency` | 0.95 | |
| `sensing_range` | null (all) | Chebyshev cells within which it sees other robots |
| `comm_range` | null (all) | shares what it sees with robots this close (both ranges must cover the distance); 0 = none |
| `localization_noise` | 0 | std (cells) of Gaussian noise on positions it observes |
| `failure.rate` | 0 | breakdown probability per operational tick |
| `failure.repair_ticks` | 0 | downtime per breakdown |

## Fleet group

| key | default | meaning |
|---|---|---|
| `type` | required | robot type name |
| `count` | 1 | |
| `name` | = type | robots are named `<name>_<i>` |
| `initial_soc` | 1.0 | value or `[low, high]` (uniform) |
| `overrides` | `{}` | any robot-type field for this group only; `energy` / `failure` are merged |

## Charger type

`name`, `power_w`, `share` (relative fraction of docks). Docks are assigned to types in
proportion to `share`, interleaved.

## Task type

| key | default | meaning |
|---|---|---|
| `name` | required | |
| `roles` | required | list of `{name, capability, count: 1, presence: adjacent \| on_cell, radius: 1, stay: false}` |
| (alternatives) | | instead of `capability`/`presence`/`radius`: `alternatives: [{capability, presence: adjacent \| on_cell, radius: 1}, ...]`. The slot can be filled through any listed capability, and the robot works at that alternative's position (first matching alternative in list order). Not allowed on the carrier role. Capability-indexed features (`task_open_capability`, the observation's open/required capability counts, `Task.required`) count such a slot under every capability that can fill it, and a state-view `open_roles` entry shows the capabilities joined by `\|` |
| `stages` | required | ordered list of `{name, at: origin \| destination \| zone:<name>, roles, duration}` |
| (presence) | | `on_cell`: the robot stands on the stage cell. In model 2.1 a non-carrier `on_cell` role shares that cell with the carrier while they work (e.g. a picker on the shelf cell beside the AGV); give a second helper its own `adjacent` role. `adjacent`: within `radius` (Chebyshev) of the stage cell |
| `carrier_role` | null | role holding the payload; it must take part in every stage; `count > 1` = joint transport (capacities add up, the carriers move as one unit, each pays for its share) |
| `weight` | 0 | number, or `{kind: fixed \| uniform \| normal \| choice, ...}` |
| `priority` | 1.0 | scales the completion reward; used by the `priority` rule |
| `deadline` | null | ticks after release; lateness is penalized via `rewards.late_per_tick` |
| `origin` | `storage` | zone the origin cell is drawn from (open tasks never share an origin) |
| `destination` | `goal` | zone the destination cell is drawn from |

Semantics:
- A stage progresses only while all its roles are filled and present (`on_cell`: on the stage cell;
  `adjacent`: within `radius`).
- A robot fills a role with `alternatives` through the first alternative whose capability it has,
  and must be present at that alternative's position. A robot never holds two roles of one task.
- Robots are recruited for the current stage only. After a stage, roles the next stage does not
  need are released unless `stay: true`.
- The payload is picked up when the first stage completes and placed when the last one completes.

## Job

`{name, tasks: [{key, type, depends_on: [keys], origin_from: "<key>.origin|destination",
destination_from: ...}]}`. Dependencies must be acyclic. A task becomes ready once all its
dependencies have completed.

## Demand stream

Give exactly one of `task_type`, `job` or `mix: {name: weight}`.

| `kind` | keys | meaning |
|---|---|---|
| `backlog` | `open` | keep this many items outstanding |
| `poisson` | `rate` or `rates: [[tick, rate], ...]` | Poisson arrivals per tick (piecewise schedule) |
| `batch` | `count` | all at tick 0 |
| `trace` | `trace: [ticks]` | explicit release times |

`max_items` caps any stream. Finite demand ends the episode (`terminated`) once everything is done.

## Rewards

`task_completed` (× priority, shared by the crew), `stage_completed`, `energy_per_kwh`,
`late_per_tick`, `stranded` (once per robot), `per_tick`.

## Observation

`task_slots` (K: tasks on the board, also the number of macro join actions) and
`task_visibility: global | sensing`.
