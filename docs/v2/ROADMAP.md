# Battery-TA-RWARE v2 roadmap

Status: development branch `v2-dev`, 2026-11-05.

## Positioning

Battery-TA-RWARE is a general-purpose simulator and benchmark for **heterogeneous multi-robot task
execution and cooperation** under battery constraints. It is not tied to any single paper. Research
code (coordinators, learning methods, experiments) lives in downstream repositories that pin a
released version, e.g. RendeCoord:

```toml
dependencies = ["battery-tarware @ git+https://github.com/Cyber-physical-Systems-Lab/Battery-TA-RWARE@<tag>"]
```
(a package-index release such as `battery-tarware==2.0.0` once one exists).

The kernel never imports research code. `tests/test_kernel_independence.py` enforces this.

## Packages in this repository

| Package | What | Stability |
|---|---|---|
| `battery_tarware` | v2 kernel: battery/charging physics, tiled layouts, SOLO/STANDARD/LARGE cooperative tasks, external arrivals, charger capacity; `interfaces` (read-only `StateView`/`LayoutView`), `tasks` (task model, DAG, lifecycle), `testing` (trajectory fingerprints) | evolving; frozen profiles regression-locked |
| `tarware` | original TA-RWARE v1.1.0 Gymnasium RL environment (tag `legacy-v1.1.0`) | frozen, unchanged |

## Version isolation (replaces "every refactor must keep every trajectory")

| Version | Goal | How it is protected |
|---|---|---|
| `legacy_v1` and `protocol_v4` **frozen profiles** | reproduce existing results exactly | `tests/test_frozen_profile_fingerprints.py`: per-tick digests recorded at RendeCoord `bt-g0-baseline`. Never updated to make a change pass. |
| v2 profiles | clearer semantics, richer heterogeneity and tasks | new profiles with their own deterministic tests and physical/task invariants; behavior changes listed in `CHANGELOG.md` |
| later versions | new robot models, cooperation mechanisms, scenarios | same rule: new profile + release note |

A v2 change may fix unreasonable semantics (navigation, resource model, broken action masks) but must
do it in a v2 profile, leaving the frozen profiles bit-identical.

## Heterogeneity model (P1 core)

| Dimension | Configurable | Effect |
|---|---|---|
| Capability | transport, lift, pick, inspect, manipulate | which robots can serve a task |
| Physical performance | speed, payload limit, load/unload time, workspace | same task, different execution |
| Resources | battery capacity, consumption, charger compatibility, payload state | available time and capability change over time |
| Information | sensing range, localization accuracy, communication | robots see different state |
| Cooperation | crew size, synchronization, joint capability, spatial limits | solo-feasible ≠ jointly feasible |

Heterogeneity **within a role** is required (two AGVs with different payload, speed and energy; two
pickers with different handling time and AGV compatibility), so team formation becomes
capability-dependent rather than a fixed AGV:picker ratio.

Central abstraction, owned by the environment:
task requirements + robot capabilities + joint execution constraints → **feasible coalitions**.
*Which* feasible coalition is chosen is the coordinator's job and stays pluggable.

## Target v2 API (not implemented yet)

```python
env = battery_tarware.make("heterogeneous-warehouse-v2", config=config)
state = env.reset(seed=42)
while not env.done:
    decision = coordinator.decide(state)          # StateView -> CoordinationDecision
    state, events = env.step(decision)
```
Gymnasium / PettingZoo / SMDP are adapters on top. C++ coordinators connect through primitive-array
`StateView`/`ActionBatch` DTOs; RendeCoord's `EnvSnapshot` is a migration reference, not a dependency.
The legacy `Warehouse.step(macro_actions) -> (done, info)` keeps working for frozen profiles.

## Benchmark tiers

- **Basic**: standard AGV/picker, simple transport, fixed resources.
- **Intermediate**: robots with different capability and performance, dynamic arrivals, shared
  charging and traffic constraints.
- **Advanced**: multi-stage dependent tasks, joint manipulation, changing capability and resources,
  failure recovery.

Difficulty comes from explicit cooperation constraints, not from more robots.

## Parallel tracks

| Track | Scope | First deliverables |
|---|---|---|
| P0 simulator engineering | OOP kernel (`Simulator`, `World`, robots, tasks, navigation, energy, events), config, C++-friendly DTOs | ✅ standalone kernel + independence test; ✅ `StateView`/`LayoutView`; ✅ frozen-profile fingerprints. Next: robot model with capability/performance/resource attributes, event recorder, typed config. |
| P1 heterogeneous benchmark | robot types, capabilities, cooperative tasks, dependencies, dynamic demand, tiered scenarios | ✅ `tasks`: `TaskSpec`, `TaskGraph` (cycle / unknown-dependency / fleet-feasibility checks, deterministic order), lifecycle state machine with retry budget and reassignment ≠ failure. Next: capability model + coalition feasibility, task generators. |
| P2 algorithm ecosystem | coordinator plugins, Gymnasium/PettingZoo, SMDP, RL examples, scheduling baselines with full definitions, MAPF interfaces | next: coordinator protocol + mock conformance tests, PettingZoo Parallel adapter (optional extra). |

All three tracks build against [SIMULATION_CONTRACTS_V1.md](SIMULATION_CONTRACTS_V1.md); a contract
change bumps its schema version.

## Known issues inherited from the source kernel

- `Warehouse.compute_valid_action_masks()` reads a missing `action_size` and raises. Fix in a v2
  profile; frozen profiles do not use it.
- Global RNG streams (`random`, `np.random` legacy MT19937) are seeded in `reset()`. v2 should move to
  per-environment streams, as a v2 profile change.
- IDs equal `index + 1` and are class-level counters; consumers must index by ID.
