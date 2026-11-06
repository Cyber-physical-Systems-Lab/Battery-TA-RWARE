# Model 2.1 behavioural gate — pre-registration

Registered 2026-11-06, before any model 2.1 code was written or run.

**Branch.** `fix/model-2.1-traffic`, isolated worktree. Base commit `797000c` (v2-dev). Its `battery_tarware/`
package tree is identical to `release/v2.0.0` (`eef06ac`); only the tests and CHANGELOG differ.

**Gate purpose.** Establish that movement, docking and charger contention have defensible semantics BEFORE
any RA-L coordination experiment runs on v2. Correctness first, performance second: a throughput gain never
compensates for a failed safety or liveness criterion.

## 1. Observed problems in model 2.0 (motivation)

Large basic, greedy FIFO, seed 0, ticks 850–950:
- 4 robots each had a distinct, free dock assigned. None moved for over 100 ticks while 4 docks stayed empty.
- **Cause.** PIBT ranks moves by a static, cached distance map in which charging/working/failed robots are not
  walls. "Stay" therefore outranks a +1 detour. The generator places docks IN row 0, which is also the top
  corridor, so a charging robot closes the corridor.
- **Scale.** 1.6k–3.3k stalled robot-ticks per episode behind charging robots (large basic, seeds 0–2), and
  440–570 behind working robots (large intermediate).

Code reading, not observed in the greedy runs:
- `can_charge()` / `send_to_charger()` neither check nor reserve the dock. Two robots can be sent to one dock
  through the API or macro actions. Greedy avoids this with its own `targeted` set.
- Charger occupancy is registered only when charging starts.

## 2. Model 2.1 changes (to be implemented)

Selected by `model: "2.1"`, the new default. `model: "2.0"` must reproduce 2.0 exactly.

**M1 — Charger admission and reservation.** Each dock has a state:

| State | Meaning |
|---|---|
| available | no reservation, no occupant |
| reserved | assigned to a robot that has not docked yet |
| occupied | a robot is charging on it |
| unavailable | a non-operational robot that does not hold the dock stands on its cell |

- `can_charge` rejects a dock reserved by or occupied by another robot. `send_to_charger` reserves it.
- Release, stop-charging, full, stranding and reassignment clear the reservation. A broken-down robot keeps
  its reservation, as it keeps its task.
- The action mask follows `can_charge`. There is no implicit queueing: a robot that cannot get a dock stays
  where it is, and the controller defers.
- Primitive `interact` obeys the same rule.

**M2 — Movement and waiting.**
- **Detour.** If every strictly-closer neighbour of a moving unit is held by a fixed unit (charging, working,
  failed, or without speed credit), its moves are ranked by distances that treat the fixed units' cells as
  walls.
- **Dock staging.** A robot whose reserved dock cell is held by a fixed unit of another robot waits at a staging
  cell (the existing `_staging_cell`) instead of pressing towards the dock.
- No new MAPF algorithm.

**L — Layout.** `map.dock_layout: "inline"` (the default, = 2.0) | `"bays"`.
- In `bays`, every row whose cells are only docks and aisle has its aisle cells turned into walls. Each dock
  then becomes a one-cell bay opening onto the corridor next to it.
- This applies to both ASCII and generated maps.
- `bays` + one-way streets is rejected as a configuration error.

## 3. Design

**Factorial** (all macro mode, greedy FIFO coordinator):

| Config | Rules | Dock layout |
|---|---|---|
| A | model 2.0 | inline |
| B | model 2.1 | inline |
| C | model 2.0 | bays |
| D | model 2.1 | bays |

**Scenarios.** 6, each in its documented configuration: `basic`, `intermediate` and `advanced` at built-in
size, and their large versions from `docs/make_gifs.py::large_scenarios()`.

**Seeds and length.** Seeds 0–4; full scenario `max_steps`. That gives 6 × 5 × 4 = 120 episodes.

**Primitive mode** (safety only). Masked random policy, model 2.1, both layouts; 3 built-in tiers × seeds 0–4
× 500 ticks.

**API probe.** Send a second robot to a dock already reserved, and to one already occupied. 2.1 must reject
both; 2.0's behaviour is recorded.

**Pairing.** Changing the layout changes spawn cells and path costs, so A/B vs C/D are NOT paired
counterfactuals. Only rule effects within a layout (A vs B, C vs D) share the same seeds and layout. They are
still descriptive, not causal estimates.

## 4. Gate (frozen)

All criteria are checked every tick by an external validator (`validation/model_2_1/gate.py`).

| ID | Criterion | Applies to | Pass |
|---|---|---|---|
| R0 | **2.0 preserved.** `model: "2.0"`, `inline`: per-tick state digests (robot cells, SoC, status, task states) identical to commit 797000c | all 6 scenarios × seeds 0–4, macro; 3 tiers primitive | 100% identical |
| S1 | **Vertex conflicts.** Two robots on one cell, except co-carriers of the same task | B, D, primitive 2.1 | 0 |
| S2 | **Edge conflicts.** Two robots swap cells in one tick | B, D, primitive 2.1 | 0 |
| S3 | **Dock exclusivity.** A dock reserved or occupied by more than one robot, or two robots charging on one dock | B, D, primitive 2.1, API probe | 0, and the probe rejects both |
| L1 | **No persistent no-progress.** A robot with a movement target (target ≠ its cell) that does not reach a new minimum static distance to an unchanged target for ≥ 100 consecutive ticks | B, D | 0 streaks |
| L2 | **No persistent oscillation.** ≥ 10 consecutive moves alternating between the same two cells | B, D | 0 |
| L3 | **No new stranding.** Stranded robots per episode | B vs A, D vs C | not greater, in every scenario × seed |
| T | **Tests.** Full test suite, plus new tests for M1, M2, L and R0 | — | all pass |

**Reported, no threshold** (for all of A–D):
- charger travel time (dispatch → charge start) and dock wait;
- stalled robot-ticks (stalls behind fixed units separately);
- moves per completed task;
- tasks completed, mean latency, energy, stranding.

## 5. Rules

- These thresholds are frozen and are not changed after any 2.1 result is seen.
- **If a criterion fails:**
  - the failure is reported;
  - the implementation may be revised;
  - the WHOLE gate is rerun;
  - every rerun is listed in the outcome section (no silent iterations).
- **Until all of R0, S1–S3, L1–L3 and T pass:** no RA-L coordination experiment on v2.
- Performance differences between A–D are descriptive. They are not a criterion and not a claim.
