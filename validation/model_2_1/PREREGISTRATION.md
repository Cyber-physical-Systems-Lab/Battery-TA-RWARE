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

## OUTCOME so far (2026-11-06): gate NOT passed after 5 full runs; stopped for decisions

**Mechanics smoke.** Before run 1, a smoke test of the validator ran on 4 episodes (basic seed 0
REF/A/B; basic_large seed 0 D). It is disclosed here, but no threshold depended on it.

**Run setup.**
- Every run is the full gate: 165 episodes, plus the REF episodes.
- REF digests come from the unchanged base commit 797000c (run 1) and are reused in later runs.
- Outputs: `validation/model_2_1/out/run{1..5}/` (report.txt, RUN_INFO).

### Runs

| Run | Commit | R0 | S1 | S2 | S3 | L1 | L2 | L3 | Revision after it |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ee7c86a | PASS 45/45 | FAIL 932 | PASS | PASS | FAIL 47 | FAIL 84 | FAIL (1 pair) | Docks are destinations only (idle robots were pushed into dead-end bays and shuttled in and out) |
| 2 | 2bba4aa | PASS | FAIL 934 | PASS | PASS | FAIL 39 | FAIL 47 | PASS | Staging only on neutral cells (loaded AGVs staged on another task's goal cell, got pushed off and came back) |
| 3 | 6132604 | PASS | FAIL 930 | PASS | PASS | FAIL 28 | FAIL 27 | PASS | Robots parked on their own stage cell route as long-lived obstacles (4-robot circular wait, ~1400 ticks, also in 2.0) |
| 4 | eda2b30 | PASS | FAIL 916 | PASS | PASS | FAIL 6 | FAIL 10 | FAIL (advanced s4 D: 2 drones stranded) | Admission for shared stage cells (hold-and-wait deadlock on the shared QC cell) |
| 5 | dd7c98f | PASS | FAIL 944 | PASS | PASS | FAIL 7 | FAIL 8 | PASS | — (stopped for decisions) |

API probe (S3): model 2.0 accepts a 2nd and 3rd robot sent to a reserved/occupied dock; 2.1 rejects both.

### Remaining failures

**S1.** Every violation is the same mechanism, verified for all 467 cases in run 1 config B:
- Two former co-carriers stay on one cell after a joint task completes (or after release), until they
  drive apart.
- The rate is identical in model 2.0 (REF). There are no other vertex conflicts.
- This follows from the joint-transport abstraction (joint carriers share one cell).
- It cannot pass S1 as registered without either redesigning joint transport, or a user-approved amendment to
  S1. Any such amendment would be flagged as made after seeing results.

**L1/L2.** These are rare residuals:

| | Model 2.0 (A) | Run 5 (B + D) |
|---|---|---|
| L1 | 33 | 7 |
| L2 | 30 | 8 |
| Longest no-progress streak | — | 160 ticks |

The residuals are dense-aisle and goal-row interactions in the large and advanced maps.

### Notes on the implementation (not the gate)

M1 and M2 as implemented differ from section 2's description in three places:
- **(a) Failure releases the dock.** A broken-down robot heading to a dock gives the dock up, as 2.0 does for a
  charging robot. Keeping it would leak a reservation, since 2.0 resumes such a robot as IDLE.
- **(b) Short-lived fixed units are waited for.** Units without speed credit, or working a stage, are not
  routed around.
- **(c) Additional 2.1 rules** were added after runs 1–4: destination-only docks, neutral staging cells,
  stationed robots as obstacles, and shared-cell admission.

### Descriptive (run 5; not a criterion)

**2.1 vs 2.0, inline layout** (paired seeds):

| Scenario | Stalled robot-ticks | Tasks | Mean latency |
|---|---|---|---|
| basic_large | 2438 → 207 | 220 → 223 | — |
| advanced | — | 293 → 280 (−4.6%) | 217 → 208 |

**Bays with 2.1 (D):** no scenario loses throughput against C.

## Runs 6–7 (2026-11-06): pickers on the shelf cell (user request), gate still NOT passed

**User observation (from the GIFs).** The picking robot did not step onto the shelf cell beside the AGV,
as it does in TA-RWARE (v1 `tarware/warehouse.py`: a picker steps onto an AGV's rack cell, one picker per
shelf location).

**Model 2.1 changes.**
- Built-in picker roles are `on_cell`.
- A task's carrier and its `on_cell` crew may share their work cell (PIBT meet-compatibility, and
  primitive shared groups).
- For tasks needing 2 pickers, one picker is `on_cell`; the second (`picker_2`) works from the next cell.
- `get_scenario(name, model="2.0")` returns the exact 2.0 definitions, so R0 still compares like with
  like.

**S1 as registered** counts this intended co-location as a violation. The validator therefore also
reports a **PROPOSED, unregistered** amendment, **S1p**. It forbids vertex sharing except:
- co-carriers of a task;
- a task's crew on that task's stage or payload cell;
- former partners parting within 10 ticks.

S1p was defined before run 6. Its partner tracking was revised after run 6 (a group that is a subset of
a group allowed on the same cell the tick before), which is disclosed here.

| Run | Commit | R0 | S1 (registered) | S1p (proposed) | S2 | S3 | L1 | L2 | L3 | Revision after it |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 3886d65 | PASS | FAIL 69811 | FAIL 2056 | PASS | PASS | FAIL 4 | FAIL 6 | PASS | One picker per shelf cell (LARGE/pallet put 2 pickers + AGV on one cell); S1p tracking revised |
| 7 | f26b250 | PASS | FAIL 42523 | PASS 0 | PASS | PASS | FAIL 5 | FAIL 6 | PASS | — (stopped for decisions) |

**Remaining L1/L2.** 11 events, against 63 in model 2.0. They are in advanced and advanced_large (and one in
intermediate_large D); the longest no-progress streak is 160 ticks.

**Descriptive (run 7, B vs A):**

| Scenario | Tasks | Mean latency |
|---|---|---|
| basic | 203 → 199 (pickers now enter the shelf cell) | — |
| basic_large | 220 → 224 | — |
| intermediate | 134.4 → 134.6 | 67.6 → 64.9 |

## Residual-event diagnosis after run 7, and the targeted revision (2026-11-06)

All 11 flagged events were traced before any code change (`validation/model_2_1/trace.py`; traces in
`out/diag7/ev*.txt`). Each trace records robot, position, target, PIBT priority, blockers, charging state,
task/stage and crew state.

| Events | Mechanism | Class |
|---|---|---|
| 5 × L1 (advanced_large s0/s1 D, s3 B ×2; intermediate_large s2 D) | A slow robot is pushed straight ahead down a corridor by a faster, higher-priority robot for 20–35 cells, away from its target, then walks back. A free side lane existed every time. | B: avoidable movement |
| 4 × L2 (advanced s4 B ×2; advanced_large s4 D ×2) | Two carriers waiting for the same goal cell repeatedly choose the same staging cell and push each other back and forth | B: avoidable oscillation |
| 2 × L2 (advanced_large s4 D) | A carrier and its own handler, both waiting for an occupied goal cell, trade the same staging cell | B: avoidable oscillation |

No event was legitimate waiting (A) or a deadlock (C).

**The one targeted revision** (commits ae30764, 98aac5b):
- **(i) Step aside.** A pushed unit prefers stepping aside over retreating straight ahead of its pusher,
  at equal distance.
- **(ii) Waiting spots.** Staging spots are kept by the robot that chose them and are never shared.

Re-running the 11 flagged episodes left one new L2 (advanced s4 B, tick 734): two idle robots recomputed
their parking target every tick, traded spots and pushed a third robot. Same mechanism as (ii), so the
revision was completed within the same iteration:
- **(iii) Parking spots.** Idle robots keep an unshared parking spot.

After that, all 11 episodes have 0 L1/L2 events (longest no-progress streak 7–49 ticks) and 0 stranding.

## AMENDMENT (post hoc, frozen 2026-11-06 before the confirmation run)

**S1 as registered remains FAILED (runs 1–7).** It is never reported as passed.

Occupancy is task-dependent compatibility, not universal cell exclusivity. Coalition membership alone does
not authorize sharing a cell: the permitted occupancy depends on the active stage and the roles.

**S1a: unintended occupancy = 0.** Two or more robots may share a cell only if:
- they are co-carriers of one task (joint transport); or
- they are a task's carrier and its `on_cell` crew, on that task's current stage cell or payload cell.
  The built-in scenarios allow one picker per shelf cell; a second picker works from the next cell.

**S1b: bounded separation.** Robots allowed to share a cell under S1a in the previous tick (or a subset of
such a group) may stay together for at most 10 ticks after their task releases them. Any longer is an S1a
violation.
- The 10-tick bound is the one accepted in the user's decision. An earlier review suggested "one movement
  opportunity"; the user's final decision accepted 10 ticks.

**Implementation.** `gate.py` key `S1p`, as at commit 98aac5b (partner tracking as revised after run 6).

**L1/L2.** Thresholds unchanged: zero.
- Any flagged event must be traced with `trace.py` and classified.
- Only class A (legitimate task/resource waiting, shown by the trace) may be excluded, and each exclusion
  is listed.
- Class B or C fails the gate.

**Confirmation run (run 8).**
- Commit 98aac5b.
- Fresh seeds 5–9: never used in any run, smoke test or diagnosis.
- All 6 scenarios × A–D, macro, plus primitive mode for the 3 tiers.
- REF episodes from base commit 797000c on seeds 5–9, for R0.
- The gate verdict uses seeds 5–9 only.
- Seeds 0–4 are re-run alongside as development seeds (descriptive, and for the advanced throughput analysis).

## CONFIRMATION RUN 8 (2026-11-06): gate PASSED under the frozen amended definitions

**Run setup.** Commit a2c4d69, fresh seeds 5–9. 210 confirmation episodes: 6 scenarios × A–D × macro,
REF on seeds 5–9, and primitive mode for the 3 tiers. Report: `out/run8/report_confirm.txt`.

| Criterion | Result |
|---|---|
| R0 (2.0 reproduced exactly) | PASS, 45/45 |
| S1 as registered | **FAIL** (40,069; intended co-location). Not reported as passed. |
| S1a/S1b (post hoc amendment, frozen before run 8) | PASS, 0 |
| S2, S3 | PASS, 0 |
| L3 | PASS |
| L1 / L2 | 1 event each. Both traced (`out/diag8/`) and classified **A, legitimate waiting**; excluded under the frozen rule, listed below |

**Excluded events:**
- **L1, advanced s7 D, robot 4, tick 4094.** A loaded AGV is on its pickup shelf cell. Its only exit, the
  single aisle cell (4,16), is held by forklift r7, which broke down at tick 3992 (repair 120 ticks, until
  4112). The robot cannot leave until the repair ends.
- **L2, advanced_large s8 D, robot 18, tick 1657.** Handler r18 waits at its standing cell beside its own
  carrier r7, which broke down at tick 1607 (repair 60 ticks). Its target never changes. The standing cell
  must be adjacent to the work cell, and it lies in the goal-row corridor. r18 steps aside 5 times, for 5
  different passing robots or carrier pairs (r0+r3, r2+r4, r17, r6+r11, r16), and returns each time. That
  is yielding to through traffic while waiting, not oscillating between targets.

**Purity fix after run 8 (commit 31c89ad).** Waiting and parking claims are now written only in the
movement phase, so `_target` is a pure query. This was found because the throughput analysis's
`_target` calls changed 2.1 trajectories.
- All 90 model-2.1 confirmation episodes (B, D, macro and primitive) were re-run on 31c89ad and are
  **byte-identical** to run 8 (`out/run8b`), so the confirmation applies to 31c89ad.
- New test: `test_target_is_a_pure_query`.

### Advanced B vs A throughput (paired, seeds 0–9, final code)

Source: `validation/model_2_1/throughput.py`; output `out/throughput_advanced_AB.txt`. Differences are
B − A with paired bootstrap 95% CIs.

| Measure | Difference [95% CI] |
|---|---|
| Tasks completed | +7.3 [−5.6, +20.2] |
| Mean latency | −16.0 [−34.9, +1.2] |
| Per task: queue wait | −13.9 [−32.3, +3.0] |
| Per task: execution | −2.1 [−3.6, −0.8] |
| Per task: crew gathering | −0.0 [−0.6, +0.4] |
| Robot-ticks waiting in place | −1220 [−1703, −828] |
| Robot-ticks moving | +434 [+267, +614] |

**Completions by type.** Demand-driven types differ by +0.2 to +1.1 per episode (order, bulky_order,
receive, cycle_count). Relocations differ by +4.4 [−10.0, +19.7].

**Reading.** The variability in "tasks completed" is dominated by `relocate`, a backlog filler stream
(one always open, taken by any free forklift; 13–76 per episode). The earlier "−4.6%" (run 5, seeds 0–4,
earlier code) was this filler noise. There is no throughput regression on demand-driven tasks.

**D − C:** tasks +15.7 [−1.9, +36.3], latency −20.2 [−37.3, −1.3].

**Open semantic detail (not a gate item).** An `on_cell` picker that arrives before its AGV waits on the
shelf cell. TA-RWARE has the picker wait just before stepping in. This was left unchanged after the
confirmation run.
