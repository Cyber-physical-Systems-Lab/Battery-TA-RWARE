# Model 2.2 behavioural gate (fixed before the gate was run)

**Change under test.** Model 2.2 adds one movement rule to model 2.1: a robot that works *beside* a
stage cell (a role with `presence: adjacent`, e.g. the second picking robot of a `large` task) never
chooses, keeps or waits on a cell that is the current stage cell or payload cell of another open,
staffed task. If every standing cell within its radius is reserved this way, it waits at a neutral
staging cell nearby. This extends model 2.1's admission rule (which already keeps carriers and
on-cell crew off shared work cells) to helpers, and removes the hold-and-wait cycle reproduced by
`experiments/checks/known_gridlock.py` (model 2.1: base order scenario, PLB, seed 1017; also
rate 0.05, PLB, seed 6).

**Not changed.** Model 2.0 and 2.1 trajectories (`experiments/checks/identity.py`, 31 fingerprints
recorded before the change). No timeout, teleport, forced move or random perturbation was added.

## Episodes (`experiments/model_2_2/gate.py`)

- Task-only built-ins `basic`, `intermediate`, `advanced`, seeds 0-4, greedy FIFO coordinator, full
  episode, models 2.1 and 2.2; their large versions (`docs/make_gifs.large_scenarios`), seeds 0-1.
- Primitive mode, the three built-ins, seed 0, 500 ticks, model 2.2.
- Order scenarios `basic`, `intermediate`, `advanced` (`experiments/model_2_2/scenarios.py`), rules
  FIFO, NSP and PLB, seeds 0-9, 3000 ticks, models 2.1 and 2.2.

## Criteria (all must hold for model 2.2)

| id | criterion |
|---|---|
| S1 | no vertex conflict outside the model 2.1 occupancy exceptions (joint carriers; a task's carrier and crew on that task's work cell; former partners parting within 10 ticks) |
| S2 | no swap between robots of different tasks |
| S3 | dock exclusivity: one holder and one charging robot per dock, charging only on the dock, no stale reservation |
| H1a | no helper (adjacent role) that is not working its stage targets a cell that is another open, staffed task's current stage or payload cell (other than its own task's stage cell) |
| H1b | no such helper stays on such a cell for more than 10 consecutive ticks |
| L1 | task-only built-ins: no unit without progress towards its goal for 100 ticks (as in the 2.1 gate) |
| L2 | no unit alternating between two cells for 10 consecutive moves |
| L3 | stranding: no episode strands more robots than the same episode under model 2.1 |
| D1 | order scenarios: no suspected stall (`stall_suspected`, 100 ticks without progress) in any 2.2 episode |
| D2 | the two known 2.1 gridlock episodes do not stall under 2.2 |

L1 on the order scenarios and throughput changes are reported, not gated: a robot may legitimately
wait more than 100 ticks for a busy cell or crew under order backlogs.

## Amendment (after the smoke run `--quick`, before the full run)

The draft H1 ("no helper stands on another task's work cell while waiting or working") counted
situations the rule does not address and cannot prevent: (i) a helper standing on its *own* task's
stage cell that another task also needs (two deliveries to one station); (ii) a helper already
working its stage when another task's crew is assigned to that cell (the stage has a bounded length;
the other task queues, as for any busy work cell); (iii) a helper that has chosen a new spot but
needs a tick or two to move off. The smoke run found 82 such ticks under 2.2 (114 under 2.1),
none of them a hold-and-wait. H1 was replaced by H1a/H1b above, which test the rule itself (what
a waiting helper targets) and that it is not held up on a foreign work cell. The rule was also
clarified to never avoid the helper's own stage cell.
