# Model 2.2 gate: outcome

Run: `python experiments/model_2_2/gate.py --jobs 30` on commit 96b7867 (criteria in `GATE.md`, fixed
before this run; the H1 amendment there was made after the smoke run and before this run).
229 episodes, 0 raised.

| id | result | detail |
|---|---|---|
| S1 vertex conflicts | PASS | 0 in 116 model 2.2 episodes |
| S2 swaps | PASS | 0 |
| S3 dock exclusivity | PASS | 0 |
| H1a helper targets a foreign work cell | PASS | 0 (model 2.1, same episodes: 778) |
| H1b helper held on a foreign work cell > 10 ticks | PASS | 0 (model 2.1: 2) |
| L1 no progress >= 100 ticks (task-only) | PASS | 0 in 21 episodes; longest streak 16 ticks |
| **L2 oscillation (10 alternating moves)** | **FAIL** | **7 events** (5 of them identical, same robot and tick, under model 2.1) |
| L3 stranding vs 2.1 | PASS | no episode strands more robots (0 stranded in all) |
| D1 suspected stalls, order scenarios | PASS | 0 in 90 episodes |
| D2 known 2.1 gridlocks | PASS | 2.1: 4 + 4 suspected stalls; 2.2: 0 + 0 |

**L2 is reported as failed.** All 7 events are in order scenarios (none in task-only episodes; the
model 2.1 gate never ran order scenarios, which is why the 5 shared events were not seen before).
Each was traced (`gate.json` holds the tick and robot):

| model | episode | robot | 10 alternating moves spread over | statuses during them |
|---|---|---|---|---|
| 2.1 and 2.2 | basic FIFO s5 | picking_robot_2 | 53 ticks | idle and task, 3 targets |
| 2.1 and 2.2 | basic FIFO s7 | picking_robot_1 | 94 ticks | idle and task, 4 targets |
| 2.1 and 2.2 | intermediate NSP s2 | picking_robot_0 | 251 ticks | idle and task, 4 targets |
| 2.1 and 2.2 | intermediate NSP s7 | picking_robot_2 | 93 ticks | idle and task, 4 targets |
| 2.1 and 2.2 | intermediate NSP s8 | picking_robot_1 | 183 ticks | task, 3 targets (several tasks at one shelf) |
| 2.2 only | basic PLB s4 | picking_robot_1 | 281 ticks | idle and task, 4 targets |
| 2.2 only | advanced NSP s7 | mobile_manipulator_0 | 80 ticks | idle and task, 4 targets |

None is a back-and-forth towards one target: in every case the robot alternates between a work
position beside a shelf and a cell next to it across several tasks and idle spells (served the same
shelf again, or stepped aside for passing traffic and came back), over 53-281 ticks. The L2
definition (inherited from the model 2.1 gate) counts alternating moves without a time window and
across target changes, so it flags this pattern. It is not reclassified as a pass here; a revised
definition (alternation towards an unchanged target within a time window) would have to be fixed
before a new gate run.

**Descriptive (not gated).** Order scenarios, 10 seeds x FIFO/NSP/PLB, 3000 ticks, mean orders
completed: basic 238.0 (2.1) vs 237.8 (2.2); intermediate 166.9 vs 167.0; advanced 98.1 vs 98.3; no
suspected stall under either model in these 90+90 episodes (the 2.1 gridlock is rare: 2 of 1480
runs in the first study). Task-only built-ins: 4835 (2.1) vs 4807 (2.2) tasks completed over 21
episodes; longest no-progress streak 16 ticks under both.
