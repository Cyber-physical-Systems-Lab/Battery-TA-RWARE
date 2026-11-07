# Second order-scheduling evaluation: model 2.2 and the expanded warehouses

This is a **new evaluation**, separate from the first study (`experiments/orders/`, model 2.1, basic
warehouse family, seeds 0-9 and 1000-1019), whose protocol, hypotheses and results are unchanged.
It has its own fresh seeds (exploratory 2000-2009, confirmatory 3000-3019) and its own protocol
(`protocol_v2.json`, committed in 6e2a2d4 before the confirmatory run; git blob 16e2384).

Model 2.2 is opt-in (`model: "2.2"`); the default stays 2.1. Everything below names the model it ran.

## Reproduce

```bash
python experiments/checks/identity.py           # model 2.0 / 2.1 trajectories unchanged (31 fingerprints)
python experiments/checks/known_gridlock.py     # 2.1 gridlock reproduced, 2.2 free of it
python experiments/model_2_2/gate.py --jobs 30  # model 2.2 behavioural gate (GATE.md, RESULT.md)
python experiments/orders_v2/exploratory.py --jobs 30     # seeds 2000-2009, 2900 runs (~1.5 h on 15 cores)
python experiments/orders_v2/confirmatory.py --jobs 30    # protocol_v2.json, seeds 3000-3019, 380 runs
python experiments/orders_v2/stress.py --ticks 18000      # 36 long runs with system checks
python experiments/orders_v2/runtime.py --jobs 8          # scheduler runtime / deadline (300 runs)
python experiments/checks/dashboard_browser.py experiments/orders_v2/results/stress/trace_complex_dependencies_2.2_nsp_0.json
```

Outputs go to `experiments/orders_v2/results/` (git-ignored). Workloads: `workloads.py`.

## 1. Model 2.2 vs 2.1 (the simulator correction)

**Gate** (`experiments/model_2_2/RESULT.md`, 229 episodes): no vertex, swap or dock-exclusivity
violation; no helper targeting another task's work cell (model 2.1: 778 such targets in the same
episodes); no no-progress streak of 100 ticks in task-only episodes; no extra stranding; no suspected
stall in 90 order episodes; both known 2.1 gridlocks absent. **Failed: L2** (oscillation, 7 events,
5 of them identical under 2.1): a picker alternating between a work position and the next cell
across several tasks over 53-281 ticks; reported as failed, not reclassified.

**Matched workloads, exploratory** (the first study's 10 configurations x 10 rules x seeds
2000-2009 under both models, 2000 runs): most (configuration, rule, seed) triples are identical;
mean throughput differences lie within +-0.41 orders/1000 ticks except CR at rate 0.13 (+2.26),
where model 2.1 stalled on one seed (the only suspected stall in the 1000 model 2.1 runs; 0 in the
1000 model 2.2 runs). No consistent shift in lead time, energy or coalition waiting.

**Confirmatory** (C12, C13 below): on the first study's base, NSP and PLB throughput under 2.2 are
equivalent to 2.1 within +-1 order/1000 ticks (differences -0.06 and -0.02). The NSP advantage over
FIFO (C1, +7.7) and the PLB advantage (C2, +1.9) hold under 2.2; under 2.1 the first study could
not confirm PLB > FIFO because of one gridlocked seed.

## 2. Confirmatory results (seeds 3000-3019, Bonferroni CI level 0.99615)

| id | hypothesis | reference mean | mean | diff [CI] | seeds with diff > 0 | supported |
|---|---|---|---|---|---|---|
| C1 | model 2.2, first-study base: NSP completes more orders per 1000 ticks than FIFO (replication of H1) | 78.54 | 86.24 | +7.7 [+5.15, +10.1] | 20/20 | yes |
| C2 | model 2.2, first-study base: PLB completes more orders than FIFO (the H3 question without the 2.1 gridlock) | 78.54 | 80.44 | +1.91 [+0.226, +3.53] | 15/20 | yes |
| C3 | basic warehouse, standard order profile (compound and sequenced orders): NSP > FIFO throughput | 78.39 | 83.87 | +5.48 [+3.26, +8.22] | 19/20 | yes |
| C4 | intermediate warehouse (heterogeneous fleet, role alternatives, joint transport): NSP > FIFO throughput | 57.74 | 58.96 | +1.22 [+0.426, +2.19] | 14/20 | yes |
| C5 | intermediate warehouse: NSP has a shorter mean order lead time than FIFO | 134.2 | 106.4 | -27.8 [-43.4, -13] | 4/20 | yes |
| C6 | advanced warehouse (3-type crews with a QC stage, forklifts, breakdowns): NSP > FIFO throughput | 32.72 | 33.69 | +0.963 [+0.037, +1.89] | 13/20 | yes |
| C7 | advanced warehouse: NSP has a shorter mean order lead time than FIFO | 187.9 | 156.5 | -31.4 [-50.3, -16.2] | 0/20 | yes |
| C8 | intermediate warehouse, large dependency graphs (2-4 lines, 70% chained): NSP > FIFO throughput | 23.46 | 24.2 | +0.741 [+0.111, +1.57] | 12/20 | yes |
| C9 | intermediate warehouse without role alternatives: NSP > FIFO throughput | 57.31 | 57.7 | +0.389 [-0.907, +1.81] | 10/20 | no |
| C10 | advanced warehouse, large dependency graphs: NSP and FIFO throughput equivalent within +-1.0 orders/1000 ticks (the advantage does not carry over) | 12.7 | 13.24 | +0.537 [-0.037, +1.2] | 9/20 | no |
| C11 | intermediate warehouse, tight batteries and half the docks: NSP and FIFO throughput equivalent within +-3.0 orders/1000 ticks | 55.54 | 56.83 | +1.3 [+0.0185, +2.5] | 14/20 | yes |
| C12 | first-study base, NSP: model 2.2 and 2.1 throughput equivalent within +-1.0 orders/1000 ticks (the correction does not change behaviour outside the gridlock) | 86.3 | 86.24 | -0.0556 [-0.185, +0] | 0/20 | yes |
| C13 | first-study base, PLB: model 2.2 and 2.1 throughput equivalent within +-1.0 orders/1000 ticks | 80.46 | 80.44 | -0.0185 [-0.481, +0.403] | 2/20 | yes |

**11 of 13 supported.** In words:
- NSP's throughput and lead-time advantage over FIFO **generalizes** to the intermediate and advanced
  warehouses and to large dependency graphs in the intermediate warehouse (C3-C8), but it is much
  **smaller** there than in the basic family: +1.2 (intermediate, 2.1%) and +1.0 (advanced, 2.9%)
  orders/1000 ticks against +7.7 (9.8%) in the first study's base. Lead-time gains remain large
  (-21% and -17%).
- **Not supported, C9:** without role alternatives in the intermediate warehouse, NSP's throughput
  gain is not confirmed (+0.39, CI [-0.91, +1.81]); lead time was shorter in the exploratory pass.
- **Not supported, C10:** for the advanced warehouse with large dependency graphs, equivalence of NSP
  and FIFO within +-1 could not be shown (CI [-0.04, +1.20] exceeds the margin): inconclusive, not
  evidence that the advantage vanishes or persists.
- C11: under tight batteries in the intermediate warehouse NSP and FIFO are equivalent within the
  pre-set +-3 margin, although the CI ([+0.02, +2.50]) also excludes 0, i.e. a small positive
  difference.
- C2: with model 2.2, PLB > FIFO is supported (+1.9, 15 of 20 seeds), the question H3 of the first
  study could not answer under 2.1. This is a new evaluation on new seeds; it does not revise H3.

Suspected stalls in the confirmatory runs: 3 events in 2 runs (advanced/standard, seed 3003, FIFO and
NSP), all resolved after 115-126 ticks while a forklift was being repaired (repair time 120 ticks >
the 100-tick stall threshold): detector false positives, not deadlocks.

## 3. Exploratory findings (seeds 2000-2009; no hypothesis fixed in advance)

Full tables: `results/exploratory/summary.md`. Selected:
- Rules matter where capacity binds: in basic with the first study's simple profile, every rule is
  within +-0.6 orders/1000 ticks of FIFO (demand-limited at 0.09 orders/tick with these orders).
- With compound and sequenced orders (standard profile) the basic warehouse is capacity-bound again:
  NSP +5.2, PLB +2.6; LWF and CR lose 3-4 orders/1000 ticks.
- Pair rules (NSP, PLB, ZLP_2D) and SPT shorten lead time in every warehouse; WSP_eq and SPT trade
  on-time rate for it in the intermediate warehouse.
- **Advanced warehouse with tight batteries (excluded from the protocol):** 22 suspected stalls in
  100 runs. Every unresolved one occurred after at least one inventory drone (36 Wh at 12 % capacity,
  60 W idle draw) had stranded; stranded robots are permanent obstacles and remove the only
  inspection capability, so crews holding AGVs wait indefinitely. A modelling limitation of the
  stress level (no rescue, no task abandonment by default), not a movement deadlock.
- Three suspected stalls under normal batteries (advanced) resolved after 110-133 ticks during
  forklift repairs (120 ticks): the 100-tick stall threshold, validated on task-only traces only, is
  too short for the advanced warehouse's breakdowns and slow forklifts (auto-retry is off by default,
  so this affects diagnostics only).

## 4. Long-horizon stress runs (18,000 ticks = 25 h simulated)

36 runs (6 stress cases x FIFO / NSP / PLB x models 2.1 and 2.2, seed 0; 16 runs in parallel).
Checks: inventory invariants every tick, inventory rebuilt from the event log equals the final
inventory, package conservation, every lifecycle transition allowed and consistent with the log,
completed orders have only completed tasks, manager counts equal log counts. **All 36 pass, 0
exceptions.**

| case | model | rule | robots | arrived | completed | unfinished (max age) | backlog slope | stalls | starving (max wait) | stranded | checks | peak MB | trace MB | dashboard MB | P99 ms | viol. % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high_demand | 2.1 | fifo | 9 | 2352 | 1409 | 475 (7600) | +26.5 | 0 | 772 (3071) | 0 | PASS | 518 | 53.9 | 54.0 | 8.7 | 0.08 |
| high_demand | 2.1 | nsp | 9 | 2352 | 1637 | 365 (12590) | +20.0 | 0 | 890 (11218) | 0 | PASS | 537 | 55.2 | 55.2 | 9.9 | 0.08 |
| high_demand | 2.1 | plb | 9 | 2352 | 1695 | 312 (17704) | +17.4 | 0 | 722 (10274) | 0 | PASS | 539 | 55.3 | 55.3 | 38.2 | 6.56 |
| high_demand | 2.2 | fifo | 9 | 2352 | 1409 | 475 (7600) | +26.5 | 0 | 772 (3071) | 0 | PASS | 518 | 53.9 | 54.0 | 8.7 | 0.07 |
| high_demand | 2.2 | nsp | 9 | 2352 | 1637 | 365 (12590) | +20.0 | 0 | 890 (11218) | 0 | PASS | 537 | 55.2 | 55.2 | 9.8 | 0.08 |
| high_demand | 2.2 | plb | 9 | 2352 | 1694 | 315 (17704) | +17.1 | 0 | 714 (15467) | 0 | PASS | 544 | 56.1 | 56.1 | 39.5 | 6.47 |
| battery_scarcity | 2.1 | fifo | 9 | 1619 | 1280 | 162 (3461) | +8.1 | 0 | 674 (2909) | 0 | PASS | 486 | 50.2 | 50.3 | 7.7 | 0.07 |
| battery_scarcity | 2.1 | nsp | 9 | 1619 | 1435 | 70 (17516) | +2.9 | 0 | 706 (9370) | 0 | PASS | 476 | 47.6 | 47.7 | 8.7 | 0.07 |
| battery_scarcity | 2.1 | plb | 9 | 1619 | 1502 | 24 (17545) | +0.6 | 0 | 642 (10810) | 0 | PASS | 486 | 49.5 | 49.5 | 33.6 | 5.01 |
| battery_scarcity | 2.2 | fifo | 9 | 1619 | 1283 | 157 (3280) | +7.9 | 0 | 695 (2861) | 0 | PASS | 487 | 50.2 | 50.3 | 7.3 | 0.07 |
| battery_scarcity | 2.2 | nsp | 9 | 1619 | 1445 | 64 (17516) | +2.9 | 0 | 714 (8435) | 0 | PASS | 471 | 47.6 | 47.6 | 8.8 | 0.07 |
| battery_scarcity | 2.2 | plb | 9 | 1619 | 1502 | 24 (17545) | +0.6 | 0 | 642 (10810) | 0 | PASS | 488 | 49.5 | 49.5 | 33.9 | 3.84 |
| charging_congestion | 2.1 | fifo | 13 | 1090 | 998 | 57 (2853) | +3.4 | 0 | 572 (3844) | 0 | PASS | 462 | 47.1 | 47.2 | 19.0 | 0.78 |
| charging_congestion | 2.1 | nsp | 13 | 1090 | 1066 | 3 (99) | +0.9 | 0 | 512 (7621) | 0 | PASS | 370 | 33.5 | 33.6 | 13.1 | 0.17 |
| charging_congestion | 2.1 | plb | 13 | 1090 | 1038 | 29 (16321) | +2.0 | 0 | 614 (14596) | 0 | PASS | 434 | 42.3 | 42.4 | 33.7 | 2.27 |
| charging_congestion | 2.2 | fifo | 13 | 1090 | 1018 | 44 (2158) | +2.3 | 0 | 649 (3579) | 0 | PASS | 446 | 44.4 | 44.5 | 17.3 | 0.74 |
| charging_congestion | 2.2 | nsp | 13 | 1090 | 1023 | 39 (12135) | +2.8 | 0 | 520 (8023) | 0 | PASS | 402 | 38.1 | 38.2 | 19.7 | 0.90 |
| charging_congestion | 2.2 | plb | 13 | 1090 | 1025 | 40 (17416) | +2.3 | 0 | 576 (11557) | 0 | PASS | 438 | 43.4 | 43.4 | 35.1 | 2.39 |
| dense_traffic | 2.1 | fifo | 18 | 2352 | 2207 | 12 (67) | -0.3 | 0 | 1129 (348) | 0 | PASS | 418 | 34.1 | 34.1 | 5.2 | 0.02 |
| dense_traffic | 2.1 | nsp | 18 | 2352 | 2209 | 10 (42) | -0.1 | 0 | 533 (359) | 0 | PASS | 382 | 29.4 | 29.4 | 3.0 | 0.03 |
| dense_traffic | 2.1 | plb | 18 | 2352 | 2210 | 11 (67) | -0.1 | 0 | 683 (1500) | 0 | PASS | 389 | 30.5 | 30.6 | 7.2 | 0.06 |
| dense_traffic | 2.2 | fifo | 18 | 2352 | 2208 | 11 (67) | -0.4 | 0 | 1110 (348) | 0 | PASS | 416 | 34.1 | 34.1 | 5.6 | 0.03 |
| dense_traffic | 2.2 | nsp | 18 | 2352 | 2209 | 10 (42) | -0.0 | 0 | 558 (371) | 0 | PASS | 386 | 29.6 | 29.7 | 3.4 | 0.03 |
| dense_traffic | 2.2 | plb | 18 | 2352 | 2210 | 11 (67) | -0.1 | 0 | 683 (1500) | 0 | PASS | 390 | 30.5 | 30.6 | 7.4 | 0.04 |
| complex_dependencies | 2.1 | fifo | 13 | 234 | 148 | 66 (7876) | +3.6 | 0 | 289 (5293) | 0 | PASS | 380 | 38.3 | 38.4 | 5.6 | 0.06 |
| complex_dependencies | 2.1 | nsp | 13 | 234 | 157 | 60 (11878) | +3.2 | 0 | 330 (5314) | 0 | PASS | 338 | 32.4 | 32.4 | 5.6 | 0.04 |
| complex_dependencies | 2.1 | plb | 13 | 234 | 196 | 31 (15332) | +1.3 | 1 | 326 (6230) | 0 | PASS | 298 | 26.1 | 26.2 | 5.8 | 0.04 |
| complex_dependencies | 2.2 | fifo | 13 | 234 | 142 | 69 (8439) | +3.8 | 0 | 275 (5656) | 0 | PASS | 391 | 40.0 | 40.1 | 6.2 | 0.05 |
| complex_dependencies | 2.2 | nsp | 13 | 234 | 179 | 44 (7862) | +2.6 | 0 | 355 (5195) | 0 | PASS | 335 | 31.6 | 31.6 | 5.0 | 0.04 |
| complex_dependencies | 2.2 | plb | 13 | 234 | 192 | 34 (15332) | +1.5 | 0 | 351 (6388) | 0 | PASS | 306 | 27.3 | 27.4 | 11.5 | 0.05 |
| sustained_backlog | 2.1 | fifo | 13 | 1617 | 1226 | 242 (10603) | +11.5 | 0 | 790 (6363) | 0 | PASS | 686 | 77.4 | 77.5 | 21.3 | 1.53 |
| sustained_backlog | 2.1 | nsp | 13 | 1617 | 1133 | 269 (17743) | +14.8 | 0 | 730 (5985) | 0 | PASS | 638 | 71.0 | 71.1 | 29.1 | 15.15 |
| sustained_backlog | 2.1 | plb | 13 | 1617 | 1244 | 239 (17627) | +13.5 | 0 | 963 (14645) | 0 | PASS | 676 | 75.2 | 75.3 | 86.3 | 19.10 |
| sustained_backlog | 2.2 | fifo | 13 | 1617 | 1209 | 240 (13119) | +11.1 | 0 | 862 (6232) | 0 | PASS | 687 | 77.6 | 77.7 | 24.4 | 3.63 |
| sustained_backlog | 2.2 | nsp | 13 | 1617 | 1163 | 265 (17743) | +14.4 | 0 | 707 (7674) | 0 | PASS | 634 | 70.8 | 70.9 | 26.7 | 4.28 |
| sustained_backlog | 2.2 | plb | 13 | 1617 | 1138 | 291 (17397) | +16.9 | 0 | 858 (11572) | 0 | PASS | 701 | 79.4 | 79.5 | 54.9 | 9.16 |

- **Deadlock:** one suspected stall in 36 runs (model 2.1, complex dependencies, PLB); none under 2.2.
- **Starvation:** under sustained overload, pair rules leave some orders unserved far longer than FIFO
  (high demand: oldest open order 12,590 ticks under NSP and 17,704 under PLB vs 7,600 under FIFO;
  starving tasks counted as waiting > 5x the median assignment latency). Pair rules choose by crew
  proximity and have no ageing term.
- **Incomplete orders:** backlog grows (positive slope) in every overloaded case, as designed;
  dense traffic (18 robots on the basic map) keeps up (slope about 0).
- **Memory and files:** peak RSS 0.3-0.7 GB per run (simulation, recording, metrics, dashboard);
  trace JSON and dashboard 26-80 MB for 18,000 ticks. The 18,000-tick dashboard (31 MB) passed the
  automated browser checks (30 sampled ticks, playback, selection, filters, exports, deep links).
- Scheduler P99 and violation rates here are measured with 16 runs in parallel plus the exploratory
  pass on the same machine; use section 5 for runtime.

## 5. Scheduler runtime and deadline compliance (model 2.2)

i9-13900 (32 cores), 8 runs in parallel, 1500 ticks, seeds 0-2, deadline 20 ms (a reporting target:
the scheduler's algorithmic budget is count-based and is never cut short). Basic-family scale cases
(S/M/L/XL: 9/18/36/72 robots, order rate scaled with the fleet); selected rules (all ten in
`results/runtime/summary.md`):

| case | robots | rule | mean | P50 | P95 | P99 | max | ticks > 20 ms |
|---|---|---|---|---|---|---|---|---|
| S | 9 | fifo | 0.48 | 0.41 | 0.95 | 1.23 | 155 | 0.02% (1) |
| S | 9 | nsp | 0.33 | 0.24 | 0.84 | 1.22 | 15 | 0.00% (0) |
| S | 9 | plb | 0.59 | 0.42 | 1.88 | 3.72 | 18 | 0.00% (0) |
| S | 9 | wsp | 0.52 | 0.48 | 1.06 | 1.26 | 16 | 0.00% (0) |
| S | 9 | zlp_2d | 0.44 | 0.40 | 0.96 | 1.37 | 12 | 0.00% (0) |
| M | 18 | fifo | 2.45 | 2.52 | 4.16 | 4.85 | 30 | 0.07% (3) |
| M | 18 | nsp | 1.64 | 1.50 | 3.07 | 4.54 | 31 | 0.09% (4) |
| M | 18 | plb | 3.77 | 2.45 | 14.79 | 21.88 | 41 | 2.24% (101) |
| M | 18 | wsp | 2.43 | 2.55 | 3.92 | 4.40 | 33 | 0.13% (6) |
| M | 18 | zlp_2d | 2.15 | 2.18 | 3.63 | 4.39 | 34 | 0.09% (4) |
| L | 36 | fifo | 9.09 | 9.94 | 14.35 | 16.37 | 53 | 0.62% (28) |
| L | 36 | nsp | 7.10 | 7.16 | 12.16 | 18.03 | 80 | 1.00% (45) |
| L | 36 | plb | 21.07 | 11.15 | 72.04 | 90.04 | 165 | 30.53% (1374) |
| L | 36 | wsp | 9.30 | 10.00 | 14.03 | 14.75 | 205 | 0.58% (26) |
| L | 36 | zlp_2d | 8.57 | 8.75 | 13.93 | 15.68 | 73 | 0.62% (28) |
| XL | 72 | fifo | 32.24 | 31.97 | 54.35 | 57.44 | 132 | 65.87% (2964) |
| XL | 72 | nsp | 36.16 | 38.29 | 61.95 | 93.26 | 172 | 71.47% (3216) |
| XL | 72 | plb | 131.17 | 60.01 | 352.94 | 406.05 | 546 | 86.98% (3914) |
| XL | 72 | wsp | 30.87 | 32.77 | 49.69 | 51.29 | 168 | 66.62% (2998) |
| XL | 72 | zlp_2d | 31.87 | 30.82 | 55.90 | 58.68 | 211 | 64.29% (2893) |

- 20 ms is met at S and M except for PLB at M (2.2% of ticks) and for isolated spikes (max up to
  185 ms at S: single ticks; P99 <= 4.7 ms for every rule and seed). At L, 0.3-1.5% of ticks violate it (PLB 31%).
  **At XL (72 robots) every rule misses it on 61-88% of ticks** (P99 49-406 ms).
- Runtime grows with the number of eligible tasks, not only with candidate coalitions: at XL even
  ticks without a feasible pair take ~70 ms (P99), because every eligible task (about 200 under
  backlog) is checked and its rejection reason classified. PLB's path-conflict term is the most
  expensive.
- Intermediate and advanced warehouses (13 robots; simple / standard / dag profiles): P99 <= 3.3 ms,
  no violation; dependency complexity lowers the number of eligible tasks and hence the cost.

## Limitations

- One seed per stress case; runtime is machine- and load-dependent wall-clock time.
- The stall threshold (100 ticks) produces false positives where repairs take 120 ticks.
- Stranded robots are never rescued and tasks waiting for a lost capability are not abandoned unless
  automatic retries are enabled; the advanced tight-battery case was therefore excluded.
- Equivalence margins (C10, C11) were set from the exploratory FIFO means; C10 was inconclusive.
- Advanced-warehouse order scenarios let AGVs and forklifts drive through the receiving and QC
  areas (needed to reach the left rack column); the built-in advanced tier is unchanged.
