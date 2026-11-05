# Provenance of the v2 kernel

`battery_tarware/` (except `interfaces/`, `tasks/`, `testing/`, which are new) is the simulator
developed in RendeCoord (`tarware/` there), imported at RendeCoord tag **`bt-g0-baseline`**
(`79b5a904259a8c3b42f3c7dcfe32460c81955d44`, 2026-11-05) after a package rename
`tarware` → `battery_tarware` with import paths rewritten. No simulation logic was changed.
`telemetry.py` was left in RendeCoord (it is the planner's episode-trace format).

It descends from the public TA-RWARE v1.1.0 in this repository (`tarware/`, tag `legacy-v1.1.0`) but
has diverged substantially. Main differences:

| | `tarware` v1.1.0 (public) | `battery_tarware` (v2 kernel) |
|---|---|---|
| API | Gymnasium env: `step -> (obs, rewards, terminated, truncated, info)`, observation spaces | plain kernel: `step(macro_actions) -> (done, info)`, `reset -> {}`; Gymnasium is an adapter to come |
| Battery | basic | OMRON LD-250-calibrated energy (5 s/step), charge rate, loaded/unloaded consumption, picker lift energy |
| Tasks | single shelf type | SOLO / STANDARD / LARGE packages with AGV + picker crew requirements and weights |
| Layouts | parametric | parametric + tiled (one-way lanes, off-lane docks, goals per rack block) |
| Demand | replenished queue | replenished queue + seeded external arrivals (`protocol_v4`) |
| Chargers | fixed | configurable rows / spread; functional-charger ratio, layout and power split (`protocol_v4`) |

## Verification

1. **G0 baseline in RendeCoord** (`docs/battery_tarware/G0_*` there): 30 configurations
   (env-only and full planner runs; Python and C++ coordinator; historical and rebuilt C++ binary),
   2000 ticks, seeds 501/502, each run twice. All deterministic; historical and rebuilt binaries
   identical; protocol v4 at defaults equals v2 tick for tick.
2. **Rename gate in RendeCoord**: the renamed tree reproduced all 60 G0 runs bit-identically.
3. **This repository**: `tests/test_frozen_profile_fingerprints.py` reproduces the six env-only G0
   fingerprints (legacy medium, tiled 2x3, v4 arrivals with charger ratio 0.5; both seeds) tick by
   tick with the kernel imported in isolation.

Fingerprint definition: `battery_tarware/testing/fingerprint.py` (identical to RendeCoord
`sh/bt_g0_trajectories.py`).
