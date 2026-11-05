# Changelog

## 2.0.0.dev0 (branch `v2-dev`, unreleased)

- Add `battery_tarware`, the v2 simulation kernel, imported from RendeCoord at tag `bt-g0-baseline`
  (79b5a90) with the package renamed from `tarware`; simulation logic unchanged
  (see `docs/v2/PROVENANCE.md`).
- Add `battery_tarware.interfaces`: read-only, value-typed `StateView` / `LayoutView` (schema 1).
- Add `battery_tarware.tasks`: `TaskSpec`, `TaskGraph` (dependency DAG with cycle, unknown-dependency
  and fleet-feasibility checks; deterministic order), task lifecycle state machine. Not wired into the
  kernel; frozen profiles do not import it.
- Add `battery_tarware.testing`: per-tick trajectory fingerprints (same definition as RendeCoord G0).
- Tests: kernel independence (no imports of `tarware`/RendeCoord; runs in isolation), frozen-profile
  fingerprints (6 RendeCoord G0 env-only runs reproduced tick by tick), state views, task model.
- Packaging: distribution renamed `battery-tarware`, version 2.0.0.dev0; fixed `readme` (pointed at
  a non-existent README.txt); `six`/`pyglet` moved to the `render` extra; both `battery_tarware` and
  the unchanged legacy `tarware` are packaged.
- The legacy `tarware` v1.1.0 Gymnasium environment is unchanged (tag `legacy-v1.1.0`).
