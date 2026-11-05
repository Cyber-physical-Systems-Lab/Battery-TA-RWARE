"""Frozen profiles reproduce the RendeCoord G0 baseline tick by tick.

Reference: tests/data/legacy_v1_env_fingerprints.json, recorded in RendeCoord at tag bt-g0-baseline
(79b5a90) with sh/bt_g0_trajectories.py (env-only cases, 2000 ticks, seeds 501/502, deterministic
across repeats). These profiles are frozen: a v2 behavior change must ship as a NEW profile, never by
updating this reference.
"""
import json
from pathlib import Path

import pytest

import battery_tarware as bt
from battery_tarware.protocol_v4 import ArrivalSpec, make_tiled_warehouse_v4
from battery_tarware.testing import TrajectoryRecorder, random_target_driver

REF = json.loads((Path(__file__).parent / "data" / "legacy_v1_env_fingerprints.json").read_text())
STEPS = REF["steps"]
TILED_2X3 = dict(tile_rows=2, tile_cols=3, num_agvs=19, num_pickers=7, max_steps=STEPS, request_queue_size=40,
                 max_inactivity_steps=STEPS, replenish_requests=True)


def build(case):
    if case == "env_legacy_medium":
        return bt.make_pkgmix_warehouse(size="medium", num_agvs=6, num_pickers=3, max_steps=STEPS,
                                        max_inactivity_steps=STEPS)
    mix = dict(bt.DEFAULT_PACKAGE_MIX)
    if case == "env_tiled_2x3":
        return bt.make_tiled_warehouse(package_distribution=mix, **TILED_2X3)
    if case == "env_v4_arrivals":
        spec = ArrivalSpec(profile="constant", rate=0.25, period=2880, amplitude=0.5, mix=(0.3, 0.4, 0.3))
        return make_tiled_warehouse_v4(charger_ratio=0.5, arrivals=spec, package_distribution=mix, **TILED_2X3)
    raise KeyError(case)


@pytest.mark.parametrize("key", sorted(REF["cases"]))
def test_frozen_profile_fingerprint(key):
    case, seed = key.split("/")
    seed = int(seed)
    ref = REF["cases"][key]
    env = build(case)
    env.reset(seed=seed)
    rec = TrajectoryRecorder(env).wrap()
    deliveries = random_target_driver(env, seed, STEPS)
    assert rec.reset_digest == ref["reset_digest"], "reset state differs"
    sampled = rec.ticks[99::100]
    first = next((100 * (i + 1) for i, (a, b) in enumerate(zip(sampled, ref["tick_digests_every_100"])) if a != b), None)
    assert first is None, "trajectory diverges by tick %d" % first
    assert (len(rec.ticks), deliveries) == (ref["ticks"], ref["deliveries"])
    assert rec.hexdigest() == ref["chain"]
