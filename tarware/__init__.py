"""
Gymnasium registration entry point for Tarware environments.

Importing this package registers size, observation, AGV-count, and picker-count
variants for the standard charging environment, plus matching package-mix
variants using SOLO / STANDARD / HEAVY / PICKER_SOLO / LARGE package
probabilities. full_registration() adds a broader registration sweep for
custom request-queue sizes and longer episodes.
"""

import itertools

import gymnasium as gym

from tarware.spaces import observation_map
from tarware.warehouse import RewardType
from tarware.definitions import PackageType

_obs_types = list(observation_map.keys())

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
    "extralarge": (4, 7),
}

_request_queues = {
    "tiny": 20,
    "small": 20,
    "medium": 20,
    "large": 60,
    "extralarge": 100,
}

_perms = itertools.product(_sizes.keys(), _obs_types, range(1,20), range(0, 10))

_DEFAULT_PKGMIX = {
    PackageType.SOLO:        0.3,   # n_a
    PackageType.STANDARD:    0.3,   # n_st
    PackageType.HEAVY:       0.3,   # n_h
    PackageType.PICKER_SOLO: 0.1,   # n_p
    PackageType.LARGE:       0.0,   # kept registered; opt-in via override
}

# Resource-competition mix: mostly single-role tasks (SOLO + PICKER_SOLO) so the
# AGV-Picker coupling is no longer mutualistic-by-design; the only coupling left
# is contention for scarce shared resources (chargers, aisles). Used with a
# reduced charger count and higher battery drain to make a NON-mutualism
# (competition / parasitism) relationship regime emerge.
_COMPETITION_PKGMIX = {
    PackageType.SOLO:        0.5,
    PackageType.PICKER_SOLO: 0.3,
    PackageType.STANDARD:    0.1,
    PackageType.HEAVY:       0.1,
    PackageType.LARGE:       0.0,
}

for size, obs_type, num_agvs, num_pickers in _perms:
    base_kwargs = {
        "column_height": 8,
        "shelf_rows": _sizes[size][0],
        "shelf_columns": _sizes[size][1],
        "num_agvs":  num_agvs,
        "num_pickers": num_pickers,
        "request_queue_size": _request_queues[size],
        "max_inactivity_steps": 100,
        "max_steps": 5000,
        "reward_type": RewardType.INDIVIDUAL,
        "observation_type": obs_type,
    }
    # normal tasks (STANDARD-only, backward compatible)
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs=base_kwargs,
    )
    # package-heterogeneous variant: SOLO / STANDARD / HEAVY / PICKER_SOLO / LARGE mix
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-pkgmix-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs={**base_kwargs, "package_distribution": dict(_DEFAULT_PKGMIX)},
    )
    # relaxed pkgmix variant: looser LARGE convoy adjacency + longer inactivity
    # tolerance, intended for runs where the original LARGE coordination
    # cliff prevents any LARGE deliveries from bootstrapping.
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-pkgmix-relaxed-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs={
            **base_kwargs,
            "max_inactivity_steps": 500,
            "package_distribution": dict(_DEFAULT_PKGMIX),
            "large_convoy_adjacency": 2,
        },
    )
    # resource-competition variant: single-role-heavy mix + scarce chargers
    # (~1 per 3 agents) + 2x battery drain. The single-role mix removes the
    # cross-role mutualism that dominates the default env; the scarce chargers
    # under elevated drain force genuine contention -> a NON-mutualism
    # (competition/parasitism) regime. The reduced charger count changes the
    # macro-action space, so this variant is characterized with the
    # action-agnostic rule controller rather than a fixed-head trained policy.
    _n_chg_comp = max(2, (num_agvs + num_pickers) // 3)
    gym.register(
        id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-pkgmix-competition-v1",
        entry_point="tarware.warehouse:Warehouse",
        disable_env_checker=True,
        kwargs={
            **base_kwargs,
            "package_distribution": dict(_COMPETITION_PKGMIX),
            "num_chargers": _n_chg_comp,
            "battery_drain_multiplier": 4.0,
        },
    )

def full_registration():
    _perms = itertools.product(_sizes.keys(), _obs_types, _request_queues, range(1,20), range(0, 10),)
    for size, obs_type, request_queue_size, num_agvs, num_pickers in _perms:
        # normal tasks with modified column height
        gym.register(
            id=f"tarware-{size}-{num_agvs}agvs-{num_pickers}pickers-{obs_type}obs-chg-v1",
            entry_point="tarware.warehouse:Warehouse",
            disable_env_checker=True,
            kwargs={
                "column_height": 8,
                "shelf_rows": _sizes[size][0],
                "shelf_columns": _sizes[size][1],
                "num_agvs":  num_agvs,
                "num_pickers": num_pickers,
                "sensor_range": 1,
                "request_queue_size": _request_queues[request_queue_size],
                "max_inactivity_steps": None,
                "max_steps": 10000,
                "reward_type": RewardType.INDIVIDUAL,
                "observation_type": obs_type,
            },
        )
