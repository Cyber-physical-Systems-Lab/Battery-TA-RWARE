from typing import Dict, Optional

from battery_tarware.definitions import PackageType
from battery_tarware.warehouse import Warehouse

SIZES = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
    "extralarge": (4, 7),
    "huge": (5, 9),
    "xxlarge": (6, 11),
}

REQUEST_QUEUES = {
    "tiny": 20,
    "small": 20,
    "medium": 20,
    "large": 60,
    "extralarge": 100,
    "huge": 200,
    "xxlarge": 500,
}

DEFAULT_PACKAGE_MIX = {
    PackageType.SOLO: 0.3,
    PackageType.STANDARD: 0.4,
    PackageType.LARGE: 0.3,
}


def make_warehouse(
    *,
    size: str = "tiny",
    num_agvs: int = 2,
    num_pickers: int = 1,
    column_height: int = 8,
    request_queue_size: Optional[int] = None,
    max_inactivity_steps: Optional[int] = 100,
    max_steps: Optional[int] = 5000,
    package_distribution: Optional[Dict] = None,
    charger_rows: int = 1,
    replenish_requests: bool = False,
    bottom_charger_rows: int = 0,
    charger_spread: bool = False,
    charger_all_rows: bool = False,
    initial_battery_profile: str = "legacy",
    initial_battery_low: float = 20.0,
    initial_battery_high: float = 100.0,
    battery_drain: bool = True,
    picker_energy_factor: float = 1.25,
) -> Warehouse:
    if size not in SIZES:
        raise ValueError(f"Unknown size '{size}'. Expected one of {list(SIZES)}")

    shelf_rows, shelf_columns = SIZES[size]
    queue_size = request_queue_size if request_queue_size is not None else REQUEST_QUEUES[size]

    return Warehouse(
        shelf_columns=shelf_columns,
        column_height=column_height,
        shelf_rows=shelf_rows,
        num_agvs=num_agvs,
        num_pickers=num_pickers,
        request_queue_size=queue_size,
        max_inactivity_steps=max_inactivity_steps,
        max_steps=max_steps,
        package_distribution=package_distribution,
        charger_rows=charger_rows,
        replenish_requests=replenish_requests,
        bottom_charger_rows=bottom_charger_rows,
        charger_spread=charger_spread,
        charger_all_rows=charger_all_rows,
        initial_battery_profile=initial_battery_profile,
        initial_battery_low=initial_battery_low,
        initial_battery_high=initial_battery_high,
        battery_drain=battery_drain,
        picker_energy_factor=picker_energy_factor,
    )


def make_tiled_warehouse(
    *,
    tile_rows: int,
    tile_cols: int,
    num_agvs: int,
    num_pickers: int,
    column_height: int = 16,
    rack_width: int = 1,
    lane_width: int = 2,
    request_queue_size: Optional[int] = None,
    max_inactivity_steps: Optional[int] = 100,
    max_steps: Optional[int] = 5000,
    package_distribution: Optional[Dict] = None,
    replenish_requests: bool = True,
    initial_battery_profile: str = "legacy",
    initial_battery_low: float = 20.0,
    initial_battery_high: float = 100.0,
    battery_drain: bool = True,
    picker_energy_factor: float = 1.25,
) -> Warehouse:
    """Tiled, traffic-stable warehouse (one-way lanes, off-lane docks, goals in every rack block)."""
    return Warehouse(
        shelf_columns=tile_cols,
        column_height=column_height,
        shelf_rows=tile_rows,
        num_agvs=num_agvs,
        num_pickers=num_pickers,
        request_queue_size=request_queue_size if request_queue_size is not None else 4 * tile_rows * tile_cols,
        max_inactivity_steps=max_inactivity_steps,
        max_steps=max_steps,
        package_distribution=package_distribution if package_distribution is not None else dict(DEFAULT_PACKAGE_MIX),
        layout="tiled",
        rack_width=rack_width,
        lane_width=lane_width,
        replenish_requests=replenish_requests,
        initial_battery_profile=initial_battery_profile,
        initial_battery_low=initial_battery_low,
        initial_battery_high=initial_battery_high,
        battery_drain=battery_drain,
        picker_energy_factor=picker_energy_factor,
    )


def make_pkgmix_warehouse(**kwargs) -> Warehouse:
    return make_warehouse(package_distribution=dict(DEFAULT_PACKAGE_MIX), **kwargs)


__version__ = "0.2.0.dev0"

__all__ = ["Warehouse", "make_warehouse", "make_pkgmix_warehouse", "make_tiled_warehouse", "SIZES", "REQUEST_QUEUES"]
