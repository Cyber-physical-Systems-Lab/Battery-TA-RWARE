"""Model 2.1: dock reservation and admission, routing around long-lived obstacles, dock bays, and
model 2.0 left unchanged when selected."""
import pytest

from battery_tarware import ScenarioConfig, Simulator
from battery_tarware.config import ConfigError
from battery_tarware.layout import Layout, dock_bays
from battery_tarware.robots import RobotStatus
from battery_tarware.traffic import PIBT, Unit
from conftest import place, run, tiny_dict


def make(**over):
    return Simulator(ScenarioConfig.from_dict(tiny_dict(**over)))


# ───────────────────────────── M1: charger admission and reservation

@pytest.mark.parametrize("model", ["2.0", "2.1"])
def test_second_robot_to_reserved_dock(model):
    sim = make(model=model)
    sim.reset(seed=0)
    assert sim.send_to_charger(0, 0)[0]
    ok, why = sim.send_to_charger(1, 0)
    if model == "2.1":
        assert not ok and "reserved" in why
        assert sim.chargers[0].reserved_by == 0
        assert sim.charger_state(sim.chargers[0]) == "reserved"
        assert sim.macro_action_mask(sim.robots[1])[1 + sim.K + 0] == 0
    else:
        assert ok                                   # model 2.0: no admission check (documented gap)


def test_occupied_dock_rejected_and_released_when_full():
    sim = make()
    sim.reset(seed=0)
    c = sim.chargers[0]
    for o in sim.robots:
        if o.cell == c.cell:
            place(sim, o.id, (1, 3))
    place(sim, 0, c.cell)
    sim.robots[0].energy_wh = sim.robots[0].spec.energy.capacity_wh - 1.0
    assert sim.send_to_charger(0, c.id)[0]
    run(sim, 1)
    assert sim.robots[0].charging and sim.charger_state(c) == "occupied"
    assert not sim.send_to_charger(1, c.id)[0]
    run(sim, 3)                                     # becomes full -> idle, dock free again
    assert sim.robots[0].status == RobotStatus.IDLE
    assert c.reserved_by is None and c.occupant is None
    assert sim.send_to_charger(1, c.id)[0]


def test_release_and_stranding_clear_reservation():
    sim = make()
    sim.reset(seed=0)
    assert sim.send_to_charger(0, 0)[0]
    assert sim.release(0)[0]
    assert sim.chargers[0].reserved_by is None
    assert sim.send_to_charger(1, 0)[0]
    sim.robots[1].energy_wh = 1e-6
    run(sim, 1)
    assert sim.robots[1].status == RobotStatus.STRANDED
    assert sim.chargers[0].reserved_by is None


def test_dock_with_stranded_robot_is_unavailable():
    sim = make()
    sim.reset(seed=0)
    c = sim.chargers[0]
    place(sim, 0, c.cell)
    sim.robots[0].status = RobotStatus.STRANDED
    assert sim.charger_state(c) == "unavailable"
    assert not sim.send_to_charger(1, c.id)[0]


# ───────────────────────────── M2: routing around long-lived obstacles

class _Sim:
    def __init__(self, ascii_map, model):
        self.layout = Layout.from_ascii(ascii_map, {}, ("aisle", "storage", "goal", "dock"))
        self.config = type("C", (), {"model": model})()

    def passable(self, zones):
        return self.layout.passable_mask(zones)


Z = ("aisle", "goal", "dock")


@pytest.mark.parametrize("model,moves", [("2.0", False), ("2.1", True)])
def test_detour_around_charging_robot(model, moves):
    # row 0 is a dock row used as a corridor; a robot charges at (0, 2); the mover at (0, 3) wants (0, 0)
    sim = _Sim("C.C..\n....G", model)
    charging = Unit(0, [], (0, 2), None, Z, None, True, 0, lasting=True)
    mover = Unit(1, [], (0, 3), (0, 0), Z, None, False, 0)
    PIBT(sim).step([charging, mover])
    assert (mover.next != mover.cell) is moves
    if moves:
        assert mover.next == (1, 3)                 # steps round the charging robot


def test_short_lived_obstacle_is_waited_for():
    sim = _Sim("C.C..\n....G", "2.1")
    working = Unit(0, [], (0, 2), None, Z, None, True, 0, lasting=False)
    mover = Unit(1, [], (0, 3), (0, 0), Z, None, False, 0)
    PIBT(sim).step([working, mover])
    assert mover.next == mover.cell


def test_no_way_around_falls_back_to_waiting():
    sim = _Sim("C.C.G", "2.1")                      # a single row: nowhere to go round
    charging = Unit(0, [], (0, 2), None, Z, None, True, 0, lasting=True)
    mover = Unit(1, [], (0, 3), (0, 0), Z, None, False, 0)
    PIBT(sim).step([charging, mover])
    assert mover.next == mover.cell


# ───────────────────────────── L: dock bays

def test_dock_bays_transform():
    assert dock_bays(".C..C.\n......\n.SS.G.") == "#C##C#\n......\n.SS.G."


def test_dock_bays_layout_and_one_way_rejected():
    d = tiny_dict()
    d["map"] = {"ascii": "C.....C\n.......\n.SS.SS.\n.......\n..G.G..", "dock_layout": "bays"}
    sim = Simulator(ScenarioConfig.from_dict(d))
    assert sim.layout.to_ascii().splitlines()[0] == "C#####C"
    d["map"] = {"generator": {"rack_rows": 1, "rack_cols": 2, "rack_height": 2, "goals": 1, "docks": 2,
                              "aisle_width": 1, "one_way": True}, "dock_layout": "bays"}
    with pytest.raises(ConfigError):
        Simulator(ScenarioConfig.from_dict(d))


def test_unknown_model_and_layout_rejected():
    with pytest.raises(ConfigError):
        ScenarioConfig.from_dict(tiny_dict(model="3.0"))
    d = tiny_dict()
    d["map"] = dict(d["map"], dock_layout="ring")
    with pytest.raises(ConfigError):
        ScenarioConfig.from_dict(d)


def test_default_model_is_2_1():
    assert ScenarioConfig.from_dict(tiny_dict()).model == "2.1"
