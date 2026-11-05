"""PIBT movement: one step per tick, no vertex or swap collisions, push chains, dead-end swaps."""
import numpy as np

from battery_tarware import ScenarioConfig, Simulator
from battery_tarware.layout import Layout
from battery_tarware.traffic import PIBT, Unit, reverse_distances
from conftest import tiny_dict


class _Sim:
    """Minimal stand-in exposing what PIBT needs."""
    def __init__(self, ascii_map):
        self.layout = Layout.from_ascii(ascii_map, {}, ("aisle", "storage", "goal", "dock"))

    def passable(self, zones):
        return self.layout.passable_mask(zones)


Z = ("aisle", "goal", "dock")


def unit(i, cell, goal, prio=0, fixed=False, zones=Z):
    return Unit(i, [], cell, goal, zones, None, fixed, prio)


def step(sim, units):
    PIBT(sim).step(units)
    nxt = [u.next for u in units]
    assert len(set(nxt)) == len(nxt), "vertex collision"
    for a in units:
        for b in units:
            if a is not b:
                assert not (a.next == b.cell and b.next == a.cell and a.next != a.cell), "swap"
        assert abs(a.next[0] - a.cell[0]) + abs(a.next[1] - a.cell[1]) <= 1
    return nxt


def test_reverse_distances_respect_one_way():
    s = _Sim(">>>\n...\n..G")
    d = reverse_distances(s.layout, s.passable(Z), (0, 2))
    assert d[0, 0] == 2 and d[0, 1] == 1
    d2 = reverse_distances(s.layout, s.passable(Z), (0, 0))
    assert d2[0, 2] > 2                                      # must leave the one-way row and come back


def test_push_chain_in_a_corridor():
    s = _Sim(".....\n####.\n....G")
    a = unit(0, (0, 0), (0, 4), prio=10)
    b = unit(1, (0, 1), None)                                # idle robot in the way
    c = unit(2, (0, 2), None)
    nxt = step(s, [a, b, c])
    assert nxt[0] == (0, 1) and nxt[1] == (0, 2) and nxt[2] == (0, 3)


def test_fixed_units_block():
    s = _Sim("....\n####\n...G")
    a = unit(0, (0, 0), (0, 3), prio=10)
    b = unit(1, (0, 1), (0, 1), fixed=True)
    assert step(s, [a, b])[0] == (0, 0)


def test_dead_end_swap_lets_a_robot_out_of_a_bay():
    # robot b sits in a one-cell bay (0,1) and wants out through a's cell; a wants into the bay
    s = _Sim("#.#\n...\n..G")
    a = unit(0, (1, 1), (0, 1), prio=10)
    b = unit(1, (0, 1), (2, 2), prio=1)
    nxt = step(s, [a, b])
    assert nxt[0] != (1, 1) and nxt[1] == (1, 1)            # a steps aside, b comes out


def test_all_tiers_reach_steady_throughput():
    import battery_tarware as bt
    for name, horizon, floor in [("basic", 1500, 90), ("intermediate", 3000, 70), ("advanced", 3000, 120)]:
        env = bt.make(name, max_steps=horizon)
        env.reset(seed=5)
        coord = bt.GreedyCoalitionCoordinator()
        last_half = 0
        for t in range(horizon):
            _, _, term, trunc, info = env.step(coord.act(env))
            if t >= horizon // 2:
                last_half += sum(e.type == "task_completed" for e in info["events"])
            if term or trunc:
                break
        m = env.sim.metrics()
        assert m["tasks_completed"] >= floor, (name, m)
        assert last_half > 0.25 * m["tasks_completed"], (name, "throughput collapsed late", m)
        assert m["stranded"] == 0, (name, m)
