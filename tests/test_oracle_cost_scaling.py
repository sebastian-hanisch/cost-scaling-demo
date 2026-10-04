"""Unabhängiges Orakel: Cost Scaling auf zufälligen Kleinstnetzen (Parallel- und Gegenkanten, Kapazität 0, Gleichstände, auch negative Kosten) gegen scipy.optimize.linprog (HiGHS).
Zu jedem Netz Mengen unterhalb, am und oberhalb des größten Flusses: Kosten und Zulässigkeit stimmen, eine unzulässige Menge wird abgelehnt, am Ende jeder Phase ist der Fluss zulässig und
epsilon-optimal (aus Fluss und Preisen des letzten Bildes der Phase nachgerechnet, nicht aus den Zählern der Phase)."""

import random

import numpy as np
import pytest

import csc_algorithm as cs
import csc_scenario as sc

linprog = pytest.importorskip("scipy.optimize").linprog


def _net(rng):
    n = rng.randint(3, 8)
    arcs = []
    for _ in range(rng.randint(n, 3 * n)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            arcs.append((u, v, rng.choice([0, 1, 1, 2, 3, 4, 5, 7]), rng.choice([rng.randint(-4, 9), rng.randint(0, 9), rng.choice([1, 2])])))
    net = sc.Net(tuple(f"n{i}" for i in range(n)), tuple(f"n{i}" for i in range(n)), tuple((i, 0) for i in range(n)), tuple((u, v, c, k, sc.K_OTHER) for u, v, c, k in arcs), 0, 1, False)
    return n, arcs, net


def _lp(n, arcs, value):
    """(zulässig?, Mindestkosten) des Flusses der Menge `value` von Knoten 0 nach 1."""
    if not arcs:
        return False, None
    a = np.zeros((n, len(arcs)))
    for i, (u, v, _c, _k) in enumerate(arcs):
        a[u, i] += 1
        a[v, i] -= 1
    b = np.zeros(n)
    b[0], b[1] = value, -value
    res = linprog([k for *_x, k in arcs], A_eq=a, b_eq=b, bounds=[(0, c) for _u, _v, c, _k in arcs], method="highs")
    return res.status == 0, (round(res.fun) if res.status == 0 else None)


def _balance(n, arcs, flow):
    bal = [0] * n
    for (u, v, c, _k), f in zip(arcs, flow):
        assert 0 <= f <= c
        bal[u] += f
        bal[v] -= f
    return bal


@pytest.mark.parametrize("alpha,selection", [(2, "fifo"), (4, "generic"), (16, "fifo")])
def test_random_small_nets_against_linprog(alpha, selection):
    rng = random.Random(20261004)
    ok = rejected = 0
    for _ in range(40):
        n, arcs, net = _net(rng)
        for value in (1, 2, 4, 9):
            feasible, expected = _lp(n, arcs, value)
            if not feasible:
                with pytest.raises(ValueError):
                    cs.cost_scaling(net, value, alpha, selection)
                rejected += 1
                continue
            res = cs.cost_scaling(net, value, alpha, selection)
            ok += 1
            assert res.total == expected
            target = [0] * n
            target[0], target[1] = value, -value
            assert _balance(n, arcs, res.flow) == target
            scale = n + 1
            for ph in res.phases:
                frame = res.frames[ph.last_frame]
                assert _balance(n, arcs, frame.flow) == target
                price = frame.prices
                for (u, v, c, k), f in zip(arcs, frame.flow):
                    if f < c:
                        assert scale * k + price[u] - price[v] >= -ph.eps
                    if f > 0:
                        assert -scale * k + price[v] - price[u] >= -ph.eps
    assert ok >= 40 and rejected >= 20
