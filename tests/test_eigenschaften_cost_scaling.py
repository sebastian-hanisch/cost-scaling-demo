"""Eigenschaftstests (metamorph) für Cost Scaling (ORACLE-PLAYBOOK §5.2). Die Orakel-Tests vergleichen die Kosten mit linprog, networkx und der Aufzählung aller Kreise; hier wird die Eingabe auf eine
Weise verändert, deren Wirkung auf den Optimalwert bekannt ist (Netze und Mengen beliebig, nur die Beziehung zählt).

1. Umnummerieren der Knoten (auch S und T) und Umsortieren der Kanten ändert den Optimalwert nicht (mehrere alpha, beide Knotenwahlen, Randfälle).
2. Alle Kosten mal c > 0: Kosten mal c, und der Fluss des Ausgangsnetzes ist auch im skalierten Netz kostenminimal.
3. Potenzialverschiebung c'(u,v) = c(u,v) + p(u) - p(v): jeder Fluss der Menge F ändert seine Kosten um F (p(S) - p(T)), also auch das Optimum (zuerst nur nichtnegative Kosten, dann auch negative wie im Orakel-Test).
4. Eine zusätzliche Kante, mehr Kapazität oder niedrigere Kosten machen das Optimum bei fester Menge nie schlechter; Kanten mit Kapazität 0 ändern nichts (auch mit negativen Kosten).
5. Die Optimalkosten sind in der Menge konvex und nicht fallend (Grenzkosten nicht fallend, wie im Kopf von csc_ssp.py für SSP genannt; bei nichtnegativen Kosten).
6. Netz umkehren (alle Kanten gedreht, S und T getauscht): gleiche Optimalkosten.
7. Schranke für Zwischenstände: am Ende jeder Phase ist der Fluss zulässig und seine Kosten liegen höchstens (n + 1)^-1 * eps * Summe |f - x*| über dem Optimum x* (aus ε-Optimalität: f* - f ist ein Kreislauf
   im Restgraphen von f, jede Restkante hat reduzierte Kosten >= -eps) und nie darunter. Optimum x* aus scipy linprog.
8. Randfälle und Gleichstände: leeres Netz, eine Kante, alle Kosten 0, alle Kosten gleich k in der Zuordnung (Kosten = Menge mal k).
Alle mit festem Seed und 20 bis 50 Zufallsnetzen je Eigenschaft."""

import random

import pytest

import csc_algorithm as cs
import csc_edmonds_karp as ek
import csc_scenario as sc

linprog = pytest.importorskip("scipy.optimize").linprog
np = pytest.importorskip("numpy")


def _net(n, arcs, s=0, t=1):
    """Netz aus (u, v, Kapazität, Kosten)-Kanten."""
    return sc.Net(tuple(f"n{i}" for i in range(n)), tuple(f"n{i}" for i in range(n)), tuple((i, 0) for i in range(n)), tuple((u, v, c, k, sc.K_OTHER) for u, v, c, k in arcs), s, t, False)


def _instances(count, seed, negative=False):
    edge = [(2, []), (2, [(0, 1, 5, 3)]), (2, [(0, 1, 0, 3)]), (3, [(0, 2, 4, 1), (2, 1, 4, 1)]), (4, [(0, 2, 1, 2), (0, 3, 1, 2), (2, 1, 1, 2), (3, 1, 1, 2)]), (3, [(0, 1, 2, 4), (0, 2, 2, 1), (2, 1, 2, 1)])]
    rng = random.Random(seed)
    out = list(edge)
    while len(out) < count:
        n = rng.randint(3, 8)
        arcs = []
        for _ in range(rng.randint(n, 3 * n)):
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v:
                cost = rng.choice([rng.randint(-4, 9), rng.randint(0, 9), rng.choice([1, 2])]) if negative else rng.choice([rng.randint(0, 9), rng.randint(1, 9), rng.choice([1, 2])])
                arcs.append((u, v, rng.choice([0, 1, 1, 2, 3, 4, 5, 7]), cost))
        out.append((n, arcs))
    return out


def _fmax(n, arcs, s=0, t=1):
    return ek.max_flow(_net(n, arcs, s, t), "bfs", keep_flows=False).value


def _total(n, arcs, value, alpha=2, selection="fifo", s=0, t=1, trace=False):
    return cs.cost_scaling(_net(n, arcs, s, t), value, alpha, selection, keep_trace=trace)


def _cost(arcs, flow):
    return sum(f * k for (_, _, _, k), f in zip(arcs, flow))


def _check_flow(n, arcs, flow, value, s=0, t=1):
    bal = [0] * n
    for (u, v, c, _), f in zip(arcs, flow):
        assert 0 <= f <= c
        bal[u] -= f
        bal[v] += f
    assert all(bal[v] == 0 for v in range(n) if v not in (s, t)) and bal[t] == value == -bal[s] if s != t else True


def test_renumbering_nodes_and_reordering_arcs_does_not_change_the_optimum():
    rng = random.Random(11)
    for n, arcs in _instances(40, 20261017):
        fm = _fmax(n, arcs)
        for value in sorted({0, fm // 2, fm}):
            base = _total(n, arcs, value).total
            perm = list(range(n))
            rng.shuffle(perm)
            arcs2 = [(perm[u], perm[v], c, k) for u, v, c, k in arcs]
            rng.shuffle(arcs2)
            for alpha, sel in ((2, "fifo"), (4, "generic"), (16, "fifo")):
                r = _total(n, arcs2, value, alpha, sel, perm[0], perm[1])
                assert r.total == base, (n, arcs, value, alpha, sel)
                _check_flow(n, arcs2, r.flow, value, perm[0], perm[1])


def test_costs_times_c_scale_the_optimum_and_the_old_flow_stays_optimal():
    for n, arcs in _instances(40, 2024, negative=True):
        value = _fmax(n, arcs)
        base = _total(n, arcs, value)
        for c in (2, 3, 10):
            arcs2 = [(u, v, cap, c * k) for u, v, cap, k in arcs]
            scaled = _total(n, arcs2, value)
            assert scaled.total == c * base.total == _cost(arcs2, base.flow), (c, n, arcs)


def test_potential_shift_changes_every_flow_of_value_f_by_f_times_the_potential_difference():
    rng = random.Random(77)
    nonneg = negative = 0
    for n, arcs in _instances(50, 99):
        value = _fmax(n, arcs)
        base = _total(n, arcs, value)
        for _ in range(3):
            pi = [rng.randint(0, 6) for _ in range(n)]
            arcs2 = [(u, v, cap, k + pi[u] - pi[v]) for u, v, cap, k in arcs]
            shifted = _total(n, arcs2, value)
            assert shifted.total == base.total + value * (pi[0] - pi[1]), (n, arcs, pi)
            assert _cost(arcs2, base.flow) == shifted.total                      # derselbe Fluss bleibt optimal
            if min((k for *_x, k in arcs2), default=0) >= 0:
                nonneg += 1
            else:
                negative += 1
    assert nonneg >= 10 and negative >= 10


def test_more_options_never_make_the_optimum_worse_and_zero_capacity_arcs_change_nothing():
    rng = random.Random(5)
    for n, arcs in _instances(50, 4242):
        fm = _fmax(n, arcs)
        value = max(0, fm - rng.randint(0, 1))
        base = _total(n, arcs, value).total
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            assert _total(n, arcs + [(u, v, rng.randint(0, 5), rng.randint(0, 9))], value).total <= base, (n, arcs, u, v)
            assert _total(n, arcs + [(u, v, 0, rng.randint(-5, 9))], value).total == base
        if arcs:
            i = rng.randrange(len(arcs))
            more = list(arcs)
            more[i] = (arcs[i][0], arcs[i][1], arcs[i][2] + rng.randint(1, 4), arcs[i][3])
            assert _total(n, more, value).total <= base, (n, arcs, i)
            cheaper = list(arcs)
            cheaper[i] = (arcs[i][0], arcs[i][1], arcs[i][2], arcs[i][3] - rng.randint(1, 4))
            assert _total(n, cheaper, value).total <= base, (n, arcs, i)


def test_the_optimal_cost_is_nondecreasing_and_convex_in_the_quantity():
    for n, arcs in _instances(40, 6006):
        fm = _fmax(n, arcs)
        g = [_total(n, arcs, f).total for f in range(fm + 1)]
        assert g[0] == 0
        marginal = [b - a for a, b in zip(g, g[1:])]
        assert all(m >= 0 for m in marginal), (n, arcs, g)
        assert all(a <= b for a, b in zip(marginal, marginal[1:])), (n, arcs, g)


def test_reversing_all_arcs_and_swapping_s_and_t_keeps_the_optimum():
    for n, arcs in _instances(40, 818, negative=True):
        value = _fmax(n, arcs)
        rev = [(v, u, c, k) for u, v, c, k in arcs]
        for alpha in (2, 8):
            a, b = _total(n, arcs, value, alpha), _total(n, rev, value, alpha, "fifo", 1, 0)
            assert a.total == b.total, (n, arcs, alpha)
            _check_flow(n, rev, b.flow, value, 1, 0)


def test_every_phase_end_is_feasible_never_below_the_optimum_and_within_eps_times_the_flow_distance():
    """Ist (f, p) eps-optimal, ist f* - f ein Kreislauf im Restgraphen von f; mit dem Zerlegen in Restkanten folgt scale * (c(f) - c(f*)) <= eps * Summe |f - f*|."""
    rng = random.Random(31)
    checked = 0
    for n, arcs in _instances(60, 1001, negative=True):
        if not arcs:
            continue
        value = _fmax(n, arcs)
        a = np.zeros((n, len(arcs)))
        for i, (u, v, _c, _k) in enumerate(arcs):
            a[u, i] += 1
            a[v, i] -= 1
        b = np.zeros(n)
        b[0], b[1] = value, -value
        lp = linprog([k for *_x, k in arcs], A_eq=a, b_eq=b, bounds=[(0, c) for _u, _v, c, _k in arcs], method="highs")
        assert lp.status == 0
        for alpha in (2, rng.choice((3, 4, 8))):
            r = _total(n, arcs, value, alpha, trace=True)
            for ph in r.phases:
                _check_flow(n, arcs, ph.flow, value)
                gap = r.scale * (_cost(arcs, ph.flow) - lp.fun)
                distance = float(np.abs(np.array(ph.flow) - lp.x).sum())
                assert -1e-6 <= gap <= ph.eps * distance + 1e-6, (n, arcs, alpha, ph.eps, gap, distance)
                checked += 1
    assert checked >= 100


def test_edge_cases_empty_net_one_arc_zero_costs_and_equal_costs_in_the_assignment():
    r = cs.cost_scaling(_net(2, []), 0)
    assert r.total == 0 and r.value == 0 and r.flow == ()
    for cap, cost in ((5, 3), (1, 0), (4, 7)):
        r = cs.cost_scaling(_net(2, [(0, 1, cap, cost)]), cap)
        assert r.total == cap * cost and r.flow == (cap,)
    for n, arcs in _instances(30, 3):
        zero = [(u, v, c, 0) for u, v, c, _ in arcs]
        fm = _fmax(n, zero)
        assert all(_total(n, zero, f).total == 0 for f in {0, fm // 2, fm})
    for size in range(1, 5):                                           # Gleichstände: S -> F_i (0) -> A_j (k) -> T (0), jede Zuordnung kostet size * k
        for k in (1, 3, 9):
            arcs = [(0, 2 + i, 1, 0) for i in range(size)] + [(2 + i, 2 + size + j, 1, k) for i in range(size) for j in range(size)] + [(2 + size + j, 1, 1, 0) for j in range(size)]
            for alpha, sel in ((2, "fifo"), (4, "generic")):
                assert _total(2 + 2 * size, arcs, size, alpha, sel).total == size * k
