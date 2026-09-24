"""Kern: Cost Scaling gegen Handfälle und unabhängige Gegenproben (networkx, scipy, Brute Force), Invarianten je Bild aus dem Trace (Überschuss-Bilanz, epsilon-Optimalität, Preise nur fallend),
Phasen (epsilon-Folge, Präfix-Eigenschaft beim vorzeitigen Anhalten), epsilon-optimal = jeder Kreis hat mittlere Kosten >= -epsilon (gegen die Aufzählung aller einfachen Kreise), Zertifikat, Skalierungsinvarianzen."""

import itertools
import math
import random

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, linprog

import csc_algorithm as cs
import csc_constants as C
import csc_edmonds_karp as ek
import csc_evaluation as ev
import csc_scenario as sc
import csc_ssp as ssp
from csc_scenario import SplitMix64


def _nets(count, sizes=((2, 2, 3), (3, 3, 6), (3, 3, 8), (4, 3, 5), (2, 4, 9), (6, 6, 12))):
    """Zufällige Distributionsnetze unterschiedlicher Größe, Dichte, Streuung und Auslastung."""
    rng = random.Random(13)
    for i in range(count):
        p, d, s = sizes[i % len(sizes)]
        yield sc.generate(p, d, s, rng.choice((20, 40, 60, 80, 100)), rng.choice((0, 25, 50, 75, 100)), rng.choice((40, 90, 120, 160)), 3000 + i)


def _value(net):
    return ek.max_flow(net, "bfs", keep_flows=False).value


def _optimum(net):
    g = nx.DiGraph()
    g.add_nodes_from(range(net.n))
    for u, v, c, k, _ in net.arcs:
        g.add_edge(u, v, capacity=c, weight=k)
    flow = nx.max_flow_min_cost(g, net.s, net.t)
    return sum(flow[net.s].values()), nx.cost_of_flow(g, flow)


def _balance(net, flow):
    bal = [0] * net.n
    for i, (u, v, _, _, _) in enumerate(net.arcs):
        bal[u] -= flow[i]
        bal[v] += flow[i]
    return bal


def _is_flow(net, flow, value):
    bal = _balance(net, flow)
    return all(0 <= flow[i] <= net.arcs[i][2] for i in range(net.m)) and all(bal[v] == 0 for v in range(net.n) if v not in (net.s, net.t)) and bal[net.t] == value == -bal[net.s]


def _residual(net, flow):
    """Restkanten (Index -> (u, v, Rest, Kosten))."""
    out = {}
    for i, (u, v, c, k, _) in enumerate(net.arcs):
        out[2 * i] = (u, v, c - flow[i], k)
        out[2 * i + 1] = (v, u, flow[i], -k)
    return out


def _min_rc(net, flow, price, scale):
    return min(scale * k + price[u] - price[v] for (u, v, rest, k) in _residual(net, flow).values() if rest > 0)


def test_splitmix64_reference_vector():
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def test_the_predecessor_copies_reproduce_their_numbers():
    """Wachen: Edmonds-Karp 52 475 durchsuchte Kanten (Mittel 524,75), SSP 53 907 (Mittel 539,07) über die 100 festen Netze; SSP-Kosten = networkx."""
    scans, ssp_scans = 0, 0
    for seed in C.DIST_SEEDS:
        net = sc.generate(3, 3, 8, 60, 50, 90, seed)
        scans += ek.max_flow(net, "bfs", keep_flows=False).scanned_total
        r = ssp.ssp(net, keep_trace=False)
        ssp_scans += r.scanned_total
        if seed < C.DIST_SEEDS[0] + 20:
            assert (r.value, r.total) == _optimum(net)
    assert scans == 52475 and ssp_scans == 53907


def test_detour_takes_the_expensive_way_first_and_the_detour_later():
    """Umweg: nach Phase 1 (epsilon 23) kostet der Fluss noch 9, am Ende 2."""
    net = sc.detour_cost()
    r = cs.cost_scaling(net, 1)
    assert r.scale == 5 and r.eps0 == 45 and [p.eps for p in r.phases] == [23, 12, 6, 3, 2, 1]
    assert r.phases[0].cost == 9 and r.total == 2 and r.flow == (1, 0, 1, 1)


def test_brute_force_on_the_detour_and_the_assignment_is_the_hungarian_optimum():
    net = sc.detour_cost()
    best = min(sum(f[i] * net.arcs[i][3] for i in range(net.m)) for f in itertools.product((0, 1), repeat=net.m) if _is_flow(net, f, 1))
    assert best == 2 == cs.cost_scaling(net, 1).total
    net = sc.assignment_cost(5)
    cost = np.zeros((5, 5), dtype=int)
    for u, v, _, k, _ in net.arcs:
        if 2 <= u < 7 and 7 <= v < 12:
            cost[u - 2, v - 7] = k
    rows, cols = linear_sum_assignment(cost)
    for alpha in C.ALPHAS:
        assert cs.cost_scaling(net, 5, alpha).total == cost[rows, cols].sum() == 10


@pytest.mark.parametrize("alpha", C.ALPHAS)
@pytest.mark.parametrize("selection", cs.SELECTIONS)
def test_cost_equals_networkx_on_the_hundred_fixed_nets(alpha, selection):
    for seed in C.DIST_SEEDS:
        net = sc.generate(3, 3, 8, 60, 50, 90, seed)
        r = cs.cost_scaling(net, _value(net), alpha, selection, keep_trace=False)
        assert (r.value, r.total) == _optimum(net), (alpha, selection, seed)
        assert _is_flow(net, r.flow, r.value)


def test_cost_equals_networkx_on_varied_nets_and_the_flow_is_feasible():
    for net in _nets(60):
        r = cs.cost_scaling(net, _value(net), 2, "fifo", keep_trace=False)
        assert (r.value, r.total) == _optimum(net) and _is_flow(net, r.flow, r.value)


def test_a_smaller_quantity_than_the_maximum_also_works():
    """Cost Scaling löst auch Mengen unterhalb des größten Flusses (billigster Fluss der Menge), gegen networkx.network_simplex mit Bedarfen."""
    for net in _nets(30):
        top = _value(net)
        if top < 2:
            continue
        value = max(1, top // 2)
        g = nx.DiGraph()
        g.add_nodes_from(range(net.n))
        for u, v, c, k, _ in net.arcs:
            g.add_edge(u, v, capacity=c, weight=k)
        g.nodes[net.s]["demand"], g.nodes[net.t]["demand"] = -value, value
        cost, _ = nx.network_simplex(g)
        r = cs.cost_scaling(net, value, 2, "fifo", keep_trace=False)
        assert r.total == cost and _is_flow(net, r.flow, value)


def test_scipy_linear_program_agrees():
    for net in list(_nets(10)):
        value = _value(net)
        m = net.m
        a_eq, b_eq = [], []
        for v in range(net.n):
            row = np.zeros(m)
            for i, (u, w, _, _, _) in enumerate(net.arcs):
                row[i] += (w == v) - (u == v)
            if v not in (net.s, net.t):
                a_eq.append(row)
                b_eq.append(0)
        a_eq.append(np.array([1.0 if net.arcs[i][0] == net.s else 0.0 for i in range(m)]))
        b_eq.append(value)
        lp = linprog([a[3] for a in net.arcs], A_eq=np.array(a_eq), b_eq=b_eq, bounds=[(0, a[2]) for a in net.arcs], method="highs")
        assert lp.status == 0 and round(lp.fun) == cs.cost_scaling(net, value, 4, "fifo", keep_trace=False).total


@pytest.mark.parametrize("alpha", [2, 4])
@pytest.mark.parametrize("selection", cs.SELECTIONS)
def test_frame_invariants_from_the_trace(alpha, selection):
    """Je Bild: Überschuss = Angebot + Zufluss - Abfluss (Summe 0), Fluss innerhalb der Kapazitäten, Preise nur fallend, epsilon-Optimalität aller Restkanten (nach dem Sättigen 0-optimal);
    am Ende jeder Phase kein Überschuss mehr und der Fluss zulässig."""
    for net in _nets(30):
        value = _value(net)
        r = cs.cost_scaling(net, value, alpha, selection)
        prev_prices = (0,) * net.n
        for f in r.frames:
            bal = _balance(net, f.flow)
            supply = [0] * net.n
            supply[net.s], supply[net.t] = value, -value
            assert list(f.excess) == [supply[v] + bal[v] for v in range(net.n)] and sum(f.excess) == 0
            assert all(0 <= f.flow[i] <= net.arcs[i][2] for i in range(net.m))
            assert all(a <= b for a, b in zip(f.prices, prev_prices))
            prev_prices = f.prices
            if f.kind == "saturate":
                assert _min_rc(net, f.flow, f.prices, r.scale) >= 0
            elif f.kind == "discharge":
                assert _min_rc(net, f.flow, f.prices, r.scale) >= -f.eps
        for ph in r.phases:
            end = r.frames[ph.last_frame]
            assert all(x == 0 for x in end.excess) and _is_flow(net, end.flow, value) and end.flow == ph.flow
            assert ph.min_rc >= -ph.eps and ph.min_rc == _min_rc(net, ph.flow, end.prices, r.scale)


def test_the_epsilon_sequence_and_phase_structure():
    for net in _nets(20):
        for alpha in C.ALPHAS:
            r = cs.cost_scaling(net, _value(net), alpha, keep_trace=False)
            assert [p.eps for p in r.phases] == cs.eps_sequence(r.eps0, alpha) and r.phases[-1].eps == 1 and r.eps0 == max(1, r.scale * max(a[3] for a in net.arcs))
            assert all(a.eps > b.eps for a, b in zip(r.phases, r.phases[1:]))
            assert len(r.phases) <= math.ceil(math.log(r.eps0, alpha)) + 1        # ceil-Division kann eine Phase mehr brauchen als der reine Logarithmus
    assert cs.eps_sequence(100, 2) == [50, 25, 13, 7, 4, 2, 1] and cs.eps_sequence(1, 2) == [1] and cs.eps_sequence(180, 4) == [45, 12, 3, 1]


def test_scanned_and_operation_counts_add_up():
    net = sc.generate(3, 3, 8, 60, 50, 90, 155)
    r = cs.cost_scaling(net, _value(net))
    assert r.scanned_total == sum(p.scanned for p in r.phases) and len(r.frames) == 1 + len(r.phases) + r.discharges
    discharges = [f for f in r.frames if f.kind == "discharge"]
    assert r.pushes == sum(len(f.pushes) for f in discharges) and r.relabels == sum(len(f.relabels) for f in discharges)
    assert sum(p.sat_pushes for p in r.phases) <= r.pushes
    assert (r.discharges, r.pushes, r.relabels, r.scanned_total) == (690, 800, 470, 5190)


def test_stopping_early_is_a_prefix_of_the_full_run_and_never_cheaper_than_the_optimum():
    for net in _nets(30):
        value = _value(net)
        full = cs.cost_scaling(net, value, 2, "fifo")
        for stop in (3, 12, 45, 10_000):
            early = cs.cost_scaling(net, value, 2, "fifo", True, stop)
            k = len(early.phases)
            assert early.phases[-1].eps <= stop or early.phases[-1].eps == 1
            assert [(p.eps, p.cost, p.scanned) for p in early.phases] == [(p.eps, p.cost, p.scanned) for p in full.phases[:k]] and early.flow == full.phases[k - 1].flow
            assert early.total >= full.total == _optimum(net)[1] and _is_flow(net, early.flow, value)


def test_epsilon_optimal_means_every_residual_cycle_has_mean_cost_at_least_minus_epsilon():
    """Bezug zu Cycle-Canceling: am Ende jeder Phase hat jeder einfache Kreis im Restgraphen einen Mittelwert der skalierten Kosten je Kante >= -epsilon (Aufzählung aller Kreise, Kleinstnetze)."""
    checked = 0
    for net in _nets(20, sizes=((2, 2, 3), (2, 3, 3), (3, 2, 4))):
        r = cs.cost_scaling(net, _value(net), 2)
        for ph in r.phases:
            g = nx.DiGraph()
            for e, (u, v, rest, k) in _residual(net, ph.flow).items():
                if rest > 0:
                    g.add_edge(u, v, weight=r.scale * k)
            for cyc in nx.simple_cycles(g):
                mean = sum(g[cyc[i]][cyc[(i + 1) % len(cyc)]]["weight"] for i in range(len(cyc))) / len(cyc)
                assert mean >= -ph.eps - 1e-9
                checked += 1
    assert checked > 100


def test_final_certificate_no_negative_cycle_and_complementary_slackness():
    for net in _nets(40):
        r = cs.cost_scaling(net, _value(net), 4, keep_trace=False)
        pi, ok = ssp.certificate(net, r.flow)
        assert ok and ev.proof(net, r.flow, pi)["valid"]
        g = nx.DiGraph()
        for i, (u, v, c, k, _) in enumerate(net.arcs):
            if c - r.flow[i] > 0:
                g.add_edge(u, v, weight=k)
            if r.flow[i] > 0:
                g.add_edge(v, u, weight=-k)
        assert not nx.negative_edge_cycle(g)


def test_a_flow_stopped_too_early_may_have_a_negative_cycle():
    net = sc.generate(3, 3, 8, 60, 50, 90, 38)
    early = cs.cost_scaling(net, _value(net), 2, "fifo", True, stop_eps=ev.stop_eps_for(net, 5))
    assert early.total == 1027 and _optimum(net)[1] == 953 and ssp.certificate(net, early.flow)[1] is False


def test_costs_times_k_scale_the_optimum_and_can_only_add_phases():
    for net in _nets(20):
        value = _value(net)
        base = cs.cost_scaling(net, value, 2, keep_trace=False)
        prev = len(base.phases)
        for k in (10, 1000):
            big = cs.cost_scaling(sc.scale_costs(net, k), value, 2, keep_trace=False)
            assert big.total == k * base.total and len(big.phases) >= prev
            prev = len(big.phases)


def test_capacities_and_quantity_times_k_keep_the_phase_count():
    for net in _nets(20):
        value = _value(net)
        base = cs.cost_scaling(net, value, 2, keep_trace=False)
        for k in (10, 1000):
            big = cs.cost_scaling(sc.scale_capacities(net, k), k * value, 2, keep_trace=False)
            assert big.total == k * base.total and [p.eps for p in big.phases] == [p.eps for p in base.phases]


def test_an_unreachable_net_has_no_work_but_the_phases_of_the_sequence():
    net = sc.generate(2, 6, 3, 20, 50, 90, 8)
    r = cs.cost_scaling(net, 0, 2)
    assert r.value == 0 and r.total == 0 and r.discharges == 0 and [p.eps for p in r.phases] == cs.eps_sequence(r.eps0, 2)


def test_an_infeasible_quantity_is_rejected_and_does_not_hang():
    for net in (sc.detour_cost(), sc.generate(3, 3, 8, 60, 50, 90, 155)):
        with pytest.raises(ValueError):
            cs.cost_scaling(net, _value(net) + 1, 2, keep_trace=False)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        cs.cost_scaling(sc.detour_cost(), 1, 2, "lifo")
    with pytest.raises(ValueError):
        cs.cost_scaling(sc.detour_cost(), 1, 1)


def test_results_are_deterministic():
    net = sc.generate(3, 3, 8, 60, 50, 90, 155)
    a, b = cs.cost_scaling(net, _value(net), 2), cs.cost_scaling(net, _value(net), 2)
    assert a.flow == b.flow and a.prices == b.prices and [f.node for f in a.frames] == [f.node for f in b.frames]
