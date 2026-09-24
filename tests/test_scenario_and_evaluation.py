"""Szenario (Aufbau, Reproduzierbarkeit, Stufen, Lehrnetze), Auswertung (Urteil, Verteilungen, α, Toleranz, Kostenspanne, Skalierung, Zuordnung)."""

import pytest

import csc_algorithm as cs
import csc_constants as C
import csc_edmonds_karp as ek
import csc_evaluation as ev
import csc_scenario as sc
import csc_ssp as ssp

DEFAULT = (C.DEFAULT_P, C.DEFAULT_D, C.DEFAULT_S, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD)


def _net(seed=C.DEFAULT_SEED, *settings):
    return sc.generate(*(settings or DEFAULT), seed)


def test_generation_is_reproducible_and_seed_dependent():
    assert _net(5) == _net(5) and _net(5) != _net(6)


def test_structure_of_a_distribution_net():
    p, d, s = 3, 3, 8
    net = _net()
    assert net.n == 2 + p + 2 * d + s and net.s == 0 and net.t == 1 and net.logistic
    kinds = [a[4] for a in net.arcs]
    assert kinds.count(sc.K_SUPPLY) == p and kinds.count(sc.K_THROUGHPUT) == d and kinds.count(sc.K_DEMAND) == s
    assert all(a[2] >= 1 for a in net.arcs)
    for u, v, _, _, kind in net.arcs:
        assert net.pos[u][1] > net.pos[v][1]


def test_costs_lie_in_the_documented_ranges():
    for seed in range(20):
        for u, v, _, cost, kind in _net(seed).arcs:
            lo, hi = {sc.K_SUPPLY: (1, 5), sc.K_LANE_IN: (1, 9), sc.K_THROUGHPUT: (1, 3), sc.K_LANE_OUT: (1, 9), sc.K_DEMAND: (0, 0)}[kind]
            assert lo <= cost <= hi


@pytest.mark.parametrize("seed", range(20))
def test_every_plant_and_store_has_a_lane_even_on_the_thinnest_net(seed):
    net = _net(seed, 3, 3, 8, C.DENSITY_MIN, C.DEFAULT_SPREAD, C.DEFAULT_LOAD)
    assert {a[1] for a in net.arcs if a[4] == sc.K_SUPPLY} <= {a[0] for a in net.arcs if a[4] == sc.K_LANE_IN}
    assert {a[0] for a in net.arcs if a[4] == sc.K_DEMAND} <= {a[1] for a in net.arcs if a[4] == sc.K_LANE_OUT}


def test_a_thinner_net_only_removes_lanes_and_keeps_all_other_values():
    for seed in range(20):
        thin, full = _net(seed, 3, 3, 8, 40, 50, 90), _net(seed, 3, 3, 8, 100, 50, 90)
        assert set(thin.arcs) <= set(full.arcs) and len(thin.arcs) < len(full.arcs)


def test_teaching_nets_and_build_ignore_the_random_settings():
    assert not sc.assignment_cost(5).logistic and sc.build("assignment", 6, 6, 12, 100, 100, 160, 1) == sc.assignment_cost(5)
    assert sc.build("detour", 6, 6, 12, 100, 100, 160, 1) == sc.detour_cost()
    assert sc.build("random", *DEFAULT, 9) == _net(9)


def test_teaching_nets_have_the_documented_shape():
    d = sc.detour_cost()
    assert d.n == 4 and d.m == 4 and [a[2] for a in d.arcs] == [1] * 4 and [a[3] for a in d.arcs] == [0, 9, 1, 1]
    a = sc.assignment_cost(5)
    assert a.n == 12 and a.m == 35 and all(x[2] == 1 for x in a.arcs) and {x[3] for x in a.arcs if 2 <= x[0] < 7} <= set(range(1, 10))
    big = sc.assignment_cost(40, 7)
    assert big.n == 82 and big.m == 40 * 40 + 80


def test_scale_costs_and_capacities_change_only_their_column():
    net = _net()
    c7, u7 = sc.scale_costs(net, 7), sc.scale_capacities(net, 7)
    assert [(a[0], a[1], a[2], a[4]) for a in c7.arcs] == [(a[0], a[1], a[2], a[4]) for a in net.arcs] and [a[3] for a in c7.arcs] == [7 * a[3] for a in net.arcs]
    assert [(a[0], a[1], a[3], a[4]) for a in u7.arcs] == [(a[0], a[1], a[3], a[4]) for a in net.arcs] and [a[2] for a in u7.arcs] == [7 * a[2] for a in net.arcs]


def test_stop_eps_for_maps_the_tolerance_to_scaled_units():
    net = _net()
    assert ev.stop_eps_for(net, 0) == 1 and ev.stop_eps_for(net, 1) == net.n + 1 and ev.stop_eps_for(net, 5) == 5 * (net.n + 1)


def test_verdict_codes():
    lvl, code, d = ev.verdict(ev.analyse(_net(1, 3, 3, 8, 100, 50, 40)))
    assert (lvl, code) == ("success", "optimal") and d["value"] == d["demand"]
    lvl, code, d = ev.verdict(ev.analyse(_net(1, 3, 3, 8, 60, 50, 160)))
    assert (lvl, code) == ("success", "optimal") and d["value"] < d["demand"] and set(d["stage_caps"]) & set(ev.STAGE_KINDS)
    lvl, code, d = ev.verdict(ev.analyse(_net(38), 2, "fifo", 5))
    assert (lvl, code) == ("info", "early") and d["total"] > d["optimal_total"] and d["phases"] < d["n_phases_full"]
    lvl, code, d = ev.verdict(ev.analyse(sc.generate(2, 6, 3, 20, 50, 90, 8)))
    assert (lvl, code) == ("warning", "disconnected") and d["value"] == 0
    # Toleranz gesetzt, aber der Fluss ist zufällig schon optimal: 'optimal'
    seed = next(s for s in range(1, 200) if ev.verdict(ev.analyse(_net(s), 2, "fifo", 2))[1] == "optimal" and ev.verdict(ev.analyse(_net(s), 2, "fifo", 2))[2]["phases"] < 8)
    assert ev.verdict(ev.analyse(_net(seed), 2, "fifo", 2))[0] == "success"


def test_verdict_data_is_consistent():
    a = ev.analyse(_net())
    _, _, d = ev.verdict(a)
    assert d["phases"] == len(d["eps"]) == len(d["costs"]) == len(d["gaps"]) and d["eps0"] == a.result.eps0 and d["scale"] == a.net.n + 1
    assert d["discharges"] == a.result.discharges and d["pushes"] == a.result.pushes and d["relabels"] == a.result.relabels and d["scanned"] == a.result.scanned_total == sum(d["phase_scans"])
    assert d["ssp_rounds"] == len(ssp.ssp(a.net, keep_trace=False).rounds) and d["optimal_total"] == d["total"] == d["costs"][-1]
    assert d["n_phases_full"] == len(cs.eps_sequence(d["eps0"], 2)) and d["saturated"][0] >= 0 and all(rc >= -e for rc, e in zip(d["min_rc"], d["eps"]))


def test_distribution_is_consistent_with_single_runs():
    seeds = tuple(range(5))
    dist = ev.distribution(*DEFAULT, seeds=seeds)
    for i, seed in enumerate(seeds):
        net = sc.generate(*DEFAULT, seed)
        r = cs.cost_scaling(net, ek.max_flow(net, "bfs", keep_flows=False).value, 2, "fifo", False)
        assert dist["cols"]["scans"][i] == r.scanned_total and dist["cols"]["discharges"][i] == r.discharges and dist["cols"]["phases"][i] == len(r.phases)
        assert dist["cols"]["ssp_scans"][i] == ssp.ssp(net, keep_trace=False).scanned_total
    assert dist["n_seeds"] == 5 and dist["share_optimal"] == 1.0 and dist["gap_mean"][-1] == 0.0 and len(dist["gap_after"]) == dist["max_phases"]


def test_tables_have_the_documented_shape():
    rows = ev.alpha_table(*DEFAULT, alphas=(2, 4), seeds=C.DIST_SEEDS[:10])
    assert [(r["alpha"], r["selection"]) for r in rows] == [(2, "fifo"), (2, "generic"), (4, "fifo"), (4, "generic")] and all(r["all_optimal"] for r in rows)
    tol = ev.tolerance_table(*DEFAULT, seeds=C.DIST_SEEDS[:10])
    assert [r["tol"] for r in tol] == list(C.TOLERANCES) and tol[0]["gap_mean"] == 0.0 and all(a["scans"] >= b["scans"] for a, b in zip(tol, tol[1:]))
    cr = ev.cost_range_table(*DEFAULT, factors=(1, 10), seeds=C.DIST_SEEDS[:5])
    assert [r["k"] for r in cr] == [1, 10] and cr[1]["phases"] >= cr[0]["phases"] and cr[0]["ssp_scans"] == cr[1]["ssp_scans"] and all(r["all_optimal"] for r in cr)
    sr = ev.scaling(sizes=C.SCALE_SIZES[:3], seeds=C.SCALE_SEEDS[:2])
    assert [r["size"] for r in sr] == list(C.SCALE_SIZES[:3]) and all(r["cs2"] > 0 and r["ssp"] > 0 for r in sr) and all(0.5 < s < 3.5 for s in ev.slopes(sr).values())
    ar = ev.assignment_table(sizes=(5, 10), seeds=(7,))
    assert [r["n"] for r in ar] == [5, 10] and all(r["all_optimal"] and r["rounds"] == r["n"] for r in ar)
    assert ev.crossing([{"cs4": 5, "ssp": 3}, {"cs4": 2, "ssp": 3}], "cs4")["cs4"] == 2 and ev.crossing([{"cs4": 5, "ssp": 3}], "cs4") is None


def test_proof_helper_counts_and_detects_violations():
    net = sc.detour_cost()
    r = cs.cost_scaling(net, 1)
    pi, ok = ssp.certificate(net, r.flow)
    p = ev.proof(net, r.flow, pi)
    assert ok and p["valid"] and p["violations"] == 0 and p["flow_arcs"] == 3
    assert not ev.proof(net, r.flow, tuple(0 for _ in range(net.n)))["valid"]                       # Potenziale 0 sind hier ungültig
