"""Jede Zahl, die README und Hilfetexte nennen, ist hier belegt (Standardeinstellungen, 100 feste Netze, Seeds 100000-100099)."""

import pytest

import csc_algorithm as cs
import csc_constants as C
import csc_evaluation as ev
import csc_scenario as sc

S = (C.DEFAULT_P, C.DEFAULT_D, C.DEFAULT_S, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD)


@pytest.fixture(scope="module")
def dist():
    return ev.distribution(*S)


def test_every_net_ends_in_the_optimum_and_the_phase_count(dist):
    """α = 2: immer 8 Phasen (ε 90, 45, 23, 12, 6, 3, 2, 1), ε₀ im Mittel 178 (höchstens 180); alle 100 Netze enden im Optimum von SSP."""
    assert dist["n_seeds"] == 100 and dist["share_optimal"] == 1.0
    assert dist["phases_mean"] == dist["phases_max"] == 8 and round(dist["eps0_mean"]) == 178 and dist["eps0_max"] == 180


def test_work_per_run(dist):
    """Entladungen 494 (Median 488, höchstens 769), Pushes 573, Relabels 349 (0,71 je Entladung), sättigende Pushes 23 % der Pushes."""
    assert (round(dist["discharges_mean"]), dist["discharges_median"], dist["discharges_max"]) == (494, 487.5, 769)
    assert (round(dist["pushes_mean"]), round(dist["relabels_mean"])) == (573, 349)
    assert round(dist["relabels_mean"] / dist["discharges_mean"], 2) == 0.71 and round(100 * sum(dist["cols"]["sat_pushes"]) / sum(dist["cols"]["pushes"])) == 23


def test_cost_scaling_loses_to_ssp_on_the_small_nets(dist):
    """Durchsuchte Kanten: Cost Scaling 3737 (Median 3699, höchstens 5655), SSP 539; das 7,2-fache (Median 7,1, höchstens 13,3); in keinem der 100 Netze höchstens so viele wie SSP."""
    assert (round(dist["scans_mean"]), dist["scans_median"], dist["scans_max"], round(dist["ssp_scans_mean"])) == (3737, 3698.5, 5655, 539)
    assert (round(dist["ratio_mean"], 1), round(dist["ratio_median"], 1), round(dist["ratio_max"], 1)) == (7.2, 7.1, 13.3) and dist["share_cs_le_ssp"] == 0.0


def test_the_quality_after_each_phase(dist):
    """Mehrkosten nach Phase 1..5: 3,0 / 1,7 / 0,56 / 0,14 / 0,01 % im Mittel (Median 2,2 / 1,15 / 0,1 / 0 / 0; größter Wert 16,5 / 8,8 / 5,2 / 2,4 / 0,4 %); optimal 13 / 19 / 46 / 79 / 98 %."""
    assert [round(g, 3) for g in dist["gap_mean"][:5]] == [3.009, 1.707, 0.557, 0.138, 0.008]
    assert [round(g, 2) for g in dist["gap_median"][:5]] == [2.2, 1.15, 0.1, 0.0, 0.0] and [round(g, 1) for g in dist["gap_max"][:5]] == [16.5, 8.8, 5.2, 2.4, 0.4]
    assert [round(100 * x) for x in dist["share_optimal_after"][:6]] == [13, 19, 46, 79, 98, 100]


def test_the_cost_does_not_fall_monotonically(dist):
    """In 60 von 100 Netzen wird der Fluss in mindestens einer Phase wieder teurer; Beispiel Seed 38: Mehrkosten 7,8 % → 1,7 % → 4,2 % → 0."""
    assert dist["share_nonmonotone"] == 0.6
    d = ev.verdict(ev.analyse(sc.generate(*S, 38), 2, "fifo", 0))[2]
    assert [round(g, 1) for g in d["gaps"][:4]] == [7.8, 1.7, 4.2, 0.0]


def test_alpha_table():
    """Durchsuchte Kanten je α (Warteschlange): 3737 / 2620 / 2581 / 3079 / 3539 für α = 2, 4, 8, 16, 32, Phasen 8 / 4 / 3 / 2 / 2; beliebige Knotenwahl bei α = 2: 3862."""
    rows = ev.alpha_table(*S)
    fifo = {r["alpha"]: r for r in rows if r["selection"] == "fifo"}
    assert [round(fifo[a]["scans"]) for a in C.ALPHA_TABLE] == [3737, 2620, 2581, 3079, 3539] and [fifo[a]["phases"] for a in C.ALPHA_TABLE] == [8, 4, 3, 2, 2]
    assert round(next(r for r in rows if r["alpha"] == 2 and r["selection"] == "generic")["scans"]) == 3862 and all(r["all_optimal"] for r in rows)
    assert min(rows, key=lambda r: r["scans"])["alpha"] in (4, 8)


def test_tolerance_table():
    """Anhalten bei Toleranz 1 / 2 / 3 / 5: Phasen 3,9 / 2,9 / 2 / 1, durchsuchte Kanten 1881 / 1438 / 1061 / 583 (gegen 3737 bis zum Optimum), Mehrkosten im Mittel 0,15 / 0,69 / 1,7 / 3,0 %, optimal 77 / 43 / 19 / 13 %."""
    rows = {r["tol"]: r for r in ev.tolerance_table(*S)}
    assert [round(rows[t]["phases"], 2) for t in (1, 2, 3, 5)] == [3.92, 2.92, 2.0, 1.0] and [round(rows[t]["scans"]) for t in (0, 1, 2, 3, 5)] == [3737, 1881, 1438, 1061, 583]
    assert [round(rows[t]["gap_mean"], 3) for t in (1, 2, 3, 5)] == [0.148, 0.69, 1.707, 3.009] and [round(100 * rows[t]["share_optimal"]) for t in (1, 2, 3, 5)] == [77, 43, 19, 13]
    assert rows[0]["gap_mean"] == 0.0 and rows[0]["share_optimal"] == 1.0


def test_cost_range():
    """Kosten × 1 / 10 / 100 / 1000: ε₀ 179 / 1790 / 17 900 / 179 000, Phasen 8 / 11 / 15 / 18 (α = 2), Cost Scaling 3829 / 5298 / 7122 / 8570 Kanten, SSP immer 570 (20 Netze)."""
    rows = ev.cost_range_table(*S)
    assert [round(r["eps0"]) for r in rows] == [179, 1790, 17900, 179000] and [round(r["phases"]) for r in rows] == [8, 11, 15, 18]
    assert [round(r["scans"]) for r in rows] == [3829, 5298, 7122, 8570] and [round(r["ssp_scans"]) for r in rows] == [570] * 4 and all(r["all_optimal"] for r in rows)


def test_the_crossing_point_on_larger_nets():
    """Cost Scaling α = 4 durchsucht ab 166 Knoten weniger Kanten als SSP, α = 2 ab 306 Knoten; bei 458 Knoten das 0,57- bzw. 0,40-fache; im kleinsten Netz (12 Knoten) das 14,5-fache (α = 2)."""
    rows = ev.scaling()
    assert [round(r["n"]) for r in rows] == [12, 19, 30, 52, 90, 166, 306, 458] and [round(r["m"]) for r in rows] == [15, 33, 70, 188, 439, 1182, 2862, 6297]
    assert [round(r["ssp"]) for r in rows] == [60, 433, 1678, 8135, 35892, 178056, 696968, 2582008]
    assert [round(r["cs2"]) for r in rows] == [863, 3601, 9231, 29087, 72773, 214748, 634068, 1483486] and [round(r["cs4"]) for r in rows] == [612, 2367, 7354, 22179, 54085, 168258, 422686, 1025685]
    assert round(ev.crossing(rows, "cs4")["n"]) == 166 and round(ev.crossing(rows, "cs2")["n"]) == 306
    assert round(rows[-1]["cs2"] / rows[-1]["ssp"], 2) == 0.57 and round(rows[-1]["cs4"] / rows[-1]["ssp"], 2) == 0.4 and round(rows[0]["cs2"] / rows[0]["ssp"], 1) == 14.5
    slopes = ev.slopes(rows)
    assert (round(slopes["ssp"], 2), round(slopes["cs2"], 2), round(slopes["cs4"], 2)) == (1.72, 1.2, 1.19)
    assert [round(r["phases2"]) for r in rows] == [7, 8, 9, 9, 10, 11, 12, 13]
    assert round(rows[0]["ssp_rounds"]) == 3 and round(rows[-1]["ssp_rounds"]) == 401


def test_assignment_crossing():
    """Zuordnung n × n (α = 4, 3 Zuordnungen): Cost Scaling ÷ SSP 5,4 / 3,4 / 2,5 / 1,5 / 1,04 / 0,80 / 0,51 für n = 5 / 10 / 20 / 40 / 60 / 80 / 120; SSP braucht n Runden, Cost Scaling 4 bis 6 Phasen."""
    rows = ev.assignment_table()
    assert [r["n"] for r in rows] == [5, 10, 20, 40, 60, 80, 120] and all(r["all_optimal"] for r in rows)
    assert [round(r["cs"] / r["ssp"], 2) for r in rows] == [5.38, 3.38, 2.52, 1.46, 1.04, 0.8, 0.51] and [r["rounds"] for r in rows] == [5, 10, 20, 40, 60, 80, 120]
    assert [round(r["phases"]) for r in rows] == [4, 4, 5, 5, 6, 6, 6]
    assert next(r["n"] for r in rows if r["cs"] < r["ssp"]) == 80


def test_default_net_numbers():
    d = ev.verdict(ev.analyse(sc.generate(*S, C.DEFAULT_SEED)))[2]
    assert (d["value"], d["demand"], d["total"], d["optimal_total"], d["phases"], d["eps0"], d["eps"]) == (76, 76, 1197, 1197, 8, 180, [90, 45, 23, 12, 6, 3, 2, 1])
    assert (d["discharges"], d["pushes"], d["relabels"], d["scanned"], d["ssp_scanned"], d["ssp_rounds"]) == (690, 800, 470, 5190, 824, 14)
    assert d["costs"][:4] == [1199, 1207, 1197, 1198] and round(d["first_gap_pct"], 1) == 0.2
    a4 = ev.verdict(ev.analyse(sc.generate(*S, C.DEFAULT_SEED), 4))[2]
    assert (a4["phases"], a4["scanned"], a4["eps"]) == (4, 3648, [45, 12, 3, 1])
    g = ev.verdict(ev.analyse(sc.generate(*S, C.DEFAULT_SEED), 2, "generic"))[2]
    assert (g["scanned"], g["total"]) == (4907, 1197)


def test_early_stop_seed_38_and_the_other_presets():
    e = ev.verdict(ev.analyse(sc.generate(*S, 38), 2, "fifo", 5))
    assert e[1] == "early" and (e[2]["total"], e[2]["optimal_total"], e[2]["gap_pct"], e[2]["phases"], e[2]["scanned"]) == (1027, 953, 7.8, 1, 590)
    assert ev.verdict(ev.analyse(sc.generate(*S, 38)))[2]["scanned"] == 3737
    k = ev.verdict(ev.analyse(sc.generate(3, 3, 8, 60, 50, 140, C.DEFAULT_SEED)))[2]
    assert (k["value"], k["demand"], k["total"]) == (89, 118, 1403)
    t = ev.verdict(ev.analyse(sc.generate(3, 3, 8, 30, 50, 90, C.DEFAULT_SEED)))[2]
    assert (t["scanned"], t["ssp_scanned"]) == (3031, 353)
    u = ev.verdict(ev.analyse(sc.detour_cost()))[2]
    assert (u["discharges"], u["scanned"], u["ssp_scanned"], u["costs"][0], u["total"], u["eps"]) == (9, 84, 7, 9, 2, [23, 12, 6, 3, 2, 1])
    z = ev.verdict(ev.analyse(sc.assignment_cost(5)))[2]
    assert (z["phases"], z["scanned"], z["ssp_scanned"], z["total"]) == (7, 1147, 150, 10)


def test_extremes_of_the_input_ranges_end_in_the_optimum():
    import networkx as nx
    for args in ((2, 2, 3, 20, 0, 40), (6, 6, 12, 100, 100, 160), (2, 6, 12, 20, 100, 160), (6, 2, 3, 100, 0, 40)):
        net = sc.generate(*args, 5)
        g = nx.DiGraph()
        for u, v, c, k, _ in net.arcs:
            g.add_edge(u, v, capacity=c, weight=k)
        flow = nx.max_flow_min_cost(g, net.s, net.t)
        value = sum(flow[net.s].values())
        for alpha in (2, 4):
            assert cs.cost_scaling(net, value, alpha, keep_trace=False).total == nx.cost_of_flow(g, flow)
