"""Kennzahlen, Urteil und die Experimente der Demo (Verteilungen über feste Netze, α und Knotenwahl, Kostenspanne, Netzgröße und Kreuzungspunkt gegen SSP, Näherung nach jeder Phase, Zuordnung).
Alles ganzzahlig gerechnet; Prozente entstehen erst bei der Ausgabe."""

from dataclasses import dataclass
from functools import lru_cache
from statistics import mean, median

import numpy as np

import csc_algorithm as cs
import csc_constants as C
import csc_edmonds_karp as ek
import csc_scenario as sc
import csc_ssp as ssp

K_SUPPLY, K_LANE_IN, K_THROUGHPUT, K_LANE_OUT, K_DEMAND = sc.K_SUPPLY, sc.K_LANE_IN, sc.K_THROUGHPUT, sc.K_LANE_OUT, sc.K_DEMAND
STAGE_KINDS = (K_SUPPLY, K_LANE_IN, K_THROUGHPUT, K_LANE_OUT)


@dataclass(frozen=True)
class Analysis:
    net: sc.Net
    result: cs.Result            # mit dem gewählten α, der Knotenwahl und der Toleranz
    optimum: ssp.Result          # Successive Shortest Paths, größter Fluss (die Messlatte)
    value: int                   # Menge F (größter Fluss)
    demand: int
    stage_caps: dict
    alpha: int
    selection: str
    tol: int


def _demand(net):
    return sum(c for _, _, c, _, kind in net.arcs if kind == K_DEMAND)


def _stage_caps(net, cut_arcs):
    caps = {}
    for i in cut_arcs:
        kind = net.arcs[i][4]
        caps[kind] = caps.get(kind, 0) + net.arcs[i][2]
    return caps


def pct(numerator, denominator, digits=1):
    return round(100 * numerator / denominator, digits) if denominator else 0.0


def stop_eps_for(net, tol):
    """Toleranz je Kante (Kosteneinheiten) -> epsilon in skalierten Einheiten: anhalten, sobald epsilon <= (n+1) * tol. 0 = bis zum Optimum (epsilon = 1)."""
    return 1 if tol == 0 else max(1, (net.n + 1) * tol)


def analyse(net, alpha=C.DEFAULT_ALPHA, selection=C.DEFAULT_SELECTION, tol=C.DEFAULT_TOLERANCE):
    e = ek.max_flow(net, "bfs")
    res = cs.cost_scaling(net, e.value, alpha, selection, True, stop_eps_for(net, tol))
    opt = ssp.ssp(net, "dijkstra", None, keep_trace=False)
    return Analysis(net, res, opt, e.value, _demand(net) if net.logistic else 0, _stage_caps(net, e.cut_arcs), alpha, selection, tol)


def verdict(a):
    """(Stufe, Code, Daten): 'optimal' = kostenminimal, 'early' = bei Toleranz angehalten, nicht kostenminimal, 'wrong' = am Ende nicht optimal (dürfte nie vorkommen), 'disconnected' = gar nichts kommt an."""
    res, opt = a.result, a.optimum.total
    phases = res.phases
    gaps = [pct(p.cost - opt, opt) if opt else 0.0 for p in phases]
    data = {
        "value": res.value, "demand": a.demand, "share": pct(res.value, a.demand) if a.demand else None, "total": res.total, "optimal_total": opt,
        "gap_pct": pct(res.total - opt, opt) if opt else 0.0, "phases": len(phases), "eps0": res.eps0, "eps": [p.eps for p in phases], "scale": res.scale,
        "discharges": res.discharges, "pushes": res.pushes, "relabels": res.relabels, "sat_pushes": sum(p.sat_pushes for p in phases), "scanned": res.scanned_total,
        "ssp_rounds": len(a.optimum.rounds), "ssp_scanned": a.optimum.scanned_total, "stage_caps": a.stage_caps, "gaps": gaps, "costs": [p.cost for p in phases],
        "first_gap_pct": gaps[0] if gaps else 0.0, "last_eps": phases[-1].eps if phases else res.eps0, "saturated": [p.saturated for p in phases],
        "min_rc": [p.min_rc for p in phases], "phase_scans": [p.scanned for p in phases], "phase_discharges": [p.discharges for p in phases],
        "n_phases_full": len(cs.eps_sequence(res.eps0, a.alpha)),
    }
    if res.value == 0:
        return "warning", "disconnected", data
    if res.total > opt:
        return ("info", "early", data) if a.tol > 0 and res.stop_eps > 1 else ("warning", "wrong", data)
    return "success", "optimal", data


@lru_cache(maxsize=64)
def _instance(p, d, s, density, spread, load, seed):
    net = sc.generate(p, d, s, density, spread, load, seed)
    return net, ek.max_flow(net, "bfs", keep_flows=False).value, ssp.ssp(net, "dijkstra", None, keep_trace=False)


@lru_cache(maxsize=128)
def _runs(p, d, s, density, spread, load, alpha=C.DEFAULT_ALPHA, selection=C.DEFAULT_SELECTION, seeds=C.DIST_SEEDS):
    """Je Netz das Optimum (SSP) und Cost Scaling bis ε = 1 (ohne Trace); von den Verteilungen gemeinsam genutzt."""
    out = []
    for seed in seeds:
        net, value, opt = _instance(p, d, s, density, spread, load, seed)
        out.append((net, opt, cs.cost_scaling(net, value, alpha, selection, False)))
    return out


@lru_cache(maxsize=128)
def distribution(p, d, s, density, spread, load, alpha=C.DEFAULT_ALPHA, selection=C.DEFAULT_SELECTION, seeds=C.DIST_SEEDS):
    """Über feste Netze: Phasen, Entladungen, Pushes, Relabels, durchsuchte Kanten gegen SSP; Näherungsgüte nach jeder Phase (Mehrkosten in %, Anteil optimal)."""
    keys = ["phases", "discharges", "pushes", "relabels", "sat_pushes", "scans", "ssp_scans", "ssp_rounds", "ratio", "value", "eps0"]
    cols = {k: [] for k in keys}
    runs = _runs(p, d, s, density, spread, load, alpha, selection, seeds)
    max_ph = max(len(r.phases) for _, _, r in runs)
    gap_after = [[] for _ in range(max_ph)]        # je Phase k die Mehrkosten in % (nach der letzten Phase eines Netzes: 0)
    optimal_after = [0] * max_ph
    optimal_final = 0
    cheaper = 0
    nonmonotone = 0
    for net, opt, r in runs:
        cols["phases"].append(len(r.phases))
        cols["discharges"].append(r.discharges)
        cols["pushes"].append(r.pushes)
        cols["relabels"].append(r.relabels)
        cols["sat_pushes"].append(sum(x.sat_pushes for x in r.phases))
        cols["scans"].append(r.scanned_total)
        cols["ssp_scans"].append(opt.scanned_total)
        cols["ssp_rounds"].append(len(opt.rounds))
        cols["ratio"].append(r.scanned_total / opt.scanned_total if opt.scanned_total else 0.0)
        cols["value"].append(r.value)
        cols["eps0"].append(r.eps0)
        optimal_final += r.total == opt.total
        cheaper += r.scanned_total <= opt.scanned_total
        costs = [x.cost for x in r.phases]
        nonmonotone += any(b > a for a, b in zip(costs, costs[1:]))
        for k in range(max_ph):
            cost = r.phases[min(k, len(r.phases) - 1)].cost
            gap_after[k].append(pct(cost - opt.total, opt.total) if opt.total else 0.0)
            optimal_after[k] += cost == opt.total
    n = len(seeds)
    out = {"n_seeds": n, "cols": cols, "share_optimal": optimal_final / n, "share_cs_le_ssp": cheaper / n, "share_nonmonotone": nonmonotone / n, "max_phases": max_ph,
           "gap_after": gap_after, "gap_mean": [mean(g) for g in gap_after], "gap_median": [median(g) for g in gap_after], "gap_max": [max(g) for g in gap_after],
           "share_optimal_after": [x / n for x in optimal_after], "nodes_mean": mean(net.n for net, *_ in runs), "edges_mean": mean(net.m for net, *_ in runs)}
    for k in keys:
        out[k + "_mean"], out[k + "_median"], out[k + "_max"] = mean(cols[k]), median(cols[k]), max(cols[k])
    return out


@lru_cache(maxsize=32)
def tolerance_table(p, d, s, density, spread, load, tolerances=C.TOLERANCES, alpha=C.DEFAULT_ALPHA, seeds=C.DIST_SEEDS):
    """Anhalten bei Toleranz δ je Kante (ε <= (n+1)·δ): genutzte Phasen, durchsuchte Kanten, Mehrkosten in % und Anteil optimal - aus den Phasen der vollständigen Läufe (jede Phase ist ein Präfix)."""
    runs = _runs(p, d, s, density, spread, load, alpha, C.DEFAULT_SELECTION, seeds)
    rows = []
    for tol in tolerances:
        used, scans, gaps, optimal = [], [], [], 0
        for net, opt, r in runs:
            stop = stop_eps_for(net, tol)
            k = next((i for i, ph in enumerate(r.phases) if ph.eps <= stop), len(r.phases) - 1)
            used.append(k + 1)
            scans.append(sum(ph.scanned for ph in r.phases[:k + 1]))
            gaps.append(pct(r.phases[k].cost - opt.total, opt.total) if opt.total else 0.0)
            optimal += r.phases[k].cost == opt.total
        rows.append({"tol": tol, "phases": mean(used), "scans": mean(scans), "gap_mean": mean(gaps), "gap_max": max(gaps), "share_optimal": optimal / len(runs), "ssp_scans": mean(o.scanned_total for _, o, _ in runs)})
    return rows


@lru_cache(maxsize=32)
def alpha_table(p, d, s, density, spread, load, alphas=C.ALPHA_TABLE, seeds=C.DIST_SEEDS):
    """Skalierungsfaktor α (Warteschlange und beliebige Knotenwahl): Phasen, Entladungen, Relabels, durchsuchte Kanten im Mittel."""
    rows = []
    for alpha in alphas:
        for selection in cs.SELECTIONS:
            runs = _runs(p, d, s, density, spread, load, alpha, selection, seeds)
            rs = [r for _, _, r in runs]
            rows.append({"alpha": alpha, "selection": selection, "phases": mean(len(r.phases) for r in rs), "discharges": mean(r.discharges for r in rs), "relabels": mean(r.relabels for r in rs),
                         "pushes": mean(r.pushes for r in rs), "scans": mean(r.scanned_total for r in rs), "all_optimal": all(r.total == o.total for _, o, r in runs)})
    return rows


@lru_cache(maxsize=16)
def cost_range_table(p, d, s, density, spread, load, factors=C.COST_FACTORS, seeds=C.COST_SEEDS, alpha=C.DEFAULT_ALPHA):
    """Kosten mal k (Kapazitäten unverändert): Phasen wachsen mit log(n·C), SSP bleibt gleich - Vergleich der durchsuchten Kanten."""
    rows = []
    for k in factors:
        cols = {"phases": [], "eps0": [], "scans": [], "ssp_scans": [], "discharges": []}
        same = []
        for seed in seeds:
            net, value, _ = _instance(p, d, s, density, spread, load, seed)
            big = sc.scale_costs(net, k)
            opt = ssp.ssp(big, "dijkstra", None, keep_trace=False)
            r = cs.cost_scaling(big, value, alpha, "fifo", False)
            same.append(r.total == opt.total)
            cols["phases"].append(len(r.phases))
            cols["eps0"].append(r.eps0)
            cols["scans"].append(r.scanned_total)
            cols["ssp_scans"].append(opt.scanned_total)
            cols["discharges"].append(r.discharges)
        rows.append({"k": k, "all_optimal": all(same), **{key: mean(v) for key, v in cols.items()}})
    return rows


@lru_cache(maxsize=32)
def scaling(sizes=C.SCALE_SIZES, seeds=C.SCALE_SEEDS):
    """Durchsuchte Kanten gegen die Netzgröße: Cost Scaling (α = 2 und α = 4) und SSP; der Kreuzungspunkt liegt dort, wo das Verhältnis unter 1 fällt."""
    rows = []
    for (p, d, s) in sizes:
        cols = {k: [] for k in ("cs2", "cs4", "ssp", "phases2", "ssp_rounds", "n", "m")}
        for seed in seeds:
            net = sc.generate(p, d, s, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD, seed)
            value = ek.max_flow(net, "bfs", keep_flows=False).value
            r2 = cs.cost_scaling(net, value, 2, "fifo", False)
            cols["cs2"].append(r2.scanned_total)
            cols["cs4"].append(cs.cost_scaling(net, value, 4, "fifo", False).scanned_total)
            o = ssp.ssp(net, "dijkstra", None, keep_trace=False)
            cols["ssp"].append(o.scanned_total)
            cols["phases2"].append(len(r2.phases))
            cols["ssp_rounds"].append(len(o.rounds))
            cols["n"].append(net.n)
            cols["m"].append(net.m)
        rows.append({"size": (p, d, s), **{k: mean(v) for k, v in cols.items()}})
    return rows


def slopes(rows):
    x = np.log([r["m"] for r in rows])
    return {k: float(np.polyfit(x, np.log([r[k] for r in rows]), 1)[0]) for k in ("cs2", "cs4", "ssp")}


def crossing(rows, key="cs4"):
    """Erste Zeile, in der Cost Scaling weniger Kanten durchsucht als SSP (oder None)."""
    return next((r for r in rows if r[key] < r["ssp"]), None)


@lru_cache(maxsize=16)
def assignment_table(sizes=C.ASSIGN_SIZES, seeds=C.ASSIGN_SEEDS, alpha=4):
    """Zuordnung n × n mit Einheitskapazitäten: SSP braucht n Runden, Cost Scaling ist von der Menge unabhängig."""
    rows = []
    for n in sizes:
        cols = {"cs": [], "ssp": [], "phases": [], "rounds": []}
        same = []
        for seed in seeds:
            net = sc.assignment_cost(n, seed)
            value = ek.max_flow(net, "bfs", keep_flows=False).value
            r = cs.cost_scaling(net, value, alpha, "fifo", False)
            o = ssp.ssp(net, "dijkstra", None, keep_trace=False)
            same.append(r.total == o.total)
            cols["cs"].append(r.scanned_total)
            cols["ssp"].append(o.scanned_total)
            cols["phases"].append(len(r.phases))
            cols["rounds"].append(len(o.rounds))
        rows.append({"n": n, "all_optimal": all(same), **{k: mean(v) for k, v in cols.items()}})
    return rows


def proof(net, flow, pi):
    """Optimalitätsprüfung mit Potenzialen pi: jede Restkante hat reduzierte Kosten >= 0 (Vorwärtsrest: c + pi(u) - pi(v) >= 0; Kante mit Fluss: <= 0).
    Rückgabe: geprüfte Restkanten, Verletzungen, Kanten mit Fluss (`flow_arcs`) und davon die mit reduzierten Kosten 0 (`tight`), kleinste reduzierte Kosten einer Vorwärts-Restkante."""
    checked = violations = tight = flow_arcs = 0
    worst = None
    for i, (u, v, cap, cost, _) in enumerate(net.arcs):
        rc = cost + pi[u] - pi[v]
        if flow[i] < cap:
            checked += 1
            violations += rc < 0
            worst = rc if worst is None else min(worst, rc)
        if flow[i] > 0:
            checked += 1
            violations += rc > 0
            flow_arcs += 1
            tight += rc == 0
    return {"valid": violations == 0, "checked": checked, "violations": violations, "tight": tight, "flow_arcs": flow_arcs, "worst": worst}
