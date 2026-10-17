"""Literatur-Referenzwerte für Cost Scaling (ORACLE-PLAYBOOK §5.1). Jeder Wert steht in einer Quelle, die beim Schreiben eingesehen wurde (Stand der Einsicht 2026-10-17); nichts davon ist mit dem Code der Demo erzeugt.

1. networkx 3.7, Dokumentation von `networkx.network_simplex` / `min_cost_flow_cost` / `min_cost_flow` (Dateien networkx/algorithms/flow/networksimplex.py Zeilen 415-426 und mincost.py Zeilen 88-98 und 181-190, im Test-venv gelesen):
   Knoten a (Bedarf -5) und d (Bedarf 5); Kanten a->b (Kosten 3, Kapazität 4), a->c (6, 10), b->d (1, 9), c->d (2, 5). Ergebnis `flowCost` = 24 und `flowDict` {a: {b: 4, c: 1}, b: {d: 4}, c: {d: 1}}.
2. networkx 3.7, Dokumentation von `networkx.max_flow_min_cost` (Datei networkx/algorithms/flow/mincost.py, Abschnitt Examples ab Zeile 304, im Test-venv gelesen): zwölf Kanten (1,2,12,4), (1,3,20,6), (2,3,6,-3), (2,6,14,1),
   (3,4,unbegrenzt,9), (3,5,10,5), (4,2,19,13), (4,5,4,0), (5,7,28,2), (6,5,11,1), (6,7,unbegrenzt,8), (7,4,6,6) als (von, nach, Kapazität, Kosten), Quelle 1, Senke 7: `cost_of_flow` = 373.
   Unbegrenzt wird hier als 10**6 geschrieben (kein Weg von S aus nimmt mehr als 32 auf). Negative Kosten und eine Kante aus der Senke heraus gehören zum Beispiel.
3. Princeton COS 226, Vorlesung 22 "Mincost Flow" (Abschnitte "Mincost flow" mit der Tabelle der drei Maximalflüsse und "Cycle canceling example", PDF https://www.cs.princeton.edu/courses/archive/spr02/cs226/lectures/mincost.4up.pdf, Frühjahr 2002, als Text gelesen):
   Kanten (Kapazität, Kosten) 0-1 (3, 3), 0-2 (3, 1), 1-3 (2, 1), 1-4 (2, 1), 2-3 (1, 4), 2-4 (2, 2), 3-5 (2, 2), 4-5 (2, 1). Drei Maximalflüsse mit den Kosten 21, 20 und 22:
   Fluss je Kante in der Reihenfolge der Tabelle (1, 3, 1, 0, 1, 2, 2, 2) = 21, (2, 2, 2, 0, 0, 2, 2, 2) = 20, (2, 2, 1, 1, 1, 1, 2, 2) = 22. Das Beispiel zum Kreise-Löschen endet bei Kosten 20, ohne dass die Folie weitere
   negative Kreise nennt; kostenminimaler Maximalfluss also 20 (der Wert des Maximalflusses ist 4).
4. MIT 6.854J (Herbst 2005 im Verzeichnis, Vorlesung 10 vom 4. Oktober 2004, Mitschrift von Wendy Chang, Dozent David Karger, Abschnitt 10.3 "MCF Scaling by Cost", PDF https://www.ocw.mit.edu/courses/6-854j-advanced-algorithms-fall-2005/b0e838a7bc121b8afd86080ce4341f63_mincostflow.pdf, als Text gelesen):
   Definition 1: p ist eps-optimal, wenn c_p(i,j) >= -eps für alle Restkanten gilt; "We start with a max flow and a zero price function, which will be C-optimal"; je Phase von eps auf eps/2;
   Behauptung 2: ein 1/(n+1)-optimaler Maximalfluss ist optimal (ganzzahlige Kosten). In der Demo sind die Kosten mit n + 1 multipliziert, das Ende ist eps = 1 in skalierten Einheiten, also 1/(n+1) in Originaleinheiten.
Offen (nicht einsehbar): Goldberg und Tarjan 1990 (Originalarbeit), Ahuja-Magnanti-Orlin Kapitel 10 (Buch), die Vorlesung MIT 15.082J Nr. 14 (Adresse nicht abrufbar bzw. nur Binärdaten ohne Text) und die OR-Tools-Dokumentation
zu `SimpleMinCostFlow` (nur eine Suchzusammenfassung gesehen, kein Zahlenbeispiel)."""

from fractions import Fraction

import pytest

import csc_algorithm as cs
import csc_edmonds_karp as ek
import csc_scenario as sc


def _net(n, arcs, s=0, t=1):
    return sc.Net(tuple(f"n{i}" for i in range(n)), tuple(f"n{i}" for i in range(n)), tuple((i, 0) for i in range(n)), tuple((u, v, c, k, sc.K_OTHER) for u, v, c, k in arcs), s, t, False)


INF = 10 ** 6
NETWORKX_7 = [(1, 2, 12, 4), (1, 3, 20, 6), (2, 3, 6, -3), (2, 6, 14, 1), (3, 4, INF, 9), (3, 5, 10, 5), (4, 2, 19, 13), (4, 5, 4, 0), (5, 7, 28, 2), (6, 5, 11, 1), (6, 7, INF, 8), (7, 4, 6, 6)]
PRINCETON = [(0, 1, 3, 3), (0, 2, 3, 1), (1, 3, 2, 1), (1, 4, 2, 1), (2, 3, 1, 4), (2, 4, 2, 2), (3, 5, 2, 2), (4, 5, 2, 1)]       # (von, nach, Kapazität, Kosten); S = 0, T = 5
CASES = [(2, "fifo"), (4, "fifo"), (8, "generic"), (16, "generic")]


@pytest.mark.parametrize("alpha,selection", CASES)
def test_networkx_documentation_example_cost_24_with_the_documented_flow(alpha, selection):
    # a = 0 (Quelle), d = 1 (Senke), b = 2, c = 3
    net = _net(4, [(0, 2, 4, 3), (0, 3, 10, 6), (2, 1, 9, 1), (3, 1, 5, 2)])
    res = cs.cost_scaling(net, 5, alpha, selection)
    assert res.total == 24
    assert res.flow == (4, 1, 4, 1)                                      # a->b 4, a->c 1, b->d 4, c->d 1


@pytest.mark.parametrize("alpha,selection", CASES)
def test_networkx_documentation_example_max_flow_min_cost_373(alpha, selection):
    node = {1: 0, 7: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}                   # Quelle 1 -> 0, Senke 7 -> 1
    net = _net(7, [(node[u], node[v], c, k) for u, v, c, k in NETWORKX_7])
    value = ek.max_flow(net, "bfs", keep_flows=False).value
    res = cs.cost_scaling(net, value, alpha, selection)
    assert res.total == 373 and res.value == value


def test_princeton_costs_of_the_three_max_flows_and_the_minimum_20():
    net = _net(6, [(u, v, c, k) for u, v, c, k in PRINCETON], 0, 5)
    for flow, cost in (((1, 3, 1, 0, 1, 2, 2, 2), 21), ((2, 2, 2, 0, 0, 2, 2, 2), 20), ((2, 2, 1, 1, 1, 1, 2, 2), 22)):
        assert cs.flow_cost(net, flow) == cost
    assert ek.max_flow(net, "bfs", keep_flows=False).value == 4
    for alpha, selection in CASES:
        res = cs.cost_scaling(net, 4, alpha, selection)
        assert res.total == 20 and res.flow == (2, 2, 2, 0, 0, 2, 2, 2)


@pytest.mark.parametrize("arcs,n,s,t", [(PRINCETON, 6, 0, 5), ([(0, 2, 4, 3), (0, 3, 10, 6), (2, 1, 9, 1), (3, 1, 5, 2)], 4, 0, 1)])
def test_mit_6854_zero_prices_c_optimal_start_halving_and_final_epsilon_one_over_n_plus_one(arcs, n, s, t):
    """Definition 1 und Behauptung 2 der Mitschrift: Start mit Preisen 0 und eps = C (größte Kosten, hier mal n + 1 skaliert); je Phase halbiert (alpha = 2, aufgerundet auf ganze skalierte Einheiten);
    am Ende eps = 1/(n+1) in Originaleinheiten, und jede Restkante hat dann reduzierte Kosten >= -1/(n+1) (exakt in Brüchen)."""
    net = _net(n, arcs, s, t)
    value = 4 if n == 6 else 5
    res = cs.cost_scaling(net, value, 2, "fifo")
    scale = n + 1
    assert res.scale == scale and Fraction(res.eps0, scale) == max(k for *_x, k in arcs)       # C-optimal: eps_0 = C
    assert res.frames[0].prices == (0,) * n
    eps = [res.eps0] + [p.eps for p in res.phases]                       # in skalierten Einheiten
    assert all(b == max(1, -(-a // 2)) for a, b in zip(eps, eps[1:]))
    assert Fraction(eps[-1], scale) == Fraction(1, n + 1)
    price = res.frames[res.phases[-1].last_frame].prices
    flow = res.phases[-1].flow
    for i, (u, v, c, k) in enumerate(arcs):
        if flow[i] < c:
            assert k + Fraction(price[u] - price[v], scale) >= -Fraction(1, n + 1)
        if flow[i] > 0:
            assert -k + Fraction(price[v] - price[u], scale) >= -Fraction(1, n + 1)
