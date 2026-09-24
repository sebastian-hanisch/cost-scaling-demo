# Cost Scaling – erst grob, dann fein – Streamlit-Demo

*(noch nicht deployed)*

Sechstes Stück der **Netzwerkfluss-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Konvergenz aus [Push-Relabel](https://github.com/sebastian-hanisch/push-relabel-demo) und der ε-Skalierung der Auktion (Demo `auction-algorithm-demo` der Matching-Linie, dort nur für die Zuordnung):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Cost Scaling** (Goldberg und Tarjan 1990) – an einem wachsenden Beispiel.
[Successive Shortest Paths](https://github.com/sebastian-hanisch/ssp-demo) beweist den billigsten Fluss Weg für Weg, [Cycle-Canceling](https://github.com/sebastian-hanisch/cycle-canceling-demo) Kreis für Kreis; beide verlangen zu jedem Zeitpunkt etwas Exaktes. Cost Scaling lockert das: jeder Knoten hat einen **Preis** p, und es genügt, dass keine Restkante *mehr als ε* zu billig ist (reduzierte Kosten c_p ≥ −ε, **ε-Optimalität**).
Es beginnt grob – ε so groß wie die größten (mit n + 1 multiplizierten) Kosten – und teilt ε in **Phasen** durch α. Jede Phase ist ein **Push-Relabel** auf den reduzierten Kosten: alle Restkanten mit negativen reduzierten Kosten werden gesättigt, die Überschüsse wandern über zulässige Kanten, ein Knoten ohne zulässige Kante senkt seinen Preis.
Am Ende jeder Phase ist der Fluss **zulässig**, bei ε = 1 **kostenminimal**. Das ist das Verfahren hinter OR-Tools `SimpleMinCostFlow` (dort mit Heuristiken). Vehikel wie in den Vorgänger-Demos: ein Distributionsnetz (Werke → Verteilzentren → Filialen) mit Kosten je Einheit, dazu zwei Lehrnetze.

**Einordnung in die Reihe (die Kanten des Graphen):** Konvergenz zweier Äste: Push-Relabel (Überschüsse, Höhen, lokale Pushes) liefert die Bauweise, die ε-Skalierung der Auktion die Idee, grob zu beginnen. Cost Scaling ist von der Menge unabhängig – SSP braucht so viele Runden, wie es Wege gibt.
Mit ε-Optimalität ist auch der Bezug zu Cycle-Canceling klar: (f, p) ist ε-optimal genau dann, wenn jeder Kreis im Restgraphen einen Mittelwert der Kosten je Kante ≥ −ε hat (Goldberg–Tarjan). Bisher gebaut: die ersten zehn Stücke.
```
edmonds-karp-demo (Wurzel: Restgraph, Rückkanten, Max-Flow = Min-Cut)                  [gebaut]
  ├─ dinic-demo (viele kürzeste Wege je Phase: Niveaugraph, blockierender Fluss)        [gebaut]
  ├─ push-relabel-demo (kein Weg: Überschüsse schieben, Höhen anheben)                 [gebaut]
  └─ ssp-demo (Kosten: der billigste Weg im Restgraphen, Potenziale)                    [gebaut]
       ├─ cycle-canceling-demo (negative Kreise löschen) → Netzwerksimplex               [gebaut]
       │    (network-flow-demo)                                                          [gebaut als Fall-Demo]
       ├─ cost-scaling-demo (Push-Relabel + ε-Skalierung, das nutzt OR-Tools)           [dieses Stück]
       └─ multicommodity-demo (mehrere Güter teilen Kapazität: Kanten-LP, Preise)       [gebaut]
            ├─ mcf-column-generation-demo (Pfade als Spalten, Pricing = Dijkstra)       [gebaut]
            ├─ garg-koenemann-demo (Näherung mit Preisen, ohne LP-Löser)                [gebaut]
            └─ fixkosten-netzdesign-demo (Fixkosten: Schranke und Schnitte)             [gebaut]
                 ├─ Benders-Zerlegung (Entwurf im Master, Fluss im Teilproblem)         [geplant]
                 └─ Slope Scaling (Heuristik für große Netze)                           [geplant]
```

## Ergebnis (Zahlen aus den Tests)

Jede hier genannte Zahl ist in `tests/test_claims.py` belegt: die Lehrnetze von Hand, die Beispielnetze über ihre Seeds, die Verteilungen über 100 feste Netze (Seeds 100000–100099, dieselben wie in den Vorgänger-Demos). Standard: 3 Werke, 3 Verteilzentren, 8 Filialen, Netzdichte 60 %, Streuung 50 %, Auslastung 90 %, α = 2, Warteschlange (FIFO), bis ε = 1. Kosten je Einheit: Werk 1–5, Lane 1–9, Verteilzentrum 1–3, Filialnachfrage 0.
Die Menge F ist der größte Fluss; Edmonds-Karp und SSP sind aus den Vorgänger-Demos kopiert, Wache-Tests: 52 475 (Edmonds-Karp) bzw. 53 907 (SSP) durchsuchte Kanten über die 100 Netze, SSP-Kosten gleich dem Optimum von `networkx`.

| Frage | Ergebnis |
|---|---|
| Ist der Fluss am Ende kostenminimal? | ✅ Ja, in allen 100 Netzen für α = 2, 4, 8, 16 und beide Knotenwahlen, jeweils gleich dem Optimum von `networkx` (`max_flow_min_cost`); auf 60 weiteren Netzen unterschiedlicher Größe, gegen ein lineares Programm (`scipy`, HiGHS), gegen `linear_sum_assignment` (Zuordnung: 10), Brute-Force-Aufzählung (Umweg: 2) und für Mengen unterhalb des größten Flusses gegen `networkx.network_simplex` mit Bedarfen. Am Ende kein negativer Kreis und gültige Potenziale. |
| Stimmen die Invarianten? | ✅ Nach jeder Entladung (aus dem Trace geprüft): Überschuss = Angebot + Zufluss − Abfluss (Summe 0), Fluss innerhalb der Kapazitäten, Preise nur fallend, ε-Optimalität aller Restkanten (nach dem Sättigen 0-optimal); am Ende jeder Phase kein Überschuss mehr und der Fluss zulässig; ε strikt fallend bis 1; ein vorzeitig angehaltener Lauf ist ein Präfix des vollständigen. |
| Heißt ε-optimal wirklich „jeder Kreis hat Mittel ≥ −ε“? | ✅ Ja: am Ende jeder Phase liegt der Mittelwert der skalierten Kosten je Kante auf **jedem** einfachen Kreis des Restgraphen bei ≥ −ε (Aufzählung aller Kreise mit `networkx.simple_cycles` auf Kleinstnetzen, über 100 Kreise geprüft). |
| Wie viele Phasen? | Bei α = 2 immer **8** (ε₀ = 180: 90, 45, 23, 12, 6, 3, 2, 1); α = 4, 8, 16, 32: 4, 3, 2, 2 Phasen. Mit Kosten × 1000 wächst ε₀ auf 179 000 und die Phasen auf 18 (α = 2). |
| Wie viel Arbeit? | Im Mittel 494 Entladungen (Median 488, höchstens 769), 573 Pushes, 349 Relabels (0,71 je Entladung); 23 % der Pushes sättigen die Kante. |
| Ist Cost Scaling schneller als SSP? | ❌ Auf den kleinen Netzen der Regler nein: **3737 gegen 539** durchsuchte Kanten, das 7,2-fache (Median 7,1, höchstens 13,3); in keinem der 100 Netze höchstens so viele wie SSP. Beispielnetz: 5190 gegen 824. |
| Ab wann gewinnt es? | ✅ Mit der Netzgröße: Steigung im doppelt logarithmischen Diagramm **1,20 (α = 2) und 1,19 (α = 4) gegen 1,72 bei SSP**. Mit α = 4 durchsucht Cost Scaling ab **166 Knoten** weniger Kanten als SSP (168 258 gegen 178 056), mit α = 2 ab **306 Knoten**; bei 458 Knoten das 0,57- bzw. 0,40-fache (α = 2 bzw. 4). Im kleinsten Netz (12 Knoten) das 14,5-fache. SSP braucht bei 458 Knoten 401 Runden, Cost Scaling 13 Phasen. |
| Und bei der Zuordnung? | ✅ Ähnlich: Cost Scaling ÷ SSP = 5,4 / 3,4 / 2,5 / 1,5 / 1,04 / 0,80 / 0,51 für n = 5 / 10 / 20 / 40 / 60 / 80 / 120; Cost Scaling gewinnt ab n = 80 (Kreuzung zwischen 60 und 80), 4 bis 6 Phasen gegen n Runden. |
| Welches α? | ⚠️ Durchsuchte Kanten im Mittel: 3737 (α = 2), **2620 (α = 4), 2581 (α = 8)**, 3079 (α = 16), 3539 (α = 32); bei α = 2 beliebige Knotenwahl 3862 – die Knotenwahl entscheidet kaum. |
| Wie gut ist der Fluss nach Phase k? | ✅ Mehrkosten gegen das Optimum nach Phase 1 bis 5: 3,0 / 1,7 / 0,56 / 0,14 / 0,01 % im Mittel (Median 2,2 / 1,15 / 0,1 / 0 / 0 %, größter Wert 16,5 / 8,8 / 5,2 / 2,4 / 0,4 %); schon optimal: 13 / 19 / 46 / 79 / 98 % der Netze. |
| Was spart eine Toleranz? | ✅ Anhalten, sobald kein Kreis mehr als t je Kante spart: t = 1: 3,9 Phasen, **1881 statt 3737 Kanten**, 0,15 % Mehrkosten (77 % optimal); t = 2: 1438 Kanten, 0,69 %; t = 3: 1061, 1,7 %; t = 5: nach der ersten Phase, 583 Kanten (etwa so viele wie SSP mit 539), 3,0 % (13 % optimal). |
| Sinken die Kosten monoton? | ❌ Nein: in **60 von 100 Netzen** wird der Fluss in mindestens einer Phase wieder teurer. Seed 38: Mehrkosten 7,8 % → 1,7 % → 4,2 % → 0. Eine feinere Phase garantiert nur die feinere Schranke, nicht den billigeren Fluss. |

## Was nicht funktioniert hat / Vorab-Hypothesen

Vor dem Schreiben der Texte wurde über die 100 Netze gemessen; einige Vermutungen aus dem Plan stimmten nicht oder nur teilweise:

- **„Cost Scaling verliert gegen SSP, und ein Kreuzungspunkt liegt nur bei riesigen Kostenspannen.“** Halb richtig: es verliert auf den kleinen Netzen um das Siebenfache, aber der Kreuzungspunkt liegt bei der **Netzgröße**, nicht bei der Kostenspanne (166 bis 306 Knoten). Größere Kosten machen Cost Scaling schlechter (Kosten × 1000: 8570 statt 3829 Kanten), SSP bleibt bei 570 – eine große Kostenspanne dürfte den Kreuzungspunkt also eher nach oben schieben (nicht gemessen).
- **„Die Mehrkosten sind durch n·ε beschränkt.“** Falsch, und im Plan so notiert: ε-Optimalität beschränkt den **Mittelwert der Kosten je Kante auf jedem Kreis** (≥ −ε), nicht die absoluten Mehrkosten des Flusses – die hängen davon ab, wie viel Fluss sich umleiten ließe. Deshalb zeigt die Demo gemessene Mehrkosten und die Toleranz je Kante, keine Schranke in Kosteneinheiten.
- **„Die Kosten fallen von Phase zu Phase.“** In 60 % der Netze nicht (siehe Tabelle).
- **„α = 2 ist die natürliche Wahl.“** Bei den durchsuchten Kanten sind α = 4 bis 8 um ein Drittel besser als α = 2 (2581 bis 2620 gegen 3737); α = 2 zeigt dafür die klarste Abfolge (8 Phasen).
- **„Größere Netze brauchen mehr Phasen.“** Nur logarithmisch: von 12 auf 458 Knoten 7 auf 13 Phasen bei α = 2, während SSP von 3 auf 401 Runden wächst.
- **Fund beim Bau (Implementierung):** Der Zielknoten T startet mit Überschuss −F, darf aber auch **positiven** Überschuss bekommen, wenn zu viel Fluss dort ankommt (Nachfragekanten haben Kosten 0). Wurde T nie entladen, endeten 34 von 100 Netzen nicht im Optimum von SSP (in den geprüften Fällen kamen 3 bis 11 Einheiten zu viel in T an). Jeder Knoten mit positivem Überschuss muss aktiv sein, auch T; seither gilt die Überschuss-Bilanz je Bild als Test.
- **Abweichungen vom Plan:** Port 8679; kein PDF-Export; die Toleranz je Kante ersetzt den geplanten „Abbruch-ε“; die Experimente 5 und 6 des Plans (Zuordnung, Anteil Relabels/Pushes) sind in den Experimenten „Zuordnung“ und der Kennzahl „Entladungen“ enthalten.

## Was die Demo zeigt

- **Phasen und Entladungen:** zwei Slider: die **Phase** (Start, Phase 1 bis P, Beweis) und die **Entladung** innerhalb der Phase (Sättigen, dann ein Bild je Entladung). Links das Netz (Knotenfarbe = Preis, roter Ring mit Menge = Überschuss, schwarzer Ring = entladener Knoten, Pushes grün, Pushes über Rückkanten orange gestrichelt), rechts ε je Phase (logarithmisch) und die Kosten am Phasenende gegen das Optimum. ▶️ zeigt die Phasenenden. Im Endbild der Beweis: bei ε = 1 gibt es keinen negativen Kreis; unabhängig bestätigen Potenziale aus Bellman-Ford alle Restkanten.
- **Vom groben zum feinen Fluss:** Menge, Phasen, Entladungen (Pushes, Relabels), durchsuchte Kanten gegen SSP; Verteilung über 100 feste Netze (Histogramm mit der Marke „Ihre Ziehung“).
- **Experimente (🔬):** α (Phasen, Entladungen, durchsuchte Kanten), Näherung nach Phase k und Toleranz, Kostenspanne (Kosten × 1000), Kreuzungspunkt gegen SSP (12 bis 458 Knoten), Zuordnung n × n bis 120 × 120.
- **Anhalten bei Toleranz:** ein zulässiger, aber nicht sicher optimaler Fluss; das Endbild sagt ehrlich, dass der Restgraph noch negative Kreise enthält.
- **Feste Netze** (Umweg: nach Phase 1 fließt noch alles über den teuren Weg für 9, erst danach über den Umweg für 2; Zuordnung mit Kosten) und zufällige Distributionsnetze; **Wo die Annahmen enden:** was Cost Scaling ohne Heuristiken der Praxis leistet.

## Modell und Verfahren

- **Netz und Restgraph:** wie in der SSP-Demo (Quelle S, Werke, Verteilzentren als Eingang und Ausgang gespalten, Filialen, Senke T; Restkanten als Paar 2i/2i+1), Kosten je Einheit c ≥ 0. Ganzzahlig, eigener Zufallsgenerator SplitMix64 statt `numpy.random`.
- **Modell:** Angebot F in S, Nachfrage F in T (F = größter Fluss), Kosten mit n + 1 multipliziert; Pseudofluss mit Überschüssen e(v), Preise p(v) (Start 0), reduzierte Kosten c_p(u, v) = c(u, v) + p(u) − p(v).
- **Phase (refine):** ε ← max(1, ⌈ε/α⌉); alle Restkanten mit c_p < 0 sättigen; solange ein Knoten mit e > 0 existiert (Warteschlange oder kleinster Index): push über zulässige Kanten (c_p < 0, Zeigerliste), sonst relabel p(v) −= min c_p + ε. ε₀ ist die größte skalierte Kostenzahl (der leere Fluss ist dafür ε₀-optimal).
- **Warum ε = 1 genügt:** ein Kreis im Restgraphen hat höchstens n Kanten, also Kosten ≥ −n·ε; mit Kosten mal n + 1 wäre ein negativer Kreis mindestens −(n + 1).
- **Toleranz t:** anhalten nach der ersten Phase mit ε ≤ (n + 1)·t – kein Kreis spart dann mehr als t je Kante.
- **Aufwand:** durchsuchte Kanten (jede in einer Adjazenzliste angesehene Restkante, auch beim Sättigen und Anheben), nie Sekunden. Kein Price Update und keine Look-Ahead-Regeln – die Heuristiken produktiver Löser fehlen.
- **Laufzeit:** O(n³ log(nC)) (FIFO), unabhängig vom Flusswert.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `csc_constants.py` | Regler-Grenzen, Presets und Hilfetexte, feste Seed-Mengen |
| `csc_presets.py` | Permalink, Preset- und Zufalls-Seed-Logik (Standardmuster des Portfolios) |
| `csc_scenario.py` | Distributionsnetz mit Kosten, eigener Zufallsgenerator, Lehrnetze (Umweg, Zuordnung mit n), Kosten- und Kapazitäts-Skalierung |
| `csc_algorithm.py` | Cost Scaling: Sättigen, push, relabel, Zeigerliste, Phasen, Toleranz, Trace je Entladung |
| `csc_edmonds_karp.py`, `csc_ssp.py` | Kopien der Vorgänger-Demos (größter Fluss; Optimum, Zertifikat und Vergleichsbasis), ohne Import, durch Tests bewacht |
| `csc_evaluation.py` | Urteil, Verteilungen, α-Tabelle, Toleranz-Tabelle, Kostenspanne, Skalierung und Kreuzungspunkt, Zuordnung, Optimalitätsprüfung |
| `csc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt für Touch-Geräte; Kantenbeschriftungen als Annotationen mit heller Hinterlegung; Hover über unsichtbare Marker entlang der Kanten) |
| `tests/` | Algorithmus (Handfälle, `networkx`/`scipy`/Brute Force als Gegenprobe, Invarianten je Bild, ε-optimal gegen die Aufzählung aller Kreise, Präfix-Eigenschaft, Skalierungsinvarianzen, unzulässige Menge), Szenario und Auswertung, Presets, belegte Zahlen, AppTest-Rauchtests |

Alle Daten sind synthetisch; die Laufzeit braucht nur numpy, pandas, plotly und streamlit (scipy und networkx sind reine Testorakel).

## Lokal starten

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -v
```

Die Logik rechnet ausschließlich mit ganzen Zahlen; die im Text genannten Anteile und Mediane sind deshalb auf jeder Plattform identisch.
Die CI (`.github/workflows/tests.yml`) läuft auf Ubuntu mit Python 3.12, bei jedem Push und wöchentlich mit den jeweils neuesten Bibliotheksversionen.
