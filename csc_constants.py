"""Konstanten, Regler-Grenzen, Presets und feste Seed-Mengen der Demo "Cost Scaling: erst grob, dann fein"."""

# --- Regler (wie in den Vorgänger-Demos) ----------------------------------------------------------------------------------------
P_MIN, P_MAX, DEFAULT_P = 2, 6, 3            # Werke
D_MIN, D_MAX, DEFAULT_D = 2, 6, 3            # Verteilzentren
S_MIN, S_MAX, DEFAULT_S = 3, 12, 8           # Filialen
DENSITY_MIN, DENSITY_MAX, DEFAULT_DENSITY = 20, 100, 60   # Anteil vorhandener Lanes in ganzen Prozent, Schritt 10
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0, 100, 50       # Streuung der Lane-Breiten in ganzen Prozent, Schritt 25
LOAD_MIN, LOAD_MAX, DEFAULT_LOAD = 40, 160, 90            # Gesamtnachfrage in Prozent der Werkskapazität, Schritt 10
DEFAULT_SEED = 155
SEED_MAX = 2_000_000_000

NETS = {
    "random": "Zufälliges Distributionsnetz",
    "detour": "Umweg (der kurze Weg ist teuer)",
    "assignment": "Zuordnung mit Kosten (5 × 5, Ungarische Methode)",
}
DEFAULT_NET = "random"
FIXED_NETS = ("detour", "assignment")

ALPHAS = (2, 4, 8, 16)
DEFAULT_ALPHA = 2
ALPHA_TABLE = (2, 4, 8, 16, 32)
SELECTION_LABELS = {"fifo": "Warteschlange (FIFO)", "generic": "Beliebig (kleinster Index)"}
DEFAULT_SELECTION = "fifo"
TOLERANCES = (0, 1, 2, 3, 5)               # Toleranz je Kante in Kosteneinheiten: anhalten, sobald kein Kreis im Restgraphen mehr als so viel je Kante spart (0 = bis zum Optimum)
DEFAULT_TOLERANCE = 0

# --- feste Seed-Mengen (dieselben wie in der Edmonds-Karp-Demo; unabhängig vom Nutzer-Seed) -----------------------------------------
DIST_SEEDS = tuple(range(100000, 100100))
SWEEP_SEEDS = DIST_SEEDS[:40]
SCALE_SIZES = ((2, 2, 4), (3, 3, 8), (4, 4, 16), (6, 6, 32), (8, 8, 64), (12, 12, 128), (16, 16, 256), (24, 24, 384))   # (Werke, DCs, Filialen)
SCALE_SEEDS = DIST_SEEDS[:3]
COST_FACTORS = (1, 10, 100, 1000)
COST_SEEDS = DIST_SEEDS[:20]
ASSIGN_SIZES = (5, 10, 20, 40, 60, 80, 120)
ASSIGN_SEEDS = (7, 8, 9)

COLORS = {
    "flow": "#1f77b4", "path": "#2ca02c", "back": "#ff7f0e", "cut": "#d62728", "reach": "#2ca02c", "dead": "#9467bd",
    "unreach": "#8c8c8c", "faint": "rgba(150,150,150,0.45)", "node": "#111111", "optimal": "#d62728", "levels": "Viridis",
}

# --- Presets -----------------------------------------------------------------------------------------------------------------
_BASE = dict(net="random", alpha=DEFAULT_ALPHA, selection=DEFAULT_SELECTION, tol=DEFAULT_TOLERANCE, p=DEFAULT_P, d=DEFAULT_D, s=DEFAULT_S, density=DEFAULT_DENSITY, spread=DEFAULT_SPREAD, load=DEFAULT_LOAD, seed=DEFAULT_SEED)
PRESETS = {
    "🚚 Zufallsnetz": {**_BASE},
    "⚡ α = 4": {**_BASE, "alpha": 4},
    "🎚️ Grobe Toleranz": {**_BASE, "tol": 5, "seed": 38},
    "🔀 Beliebige Knotenwahl": {**_BASE, "selection": "generic"},
    "🏭 Werke knapp": {**_BASE, "load": 140},
    "🕸️ Dünnes Netz": {**_BASE, "density": 30},
    "🧭 Umweg": {**_BASE, "net": "detour"},
    "💑 Zuordnung mit Kosten": {**_BASE, "net": "assignment"},
}
PRESET_HELP = {
    "🚚 Zufallsnetz": "Das Netz der SSP- und Cycle-Canceling-Demo (Seed 155): 8 Phasen von ε = 90 bis 1, 690 Entladungen, 5190 durchsuchte Kanten - SSP braucht für dasselbe Netz 824. Der Fluss kostet nach der ersten Phase 1199 statt 1197.",
    "⚡ α = 4": "ε wird je Phase durch 4 statt 2 geteilt: nur 4 statt 8 Phasen und 3648 statt 5190 durchsuchte Kanten. Über 100 Netze ist α = 4 (2620) und α = 8 (2581) besser als α = 2 (3737); α = 16 (3079) schon wieder schlechter.",
    "🎚️ Grobe Toleranz": "Seed 38, Anhalten bei Toleranz 5 je Kante: schon nach der ersten Phase (ε = 90) liegt ein zulässiger Fluss vor - er kostet 1027 statt 953 (+7,8 %), die Suche hat 590 Kanten durchsucht statt 3737 für den ganzen Lauf. Der Fluss wird danach billiger, aber nicht monoton: 7,8 % → 1,7 % → 4,2 % → 0.",
    "🔀 Beliebige Knotenwahl": "Statt der Warteschlange wird stets der Knoten mit dem kleinsten Index entladen: dieselben Kosten, 4907 statt 5190 durchsuchte Kanten. Die Knotenwahl entscheidet hier kaum etwas.",
    "🏭 Werke knapp": "Nachfrage 140 % der Werkskapazität: das Netz schafft nur 89 von 118 Einheiten; Cost Scaling bekommt F = 89 aus dem größten Fluss und liefert die billigsten 89 (1403).",
    "🕸️ Dünnes Netz": "Nur 30 % der möglichen Lanes: weniger Kanten, 3031 durchsuchte Kanten gegen 353 bei SSP.",
    "🧭 Umweg": "Vier Knoten: nach der ersten Phase (ε = 23) fließt die Einheit noch über den teuren Weg S-X-T (Kosten 9); erst die feineren Phasen finden den Umweg S-X-A-T für 2. Neun Entladungen, 84 durchsuchte Kanten gegen 7 bei SSP.",
    "💑 Zuordnung mit Kosten": "5 Fahrzeuge, 5 Aufträge, Kosten 1 bis 9: Cost Scaling braucht 7 Phasen und 1147 durchsuchte Kanten, SSP (die Ungarische Methode) 150. Erst zwischen n = 60 und n = 80 gewinnt Cost Scaling - siehe Experiment „Zuordnung“.",
}
