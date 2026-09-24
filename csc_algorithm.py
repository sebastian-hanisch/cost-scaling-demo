"""Cost Scaling (Goldberg und Tarjan 1990): kostenminimaler Fluss durch Push-Relabel auf reduzierten Kosten und schrittweise feineres epsilon.

Gegeben ein Netz mit Kapazitäten und ganzzahligen Kosten und eine Menge F (hier: der größte Fluss). Modell als Transshipment: Angebot F in S, Nachfrage F in T. Kosten werden mit n+1 multipliziert (n = Knotenzahl),
damit epsilon = 1 genügt. Zu Preisen p (Potenzialen) sind die reduzierten Kosten einer Restkante c_p(u,v) = c(u,v) + p(u) - p(v); ein Pseudofluss ist **epsilon-optimal**, wenn c_p >= -epsilon auf allen Restkanten gilt.
Jede Restkante mit c_p < 0 heißt zulässig (Push erlaubt).

Ablauf: Start mit Fluss 0 und Preisen 0 (epsilon_0 = größte skalierte Kosten, so ist der leere Pseudofluss epsilon_0-optimal). Dann Phasen: epsilon <- max(1, ceil(epsilon / alpha)) und **refine**:
1. alle Restkanten mit c_p < 0 werden gesättigt (der Pseudofluss ist danach 0-optimal, es entstehen Überschüsse),
2. aktive Knoten (Überschuss > 0) werden entladen: push über zulässige Kanten (Zeigerliste), gibt es keine, wird der Knoten angehoben: p(v) -= min c_p + epsilon (dann hat er eine zulässige Kante mit c_p = -epsilon).
Am Ende jeder Phase ist der Fluss zulässig und epsilon-optimal; bei epsilon = 1 (skalierte Kosten) ist er kostenminimal. Zwischenphasen sind Näherungen: die Mehrkosten sind höchstens n * epsilon / (n+1) in Originaleinheiten.

Restkanten wie in den Vorgänger-Demos: Kante 2i ist die Vorwärtskante der Netzkante i (Rest = Kapazität - Fluss, Kosten c), Kante 2i+1 die Rückkante (Rest = Fluss, Kosten -c).
Ein Knoten ist aktiv, sobald sein Überschuss positiv ist - auch T, wenn zu viel Fluss dort ankommt (T startet mit Überschuss -F und darf nie über 0 bleiben).
Aufwand wird in durchsuchten Kanten gezählt (jede in einer Adjazenzliste angesehene Restkante, auch beim Sättigen und Anheben), nie in Sekunden.
"""

from collections import deque
from dataclasses import dataclass

from csc_edmonds_karp import _adjacency

SELECTIONS = ("fifo", "generic")
ALPHAS = (2, 4, 8, 16)


@dataclass(frozen=True)
class Frame:
    kind: str              # 'start' | 'saturate' | 'discharge'
    phase: int             # 0 = Start, sonst Nummer der Phase (1-basiert)
    eps: int
    node: object           # entladener Knoten (nur discharge)
    pushes: tuple          # ((Restkante, Menge), ...)
    relabels: tuple        # ((alter Preis, neuer Preis), ...)
    excess: tuple
    prices: tuple
    flow: tuple            # Fluss je Netzkante (Pseudofluss)
    scanned: int


@dataclass(frozen=True)
class Phase:
    eps: int
    saturated: int         # Restkanten, die zu Beginn gesättigt wurden
    discharges: int
    pushes: int
    sat_pushes: int        # davon sättigende Pushes (in Entladungen)
    relabels: int
    scanned: int
    cost: int              # Kosten des Flusses am Phasenende (Originaleinheiten)
    min_rc: int            # kleinste reduzierte Kosten einer Restkante am Phasenende (skalierte Einheiten)
    first_frame: int       # Index des ersten Bildes der Phase (Sättigen) in Result.frames
    last_frame: int        # Index des letzten Bildes der Phase
    flow: tuple            # Fluss am Phasenende (leer ohne Trace)


@dataclass(frozen=True)
class Result:
    phases: tuple
    frames: tuple
    flow: tuple
    prices: tuple
    value: int
    total: int             # Kosten des Flusses (Originaleinheiten)
    scale: int             # n + 1
    eps0: int
    alpha: int
    selection: str
    stop_eps: int
    scanned_total: int

    @property
    def discharges(self):
        return sum(p.discharges for p in self.phases)

    @property
    def pushes(self):
        return sum(p.pushes for p in self.phases)

    @property
    def relabels(self):
        return sum(p.relabels for p in self.phases)

    @property
    def final_eps(self):
        return self.phases[-1].eps if self.phases else self.eps0


def eps_sequence(eps0, alpha):
    """epsilon je Phase: ceil(eps / alpha), mindestens 1, bis 1 erreicht ist (mindestens eine Phase)."""
    out, eps = [], eps0
    while True:
        eps = max(1, -(-eps // alpha))
        out.append(eps)
        if eps == 1:
            return out


def cost_scaling(net, value, alpha=2, selection="fifo", keep_trace=True, stop_eps=1):
    """Kostenminimaler Fluss der Menge `value` (muss zulässig sein). `stop_eps` > 1: nach der ersten Phase mit epsilon <= stop_eps anhalten (Näherung, Negativkontrolle 'zu früh')."""
    if selection not in SELECTIONS:
        raise ValueError(selection)
    if alpha < 2:
        raise ValueError(alpha)
    adj, head = _adjacency(net)
    n, m, s, t = net.n, net.m, net.s, net.t
    scale = n + 1
    cost = [0] * (2 * m)
    res = [0] * (2 * m)
    for i, (_, _, cap, c, _) in enumerate(net.arcs):
        cost[2 * i], cost[2 * i + 1] = scale * c, -scale * c
        res[2 * i] = cap
    eps0 = max(1, max((abs(c) for c in cost), default=1))
    price = [0] * n
    excess = [0] * n
    excess[s], excess[t] = value, -value
    frames, phases = [], []
    scanned_total = 0

    def flow_now():
        return tuple(res[2 * i + 1] for i in range(m))

    def add_frame(kind, phase, eps, node, pushes, relabels, scanned):
        if keep_trace:
            frames.append(Frame(kind, phase, eps, node, tuple(pushes), tuple(relabels), tuple(excess), tuple(price), flow_now(), scanned))

    add_frame("start", 0, eps0, None, (), (), 0)
    eps = eps0
    while True:
        eps = max(1, -(-eps // alpha))
        phase_no = len(phases) + 1
        first_frame = len(frames)
        scanned = discharges = pushes = sat_pushes = relabels = 0
        # 1. alle Restkanten mit negativen reduzierten Kosten sättigen
        sat, sat_list = 0, []
        for u in range(n):
            for e in adj[u]:
                scanned += 1
                if res[e] > 0 and cost[e] + price[u] - price[head[e]] < 0:
                    v, r = head[e], res[e]
                    res[e] = 0
                    res[e ^ 1] += r
                    excess[u] -= r
                    excess[v] += r
                    sat += 1
                    sat_list.append((e, r))
        add_frame("saturate", phase_no, eps, None, sat_list, (), scanned)
        # 2. aktive Knoten entladen
        cur = [0] * n
        fifo = deque(v for v in range(n) if excess[v] > 0)
        queued = [excess[v] > 0 for v in range(n)]
        while fifo if selection == "fifo" else any(queued):
            if selection == "fifo":
                u = fifo.popleft()
            else:
                u = next(v for v in range(n) if queued[v])
            queued[u] = False
            d_scanned, d_pushes, d_relabels = 0, [], []
            while excess[u] > 0:
                while cur[u] < len(adj[u]):
                    e = adj[u][cur[u]]
                    d_scanned += 1
                    v = head[e]
                    if res[e] > 0 and cost[e] + price[u] - price[v] < 0:
                        delta = min(excess[u], res[e])
                        res[e] -= delta
                        res[e ^ 1] += delta
                        excess[u] -= delta
                        excess[v] += delta
                        d_pushes.append((e, delta))
                        if res[e] == 0:
                            sat_pushes += 1
                        if not queued[v] and excess[v] > 0:
                            queued[v] = True
                            if selection == "fifo":
                                fifo.append(v)
                        if excess[u] == 0:
                            break
                    cur[u] += 1
                if excess[u] == 0:
                    break
                if relabels + len(d_relabels) > 6 * n * n + 100:
                    raise ValueError("Die Menge ist nicht zulässig: die Preise fallen ohne Ende")
                # relabel: keine zulässige Kante mehr
                best = None
                for e in adj[u]:
                    d_scanned += 1
                    if res[e] > 0:
                        c = cost[e] + price[u] - price[head[e]]
                        if best is None or c < best:
                            best = c
                if best is None:
                    raise ValueError("Knoten mit Überschuss ohne Restkante: die Menge ist nicht zulässig")
                old = price[u]
                price[u] -= best + eps
                d_relabels.append((old, price[u]))
                cur[u] = 0
            discharges += 1
            pushes += len(d_pushes)
            relabels += len(d_relabels)
            scanned += d_scanned
            add_frame("discharge", phase_no, eps, u, d_pushes, d_relabels, d_scanned)
        scanned_total += scanned
        flow = flow_now()
        min_rc = min((cost[e] + price[head[e ^ 1]] - price[head[e]] for e in range(2 * m) if res[e] > 0), default=0)
        phases.append(Phase(eps, sat, discharges, pushes, sat_pushes, relabels, scanned, sum(flow[i] * net.arcs[i][3] for i in range(m)), min_rc, first_frame, len(frames) - 1 if keep_trace else -1, flow if keep_trace else ()))
        if eps == 1 or eps <= stop_eps:
            break
    flow = flow_now()
    return Result(tuple(phases), tuple(frames), flow, tuple(price), value, sum(flow[i] * net.arcs[i][3] for i in range(m)), scale, eps0, alpha, selection, stop_eps, scanned_total)


def flow_cost(net, flow):
    return sum(flow[i] * net.arcs[i][3] for i in range(net.m))
