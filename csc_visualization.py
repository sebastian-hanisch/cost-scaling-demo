"""Plotly-Abbildungen: Netz mit Preisen, Überschüssen und Pushes, ε und Kosten je Phase, Näherungsgüte, Verteilungen, Aufwand und Kreuzungspunkt.
Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen. Kanten haben über unsichtbare Marker einen Hover-Text
(Plotly-Linien reagieren nur an ihren Stützpunkten)."""

from math import atan2, degrees, hypot

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import csc_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _layout(fig, net, height):
    xs = [p[0] for p in net.pos]
    ys = [p[1] for p in net.pos]
    pad = 9
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    return _base(fig, height)


def _curve(p0, p1, bulge, steps=8):
    """Punkte von p0 nach p1; mit `bulge` > 0 als flacher Bogen nach rechts (so trennen sich Vorwärts- und Rückkante). Dazu der Pfeilwinkel bei 65 %."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = hypot(dx, dy) or 1.0
    cx, cy = (x0 + x1) / 2 + bulge * length * dy / length, (y0 + y1) / 2 - bulge * length * dx / length
    ts = [k / steps for k in range(steps + 1)]
    xs = [(1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1 for t in ts]
    ys = [(1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1 for t in ts]
    t = 0.65
    tx = 2 * (1 - t) * (cx - x0) + 2 * t * (x1 - cx)
    ty = 2 * (1 - t) * (cy - y0) + 2 * t * (y1 - cy)
    ax = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t * t * x1
    ay = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t * t * y1
    return xs, ys, (ax, ay, degrees(atan2(tx, ty))), (xs[steps // 2], ys[steps // 2])


def _segments(curves):
    x, y = [], []
    for xs, ys, _, _ in curves:
        x += xs + [None]
        y += ys + [None]
    return x, y


def _lines(fig, curves, color, width, name, dash=None, showlegend=True):
    if not curves:
        return
    x, y = _segments(curves)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", name=name, showlegend=showlegend))


def _arrows(fig, curves, color, size=9):
    if not curves:
        return
    fig.add_trace(go.Scatter(x=[c[2][0] for c in curves], y=[c[2][1] for c in curves], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(symbol="arrow", size=size, color=color, angle=[c[2][2] for c in curves])))


def _hover_points(fig, net, entries):
    """Unsichtbare Marker entlang jeder Kante, damit der Hover-Text überall auf der Kante erscheint. entries: [(Kurve, Text)]"""
    x, y, text = [], [], []
    for curve, label in entries:
        xs, ys = curve[0], curve[1]
        for k in range(1, len(xs) - 1):
            x.append(xs[k]); y.append(ys[k]); text.append(label)
    if x:
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=9, opacity=0), hovertext=text, hoverinfo="text", showlegend=False))


def _labels(fig, points):
    """points: [(x, y, Text)] - als Annotationen mit heller Hinterlegung, damit sie Kanten, Pfeile und Knotenbeschriftungen nicht unlesbar machen."""
    for x, y, text in points:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", font=dict(size=11, color="#111"), bgcolor="rgba(255,255,255,0.88)", borderpad=1)


def _arc_name(net, i):
    u, v = net.arcs[i][0], net.arcs[i][1]
    return f"{net.names[u]} → {net.names[v]}"


def _nodes(fig, net, reach=None):
    """Knoten: S und T als Quadrate, alle anderen als Kreise; mit `reach` grün (von S erreichbar) oder grau eingefärbt."""
    text_pos = {0: "top center", 1: "bottom center"}
    for kind, idx in (("Quelle/Senke", [net.s, net.t]), ("Knoten", [v for v in range(net.n) if v not in (net.s, net.t)])):
        colors = [C.COLORS["node"] if reach is None else (C.COLORS["reach"] if reach[v] else C.COLORS["unreach"]) for v in idx]
        pos = [text_pos.get(v, "top center" if net.pos[v][1] > 70 else ("bottom center" if net.pos[v][1] < 30 else "middle left")) for v in idx]
        if net.logistic and kind == "Knoten":
            pos = ["top center" if net.names[v].startswith("Werk") else "bottom center" if net.names[v].startswith("Filiale") else "middle left" for v in idx]
        fig.add_trace(go.Scatter(
            x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", showlegend=False,
            text=[net.labels[v] for v in idx], textposition=pos, hovertext=[net.names[v] for v in idx], hoverinfo="text",
            marker=dict(symbol="square" if kind == "Quelle/Senke" else "circle", size=13 if kind == "Quelle/Senke" else 10, color=colors, line=dict(width=1.5, color="#333"))))


def _wscale(net):
    return max(c for _, _, c, _, _ in net.arcs)


def _width(amount, top, lo=1.0, hi=6.0):
    return lo + (hi - lo) * amount / top if top else lo


def _node_text_positions(net, idx):
    if net.logistic:
        return ["top center" if (v == 0 or net.names[v].startswith("Werk")) else "bottom center" if (v == 1 or net.names[v].startswith("Filiale")) else "middle left" for v in idx]
    return ["top center" if v == net.s else "bottom center" if v == net.t else "middle left" for v in idx]


def _colorbar(lo, hi):
    return dict(title=dict(text="Schattenpreis π", side="top"), orientation="h", thickness=9, len=0.6, x=0.5, xanchor="center", y=-0.02, yanchor="top", tickmode="linear", tick0=lo, dtick=max(1, -(-(hi - lo) // 4)))


def build_network(net, flow, pi=None, path=None, height=460):
    """Netz mit Fluss: Breite ~ Fluss (dunkelblau = Kante voll, blass = ungenutzt), Knotenfarbe ~ Schattenpreis π (falls gegeben).
    `path`: [(Netzkante, vorwärts?, Menge)] des gelöschten Kreises - grün (vorwärts, +Menge) bzw. orange gestrichelt (Rückkante, −Menge), beschriftet mit den Kosten je Einheit.
    Bei kleinen Lehrnetzen trägt jede Kante Fluss/Kapazität und Kosten."""
    fig = go.Figure()
    top = _wscale(net)
    path_map = {i: (fwd, amount) for i, fwd, amount in (path or ())}
    groups = {"idle": [], "part": [], "full": []}
    fwd_path, back_path, hover, labels = [], [], [], []
    for i, (u, v, cap, cost, _) in enumerate(net.arcs):
        curve = _curve(net.pos[u], net.pos[v], 0.0)
        rc = "" if pi is None else f", reduzierte Kosten {cost + pi[u] - pi[v]}"
        hover.append((curve, f"{_arc_name(net, i)}: Fluss {flow[i]} von {cap}, Kosten {cost} je Einheit{rc}"))
        if i in path_map:
            fwd, amount = path_map[i]
            (fwd_path if fwd else back_path).append(curve)
            labels.append((curve[3][0] + 1.5, curve[3][1], f"{'+' if fwd else '−'}{amount} × {cost if fwd else -cost}"))
        else:
            groups["idle" if flow[i] == 0 else "full" if flow[i] == cap else "part"].append((curve, flow[i]))
            if net.m <= 12:
                labels.append((curve[3][0] + 1.5, curve[3][1], f"{flow[i]}/{cap} · {cost}"))
    _lines(fig, [c for c, _ in groups["idle"]], C.COLORS["faint"], 1.2, "ungenutzt")
    for group, color, name in (("part", "rgba(31,119,180,0.85)", "Fluss (nicht voll)"), ("full", "#0b3d91", "Fluss (Kante voll)")):
        by_width = {}
        for c, f in groups[group]:
            by_width.setdefault(round(_width(f, top)), []).append(c)      # ganze Breiten: wenige Spuren statt einer je Kante
        for w, curves in by_width.items():
            _lines(fig, curves, color, w, name, showlegend=False)
        if groups[group]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, width=4), name=name))
    _lines(fig, fwd_path, C.COLORS["path"], 5.5, "Kreis: Kante vorwärts (+Menge × Kosten je Einheit)")
    _lines(fig, back_path, C.COLORS["back"], 5.5, "Kreis: Rückkante (nimmt Fluss zurück, spart die Kosten)", dash="dash")
    if net.m <= 80:
        _arrows(fig, [c for c, _ in groups["idle"]] + [c for c, _ in groups["part"]] + [c for c, _ in groups["full"]], "rgba(60,60,60,0.7)", 8)
    _arrows(fig, fwd_path + back_path, "rgba(30,30,30,0.9)", 11)
    _hover_points(fig, net, hover)
    _labels(fig, labels)
    idx = list(range(net.n))
    marker = dict(symbol=["square" if v in (net.s, net.t) else "circle" for v in idx], size=[13 if v in (net.s, net.t) else 11 for v in idx], line=dict(width=1.5, color="#333"))
    if pi is None:
        marker["color"] = C.COLORS["node"]
        hover_nodes = [net.names[v] for v in idx]
    else:
        lo_pi, hi_pi = min(0, min(pi)), max(1, max(pi))
        marker.update(color=list(pi), colorscale=C.COLORS["levels"], cmin=lo_pi, cmax=hi_pi, showscale=True, colorbar=_colorbar(lo_pi, hi_pi))
        hover_nodes = [f"{net.names[v]}: Schattenpreis π = {pi[v]}" for v in idx]
    fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", showlegend=False, text=[net.labels[v] for v in idx],
                             textposition=_node_text_positions(net, idx), hovertext=hover_nodes, hoverinfo="text", marker=marker))
    fig = _layout(fig, net, height)
    if pi is not None:
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=90), legend=dict(orientation="h", y=-0.3))
    return fig


def _price_colorbar(lo, hi):
    return dict(title=dict(text="Preis p (skaliert)", side="top"), orientation="h", thickness=9, len=0.6, x=0.5, xanchor="center", y=-0.02, yanchor="top", tickmode="linear", tick0=lo, dtick=max(1, -(-(hi - lo) // 4)))


def build_state(net, frame, scale, height=460):
    """Netz nach einer Entladung: Knotenfarbe = Preis p (skaliert), roter Ring = Knoten mit Überschuss (Größe ~ Menge, beschriftet), dicker schwarzer Ring = der entladene Knoten,
    T mit Nachfrage (negativer Überschuss) als Raute; Fluss (Pseudofluss) wie sonst (Breite ~ Fluss); die Pushes der Entladung grün mit Menge (orange gestrichelt: Push über eine Rückkante nimmt Fluss zurück)."""
    fig = go.Figure()
    flow, price, excess = frame.flow, frame.prices, frame.excess
    top = _wscale(net)
    push_arcs = {}
    for e, amount in frame.pushes:
        push_arcs[e // 2] = (e % 2 == 0, push_arcs.get(e // 2, (True, 0))[1] + amount)
    groups = {"idle": [], "part": [], "full": []}
    fwd_push, back_push, hover, labels = [], [], [], []
    for i, (u, v, cap, cost, _) in enumerate(net.arcs):
        curve = _curve(net.pos[u], net.pos[v], 0.0)
        rc = scale * cost + price[u] - price[v]
        hover.append((curve, f"{_arc_name(net, i)}: Fluss {flow[i]} von {cap}, Kosten {cost} je Einheit; reduzierte Kosten {rc} (skaliert)"))
        if i in push_arcs:
            forward, amount = push_arcs[i]
            (fwd_push if forward else back_push).append(curve)
            labels.append((curve[3][0] + 1.5, curve[3][1], f"{'+' if forward else '−'}{amount}"))
        else:
            groups["idle" if flow[i] == 0 else "full" if flow[i] == cap else "part"].append((curve, flow[i]))
    _lines(fig, [c for c, _ in groups["idle"]], C.COLORS["faint"], 1.2, "ungenutzt")
    for group, color, name in (("part", "rgba(31,119,180,0.85)", "Fluss (nicht voll)"), ("full", "#0b3d91", "Fluss (Kante voll)")):
        by_width = {}
        for c, f in groups[group]:
            by_width.setdefault(round(_width(f, top)), []).append(c)
        for w, curves in by_width.items():
            _lines(fig, curves, color, w, name, showlegend=False)
        if groups[group]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color, width=4), name=name))
    _lines(fig, fwd_push, C.COLORS["path"], 5.5, "Push in dieser Entladung (+Menge)")
    _lines(fig, back_push, C.COLORS["back"], 5.5, "Push über eine Rückkante (nimmt Fluss zurück)", dash="dash")
    if net.m <= 80:
        _arrows(fig, [c for c, _ in groups["idle"]] + [c for c, _ in groups["part"]] + [c for c, _ in groups["full"]], "rgba(60,60,60,0.7)", 8)
    _arrows(fig, fwd_push + back_push, "rgba(30,30,30,0.9)", 11)
    _hover_points(fig, net, hover)
    _labels(fig, labels)
    idx = list(range(net.n))
    lo_p, hi_p = min(0, min(price)), max(1, max(price))
    fig.add_trace(go.Scatter(
        x=[net.pos[v][0] for v in idx], y=[net.pos[v][1] for v in idx], mode="markers+text", showlegend=False, text=[net.labels[v] for v in idx], textposition=_node_text_positions(net, idx),
        hovertext=[f"{net.names[v]}: Preis p = {price[v]}, Überschuss {excess[v]}" for v in idx], hoverinfo="text",
        marker=dict(symbol=["square" if v in (net.s, net.t) else "circle" for v in idx], size=[13 if v in (net.s, net.t) else 11 for v in idx], color=list(price), colorscale=C.COLORS["levels"], cmin=lo_p, cmax=hi_p,
                    showscale=True, colorbar=_price_colorbar(lo_p, hi_p), line=dict(width=1.5, color="#333"))))
    active = [v for v in idx if excess[v] > 0]
    if active:
        fig.add_trace(go.Scatter(x=[net.pos[v][0] for v in active], y=[net.pos[v][1] for v in active], mode="markers", name="Knoten mit Überschuss", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=[17 + min(14, 3 * excess[v] ** 0.5) for v in active], color=C.COLORS["cut"], line=dict(width=2.5))))
        _labels(fig, [(net.pos[v][0] + 2.2, net.pos[v][1] + 3.2, f"+{excess[v]}") for v in active])
    if frame.node is not None:
        v = frame.node
        fig.add_trace(go.Scatter(x=[net.pos[v][0]], y=[net.pos[v][1]], mode="markers", name="entladener Knoten", hoverinfo="skip", marker=dict(symbol="circle-open", size=27, color="#111", line=dict(width=3.5))))
    fig = _layout(fig, net, height)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=90), legend=dict(orientation="h", y=-0.3))
    return fig


def build_eps(eps_list, costs, k, optimum, height=300):
    """Oben ε je Phase (logarithmisch), unten die Kosten des Flusses am Ende jeder Phase gegen das Optimum. Bisherige Phasen voll, kommende blass; k = Zahl der abgeschlossenen Phasen."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.45, 0.55])
    xs = list(range(1, len(eps_list) + 1))
    fig.add_trace(go.Bar(x=xs, y=eps_list, marker_color=[C.COLORS["flow"] if x <= k else C.COLORS["faint"] for x in xs], showlegend=False, hovertext=[f"Phase {x}: ε = {e}" for x, e in zip(xs, eps_list)], hoverinfo="text"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs[:k], y=costs[:k], mode="lines+markers", line=dict(color=C.COLORS["flow"], width=3), marker=dict(size=7), showlegend=False,
                             hovertext=[f"nach Phase {x}: Kosten {c}" for x, c in zip(xs[:k], costs[:k])], hoverinfo="text"), row=2, col=1)
    if k < len(costs):
        fig.add_trace(go.Scatter(x=xs[max(0, k - 1):], y=costs[max(0, k - 1):], mode="lines", line=dict(color=C.COLORS["faint"], width=3), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[optimum, optimum], mode="lines", line=dict(color="#555", dash="dash"), name=f"Optimum (SSP) {optimum}", hoverinfo="skip"), row=2, col=1)
    fig.update_yaxes(title="ε", type="log", row=1, col=1)
    fig.update_yaxes(title="Kosten", range=[optimum * 0.97, max(max(costs), optimum) * 1.03], row=2, col=1)
    fig.update_xaxes(title="Phase", dtick=1 if len(xs) <= 20 else None, row=2, col=1)
    fig = _base(fig, height + 30)
    fig.update_layout(legend=dict(orientation="h", y=-0.22), margin=dict(l=10, r=10, t=10, b=50))
    return fig


def build_anytime(gap_mean, gap_max, share_optimal, height=300):
    """Näherungsgüte nach jeder Phase über die Netze: mittlere und größte Mehrkosten in % (Balken, Punkte) und Anteil der Netze, die schon optimal sind (Linie, rechte Achse)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    xs = list(range(1, len(gap_mean) + 1))
    fig.add_trace(go.Bar(x=xs, y=gap_mean, name="Mehrkosten im Mittel [%]", marker_color=C.COLORS["flow"], opacity=0.8), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=gap_max, mode="markers", name="Mehrkosten, größter Wert [%]", marker=dict(color=C.COLORS["cut"], size=9, symbol="diamond")), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=[100 * x for x in share_optimal], mode="lines+markers", name="schon optimal [% der Netze]", line=dict(color=C.COLORS["path"], width=3)), secondary_y=True)
    fig.update_xaxes(title="nach Phase", dtick=1)
    fig.update_yaxes(title="Mehrkosten [%]", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="optimal [%]", secondary_y=True, range=[0, 100], showgrid=False)
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.35), height=height + 60)
    return fig


def build_ratio_hist(ratios, current=None, height=300):
    """Durchsuchte Kanten von Cost Scaling im Verhältnis zu SSP, je Netz (1 = gleich viele; darunter gewinnt Cost Scaling)."""
    fig = go.Figure(go.Histogram(x=ratios, xbins=dict(size=0.5), marker_color=C.COLORS["back"], opacity=0.85, name="Netze"))
    fig.add_vline(x=1, line=dict(color="#555", dash="dot"))
    if current is not None:
        fig.add_vline(x=current, line=dict(color=C.COLORS["optimal"], dash="dash"), annotation_text="Ihre Ziehung", annotation_position="top")
    fig.update_xaxes(title="durchsuchte Kanten: Cost Scaling ÷ SSP")
    fig.update_yaxes(title="Netze")
    fig = _base(fig, height)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30 if current is not None else 10, b=10))
    return fig


def build_alpha(rows, height=320):
    """Durchsuchte Kanten (Balken, links) und Phasen (Linie, rechts) je Skalierungsfaktor α, Warteschlange und beliebige Knotenwahl."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    alphas = sorted({r["alpha"] for r in rows})
    for sel, name, color in (("fifo", "Warteschlange", "#1f77b4"), ("generic", "beliebig", "#ff7f0e")):
        sub = [r for r in rows if r["selection"] == sel]
        fig.add_trace(go.Bar(x=[str(r["alpha"]) for r in sub], y=[r["scans"] for r in sub], name=name, marker_color=color, opacity=0.85), secondary_y=False)
    fig.add_trace(go.Scatter(x=[str(a) for a in alphas], y=[next(r["phases"] for r in rows if r["alpha"] == a) for a in alphas], mode="lines+markers", name="Phasen", line=dict(color="#2ca02c", width=3)), secondary_y=True)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Skalierungsfaktor α")
    fig.update_yaxes(title="durchsuchte Kanten (Mittel)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Phasen", secondary_y=True, rangemode="tozero", showgrid=False)
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 50)
    return fig


def build_cost_range(rows, height=320):
    """Kosten mal k: durchsuchte Kanten von Cost Scaling (wächst mit log C) und SSP (bleibt gleich); dazu die Phasen."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ks = [str(r["k"]) for r in rows]
    fig.add_trace(go.Scatter(x=ks, y=[r["scans"] for r in rows], mode="lines+markers", name="Cost Scaling", line=dict(color="#d62728", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=[r["ssp_scans"] for r in rows], mode="lines+markers", name="SSP", line=dict(color="#2ca02c", width=3)), secondary_y=False)
    fig.add_trace(go.Bar(x=ks, y=[r["phases"] for r in rows], name="Phasen (rechts)", marker_color="#1f77b4", opacity=0.35), secondary_y=True)
    fig.update_xaxes(title="Kosten × k")
    fig.update_yaxes(title="durchsuchte Kanten (Mittel)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title="Phasen", secondary_y=True, rangemode="tozero", showgrid=False)
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 50)
    return fig


def build_scaling(rows, height=340):
    """Durchsuchte Kanten gegen die Kantenzahl (doppelt logarithmisch): Cost Scaling (α = 2 und α = 4) und SSP; der Kreuzungspunkt liegt dort, wo Cost Scaling unter SSP fällt."""
    fig = go.Figure()
    m = [r["m"] for r in rows]
    for key, label, color, dash in (("cs2", "Cost Scaling (α = 2)", "#d62728", "solid"), ("cs4", "Cost Scaling (α = 4)", "#ff7f0e", "solid"), ("ssp", "Successive Shortest Paths", "#2ca02c", "solid")):
        fig.add_trace(go.Scatter(x=m, y=[r[key] for r in rows], mode="lines+markers", name=label, line=dict(color=color, dash=dash)))
    fig.add_trace(go.Scatter(x=m, y=m, mode="lines", name="Kanten des Netzes", line=dict(color="#555", dash="dashdot")))
    fig.update_xaxes(title="Kanten des Netzes", type="log")
    fig.update_yaxes(title="durchsuchte Kanten", type="log")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.4), height=height + 70)
    return fig


def build_assignment(rows, height=320):
    """Zuordnung n × n: durchsuchte Kanten von Cost Scaling und SSP (SSP braucht n Runden); doppelt logarithmisch über n."""
    fig = go.Figure()
    ns = [r["n"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["cs"] for r in rows], mode="lines+markers", name="Cost Scaling (α = 4)", line=dict(color="#ff7f0e", width=3)))
    fig.add_trace(go.Scatter(x=ns, y=[r["ssp"] for r in rows], mode="lines+markers", name="Successive Shortest Paths", line=dict(color="#2ca02c", width=3)))
    fig.update_xaxes(title="Größe n der Zuordnung (n × n)", type="log")
    fig.update_yaxes(title="durchsuchte Kanten", type="log")
    fig = _base(fig, height)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), height=height + 50)
    return fig
