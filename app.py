"""Cost Scaling - erst grob, dann fein - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Cost Scaling: Push-Relabel auf reduzierten Kosten mit schrittweise feinerem epsilon - und lässt stattdessen das Beispiel wachsen.
Sechstes Stück der Netzwerkfluss-Linie der "Konzepte"-Reihe, Konvergenz aus "Push-Relabel" und der epsilon-Skalierung der Auktion. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import csc_constants as C
import csc_evaluation as ev
import csc_scenario as sc
import csc_ssp as ssp
from csc_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from csc_scenario import build
from csc_visualization import (
    build_alpha,
    build_anytime,
    build_assignment,
    build_cost_range,
    build_eps,
    build_network,
    build_ratio_hist,
    build_scaling,
    build_state,
)

st.set_page_config(page_title="Cost Scaling – Sebastian Hanisch", layout="wide")

SEL_SHORT = {"fifo": "Warteschlange", "generic": "beliebig"}


def _pct(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f} %".replace(".", ",")


def _share(x):
    """Anteil (0..1) als 'nn %'."""
    return f"{100 * x:.0f} %"


def _f(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}".replace(".", ",")


def _int(x):
    return f"{int(round(x)):,}".replace(",", " ")


def _stage_text(stage_caps):
    parts = [f"{sc.KIND_LABELS[k]} {v}" for k, v in stage_caps.items() if k in ev.STAGE_KINDS]
    return ", ".join(parts)


@st.cache_resource(show_spinner=False, max_entries=32)
def _analysis(params, alpha, selection, tol):
    return ev.analyse(build(*params), alpha, selection, tol)


st.title("🪜 Cost Scaling – erst grob, dann fein")
st.markdown(
    """
**Successive Shortest Paths** beweist den billigsten Fluss Weg für Weg, **Cycle-Canceling** Kreis für Kreis - beide verlangen zu jedem Zeitpunkt etwas Exaktes. **Cost Scaling** lockert das: jeder Knoten hat einen **Preis** $p$, und es genügt, dass keine Restkante *mehr* als $\\varepsilon$ zu billig ist ($c_p\\ge-\\varepsilon$, **$\\varepsilon$-Optimalität**).
Es beginnt grob - $\\varepsilon$ so groß wie die größten Kosten - und teilt $\\varepsilon$ in **Phasen** durch $\\alpha$. Jede Phase ist ein **Push-Relabel** wie im dritten Stück, nur auf den reduzierten Kosten: Kanten mit negativen reduzierten Kosten werden gesättigt, die Überschüsse wandern lokal über zulässige Kanten, ein Knoten ohne zulässige Kante hebt seinen Preis.
Am Ende jeder Phase ist der Fluss **zulässig**; bei $\\varepsilon<1$ (Kosten mit $n+1$ multipliziert) ist er **kostenminimal**. Diese Demo zeigt die Phasen und Entladungen, wie gut die Zwischenflüsse schon sind - und ab welcher Netzgröße das Verfahren SSP schlägt: es ist das Verfahren hinter OR-Tools `SimpleMinCostFlow`.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der Netzwerkfluss-Linie der \"Konzepte\"-Reihe, Konvergenz aus \"Push-Relabel\" und der ε-Skalierung der Auktion (Demo \"auction-algorithm\" der Matching-Linie, dort nur für die Zuordnung) - **ein** Verfahren an einem wachsenden Beispiel. "
    "Verwandt: **Cycle-Canceling** (ε-optimal heißt: jeder Kreis im Restgraphen hat im Mittel Kosten ≥ −ε je Kante) und der **Netzwerksimplex** (Fall-Demo \"Distributionsnetzwerk-Optimierung\")."
)

with st.expander("So funktioniert Cost Scaling", expanded=True):
    st.markdown(
        r"""
1. **Modell:** Angebot $F$ in S, Nachfrage $F$ in T ($F$ = größter Fluss), Kosten mit $n+1$ multipliziert. Ein *Pseudofluss* darf Überschüsse $e(v)$ haben; Preise $p(v)$ starten bei 0. Reduzierte Kosten $c_p(u,v)=c(u,v)+p(u)-p(v)$.
2. **$\varepsilon$-optimal:** $c_p\ge-\varepsilon$ auf allen Restkanten. Der leere Fluss ist es für $\varepsilon_0=$ größte Kosten. Eine Restkante mit $c_p<0$ heißt **zulässig** - nur über sie wird geschoben.
3. **Phase (refine):** $\varepsilon\leftarrow\lceil\varepsilon/\alpha\rceil$. Erst alle zulässigen Restkanten **sättigen** (danach ist alles 0-optimal, es entstehen Überschüsse). Dann **entladen**: ein Knoten mit Überschuss schiebt über zulässige Kanten; hat er keine mehr, **hebt** er sich an: $p(v)\leftarrow p(v)-(\min c_p+\varepsilon)$.
4. **Ende der Phase:** kein Überschuss mehr - der Fluss ist zulässig und $\varepsilon$-optimal, aber noch nicht billigst.
5. **Warum das am Ende genügt:** jeder Kreis im Restgraphen hat höchstens $n$ Kanten, also Kosten $\ge -n\varepsilon$; bei $\varepsilon=1$ und Kosten mal $n+1$ wäre ein negativer Kreis mindestens $-(n+1)$ - den gibt es nicht. Nach $\lceil\log_\alpha \varepsilon_0\rceil$ Phasen ist der Fluss kostenminimal.
6. **Toleranz:** wer nur **fast** optimal braucht, hält früher an - dann spart kein Kreis im Restgraphen mehr als $\varepsilon/(n+1)$ je Kante.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
names = list(C.PRESETS.keys())
for row in range(0, len(names), 4):
    preset_cols = st.columns(4)
    for col, name in zip(preset_cols, names[row:row + 4]):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", list(C.NETS), key="net_select", format_func=lambda k: C.NETS[k],
        help="Ein zufälliges Distributionsnetz mit Kosten je Einheit, oder eines der festen Lehrnetze: der Umweg (nach der ersten Phase fließt noch alles über den teuren Weg) und die Zuordnung mit Kosten.",
    )
    alpha = st.radio(
        "Skalierungsfaktor α", list(C.ALPHAS), key="alpha_radio", format_func=lambda a: f"α = {a} (ε durch {a} teilen)",
        help="Um welchen Faktor ε je Phase sinkt. Große α: wenige Phasen, aber je Phase mehr Arbeit. Im Standardnetz-Mittel durchsuchen α = 2, 4, 8, 16 im Mittel 3737, 2620, 2581 und 3079 Kanten - α = 4 bis 8 ist am besten.",
    )
    selection = st.radio(
        "Welcher Knoten wird entladen?", list(C.SELECTION_LABELS), key="selection_radio", format_func=lambda k: C.SELECTION_LABELS[k],
        help="Warteschlange (FIFO) oder stets der Knoten mit dem kleinsten Index. Bei α = 2 im Mittel 3737 gegen 3862 durchsuchte Kanten: kaum ein Unterschied.",
    )
    tol = st.radio(
        "Anhalten bei Toleranz", list(C.TOLERANCES), key="tol_radio", format_func=lambda t: "0 – bis zum Optimum (ε = 1)" if t == 0 else f"{t} – kein Kreis spart mehr als {t} je Kante",
        help="Anhalten, sobald ε ≤ (n+1) × Toleranz: der Fluss ist zulässig und ε-optimal, aber nicht mehr sicher kostenminimal. Über 100 Netze: Toleranz 1 halbiert die Arbeit (1881 statt 3737 durchsuchte Kanten) bei 0,15 % Mehrkosten im Mittel; Toleranz 5 hält nach der ersten Phase an (583 Kanten, 3,0 % Mehrkosten).",
    )
    if net_key == "random":
        seed_widget("p_slider")
        p = st.slider("Werke", *bounds("p_slider"), key="p_slider", help="Anzahl der Werke (oben im Netz); Kosten je Einheit 1 bis 5.")
        st.session_state[KEPT["p_slider"]] = p
        seed_widget("d_slider")
        d = st.slider("Verteilzentren", *bounds("d_slider"), key="d_slider", help="Anzahl der Verteilzentren; Umschlagkosten 1 bis 3 je Einheit, Durchsatz 30 bis 60 % der gesamten Werkskapazität.")
        st.session_state[KEPT["d_slider"]] = d
        seed_widget("s_slider")
        s = st.slider("Filialen", *bounds("s_slider"), key="s_slider", help="Anzahl der Filialen (unten im Netz).")
        st.session_state[KEPT["s_slider"]] = s
        seed_widget("density_slider")
        density = st.slider("Netzdichte [%]", *bounds("density_slider"), key="density_slider", step=10, help="Anteil der möglichen Lanes (Werk → Verteilzentrum, Verteilzentrum → Filiale), die es gibt; Kosten je Lane 1 bis 9.")
        st.session_state[KEPT["density_slider"]] = density
        seed_widget("spread_slider")
        spread = st.slider("Streuung der Lane-Breiten [%]", *bounds("spread_slider"), key="spread_slider", step=25, help="0 = alle Lanes einer Stufe gleich breit, 100 = Kapazitäten gleichverteilt von 1 bis zum Doppelten der Grundbreite.")
        st.session_state[KEPT["spread_slider"]] = spread
        seed_widget("load_slider")
        load = st.slider("Auslastung [% der Werkskapazität]", *bounds("load_slider"), key="load_slider", step=10, help="Gesamtnachfrage der Filialen in Prozent der Werkskapazität. Über 100 % kann das Netz die Nachfrage nicht mehr decken.")
        st.session_state[KEPT["load_slider"]] = load
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed. Die Verteilungen über 100 feste Netze weiter unten ändern sich dabei nicht - nur die Marke „Ihre Ziehung“ wandert.")
    else:
        p = int(st.session_state.get(KEPT["p_slider"], C.DEFAULT_P))
        d = int(st.session_state.get(KEPT["d_slider"], C.DEFAULT_D))
        s = int(st.session_state.get(KEPT["s_slider"], C.DEFAULT_S))
        density = int(st.session_state.get(KEPT["density_slider"], C.DEFAULT_DENSITY))
        spread = int(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
        load = int(st.session_state.get(KEPT["load_slider"], C.DEFAULT_LOAD))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen. Zahl der Werke, Verteilzentren und Filialen, Netzdichte, Streuung, Auslastung und Seed gehören zum zufälligen Netz.")

sync_query_params({"net_select": net_key, "alpha_radio": int(alpha), "selection_radio": selection, "tol_radio": int(tol), "p_slider": int(p), "d_slider": int(d), "s_slider": int(s),
                   "density_slider": int(density), "spread_slider": int(spread), "load_slider": int(load), "seed_input": int(seed)})

# feste Netze ignorieren die Zufallsregler: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
params = (net_key, int(p), int(d), int(s), int(density), int(spread), int(load), int(seed))
if net_key in C.FIXED_NETS:
    params = (net_key, C.DEFAULT_P, C.DEFAULT_D, C.DEFAULT_S, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD, C.DEFAULT_SEED)
with st.spinner("Rechne..."):
    a = _analysis(params, int(alpha), selection, int(tol))
net, res = a.net, a.result
level, code, dat = ev.verdict(a)
settings = (int(p), int(d), int(s), int(density), int(spread), int(load))
phases, frames = res.phases, res.frames
n_phase_frames = len(phases) + 2                # Bild 0 = Start, je Phase ein Bild, zuletzt der Beweis
is_fixed = net_key in C.FIXED_NETS
cert_pi, cert_ok = ssp.certificate(net, res.flow)
final_pi = tuple(x - cert_pi[net.s] for x in cert_pi) if cert_ok else None
proof_data = ev.proof(net, res.flow, final_pi) if final_pi is not None else None

# --- Phasen und Entladungen ------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Phasen und Entladungen")
owner = (params, int(alpha), selection, int(tol))
if st.session_state.get("csc_owner") != owner:
    st.session_state["csc_phase"] = n_phase_frames - 1
    st.session_state["csc_owner"] = owner
    st.session_state["csc_step_owner"] = None
step_col, play_col = st.columns([5, 2])
with step_col:
    phase = st.slider("Phase", 0, n_phase_frames - 1, key="csc_phase", help="0 ist der Start (leerer Fluss, alle Preise 0); jede weitere Stufe ist eine Phase mit eigenem ε; ganz rechts der fertige Fluss und der Beweis.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
inner_slot = st.container()
view_slot = st.empty()


def _push_text(frame):
    parts = []
    for e, amount in frame.pushes:
        u, v = net.arcs[e // 2][0], net.arcs[e // 2][1]
        target = v if e % 2 == 0 else u
        parts.append(f"{amount} → {net.names[target]}")
    return ", ".join(parts)


def _render(k, j):
    """Phase k (0 = Start, 1..P = Phasen, P+1 = Beweis), Bild j innerhalb der Phase."""
    tag = f"{k}_{j}"
    with view_slot.container():
        c1, c2 = st.columns(2)
        if k >= len(phases) + 1:
            c1.markdown(f"**Fertiger Fluss** - Menge {res.value}, Gesamtkosten {res.total}")
            if final_pi is not None:
                c1.plotly_chart(build_network(net, res.flow, pi=final_pi), width="stretch", key=f"result_map_{k}")
            else:
                c1.plotly_chart(build_state(net, frames[-1], res.scale), width="stretch", key=f"result_state_{k}")
            c2.markdown("**Beweis:** ε und Kosten je Phase")
            c2.plotly_chart(build_eps(dat["eps"], dat["costs"], len(phases), dat["optimal_total"]), width="stretch", key=f"proof_eps_{k}")
            if final_pi is None:
                st.caption(f"Angehalten nach Phase {len(phases)} bei ε = {dat['last_eps']} (Toleranz {a.tol} je Kante): der Fluss ist zulässig und ε-optimal, aber nicht sicher kostenminimal - er kostet {res.total}, das Optimum {dat['optimal_total']} ({_pct(dat['gap_pct'])} weniger). "
                           f"Kein Kreis im Restgraphen spart mehr als ε/(n+1) = {_f(dat['last_eps'] / res.scale, 1)} je Kante. Der Restgraph enthält noch negative Kreise, deshalb gibt es keine gültigen Schattenpreise.")
            else:
                st.caption(f"Bei ε = 1 (Kosten mal {res.scale}) sind alle reduzierten Kosten ≥ −1; ein negativer Kreis hätte höchstens {net.n} Kanten, also Kosten ≥ −{net.n} - aber alle Kreiskosten sind Vielfache von {res.scale}: es gibt keinen negativen Kreis. "
                           f"Unabhängig prüft der Restgraph mit Potenzialen aus Bellman-Ford: **alle {proof_data['checked']} Restkanten** haben reduzierte Kosten ≥ 0 (Vorwärtsrest) bzw. ≤ 0 (Kante mit Fluss), {proof_data['tight']} der {proof_data['flow_arcs']} Flusskanten genau 0. "
                           f"**Kein Fluss dieser Menge ist billiger.** Rechts: ε fällt von {dat['eps0']} in {len(phases)} Phasen auf 1, die Kosten erreichen das Optimum {dat['optimal_total']}.")
            return
        if k == 0:
            c1.markdown("**Start:** leerer Fluss, Preise 0")
            c2.markdown("**ε und Kosten je Phase** - noch keine Phase")
            c1.plotly_chart(build_state(net, frames[0], res.scale), width="stretch", key=f"state_map_{tag}")
            c2.plotly_chart(build_eps(dat["eps"], dat["costs"], 0, dat["optimal_total"]), width="stretch", key=f"eps_chart_{tag}")
            st.caption(f"Alle {a.value} Einheiten stehen in S als Überschuss (roter Ring), T hat die Nachfrage. Der leere Fluss ist ε₀-optimal für ε₀ = {res.eps0} (die größten Kosten mal {res.scale}). Phase 1 teilt ε durch {a.alpha}.")
            return
        ph = phases[k - 1]
        f = frames[ph.first_frame + j]
        done = k if ph.first_frame + j == ph.last_frame else k - 1
        c1.markdown(f"**Phase {k}** (ε = {ph.eps}): " + ("Sättigen" if f.kind == "saturate" else f"Entladung {j} von {ph.last_frame - ph.first_frame}"))
        c2.markdown(f"**ε und Kosten je Phase** - {done} von {len(phases)} Phasen abgeschlossen")
        c1.plotly_chart(build_state(net, f, res.scale), width="stretch", key=f"state_map_{tag}")
        c2.plotly_chart(build_eps(dat["eps"], dat["costs"], done, dat["optimal_total"]), width="stretch", key=f"eps_chart_{tag}")
        if f.kind == "saturate":
            cap = (f"Phase {k} beginnt mit ε = {ph.eps}: {ph.saturated} Restkanten mit negativen reduzierten Kosten werden gesättigt (grün, mit Menge). Danach ist alles 0-optimal, aber es gibt Überschüsse (rote Ringe) und Nachfrageknoten mit Mangel; "
                   f"{ph.discharges} Entladungen folgen.")
        else:
            parts = []
            if f.pushes:
                parts.append("schiebt " + _push_text(f))
            if f.relabels:
                parts.append("senkt seinen Preis " + ", ".join(f"von {o} auf {n}" for o, n in f.relabels))
            cap = f"{net.names[f.node]} " + " und ".join(parts) + f" ({f.scanned} Kanten durchsucht)."
            if ph.first_frame + j == ph.last_frame:
                cap += (f" **Ende von Phase {k}:** kein Überschuss mehr, der Fluss ist zulässig und {ph.eps}-optimal (kleinste reduzierte Kosten {ph.min_rc} ≥ −{ph.eps}); Kosten {ph.cost}, " + (f"das sind {_pct(dat['gaps'][k - 1])} mehr als das Optimum." if ph.cost > dat["optimal_total"] else "das Optimum."))
        st.caption(cap)


# innerer Slider: Entladungen der gewählten Phase
inner = 0
if 1 <= phase <= len(phases):
    ph_sel = phases[phase - 1]
    n_inner = ph_sel.last_frame - ph_sel.first_frame + 1
    inner_owner = (owner, phase)
    if st.session_state.get("csc_step_owner") != inner_owner:
        st.session_state["csc_step"] = n_inner - 1
        st.session_state["csc_step_owner"] = inner_owner
    with inner_slot:
        if n_inner > 1:
            inner = st.slider(f"Entladung in Phase {phase}", 0, n_inner - 1, key="csc_step", help="0 ist das Sättigen zu Beginn der Phase, dann eine Entladung je Bild; ganz rechts der Zustand am Ende der Phase.")
        else:
            inner = 0
            st.caption(f"Phase {phase} hat nur den Sättigungsschritt - keine Entladung.")

if auto_play:
    for k in range(n_phase_frames):
        if 1 <= k <= len(phases):
            _render(k, phases[k - 1].last_frame - phases[k - 1].first_frame)
        else:
            _render(k, 0)
        time.sleep(min(0.9, 8.0 / max(n_phase_frames, 1)))
    phase = n_phase_frames - 1
else:
    _render(phase, inner)

st.caption("Links: Knotenfarbe = Preis p (skaliert mit n + 1), roter Ring = Überschuss (mit Menge), dicker schwarzer Ring = der entladene Knoten, grüne Kanten = Pushes dieser Entladung (mit Menge), orange gestrichelt = Push über eine Rückkante. "
           "Rechts oben ε je Phase (logarithmisch), unten die Kosten des zulässigen Flusses am Ende jeder Phase, gestrichelt das Optimum von SSP.")

st.markdown("---")

# --- Kernfrage ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Vom groben zum feinen Fluss")
st.caption("**Phasen** = wie oft ε durch α geteilt wird; **Entladungen** = wie oft ein Knoten geleert wird (Pushes und, wenn nötig, Relabels); **durchsuchte Kanten** = Aufwand (jede in einer Adjazenzliste angesehene Restkante, auch beim Sättigen und Anheben), nie Sekunden. Zum Vergleich: SSP baut denselben Fluss in Runden auf.")
m1, m2, m3, m4 = st.columns(4)
if net.logistic:
    m1.metric("Menge", f"{dat['value']} von {dat['demand']}", delta=f"{_pct(dat['share'])} der Nachfrage", delta_color="off", help="Die Menge F ist der größte Fluss des Netzes; Cost Scaling ändert sie nicht.")
else:
    m1.metric("Menge", f"{dat['value']}", help="Menge von S nach T (der größte Fluss).")
m2.metric("Phasen", f"{dat['phases']}", delta=f"ε {dat['eps0']} → {dat['last_eps']}" + ("" if dat["phases"] == dat["n_phases_full"] else f" (angehalten, voll wären {dat['n_phases_full']})"), delta_color="off", help=f"Je Phase wird ε durch α = {a.alpha} geteilt (aufgerundet); mit Kosten mal {res.scale} genügt ε = 1.")
m3.metric("Entladungen", f"{dat['discharges']}", delta=f"{dat['pushes']} Pushes, {dat['relabels']} Relabels", delta_color="off", help=f"Von den {dat['pushes']} Pushes sättigen {dat['sat_pushes']} die Kante.")
m4.metric("Durchsuchte Kanten", f"{dat['scanned']}", delta=f"SSP: {dat['ssp_scanned']}", delta_color="off", help=f"Successive Shortest Paths braucht für denselben Fluss (von Grund auf) {dat['ssp_scanned']} durchsuchte Kanten in {dat['ssp_rounds']} Runden.")

if code == "optimal":
    served = ""
    if net.logistic:
        served = " Die gesamte Nachfrage wird geliefert." if dat["value"] == dat["demand"] else f" Das Netz schafft höchstens {dat['value']} von {dat['demand']} Einheiten ({_f(dat['share'], 0)} %); Engpass: **{_stage_text(dat['stage_caps'])}**."
    early_note = f" (bei Toleranz {a.tol} schon nach {dat['phases']} Phasen - hier zufällig optimal)" if a.tol > 0 and dat["phases"] < dat["n_phases_full"] else ""
    st.success(f"✅ Kostenminimal: {dat['phases']} Phasen (ε von {dat['eps0']} auf {dat['last_eps']}){early_note}, {dat['discharges']} Entladungen, Kosten {dat['total']} - dasselbe Optimum wie SSP. Nach der ersten Phase war der Fluss {_pct(dat['first_gap_pct'])} zu teuer.{served} "
               f"Dafür durchsucht Cost Scaling {dat['scanned']} Kanten, SSP {dat['ssp_scanned']} ({_f(dat['scanned'] / dat['ssp_scanned'], 1)}-fach).")
elif code == "early":
    st.info(f"ℹ️ Bei Toleranz {a.tol} nach {dat['phases']} von {dat['n_phases_full']} Phasen angehalten (ε = {dat['last_eps']}): zulässiger Fluss mit Kosten {dat['total']}, {_pct(dat['gap_pct'])} über dem Optimum {dat['optimal_total']}. "
            f"Durchsucht wurden {dat['scanned']} Kanten - SSP braucht für den optimalen Fluss {dat['ssp_scanned']}.")
elif code == "wrong":
    st.warning(f"⚠️ **Nicht kostenminimal:** der Fluss kostet {dat['total']}, der billigste {dat['optimal_total']}. Das darf bei ε = 1 nicht passieren.")
else:
    st.warning("⚠️ Es kommt gar nichts an: kein Weg führt von einem Werk über ein Verteilzentrum zu einer Filiale. Die Menge ist 0, es gibt keinen Fluss und nichts zu skalieren.")

if is_fixed:
    st.info("Festes Netz: es gibt nur diese eine Ziehung. Für die Verteilungen über viele Netze ein zufälliges Distributionsnetz wählen.")
    dist = None
else:
    dist = ev.distribution(*settings, int(alpha), selection)
    st.markdown(f"**Nicht nur dieses eine Netz:** {len(C.DIST_SEEDS)} feste Netze mit denselben Einstellungen (Werke {p}, Verteilzentren {d}, Filialen {s}, Netzdichte {density} %, Streuung {spread} %, Auslastung {load} %), getrennt vom Seed oben; α = {alpha}, {SEL_SHORT[selection]}, bis ε = 1.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Phasen", _f(dist["phases_mean"], 1), delta=f"höchstens {dist['phases_max']}", delta_color="off", help="Mittel über die Netze.")
    p2.metric("Entladungen", _f(dist["discharges_mean"], 1), delta=f"{_f(dist['pushes_mean'], 1)} Pushes, {_f(dist['relabels_mean'], 1)} Relabels", delta_color="off", help="Mittel über die Netze.")
    p3.metric("Durchsuchte Kanten", _f(dist["scans_mean"], 0), delta=f"SSP {_f(dist['ssp_scans_mean'], 0)}", delta_color="off", help="Mittel über die Netze: Cost Scaling gegen SSP.")
    p4.metric("Höchstens so viele wie SSP", _share(dist["share_cs_le_ssp"]), delta=f"Kosten optimal: {_share(dist['share_optimal'])}", delta_color="off", help="Anteil der Netze, in denen Cost Scaling höchstens so viele Kanten durchsucht wie SSP.")
    st.plotly_chart(build_ratio_hist(dist["cols"]["ratio"], current=dat["scanned"] / dat["ssp_scanned"] if (net.logistic and dat["ssp_scanned"]) else None), width="stretch", key="ratio_chart")
    st.caption(f"Über {len(C.DIST_SEEDS)} feste Netze durchsucht Cost Scaling im Mittel das {_f(dist['ratio_mean'], 1)}-fache der Kanten von SSP (Median {_f(dist['ratio_median'], 1)}, höchstens {_f(dist['ratio_max'], 1)}); in {_share(dist['share_cs_le_ssp'])} der Netze höchstens so viele. Auf so kleinen Netzen ist SSP im Vorteil - das kippt mit der Netzgröße (Experiment unten).")

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Phasen im Protokoll"):
    st.markdown("**Was jede Phase für das Netz oben leistet** (aktuelle Einstellung)")
    st.table({"Phase": [str(i + 1) for i in range(len(phases))], "ε": [str(p_.eps) for p_ in phases], "gesättigte Kanten": [str(p_.saturated) for p_ in phases], "Entladungen": [str(p_.discharges) for p_ in phases],
              "Pushes": [str(p_.pushes) for p_ in phases], "Relabels": [str(p_.relabels) for p_ in phases], "durchsuchte Kanten": [str(p_.scanned) for p_ in phases], "Kosten": [str(p_.cost) for p_ in phases],
              "Mehrkosten": [_pct(g) for g in dat["gaps"]], "kleinste reduzierte Kosten": [str(p_.min_rc) for p_ in phases]})
    st.caption("Am Ende jeder Phase ist der Fluss zulässig und ε-optimal: die kleinsten reduzierten Kosten einer Restkante liegen nicht unter −ε (in Einheiten mal n + 1). Die Kosten fallen nicht in jeder Phase - eine feinere Phase darf einen Fluss verschlechtern, sie garantiert nur die feinere Schranke.")
    st.markdown("**Vergleich mit SSP:** " + f"Cost Scaling {dat['scanned']} durchsuchte Kanten in {dat['phases']} Phasen, SSP {dat['ssp_scanned']} in {dat['ssp_rounds']} Runden (beide {dat['total'] if code == 'optimal' else dat['optimal_total']} Kosten).")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie fein soll skaliert werden – α")
st.caption("Große α heißen wenige, grobe Sprünge: weniger Phasen, aber in jeder Phase mehr Überschuss zu verschieben. Kleine α heißen viele Phasen mit wenig Arbeit je Phase. Wo liegt der beste Wert?")
if is_fixed:
    st.info("Für die Verteilung über viele Netze ein zufälliges Distributionsnetz wählen. Für das feste Netz oben zeigt die Tabelle im Expander die Arbeit je Phase.")
else:
    alpha_rows = ev.alpha_table(*settings)
    st.plotly_chart(build_alpha(alpha_rows), width="stretch", key="alpha_chart")
    st.table({"α": [str(r["alpha"]) for r in alpha_rows], "Knotenwahl": [SEL_SHORT[r["selection"]] for r in alpha_rows], "Phasen": [_f(r["phases"], 1) for r in alpha_rows], "Entladungen": [_f(r["discharges"], 1) for r in alpha_rows],
              "Pushes": [_f(r["pushes"], 1) for r in alpha_rows], "Relabels": [_f(r["relabels"], 1) for r in alpha_rows], "durchsuchte Kanten (Mittel)": [_f(r["scans"], 0) for r in alpha_rows]})
    best = min(alpha_rows, key=lambda r: r["scans"])
    st.caption(f"Mittel über {len(C.DIST_SEEDS)} feste Netze, alle enden im Optimum. Am wenigsten durchsucht α = {best['alpha']} ({SEL_SHORT[best['selection']]}) mit {_f(best['scans'], 0)} Kanten in {_f(best['phases'], 1)} Phasen; α = 2 braucht {_f(next(r['scans'] for r in alpha_rows if r['alpha'] == 2 and r['selection'] == 'fifo'), 0)}, α = 32 {_f(next(r['scans'] for r in alpha_rows if r['alpha'] == 32 and r['selection'] == 'fifo'), 0)}. "
               "Die Knotenwahl (Warteschlange gegen beliebig) macht dagegen kaum einen Unterschied.")

st.subheader("🔬 Näherung: wie gut ist der Fluss nach Phase k?")
st.caption("Jede Phase liefert einen zulässigen Fluss. Wer nicht das Optimum, sondern nur einen guten Fluss braucht, hält früher an. Wie viel Mehrkosten bleiben, und wie viel Arbeit spart das?")
if dist is None:
    st.info("Für die Verteilung über viele Netze ein zufälliges Distributionsnetz wählen. Für das feste Netz oben zeigt die Tabelle im Expander die Mehrkosten je Phase.")
else:
    st.plotly_chart(build_anytime(dist["gap_mean"], dist["gap_max"], dist["share_optimal_after"]), width="stretch", key="anytime_chart")
    st.table({"nach Phase": [str(k + 1) for k in range(dist["max_phases"])], "Mehrkosten (Mittel)": [_pct(g, 2) for g in dist["gap_mean"]], "Mehrkosten (Median)": [_pct(g, 2) for g in dist["gap_median"]],
              "Mehrkosten (größter Wert)": [_pct(g, 1) for g in dist["gap_max"]], "schon optimal": [_share(x) for x in dist["share_optimal_after"]]})
    tol_rows = ev.tolerance_table(*settings)
    st.markdown("**Anhalten bei Toleranz** (α = 2, Warteschlange): kein Kreis im Restgraphen spart mehr als die Toleranz je Kante.")
    st.table({"Toleranz je Kante": [str(r["tol"]) for r in tol_rows], "Phasen (Mittel)": [_f(r["phases"], 1) for r in tol_rows], "durchsuchte Kanten (Mittel)": [_f(r["scans"], 0) for r in tol_rows],
              "Mehrkosten (Mittel)": [_pct(r["gap_mean"], 2) for r in tol_rows], "Mehrkosten (größter Wert)": [_pct(r["gap_max"], 1) for r in tol_rows], "schon optimal": [_share(r["share_optimal"]) for r in tol_rows]})
    last = dist["max_phases"] - 1
    gm = lambda k: dist["gap_mean"][min(k, last)]
    st.caption(f"Über {len(C.DIST_SEEDS)} feste Netze: nach der ersten Phase kostet der Fluss im Mittel {_pct(gm(0))} mehr als das Optimum (Median {_pct(dist['gap_median'][0])}, höchstens {_pct(dist['gap_max'][0])}), nach der dritten {_pct(gm(2))}, nach der fünften {_pct(gm(4), 2)}; "
               f"nach der vierten Phase sind {_share(dist['share_optimal_after'][min(3, last)])} der Netze schon optimal. Die Kosten sinken **nicht monoton**: in {_share(dist['share_nonmonotone'])} der Netze wird der Fluss in mindestens einer Phase wieder teurer, bevor er billiger wird. "
               f"Toleranz 1 spart etwa die Hälfte der Arbeit ({_f(tol_rows[1]['scans'], 0)} statt {_f(tol_rows[0]['scans'], 0)} Kanten) bei {_pct(tol_rows[1]['gap_mean'], 2)} Mehrkosten im Mittel.")

st.subheader("🔬 Kostenspanne: log(n·C) Phasen")
st.caption("Die Phasenzahl hängt an der Spanne der Kosten, nicht an der Menge: ε beginnt bei den größten Kosten. Kosten mal 1000 bedeuten zehn Phasen mehr bei α = 2 - und SSP bleibt unberührt, weil Wegkosten für Dijkstra nur Zahlen sind.")
if is_fixed:
    st.info("Für dieses Experiment ein zufälliges Distributionsnetz wählen.")
else:
    if st.button("Kosten mit 1, 10, 100 und 1000 multiplizieren (20 Netze)", key="cost_range_start"):
        st.session_state["cost_range_on"] = settings
    if st.session_state.get("cost_range_on") == settings:
        cr_rows = ev.cost_range_table(*settings)
        st.plotly_chart(build_cost_range(cr_rows), width="stretch", key="cost_range_chart")
        st.table({"Kosten ×": [str(r["k"]) for r in cr_rows], "ε₀": [_int(r["eps0"]) for r in cr_rows], "Phasen (Mittel)": [_f(r["phases"], 1) for r in cr_rows], "Entladungen (Mittel)": [_f(r["discharges"], 0) for r in cr_rows],
                  "Cost Scaling: durchsuchte Kanten": [_int(r["scans"]) for r in cr_rows], "SSP: durchsuchte Kanten": [_int(r["ssp_scans"]) for r in cr_rows]})
        st.caption(f"Mittel über 20 feste Netze (α = 2). Mit Kosten mal 1000 steigt ε₀ auf {_int(cr_rows[-1]['eps0'])}, die Phasen von {_f(cr_rows[0]['phases'], 0)} auf {_f(cr_rows[-1]['phases'], 0)}, die durchsuchten Kanten von {_int(cr_rows[0]['scans'])} auf {_int(cr_rows[-1]['scans'])} ({_f(cr_rows[-1]['scans'] / cr_rows[0]['scans'], 1)}-fach) - SSP bleibt bei {_int(cr_rows[0]['ssp_scans'])}. "
                   "Die Spanne der Kosten kostet Cost Scaling also nur logarithmisch mehr Phasen.")

st.subheader("🔬 Kreuzungspunkt: ab welcher Netzgröße gewinnt Cost Scaling?")
st.caption("SSP braucht so viele Runden wie es Wege gibt und sucht in jeder das ganze Netz ab; Cost Scaling ist von der Menge unabhängig und wächst flacher. Auf den kleinen Netzen der Regler verliert es - wo kippt das?")
if dist is None:
    st.info("Für die Verteilung über viele Netze ein zufälliges Distributionsnetz wählen. Für das feste Netz oben zeigen die Kennzahlen „Durchsuchte Kanten“ den Vergleich.")
else:
    if st.button("Netze von 12 bis 458 Knoten durchrechnen (dauert einige Sekunden)", key="scaling_start"):
        st.session_state["scaling_on"] = True
    if st.session_state.get("scaling_on"):
        with st.spinner("Rechne 8 Netzgrößen × 3 Netze × 3 Verfahren..."):
            sc_rows = ev.scaling()
        slopes = ev.slopes(sc_rows)
        cross4, cross2 = ev.crossing(sc_rows, "cs4"), ev.crossing(sc_rows, "cs2")
        st.plotly_chart(build_scaling(sc_rows), width="stretch", key="scaling_chart")
        st.table({"Werke / DCs / Filialen": [f"{r['size'][0]} / {r['size'][1]} / {r['size'][2]}" for r in sc_rows], "Knoten": [_f(r["n"], 0) for r in sc_rows], "Kanten": [_f(r["m"], 0) for r in sc_rows], "SSP-Runden": [_f(r["ssp_rounds"], 0) for r in sc_rows],
                  "SSP": [_int(r["ssp"]) for r in sc_rows], "Cost Scaling α = 2": [_int(r["cs2"]) for r in sc_rows], "Cost Scaling α = 4": [_int(r["cs4"]) for r in sc_rows], "α = 4 ÷ SSP": [_f(r["cs4"] / r["ssp"], 2) for r in sc_rows]})
        st.caption(f"Mittel über 3 feste Netze je Größe (Netzdichte 60 %, Streuung und Auslastung auf den Standardwerten). Steigung im doppelt logarithmischen Diagramm: SSP {_f(slopes['ssp'], 2)}, Cost Scaling {_f(slopes['cs2'], 2)} (α = 2) und {_f(slopes['cs4'], 2)} (α = 4). "
                   + (f"Cost Scaling mit α = 4 durchsucht ab {_f(cross4['n'], 0)} Knoten weniger Kanten als SSP, mit α = 2 ab {_f(cross2['n'], 0)} Knoten. " if cross4 and cross2 else "Im untersuchten Bereich kippt es nicht. ")
                   + f"Im kleinsten Netz (12 Knoten) durchsucht es das {_f(sc_rows[0]['cs2'] / sc_rows[0]['ssp'], 1)}-fache, im größten das {_f(sc_rows[-1]['cs2'] / sc_rows[-1]['ssp'], 2)}-fache (α = 2).")

st.subheader("🔬 Zuordnung n × n: wo SSP n Runden braucht")
st.caption("Auf einem Netz mit Einheitskapazitäten zwischen n Fahrzeugen und n Aufträgen ist SSP die Ungarische Methode: n Runden, jede sucht das ganze Netz ab. Cost Scaling ist hier die Idee der Auktion mit ε-Skalierung.")
if st.button("Zuordnungen von 5 × 5 bis 120 × 120 durchrechnen (dauert einige Sekunden)", key="assignment_start"):
    st.session_state["assignment_on"] = True
if st.session_state.get("assignment_on"):
    with st.spinner("Rechne 7 Größen × 3 Zuordnungen × 2 Verfahren..."):
        as_rows = ev.assignment_table()
    st.plotly_chart(build_assignment(as_rows), width="stretch", key="assignment_chart")
    st.table({"n × n": [f"{r['n']} × {r['n']}" for r in as_rows], "Kanten": [str(r["n"] * r["n"] + 2 * r["n"]) for r in as_rows], "SSP-Runden": [_f(r["rounds"], 0) for r in as_rows], "Cost-Scaling-Phasen": [_f(r["phases"], 0) for r in as_rows],
              "SSP: durchsuchte Kanten": [_int(r["ssp"]) for r in as_rows], "Cost Scaling α = 4": [_int(r["cs"]) for r in as_rows], "Cost Scaling ÷ SSP": [_f(r["cs"] / r["ssp"], 2) for r in as_rows]})
    below = next((r for r in as_rows if r["cs"] < r["ssp"]), None)
    st.caption(f"Mittel über 3 Zuordnungen je Größe (α = 4). Cost Scaling braucht nur {_f(as_rows[0]['phases'], 0)} bis {_f(as_rows[-1]['phases'], 0)} Phasen, SSP n Runden. Bei 5 × 5 durchsucht Cost Scaling das {_f(as_rows[0]['cs'] / as_rows[0]['ssp'], 1)}-fache von SSP, bei {as_rows[-1]['n']} × {as_rows[-1]['n']} das {_f(as_rows[-1]['cs'] / as_rows[-1]['ssp'], 2)}-fache"
               + (f" - es gewinnt ab n = {below['n']}." if below else "."))

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist - und wer setzt an |
|---|---|
| **Ganzzahlige Kosten** | Die Skalierung mit n + 1 und der Abbruch bei ε = 1 setzen ganze Zahlen voraus; bei gebrochenen Kosten wäre ε < 1/n nötig, mehr Phasen. **Ansatzpunkt:** Skalierung der Kosten vorab. |
| **Ohne die Heuristiken der Praxis** | Hier läuft nacktes Push-Relabel je Phase. Produktive Löser (OR-Tools `SimpleMinCostFlow`) ergänzen eine globale Preisaktualisierung und Look-Ahead-Regeln und sind dadurch auch auf kleinen Netzen schnell. **Ansatzpunkt:** Heuristiken wie bei Push-Relabel (Gap, Global Relabeling). |
| **Sequentielle Rechnung** | Push-Relabel ist lokal und parallelisierbar - gezählt wird nacheinander. **Ansatzpunkt:** nicht Thema dieser Linie. |
| **Ein Gut, teilbar** | Alle Waren sind gleich und beliebig teilbar. Mehrere Güter auf gemeinsamen Kanten machen den Fluss im Allgemeinen gebrochen. **Ansatzpunkt: Mehrgüterfluss** (gebaut: multicommodity-demo). |
| **Keine Zeit** | Ein Fluss ist eine Momentaufnahme; Wartezeiten und Fahrpläne fehlen. **Ansatzpunkt:** Zeit-Raum-Netz in der Demo „leercontainer-demo“. |
"""
)
st.caption("Die Netzwerkfluss-Linie ist als Ganzes geplant: Edmonds-Karp, Dinic, Push-Relabel, Successive Shortest Paths, Cycle-Canceling, Cost Scaling (dieses Stück), Mehrgüterfluss (gebaut), Column Generation (gebaut), Garg-Könemann (gebaut), Fixkosten-Netzwerkdesign (gebaut), Benders-Zerlegung (gebaut) und Slope Scaling (gebaut) - bisher sind alle zwölf Stücke der Hauptlinie gebaut.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell (Min-Cost-Flow).** Gerichteter Graph $G=(V,E)$, $n=|V|$, ganzzahlige Kapazitäten $u_e$ und Kosten $c_e$; Angebot $b(s)=F$, $b(t)=-F$. Kosten werden mit $n+1$ multipliziert: $c'_e=(n+1)c_e$.

**Pseudofluss und Preise.** $f$ mit $0\le f_e\le u_e$ und Überschuss $e_f(v)=b(v)+\sum_{\text{ein}}f-\sum_{\text{aus}}f$ (hier: Zufluss minus Abfluss plus Angebot). Preise $p:V\to\mathbb Z$, reduzierte Kosten $c_p(v,w)=c'(v,w)+p(v)-p(w)$ für jede Restkante.

**$\varepsilon$-Optimalität.** $(f,p)$ ist $\varepsilon$-optimal, wenn $c_p(v,w)\ge-\varepsilon$ für jede Restkante $(v,w)$. Das ist äquivalent dazu, dass jeder Kreis des Restgraphen einen Mittelwert der Kosten je Kante $\ge-\varepsilon$ hat (Bezug zu Minimum-Mean-Cycle-Canceling).

**Satz.** Ist $f$ zulässig und $\varepsilon$-optimal mit $\varepsilon<1$ bei ganzzahligen Kosten $c'$ (Vielfache von $n+1$) und $n$-Kanten-Kreisen, so ist $f$ kostenminimal: ein negativer Kreis hätte Kosten $\le-(n+1)$, aber $\ge-n\varepsilon>-(n+1)$.

**Refine.** Gegeben ein $2\varepsilon$-optimales $(f,p)$: (1) setze $f_e\leftarrow u_e$ für alle Restkanten mit $c_p<0$ (danach $0$-optimal, Überschüsse); (2) solange ein Knoten mit $e_f(v)>0$ existiert: **push** über eine zulässige Kante $c_p<0$ ($\delta=\min(e_f(v),r)$), oder **relabel** $p(v)\leftarrow p(v)-(\min_{(v,w)}c_p+\varepsilon)$, wenn keine zulässige Kante existiert. Beide erhalten $\varepsilon$-Optimalität; am Ende ist $f$ zulässig.

**Phasen.** $\varepsilon_0=\max|c'|$, $\varepsilon_{k+1}=\lceil\varepsilon_k/\alpha\rceil$: nach $\lceil\log_\alpha\varepsilon_0\rceil\approx\log_\alpha((n+1)C)$ Phasen ist $\varepsilon=1$. **Laufzeit:** je Phase $O(n^2m)$ (Push-Relabel) bzw. $O(n^3)$ mit FIFO, insgesamt $O(n^3\log(nC))$ - unabhängig vom Flusswert. **Näherung:** nach Phase $k$ ist $f$ zulässig und $\varepsilon_k$-optimal, kein Kreis spart mehr als $\varepsilon_k/(n+1)$ je Kante.

**Zertifikat.** Aus $\varepsilon=1$ folgen ganzzahlige Preise; Potenziale aus Bellman-Ford im Restgraphen bestätigen $c(u,v)+\pi(u)-\pi(v)\ge 0$ - dasselbe Zertifikat wie bei SSP und Cycle-Canceling.

Implementiert in `csc_scenario.py` (Netze, eigener Zufallsgenerator, Kostenskalierung), `csc_algorithm.py` (Sättigen, push, relabel, Zeigerliste, Phasen, Trace je Entladung), `csc_edmonds_karp.py` und `csc_ssp.py` (Kopien der Vorgänger-Demos: größter Fluss und Vergleichsbasis), `csc_evaluation.py` (Kennzahlen, Verteilungen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
