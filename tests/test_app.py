"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle α × Knotenwahl × Toleranz, Randgrößen, beide Slider-Ebenen, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel und Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import csc_constants as C
import csc_evaluation as ev
import csc_scenario as sc
from csc_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"

# Anfang der Meldung zum gezeigten Netz (Streamlit legt das führende Emoji in `icon`, nicht in `value`)
EXPECTED = {
    "🚚 Zufallsnetz": "Kostenminimal: 8 Phasen (ε von 180 auf 1), 690 Entladungen, Kosten 1197 - dasselbe Optimum wie SSP. Nach der ersten Phase war der Fluss 0,2 % zu teuer. Die gesamte Nachfrage wird geliefert. Dafür durchsucht Cost Scaling 5190 Kanten, SSP 824 (6,3-fach).",
    "⚡ α = 4": "Kostenminimal: 4 Phasen (ε von 180 auf 1), 502 Entladungen, Kosten 1197 - dasselbe Optimum wie SSP. Nach der ersten Phase war der Fluss 1,3 % zu teuer.",
    "🎚️ Grobe Toleranz": "Bei Toleranz 5 nach 1 von 8 Phasen angehalten (ε = 90): zulässiger Fluss mit Kosten 1027, 7,8 % über dem Optimum 953. Durchsucht wurden 590 Kanten - SSP braucht für den optimalen Fluss 814.",
    "🔀 Beliebige Knotenwahl": "Kostenminimal: 8 Phasen (ε von 180 auf 1), 696 Entladungen, Kosten 1197",
    "🏭 Werke knapp": "Kostenminimal: 8 Phasen (ε von 180 auf 1), 668 Entladungen, Kosten 1403 - dasselbe Optimum wie SSP. Nach der ersten Phase war der Fluss 3,6 % zu teuer. Das Netz schafft höchstens 89 von 118 Einheiten (75 %); Engpass: **Werkskapazität 89**.",
    "🕸️ Dünnes Netz": "Kostenminimal: 8 Phasen (ε von 160 auf 1), 425 Entladungen, Kosten 900",
    "🧭 Umweg": "Kostenminimal: 6 Phasen (ε von 45 auf 1), 9 Entladungen, Kosten 2 - dasselbe Optimum wie SSP. Nach der ersten Phase war der Fluss 350,0 % zu teuer. Dafür durchsucht Cost Scaling 84 Kanten, SSP 7 (12,0-fach).",
    "💑 Zuordnung mit Kosten": "Kostenminimal: 7 Phasen (ε von 117 auf 1), 68 Entladungen, Kosten 10 - dasselbe Optimum wie SSP. Nach der ersten Phase war der Fluss 50,0 % zu teuer. Dafür durchsucht Cost Scaling 1147 Kanten, SSP 150 (7,6-fach).",
}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input) + list(at.sidebar.radio)}


def _texts(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info)]


def _has(at, prefix):
    return any(t.startswith(prefix) for t in _texts(at))


def _phase(at):
    found = [s for s in at.slider if s.key == "csc_phase"]
    return found[0] if found else None


def _inner(at):
    found = [s for s in at.slider if s.key == "csc_step"]
    return found[0] if found else None


def _metric(at, label):
    return [m.value for m in at.metric if m.label == label]


def _captions(at):
    return [c.value for c in at.caption]


def test_default_renders_without_exception():
    at = _run()
    assert any("Phasen und Entladungen" in m.value for m in at.markdown)
    assert _has(at, EXPECTED["🚚 Zufallsnetz"]) and not at.error
    assert _metric(at, "Menge")[0] == "76 von 76" and _metric(at, "Phasen")[0] == "8" and _metric(at, "Entladungen")[0] == "690" and _metric(at, "Durchsuchte Kanten")[0] == "5190"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdicts(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert _has(at, EXPECTED[name]), _texts(at)
    if C.PRESETS[name]["net"] in C.FIXED_NETS:
        assert any(t.startswith("Festes Netz") for t in _texts(at))
    else:
        assert any(t.startswith("**Nicht nur dieses eine Netz:**") for t in [m.value for m in at.markdown])


@pytest.mark.parametrize("tol", C.TOLERANCES)
@pytest.mark.parametrize("selection", list(C.SELECTION_LABELS))
@pytest.mark.parametrize("alpha", C.ALPHAS)
def test_every_alpha_selection_and_tolerance_renders(alpha, selection, tol):
    def setup(at):
        at.session_state["alpha_radio"] = alpha
        at.session_state["selection_radio"] = selection
        at.session_state["tol_radio"] = tol
    at = _run(setup)
    assert not at.error
    assert _phase(at).value == _phase(at).max and any("Fluss" in c for c in _captions(at))


def test_extreme_sizes_render():
    for p, d, s, dens, spread, load in ((C.P_MIN, C.D_MIN, C.S_MIN, C.DENSITY_MIN, C.SPREAD_MIN, C.LOAD_MIN), (C.P_MAX, C.D_MAX, C.S_MAX, C.DENSITY_MAX, C.SPREAD_MAX, C.LOAD_MAX),
                                        (C.P_MIN, C.D_MAX, C.S_MAX, C.DENSITY_MIN, C.SPREAD_MAX, C.LOAD_MAX), (C.P_MAX, C.D_MIN, C.S_MIN, C.DENSITY_MAX, C.SPREAD_MIN, C.LOAD_MIN)):
        def setup(at, vals=(p, d, s, dens, spread, load)):
            for key, value in zip(("p_slider", "d_slider", "s_slider", "density_slider", "spread_slider", "load_slider"), vals):
                at.session_state[key] = value
        at = _run(setup)
        step = _phase(at)
        assert step is not None and step.value == step.max


def test_a_net_where_nothing_arrives_renders_and_says_so():
    """Zwei Werke, sechs Verteilzentren, drei Filialen, dünnes Netz: kein Weg von S nach T, keine Entladung; jede Phase hat nur den Sättigungsschritt (kein innerer Slider mit min = max)."""
    def setup(at):
        for key, value in (("p_slider", 2), ("d_slider", 6), ("s_slider", 3), ("density_slider", 20), ("spread_slider", 50), ("load_slider", 90), ("seed_input", 8)):
            at.session_state[key] = value
    at = _run(setup)
    assert _has(at, "Es kommt gar nichts an") and _metric(at, "Menge")[0].startswith("0 von ")
    phase = _phase(at)
    phase.set_value(1)
    at.run()
    assert not at.exception and _inner(at) is None and any("hat nur den Sättigungsschritt" in c for c in _captions(at))


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(lambda a: a.session_state.__setitem__("net_select", net)))
    random_labels, fixed = labels_for("random"), labels_for("detour")
    assert {"Netz", "Skalierungsfaktor α", "Welcher Knoten wird entladen?", "Anhalten bei Toleranz", "Werke", "Verteilzentren", "Filialen", "Netzdichte [%]", "Streuung der Lane-Breiten [%]", "Auslastung [% der Werkskapazität]", "Zufalls-Seed"} <= random_labels
    assert fixed == {"Netz", "Skalierungsfaktor α", "Welcher Knoten wird entladen?", "Anhalten bei Toleranz"}                # keine toten Regler bei festen Netzen


def test_hidden_slider_values_come_back_when_the_random_net_is_shown_again():
    at = _run(lambda a: a.session_state.__setitem__("density_slider", 80))
    at.session_state["net_select"] = "detour"
    at.run()
    at.session_state["net_select"] = "random"
    at.run()
    assert not at.exception and at.slider(key="density_slider").value == 80


def test_both_slider_levels():
    at = _run()
    assert _phase(at).max == 9 and _phase(at).value == 9                                # Start + 8 Phasen + Beweis
    _phase(at).set_value(3)
    at.run()
    assert not at.exception and _phase(at).value == 3 and _inner(at) is not None and _inner(at).value == _inner(at).max          # innen zunächst das Ende der Phase
    top = _inner(at).max
    _inner(at).set_value(1)
    at.run()
    assert _inner(at).value == 1 and any("schiebt" in c or "senkt seinen Preis" in c for c in _captions(at))
    _inner(at).set_value(0)
    at.run()
    assert any("Sättigen" in m.value for m in at.markdown) and any("Restkanten mit negativen reduzierten Kosten werden gesättigt" in c for c in _captions(at))
    _phase(at).set_value(4)
    at.run()
    assert _inner(at).value == _inner(at).max                                             # Phasenwechsel: innen wieder ans Ende
    at.session_state["net_select"] = "detour"
    at.run()
    assert not at.exception and _phase(at).value == _phase(at).max == 7                   # Start + 6 Phasen + Beweis
    assert top >= 2


def test_captions_for_start_phase_end_and_proof():
    at = _run()
    for k, needle in ((0, "Alle 76 Einheiten stehen in S als Überschuss"), (2, "**Ende von Phase 2:**"), (9, "es gibt keinen negativen Kreis")):
        _phase(at).set_value(k)
        at.run()
        assert not at.exception and any(needle in c for c in _captions(at)), (k, needle)
    at = _run(lambda a: _apply(a, C.PRESETS["🎚️ Grobe Toleranz"]))
    _phase(at).set_value(_phase(at).max)
    at.run()
    assert any("aber nicht sicher kostenminimal" in c for c in _captions(at))


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb Phase und Bild."""
    at = _run(lambda a: a.session_state.__setitem__("net_select", "assignment"))
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped_and_snapped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["density"] = "9999"
    at.query_params["spread"] = "abc"
    at.query_params["p"] = "-5"
    at.run()
    assert not at.exception
    assert at.slider(key="density_slider").value == C.DENSITY_MAX and at.slider(key="spread_slider").value == C.DEFAULT_SPREAD and at.slider(key="p_slider").value == C.P_MIN
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["density"] = "63"
    at.query_params["spread"] = "60"
    at.query_params["load"] = "94"
    at.query_params["alpha"] = "4"
    at.query_params["tol"] = "3"
    at.run()
    assert at.slider(key="density_slider").value == 60 and at.slider(key="spread_slider").value == 50 and at.slider(key="load_slider").value == 90
    assert at.radio(key="alpha_radio").value == 4 and at.radio(key="tol_radio").value == 3


def test_unknown_values_in_the_permalink_fall_back_to_the_defaults():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.query_params["selection"] = "lifo"
    at.query_params["alpha"] = "3"
    at.query_params["tol"] = "abc"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET
    assert at.radio(key="selection_radio").value == C.DEFAULT_SELECTION and at.radio(key="alpha_radio").value == C.DEFAULT_ALPHA and at.radio(key="tol_radio").value == C.DEFAULT_TOLERANCE


def test_randomize_moves_the_seed_but_not_the_distribution():
    at = _run()
    pick = lambda a: ([m.value for m in a.metric if m.label == "Phasen"][1], [m.value for m in a.metric if m.label == "Durchsuchte Kanten"][1], _metric(a, "Höchstens so viele wie SSP"))
    before = pick(at)
    seed_before = at.number_input(key="seed_input").value
    [b for b in at.sidebar.button if "Neues Netz" in b.label][0].click()
    at.run()
    assert not at.exception and at.number_input(key="seed_input").value != seed_before and pick(at) == before


def test_experiments_run_on_demand():
    at = _run()
    for text in ("Mittel über 20 feste Netze (α = 2)", "Mittel über 3 feste Netze je Größe", "Mittel über 3 Zuordnungen je Größe"):
        assert not any(text in c for c in _captions(at))
    for key in ("cost_range_start", "scaling_start", "assignment_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    text = " ".join(_captions(at))
    assert "Mittel über 20 feste Netze (α = 2)" in text and "Mittel über 3 feste Netze je Größe" in text and "Mittel über 3 Zuordnungen je Größe" in text
    assert "ab 166 Knoten weniger Kanten als SSP, mit α = 2 ab 306 Knoten" in text and "es gewinnt ab n = 80" in text


def test_experiments_on_a_fixed_net_show_hints_instead_of_dead_controls():
    at = _run(lambda a: a.session_state.__setitem__("net_select", "detour"))
    assert not [b for b in at.button if b.key in ("cost_range_start", "scaling_start")]
    assert sum("zufälliges Distributionsnetz wählen" in t for t in _texts(at)) >= 5


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert all(re.search(r'key=f?"[a-z_]+(_\{\w+\})?"', c) for c in calls), calls
    assert len(calls) == 13 and len(set(keys)) == 11 and {k for k in keys if keys.count(k) > 1} == {"state_map", "eps_chart"}      # Start und Phasen teilen sich die Namen (gegenseitig ausschließend)
    viz = (ROOT / "csc_visualization.py").read_text(encoding="utf-8")
    bodies = [b for b in viz.split(chr(10) + "def ") if b.startswith("build_")]
    assert "fixedrange=True" in viz and len(bodies) == 9 and all("_base(" in b or "_layout(" in b for b in bodies)


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_footer_is_verbatim():
    src = APP.read_text(encoding="utf-8")
    assert "https://sebastianhanisch.net/kontakt.html" in src and "Interesse an einer maßgeschneiderten Lösung für" in src and "Operations Research und Machine Learning" in src


def test_runtime_needs_only_numpy_pandas_plotly_streamlit():
    """Konvention der Konzepte-Wurzeln und -Stücke: Referenzbibliotheken (scipy, networkx) nur als Testorakel."""
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "scipy" not in req and "networkx" not in req
    for path in ROOT.glob("*.py"):
        assert not re.search(r"^\s*(import|from)\s+(scipy|networkx)\b", path.read_text(encoding="utf-8"), re.M), path.name
