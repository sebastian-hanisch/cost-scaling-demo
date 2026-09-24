"""Presets: vollständig, in den Grenzen, und jedes Beispielnetz zeigt, was sein Hilfetext behauptet."""

import pytest

import csc_constants as C
import csc_evaluation as ev
import csc_presets as P
import csc_scenario as sc

KEYS = set(P.PRESET_KEYS)


def _net(p):
    return sc.build(p["net"], p["p"], p["d"], p["s"], p["density"], p["spread"], p["load"], p["seed"])


def _analyse(p):
    return ev.analyse(_net(p), p["alpha"], p["selection"], p["tol"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 8
    assert all(C.PRESET_HELP[name].strip() for name in C.PRESETS)
    for name, p in C.PRESETS.items():
        assert set(p) == KEYS, name


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_values_are_inside_the_bounds_and_on_the_step_grid(name):
    p = C.PRESETS[name]
    assert p["net"] in C.NETS and p["selection"] in C.SELECTION_LABELS and p["alpha"] in C.ALPHAS and p["tol"] in C.TOLERANCES
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        if spec.lo is not None:
            assert spec.lo <= p[key] <= spec.hi, (name, key)
    assert (p["density"] - C.DENSITY_MIN) % 10 == 0 and p["spread"] % 25 == 0 and (p["load"] - C.LOAD_MIN) % 10 == 0


def test_setting_specs_have_room_to_move():
    """Ein Regler mit lo == hi würde Streamlit abstürzen lassen."""
    assert all(spec.lo < spec.hi for spec in P.SETTING_SPECS.values() if spec.lo is not None)


def test_presets_use_seeds_outside_the_distribution_set():
    for name, p in C.PRESETS.items():
        assert p["seed"] not in C.DIST_SEEDS, name


def test_defaults_equal_the_random_net_preset():
    p = C.PRESETS["🚚 Zufallsnetz"]
    assert (p["net"], p["alpha"], p["selection"], p["tol"], p["p"], p["d"], p["s"], p["density"], p["spread"], p["load"], p["seed"]) == (
        C.DEFAULT_NET, C.DEFAULT_ALPHA, C.DEFAULT_SELECTION, C.DEFAULT_TOLERANCE, C.DEFAULT_P, C.DEFAULT_D, C.DEFAULT_S, C.DEFAULT_DENSITY, C.DEFAULT_SPREAD, C.DEFAULT_LOAD, C.DEFAULT_SEED)


def test_the_variant_presets_change_one_setting():
    base = C.PRESETS["🚚 Zufallsnetz"]
    for name, changed in (("⚡ α = 4", ("alpha",)), ("🔀 Beliebige Knotenwahl", ("selection",)), ("🏭 Werke knapp", ("load",)), ("🕸️ Dünnes Netz", ("density",)), ("🎚️ Grobe Toleranz", ("tol", "seed"))):
        assert all(C.PRESETS[name][k] == base[k] for k in base if k not in changed), name
    assert C.PRESETS["⚡ α = 4"]["alpha"] == 4 and C.PRESETS["🔀 Beliebige Knotenwahl"]["selection"] == "generic" and C.PRESETS["🎚️ Grobe Toleranz"]["tol"] == 5


def test_fixed_presets_hide_the_random_controls():
    assert {n for n, p in C.PRESETS.items() if p["net"] in C.FIXED_NETS} == {"🧭 Umweg", "💑 Zuordnung mit Kosten"}


def test_default_net_is_a_typical_draw():
    """Das Beispielnetz (Seed 155, dasselbe wie in der SSP- und Cycle-Canceling-Demo) ist größer als der Median (5190 gegen 3699 Kanten, SSP 824 gegen 540), aber beim Verhältnis Cost Scaling ÷ SSP typisch (±20 % des Medians); 8 Phasen, ganze Nachfrage."""
    p = C.PRESETS["🚚 Zufallsnetz"]
    dist = ev.distribution(p["p"], p["d"], p["s"], p["density"], p["spread"], p["load"])
    _, code, d = ev.verdict(_analyse(p))
    assert code == "optimal" and abs(d["scanned"] / d["ssp_scanned"] - dist["ratio_median"]) <= 0.2 * dist["ratio_median"] and d["phases"] == 8 and d["value"] == d["demand"] and p["seed"] == 155


def test_each_preset_shows_what_its_help_text_says():
    a = {name: _analyse(p) for name, p in C.PRESETS.items()}
    v = {name: ev.verdict(x) for name, x in a.items()}
    d = v["🚚 Zufallsnetz"][2]
    assert (d["phases"], d["eps0"], d["eps"][0], d["discharges"], d["scanned"], d["ssp_scanned"], d["costs"][0], d["total"]) == (8, 180, 90, 690, 5190, 824, 1199, 1197)
    q = v["⚡ α = 4"][2]
    assert (q["phases"], q["scanned"]) == (4, 3648)
    t = v["🎚️ Grobe Toleranz"]
    assert t[1] == "early" and (t[2]["total"], t[2]["optimal_total"], t[2]["gap_pct"], t[2]["scanned"], t[2]["phases"]) == (1027, 953, 7.8, 590, 1)
    assert [round(g, 1) for g in ev.verdict(ev.analyse(_net(C.PRESETS["🎚️ Grobe Toleranz"]), 2, "fifo", 0))[2]["gaps"][:4]] == [7.8, 1.7, 4.2, 0.0]
    assert v["🔀 Beliebige Knotenwahl"][2]["scanned"] == 4907
    k = v["🏭 Werke knapp"][2]
    assert (k["value"], k["demand"], k["total"]) == (89, 118, 1403)
    assert (v["🕸️ Dünnes Netz"][2]["scanned"], v["🕸️ Dünnes Netz"][2]["ssp_scanned"]) == (3031, 353)
    u = v["🧭 Umweg"][2]
    assert (u["discharges"], u["scanned"], u["ssp_scanned"], u["costs"][0], u["total"]) == (9, 84, 7, 9, 2)
    z = v["💑 Zuordnung mit Kosten"][2]
    assert (z["phases"], z["scanned"], z["ssp_scanned"]) == (7, 1147, 150)


def test_the_presets_show_both_good_and_bad_news():
    """Gute Nachricht: jede vollständige Rechnung ist kostenminimal, nach der ersten Phase schon nah dran; schlechte: auf allen kleinen Beispielnetzen durchsucht Cost Scaling mehr Kanten als SSP, und grob angehalten bleibt der Fluss zu teuer."""
    a = {name: ev.verdict(_analyse(p)) for name, p in C.PRESETS.items()}
    assert all(x[1] == "optimal" for n, x in a.items() if n != "🎚️ Grobe Toleranz") and a["🎚️ Grobe Toleranz"][0] == "info"
    assert all(x[2]["scanned"] > x[2]["ssp_scanned"] for n, x in a.items() if n != "🎚️ Grobe Toleranz")
