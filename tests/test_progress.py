"""solve(..., progress=callback): events, fractions, scaling across compare, and cancellation."""

import pytest

import lightsmith as ls
from lightsmith.progress import Cancelled, report, reporting, scaled


def _sim(wavelengths=(500.0, 600.0, 700.0), **dda):
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=ls.ConstantIndex(n_real=1.5))],
                         wavelengths=list(wavelengths), options={"dda": {"dipoles_per_diameter": 8, **dda},
                                                                 "fdtd": {"spacing_nm": 5.0}})


def test_report_is_a_no_op_without_a_callback():
    report("anything", 0.5)  # nothing installed: nothing happens
    seen = []
    with reporting(seen.append):
        report("a", 0.5, x=1)
    report("b")
    assert seen == [{"stage": "a", "fraction": 0.5, "x": 1}]


def test_scaled_maps_fractions_and_keeps_unknown_ones():
    seen = []
    cb = scaled(seen.append, 0.5, 1.0, part="b")
    cb({"stage": "s", "fraction": 0.5})
    cb({"stage": "s", "fraction": None})
    assert seen == [{"stage": "s", "fraction": 0.75, "part": "b"}, {"stage": "s", "fraction": None, "part": "b"}]
    assert scaled(None, 0, 1) is None


def test_dda_reports_every_wavelength_in_order():
    seen = []
    res = ls.solve(_sim(), "dda", "cpu", progress=seen.append)
    assert seen[0]["stage"] == "solve_started" and seen[0]["backend"] == "lightsmith-dda" and seen[0]["fraction"] == 0
    assert seen[-1]["stage"] == "solve_done" and seen[-1]["fraction"] == 1
    steps = [e for e in seen if e["stage"] == "dda_solve"]
    assert [e["wavelength_nm"] for e in steps] == res.wavelengths_nm
    fractions = [e["fraction"] for e in seen]
    assert fractions == sorted(fractions) and all(0 <= f <= 1 for f in fractions)


def test_a_backend_without_reports_still_says_start_and_done():
    seen = []
    ls.solve(_sim(), "mie", "cpu", progress=seen.append)
    assert [e["stage"] for e in seen] == ["solve_started", "solve_done"]


def test_bem_and_fdfd_report_per_wavelength():
    seen = []
    sim = _sim((500.0, 600.0)).model_copy(update={"options": {"bem": {"mesh_size_nm": 8}}})
    ls.solve(sim, "bem", "cpu", progress=seen.append)
    assert [e["wavelength_nm"] for e in seen if e["stage"] == "bem_solve"] == [500.0, 600.0]

    from lightsmith.mcp_server.kinds import EXAMPLES

    seen.clear()
    ls.solve({**EXAMPLES["device"], "wavelengths": [1540.0, 1560.0]}, "fdfd", "cpu", progress=seen.append)
    assert [e["wavelength_nm"] for e in seen if e["stage"] == "fdfd_solve"] == [1540.0, 1560.0]


def test_fdtd_reports_steps_with_a_fraction_only_when_the_end_is_known():
    seen = []
    ls.solve(_sim((600.0,)), "fdtd", "cpu", progress=seen.append)
    steps = [e for e in seen if e["stage"] == "fdtd_steps"]
    assert steps and all(e["fraction"] is None and e["max_steps"] >= e["steps"] for e in steps)
    assert [e["steps"] for e in steps] == sorted(e["steps"] for e in steps)

    seen.clear()
    sim = _sim((600.0,)).model_copy(update={"options": {"fdtd": {"spacing_nm": 5.0, "run_time_fs": 2.0}}})
    ls.solve(sim, "fdtd", "cpu", progress=seen.append)
    fractions = [e["fraction"] for e in seen if e["stage"] == "fdtd_steps"]
    assert fractions and all(0 < f <= 0.95 for f in fractions) and fractions == sorted(fractions)


def test_inverse_design_reports_iterations():
    pytest.importorskip("torch")
    from test_inverse import bend_spec

    seen = []
    ls.solve(bend_spec(max_iter=2, dl=60.0), "inverse", "cpu", progress=seen.append)
    its = [e for e in seen if e["stage"] == "iteration"]
    assert [e["iteration"] for e in its] == [0, 1] and "objective" in its[0] and its[-1]["max_iter"] == 2


def test_compare_shares_the_range_between_methods():
    seen = []
    ls.compare(_sim((600.0,)), ["mie", "dda"], "cpu", progress=seen.append)
    assert {e["compare_method"] for e in seen} == {"mie", "dda"}
    assert max(e["fraction"] for e in seen if e["compare_method"] == "mie") <= 0.5
    assert min(e["fraction"] for e in seen if e["compare_method"] == "dda") >= 0.5
    assert seen[-1]["fraction"] == 1


def test_a_callback_can_cancel():
    def stop(event):
        if event["stage"] == "dda_solve":
            raise Cancelled

    with pytest.raises(Cancelled):
        ls.solve(_sim(), "dda", "cpu", progress=stop)
    assert ls.solve(_sim(), "mie", "cpu").c_ext_nm2  # no callback left behind
    report("after")  # still a no-op


def test_the_callback_does_not_change_the_result():
    a = ls.solve(_sim(), "dda", "cpu")
    b = ls.solve(_sim(), "dda", "cpu", progress=lambda e: None)
    assert a.c_ext_nm2 == b.c_ext_nm2
