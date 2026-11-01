"""Component-to-circuit flow (lightsmith.pdk): technology, cells, simulations against references, fits,
the model cache, netlists by name, and an MZI circuit against a direct simulation of the whole layout.

All device simulations are 2D (plan view, effective-index materials of the generic 220 nm SOI stack)."""

import math

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.backends._fdfd.modes import slab_neff_analytic
from lightsmith.layout import read_gds
from lightsmith.pdk import Assembly, CELLS, Technology, generic_soi, get_component, get_technology
from lightsmith.pdk import flow as F
from lightsmith.pdk.cells import mzi_arm_lengths
from lightsmith.pdk.fitting import fit_sparameters
from lightsmith.pdk.technology import asymmetric_slab_neff, register_technology
from lightsmith.problems import CircuitProblem

TECH = generic_soi()
LAM = 1550.0


@pytest.fixture(autouse=True)
def _model_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTSMITH_PDK_CACHE", str(tmp_path / "pdk-cache"))
    F.clear_memory_cache()
    yield
    F.clear_memory_cache()


# --- technology --------------------------------------------------------------------------


def test_generic_soi_is_open_data_and_round_trips_as_json(tmp_path):
    t = generic_soi()
    for ld in t.layers.values():
        if ld.material is not None:
            assert ld.material.kind == "library"  # bundled refractiveindex.info data (CC0)
    assert t.cladding.kind == t.box.kind == "library"
    assert t.core_thickness_nm() == 220 and t.core_thickness_nm("SHALLOW_ETCH") == 150
    path = tmp_path / "tech.json"
    path.write_text(t.model_dump_json())
    assert get_technology(str(path)) == t
    other = Technology.model_validate({**t.model_dump(mode="json"), "name": "thin_soi",
                                       "layers": {**t.model_dump(mode="json")["layers"],
                                                  "WG": {**t.layers["WG"].model_dump(mode="json"),
                                                         "z_max_nm": 150.0}}})
    register_technology(other)
    assert get_technology("thin_soi").core_thickness_nm() == 150
    assert set(t.layer_map()) == {(1, 0), (2, 0), (3, 0)}  # drawing layers carry no material


def test_slab_effective_index_against_the_symmetric_slab_formula():
    """Equal box and cladding: the three-layer TE solution equals the symmetric slab (tolerance 1e-9)."""
    n = asymmetric_slab_neff(1.444, 3.4757, 1.444, 220.0, LAM, "te")
    assert n == pytest.approx(slab_neff_analytic(3.4757, 1.444, 220.0, LAM, "tm"), abs=1e-9)
    # Thinner core and air cladding lower neff.
    assert asymmetric_slab_neff(1.444, 3.4757, 1.0, 220.0, LAM) < n
    assert TECH.slab_neff(150, [LAM])[0] < TECH.slab_neff(220, [LAM])[0]
    ng = TECH.group_index([LAM])[0]
    assert 3.8 < ng < 4.3  # strip waveguide, 2D EIM


# --- cells, ports and layout --------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CELLS))
def test_cells_have_consistent_ports_and_write_gds(name, tmp_path):
    c = get_component(name)
    assert c.port_names == list(CELLS[name].ports)
    assert all(p.name.startswith("o") for p in c.ports)
    path = c.write_gds(tmp_path / f"{name}.gds", TECH)
    back = read_gds(path)
    flat = c.flat_polygons()
    for layer, polys in flat.items():
        key = TECH.gds_layer(layer)
        area = sum(abs(_area(p)) for p in polys)
        area_back = sum(abs(_area(p)) for p in back[key])
        assert area_back == pytest.approx(area, rel=1e-3)  # vertices snap to the 1 nm database grid


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_assembly_places_ports_face_to_face():
    asm = Assembly()
    asm.add("a", get_component("straight", length_nm=1000))
    asm.connect("b", get_component("bend_circular", radius_nm=2000), "o1", "a,o2")
    asm.connect("c", get_component("straight", length_nm=500), "o1", "b,o2")
    end = asm.port("c,o2")
    assert end.center == pytest.approx((3000.0, 2500.0), abs=1e-6)
    assert end.orientation_deg == pytest.approx(90.0)
    ring = get_component("ring_single")
    assert {tuple(sorted(c)) for c in ring.netlist["connections"]} >= {("bend_left,o2", "coupler,o2")}


def test_device_problems_in_3d_use_the_layer_stack():
    p = get_component("mmi1x2").device_problem(TECH, [LAM], dimension=3)
    assert p.dimension == 3
    zs = {(s.geometry.z_min_nm, s.geometry.z_max_nm) for s in p.structures}
    assert zs == {(0.0, 220.0)}
    assert all(port.size_nm[2] > 220 for port in p.ports)
    est = ls.estimate(p, "eme")
    assert est["axis"] == "x" and est["fits"]
    gc = get_component("grating_coupler").device_problem(TECH, [LAM])
    assert gc.options["fdfd"]["polarization"] == "tm"  # side view: TE strip mode has E normal to the plane
    assert ls.choose_method(gc)[0].name == "lightsmith-fdfd"


def test_mzi_layout_arm_lengths():
    p = {"delta_length_nm": 4000.0, "radius_nm": 3000.0}
    m = get_component("mzi", **p)
    ref, delay = mzi_arm_lengths(p)
    a, b = m.port("o1"), m.port("o2")
    assert b.center[1] == pytest.approx(a.center[1], abs=1e-6)
    straights = [i for i in m.instances if i.component.kind == "straight"]
    assert sum(i.component.settings["length_nm"] for i in straights if i.name == "ref") == pytest.approx(ref)
    assert delay - ref == pytest.approx(4000 + (2 * math.pi - 4) * 3000)


# --- simulations against references ----------------------------------------------------------


def test_straight_phase_eme_vs_mode_solve():
    """A device solve (EME) of a straight reproduces the slab mode phase (tolerance 1e-2 rad over 5 um)."""
    c = get_component("straight", length_nm=5000)
    mode = F.simulate(c, wavelengths=[1500, LAM, 1600])
    dev = F.simulate(c, wavelengths=[1500, LAM, 1600], direct=True)
    assert dev.backend == "lightsmith-eme"
    np.testing.assert_allclose(np.abs(dev.s()[:, 1, 0]), 1, atol=1e-4)
    assert np.abs(np.angle(dev.s()[:, 1, 0] / mode.s()[:, 1, 0])).max() < 1e-2


def test_fdfd_reference_planes_follow_the_component_not_the_grid():
    """Shifting a straight by half a grid cell changes nothing once FDFD's snapped port planes are moved
    back onto the port centres (phase tolerance 5e-3 rad; uncorrected the shift would be ~0.07 rad)."""
    s = get_component("straight", length_nm=3000)
    out = []
    for dx in (0.0, 13.0):
        asm = Assembly()
        asm.add("w", s, origin=(dx, 0.0))
        asm.expose("o1", "w,o1")
        asm.expose("o2", "w,o2")
        comp = asm.build("shifted", {"dx": dx})
        r = F.simulate(comp, wavelengths=[LAM], method="fdfd", direct=True, options={"fdfd": {"dl_nm": 26.0}})
        out.append(r.s()[0, 1, 0])
    assert abs(np.angle(out[1] / out[0])) < 5e-3


def test_mmi1x2_splits_50_50_with_low_loss():
    r = F.simulate("mmi1x2", wavelengths=[LAM])
    s = np.abs(r.s()[0]) ** 2
    assert r.backend == "lightsmith-eme" and r.ports == ["o1", "o2", "o3"]
    assert s[1, 0] == pytest.approx(s[2, 0], rel=1e-6)  # symmetric splitter
    assert s[1, 0] + s[2, 0] > 0.97  # excess loss < 0.13 dB
    assert s[0, 0] < 1e-3


def test_mmi2x2_is_a_3db_90_degree_coupler():
    r = F.simulate("mmi2x2", wavelengths=[LAM])
    S = r.s()[0]
    bar, cross = abs(S[3, 0]) ** 2, abs(S[2, 0]) ** 2
    assert bar + cross > 0.92
    assert abs(10 * np.log10(bar / cross)) < 1.0  # imbalance < 1 dB
    assert abs(abs(np.degrees(np.angle(S[2, 0] / S[3, 0]))) - 90) < 3


def _supermodes(n1, nc, w, g, lam):
    """Even/odd TE (E in plane) supermodes of two identical slabs (width w, gap g): field continuity of H_z and
    (1/eps) dH_z/dx at the four interfaces, solved by bisection. Marcuse (1991), ch. 1 / 3."""
    k0, e1, ec = 2 * np.pi / lam, n1**2, nc**2

    def f(ne, even):
        kap, gam = k0 * np.sqrt(n1**2 - ne**2), k0 * np.sqrt(ne**2 - nc**2)
        a = np.cosh(gam * g / 2) if even else np.sinh(gam * g / 2)
        b = (e1 / ec) * (gam / kap) * (np.sinh(gam * g / 2) if even else np.cosh(gam * g / 2))
        return (kap / e1) * (-a * np.sin(kap * w) + b * np.cos(kap * w)) + \
            (gam / ec) * (a * np.cos(kap * w) + b * np.sin(kap * w))

    out = []
    for even in (True, False):
        ns = np.linspace(n1 - 1e-9, nc + 1e-9, 4001)
        v = np.array([f(x, even) for x in ns])
        i = np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0][0]
        lo, hi = ns[i + 1], ns[i]
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.sign(f(mid, even)) == np.sign(f(lo, even)) else (lo, mid)
        out.append((lo + hi) / 2)
    return out


@pytest.mark.slow
def test_directional_coupler_follows_coupled_mode_theory():
    """Extra coupling length dL adds pi (n_even - n_odd) dL / lambda to the coupling angle
    theta = atan(|S_cross| / |S_bar|) (supermode / coupled-mode theory); the S-bends cancel in the
    difference. Tolerance 2%."""
    n1 = float(TECH.slab_neff(220, [LAM])[0])
    nc = float(np.real(TECH.cladding.n([LAM]))[0])
    ne, no = _supermodes(n1, nc, 500.0, 200.0, LAM)
    th = []
    for length in (1000.0, 3000.0):
        S = F.simulate("coupler", wavelengths=[LAM], length_nm=length).s()[0]
        th.append(math.atan2(abs(S[2, 0]), abs(S[3, 0])))
        assert abs(S[2, 0]) ** 2 + abs(S[3, 0]) ** 2 > 0.97
    assert th[1] - th[0] == pytest.approx(np.pi * (ne - no) * 2000.0 / LAM, rel=0.02)


def test_bends_are_low_loss_and_euler_is_not_worse():
    t = {}
    for name in ("bend_circular", "bend_euler"):
        r = F.simulate(name, wavelengths=[LAM], radius_nm=3000.0)
        assert r.backend == "lightsmith-fdfd"
        t[name] = abs(r.s()[0, 1, 0]) ** 2
    assert t["bend_circular"] > 0.97
    assert t["bend_euler"] > t["bend_circular"] - 2e-3


def test_crossing_is_symmetric_with_low_crosstalk():
    r = F.simulate("crossing", wavelengths=[LAM])
    S = r.s()[0]
    assert r.ports == ["o1", "o2", "o3", "o4"]
    assert abs(S[2, 0]) ** 2 > 0.85  # through
    assert abs(S[1, 0]) ** 2 < 1e-2 and abs(S[3, 0]) ** 2 < 1e-2  # crosstalk < -20 dB
    assert abs(S[2, 0] - S[0, 2]) < 1e-3  # reciprocal
    assert abs(abs(S[3, 1]) - abs(S[2, 0])) < 1e-2  # the two paths are identical by symmetry


def test_y_branch_and_taper():
    s = np.abs(F.simulate("y_branch", wavelengths=[LAM]).s()[0]) ** 2
    assert s[1, 0] == pytest.approx(s[2, 0], rel=1e-6)
    assert s[1, 0] + s[2, 0] > 0.9
    S = F.simulate("taper", wavelengths=[LAM], length_nm=10_000.0, width2_nm=2000.0).s()[0]
    assert abs(S[1, 0]) ** 2 > 0.98 and abs(S[1, 0] - S[0, 1]) < 1e-3


@pytest.mark.slow
def test_grating_coupler_peak_follows_phase_matching():
    """Vertical emission when lambda = period * n_avg(lambda), n_avg the fill-factor-weighted slab indices of
    the etched and unetched regions (first-order Bragg condition). Tolerance 4% on the peak wavelength."""
    gc = get_component("grating_coupler")
    p = gc.settings
    wl = np.linspace(1450, 1650, 9)
    r = F.simulate(gc, wavelengths=wl)
    assert r.backend == "lightsmith-fdfd" and r.ports == ["o1", "o2"]
    t = np.abs(r.s()[:, 1, 0]) ** 2
    assert np.all(np.sum(np.abs(r.s()) ** 2, axis=1) < 1.02)
    lam = LAM
    for _ in range(20):
        n_avg = p["fill_factor"] * TECH.slab_neff(220, [lam])[0] + (1 - p["fill_factor"]) * TECH.slab_neff(150, [lam])[0]
        lam = p["period_nm"] * n_avg
    peak = wl[np.argmax(t)]
    assert abs(peak - lam) / lam < 0.04
    assert 0.1 < t.max() < 0.9


# --- fits, cache and netlists -------------------------------------------------------------------


def test_fit_reproduces_samples_and_held_out_wavelengths():
    """Compact model of an MMI from 7 samples over 1500-1600 nm, checked against direct EME solves between the
    samples: transmission o1 -> o2, o3 within 5e-3; every entry, including the weak (|S| ~ 0.1) reflections
    between the outputs that vary fast with wavelength, within 3e-2 and within 3x the fit's own held-out
    estimate. At the samples the error is < 5e-3."""
    flow = F.FlowSettings(num_points=7)
    m = F.component_model("mmi1x2", flow)
    assert m.fit.fit_error < 5e-3
    mid = [1508.3, 1541.7, 1575.0]
    err = np.abs(m.evaluate(mid) - F.simulate("mmi1x2", wavelengths=mid).s())
    assert err[:, 1:, 0].max() < 5e-3
    assert err.max() < 3e-2
    assert err.max() < 3 * m.fit.validation_error
    with pytest.raises(ValueError, match="widen"):
        m.evaluate([1650.0])


def test_vector_fit_recovers_a_known_pole_residue_model():
    wl = np.linspace(1500, 1600, 21)
    x = (1 / wl - (1 / 1500 + 1 / 1600) / 2) / ((1 / 1500 - 1 / 1600) / 2)
    pole = 0.3 - 0.4j
    s = np.zeros((len(wl), 2, 2), complex)
    s[:, 1, 0] = s[:, 0, 1] = (0.5 + 0.1 / (x - pole)) * np.exp(1j * 7.0 * x)
    f = fit_sparameters(wl, s, "pole_residue", tol=1e-8)
    assert f.kind == "pole_residue" and f.fit_error < 1e-8
    assert np.min(np.abs(f.poles - pole)) < 1e-6


def test_models_are_cached_by_parameter_hash(monkeypatch):
    F.component_model("taper", length_nm=3000.0, flow={"num_points": 3})
    F.clear_memory_cache()

    def boom(*a, **k):
        raise AssertionError("simulated again although a cached model matches")

    monkeypatch.setattr(F, "simulate", boom)
    m = F.component_model("taper", length_nm=3000.0, flow={"num_points": 3})  # from disk
    assert m.result.backend == "lightsmith-eme"
    with pytest.raises(AssertionError):
        F.component_model("taper", length_nm=3100.0, flow={"num_points": 3})  # other parameters: other key
    k1 = F.model_key("taper", {"a": 1}, TECH, F.FlowSettings())
    assert k1 != F.model_key("taper", {"a": 1}, TECH, F.FlowSettings(method="fdfd"))
    assert k1 == F.model_key("taper", {"a": 1}, TECH, F.FlowSettings(device="cpu"))


def test_netlist_references_components_by_name_and_parameters():
    wl = {"start_nm": 1520, "stop_nm": 1580, "num": 4}
    flow = {"num_points": 3}
    prob = CircuitProblem(
        instances={"split": F.instance("mmi1x2", flow),
                   "up": {"component": "pdk:straight", "settings": {"length_nm": 20_000.0, **flow}},
                   "down": F.instance("straight", flow, length_nm=10_000.0)},
        connections=[("split,o2", "up,o1"), ("split,o3", "down,o1")],
        ports={"in": "split,o1", "a": "up,o2", "b": "down,o2"}, wavelengths=wl)
    r = ls.solve(prob)
    s = r.s()
    lam = np.asarray(r.wavelengths_nm)
    m = F.component_model("mmi1x2", flow)
    up = F.component_model("straight", flow, length_nm=20_000.0).evaluate(lam)[:, 1, 0]
    np.testing.assert_allclose(s[:, r.ports.index("a"), 0], m.evaluate(lam)[:, 1, 0] * up, atol=1e-9)
    # The straight's model (neff polynomial through 3 mode solves) keeps the phase of 20 um within 1e-2 rad.
    exact = np.exp(2j * np.pi * TECH.waveguide_neff(lam) * 20_000 / lam)
    assert np.abs(np.angle(up / exact)).max() < 1e-2
    assert "pdk:<cell>" in ls.circuits.model_names()


def test_ring_fsr_matches_the_group_index():
    """Ring (R = 3 um) built from FDFD-simulated coupler_ring and bends: FSR = lambda^2 / (ng 2 pi R) with the
    straight waveguide's group index (tolerance 2%: the bent mode's ng differs slightly)."""
    r_nm = 3000.0
    flow = {"num_points": 6, "wl_min_nm": 1530.0, "wl_max_nm": 1570.0}
    prob = CircuitProblem(instances={"ring": F.instance("ring_single", flow, radius_nm=r_nm, gap_nm=150.0)},
                          ports={"in": "ring,o1", "out": "ring,o2"},
                          wavelengths={"start_nm": 1530, "stop_nm": 1570, "num": 4001})
    res = ls.solve(prob)
    wl = np.asarray(res.wavelengths_nm)
    t = np.abs(res.s()[:, 1, 0]) ** 2
    dips = wl[1:-1][(t[1:-1] < t[:-2]) & (t[1:-1] < t[2:]) & (t[1:-1] < 0.9)]
    assert len(dips) >= 2
    fsr = np.diff(dips).mean()
    lam0 = dips.mean()
    expect = F.fsr_estimate(TECH, 2 * np.pi * r_nm, lam0)
    assert fsr == pytest.approx(expect, rel=0.02)


@pytest.mark.slow
def test_mzi_circuit_matches_direct_simulation():
    """Circuit of per-component FDFD simulations vs one FDFD simulation of the whole MZI layout, same grid
    step and 16x subpixel sampling: |S21|^2 within 0.15 at every wavelength (measured: max 0.12; fringe
    minima at the same wavelengths). Runs ~2 h on a shared GPU.

    What remains is physics the circuit leaves out by construction: the field is forced through the straight
    waveguide mode at every junction (e.g. two bends meeting directly), measured at ~0.03-0.06 rad and a few
    % of power per arm, plus residual grid-alignment neff differences between instances (~2e-3 at 16x; at 4x
    subpixel sampling ~7e-3, and the mismatch was 0.19)."""
    params = {"delta_length_nm": 2000.0, "radius_nm": 2500.0}
    mzi = get_component("mzi", **params)
    opts = {"fdfd": {"dl_nm": 30.0, "subsamples": 16}}
    flow = F.FlowSettings(wl_min_nm=1500, wl_max_nm=1600, num_points=5, method="fdfd", waveguide_model="device",
                          sim_options=opts)
    wl = np.linspace(1500, 1600, 5)
    circ = ls.solve(F.netlist_problem(mzi, wl.tolist(), flow))
    t_c = np.abs(circ.s()[:, 1, 0]) ** 2
    whole = F.simulate(mzi, wavelengths=wl, method="fdfd", direct=True, options=opts)
    t_d = np.abs(whole.s()[:, 1, 0]) ** 2
    assert t_d.max() - t_d.min() > 0.3  # a fringe is resolved
    np.testing.assert_allclose(t_c, t_d, atol=0.15)
