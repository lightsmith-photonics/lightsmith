import json
import math

import numpy as np
import pytest

import omniscatter as osc
from omniscatter.circuits import (
    NetlistError,
    circuit_sparameters,
    list_models,
    waveguide_from_modes,
)
from omniscatter.circuits.touchstone import read_touchstone, write_touchstone
from omniscatter.problems import CircuitProblem
from omniscatter.results import ModeData, ModeResult

NEFF, NG, WL0 = 2.4, 4.2, 1550.0


def wg(length, **kw):
    return {"component": "waveguide", "settings": {"length_nm": length, "neff": NEFF, "ng": NG, "wl0_nm": WL0, **kw}}


def upd(p, **kw):
    """Validated copy of a problem with some fields replaced (model_copy does not validate)."""
    return type(p).model_validate({**p.model_dump(), **kw})


def neff_of(wl):
    return NEFF + (NEFF - NG) * (wl - WL0) / WL0


def mzi(dl=20_000.0, wl=None, both_inputs=False):
    ports = {"in": "cp1,o1", "out_bar": "cp2,o4", "out_cross": "cp2,o3"}
    instances = {"cp1": {"component": "coupler"}, "cp2": {"component": "coupler"},
                 "top": wg(50_000.0 + dl), "bot": wg(50_000.0)}
    connections = [("cp1,o3", "top,o1"), ("top,o2", "cp2,o2"), ("cp1,o4", "bot,o1"), ("bot,o2", "cp2,o1")]
    if both_inputs:
        ports["in2"] = "cp1,o2"
    else:
        instances["term"] = {"component": "terminator"}
        connections.append(("cp1,o2", "term,o1"))
    return CircuitProblem(instances=instances, connections=connections, ports=ports,
                          wavelengths=list(wl if wl is not None else np.linspace(1500, 1600, 801)))


def ring_netlist(L, coupling, loss_db_per_cm=0.0, wl=None):
    return CircuitProblem(
        instances={"dc": {"component": "coupler", "settings": {"coupling": coupling}},
                   "loop": wg(L, loss_db_per_cm=loss_db_per_cm)},
        connections=[("dc,o3", "loop,o1"), ("loop,o2", "dc,o2")],
        ports={"in": "dc,o1", "out": "dc,o4"}, wavelengths=list(wl))


def allpass(wl, L, coupling, loss_db_per_cm):
    t = math.sqrt(1 - coupling)
    a = 10 ** (-loss_db_per_cm * L * 1e-7 / 20)
    phi = 2 * np.pi * neff_of(wl) * L / wl
    return (t - a * np.exp(1j * phi)) / (1 - t * a * np.exp(1j * phi))


def test_mzi_matches_analytic():
    p = mzi()
    res = osc.solve(p, method="circuit", device="cpu")
    wl = np.asarray(res.wavelengths_nm)
    s = res.s()
    phi_t = 2 * np.pi * neff_of(wl) * 70_000.0 / wl
    phi_b = 2 * np.pi * neff_of(wl) * 50_000.0 / wl
    bar = 0.5 * (np.exp(1j * phi_b) - np.exp(1j * phi_t))
    cross = 0.5j * (np.exp(1j * phi_b) + np.exp(1j * phi_t))
    i, ob, oc = res.ports.index("in"), res.ports.index("out_bar"), res.ports.index("out_cross")
    assert np.max(np.abs(s[:, ob, i] - bar)) < 1e-12
    assert np.max(np.abs(s[:, oc, i] - cross)) < 1e-12
    # Power: sin^2 / cos^2 of half the phase difference, and FSR = lambda^2 / (ng dL).
    assert np.allclose(np.abs(s[:, ob, i]) ** 2, np.sin((phi_t - phi_b) / 2) ** 2, atol=1e-12)
    tb = np.abs(s[:, ob, i]) ** 2
    mins = wl[1:-1][(tb[1:-1] < tb[:-2]) & (tb[1:-1] < tb[2:])]
    fsr = np.diff(mins)
    expect = ((mins[:-1] + mins[1:]) / 2) ** 2 / (NG * 20_000.0)
    assert np.allclose(fsr, expect, rtol=0.01)
    assert res.warnings == []


def test_ring_netlist_and_model_match_allpass_formula():
    R = 10_000.0
    L = 2 * np.pi * R
    wl = np.linspace(1540, 1560, 2001)
    loss = 3.0  # dB/cm
    p = ring_netlist(L, 0.1, loss, wl)
    s = osc.solve(p, device="cpu").s()[:, 1, 0]
    ref = allpass(wl, L, 0.1, loss)
    assert np.max(np.abs(s - ref)) < 1e-10
    model = CircuitProblem(instances={"r": {"component": "ring_single", "settings": {
        "radius_nm": R, "coupling": 0.1, "loss_db_per_cm": loss, "neff": NEFF, "ng": NG, "wl0_nm": WL0}}},
        ports={"in": "r,o1", "out": "r,o2"}, wavelengths=list(wl))
    sm = osc.solve(model, device="cpu").s()[:, 1, 0]
    assert np.max(np.abs(sm - ref)) < 1e-12
    # FSR = lambda^2 / (ng L) from adjacent resonances.
    T = np.abs(s) ** 2
    mins = wl[1:-1][(T[1:-1] < T[:-2]) & (T[1:-1] < T[2:])]
    assert len(mins) >= 2
    fsr = np.diff(mins)
    expect = ((mins[:-1] + mins[1:]) / 2) ** 2 / (NG * L)
    assert np.allclose(fsr, expect, rtol=0.01)


def test_ring_critical_coupling_extinction():
    m = 100
    L = m * WL0 / NEFF  # resonant exactly at 1550 nm
    loss = 5.0
    a = 10 ** (-loss * L * 1e-7 / 20)
    coupling = 1 - a**2  # t = a
    p = CircuitProblem(instances={"r": {"component": "ring_single", "settings": {
        "radius_nm": L / (2 * np.pi), "coupling": coupling, "loss_db_per_cm": loss, "neff": NEFF, "ng": NG}}},
        ports={"in": "r,o1", "out": "r,o2"}, wavelengths=[WL0, WL0 + 1.0])
    s = osc.solve(p, device="cpu").s()
    assert abs(s[0, 1, 0]) ** 2 < 1e-20  # critical coupling: zero transmission on resonance
    # Same with the netlist built from a coupler and a waveguide.
    s2 = osc.solve(ring_netlist(L, coupling, loss, [WL0]), device="cpu").s()
    assert abs(s2[0, 1, 0]) ** 2 < 1e-20
    # Under/over-coupled: finite extinction.
    p2 = upd(p, instances={"r": {"component": "ring_single", "settings": {
        "radius_nm": L / (2 * np.pi), "coupling": coupling / 4, "loss_db_per_cm": loss, "neff": NEFF, "ng": NG}}})
    assert abs(osc.solve(p2, device="cpu").s()[0, 1, 0]) ** 2 > 1e-3


def _lossless_network(wl):
    """Two-input interferometer with a ring and a feedback loop: lossless, 4 external ports."""
    return CircuitProblem(
        instances={"c1": {"component": "coupler", "settings": {"coupling": 0.3, "coupling_dispersion_per_nm": 2e-3}},
                   "c2": {"component": "mmi2x2", "settings": {"loss_db": 0.0, "imbalance_db": 0.4}},
                   "c3": {"component": "coupler", "settings": {"coupling": 0.7}},
                   "a": wg(30_000.0), "b": wg(41_000.0), "fb": wg(25_000.0),
                   "ring": {"component": "ring_double", "settings": {"radius_nm": 8_000.0, "coupling": 0.2,
                                                                     "coupling_drop": 0.15}},
                   "ps": {"component": "phase_shifter", "settings": {"phase_per_volt": 0.3, "voltage": 2.0}}},
        connections=[("c1,o3", "a,o1"), ("a,o2", "ring,o1"), ("ring,o2", "c2,o2"), ("c1,o4", "b,o1"),
                     ("b,o2", "ps,o1"), ("ps,o2", "c2,o1"), ("c2,o3", "c3,o1"), ("c3,o3", "fb,o1"),
                     ("fb,o2", "ring,o3")],
        ports={"in1": "c1,o1", "in2": "c1,o2", "drop": "ring,o4", "out1": "c2,o4", "out2": "c3,o2", "out3": "c3,o4"},
        wavelengths=list(wl))


def test_unitarity_and_reciprocity():
    wl = np.linspace(1530, 1570, 201)
    for algorithm in ("dense", "reduce"):
        p = upd(_lossless_network(wl), options={"circuit": {"algorithm": algorithm}})
        res = osc.solve(p, device="cpu")
        s = res.s()
        eye = np.eye(s.shape[1])
        err = np.max(np.abs(np.conj(np.swapaxes(s, 1, 2)) @ s - eye))
        assert err < 1e-12, algorithm
        assert np.max(np.abs(s - np.swapaxes(s, 1, 2))) < 1e-12, algorithm
        assert res.provenance["algorithm"] == algorithm
    # MZI with both inputs exposed is a lossless 4-port.
    s = osc.solve(mzi(both_inputs=True, wl=np.linspace(1500, 1600, 51)), device="cpu").s()
    assert np.max(np.abs(np.conj(np.swapaxes(s, 1, 2)) @ s - np.eye(4))) < 1e-12


def test_dense_and_reduce_agree_on_lossy_circuit():
    wl = np.linspace(1540, 1560, 101)
    p = _lossless_network(wl)
    inst = dict(p.instances)
    inst["a"] = wg(30_000.0, loss_db_per_cm=300.0)
    p = upd(p, instances=inst)
    sd = osc.solve(upd(p, options={"circuit": {"algorithm": "dense"}}), device="cpu").s()
    sr = osc.solve(upd(p, options={"circuit": {"algorithm": "reduce"}}), device="cpu").s()
    assert np.max(np.abs(sd - sr)) < 1e-12
    sv = np.linalg.svd(sd, compute_uv=False)
    assert sv.max() <= 1 + 1e-12 and sv.min() < 0.95  # passive and lossy


def test_splitter_tree_power():
    p = CircuitProblem(
        instances={"y0": {"component": "y_branch"}, "y1": {"component": "y_branch"}, "y2": {"component": "y_branch"},
                   "w1": wg(20_000.0), "w2": wg(20_000.0)},
        connections=[("y0,o2", "w1,o1"), ("w1,o2", "y1,o1"), ("y0,o3", "w2,o1"), ("w2,o2", "y2,o1")],
        ports={"in": "y0,o1", "out1": "y1,o2", "out2": "y1,o3", "out3": "y2,o2", "out4": "y2,o3"},
        wavelengths=[1550.0])
    s = osc.solve(p, device="cpu").s()[0]
    assert np.allclose(np.abs(s[1:, 0]) ** 2, 0.25, atol=1e-14)
    assert np.isclose(np.sum(np.abs(s[:, 0]) ** 2), 1.0, atol=1e-14)


def test_ring_double_drop_and_grating_passband():
    m = 80
    L = m * WL0 / NEFF
    p = CircuitProblem(instances={"r": {"component": "ring_double", "settings": {
        "radius_nm": L / (2 * np.pi), "coupling": 0.05}}},
        ports={"in": "r,o1", "thru": "r,o2", "add": "r,o3", "drop": "r,o4"}, wavelengths=[WL0, 1551.3])
    s = osc.solve(p, device="cpu").s()
    assert abs(s[0, 3, 0]) ** 2 > 1 - 1e-12  # symmetric lossless add-drop: full drop on resonance
    assert abs(s[0, 1, 0]) ** 2 < 1e-12
    assert np.allclose(np.conj(np.swapaxes(s, 1, 2)) @ s, np.eye(4), atol=1e-12)
    g = CircuitProblem(instances={"gc": {"component": "grating_coupler", "settings": {
        "peak_loss_db": 2.5, "wl0_nm": 1550, "bandwidth_1db_nm": 30}}},
        ports={"wg": "gc,o1", "fiber": "gc,o2"}, wavelengths=[1550.0, 1565.0, 1535.0])
    t = osc.solve(g, device="cpu").s()[:, 1, 0]
    loss_db = -20 * np.log10(np.abs(t))
    assert np.allclose(loss_db, [2.5, 3.5, 3.5], atol=1e-12)


def test_gradient_matches_finite_difference_through_ring_resonance():
    torch = pytest.importorskip("torch")
    m = 100
    L = m * WL0 / NEFF
    p = CircuitProblem(
        instances={"dc": {"component": "coupler", "settings": {"coupling": 0.05}},
                   "loop": wg(L, loss_db_per_cm=2.0)},
        connections=[("dc,o3", "loop,o1"), ("loop,o2", "dc,o2")],
        ports={"in": "dc,o1", "out": "dc,o4"}, wavelengths=[WL0 + 0.05])  # on the resonance flank

    def transmission(neff, coupling):
        sol = circuit_sparameters(p, device="torch-cpu", params={"loop": {"neff": neff}, "dc": {"coupling": coupling}})
        return (sol["out", "in"].abs() ** 2)[0]

    neff = torch.tensor(NEFF, dtype=torch.float64, requires_grad=True)
    kap = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    T = transmission(neff, kap)
    T.backward()
    h = 1e-7
    fd_n = (transmission(torch.tensor(NEFF + h, dtype=torch.float64), 0.05)
            - transmission(torch.tensor(NEFF - h, dtype=torch.float64), 0.05)) / (2 * h)
    fd_k = (transmission(NEFF, torch.tensor(0.05 + h, dtype=torch.float64))
            - transmission(NEFF, torch.tensor(0.05 - h, dtype=torch.float64))) / (2 * h)
    assert abs(neff.grad.item()) > 10  # steep on the resonance flank
    assert neff.grad.item() == pytest.approx(fd_n.item(), rel=1e-5)
    assert kap.grad.item() == pytest.approx(fd_k.item(), rel=1e-5)
    # Gradients survive the reduce algorithm too.
    neff2 = torch.tensor(NEFF, dtype=torch.float64, requires_grad=True)
    sol = circuit_sparameters(p, device="torch-cpu", params={"loop": {"neff": neff2}}, algorithm="reduce")
    (sol["out", "in"].abs() ** 2)[0].backward()
    assert neff2.grad.item() == pytest.approx(fd_n.item(), rel=1e-5)


@pytest.mark.parametrize("n_ports", [1, 2, 3, 4])
def test_touchstone_round_trip(tmp_path, n_ports):
    rng = np.random.default_rng(n_ports)
    wl = np.linspace(1500, 1600, 7)
    s = rng.normal(size=(7, n_ports, n_ports)) + 1j * rng.normal(size=(7, n_ports, n_ports))
    names = [f"p{i}" for i in range(n_ports)]
    path = write_touchstone(tmp_path / f"dev.s{n_ports}p", wl, s, names)
    wl2, ports, s2 = read_touchstone(path)
    assert ports == names
    assert np.allclose(wl2, wl, rtol=1e-12)
    assert np.max(np.abs(s2 - s)) < 1e-11
    if n_ports == 2:  # the standard two-port column order S11 S21 S12 S22, conjugated (exp(+jwt))
        first = [float(x) for x in path.read_text().splitlines()[-1].split()]
        k = int(np.argmin(wl))  # last line = highest frequency = shortest wavelength
        assert first[3] == pytest.approx(s[k, 1, 0].real) and first[4] == pytest.approx(-s[k, 1, 0].imag)


def test_sparams_component_round_trip(tmp_path):
    wl = np.linspace(1500, 1600, 41)
    ref = osc.solve(mzi(wl=wl, both_inputs=True), device="cpu")
    write_touchstone(tmp_path / "mzi.s4p", ref.wavelengths_nm, ref.s(), ref.ports)
    (tmp_path / "mzi.json").write_text(ref.model_dump_json())
    for fname in ("mzi.s4p", "mzi.json"):
        p = CircuitProblem(instances={"dev": {"component": f"sparams:{fname}"}},
                           ports={n: f"dev,{n}" for n in ref.ports}, wavelengths=list(wl),
                           options={"circuit": {"base_dir": str(tmp_path)}})
        got = osc.solve(p, device="cpu")
        assert got.ports == ref.ports
        # JSON is exact; Touchstone goes through frequency (12 significant digits) and back.
        assert np.max(np.abs(got.s() - ref.s())) < (1e-14 if fname.endswith(".json") else 1e-8), fname
    # Interpolation between samples, and the out-of-range error.
    p = CircuitProblem(instances={"dev": {"component": f"sparams:{tmp_path / 'mzi.s4p'}",
                                          "settings": {"interpolation": "linear"}}},
                       ports={"a": "dev,in", "b": "dev,out_bar"}, wavelengths=[1700.0])
    with pytest.raises(ValueError, match="1500-1600 nm"):
        osc.solve(p, device="cpu")


def test_netlist_errors_name_ports():
    base = mzi(wl=[1550.0])
    cut = [c for c in base.connections if tuple(c) != ("bot,o2", "cp2,o1")]
    p = upd(base, connections=cut)
    with pytest.raises(NetlistError, match=r"unconnected port\(s\): 'cp2,o1', 'bot,o2'\. Connect them"):
        osc.solve(p, device="cpu")
    p = upd(base, ports={**base.ports, "again": "top,o2"})
    with pytest.raises(NetlistError, match=r"duplicate port use.*'top,o2' in connection #1 .* and external port 'again'"):
        osc.solve(p, device="cpu")
    p = upd(base, connections=[*cut, ("bot,o9", "cp2,o1")])
    with pytest.raises(NetlistError, match=r"instance 'bot' \(waveguide\) has no port 'o9' \(ports: o1, o2\)"):
        osc.solve(p, device="cpu")
    p = upd(base, instances={**base.model_dump()["instances"], "x": {"component": "flux_capacitor"}})
    with pytest.raises(ValueError, match="unknown component model.*flux_capacitor"):
        osc.solve(p, device="cpu")
    p = upd(base, instances={**base.model_dump()["instances"], "top": wg(-5.0)})
    with pytest.raises(NetlistError, match="instance 'top' .*length_nm"):
        osc.solve(p, device="cpu")
    # 'ignore' treats loose ports as matched ends and warns.
    p = upd(base, connections=cut, options={"circuit": {"unconnected": "ignore"}})
    res = osc.solve(p, device="cpu")
    assert any("bot,o2" in w for w in res.warnings)
    assert abs(res.s()[0, 2, 0]) ** 2 == pytest.approx(0.25)


def test_waveguide_from_mode_result():
    wl = [1500.0, 1550.0, 1600.0]
    neff = [2.50, 2.44, 2.38]
    modes = ModeResult(method="fde", backend="test", wavelengths_nm=wl,
                       modes=[[ModeData(neff_real=n, neff_imag=1e-6)] for n in neff])
    inst = waveguide_from_modes(modes, length_nm=1e5)
    assert inst.component == "waveguide" and inst.settings["neff_table"] == neff
    assert inst.settings["ng"] == pytest.approx(2.44 + 1550 * 0.0012)
    loss = inst.settings["loss_table_db_per_cm"][1]
    assert loss == pytest.approx(10 / math.log(10) * 4 * math.pi * 1e-6 / 1550e-7)
    p = CircuitProblem(instances={"w": inst.model_dump()}, ports={"a": "w,o1", "b": "w,o2"}, wavelengths=[1550.0])
    t = osc.solve(p, device="cpu").s()[0, 1, 0]
    assert np.angle(t) == pytest.approx(np.angle(np.exp(2j * np.pi * 2.44 * 1e5 / 1550)), abs=1e-9)
    assert -20 * np.log10(abs(t)) == pytest.approx(loss * 1e5 * 1e-7)
    single = ModeResult(method="fde", backend="test", wavelengths_nm=[1550.0],
                        modes=[[ModeData(neff_real=2.44, neff_imag=0.0, ng=4.1, loss_db_per_cm=1.5)]])
    s = waveguide_from_modes(single, 2e4).settings
    assert (s["neff"], s["ng"], s["loss_db_per_cm"]) == (2.44, 4.1, 1.5)


def test_models_are_documented_and_json_round_trips():
    models = {m["name"]: m for m in list_models()}
    for name in ("waveguide", "bend", "coupler", "mmi1x2", "mmi2x2", "y_branch", "phase_shifter", "ring_single",
                 "ring_double", "racetrack_single", "racetrack_double", "grating_coupler", "terminator",
                 "sparams:<path>"):
        assert models[name]["doc"] and "properties" in models[name]["settings_schema"], name
    p = mzi(wl=[1550.0])
    r1 = osc.solve(json.loads(p.model_dump_json()), device="cpu")
    r2 = osc.solve(p.model_dump_json(), device="cpu")
    assert r1.s_real == r2.s_real
    assert osc.estimate(p, device="cpu")["internal_ports"] == 10  # 4 arms + terminator


def test_phase_shifter_switches_mzi():
    torch = pytest.importorskip("torch")  # noqa: F841 (exercises the torch path of the models)
    p = mzi(dl=0.0, wl=[1550.0])
    inst = dict(p.instances)
    inst["ps"] = {"component": "phase_shifter", "settings": {"mechanism": "electro-optic",
                                                            "phase_per_volt": math.pi / 3.0, "length_nm": 0.0}}
    conns = [("cp1,o3", "top,o1"), ("top,o2", "ps,o1"), ("ps,o2", "cp2,o2"), ("cp1,o4", "bot,o1"),
             ("bot,o2", "cp2,o1"), ("cp1,o2", "term,o1")]
    for v, bar in ((0.0, 0.0), (3.0, 1.0), (1.5, 0.5)):
        inst["ps"]["settings"]["voltage"] = v
        q = upd(p, instances=inst, connections=conns)
        s = circuit_sparameters(q, device="torch-cpu").numpy()
        assert abs(s[0, 1, 0]) ** 2 == pytest.approx(bar, abs=1e-12)


@pytest.mark.gpu
def test_gpu_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    p = _lossless_network(np.linspace(1530, 1570, 101))
    cpu = osc.solve(p, device="cpu").s()
    gpu = osc.solve(p, device="cuda").s()
    assert np.max(np.abs(cpu - gpu)) < 1e-10
