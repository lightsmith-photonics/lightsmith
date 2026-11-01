"""Layout integration. The GDS tests need nothing extra; gdsfactory tests run with
``uv run --with gdsfactory pytest tests/test_layout.py`` and skip otherwise."""

import warnings

import numpy as np
import pytest

import lightsmith as ls
from lightsmith.layout import (
    from_gdsfactory_component,
    from_gdsfactory_netlist,
    read_gds,
    resolve_layout,
    resolved_device,
    write_gds,
)
from lightsmith.problems import DeviceProblem, GdsLayout

SI = {"kind": "constant_index", "n_real": 3.48}
SIO2 = {"kind": "constant_index", "n_real": 1.44}
LAYER_MAP = {"1/0": {"material": SI, "z_min_nm": 0.0, "z_max_nm": 220.0}}


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _signature(polys):
    """(centroid x, centroid y, area) per polygon, sorted: an order-independent fingerprint."""
    def centroid(p):
        x, y = p[:, 0], p[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = cross.sum() / 2
        return (x + np.roll(x, -1)) @ cross / (6 * a), (y + np.roll(y, -1)) @ cross / (6 * a)

    rows = sorted((round(float(centroid(p)[0]), 2), round(float(centroid(p)[1]), 2), round(_area(p), 1))
                  for p in polys)
    return np.asarray(rows)


def test_gds_round_trip_with_references(tmp_path):
    tri = np.array([[0.0, 0.0], [400.0, 0.0], [0.0, 200.0]])
    cells = {
        "unit": {"polygons": {(1, 0): [tri], "2/5": [np.array([[0, 0], [10, 0], [10, 10], [0, 10]])]}},
        "top": {"polygons": {(1, 0): [np.array([[-1000, -250], [1000, -250], [1000, 250], [-1000, 250]])]},
                "refs": [{"cell": "unit", "origin_nm": (5000, 0), "angle_deg": 90},
                         {"cell": "unit", "origin_nm": (0, 3000), "x_reflection": True, "magnification": 2},
                         {"cell": "unit", "origin_nm": (-8000, 0), "columns": 3, "rows": 2,
                          "spacing_nm": ((1000, 0), (0, 700))}]},
    }
    path = write_gds(tmp_path / "t.gds", cells)
    polys = read_gds(path)
    assert sorted(polys) == [(1, 0), (2, 5)]
    assert len(polys[(1, 0)]) == 1 + 1 + 1 + 6 and len(polys[(2, 5)]) == 8
    got = {tuple(np.round(p[1], 6)) for p in polys[(1, 0)]}  # second vertex of each polygon
    assert (5000.0, 400.0) in got  # rotated by 90 degrees: (400, 0) -> (0, 400), shifted
    assert (800.0, 3000.0) in got  # magnified by 2: (400, 0) -> (800, 0); reflection keeps y=0
    third = {tuple(np.round(p[2], 6)) for p in polys[(1, 0)]}
    assert (0.0, 3000.0 - 400.0) in third  # reflected about x before shifting: (0, 200) -> (0, -400)
    assert (-8000.0 + 2000.0, 700.0 + 200.0) in third  # AREF column 2, row 1
    with pytest.raises(ValueError, match="no cell 'nope'"):
        read_gds(path, "nope")


def test_resolve_layout_extrudes_layers(tmp_path):
    rect = np.array([[-2000, -250], [2000, -250], [2000, 250], [-2000, 250]], dtype=float)
    write_gds(tmp_path / "wg.gds", {"wg": {"polygons": {(1, 0): [rect], (99, 0): [rect]}}})
    box = {"name": "box", "geometry": {"kind": "box", "center_nm": (0, 0, -1000), "size_nm": (6000, 3000, 2000)},
           "material": SIO2}
    dev = DeviceProblem(structures=[box], layout=GdsLayout(path="wg.gds", layer_map=LAYER_MAP), background=SIO2,
                        domain_size_nm=(6000, 3000, 3000), wavelengths=[1550.0],
                        ports=[{"name": "o1", "center_nm": (-2000, 0, 110), "size_nm": (0, 2000, 2000), "direction": "+"}])
    with pytest.warns(UserWarning, match="99/0"):
        structs = resolve_layout(dev, base_dir=tmp_path)
    assert [s.name for s in structs] == ["box", "1/0#0"]
    prism = structs[1].geometry
    assert (prism.z_min_nm, prism.z_max_nm) == (0.0, 220.0)
    assert np.allclose(prism.profile.vertices_nm, rect)
    assert structs[1].material.n_real == 3.48
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flat = resolved_device(dev, base_dir=tmp_path)
    assert flat.layout is None and len(flat.structures) == 2


# --- gdsfactory ---------------------------------------------------------------


@pytest.fixture(scope="module")
def gf():
    gf = pytest.importorskip("gdsfactory")
    gf.gpdk.PDK.activate()
    return gf


def test_component_to_device_problem(gf):
    c = gf.components.mmi1x2()
    dev = from_gdsfactory_component(c, {gf.gpdk.LAYER.WG: {"material": SI, "z_min_nm": 0, "z_max_nm": 220}},
                                    background=SIO2, wavelengths=[1500.0, 1600.0], port_margin_nm=800, merge=True)
    assert isinstance(dev, DeviceProblem)
    ports = {p.name: p for p in dev.ports}
    assert set(ports) == {"o1", "o2", "o3"}
    assert ports["o1"].direction == "+" and ports["o2"].direction == "-" and ports["o3"].direction == "-"
    assert ports["o1"].center_nm == pytest.approx((-10_000.0, 0.0, 110.0))
    assert ports["o2"].center_nm == pytest.approx((15_500.0, 625.0, 110.0))
    assert ports["o1"].size_nm == pytest.approx((0.0, 500.0 + 1600.0, 220.0 + 1600.0))
    # Polygon areas match gdsfactory's own geometry (um^2 -> nm^2).
    ref = c.dup().get_polygons_points(merge=True, by="tuple")[(1, 0)]
    area_ref = sum(_area(np.asarray(p)) for p in ref) * 1e6
    area = sum(_area(np.asarray(s.geometry.profile.vertices_nm)) for s in dev.structures)
    assert area == pytest.approx(area_ref, rel=1e-9)
    assert dev.domain_size_nm[0] == pytest.approx(25_500.0 + 2000.0)
    # The problem survives a JSON round trip (agents pass it around as JSON).
    assert DeviceProblem.model_validate_json(dev.model_dump_json()) == dev


def test_builtin_gds_reader_matches_gdsfactory(gf, tmp_path):
    c = gf.components.mzi()  # hierarchical, with rotated and mirrored references
    path = c.write_gds(tmp_path / "mzi.gds")
    ours = read_gds(path)
    theirs = c.get_polygons_points(merge=False, by="tuple")
    assert sorted(ours) == sorted(theirs)
    for k, polys in theirs.items():
        ref = _signature([np.asarray(p) * 1000.0 for p in polys])
        assert np.allclose(_signature(ours[k]), ref, rtol=1e-9, atol=0.02), k


def _arm_length(problem, start):
    """Total waveguide/bend length from 'inst,port' through two-port instances."""
    links = {}
    for a, b in problem.connections:
        links[a], links[b] = b, a
    total, ref = 0.0, start
    while True:
        inst, port = links[ref].split(",")
        comp = problem.instances[inst]
        if comp.component not in ("waveguide", "bend"):
            return total
        s = comp.settings
        total += s.get("length_nm") or 0.0
        ref = f"{inst},{'o2' if port == 'o1' else 'o1'}"


def test_netlist_to_circuit_mzi(gf):
    c = gf.components.mzi(delta_length=40.0)
    wl = np.linspace(1500, 1600, 2001)
    guide = {"neff": 2.4, "ng": 4.2, "wl0_nm": 1550.0}
    p = from_gdsfactory_netlist(c, list(wl), defaults={"waveguide": guide, "bend": guide,
                                                     "mmi1x2": {"loss_db": 0.0}})
    assert {i.component for i in p.instances.values()} == {"waveguide", "bend", "mmi1x2"}
    res = ls.solve(p, device="cpu")
    assert res.warnings == []
    t = np.abs(res.s()[:, 1, 0]) ** 2
    dl = abs(_arm_length(p, "cp1,o2") - _arm_length(p, "cp1,o3"))
    assert dl == pytest.approx(40_000.0, rel=1e-3)  # gdsfactory's delta_length is the arm length difference
    mins = wl[1:-1][(t[1:-1] < t[:-2]) & (t[1:-1] < t[2:])]
    fsr = np.diff(mins)
    assert np.allclose(fsr, ((mins[1:] + mins[:-1]) / 2) ** 2 / (4.2 * dl), rtol=0.01)
    assert t.max() == pytest.approx(1.0, abs=1e-6)


def test_netlist_unmapped_cells_are_named(gf):
    p = from_gdsfactory_netlist(gf.components.mzi2x2_2x2(), [1550.0], defaults={"mmi2x2": {"loss_db": 0.0}})
    assert sorted(p.ports) == ["o1", "o2", "o3", "o4"]
    s = ls.solve(p, device="cpu").s()[0]
    assert np.allclose(s.conj().T @ s, np.eye(4), atol=1e-12)  # lossless 2x2 MZI is unitary
    netlist = {"instances": {"s1": {"component": "spiral_thing"}, "w": {"component": "straight"}},
               "nets": [], "ports": {}}
    with pytest.raises(ValueError, match=r"spiral_thing \(instances s1\)"):
        from_gdsfactory_netlist(netlist, [1550.0])
    p = from_gdsfactory_netlist(netlist, [1550.0], models={"spiral_thing": "waveguide"})
    assert p.instances["s1"].component == "waveguide"
