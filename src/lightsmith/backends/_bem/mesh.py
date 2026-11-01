"""Closed triangle meshes of the particle shapes, and their RWG edge structure.

Every surface is a closed, outward-oriented triangle mesh. A :class:`Mesh`
joins the surfaces of all particles (several per layered sphere) and numbers
the regions they separate: region 0 is the host medium, and each surface
records the region on its normal (outer) side and on its inner side.

RWG basis function m lives on the two triangles sharing edge m. On its plus
triangle it is ``l/(2A+) (r - p+)``, on its minus triangle ``l/(2A-) (p- - r)``,
with p the vertex opposite the edge (Rao, Wilton & Glisson, IEEE TAP 30, 409
(1982)). In local terms: on triangle t, the function attached to local edge i
(opposite vertex i) is ``sigma l/(2A) (r - v_i)`` with sigma = +1 or -1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...spec import Cylinder, LayeredSphere, Simulation, Sphere, Spheroid

# --- primitive surfaces -----------------------------------------------------


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    return v / np.linalg.norm(v, axis=1, keepdims=True), f


def _merge(verts: np.ndarray, tris: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and drop degenerate triangles."""
    scale = max(float(np.abs(verts).max()), 1e-300)
    key = np.round(verts / (scale * tol)).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    tris = inv[tris]
    ok = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return verts[first], tris[ok]


def unit_sphere(freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic sphere: icosahedron faces split into freq^2 triangles, projected to |r| = 1."""
    v0, f0 = _icosahedron()
    verts, tris = [], []
    idx = {}
    n = freq
    # barycentric grid on one face: point (i, j) with i + j <= n
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[i, j] = len(idx)
    loc = np.array([[i, j] for (i, j) in idx], dtype=float) / n
    local_tris = []
    for i in range(n):
        for j in range(n - i):
            local_tris.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            if i + j < n - 1:
                local_tris.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    local_tris = np.array(local_tris)
    for a, b, c in f0:
        p = v0[a] + loc[:, :1] * (v0[b] - v0[a]) + loc[:, 1:] * (v0[c] - v0[a])
        tris.append(local_tris + sum(len(x) for x in verts))
        verts.append(p / np.linalg.norm(p, axis=1, keepdims=True))
    return _merge(np.concatenate(verts), np.concatenate(tris))


def _sphere_freq(area: float, h: float) -> int:
    """Geodesic frequency giving mean edge length ~h on a surface of the given area."""
    return max(2, int(round(np.sqrt(area / (20 * np.sqrt(3) / 4 * h**2)))))


def sphere_surface(radius: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    v, t = unit_sphere(_sphere_freq(4 * np.pi * radius**2, h))
    return radius * v, t


def _spheroid_area(a: float, c: float) -> float:
    t = np.linspace(0, np.pi, 2001)
    return float(np.trapezoid(2 * np.pi * a * np.sin(t) * np.hypot(a * np.cos(t), c * np.sin(t)), t))


def spheroid_surface(a: float, c: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic sphere mapped to the spheroid, with polar angle reparametrised by
    meridian arc length so elongated or flattened bodies keep evenly sized triangles."""
    v, tris = unit_sphere(_sphere_freq(_spheroid_area(a, c), h))
    t_grid = np.linspace(0, np.pi, 4001)
    ds = np.hypot(a * np.cos(t_grid), c * np.sin(t_grid))
    s = np.concatenate([[0], np.cumsum((ds[1:] + ds[:-1]) / 2 * np.diff(t_grid))])
    theta = np.arccos(np.clip(v[:, 2], -1, 1))
    t = np.interp(theta / np.pi * s[-1], s, t_grid)
    phi = np.arctan2(v[:, 1], v[:, 0])
    out = np.stack([a * np.sin(t) * np.cos(phi), a * np.sin(t) * np.sin(phi), c * np.cos(t)], axis=1)
    return out, tris


def _zip_rings(a_idx, a_ang, b_idx, b_ang) -> list[tuple[int, int, int]]:
    """Triangulate the band between two closed rings of points given by azimuth."""
    if len(a_idx) == 1 or len(b_idx) == 1:
        c, ring = (a_idx[0], b_idx) if len(a_idx) == 1 else (b_idx[0], a_idx)
        return [(c, ring[j], ring[(j + 1) % len(ring)]) for j in range(len(ring))]
    oa, ob = np.argsort(a_ang), np.argsort(b_ang)
    a_idx, a_ang = np.asarray(a_idx)[oa], np.asarray(a_ang)[oa]
    b_idx, b_ang = np.asarray(b_idx)[ob], np.asarray(b_ang)[ob]
    na, nb = len(a_idx), len(b_idx)
    # unwrap both rings to start near the same angle and close with one extra point each
    shift = int(np.argmin(np.abs(np.angle(np.exp(1j * (b_ang - a_ang[0]))))))
    b_idx, b_ang = np.roll(b_idx, -shift), np.roll(b_ang, -shift)
    b_ang = np.unwrap(b_ang)
    a_ang = np.unwrap(a_ang)
    b_ang = b_ang - 2 * np.pi * np.round((b_ang[0] - a_ang[0]) / (2 * np.pi))
    a_ext = np.append(a_ang, a_ang[0] + 2 * np.pi)
    b_ext = np.append(b_ang, b_ang[0] + 2 * np.pi)
    i = j = 0
    out = []
    while i < na or j < nb:
        if j >= nb or (i < na and a_ext[i + 1] <= b_ext[j + 1]):
            out.append((a_idx[i % na], b_idx[j % nb], a_idx[(i + 1) % na]))
            i += 1
        else:
            out.append((a_idx[i % na], b_idx[j % nb], b_idx[(j + 1) % nb]))
            j += 1
    return out


def cylinder_surface(radius: float, length: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Flat-ended cylinder: concentric-ring caps (6i points on ring i) and a wall of
    staggered rings, sharing the rim points. Rings are graded towards the rims, where
    the field is singular at the sharp edge."""
    n_r = max(2, int(round(radius / h)))
    n_phi = 6 * n_r
    h_az = 2 * np.pi * radius / n_phi
    n_z = max(2, int(round(length / (h_az * np.sqrt(3) / 2))))
    # mild grading: blend uniform with Chebyshev-like spacing
    u = np.linspace(0, 1, n_z + 1)
    z = (0.5 * u + 0.5 * (1 - np.cos(np.pi * u)) / 2 - 0.5) * length
    rho = np.linspace(0, 1, n_r + 1)
    rho = 0.5 * rho + 0.5 * np.sin(np.pi * rho / 2)
    verts: list[np.ndarray] = []
    rings: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    def add_ring(key, r, zz, count, phase):
        ang = phase + 2 * np.pi * np.arange(count) / count
        start = sum(len(x) for x in verts)
        verts.append(np.stack([r * np.cos(ang), r * np.sin(ang), np.full(count, zz)], axis=1))
        rings[key] = (np.arange(start, start + count), np.mod(ang, 2 * np.pi))

    half = np.pi / n_phi
    for k in range(n_z + 1):
        add_ring(("wall", k), radius, z[k], n_phi, half * (k % 2))
    for side, k_rim in (("bot", 0), ("top", n_z)):
        rings[(side, n_r)] = rings[("wall", k_rim)]
        zz = z[k_rim]
        for i in range(n_r):
            add_ring((side, i), radius * rho[i], zz, max(1, 6 * i), half * (k_rim % 2) + (half * (i % 2)))
    tris: list[tuple[int, int, int]] = []
    for k in range(n_z):
        tris += _zip_rings(*rings[("wall", k)], *rings[("wall", k + 1)])
    for side in ("bot", "top"):
        for i in range(n_r):
            tris += _zip_rings(*rings[(side, i)], *rings[(side, i + 1)])
    return np.concatenate(verts), np.array(tris)


def orient_outward(verts: np.ndarray, tris: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Flip triangles whose normal points towards `centre` (valid for convex bodies)."""
    p = verts[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    flip = np.sum(nrm * (p.mean(axis=1) - centre), axis=1) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def enclosed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    p = verts[tris]
    return float(np.sum(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2]))) / 6)


# --- RWG mesh of a whole scene ----------------------------------------------


@dataclass
class Surface:
    verts: np.ndarray  # (V, 3), nm, scene frame
    tris: np.ndarray  # (T, 3), outward oriented
    outer: int  # region on the normal side
    inner: int  # region inside


@dataclass
class Mesh:
    surfaces: list[Surface]
    region_materials: list  # Material per region; index 0 (host) is None
    verts: np.ndarray = field(init=False)
    tris: np.ndarray = field(init=False)
    tri_surface: np.ndarray = field(init=False)
    edge_surface: np.ndarray = field(init=False)
    tri_edge: np.ndarray = field(init=False)  # (T, 3) edge attached to local edge i (opposite vertex i)
    tri_sign: np.ndarray = field(init=False)  # (T, 3) +1 on the plus triangle, -1 on the minus one
    edge_len: np.ndarray = field(init=False)
    edge_tri: np.ndarray = field(init=False)  # (N, 2) plus, minus triangle
    edge_loc: np.ndarray = field(init=False)  # (N, 2) local index on those triangles

    def __post_init__(self):
        verts, tris, tsurf, off = [], [], [], 0
        for s, surf in enumerate(self.surfaces):
            verts.append(surf.verts)
            tris.append(surf.tris + off)
            tsurf.append(np.full(len(surf.tris), s))
            off += len(surf.verts)
        self.verts = np.concatenate(verts)
        self.tris = np.concatenate(tris)
        self.tri_surface = np.concatenate(tsurf)
        T = len(self.tris)
        # local edge i joins vertices i+1, i+2
        a = self.tris[:, [1, 2, 0]].reshape(-1)
        b = self.tris[:, [2, 0, 1]].reshape(-1)
        key = np.minimum(a, b) * len(self.verts) + np.maximum(a, b)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        if len(ks) % 2 or np.any(ks[0::2] != ks[1::2]) or (len(ks) > 2 and np.any(ks[2::2] == ks[1:-1:2])):
            raise ValueError("BEM mesh is not a closed manifold (every edge must join exactly two triangles)")
        first, second = order[0::2], order[1::2]
        n_edge = len(first)
        self.tri_edge = np.empty(3 * T, dtype=np.int64)
        self.tri_sign = np.empty(3 * T)
        self.tri_edge[first] = np.arange(n_edge)
        self.tri_edge[second] = np.arange(n_edge)
        self.tri_sign[first], self.tri_sign[second] = 1.0, -1.0
        self.tri_edge = self.tri_edge.reshape(T, 3)
        self.tri_sign = self.tri_sign.reshape(T, 3)
        self.edge_tri = np.stack([first // 3, second // 3], axis=1)
        self.edge_loc = np.stack([first % 3, second % 3], axis=1)
        self.edge_len = np.linalg.norm(self.verts[a[first]] - self.verts[b[first]], axis=1)
        self.edge_surface = self.tri_surface[self.edge_tri[:, 0]]
        # sort edges by surface so each surface owns a contiguous range of unknowns
        perm = np.argsort(self.edge_surface, kind="stable")
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n_edge)
        self.tri_edge = inv[self.tri_edge]
        for name in ("edge_tri", "edge_loc", "edge_len", "edge_surface"):
            setattr(self, name, getattr(self, name)[perm])

    @property
    def n_edges(self) -> int:
        return len(self.edge_len)

    @property
    def n_regions(self) -> int:
        return len(self.region_materials)

    def tri_vertices(self) -> np.ndarray:
        return self.verts[self.tris]  # (T, 3, 3)

    def mean_edge_length(self) -> float:
        return float(self.edge_len.mean())

    def region_surfaces(self, r: int) -> list[tuple[int, float]]:
        """Surfaces bounding region r, with +1 if r is on their normal side, else -1."""
        out = []
        for s, surf in enumerate(self.surfaces):
            if surf.outer == r:
                out.append((s, 1.0))
            elif surf.inner == r:
                out.append((s, -1.0))
        return out


def _shape_surfaces(shape, h: float, volume_correction: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Surfaces of one particle in its own frame, innermost first."""
    if isinstance(shape, Sphere):
        specs = [("sphere", shape.radius_nm)]
    elif isinstance(shape, LayeredSphere):
        specs = [("sphere", r) for r in shape.radii_nm]
    elif isinstance(shape, Spheroid):
        specs = [("spheroid", (shape.semi_axis_xy_nm, shape.semi_axis_z_nm))]
    elif isinstance(shape, Cylinder):
        specs = [("cylinder", (shape.radius_nm, shape.length_nm))]
    else:  # pragma: no cover - guarded by supports()
        raise ValueError(f"BEM cannot mesh shape {shape.kind!r}")
    out = []
    for kind, p in specs:
        if kind == "sphere":
            v, t = sphere_surface(p, h)
            v_true = 4 / 3 * np.pi * p**3
        elif kind == "spheroid":
            v, t = spheroid_surface(*p, h)
            v_true = 4 / 3 * np.pi * p[0] ** 2 * p[1]
        else:
            v, t = cylinder_surface(*p, h)
            v_true = np.pi * p[0] ** 2 * p[1]
        t = orient_outward(v, t, np.zeros(3))
        if volume_correction:
            # The faceted surface is inscribed; rescale it to the true volume (radially only
            # for cylinders, whose flat ends are exact).
            ratio = v_true / enclosed_volume(v, t)
            if kind == "cylinder":
                v = v * np.array([np.sqrt(ratio), np.sqrt(ratio), 1.0])
            else:
                v = v * ratio ** (1 / 3)
        out.append((v, t))
    return out


def default_mesh_size(sim: Simulation, elements_per_wavelength: float, elements_per_size: float) -> float:
    """Mean edge length: resolve the smallest particle and the shortest wavelength (or skin depth) inside it."""
    wl = sim.wavelength_array()
    size = min(2 * p.shape.bounding_half_extent().min() for p in sim.particles)
    if isinstance(sim.particles[0].shape, LayeredSphere):
        size = min(size, min(s.thickness_nm for p in sim.particles for s in getattr(p.shape, "shells", [])) * 2)
    n_max = max(float(np.abs(np.sqrt(np.asarray(m.eps(wl), dtype=complex))).max())
                for p in sim.particles for m in p.region_materials())
    n_max = max(n_max, sim.medium_index)
    return float(min(size / elements_per_size, wl.min() / (n_max * elements_per_wavelength)))


def build_mesh(sim: Simulation, h: float, volume_correction: bool = True) -> Mesh:
    surfaces: list[Surface] = []
    materials: list = [None]
    for p in sim.particles:
        mats = p.region_materials()  # [core, shell 1, ...]
        first = len(materials)
        materials.extend(mats)
        pos = np.asarray(p.position_nm, dtype=float)
        surfs = _shape_surfaces(p.shape, h, volume_correction)
        for i, (v, t) in enumerate(surfs):
            inner = first + i
            outer = first + i + 1 if i + 1 < len(surfs) else 0
            surfaces.append(Surface(v + pos, t, outer, inner))
    return Mesh(surfaces, materials)


def check_overlaps(sim: Simulation, mesh: Mesh) -> str:
    """'' if particles are disjoint, else a message naming the first overlapping pair."""
    outer = [s for s in mesh.surfaces if s.outer == 0]
    for i, pi in enumerate(sim.particles):
        for j, pj in enumerate(sim.particles):
            if i == j:
                continue
            ri = np.linalg.norm(pi.shape.bounding_half_extent())
            rj = np.linalg.norm(pj.shape.bounding_half_extent())
            if np.linalg.norm(np.subtract(pi.position_nm, pj.position_nm)) > ri + rj:
                continue
            if np.any(pj.shape.contains(outer[i].verts - np.asarray(pj.position_nm))):
                return f"particles {i} and {j} overlap or touch; BEM needs disjoint particles"
    return ""
