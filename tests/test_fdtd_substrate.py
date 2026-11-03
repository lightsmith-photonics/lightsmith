"""FDTD scattering by particles on a substrate: normal incidence (layered TFSF fed by an auxiliary column) and
oblique incidence (tables of the exact discrete background, layered.py).

Reference: a small sphere as a point dipole (Mie a_1 polarizability) driven by the background wave (Fresnel)
and coupled to its own reflected field through Sommerfeld integrals (G_par, G_perp); exact in the dipole limit.

Measured (RTX 5070, float32), n = 2, r = 20 nm sphere in air above glass (n = 1.5), 400-800 nm:
- centre 40 nm above the glass, from above: C_sca within 0.17 % (dx = 2 nm) and 0.11 % (dx = 1.33 nm);
- centre 100 nm above: 0.23 % and 0.19 % (with 10 PML cells instead of 16: 1.3 % at 800 nm);
- through the glass (direction +z): 0.23 %.
A particle of the host's own index scatters exactly nothing (C_sca = 0 to 1e-6 nm^2); its C_abs is
float32 cancellation noise (~0.004 nm^2 for this box, exactly 0 in double precision).

Oblique incidence, same sphere 40 nm above the glass, dx = 2 nm: from above at 45 deg, s 0.17 %, p 0.33 %;
through the glass at 30 deg (p) 0.20 %; total internal reflection at 60 deg, s 0.47 %, p 0.7-1.7 % (the
evanescent field varies across the sphere, which the dipole model misses). Null particle in double precision:
C_sca <= 2e-14 nm^2 (TIR) and 2e-17 nm^2 (from above), C_abs <= 3e-5 nm^2.

Absorbing substrates (C_abs the particles' own absorption, C_sca the far field into the upper half-space, C_ext
the power taken from the reflected wave). Reference: the sphere's full Mie response to the incident and reflected
waves, plus its dipole's coupling to its own reflected field (Sommerfeld), radiating directly and by specular
reflection. A dipole-only model is off by the b_1 and quadrupole terms, ~3 % of the dipole amplitude at x = 0.3,
which C_ext (one direction, sensitive to the reactive response) and absorption feel at first order. Measured,
r = 20 nm, centre 100 nm above an absorbing dielectric (eps ~ 14 + 3i), dx = 2 nm, float32:
- n = 2: C_sca within 0.01-0.07 %, C_ext within 0.1-0.3 nm^2 (~5e-4 of its interference scale);
- lossy sphere (eps ~ 4.5 + 1i): C_abs and C_sca are off by the sphere's own discretisation (-0.7 % and -1.4 %
  in free space too); relative to the free-space run, within 0.1 %;
- centre 40 nm above (20 nm gap): within 0.3 % (the higher multipoles' own coupling to the substrate is missing
  from the reference);
- the volume absorption equals the net inflow through a closed host box around the sphere to 1e-6.
On a transparent substrate the same quantities (far field over both half-spaces, optical theorem of the reflected
and transmitted waves, volume absorption) match the box fluxes to 3e-4, including total internal reflection.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn, spherical_yn

import lightsmith as ls
from lightsmith.device import _torch_cuda
from lightsmith.spec import FarFieldRequest, NearFieldRequest, Outputs, Substrate

needs_cuda = pytest.mark.skipif(not _torch_cuda(), reason="no CUDA GPU with PyTorch")
GLASS = Substrate(material=ls.ConstantIndex(n_real=1.5))
DOWN = ls.PlaneWave(direction=(0, 0, -1), polarization=(1, 0, 0))
# absorbing substrates: a Drude metal (eps ~ -9.4 + 0.6i at 600 nm) and a dielectric absorber (eps ~ 14 + 3i)
GOLD = ls.Drude(eps_inf=9.5, wp_ev=9.0, gamma_ev=0.07)
ABSORBER = ls.DrudeLorentz(eps_inf=2.0, poles=[{"delta_eps": 8.0, "omega0_ev": 4.0, "gamma_ev": 1.0}])
LOSSY = ls.DrudeLorentz(eps_inf=3.0, poles=[{"delta_eps": 1.5, "omega0_ev": 3.0, "gamma_ev": 1.5}])  # ~4.5 + 1i


def _eps(material, wavelengths):
    from lightsmith.backends._fdtd.media import to_medium

    wl = np.asarray(wavelengths, dtype=float)
    return to_medium(material, wl, []).eps(2 * np.pi / wl)


def _a1(m, x):
    def psi(z):
        return z * spherical_jn(1, z)

    def dpsi(z):
        return spherical_jn(1, z) + z * spherical_jn(1, z, derivative=True)

    def h(z, d=False):
        return spherical_jn(1, z, derivative=d) + 1j * spherical_yn(1, z, derivative=d)

    xi, dxi = x * h(x), h(x) + x * h(x, True)
    return (m * psi(m * x) * dpsi(x) - psi(x) * dpsi(m * x)) / (m * psi(m * x) * dxi - xi * dpsi(m * x))


def _sommerfeld(k1, e1, e2, h, fn):
    """int_0^inf fn(s, s_z, r_s, r_p) e^{2 i k1 h s_z} ds, with fn carrying its 1/s_z (s = sin t, then cosh t;
    a metal's surface-plasmon pole is passed to quad as a break point)."""
    def f(s_, sz):
        kz2 = np.sqrt(complex(e2 / e1 - s_**2))
        rs, rp = (sz - kz2) / (sz + kz2), (e2 * sz - e1 * kz2) / (e2 * sz + e1 * kz2)
        return fn(s_, sz, rs, rp) * np.exp(2j * k1 * h * sz)

    def part(g, a, b, pts=None):
        kw = {"limit": 800, "points": pts} if pts else {"limit": 400}
        return quad(lambda t: g(t).real, a, b, **kw)[0] + 1j * quad(lambda t: g(t).imag, a, b, **kw)[0]

    total = part(lambda t: f(np.sin(t), np.cos(t)) * np.cos(t), 0, np.pi / 2)
    t_max = np.arcsinh(20 / (k1 * h))
    s_p = np.sqrt(complex(e2 / (e1 + e2))).real
    pts = [float(np.arccosh(s_p))] if 1 < s_p < np.cosh(t_max) else None
    return total + part(lambda t: f(np.cosh(t), 1j * np.sinh(t)) * np.sinh(t), 0, t_max, pts)


def dipole_reference(r, n_p, n1, n2, h, wavelengths, theta_deg=0.0, pol="s", from_below=False):
    """C_sca (nm^2) of a sphere centred h above the interface (host n1 above, n2 below), lit in the xz plane at
    theta from the normal of the incidence medium; s: E along y, p: in the xz plane; per unit incident intensity."""
    out = []
    th = np.radians(theta_deg)
    e1, e2 = n1**2, n2**2
    for lam in wavelengths:
        k0 = 2 * np.pi / lam
        k1, k2 = n1 * k0, n2 * k0
        kx = (n2 if from_below else n1) * k0 * np.sin(th)
        k1z, k2z = np.sqrt(complex(k1**2 - kx**2)), np.sqrt(complex(k2**2 - kx**2))
        if from_below:  # transmitted (or evanescent) wave in the host; incident intensity n2 |E|^2 / 2
            if pol == "s":
                E = np.array([0, 2 * k2z / (k2z + k1z) * np.exp(1j * k1z * h), 0])
            else:
                ph = (e2 / e1) * 2 * e1 * k2z / (e1 * k2z + e2 * k1z) * np.exp(1j * k1z * h) / k2
                E = np.array([k1z * ph, 0, -kx * ph])
            norm = n1 / n2
        else:
            dn, up = np.exp(-1j * k1z * h), np.exp(1j * k1z * h)
            if pol == "s":
                E = np.array([0, dn + (k1z - k2z) / (k1z + k2z) * up, 0])
            else:
                rp = (e2 * k1z - e1 * k2z) / (e2 * k1z + e1 * k2z)
                E = np.array([-(k1z / k1) * (dn - rp * up), 0, -(kx / k1) * (dn + rp * up)])
            norm = 1.0
        alpha = 6j * np.pi * _a1(n_p / n1, k1 * r) / k1**3
        g_par = k1**3 / (8 * np.pi) * 1j * _sommerfeld(k1, e1, e2, h, lambda s_, sz, rs, rp: s_ / sz * (rs - sz**2 * rp))
        g_perp = k1**3 / (4 * np.pi) * 1j * _sommerfeld(k1, e1, e2, h, lambda s_, sz, rs, rp: s_**3 / sz * rp)
        a_par, a_perp = alpha / (1 - alpha * g_par), alpha / (1 - alpha * g_perp)
        P = ((abs(a_par * E[0]) ** 2 + abs(a_par * E[1]) ** 2) * (1 + 6 * np.pi / k1**3 * g_par.imag)
             + abs(a_perp * E[2]) ** 2 * (1 + 6 * np.pi / k1**3 * g_perp.imag))
        out.append(norm * k1**4 / (6 * np.pi) * P)
    return np.array(out)


def _mie_ab(m, x, nmax):
    out = []
    for n in range(1, nmax + 1):
        def psi(z):
            return z * spherical_jn(n, z)

        def dpsi(z):
            return spherical_jn(n, z) + z * spherical_jn(n, z, derivative=True)

        hn = spherical_jn(n, x) + 1j * spherical_yn(n, x)
        dhn = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
        xi, dxi = x * hn, hn + x * dhn
        a = (m * psi(m * x) * dpsi(x) - psi(x) * dpsi(m * x)) / (m * psi(m * x) * dxi - xi * dpsi(m * x))
        b = (psi(m * x) * dpsi(x) - m * psi(x) * dpsi(m * x)) / (psi(m * x) * dxi - m * xi * dpsi(m * x))
        out.append((a, b))
    return out


def _mie_far(ab, k, kin, E, nhat):
    """Far amplitude f (E_sca ~ f e^{ikr} / r, phase at the centre) in directions nhat (n, 3) of a sphere lit by a
    plane wave along kin with amplitude vector E at its centre (Bohren & Huffman S_1, S_2)."""
    mu = np.clip(nhat @ kin, -1, 1)
    p0, p1 = np.zeros_like(mu), np.ones_like(mu)
    S1 = S2 = 0j
    for n, (a, b) in enumerate(ab, start=1):
        if n > 1:
            p0, p1 = p1, ((2 * n - 1) / (n - 1)) * mu * p1 - (n / (n - 1)) * p0
        tau = n * mu * p1 - (n + 1) * (p0 if n > 1 else 0 * mu)
        c = (2 * n + 1) / (n * (n + 1))
        S1, S2 = S1 + c * (a * p1 + b * tau), S2 + c * (a * tau + b * p1)
    perp = np.cross(kin, nhat)
    nrm = np.linalg.norm(perp, axis=1)
    alt = np.cross(kin, [1.0, 0.3, 0.1])
    perp = np.where(nrm[:, None] > 1e-12, perp / np.maximum(nrm, 1e-300)[:, None], alt / np.linalg.norm(alt))
    par_i, par_s = np.cross(perp, kin), np.cross(perp, nhat)
    return (1j / k) * ((S2 * (par_i @ E))[:, None] * par_s + (S1 * (perp @ E))[:, None] * perp)


def sphere_on_substrate(r, eps_p, n1, eps2, h, wavelengths, theta_deg=0.0, pol="s", dirs=None, nmax=5):
    """A sphere centred h above any substrate, lit from above in the xz plane at theta (s: E along y), per unit
    incident intensity: the full Mie response to the incident and reflected waves, plus the dipole's coupling to
    its own reflected field; the far field above is the direct plus the specularly reflected radiation.

    Returns (C_ext of the reflected wave, C_sca into the upper half-space, C_abs, scale, dC_sca/dOmega at dirs);
    scale = 4 pi / k |f(k_r)| |E_r| is the size of the interference terms that C_ext balances."""
    th = np.radians(theta_deg)
    e1 = n1**2
    x, wx = np.polynomial.legendre.leggauss(160)
    ct, wct = 0.5 * (x + 1), 0.5 * wx
    ph = 2 * np.pi * np.arange(64) / 64
    CT, PH = np.meshgrid(ct, ph, indexing="ij")
    up_dirs = np.stack([np.sqrt(1 - CT**2) * np.cos(PH), np.sqrt(1 - CT**2) * np.sin(PH), CT], -1).reshape(-1, 3)
    w_up = np.repeat(wct * 2 * np.pi / 64, 64)
    flip = np.array([1.0, 1.0, -1.0])
    out = []
    for lam, ep, e2 in zip(wavelengths, eps_p, eps2):
        k0 = 2 * np.pi / lam
        k1 = n1 * k0
        kx = k1 * np.sin(th)
        k1z, k2z = np.sqrt(complex(k1**2 - kx**2)), np.sqrt(complex(k0**2 * e2 - kx**2))
        rs = (k1z - k2z) / (k1z + k2z)
        rp = (e2 * k1z - e1 * k2z) / (e2 * k1z + e1 * k2z)
        dn, up = np.exp(-1j * k1z * h), np.exp(1j * k1z * h)
        kin, kr = np.array([np.sin(th), 0, -np.cos(th)]), np.array([np.sin(th), 0, np.cos(th)])
        if pol == "s":
            Ei, Er = np.array([0, 1, 0]) + 0j, np.array([0, rs, 0])
        else:
            Ei, Er = np.array([-(k1z / k1), 0, -(kx / k1)]), np.array([(k1z / k1) * rp, 0, -(kx / k1) * rp])
        waves = [(kin, Ei * dn), (kr, Er * up)]
        e_bg = Ei * dn + Er * up
        ab = _mie_ab(np.sqrt(complex(ep)) / n1, k1 * r, nmax)
        ab_hi = [(0j, b) if n == 0 else (a, b) for n, (a, b) in enumerate(ab)]
        alpha = 6j * np.pi * ab[0][0] / k1**3
        g_par = k1**3 / (8 * np.pi) * 1j * _sommerfeld(k1, e1, e2, h, lambda s_, sz, rs_, rp_: s_ / sz * (rs_ - sz**2 * rp_))
        g_perp = k1**3 / (4 * np.pi) * 1j * _sommerfeld(k1, e1, e2, h, lambda s_, sz, rs_, rp_: s_**3 / sz * rp_)
        p = alpha * e_bg / (1 - alpha * np.array([g_par, g_par, g_perp]))
        dp = p - alpha * e_bg

        def free(nh, coeffs=ab, coupled=True):
            f = sum(_mie_far(coeffs, k1, kw, Ew, nh) for kw, Ew in waves)
            return f + (k1**2 / (4 * np.pi) * (dp[None, :] - (nh @ dp)[:, None] * nh) if coupled else 0)

        def f_up(nh):
            c, st = nh[:, 2], np.sqrt(np.clip(1 - nh[:, 2] ** 2, 0, None))
            phi = np.arctan2(nh[:, 1], nh[:, 0])
            kz2 = np.sqrt(k0**2 * e2 - (k1 * st) ** 2 + 0j)
            rs_, rp_ = (k1 * c - kz2) / (k1 * c + kz2), (e2 * k1 * c - e1 * kz2) / (e2 * k1 * c + e1 * kz2)
            s_hat = np.stack([-np.sin(phi), np.cos(phi), 0 * phi], -1)
            t_hat = np.stack([c * np.cos(phi), c * np.sin(phi), -st], -1)
            t_img = np.stack([-c * np.cos(phi), -c * np.sin(phi), -st], -1)
            fd, fm = free(nh), free(nh * flip)
            a, b = np.exp(-1j * k1 * h * c), np.exp(1j * k1 * h * c)
            fs = np.sum(fd * s_hat, 1) * a + rs_ * np.sum(fm * s_hat, 1) * b
            fp = np.sum(fd * t_hat, 1) * a + rp_ * np.sum(fm * t_img, 1) * b
            return fs[:, None] * s_hat + fp[:, None] * t_hat

        c_sca = np.sum(w_up * np.sum(np.abs(f_up(up_dirs)) ** 2, 1))
        f_r = f_up(kr[None, :])[0]
        c_ext = 4 * np.pi / k1 * np.imag(f_r @ np.conj(Er))
        scale = 4 * np.pi / k1 * np.linalg.norm(f_r) * np.linalg.norm(Er)
        # absorption is diagonal in the multipoles: the coupled dipole, plus the higher orders driven by the
        # background (extinction of each wave minus their scattering over the whole sphere)
        c_abs = np.sum(np.abs(p) ** 2) * (k1 * np.imag(1 / np.conj(alpha)) - k1**4 / (6 * np.pi))
        all_dirs = np.concatenate([up_dirs, up_dirs * flip])
        c_abs += sum(4 * np.pi / k1 * np.imag(free(kw[None, :], ab_hi, False)[0] @ np.conj(Ew)) for kw, Ew in waves)
        c_abs -= np.sum(np.concatenate([w_up, w_up]) * np.sum(np.abs(free(all_dirs, ab_hi, False)) ** 2, 1))
        pattern = None if dirs is None else np.sum(np.abs(f_up(np.asarray(dirs, dtype=float))) ** 2, 1)
        out.append((c_ext, c_sca, c_abs, scale, pattern))
    c_ext, c_sca, c_abs, scale, pattern = zip(*out)
    return np.array(c_ext), np.array(c_sca), np.array(c_abs), np.array(scale), pattern


def oblique(theta_deg, pol, from_below=False):
    th = np.radians(theta_deg)
    k = np.array([np.sin(th), 0.0, np.cos(th) if from_below else -np.cos(th)])
    e = np.array([0.0, 1.0, 0.0]) if pol == "s" else np.cross([0.0, 1.0, 0.0], k)
    return ls.PlaneWave(direction=tuple(k), polarization=tuple(e))


def sim_on(sub, n_p=2.0, r=20.0, h=40.0, wl=(400.0, 500.0, 600.0, 800.0), illumination=DOWN, **fdtd):
    return ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=r), material=ls.ConstantIndex(n_real=n_p),
                                                position_nm=(0, 0, h))],
                         wavelengths=list(wl), illumination=illumination, substrate=sub, options={"fdtd": fdtd})


def test_dipole_reference_limits():
    wl = [400.0, 600.0]
    k = 2 * np.pi / np.array(wl)
    free = 6 * np.pi * np.abs(_a1(2.0, k * 20)) ** 2 / k**2  # Mie dipole term
    for theta, pol, below in ((0.0, "s", False), (0.0, "s", True), (37.0, "s", False), (37.0, "p", True)):
        np.testing.assert_allclose(dipole_reference(20, 2.0, 1.0, 1.0, 40, wl, theta, pol, below), free, rtol=1e-10)
    # normal incidence: s and p are the same wave
    np.testing.assert_allclose(dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, 0.0, "p"),
                               dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, 0.0, "s"), rtol=1e-12)


def test_discrete_layered_solver_matches_fresnel():
    """The exact discrete background (layered.solve) reflects as continuum Fresnel when dz << lambda, including
    total internal reflection, for s and p (measured |r| within 3e-5 at dz = 2 nm, 0 at 0.5 nm)."""
    from lightsmith.backends._fdtd.layered import layer_eta, solve
    from lightsmith.backends._fdtd.media import Medium

    host, sub, dz, lam = Medium(1.0), Medium(2.25), 1.0, 600.0
    dt = 0.5 * dz / np.sqrt(3)
    z = (np.arange(80) - 40) * dz
    om = np.array([2 * np.pi / lam])
    for from_top, theta in ((True, 30.0), (True, 60.0), (False, 30.0), (False, 60.0)):
        n1, n2 = (1.0, 1.5) if from_top else (1.5, 1.0)
        sx = n1 * np.sin(np.radians(theta))
        Kx, Om = 2 * np.sin(om * sx * dz / 2) / dz, 2 * np.sin(om * dt / 2) / dt
        eta_t = layer_eta(z, False, om, dt, dz, 1e-9, host, sub)
        eta_z = layer_eta(z + dz / 2, True, om, dt, dz, 1e-9, host, sub)
        kz1 = np.sqrt(complex((n1**2 - sx**2))) * om[0]
        kz2 = np.sqrt(complex((n2**2 - sx**2))) * om[0]
        kz2 = kz2 if kz2.imag >= 0 else -kz2
        for pol, e_t, r_ana, sign in (("s", (0.0, 1.0), (kz1 - kz2) / (kz1 + kz2), 1),
                                      ("p", (1.0, 0.0), (n2**2 * kz1 - n1**2 * kz2) / (n2**2 * kz1 + n1**2 * kz2), -1)):
            f = solve(eta_t, eta_z, Kx, np.zeros(1), Om, dz, e_t, from_top)
            end, i = (-1 if from_top else 0), (1 if pol == "s" else 0)
            comp = "Ey" if pol == "s" else "Ex"
            r_num = sign * (f[comp][0, end] / f["inc"][0, i] - 1) * np.exp(-2j * kz1 * abs(z[end]))
            assert abs(r_num - r_ana) < 2e-4, (from_top, theta, pol, r_num, r_ana)


def test_layered_background_meshes_the_plane_exactly():
    from lightsmith.backends._fdtd.media import Medium
    from lightsmith.backends._fdtd.substrate import layered_background

    host, sub = Medium(1.0), Medium(2.25)
    # nodes at z = 0, 1, 2, ...; the plane at z = 2.3 cuts E_x/E_y pixels at node 2 (f = 0.8) and E_z at 1.5 / 2.5
    inv, groups = layered_background((4, 4, 6), np.zeros(3), 1.0, 2.3, host, sub)
    assert groups == []
    assert inv[0][2, 2, 2] == pytest.approx(1 / (0.8 * 2.25 + 0.2))  # tangential: 1/<eps>
    assert inv[2][2, 2, 2] == pytest.approx(0.3 / 2.25 + 0.7)  # normal, pixel [2, 3], f = 0.3: <1/eps>
    assert inv[0][2, 2, 1] == pytest.approx(1 / 2.25) and inv[0][2, 2, 3] == 1.0


def test_substrate_support_and_refusals():
    fdtd = ls.get_backend("fdtd")
    assert fdtd.supports(sim_on(GLASS))[0]
    assert ls.choose_method(sim_on(GLASS))[0].method == "fdtd"
    assert not ls.get_backend("mie").supports(sim_on(GLASS))[0]
    assert fdtd.supports(sim_on(GLASS, illumination=oblique(40, "p")))[0]
    assert fdtd.supports(sim_on(GLASS, illumination=oblique(60, "s", from_below=True)))[0]  # TIR
    graze = ls.PlaneWave(direction=(0.999, 0, -0.0447), polarization=(0, 1, 0))
    assert "graze" in fdtd.supports(sim_on(GLASS, illumination=graze))[1]
    fitted = Substrate(material=ls.DrudeLorentz(eps_inf=2.25, poles=[]))
    assert "constant-index" in fdtd.supports(sim_on(fitted, illumination=oblique(30, "s", from_below=True)))[1]
    far = Outputs(far_field=FarFieldRequest(theta_deg=[0], wavelengths_nm=[500]))
    assert fdtd.supports(sim_on(GLASS).model_copy(update={"outputs": far}))[0]
    up = ls.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0))
    assert "constant-index" in fdtd.supports(sim_on(fitted, illumination=up).model_copy(update={"outputs": far}))[1]
    gold = Substrate(material=GOLD)
    assert fdtd.supports(sim_on(gold))[0]
    assert fdtd.supports(sim_on(gold, illumination=oblique(40, "p")))[0]
    assert "absorbing" in fdtd.supports(sim_on(gold, illumination=up))[1]


def test_substrate_tfsf_does_not_leak_cpu():
    """A sphere of the host medium on glass: the layered TFSF box must scatter nothing."""
    sim = sim_on(GLASS, n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), spacing_nm=4.0, pml_cells=6, padding_nm=8,
                 pml_gap_nm=8)  # small grid: the check needs no PML quality
    res = ls.solve(sim, "fdtd", "cpu")
    assert np.max(np.abs(res.c_sca_nm2)) < 1e-9
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-6  # float64
    assert res.provenance["substrate"]["incidence"] == "from above"


def test_oblique_substrate_tfsf_does_not_leak_cpu():
    """A sphere of the host medium on glass, lit at 40 degrees (s): the tables are the grid's own background,
    so nothing scatters, and the near field is |e^{-ik_z z} + r_s e^{ik_z z}|^2 above, |1 + r_s|^2 below
    (measured within 2e-4), uniform along x and with the phase k sin(theta) x."""
    lam, th = 600.0, np.radians(40)
    sim = sim_on(GLASS, n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), illumination=oblique(40, "s"), spacing_nm=4.0,
                 pml_cells=6, padding_nm=8, pml_gap_nm=8)
    sim = sim.model_copy(update={"outputs": Outputs(near_field=NearFieldRequest(
        plane="xz", offset_nm=0.0, half_width_nm=60.0, points=31, wavelengths_nm=[lam]))})
    res = ls.solve(sim, "fdtd", "cpu")
    assert np.max(np.abs(res.c_sca_nm2)) < 1e-12
    assert np.max(np.abs(res.c_abs_nm2)) < 1e-4  # measured ~1e-6 (float64)
    assert res.provenance["substrate"]["source"] == "layered tables"
    nf = res.near_field[0]
    z, enh = np.asarray(nf.axis2_nm), np.asarray(nf.enhancement)
    k0 = 2 * np.pi / lam
    k1z, k2z = k0 * np.cos(th), k0 * np.sqrt(2.25 - np.sin(th) ** 2)
    rs = (k1z - k2z) / (k1z + k2z)
    ana = np.where(z > 0, np.abs(np.exp(-1j * k1z * z) + rs * np.exp(1j * k1z * z)) ** 2, abs(1 + rs) ** 2)
    ok = np.isfinite(enh[15]) & (np.abs(z) > 4)
    assert ok.sum() > 15
    np.testing.assert_allclose(enh[15][ok], ana[ok], rtol=2e-3)
    np.testing.assert_allclose(enh[:, ok], np.broadcast_to(enh[15][ok], enh[:, ok].shape), rtol=1e-8)
    ey = np.asarray(nf.e_real)[1] + 1j * np.asarray(nf.e_imag)[1]
    x = np.asarray(nf.axis1_nm)
    step = np.angle(ey[1:, 25] / ey[:-1, 25])
    np.testing.assert_allclose(step, k0 * np.sin(th) * (x[1] - x[0]), atol=1e-6)


def test_substrate_near_field_is_the_standing_wave_cpu():
    """Around a sphere of the host medium the near field is the background: |e^{-ikz} + r e^{ikz}|^2 above the
    glass and |t|^2 below (measured within 0.03 % at dx = 2 nm); points beyond the column's range are NaN."""
    lam = 600.0
    sim = sim_on(GLASS, n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), spacing_nm=4.0, pml_cells=6, padding_nm=8,
                 pml_gap_nm=8)
    sim = sim.model_copy(update={"outputs": Outputs(near_field=NearFieldRequest(
        plane="xz", offset_nm=0.0, half_width_nm=60.0, points=31, wavelengths_nm=[lam]))})
    res = ls.solve(sim, "fdtd", "cpu")
    enh = np.asarray(res.near_field[0].enhancement)
    z = np.asarray(res.near_field[0].axis2_nm)
    k1, r = 2 * np.pi / lam, (1 - 1.5) / 2.5
    ana = np.where(z > 0, np.abs(np.exp(-1j * k1 * z) + r * np.exp(1j * k1 * z)) ** 2, (2 / 2.5) ** 2)
    ok = np.isfinite(enh[15]) & (np.abs(z) > 4)
    assert ok.sum() > 15
    np.testing.assert_allclose(enh[15][ok], ana[ok], rtol=5e-3)
    np.testing.assert_allclose(enh[:, ok], np.broadcast_to(enh[15][ok], enh[:, ok].shape), rtol=1e-9)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("h", [40.0, 100.0])
def test_sphere_above_glass_matches_dipole_reference(h):
    wl = (400.0, 500.0, 600.0, 800.0)
    res = ls.solve(sim_on(GLASS, h=h, wl=wl, spacing_nm=2.0), "fdtd", "gpu")
    ref = dipole_reference(20, 2.0, 1.0, 1.5, h, wl)
    np.testing.assert_allclose(res.c_sca_nm2, ref, rtol=5e-3)  # measured 0.17 % (h 40), 0.23 % (h 100)
    assert res.provenance["pml_cells"] == 16


@pytest.mark.gpu
@needs_cuda
def test_sphere_lit_through_glass_matches_dipole_reference():
    wl = (400.0, 500.0, 600.0, 800.0)
    up = ls.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0))
    res = ls.solve(sim_on(GLASS, wl=wl, illumination=up, spacing_nm=2.0), "fdtd", "gpu")
    np.testing.assert_allclose(res.c_sca_nm2, dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, from_below=True),
                               rtol=5e-3)  # measured 0.23 %
    assert res.provenance["substrate"]["incidence"] == "through the substrate"


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("theta, pol, below, rtol", [(45.0, "s", False, 5e-3), (45.0, "p", False, 8e-3),
                                                     (30.0, "p", True, 5e-3), (60.0, "s", True, 1e-2),
                                                     (60.0, "p", True, 3e-2)])
def test_oblique_sphere_above_glass_matches_dipole_reference(theta, pol, below, rtol):
    """Oblique s and p from above, through the glass, and total internal reflection (60 deg from the glass:
    the sphere sits in the evanescent field; the dipole model is least exact there)."""
    wl = (400.0, 500.0, 600.0, 800.0)
    res = ls.solve(sim_on(GLASS, wl=wl, illumination=oblique(theta, pol, below), spacing_nm=2.0), "fdtd", "gpu")
    np.testing.assert_allclose(res.c_sca_nm2, dipole_reference(20, 2.0, 1.0, 1.5, 40, wl, theta, pol, below),
                               rtol=rtol)


# --- absorbing substrates ------------------------------------------------------------


def test_sphere_on_substrate_reference_limits():
    """Without a substrate the multipole reference absorbs exactly as Mie, and reflects nothing."""
    wl = [400.0, 600.0]
    free = ls.solve(ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=20), material=LOSSY)],
                                  wavelengths=wl), "mie")
    c_ext, c_sca, c_abs, _, _ = sphere_on_substrate(20, _eps(LOSSY, wl), 1.0, np.ones(2) + 0j, 50, wl, 30.0, "p")
    np.testing.assert_allclose(c_abs, free.c_abs_nm2, rtol=1e-9)
    np.testing.assert_allclose(c_ext, 0, atol=1e-12)
    assert np.all(c_sca < np.asarray(free.c_sca_nm2))


def test_absorbing_substrate_null_particle_cpu():
    """A sphere of the host medium over an absorbing substrate scatters nothing, absorbs nothing and takes
    nothing from the reflected wave; its far field is zero (and exactly zero into the substrate)."""
    for ill in (DOWN, oblique(40, "s")):
        sim = sim_on(Substrate(material=ABSORBER), n_p=1.0, r=10.0, h=15.0, wl=(400.0, 800.0), illumination=ill,
                     spacing_nm=4.0, pml_cells=6, padding_nm=8, pml_gap_nm=8)
        sim = sim.model_copy(update={"outputs": Outputs(far_field=FarFieldRequest(
            theta_deg=[0, 60, 120, 180], phi_deg=[0, 90], wavelengths_nm=[600.0]))})
        res = ls.solve(sim, "fdtd", "cpu")
        assert res.provenance["substrate"]["absorbing"] and res.provenance["substrate"]["source"] == "layered tables"
        assert np.max(np.abs(res.c_sca_nm2)) < 1e-12
        assert np.all(np.asarray(res.c_abs_nm2) == 0)  # no lossy particle node
        assert np.max(np.abs(res.c_ext_nm2)) < 1e-4  # linear in the leak (~1e-8 of the field), measured 1e-5
        dcs = np.asarray(res.far_field[0].dcsca_domega_nm2).ravel()
        down = _directions(*ill.unit_vectors(), np.array([0, 60, 120, 180.0]), np.array([0, 90.0]))[:, 2] < 0
        assert down.any() and np.all(dcs[down] == 0) and np.max(dcs) < 1e-12


def test_transparent_substrate_far_field_quantities_match_box_cpu(monkeypatch):
    """The quantities reported over an absorbing substrate, computed for glass: volume absorption, the far field
    over both half-spaces and the optical theorem of the reflected and transmitted waves, against the box fluxes
    (measured within 3e-4 at dx = 2.5 nm, from above and through the glass, including total internal reflection)."""
    from lightsmith.backends._fdtd import scattering

    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=12), material=LOSSY, position_nm=(0, 0, 20))],
                        wavelengths=[450.0, 700.0], illumination=oblique(45, "p"), substrate=GLASS,
                        options={"fdtd": {"spacing_nm": 4.0, "pml_cells": 10}})
    box = ls.solve(sim, "fdtd", "cpu")
    monkeypatch.setattr(scattering, "_absorbing", lambda s: True)
    new = ls.solve(sim, "fdtd", "cpu")
    assert "particle_absorption_nodes" in new.provenance["substrate"]
    for k in ("c_ext_nm2", "c_sca_nm2", "c_abs_nm2"):
        np.testing.assert_allclose(getattr(new, k), getattr(box, k), rtol=2e-3, err_msg=k)


def test_particle_absorption_equals_host_box_inflow_cpu(monkeypatch):
    """A lossy sphere lifted over an absorbing substrate: its volume absorption, (Omega / 2) Im(E* D) over its
    nodes, is the net discrete Poynting inflow through a closed box around it in the host (float64)."""
    from lightsmith.backends._fdtd import scattering
    from lightsmith.backends._fdtd.absorption import ParticleAbsorption
    from lightsmith.backends._fdtd.engine import DFTMonitor
    from lightsmith.backends._fdtd.options import FDTDOptions
    from lightsmith.backends._fdtd.setup import flux, flux_points

    sim = ls.Simulation(particles=[ls.Particle(shape=ls.Sphere(radius_nm=12), material=LOSSY, position_nm=(0, 0, 30))],
                        wavelengths=[450.0, 700.0], illumination=oblique(45, "p"),
                        substrate=Substrate(material=ABSORBER),
                        options={"fdtd": {"spacing_nm": 4.0, "pml_cells": 6, "padding_nm": 8, "pml_gap_nm": 8}})
    lay = scattering.plan(sim, FDTDOptions(**sim.options["fdtd"]), False)["layout"]
    dx = lay.dx
    lo = np.floor((np.array([0, 0, 30.0]) - 12 - 2 * dx - lay.origin) / dx).astype(int)
    hi = np.ceil((np.array([0, 0, 30.0]) + 12 + 2 * dx - lay.origin) / dx).astype(int)
    assert lo[2] * dx + lay.origin[2] >= 2 * dx  # the box stays in the host
    held = {}

    class WithBox(ParticleAbsorption):
        def __init__(self, *a):
            super().__init__(*a)
            e, h1, h2, self.coef = flux_points(lo, hi, lay.shape, dx)
            self.box = DFTMonitor(self.omega, e, h1, h2)
            held["pa"] = self

        @property
        def monitors(self):
            return super().monitors + [self.box]

    monkeypatch.setattr(scattering, "ParticleAbsorption", WithBox)
    res = ls.solve(sim, "fdtd", "cpu")
    pa = held["pa"]
    inflow = -flux(*pa.box.results(), pa.coef)
    np.testing.assert_allclose(pa.power(dx**3), inflow, rtol=1e-6)
    assert np.all(np.asarray(res.c_abs_nm2) > 0)


def _directions(khat, ehat, theta_deg, phi_deg):
    th, ph = np.radians(theta_deg)[:, None], np.radians(phi_deg)[None, :]
    hhat = np.cross(khat, ehat)
    n = np.cos(th)[..., None] * khat + np.sin(th)[..., None] * (np.cos(ph)[..., None] * ehat
                                                                + np.sin(ph)[..., None] * hhat)
    return n.reshape(-1, 3)


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("theta, pol", [(0.0, "p"), (45.0, "p"), (45.0, "s")])  # p at 0: E along x, as DOWN
def test_sphere_above_absorber_matches_multipole_reference(theta, pol):
    """n = 2, r = 20 nm, centre 100 nm above the absorbing dielectric, dx = 2 nm: C_sca (upper half-space) and the
    far-field pattern, and C_ext (taken from the reflected wave; a balance of terms ~ scale, measured within 5e-4
    of it); C_abs is exactly 0."""
    wl = (400.0, 500.0, 600.0, 800.0)
    ill = DOWN if theta == 0 else oblique(theta, pol)
    th_ff, ph_ff = np.array([100.0, 130.0, 160.0, 180.0]), np.array([0.0, 90.0])
    sim = sim_on(Substrate(material=ABSORBER), h=100.0, wl=wl, illumination=ill, spacing_nm=2.0)
    sim = sim.model_copy(update={"outputs": Outputs(far_field=FarFieldRequest(
        theta_deg=[30.0, *th_ff], phi_deg=list(ph_ff), wavelengths_nm=[500.0]))})
    res = ls.solve(sim, "fdtd", "gpu")
    khat, ehat = ill.unit_vectors()
    dirs = _directions(khat, ehat, th_ff, ph_ff)
    c_ext, c_sca, _, scale, pattern = sphere_on_substrate(20, np.full(4, 4.0), 1.0, _eps(ABSORBER, wl), 100.0, wl,
                                                          theta, pol, dirs=dirs)
    np.testing.assert_allclose(res.c_sca_nm2, c_sca, rtol=2e-3)  # measured 0.01-0.07 %
    assert np.all(np.abs(np.asarray(res.c_ext_nm2) - c_ext) < 2e-3 * scale)
    assert np.all(np.asarray(res.c_abs_nm2) == 0)
    dcs = np.asarray(res.far_field[0].dcsca_domega_nm2)
    assert np.all(dcs[0] == 0)  # theta = 30 deg from the incident direction: into the absorber
    up = dirs[:, 2] > 0
    assert np.all(dcs[1:].ravel()[~up] == 0)
    np.testing.assert_allclose(dcs[1:].ravel()[up], pattern[1][up], rtol=5e-3)  # measured 0.2 %


@pytest.mark.gpu
@needs_cuda
def test_lossy_sphere_above_absorber_matches_multipole_reference():
    """A lossy sphere (eps ~ 4.5 + 1i) 100 nm above the absorber: C_abs, C_sca and C_ext relative to the same
    sphere in free space (whose own discretisation, -0.7 % in C_abs at dx = 2 nm, cancels) against the reference
    relative to Mie (measured within 0.1 %)."""
    wl = (400.0, 500.0, 600.0, 800.0)
    sim = sim_on(Substrate(material=ABSORBER), h=100.0, wl=wl, spacing_nm=2.0)
    part = ls.Particle(shape=ls.Sphere(radius_nm=20), material=LOSSY, position_nm=(0, 0, 100.0))
    sim = sim.model_copy(update={"particles": [part]})
    res = ls.solve(sim, "fdtd", "gpu")
    free_sim = sim.model_copy(update={"substrate": None})
    free, mie = ls.solve(free_sim, "fdtd", "gpu"), ls.solve(free_sim, "mie")
    c_ext, c_sca, c_abs, _, _ = sphere_on_substrate(20, _eps(LOSSY, wl), 1.0, _eps(ABSORBER, wl), 100.0, wl)
    corr = np.asarray(free.c_abs_nm2) / np.asarray(mie.c_abs_nm2)
    np.testing.assert_allclose(np.asarray(res.c_abs_nm2) / corr, c_abs, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(res.c_ext_nm2) / corr, c_ext, rtol=4e-3)
    np.testing.assert_allclose(res.c_sca_nm2 / (np.asarray(free.c_sca_nm2) / np.asarray(mie.c_sca_nm2)), c_sca,
                               rtol=3e-3)
    assert res.provenance["substrate"]["particle_absorption_nodes"] > 0


@pytest.mark.slow
@pytest.mark.gpu
@needs_cuda
def test_sphere_above_metal_matches_multipole_reference():
    """n = 2, r = 20 nm, centre 100 nm above a Drude metal (surface plasmons from 500 nm up), dx = 2 nm. "auto"
    runs it in double precision: in float32 the long metal ring-down left rounding noise, C_sca off by up to
    1.2 % and growing with the box. Measured: C_sca within 0.11-0.14 %, C_ext within 5e-4 of its scale at
    400 nm (gold barely reflects there) and 2e-5 above."""
    wl = (400.0, 500.0, 600.0, 800.0)
    res = ls.solve(sim_on(Substrate(material=GOLD), h=100.0, wl=wl, spacing_nm=2.0), "fdtd", "gpu")
    assert res.provenance["precision"] == "double"
    c_ext, c_sca, _, scale, _ = sphere_on_substrate(20, np.full(4, 4.0), 1.0, _eps(GOLD, wl), 100.0, wl, 0.0, "p")
    np.testing.assert_allclose(res.c_sca_nm2, c_sca, rtol=3e-3)
    assert np.all(np.abs(np.asarray(res.c_ext_nm2) - c_ext) < 2e-3 * scale)
