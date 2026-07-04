"""config.py — TOML configuration, derived numerical coefficients, and the static
profile fields for the Beklemishev vortex-confinement model (Eqs. 7-8 of
Beklemishev et al., FST 57 (2010) 351).

Input is a GDB5B-style TOML file (see input.toml), NOT the Fortran namelist used by
the Hasegawa-Wakatani code.  This module produces a single `Config` object carrying:
  * the physics scalars           (U, H, kappa, nu4, nu5, nu4p, nu5p, ...)
  * the finite-difference coeffs  (rdx2d3, ... rakw)  — consumed by kernels.py
  * the spectral array            (rksq = 1/k^2)      — consumed by getphi (Poisson)
  * the STATIC 2-D profile fields (on the centred Cartesian grid, axis0=y, axis1=x):
        X, Y, r2         : coordinates and r^2 = x^2 + y^2  (box centred at r=0)
        phi_w            : wall-bias potential from the 11 concentric biasing rings
                           (outermost ring = limiter, held constant out to the box edge)
        nu5_field        : vorticity axial-loss rate, ×limiter_factor beyond r_lim
        nu5p_field       : pressure  axial-loss rate, ×limiter_factor beyond r_lim
        Qp_field         : pressure source term
        pres0            : initial pressure profile (+ seed perturbation)

Naming is unambiguous on purpose: `phi` = electrostatic potential, `vort` = Δφ =
vorticity (the evolved field), `pres` = pressure P.  (The paper writes `w` for the
potential and `Δw` for vorticity — the OPPOSITE of the HW code's `w`=vorticity — so
we avoid the bare name `w` entirely.)
"""
from __future__ import annotations

import tomllib
import numpy as np

from .backend import asarray


class Config:
    """Runtime configuration + derived coefficients + static profile fields."""

    def __init__(self, d: dict):
        dom = d.get("domain", {}); phy = d.get("physics", {})
        bias = d.get("bias", {}); ini = d.get("initial", {})
        tim = d.get("time", {})
        out = d.get("output", {}); rst = d.get("restart", {})

        # ---- domain / resolution ----
        self.nx0 = int(dom.get("nx", 512)); self.ny0 = int(dom.get("ny", 512))
        self.Lx = float(dom.get("Lx", 6.0)); self.Ly = float(dom.get("Ly", 6.0))

        # ---- input-unit conventions (physical -> normalized at load time) ----
        # See vortex_normalization.pdf Sec. 5.  Lengths -> /L0, voltages -> /phi_bar
        # (=T0), pressures -> /P_bar.  Defaults: lengths & pressure already
        # normalized; ring voltages given in physical VOLTS (converted here).
        self.length_units = str(dom.get("length_units", "L0")).lower()
        self.bias_units = str(bias.get("units", "volts")).lower()
        self.pres_units = str(ini.get("pres_units", "normalized")).lower()
        self._len_phys = self.length_units in ("m", "meter", "meters", "physical")
        self._volt_phys = self.bias_units in ("volts", "v", "physical")
        self._pres_phys = self.pres_units in ("pa", "pascal", "physical")

        # ---- Beklemishev physics scalars (Eqs. 7-8) ----
        self.U = float(phy.get("U", 0.0))                 # FLR (0 = Fig 5; -5 = Fig 6)
        self.H = float(phy.get("H", 10.0))                # plasma-wall line-tying
        self.kappa = float(phy.get("kappa", 1.0))         # magnetic curvature drive
        self.nu4 = float(phy.get("nu4", 0.002))           # vorticity perp diffusion
        self.nu5 = float(phy.get("nu5", 0.02))            # vorticity axial loss
        self.nu4p = float(phy.get("nu4p", 0.002))         # pressure perp diffusion
        self.nu5p = float(phy.get("nu5p", 0.02))          # pressure axial loss (base)
        self.limiter_radius = float(phy.get("limiter_radius", 1.5))
        self.limiter_factor = float(phy.get("limiter_factor", 20.0))
        # optional: flatten phi in the r>limiter shadow (models a conducting/equipotential
        # limiter; suppresses the square-box / periodic-BC m=4 imprint on the plasma).
        self.limiter_phi_clamp = bool(phy.get("limiter_phi_clamp", False))
        self.limiter_clamp_width = float(phy.get("limiter_clamp_width", 0.08))

        # (Grid-scale regularization is the PHYSICAL diffusion nu4·Δvort / nu4p·ΔP in
        #  the RHS — no separate numerical hyper-diffusion.)

        # ---- time stepping / output / restart ----
        self.dt = float(tim.get("dt", 4.0e-5))
        self.nts = int(tim.get("nts", 25000))
        self.nframes = int(tim.get("nframes", 100))
        self.nfdump = int(tim.get("nfdump", 10))
        self.outdir = str(out.get("outdir", "./out"))
        self.nrst = int(rst.get("nrst", 0))
        self.restartdir = str(rst.get("restartdir", self.outdir))
        self.restart_file = str(rst.get("restart_file", "restart.h5"))

        self._bias = bias
        self._ini = ini
        self._normalization(d.get("normalization", {}))
        self._apply_units()
        self._derive()
        self._build_profiles()

    # -----------------------------------------------------------------
    def _apply_units(self):
        """Convert physical inputs to normalized units, using the bases from
        _normalization (L0, phi_bar, P_bar).  Length inputs consumed downstream
        (Lx, Ly, limiter_radius here; ring_radii, pres_width in _build_profiles)
        are divided by L0; voltages by phi_bar; pressures by P_bar.  Idempotent
        factors are stored for the profile builder."""
        self._Lfac = self.L0 if self._len_phys else 1.0        # divide lengths by this
        self._Vfac = self.phi_bar if self._volt_phys else 1.0  # divide volts by this
        self._Pfac = self.P_bar if self._pres_phys else 1.0    # divide pressure by this
        if self._len_phys:
            self.Lx /= self._Lfac; self.Ly /= self._Lfac
            self.limiter_radius /= self._Lfac

    # -----------------------------------------------------------------
    def _normalization(self, nrm):
        """Physical reference bases from user-set B, n, T (for a realistic machine).

        The evolved Eqs. 7-8 are DIMENSIONLESS (U, H, kappa, nu... are given directly),
        so these bases do not change the dynamics — they set the physical <-> normalized
        conversion for interpreting inputs/outputs:
            potential  phi_bar = T0            [V]      (T/e in volts)
            pressure   P_bar   = n0 * T0 * e   [Pa]     (n0 T0)
            length     L0      = ref length    [m]      (defaults to rho_s)
            time       t_bar   = B0 L0^2 / phi_bar [s]  (E×B drift time, paper's tau)
            velocity   v_bar   = phi_bar/(B0 L0)   [m/s] (E×B reference velocity)
        """
        e = 1.602176634e-19; mp = 1.67262192e-27
        self.B0 = float(nrm.get("B0", 1.0))        # reference magnetic field  [T]
        self.n0 = float(nrm.get("n0", 1.0e19))     # reference density         [m^-3]
        self.T0 = float(nrm.get("T0", 100.0))      # reference temperature     [eV]
        self.Ai = float(nrm.get("Ai", 1.0))        # ion mass number
        self.Zi = float(nrm.get("Zi", 1.0))        # ion charge number
        L0 = float(nrm.get("L0", 0.0))             # reference length [m] (<=0 -> rho_s)
        mi = self.Ai * mp
        self.Omega_i = self.Zi * e * self.B0 / mi                   # ion gyrofreq [rad/s]
        self.cs = np.sqrt(self.Zi * self.T0 * e / mi)               # sound speed  [m/s]
        self.rho_s = self.cs / self.Omega_i                         # sound gyroradius [m]
        self.L0 = L0 if L0 > 0.0 else self.rho_s
        self.phi_bar = self.T0                                      # [V]
        self.P_bar = self.n0 * self.T0 * e                          # [Pa]
        self.v_bar = self.phi_bar / (self.B0 * self.L0)             # [m/s]
        self.t_bar = self.L0 / self.v_bar                           # [s]

    def print_normalization(self):
        print("*** Normalization bases (physical <-> normalized) ***")
        print(f"    B0      = {self.B0:.4g} T     n0 = {self.n0:.3e} /m^3   "
              f"T0 = {self.T0:.4g} eV  (Ai={self.Ai:g}, Zi={self.Zi:g})")
        print(f"    Omega_i = {self.Omega_i:.4e} rad/s   cs = {self.cs:.4e} m/s   "
              f"rho_s = {self.rho_s:.4e} m")
        print(f"    L0(len) = {self.L0:.4e} m   t_bar(time) = {self.t_bar:.4e} s   "
              f"v_bar = {self.v_bar:.4e} m/s")
        print(f"    phi_bar(potential) = {self.phi_bar:.4g} V   "
              f"P_bar(pressure) = {self.P_bar:.4e} Pa")
        print(f"    input units: lengths={'meters/L0' if self._len_phys else 'L0 (normalized)'}, "
              f"ring_volts={'VOLTS/phi_bar' if self._volt_phys else 'normalized'}, "
              f"pres_amp={'Pa/P_bar' if self._pres_phys else 'normalized'}")
        print(f"    => a ring voltage of X volts = X/{self.phi_bar:g} normalized; "
              f"dt(phys) = dt*{self.t_bar:.3e} s")

    # -----------------------------------------------------------------
    def _derive(self):
        """4th-order FD stencil coefficients + spectral FFT arrays (periodic)."""
        p = self
        # Lx, Ly are HALF-widths: the domain (and the periodic FFT period) spans
        # [-Lx, Lx] x [-Ly, Ly], so the full width is 2*Lx.
        Wx = 2.0 * p.Lx; Wy = 2.0 * p.Ly
        p.dx = Wx / p.nx0; p.dy = Wy / p.ny0
        p.rdx = 1.0 / p.dx; p.rdy = 1.0 / p.dy
        rdxsq = 1.0 / p.dx ** 2; rdysq = 1.0 / p.dy ** 2
        # first derivative (derx/dery)
        p.rdx2d3 = 2.0 * p.rdx / 3.0; p.rdy2d3 = 2.0 * p.rdy / 3.0
        p.rdxd12 = p.rdx / 12.0;      p.rdyd12 = p.rdy / 12.0
        # second derivative / Laplacian (delsq)
        p.rdxsq4d3 = 4.0 * rdxsq / 3.0; p.rdysq4d3 = 4.0 * rdysq / 3.0
        p.rdxsqd12 = rdxsq / 12.0;      p.rdysqd12 = rdysq / 12.0
        p.mu2s5d2 = 2.5 * rdxsq + 2.5 * rdysq
        p.rakw = 1.0 / (12.0 * p.dx * p.dy)                 # Arakawa prefactor
        # cell-centred 1-D coordinates: x in [-Lx+dx/2, Lx-dx/2] (r=0 at grid centre)
        p.x = (np.arange(p.nx0) + 0.5) * p.dx - p.Lx
        p.y = (np.arange(p.ny0) + 0.5) * p.dy - p.Ly

        # spectral: rfft2 layout (nx02 = nx0//2+1 columns), periodic Poisson Δφ=vort.
        # rksq = 1/k^2 (0 at k=0) is the only spectral array needed (by getphi).
        nx02 = p.nx0 // 2 + 1
        twopi = 2.0 * np.pi
        kx = (np.arange(nx02) / Wx) * twopi                 # period = 2*Lx
        ky = np.empty(p.ny0)
        half = p.ny0 // 2 + 1
        ky[:half] = (np.arange(half) / Wy) * twopi
        ky[half:] = ((np.arange(half, p.ny0) - p.ny0) / Wy) * twopi
        KX, KY = np.meshgrid(kx, ky)
        ksq = KX ** 2 + KY ** 2
        rksq = np.zeros_like(ksq); nz = ksq > 0.0; rksq[nz] = 1.0 / ksq[nz]
        p.rksq = asarray(rksq)

    # -----------------------------------------------------------------
    def _build_profiles(self):
        """Static 2-D fields on the centred (ny0, nx0) grid (axis0=y, axis1=x)."""
        p = self
        X, Y = np.meshgrid(p.x, p.y)          # (ny0, nx0): X varies along x, Y along y
        r2 = X ** 2 + Y ** 2
        r = np.sqrt(r2)
        p.X = asarray(X); p.Y = asarray(Y); p.r2 = asarray(r2)

        # --- 11-ring wall-bias potential phi_w(r) --------------------------------
        # Concentric rings with outer radii `ring_radii` (ascending) and applied
        # voltages `ring_volts`.  A point at radius r belongs to the first ring whose
        # outer radius >= r; beyond the outermost ring (the limiter) the voltage is
        # held constant to the square boundary (that ring extends to the corners).
        volts = np.asarray(self._bias.get(
            "ring_volts", [0.0] * 11), dtype=np.float64) / self._Vfac   # V -> phi/phi_bar
        radii = np.asarray(self._bias.get(
            "ring_radii", list(np.linspace(0.3, 3.0, len(volts)))),
            dtype=np.float64) / self._Lfac                             # m -> r/L0
        idx = np.clip(np.searchsorted(radii, r, side="left"), 0, volts.size - 1)
        p.phi_w = asarray(volts[idx])
        p.n_rings = int(volts.size)

        # --- limiter-enhanced axial loss (ν5, ν5p) outside r_lim -----------------
        mult = np.where(r > p.limiter_radius, p.limiter_factor, 1.0)
        p.nu5_field = asarray(p.nu5 * mult)
        p.nu5p_field = asarray(p.nu5p * mult)

        # --- optional limiter potential clamp: flatten phi(r>r_lim) toward its value
        #     at the limiter, with a smooth tanh transition (no hard discontinuity).
        #     wlim = 1 inside, 0 outside; ring_w = normalized annulus weights at r_lim
        #     used to take the (azimuthal+radial) mean phi that fills the shadow. ------
        rl = p.limiter_radius; wd = max(p.limiter_clamp_width, 1.0e-6)
        p.wlim = asarray(0.5 * (1.0 - np.tanh((r - rl) / (0.5 * wd))))
        _ring = np.exp(-((r - rl) / wd) ** 2)
        p.ring_w = asarray(_ring / _ring.sum())

        # --- pressure source Q_p (scalar or 'gaussian' profile) ------------------
        # Qp is a normalized rate (already dimensionless); widths are lengths (->/L0).
        Qp = self._ini.get("Qp", 0.0)
        if isinstance(Qp, str) and Qp == "gaussian":
            w = float(self._ini.get("Qp_width", self._ini.get("pres_width", 1.0))) / self._Lfac
            a = float(self._ini.get("Qp_amp", 0.0))
            p.Qp_field = asarray(a * np.exp(-r2 / (w * w)))
        else:
            p.Qp_field = asarray(np.full_like(r2, float(Qp)))

        # --- initial pressure profile + seed perturbation ------------------------
        # pres_amp -> /P_bar if physical (Pa); pres_width is a length -> /L0.
        ini = self._ini
        amp = float(ini.get("pres_amp", 1.0)) / self._Pfac
        wid = float(ini.get("pres_width", 1.0)) / self._Lfac
        ptype = str(ini.get("pres_type", "gaussian"))
        if ptype == "parabolic":
            base = amp * np.clip(1.0 - r2 / (wid * wid), 0.0, None)
        else:                                                       # gaussian (default)
            base = amp * np.exp(-r2 / (wid * wid))
        perturb = float(ini.get("perturb_amp", 1.0e-3))
        rng = np.random.default_rng(int(ini.get("seed", 12345)))
        base = base * (1.0 + perturb * rng.standard_normal(base.shape))
        p.pres0 = asarray(base)

    # -----------------------------------------------------------------
    @property
    def nx(self):  # alias used by some callers
        return self.nx0

    @property
    def ny(self):
        return self.ny0


def load_config(path: str) -> Config:
    with open(path, "rb") as fh:
        return Config(tomllib.load(fh))
