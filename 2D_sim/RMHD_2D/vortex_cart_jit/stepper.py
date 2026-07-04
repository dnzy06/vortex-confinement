"""stepper.py — Beklemishev vortex-confinement RHS + jit/scan trap-leapfrog.

Solves (Beklemishev et al., FST 57 (2010) 351, Eqs. 7-8), with vort = Δφ:

  (8) pressure :  ∂ₜP    = −{φ,P}    + ν₄ᵖ ΔP    − ν₅ᵖ(r) P + Q_p
  (7) vorticity:  ∂ₜvort = −{φ,vort} + U∇·{∇φ,P} + H(φ − φ_w) + κ{P,r²}
                           + ν₄ Δvort − ν₅(r) vort − UΔS_p        (S_p ≈ Q_p)

with φ recovered from vort each substep by the periodic FFT Poisson solve
(getphi: Δφ = vort).  Grid-scale regularization is the PHYSICAL diffusion
(ν₄ Δvort, ν₄ᵖ ΔP) in the RHS — no numerical hyper-diffusion.  {a,b} is the
Arakawa Jacobian `akw`.

The step mirrors the validated HW trap-leapfrog exactly (rotate registers →
predictor half-step → corrector full-step from the rotated level); only the RHS
and the fields differ.
"""
from __future__ import annotations

from types import SimpleNamespace

from .backend import jit, scan_lower, HAS_JIT
from .kernels import akw, delsq, derx, dery, getphi
from .state import State


def _kpars(cfg):
    """Namespace of the FD/FFT coefficients the copied kernels expect."""
    return SimpleNamespace(
        rdx2d3=float(cfg.rdx2d3), rdy2d3=float(cfg.rdy2d3),
        rdxd12=float(cfg.rdxd12), rdyd12=float(cfg.rdyd12),
        rdxsq4d3=float(cfg.rdxsq4d3), rdysq4d3=float(cfg.rdysq4d3),
        rdxsqd12=float(cfg.rdxsqd12), rdysqd12=float(cfg.rdysqd12),
        mu2s5d2=float(cfg.mu2s5d2), rakw=float(cfg.rakw),
        rksq=cfg.rksq,
        nx0=int(cfg.nx0), ny0=int(cfg.ny0),
    )


def make_step(cfg):
    """Build the jitted trap-leapfrog step bound to `cfg`."""
    kp = _kpars(cfg)
    dt = float(cfg.dt)
    H = float(cfg.H); kappa = float(cfg.kappa)
    nu4 = float(cfg.nu4); nu4p = float(cfg.nu4p)
    U = float(cfg.U)
    apply_U = abs(U) > 1.0e-12                             # FLR (Fig. 6) terms on/off
    # static profile device arrays (closed over → baked into the XLA graph)
    phi_w = cfg.phi_w; r2 = cfg.r2
    nu5_field = cfg.nu5_field; nu5p_field = cfg.nu5p_field; Qp = cfg.Qp_field
    # FLR source −U·Δ(S_p) with S_p ≈ Q_p(r) (paper approximation): static → precompute.
    U_source = (-U) * delsq(Qp, kp) if apply_U else None

    # optional limiter potential clamp: after each Poisson solve, flatten phi in the
    # r>limiter shadow toward its limiter-ring mean (kills the square-box m=4 imprint).
    apply_clamp = bool(getattr(cfg, "limiter_phi_clamp", False))
    wlim = getattr(cfg, "wlim", None); ring_w = getattr(cfg, "ring_w", None)

    def solve_phi(vort):
        phi = getphi(vort, kp)                            # periodic FFT Poisson Δφ = vort
        if apply_clamp:
            phi_ref = (phi * ring_w).sum()               # mean phi at the limiter (scalar)
            phi = phi * wlim + phi_ref * (1.0 - wlim)     # phi -> phi_ref beyond r_lim
        return phi

    def rhs(phi, vort, pres):
        # Eq. 8 — pressure
        presdot = (-akw(phi, pres, kp)
                   + nu4p * delsq(pres, kp)
                   - nu5p_field * pres
                   + Qp)
        # Eq. 7 — vorticity.  κ{P,r²} = κ·akw(P, r²) (same Arakawa scheme).
        vortdot = (-akw(phi, vort, kp)
                   + H * (phi - phi_w)
                   + kappa * akw(pres, r2, kp)
                   + nu4 * delsq(vort, kp)
                   - nu5_field * vort)
        if apply_U:
            # FLR polarization: +U·∇·{∇φ,P} = U·(∂ₓ{φₓ,P} + ∂_y{φ_y,P})
            # (identity ∇·{∇φ,P} = {Δφ,P} + stress; computed here directly from the
            #  Arakawa bracket of the potential gradients + a FD divergence — no 3rd
            #  derivatives).  Plus the static FLR source −U·Δ(S_p).
            flr = (derx(akw(derx(phi, kp), pres, kp), kp)
                   + dery(akw(dery(phi, kp), pres, kp), kp))
            vortdot = vortdot + U * flr + U_source
        return presdot, vortdot

    def step(state):
        vort, vorti, pres, presi, phi = state

        # ---- RHS at the current state ----
        pdot, wdot = rhs(phi, vorti, presi)

        # ---- predictor: rotate registers + half-step ----
        pres_new = presi
        presi_p = 0.5 * (pres + presi) + pdot * dt
        vort_new = vorti
        vorti_p = 0.5 * (vort + vorti) + wdot * dt
        phi_p = solve_phi(vorti_p)                        # Poisson (+ optional clamp)

        # ---- RHS at the predictor (midpoint) ----
        pdot_p, wdot_p = rhs(phi_p, vorti_p, presi_p)

        # ---- corrector: full step from the rotated level ----
        presi_c = pres_new + pdot_p * dt
        vorti_c = vort_new + wdot_p * dt
        phi_c = solve_phi(vorti_c)

        return State(vort=vort_new, vorti=vorti_c,
                     pres=pres_new, presi=presi_c, phi=phi_c)

    if HAS_JIT:
        step = jit(step)
    return step


def make_run_frame(cfg):
    """Build a jitted, scanned `run_frame(state) -> state` that runs cfg.nts steps
    as one XLA program (single GPU launch after compile)."""
    step = make_step(cfg)
    nts = int(cfg.nts)

    def run_frame(state):
        def body(s, _):
            return step(s), None
        return scan_lower(body, state, nts)

    if HAS_JIT:
        run_frame = jit(run_frame)
    return run_frame
