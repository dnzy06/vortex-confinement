"""Pure-functional numerical kernels for the 2-D vortex model.

Copied (unchanged math) from the validated Hasegawa-Wakatani `phw_gpu_jit` code,
trimmed to ONLY what the Beklemishev model uses:

  derx, dery   4th-order central first derivatives (periodic)
  delsq        4th-order Laplacian (periodic)
  akw          Arakawa J_T 9-point Jacobian  {a,b} = a_x b_y − a_y b_x
  getphi       periodic spectral Poisson solve  Δφ = vort  →  φ

All functions are PURE (inputs → new arrays, no mutation), composable under
`jax.jit` / `jax.lax.scan`.  Axis 0 = y, axis 1 = x.  Grid-scale regularization
comes from the PHYSICAL diffusion (ν₄ Δvort, ν₄ᵖ ΔP) in the RHS — there is no
extra numerical hyper-diffusion.
"""
from __future__ import annotations

from .backend import xp, rfft2, irfft2


# ---------------------------------------------------------------------
# Finite-difference derivatives (axis 0 = y, axis 1 = x), periodic BC
# ---------------------------------------------------------------------

def derx(a, pars):
    """4th-order x derivative."""
    am2 = xp.roll(a, +2, axis=1); am1 = xp.roll(a, +1, axis=1)
    ap1 = xp.roll(a, -1, axis=1); ap2 = xp.roll(a, -2, axis=1)
    return (ap1 - am1) * pars.rdx2d3 - (ap2 - am2) * pars.rdxd12


def dery(a, pars):
    """4th-order y derivative."""
    am2 = xp.roll(a, +2, axis=0); am1 = xp.roll(a, +1, axis=0)
    ap1 = xp.roll(a, -1, axis=0); ap2 = xp.roll(a, -2, axis=0)
    return (ap1 - am1) * pars.rdy2d3 - (ap2 - am2) * pars.rdyd12


def delsq(a, pars):
    """4th-order Laplacian."""
    axm2 = xp.roll(a, +2, axis=1); axm1 = xp.roll(a, +1, axis=1)
    axp1 = xp.roll(a, -1, axis=1); axp2 = xp.roll(a, -2, axis=1)
    aym2 = xp.roll(a, +2, axis=0); aym1 = xp.roll(a, +1, axis=0)
    ayp1 = xp.roll(a, -1, axis=0); ayp2 = xp.roll(a, -2, axis=0)
    return (pars.rdxsq4d3 * (axp1 + axm1)
            - pars.rdxsqd12 * (axp2 + axm2)
            + pars.rdysq4d3 * (ayp1 + aym1)
            - pars.rdysqd12 * (ayp2 + aym2)
            - pars.mu2s5d2 * a)


# ---------------------------------------------------------------------
# Arakawa J_T Jacobian (9-point, periodic) — energy/enstrophy conserving
# ---------------------------------------------------------------------

def _shift(a, ix, iy):
    """out[j,i] = a[j+iy, i+ix]  (periodic)."""
    out = a
    if ix != 0:
        out = xp.roll(out, -ix, axis=1)
    if iy != 0:
        out = xp.roll(out, -iy, axis=0)
    return out


def akw(a, b, pars):
    """Arakawa J_T bracket {a,b} = a_x b_y − a_y b_x (periodic BC)."""
    aE  = _shift(a,  1,  0); aW  = _shift(a, -1,  0)
    aN  = _shift(a,  0,  1); aS  = _shift(a,  0, -1)
    aNE = _shift(a,  1,  1); aNW = _shift(a, -1,  1)
    aSE = _shift(a,  1, -1); aSW = _shift(a, -1, -1)

    bE  = _shift(b,  1,  0); bW  = _shift(b, -1,  0)
    bN  = _shift(b,  0,  1); bS  = _shift(b,  0, -1)
    bNE = _shift(b,  1,  1); bNW = _shift(b, -1,  1)
    bSE = _shift(b,  1, -1); bSW = _shift(b, -1, -1)

    out = (
        (aS  + aSE - aN  - aNE) * (bE  - b)
      + (aSW + aS  - aNW - aN ) * (b   - bW)
      + (aE  + aNE - aW  - aNW) * (bN  - b)
      + (aSE + aE  - aSW - aW ) * (b   - bS)
      + (aE  - aN ) * (bNE - b)
      + (aS  - aW ) * (b   - bSW)
      + (aN  - aW ) * (bNW - b)
      + (aE  - aS ) * (b   - bSE)
    )
    return pars.rakw * out


# ---------------------------------------------------------------------
# Spectral Poisson solve:  Δφ = vort  (periodic)
# ---------------------------------------------------------------------

def getphi(vort, pars):
    """Recover the potential from the vorticity: Δφ = vort.
    In Fourier space φ̂ = −v̂ort / k²  (pars.rksq = 1/k², zero at k=0)."""
    vk = rfft2(vort)
    phik = -vk * pars.rksq
    return irfft2(phik, s=(pars.ny0, pars.nx0))
