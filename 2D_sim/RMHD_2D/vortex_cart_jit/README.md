# `src_jit` — Beklemishev vortex-confinement model (JAX/GPU)

Lightweight GPU solver for the Beklemishev vortex-confinement equations
(A.D. Beklemishev et al., *Fusion Sci. Technol.* **57** (2010) 351), **Eqs. 7–8**:

```
(8) pressure :  ∂ₜP    = −{φ,P}    + ν₄ᵖ ΔP     − ν₅ᵖ(r) P + Q_p
(7) vorticity:  ∂ₜvort = −{φ,vort} + H(φ − φ_w) + κ{P,r²} + ν₄ Δvort − ν₅(r) vort
```

with `vort = Δφ`, `φ` recovered from `vort` by a periodic FFT Poisson solve, and
`{a,b}` the Arakawa Jacobian.  2-D uniform Cartesian mesh, periodic BCs.

It **reuses the validated numerical kernels** (`backend.py`, `kernels.py`: Arakawa
bracket `akw`, FFT Poisson `getphi`, 4th-order FD `derx/dery/delsq`, spectral
hyper-diffusion, jit+scan trap-leapfrog) copied from the Hasegawa–Wakatani
`phw_gpu_jit` code — only the RHS physics, the input format, and the r-dependent
setup are new.

## Naming
`phi` = potential, `vort` = Δφ = vorticity (the evolved field), `pres` = pressure P.
(The paper writes `w` for the potential — the OPPOSITE of the HW code's `w` = vorticity
— so we never use bare `w`.)

## Input — GDB5B-style TOML (`input.toml`)
Sections: `[precision]` (`mode = "double"` | `"single"`), `[normalization]` (physical
B/n/T reference → bases printed at startup), `[domain]`, `[physics]` (U, H, κ, ν₄, ν₅,
ν₄ᵖ, ν₅ᵖ, limiter), `[bias]` (11 concentric ring voltages — **the placeholder you edit
for your machine**; outermost ring = limiter, constant to the boundary), `[initial]`
(pressure profile + seed), `[time]`, `[output]`, `[restart]`.

### Units (physical → normalized on read)
The solver runs in the **GDB5B normalization** (length `L0`, potential `φ̄ = T0/e` [V],
pressure `P̄ = n0·T0` [Pa], time `t̄ = B0·L0²/φ̄`).  Physical inputs are converted at
load time via explicit switches — see **`../Vortex/vortex_normalization.pdf`** for the
full derivation from un-normalized reduced drift-MHD and the coefficient-rescaling table:

| input | switch | default | conversion |
|---|---|---|---|
| `[bias] ring_volts` | `[bias] units` | `"volts"` (physical) | `φ_w = V / T0` |
| `[domain] Lx,Ly`, `ring_radii`, `limiter_radius`, `pres_width` | `[domain] length_units` | `"L0"` (normalized) | `/L0` if `"m"` |
| `[initial] pres_amp` | `[initial] pres_units` | `"normalized"` | `/P̄` if `"Pa"` |
| `U, H, κ, ν₄, ν₅, ...` | — | dimensionless | rescale by β_φ, β_P (see PDF §4) |
| `[time] dt` | — | units of `t̄` | physical Δt = `dt·t̄` |

Beklemishev's Fig. 4–6 values (`H=10, κ=1, ν₅=0.02, ...`) apply **unchanged** when the
applied bias `φ0 = T0/e` and on-axis `p0 = n0·T0` (β_φ = β_P = 1); otherwise rescale per
the PDF.  A ring biased to `1.2·T0` volts reproduces the paper's "bowl of unit depth."

Grid-scale regularization is the **physical** diffusion `ν₄ Δvort` / `ν₄ᵖ ΔP` in the RHS —
there is no separate numerical hyper-diffusion (and no unused kernels: only `derx, dery,
delsq, akw, getphi` are kept).  Defaults reproduce the **Fig. 5** parameters
(`κ=1, H=10, ν₅=0.02, ν₄=0.002, U=0`).

## Precision
`[precision] mode` (or env var `VORTEX_PREC`, which wins): `"double"` = float64
(default, jax_enable_x64) — use for production/invariant accuracy; `"single"` = float32 —
~2× faster, for prototyping.

## Run
```bash
cd vortex                       # so the package `vortex_jit` is importable
JAX_PLATFORMS=cpu  python -m vortex_jit vortex_jit/input.toml     # local (CPU) test
JAX_PLATFORMS=cuda python -m vortex_jit vortex_jit/input.toml     # Perlmutter A100
VORTEX_PREC=single python -m vortex_jit vortex_jit/input.toml     # force single precision
```
Output: `<outdir>/{setup,fields,restart}.h5` (`outdir/<SLURM_JOBID>` under SLURM).
Restart: set `[restart] nrst=1`, point `restartdir` at a previous run.
View results: `python plot_vortex.py <rundir>` (pressure + potential side by side, frame by frame).

## Status / TODO
- **U-term (FLR) implemented** — `+U∇·{∇φ,P}` (as `∂ₓ{φₓ,P}+∂_y{φ_y,P}`) and the source
  `−UΔS_p` (`S_p≈Q_p`).  `U=0` → Fig. 5; `U=−5` → Fig. 6.
- Initial `P(r)` and the 11 ring voltages are configurable defaults; **match them to
  your experiment / the paper** to reproduce Figs. 4–6.
- Periodic BC (FFT Poisson); valid because the box is large and the limiter edge is
  lossy — the potential is set by the interior line-tying `H(φ−φ_w)`, not a hard wall.
