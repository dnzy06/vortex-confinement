"""vortex_jit — lightweight GPU (JAX) solver for the Beklemishev vortex-confinement
model (FST 57 (2010) 351, Eqs. 7-8), reusing the validated numerical kernels
(Arakawa bracket, FFT Poisson, 4th-order periodic FD, trap-leapfrog) from the
Hasegawa-Wakatani `phw_gpu_jit` code, with a GDB5B-style TOML input and a physical
normalization section.  Run:  `python -m vortex_jit input.toml`.
"""
