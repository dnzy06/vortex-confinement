"""Backend selection -- Stage 2 prefers JAX but degrades gracefully.

The Stage-2 design is fundamentally JAX-shaped:
  - immutable pytree State
  - `jax.jit` on the inner step
  - `jax.lax.scan` over the `nts` loop

If JAX is unavailable we fall back to CuPy (no jit, no scan -- still on
GPU, just unfused) or NumPy.  In the non-JAX paths we replace
`lax.scan` with a Python for-loop.  Correctness is preserved, only
speed degrades.
"""

import os

import numpy as _np

# Precision: "double" (float64, default) or "single" (float32).  Set via env var
# VORTEX_PREC (or the [precision] mode in the TOML, which __main__ forwards to it).
_PREC = os.environ.get("VORTEX_PREC", "double").lower()
_SINGLE = _PREC in ("single", "float32", "f32", "32")
DTYPE = "float32" if _SINGLE else "float64"
CDTYPE = "complex64" if _SINGLE else "complex128"

_requested = os.environ.get("PHW_BACKEND", "jax").lower()
BACKEND = None
HAS_JIT = False
HAS_SCAN = False


def _try_jax():
    global xp, rfft2, irfft2, jit, scan_lower, to_device, to_host, asarray
    global BACKEND, HAS_JIT, HAS_SCAN, jax
    import jax as _jax
    _jax.config.update("jax_enable_x64", not _SINGLE)   # x64 only for double precision
    import jax.numpy as _xp
    from jax import jit as _jit
    from jax.lax import scan as _scan
    from jax.numpy.fft import rfft2 as _rfft2, irfft2 as _irfft2

    jax = _jax
    xp = _xp
    rfft2 = _rfft2
    irfft2 = _irfft2

    def jit(f=None, **kw):
        if f is None:
            return lambda g: _jit(g, **kw)
        return _jit(f, **kw)

    def scan_lower(body, init, length):
        """Run `length` steps of `body(state, _) -> (state, _)` and
        return the final state.  Uses jax.lax.scan when available.
        """
        state, _ = _scan(body, init, None, length=length)
        return state

    def to_device(a):
        return _jax.device_put(_xp.asarray(a, dtype=DTYPE))

    def to_host(a):
        return _np.asarray(a)

    def asarray(a, dtype=None):
        return _xp.asarray(a, dtype=dtype if dtype is not None else DTYPE)

    BACKEND = "jax"
    HAS_JIT = True
    HAS_SCAN = True


def _try_cupy():
    global xp, rfft2, irfft2, jit, scan_lower, to_device, to_host, asarray
    global BACKEND, HAS_JIT, HAS_SCAN
    import cupy as _xp
    from cupy.fft import rfft2 as _rfft2, irfft2 as _irfft2

    xp = _xp
    rfft2 = _rfft2
    irfft2 = _irfft2

    def jit(f=None, **kw):
        # No-op for CuPy
        if f is None:
            return lambda g: g
        return f

    def scan_lower(body, init, length):
        # Python for-loop fallback.  Each iteration synchronizes on the
        # GPU only at the end; the body returns a new State.
        state = init
        for _ in range(length):
            state, _ = body(state, None)
        return state

    def to_device(a):
        return _xp.asarray(a, dtype=DTYPE)

    def to_host(a):
        return _xp.asnumpy(a)

    def asarray(a, dtype=None):
        return _xp.asarray(a, dtype=dtype if dtype is not None else DTYPE)

    BACKEND = "cupy"
    HAS_JIT = False
    HAS_SCAN = False


def _try_numpy():
    global xp, rfft2, irfft2, jit, scan_lower, to_device, to_host, asarray
    global BACKEND, HAS_JIT, HAS_SCAN
    import numpy as _xp
    from numpy.fft import rfft2 as _rfft2, irfft2 as _irfft2

    xp = _xp
    rfft2 = _rfft2
    irfft2 = _irfft2

    def jit(f=None, **kw):
        if f is None:
            return lambda g: g
        return f

    def scan_lower(body, init, length):
        state = init
        for _ in range(length):
            state, _ = body(state, None)
        return state

    def to_device(a):
        return _np.asarray(a, dtype=DTYPE)

    def to_host(a):
        return _np.asarray(a)

    def asarray(a, dtype=None):
        return _np.asarray(a, dtype=dtype if dtype is not None else DTYPE)

    BACKEND = "numpy"
    HAS_JIT = False
    HAS_SCAN = False


_chain = {
    "jax":   [_try_jax, _try_cupy, _try_numpy],
    "cupy":  [_try_cupy, _try_numpy],
    "numpy": [_try_numpy],
}.get(_requested, [_try_jax, _try_cupy, _try_numpy])

_err = None
for _fn in _chain:
    try:
        _fn()
        break
    except Exception as exc:
        _err = exc
        continue

if BACKEND is None:
    raise ImportError(f"no backend initialized; last error: {_err}")

if os.environ.get("PHW_QUIET", "0") != "1":
    extras = []
    if HAS_JIT:
        extras.append("jit")
    if HAS_SCAN:
        extras.append("scan")
    suffix = (" [" + " + ".join(extras) + "]") if extras else " [eager]"
    print(f"[2D_Vortex_JIT] backend = {BACKEND}{suffix}")
