"""output_hdf5.py — lightweight HDF5 output for the Beklemishev vortex model.

  setup.h5    : grid + static profiles (x, y, phi_w, r2, nu5p_field, pres0) +
                normalization bases as attributes.
  fields.h5   : resizable datasets phi, vort, pres (nframe, ny, nx) + time t.
  restart.h5  : vort, vorti, pres, presi + step/time attrs (for nrst=1 continuation).
"""
from __future__ import annotations

import os
import numpy as np
import h5py

from .backend import to_host, DTYPE


class Output:
    def __init__(self, cfg, rundir):
        os.makedirs(rundir, exist_ok=True)
        self.rundir = rundir
        self.path_fields = os.path.join(rundir, "fields.h5")
        self.path_restart = os.path.join(rundir, cfg.restart_file)
        self.nframe = 0

        with h5py.File(os.path.join(rundir, "setup.h5"), "w") as f:
            f["nx"] = cfg.nx0; f["ny"] = cfg.ny0
            f["Lx"] = cfg.Lx;  f["Ly"] = cfg.Ly
            f["x"] = np.asarray(cfg.x); f["y"] = np.asarray(cfg.y)
            f["phi_w"] = to_host(cfg.phi_w)
            f["r2"] = to_host(cfg.r2)
            f["nu5p_field"] = to_host(cfg.nu5p_field)
            f["pres0"] = to_host(cfg.pres0)
            for k in ("B0", "n0", "T0", "L0", "rho_s", "cs", "Omega_i",
                      "t_bar", "phi_bar", "P_bar", "v_bar",
                      "U", "H", "kappa", "nu4", "nu5", "nu4p", "nu5p",
                      "limiter_radius", "limiter_factor", "dt"):
                f.attrs[k] = float(getattr(cfg, k))

        with h5py.File(self.path_fields, "w") as f:
            for name in ("phi", "vort", "pres"):
                f.create_dataset(name, shape=(0, cfg.ny0, cfg.nx0),
                                 maxshape=(None, cfg.ny0, cfg.nx0),
                                 chunks=(1, cfg.ny0, cfg.nx0), dtype=DTYPE)
            f.create_dataset("t", shape=(0,), maxshape=(None,), dtype="float64")

    def write_frame(self, state, t):
        with h5py.File(self.path_fields, "a") as f:
            for name, arr in (("phi", state.phi), ("vort", state.vorti),
                              ("pres", state.presi)):
                d = f[name]; d.resize(self.nframe + 1, axis=0)
                d[self.nframe] = to_host(arr)
            d = f["t"]; d.resize(self.nframe + 1, axis=0); d[self.nframe] = t
        self.nframe += 1

    def write_restart(self, state, step, t):
        tmp = self.path_restart + ".tmp"
        with h5py.File(tmp, "w") as f:
            for name in ("vort", "vorti", "pres", "presi"):
                f[name] = to_host(getattr(state, name))
            f.attrs["step"] = int(step); f.attrs["t"] = float(t)
        os.replace(tmp, self.path_restart)      # atomic


def load_restart(path):
    with h5py.File(path, "r") as f:
        arrs = {n: np.asarray(f[n]) for n in ("vort", "vorti", "pres", "presi")}
        step = int(f.attrs["step"]); t = float(f.attrs["t"])
    return arrs, step, t
