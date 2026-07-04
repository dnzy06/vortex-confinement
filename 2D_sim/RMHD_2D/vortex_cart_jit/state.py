"""State pytree for the Beklemishev vortex model.

Trap-leapfrog carries two time levels of each evolved field plus the derived
potential.  Names are unambiguous: `vort` = Δφ (vorticity, the evolved field),
`phi` = electrostatic potential (from the Poisson inversion of `vorti`), `pres` = P.
"""
from __future__ import annotations

from typing import NamedTuple, Any


class State(NamedTuple):
    vort:  Any   # vorticity Δφ, time level n      (old register)
    vorti: Any   # vorticity Δφ, time level n+1    (current register)
    pres:  Any   # pressure P,  time level n
    presi: Any   # pressure P,  time level n+1
    phi:   Any   # potential consistent with `vorti` via Δφ = vorti
