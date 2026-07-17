"""
2-D LIA + O'Brien — with optional Numba JIT  (v1.0)
=========================================================

Numba is used if it is installed and compatible with NumPy.
If not (for example, NumPy 1.20 in QGIS 3.32) — pure NumPy
is used automatically. The physics and the results are identical.

Scheme:   Local Inertial Approximation (Bates et al., 2010)
Rheology: O'Brien & Julien (1985, 1988)

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import logging
import time

from .rheology import OBrienParameters

log = logging.getLogger(__name__)
GRAVITY: float = 9.81

# ─────────────────────────────────────────────────────────────────────────────
# Optional Numba import
# ─────────────────────────────────────────────────────────────────────────────

_NUMBA_AVAILABLE = False

try:
    import numba as _numba
    import numpy as _np_ver

    np_ver = tuple(int(x) for x in _np_ver.__version__.split(".")[:2])
    nb_ver = tuple(int(x) for x in _numba.__version__.split(".")[:2])
    np_min = (1, 22) if nb_ver >= (0, 56) else (1, 18)

    if np_ver >= np_min:
        from numba import njit
        _NUMBA_AVAILABLE = True
        log.info(f"Numba {_numba.__version__} active.")
    else:
        log.warning(
            f"Numba {_numba.__version__}: NumPy {_np_ver.__version__} < "
            f"{'.'.join(map(str, np_min))} — using NumPy."
        )
except ImportError as e:
    log.warning(f"Numba not found (ImportError): {e}")
except Exception as e:
    # OSError, RuntimeError, etc. — Numba is installed but not functional
    log.warning(f"Numba is installed but does not initialise ({type(e).__name__}): {e}")

_JIT_READY = False


# ─────────────────────────────────────────────────────────────────────────────
# TerrainGrid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TerrainGrid:
    """DEM with a precomputed nodata mask."""
    dem:    np.ndarray
    dx:     float
    dy:     float
    nodata: float = -9999.0

    mask_valid: np.ndarray = field(init=False)
    ny: int = field(init=False)
    nx: int = field(init=False)

    def __post_init__(self) -> None:
        self.ny, self.nx = self.dem.shape
        self.mask_valid  = self.dem != self.nodata
        # nodata → inf: flow does not go there (h_face ≤ 0)
        self._z = np.where(self.mask_valid, self.dem.astype(np.float64), np.inf)

    @property
    def z_bed(self) -> np.ndarray:
        return self._z


# ─────────────────────────────────────────────────────────────────────────────
# InflowSource + SourceCondition + SimulationConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InflowSource:
    """A single inflow zone with hydrograph Q(t) [m³/s]."""
    rows:    np.ndarray
    cols:    np.ndarray
    times:   np.ndarray
    q_total: np.ndarray
    name:    str = "source"

    def q_at(self, t: float) -> float:
        return float(np.interp(t, self.times, self.q_total, left=0., right=0.))

    @property
    def n_cells(self) -> int:
        return len(self.rows)


@dataclass
class SourceCondition:
    """Initial and boundary conditions."""
    h0_array:       Optional[np.ndarray] = None
    inflow_sources: List[InflowSource]   = field(default_factory=list)
    # Legacy API (backward compatibility)
    inflow_rows:  Optional[np.ndarray] = field(default=None, repr=False)
    inflow_cols:  Optional[np.ndarray] = field(default=None, repr=False)
    inflow_times: Optional[np.ndarray] = field(default=None, repr=False)
    inflow_q:     Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (self.inflow_rows is not None
                and self.inflow_times is not None
                and not self.inflow_sources):
            q_arr = (self.inflow_q[:, 0] if self.inflow_q is not None
                     else np.zeros_like(self.inflow_times))
            self.inflow_sources.append(InflowSource(
                rows=self.inflow_rows, cols=self.inflow_cols,
                times=self.inflow_times, q_total=q_arr,
                name="source_legacy"))

    def has_hydrograph(self) -> bool:
        return len(self.inflow_sources) > 0

    @classmethod
    def from_sources(cls, sources, h0_array=None):
        return cls(h0_array=h0_array, inflow_sources=list(sources))


@dataclass
class SimulationConfig:
    """
    Solver parameters.

    dt_max          — maximum time step [s].
                      The actual step is ALWAYS ≤ dt_max and is
                      additionally limited by the CFL condition (stability).
                      Not to be confused with output_interval.

    output_interval — interval [s] between progress-bar updates.
                      Does not affect the physics: during this interval
                      output_interval / dt_eff simulation steps are performed.
    """
    t_end:             float    = 3600.0
    dt_max:            float    = 30.0
    dt_min:            float    = 0.001
    cfl_number:        float    = 0.5
    output_interval:   float    = 60.0
    slope_correction:  bool     = False   # Phase 1 (C): cos(theta) steep-slope correction
    entrainment:       bool     = False   # Phase 2 (A): bed entrainment + evolving Cv
    eight_connectivity: bool    = False   # Phase 3 (B): diagonal fluxes (8-connectivity)
    two_phase:         bool     = False   # Phase 4 (D): pore-pressure ratio λ (Iverson)
    progress_callback: Optional[Callable[[float, float], None]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Computational kernels
# ─────────────────────────────────────────────────────────────────────────────

if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, nogil=True)
    def _lia_step(h, Qx, Qy, Qd1, Qd2, z, mask, dx, dy, dd, dt, h_min, n2, K_eta_8gm, tau_gm, cos_x, cos_y, cos_d1, cos_d2, diag, wc, wd):
        G = 9.81
        ny, nx = h.shape[0], h.shape[1]
        nxm, nym = nx - 1, ny - 1

        # X-faces
        Qx_new = np.empty_like(Qx)
        for i in range(ny):
            for j in range(nxm):
                if not (mask[i, j] and mask[i, j+1]):
                    Qx_new[i, j] = 0.0; continue
                eL = z[i, j]   + h[i, j]
                eR = z[i, j+1] + h[i, j+1]
                hf = (eL if eL > eR else eR) - (z[i,j] if z[i,j] > z[i,j+1] else z[i,j+1])
                if hf != hf or hf <= h_min:
                    Qx_new[i, j] = 0.0; continue
                hf = min(hf, 0.5*(h[i, j] + h[i, j+1]))   # limiter Almeida et al. (2012), eq. 2
                if hf <= h_min:
                    Qx_new[i, j] = 0.0; continue
                q  = Qx[i, j]
                vm = (q if q >= 0. else -q) / hf
                # Per-cell rheology averaged to the face (Phase 2 — uniform field
                # reproduces the scalar baseline exactly).
                Kf  = 0.5 * (K_eta_8gm[i, j] + K_eta_8gm[i, j+1])
                tyf = 0.5 * (tau_gm[i, j]    + tau_gm[i, j+1])
                # Manning (quadratic) and viscosity (linear) — semi-implicit in the denominator (Bates 2010)
                # τy/(γm·h) — explicit impulse after step 1 (O'Brien & Julien 1988)
                sf = n2 * vm / hf**(4./3.) + Kf / hf**2
                d  = 1. + G * dt * sf
                if d < 1.: d = 1.
                q_star = (q - G * hf * dt * (eR - eL) / dx * cos_x[i, j]) / d
                yi = G * dt * tyf             # τy-impulse = g·Δt·τy/γm [m²/s]
                aq = q_star if q_star >= 0. else -q_star
                Qx_new[i, j] = 0. if aq <= yi else q_star - (yi if q_star >= 0. else -yi)

        # Y-faces
        Qy_new = np.empty_like(Qy)
        for i in range(nym):
            for j in range(nx):
                if not (mask[i, j] and mask[i+1, j]):
                    Qy_new[i, j] = 0.0; continue
                eU = z[i, j]   + h[i, j]
                eD = z[i+1, j] + h[i+1, j]
                hf = (eU if eU > eD else eD) - (z[i,j] if z[i,j] > z[i+1,j] else z[i+1,j])
                if hf != hf or hf <= h_min:
                    Qy_new[i, j] = 0.0; continue
                hf = min(hf, 0.5*(h[i, j] + h[i+1, j]))
                if hf <= h_min:
                    Qy_new[i, j] = 0.0; continue
                q  = Qy[i, j]
                vm = (q if q >= 0. else -q) / hf
                Kf  = 0.5 * (K_eta_8gm[i, j] + K_eta_8gm[i+1, j])
                tyf = 0.5 * (tau_gm[i, j]    + tau_gm[i+1, j])
                sf = n2 * vm / hf**(4./3.) + Kf / hf**2
                d  = 1. + G * dt * sf
                if d < 1.: d = 1.
                q_star = (q - G * hf * dt * (eD - eU) / dy * cos_y[i, j]) / d
                yi = G * dt * tyf
                aq = q_star if q_star >= 0. else -q_star
                Qy_new[i, j] = 0. if aq <= yi else q_star - (yi if q_star >= 0. else -yi)

        # Diagonal faces (Phase 3 — 8-connectivity). Stay zero unless diag is on.
        Qd1_new = np.zeros_like(Qd1)
        Qd2_new = np.zeros_like(Qd2)
        if diag:
            # D1: between (i,j) and (i+1,j+1)
            for i in range(nym):
                for j in range(nxm):
                    if not (mask[i, j] and mask[i+1, j+1]):
                        continue
                    eA = z[i, j]     + h[i, j]
                    eB = z[i+1, j+1] + h[i+1, j+1]
                    hf = (eA if eA > eB else eB) - (z[i,j] if z[i,j] > z[i+1,j+1] else z[i+1,j+1])
                    if hf != hf or hf <= h_min:
                        continue
                    hf = min(hf, 0.5*(h[i, j] + h[i+1, j+1]))
                    if hf <= h_min:
                        continue
                    q  = Qd1[i, j]
                    vm = (q if q >= 0. else -q) / hf
                    Kf  = 0.5 * (K_eta_8gm[i, j] + K_eta_8gm[i+1, j+1])
                    tyf = 0.5 * (tau_gm[i, j]    + tau_gm[i+1, j+1])
                    sf = n2 * vm / hf**(4./3.) + Kf / hf**2
                    d  = 1. + G * dt * sf
                    if d < 1.: d = 1.
                    q_star = (q - G * hf * dt * (eB - eA) / dd * cos_d1[i, j]) / d
                    yi = G * dt * tyf
                    aq = q_star if q_star >= 0. else -q_star
                    Qd1_new[i, j] = 0. if aq <= yi else q_star - (yi if q_star >= 0. else -yi)
            # D2: between (i,j+1) and (i+1,j)
            for i in range(nym):
                for j in range(nxm):
                    if not (mask[i, j+1] and mask[i+1, j]):
                        continue
                    eA = z[i, j+1] + h[i, j+1]
                    eB = z[i+1, j] + h[i+1, j]
                    hf = (eA if eA > eB else eB) - (z[i,j+1] if z[i,j+1] > z[i+1,j] else z[i+1,j])
                    if hf != hf or hf <= h_min:
                        continue
                    hf = min(hf, 0.5*(h[i, j+1] + h[i+1, j]))
                    if hf <= h_min:
                        continue
                    q  = Qd2[i, j]
                    vm = (q if q >= 0. else -q) / hf
                    Kf  = 0.5 * (K_eta_8gm[i, j+1] + K_eta_8gm[i+1, j])
                    tyf = 0.5 * (tau_gm[i, j+1]    + tau_gm[i+1, j])
                    sf = n2 * vm / hf**(4./3.) + Kf / hf**2
                    d  = 1. + G * dt * sf
                    if d < 1.: d = 1.
                    q_star = (q - G * hf * dt * (eB - eA) / dd * cos_d2[i, j]) / d
                    yi = G * dt * tyf
                    aq = q_star if q_star >= 0. else -q_star
                    Qd2_new[i, j] = 0. if aq <= yi else q_star - (yi if q_star >= 0. else -yi)

        # Outflow limiter — two-pass scheme (identical to the NumPy backend)
        # Pass 1: coefficient sc for each cell
        sc = np.ones((ny, nx), dtype=np.float64)
        for i in range(ny):
            for j in range(nx):
                if not mask[i, j]: continue
                ox = 0.
                if j < nxm:
                    q = Qx_new[i, j];   ox += q if q > 0. else 0.
                if j > 0:
                    q = Qx_new[i, j-1]; ox += -q if q < 0. else 0.
                oy = 0.
                if i < nym:
                    q = Qy_new[i, j];   oy += q if q > 0. else 0.
                if i > 0:
                    q = Qy_new[i-1, j]; oy += -q if q < 0. else 0.
                tot = wc * (ox * dt / dx + oy * dt / dy)
                if diag:
                    od = 0.
                    if i < nym and j < nxm:
                        q = Qd1_new[i, j];     od += q if q > 0. else 0.
                    if i > 0 and j > 0:
                        q = Qd1_new[i-1, j-1]; od += -q if q < 0. else 0.
                    if i < nym and j > 0:
                        q = Qd2_new[i, j-1];   od += q if q > 0. else 0.
                    if i > 0 and j < nxm:
                        q = Qd2_new[i-1, j];   od += -q if q < 0. else 0.
                    tot += wd * od * dt / dd
                if tot > h[i, j] and tot > 1e-12:
                    sc[i, j] = 0.99 * h[i, j] / tot
        # Pass 2: apply min(sc_L, sc_R) to each face — once
        for i in range(ny):
            for j in range(nxm):
                f = sc[i, j] if sc[i, j] < sc[i, j+1] else sc[i, j+1]
                if f < 1.: Qx_new[i, j] *= f
        for i in range(nym):
            for j in range(nx):
                f = sc[i, j] if sc[i, j] < sc[i+1, j] else sc[i+1, j]
                if f < 1.: Qy_new[i, j] *= f
        if diag:
            for i in range(nym):
                for j in range(nxm):
                    f = sc[i, j] if sc[i, j] < sc[i+1, j+1] else sc[i+1, j+1]
                    if f < 1.: Qd1_new[i, j] *= f
                    g = sc[i, j+1] if sc[i, j+1] < sc[i+1, j] else sc[i+1, j]
                    if g < 1.: Qd2_new[i, j] *= g

        # Continuity
        h_new = np.empty_like(h)
        for i in range(ny):
            for j in range(nx):
                if not mask[i, j]: h_new[i, j] = 0.0; continue
                div = 0.
                if j < nxm: div -= wc * Qx_new[i, j]   / dx
                if j > 0:   div += wc * Qx_new[i, j-1] / dx
                if i < nym: div -= wc * Qy_new[i, j]   / dy
                if i > 0:   div += wc * Qy_new[i-1, j] / dy
                if diag:
                    if i < nym and j < nxm: div -= wd * Qd1_new[i, j]     / dd
                    if i > 0   and j > 0:   div += wd * Qd1_new[i-1, j-1] / dd
                    if i < nym and j > 0:   div -= wd * Qd2_new[i, j-1]   / dd
                    if i > 0   and j < nxm: div += wd * Qd2_new[i-1, j]   / dd
                v = h[i, j] + dt * div
                h_new[i, j] = v if v > 0. else 0.
        return h_new, Qx_new, Qy_new, Qd1_new, Qd2_new

    @njit(cache=True, fastmath=True, nogil=True)
    def _cfl_vol_wet(h, mask, cfl, dx_min, dx, dy, h_min):
        G = 9.81; max_c = 0.; vol = 0.; wet = 0
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if not mask[i, j]: continue
                hv = h[i, j]
                if hv > 0.:
                    c = (G * hv)**0.5
                    if c > max_c: max_c = c
                    vol += hv
                if hv >= h_min: wet += 1
        return (cfl * dx_min / max_c) if max_c > 1e-10 else 1e9, vol * dx * dy, wet

    @njit(cache=True, fastmath=True, nogil=True)
    def _add_inflow(h, rows, cols, rate, dt):
        for k in range(rows.shape[0]):
            h[rows[k], cols[k]] += rate * dt

    @njit(cache=True, fastmath=True, nogil=True)
    def _update_max(h, Qx, Qy, h_max, V_max, t_arrival, h_min, t):
        ny, nx = h.shape[0], h.shape[1]
        nxm, nym = nx - 1, ny - 1
        for i in range(ny):
            for j in range(nx):
                hv = h[i, j]
                if hv > h_max[i, j]: h_max[i, j] = hv
                if hv < h_min: continue
                hd = hv if hv > h_min else h_min
                qx = 0.
                if 0 < j < nx-1: qx = 0.5*(Qx[i,j-1]+Qx[i,j])
                elif j == 0   and nxm > 0: qx = Qx[i, 0]
                elif j == nx-1 and nxm > 0: qx = Qx[i, nxm-1]
                qy = 0.
                if 0 < i < ny-1: qy = 0.5*(Qy[i-1,j]+Qy[i,j])
                elif i == 0   and nym > 0: qy = Qy[0, j]
                elif i == ny-1 and nym > 0: qy = Qy[nym-1, j]
                vm = min((qx*qx+qy*qy)**0.5/hd, 30.)
                if vm > V_max[i, j]: V_max[i, j] = vm
                if t_arrival[i, j] > 1e30: t_arrival[i, j] = t

else:
    # ── NumPy fallback ────────────────────────────────────────────────────────

    def _lia_step(h, Qx, Qy, Qd1, Qd2, z, mask, dx, dy, dd, dt, h_min, n2, K_eta_8gm, tau_gm, cos_x, cos_y, cos_d1, cos_d2, diag, wc, wd):
        G  = 9.81
        mv = mask
        eta = z + h

        # X-faces
        eL, eR = eta[:, :-1], eta[:, 1:]
        hf_x = np.maximum(np.nan_to_num(
            np.maximum(eL, eR) - np.maximum(z[:,:-1], z[:,1:]),
            nan=0., posinf=0.), 0.)
        hf_x = np.minimum(hf_x, 0.5*(h[:,:-1]+h[:,1:]))           # limiter Almeida et al. (2012)
        wet_x = (hf_x > h_min) & mv[:, :-1] & mv[:, 1:]
        vm_x  = np.abs(Qx) / np.maximum(hf_x, 1e-12)
        # Per-cell rheology averaged to the X-faces (Phase 2).
        K_x   = 0.5 * (K_eta_8gm[:, :-1] + K_eta_8gm[:, 1:])
        ty_x  = 0.5 * (tau_gm[:, :-1]    + tau_gm[:, 1:])
        sf_x  = (n2 * vm_x / np.maximum(hf_x,1e-12)**(4./3.)
                 + K_x / np.maximum(hf_x,1e-12)**2)
        denom_x = np.maximum(1. + G*dt*sf_x, 1.)
        q_star_x = np.where(wet_x,
                            (Qx - G*hf_x*dt*np.nan_to_num((eR-eL)/dx)*cos_x) / denom_x, 0.)
        # τy — explicit impulse with sign-inversion limiting (O'Brien & Julien 1988)
        yield_x = G * dt * ty_x
        Qx_new = np.where(np.abs(q_star_x) > yield_x,
                          q_star_x - yield_x * np.sign(q_star_x), 0.)

        # Y-faces
        eU, eD = eta[:-1, :], eta[1:, :]
        hf_y = np.maximum(np.nan_to_num(
            np.maximum(eU, eD) - np.maximum(z[:-1,:], z[1:,:]),
            nan=0., posinf=0.), 0.)
        hf_y = np.minimum(hf_y, 0.5*(h[:-1,:]+h[1:,:]))
        wet_y = (hf_y > h_min) & mv[:-1,:] & mv[1:,:]
        vm_y  = np.abs(Qy) / np.maximum(hf_y, 1e-12)
        K_y   = 0.5 * (K_eta_8gm[:-1, :] + K_eta_8gm[1:, :])
        ty_y  = 0.5 * (tau_gm[:-1, :]    + tau_gm[1:, :])
        sf_y  = (n2 * vm_y / np.maximum(hf_y,1e-12)**(4./3.)
                 + K_y / np.maximum(hf_y,1e-12)**2)
        denom_y = np.maximum(1. + G*dt*sf_y, 1.)
        q_star_y = np.where(wet_y,
                            (Qy - G*hf_y*dt*np.nan_to_num((eD-eU)/dy)*cos_y) / denom_y, 0.)
        yield_y = G * dt * ty_y
        Qy_new = np.where(np.abs(q_star_y) > yield_y,
                          q_star_y - yield_y * np.sign(q_star_y), 0.)

        # Diagonal faces (Phase 3 — 8-connectivity). Stay zero unless diag is on.
        Qd1_new = np.zeros_like(Qd1)
        Qd2_new = np.zeros_like(Qd2)
        if diag:
            # D1: between (i,j) and (i+1,j+1)
            eA1, eB1 = eta[:-1, :-1], eta[1:, 1:]
            hf_d1 = np.maximum(np.nan_to_num(
                np.maximum(eA1, eB1) - np.maximum(z[:-1,:-1], z[1:,1:]),
                nan=0., posinf=0.), 0.)
            hf_d1 = np.minimum(hf_d1, 0.5*(h[:-1,:-1]+h[1:,1:]))
            wet_d1 = (hf_d1 > h_min) & mv[:-1,:-1] & mv[1:,1:]
            vm_d1  = np.abs(Qd1) / np.maximum(hf_d1, 1e-12)
            K_d1   = 0.5 * (K_eta_8gm[:-1,:-1] + K_eta_8gm[1:,1:])
            ty_d1  = 0.5 * (tau_gm[:-1,:-1]    + tau_gm[1:,1:])
            sf_d1  = (n2 * vm_d1 / np.maximum(hf_d1,1e-12)**(4./3.)
                      + K_d1 / np.maximum(hf_d1,1e-12)**2)
            den_d1 = np.maximum(1. + G*dt*sf_d1, 1.)
            qs_d1  = np.where(wet_d1,
                              (Qd1 - G*hf_d1*dt*np.nan_to_num((eB1-eA1)/dd)*cos_d1) / den_d1, 0.)
            yld_d1 = G * dt * ty_d1
            Qd1_new = np.where(np.abs(qs_d1) > yld_d1,
                               qs_d1 - yld_d1 * np.sign(qs_d1), 0.)
            # D2: between (i,j+1) and (i+1,j)
            eA2, eB2 = eta[:-1, 1:], eta[1:, :-1]
            hf_d2 = np.maximum(np.nan_to_num(
                np.maximum(eA2, eB2) - np.maximum(z[:-1,1:], z[1:,:-1]),
                nan=0., posinf=0.), 0.)
            hf_d2 = np.minimum(hf_d2, 0.5*(h[:-1,1:]+h[1:,:-1]))
            wet_d2 = (hf_d2 > h_min) & mv[:-1,1:] & mv[1:,:-1]
            vm_d2  = np.abs(Qd2) / np.maximum(hf_d2, 1e-12)
            K_d2   = 0.5 * (K_eta_8gm[:-1,1:] + K_eta_8gm[1:,:-1])
            ty_d2  = 0.5 * (tau_gm[:-1,1:]    + tau_gm[1:,:-1])
            sf_d2  = (n2 * vm_d2 / np.maximum(hf_d2,1e-12)**(4./3.)
                      + K_d2 / np.maximum(hf_d2,1e-12)**2)
            den_d2 = np.maximum(1. + G*dt*sf_d2, 1.)
            qs_d2  = np.where(wet_d2,
                              (Qd2 - G*hf_d2*dt*np.nan_to_num((eB2-eA2)/dd)*cos_d2) / den_d2, 0.)
            yld_d2 = G * dt * ty_d2
            Qd2_new = np.where(np.abs(qs_d2) > yld_d2,
                               qs_d2 - yld_d2 * np.sign(qs_d2), 0.)

        # Outflow limiter
        ox = np.zeros_like(h)
        ox[:,:-1] += np.maximum(Qx_new,0.); ox[:,1:] += np.maximum(-Qx_new,0.)
        oy = np.zeros_like(h)
        oy[:-1,:] += np.maximum(Qy_new,0.); oy[1:,:] += np.maximum(-Qy_new,0.)
        tot = wc*(ox*dt/dx + oy*dt/dy)
        if diag:
            od = np.zeros_like(h)
            od[:-1,:-1] += np.maximum(Qd1_new,0.); od[1:,1:]  += np.maximum(-Qd1_new,0.)
            od[:-1,1:]  += np.maximum(Qd2_new,0.); od[1:,:-1] += np.maximum(-Qd2_new,0.)
            tot = tot + wd*od*dt/dd
        sc = np.where(tot > h, 0.99*h/(tot+1e-12), 1.)
        Qx_new *= np.minimum(sc[:,:-1], sc[:,1:])
        Qy_new *= np.minimum(sc[:-1,:], sc[1:,:])
        if diag:
            Qd1_new *= np.minimum(sc[:-1,:-1], sc[1:,1:])
            Qd2_new *= np.minimum(sc[:-1,1:], sc[1:,:-1])

        # Continuity
        div = np.zeros_like(h)
        div[:,:-1] -= wc*Qx_new/dx; div[:,1:]  += wc*Qx_new/dx
        div[:-1,:] -= wc*Qy_new/dy; div[1:,:]  += wc*Qy_new/dy
        if diag:
            div[:-1,:-1] -= wd*Qd1_new/dd; div[1:,1:]  += wd*Qd1_new/dd
            div[:-1,1:]  -= wd*Qd2_new/dd; div[1:,:-1] += wd*Qd2_new/dd
        h_new = np.maximum(h + dt*div, 0.)
        h_new[~mv] = 0.
        return h_new, Qx_new, Qy_new, Qd1_new, Qd2_new

    def _cfl_vol_wet(h, mask, cfl, dx_min, dx, dy, h_min):
        hv    = h[mask]
        pos   = hv[hv > 0.]
        max_c = (9.81*pos.max())**0.5 if pos.size > 0 else 0.
        dt    = (cfl*dx_min/max_c) if max_c > 1e-10 else 1e9
        return dt, pos.sum()*dx*dy, int((hv >= h_min).sum())

    def _add_inflow(h, rows, cols, rate, dt):
        h[rows, cols] += rate * dt

    def _update_max(h, Qx, Qy, h_max, V_max, t_arrival, h_min, t):
        np.maximum(h_max, h, out=h_max)
        ny, nx = h.shape
        hd = np.maximum(h, h_min)
        Qxc = np.zeros((ny, nx))
        Qxc[:,1:-1]=0.5*(Qx[:,:-1]+Qx[:,1:]); Qxc[:,0]=Qx[:,0]; Qxc[:,-1]=Qx[:,-1]
        Qyc = np.zeros((ny, nx))
        Qyc[1:-1,:]=0.5*(Qy[:-1,:]+Qy[1:,:]); Qyc[0,:]=Qy[0,:]; Qyc[-1,:]=Qy[-1,:]
        vm = np.minimum(np.sqrt(Qxc**2+Qyc**2)/hd, 30.)
        np.maximum(V_max, vm, out=V_max)
        t_arrival[(h >= h_min) & np.isinf(t_arrival)] = t


# ─────────────────────────────────────────────────────────────────────────────
# JIT warmup
# ─────────────────────────────────────────────────────────────────────────────

def warmup_jit(h_min: float = 0.005) -> float:
    if not _NUMBA_AVAILABLE:
        return 0.0
    t0 = time.time()
    h  = np.ones((4,4),dtype=np.float64)*0.5
    Qx = np.zeros((4,3),dtype=np.float64)
    Qy = np.zeros((3,4),dtype=np.float64)
    z  = np.zeros((4,4),dtype=np.float64)
    mv = np.ones((4,4), dtype=np.bool_)
    hm = np.zeros((4,4),dtype=np.float64)
    vm = np.zeros((4,4),dtype=np.float64)
    ta = np.full((4,4), np.inf, dtype=np.float64)
    r  = np.zeros(2, dtype=np.int64)
    c  = np.arange(2, dtype=np.int64)
    cx = np.ones((4,3),dtype=np.float64)
    cy = np.ones((3,4),dtype=np.float64)
    Ke = np.full((4,4),1e-4,dtype=np.float64)
    tg = np.full((4,4),0.001,dtype=np.float64)
    Qd1 = np.zeros((3,3),dtype=np.float64)
    Qd2 = np.zeros((3,3),dtype=np.float64)
    cd1 = np.ones((3,3),dtype=np.float64)
    cd2 = np.ones((3,3),dtype=np.float64)
    dd  = (10.**2 + 10.**2)**0.5
    _lia_step(h,Qx,Qy,Qd1,Qd2,z,mv,10.,10.,dd,1.,h_min,0.0036,Ke,tg,cx,cy,cd1,cd2,True,0.8,0.2)
    _cfl_vol_wet(h,mv,0.4,10.,10.,10.,h_min)
    _update_max(h,Qx,Qy,hm,vm,ta,h_min,0.)
    _add_inflow(h,r,c,0.1,1.0)
    return time.time()-t0


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

class DebrisFlowSolver2D:

    def __init__(self, terrain, params, source, config):
        self.terrain = terrain
        self.params  = params
        self.source  = source
        self.config  = config

        ny, nx = terrain.ny, terrain.nx
        self.h   = np.zeros((ny, nx), dtype=np.float64)
        self.Qx  = np.zeros((ny, nx-1), dtype=np.float64)
        self.Qy  = np.zeros((ny-1, nx), dtype=np.float64)

        self.h_max     = np.zeros((ny, nx), dtype=np.float64)
        self.V_max     = np.zeros((ny, nx), dtype=np.float64)
        self.t_arrival = np.full((ny, nx), np.inf, dtype=np.float64)

        self.t               = 0.0
        self.n_steps         = 0
        self.volume_initial  = 0.0
        self.volume_history: list = []
        self._total_inflow_vol = 0.0

        # Precomputed kernel constants. Manning n is Cv-independent (scalar); the
        # O'Brien yield/viscous terms become per-cell FIELDS (Phase 2) so the
        # entrainment substep can vary them. A uniform Cv reproduces the scalar
        # baseline exactly (so entrainment OFF is bit-for-bit the planar model).
        self._n2 = params.manning_n**2
        self.Cv  = np.full((ny, nx), params.Cv, dtype=np.float64)
        self._tau_gm_field, self._K_eta_8gm_field = params.kernel_fields(self.Cv)

        self._z      = np.ascontiguousarray(terrain.z_bed, dtype=np.float64)
        self._mv     = np.ascontiguousarray(terrain.mask_valid)
        self._dx_min = min(terrain.dx, terrain.dy)

        # Phase 1 (C): per-face cos(theta) for the optional steep-slope
        # correction. The surface-gradient driving term is multiplied by this so
        # the gradient is measured along the bed surface (true distance dx/cos)
        # instead of the horizontal grid spacing. OFF -> all ones -> bit-for-bit
        # identical to the planar FLO-2D baseline.
        self._cos_x, self._cos_y, self._cos_d1, self._cos_d2 = self._slope_cos_faces(
            bool(getattr(config, "slope_correction", False)))

        # Phase 3 (B): diagonal (8-connectivity) momentum state + geometry.
        self._eight_conn = bool(getattr(config, "eight_connectivity", False))
        self._dd  = (terrain.dx ** 2 + terrain.dy ** 2) ** 0.5
        self.Qd1  = np.zeros((ny - 1, nx - 1), dtype=np.float64)
        self.Qd2  = np.zeros((ny - 1, nx - 1), dtype=np.float64)
        # Mobility-preserving 8-connectivity weights (cardinal:diagonal = 4:1,
        # isotropic 9-point). Both directional sums reproduce the gradient, so
        # wc + wd = 1 keeps total transport equal to 4-connectivity while removing
        # higher-order grid anisotropy. OFF -> (1, 0) == baseline, bit-for-bit.
        self._wc, self._wd = (0.8, 0.2) if self._eight_conn else (1.0, 0.0)

        # Phase 2 (A): entrainment precompute. Per-cell terrain slope tan(theta)
        # from the initial bed and the resulting Takahashi equilibrium
        # concentration. The flow erodes toward Cv_eq on steep cells and deposits
        # on flat ones; the rising Cv auto-stiffens τy,η via the exp(Cv) closure.
        self._entrain    = bool(getattr(config, "entrainment", False))
        self._erode_frac = 0.2     # max |bed change| per step as a fraction of depth
        self._slope_tan  = self._slope_tan_cells()
        self._cv_eq      = params.equilibrium_cv(self._slope_tan)
        self._entrained_vol = 0.0

        # Phase 4 (D): two-phase / basal pore-pressure ratio λ (Iverson). λ scales
        # the yield stress by (1−λ); it is advected with the flow and relaxes
        # toward drained over the consolidation time. OFF -> λ=0 -> baseline.
        self._two_phase = bool(getattr(config, "two_phase", False))
        self._lambda0   = float(params.pore_pressure_lambda0)
        self.P          = np.zeros((ny, nx), dtype=np.float64)   # P = λ·h content

    def _slope_cos_faces(self, enabled: bool):
        """Per-face cos(theta) from the bed (cardinal + diagonal); ones when off."""
        ny, nx = self.terrain.ny, self.terrain.nx
        cos_x  = np.ones((ny, nx - 1), dtype=np.float64)
        cos_y  = np.ones((ny - 1, nx), dtype=np.float64)
        cos_d1 = np.ones((ny - 1, nx - 1), dtype=np.float64)
        cos_d2 = np.ones((ny - 1, nx - 1), dtype=np.float64)
        if not enabled:
            return cos_x, cos_y, cos_d1, cos_d2
        z  = self.terrain.dem.astype(np.float64)
        mv = self.terrain.mask_valid
        dd = (self.terrain.dx ** 2 + self.terrain.dy ** 2) ** 0.5
        dzdx  = (z[:, 1:] - z[:, :-1]) / self.terrain.dx
        cos_x = np.where(mv[:, :-1] & mv[:, 1:],
                         1.0 / np.sqrt(1.0 + dzdx * dzdx), 1.0)
        dzdy  = (z[1:, :] - z[:-1, :]) / self.terrain.dy
        cos_y = np.where(mv[:-1, :] & mv[1:, :],
                         1.0 / np.sqrt(1.0 + dzdy * dzdy), 1.0)
        dzd1   = (z[1:, 1:] - z[:-1, :-1]) / dd
        cos_d1 = np.where(mv[:-1, :-1] & mv[1:, 1:],
                          1.0 / np.sqrt(1.0 + dzd1 * dzd1), 1.0)
        dzd2   = (z[1:, :-1] - z[:-1, 1:]) / dd
        cos_d2 = np.where(mv[:-1, 1:] & mv[1:, :-1],
                          1.0 / np.sqrt(1.0 + dzd2 * dzd2), 1.0)
        return (np.ascontiguousarray(cos_x),  np.ascontiguousarray(cos_y),
                np.ascontiguousarray(cos_d1), np.ascontiguousarray(cos_d2))

    def _slope_tan_cells(self):
        """Per-cell terrain slope magnitude tan(theta) from the initial bed."""
        mv = self.terrain.mask_valid
        zf = self.terrain.dem.astype(np.float64).copy()
        zf[~mv] = np.nan
        gy, gx = np.gradient(zf, self.terrain.dy, self.terrain.dx)
        tan = np.sqrt(np.nan_to_num(gx) ** 2 + np.nan_to_num(gy) ** 2)
        tan[~mv] = 0.0
        return np.ascontiguousarray(tan)

    def _entrain_step(self, h, z, Cv, Qx, Qy, dt):
        """
        One bed-entrainment substep (Egashira / Takahashi). Erodes (Cv < Cv_eq)
        or deposits (Cv > Cv_eq) bed material at a velocity-dependent rate, then
        updates the bed z, flow depth h and concentration Cv by mass balance and
        recomputes the per-cell rheology fields. Pure NumPy → identical on both
        backends (only _lia_step is backend-specific).

        Mass balance per cell over the step (bed packing concentration Cv_bed):
            h' = h + dz                      (dz>0 erodes, dz<0 deposits)
            Cv' = (Cv·h + Cv_bed·dz) / h'
            z' = z − dz
        Returns (h, z, Cv, tau_gm_field, K_eta_8gm_field, entrained_volume).
        """
        p = self.params
        h_min = p.h_min
        mv = self._mv
        ny, nx = h.shape
        wet = (h > h_min) & mv

        # cell-centred speed (same convention as _update_max)
        Qxc = np.zeros((ny, nx)); Qyc = np.zeros((ny, nx))
        Qxc[:, 1:-1] = 0.5 * (Qx[:, :-1] + Qx[:, 1:])
        Qxc[:, 0] = Qx[:, 0]; Qxc[:, -1] = Qx[:, -1]
        Qyc[1:-1, :] = 0.5 * (Qy[:-1, :] + Qy[1:, :])
        Qyc[0, :] = Qy[0, :]; Qyc[-1, :] = Qy[-1, :]
        hd = np.maximum(h, h_min)
        V = np.minimum(np.sqrt(Qxc ** 2 + Qyc ** 2) / hd, 30.0)

        # Egashira-style bed-change increment: +erosion (Cv<Cv_eq), −deposition.
        dz = p.entrain_coef * (self._cv_eq - Cv) * V * dt
        Cv_bed = p.Cv_bed
        ero_cap = self._erode_frac * hd                          # erosion limited per step
        dep_cap = -np.minimum(hd, Cv * hd / max(Cv_bed, 1e-6))    # deposition ≤ available solids
        dz = np.where(wet, np.clip(dz, dep_cap, ero_cap), 0.0)

        h_new  = h + dz
        solid  = Cv * h + Cv_bed * dz
        h_safe = np.maximum(h_new, h_min)
        Cv_new = np.where(wet, np.clip(solid / h_safe, 0.0, p.Cv_max), Cv)
        z_new  = z.copy(); z_new[wet] = z[wet] - dz[wet]
        h_new  = np.maximum(h_new, 0.0)

        entrained = float(np.sum(dz[wet])) * self.terrain.dx * self.terrain.dy
        tau_gm, K_eta_8gm = p.kernel_fields(Cv_new)
        return h_new, z_new, Cv_new, tau_gm, K_eta_8gm, entrained

    def _pore_pressure_step(self, P, lam, Qx, Qy, h_new, dt, wc):
        """
        One pore-pressure substep (Phase 4). Advects the pore-pressure content
        P = λ·h with the post-step cardinal fluxes (upwind) and relaxes it toward
        drained over the consolidation time T_c. λ = P/h is the basal
        pore-pressure ratio, which scales the yield stress by (1−λ): high λ → low
        yield → high mobility; λ decays as the flow consolidates → it stops.
        Pure NumPy → identical on both backends. (Diagonal flux transport of λ is
        neglected; 8-connectivity is ≈inert here anyway.)
        """
        p = self.params
        dx, dy = self.terrain.dx, self.terrain.dy
        # upwind λ on each face; depth transferred = wc·flux·dt/spacing
        fx    = wc * Qx * dt / dx
        lam_x = np.where(Qx >= 0.0, lam[:, :-1], lam[:, 1:])
        tx    = fx * lam_x
        dP = np.zeros_like(P)
        dP[:, :-1] -= tx; dP[:, 1:] += tx
        fy    = wc * Qy * dt / dy
        lam_y = np.where(Qy >= 0.0, lam[:-1, :], lam[1:, :])
        ty_   = fy * lam_y
        dP[:-1, :] -= ty_; dP[1:, :] += ty_
        P_new = P + dP
        # consolidation: excess pore pressure relaxes toward drained (λ → 0)
        P_new *= np.exp(-dt / p.pore_consolidation_time)
        P_new[~self._mv] = 0.0
        # λ = P/h must stay in [0, 1]  →  P in [0, h]
        return np.clip(P_new, 0.0, np.maximum(h_new, 0.0))

    def _initialise(self) -> None:
        global _JIT_READY
        if self.source.h0_array is not None:
            self.h[:] = np.maximum(self.source.h0_array, 0.)
        self.h[~self.terrain.mask_valid] = 0.
        np.copyto(self.h_max, self.h)
        self.t_arrival[self.h > self.params.h_min] = 0.
        self.volume_initial = np.sum(self.h)*self.terrain.dx*self.terrain.dy

        backend = "Numba JIT" if _NUMBA_AVAILABLE else "NumPy"
        log.info(f"[{backend}] wet={np.sum(self.h>self.params.h_min)}, "
                 f"V₀={self.volume_initial:.1f} m³, "
                 f"sources={len(self.source.inflow_sources)}")

        if _NUMBA_AVAILABLE and not _JIT_READY:
            log.info("Compiling Numba JIT (~5 s)...")
            log.info(f"JIT ready in {warmup_jit(self.params.h_min):.1f} s")
            _JIT_READY = True

    def _cell_velocities(self):
        ny, nx = self.terrain.ny, self.terrain.nx
        hd = np.maximum(self.h, self.params.h_min)
        Qxc = np.zeros((ny, nx))
        Qxc[:,1:-1]=0.5*(self.Qx[:,:-1]+self.Qx[:,1:])
        Qxc[:,0]=self.Qx[:,0]; Qxc[:,-1]=self.Qx[:,-1]
        Qyc = np.zeros((ny, nx))
        Qyc[1:-1,:]=0.5*(self.Qy[:-1,:]+self.Qy[1:,:])
        Qyc[0,:]=self.Qy[0,:]; Qyc[-1,:]=self.Qy[-1,:]
        Vx, Vy = Qxc/hd, Qyc/hd
        vm = np.sqrt(Vx**2+Vy**2)
        sc = np.where(vm>30., 30./(vm+1e-10), 1.)
        Vx*=sc; Vy*=sc
        dry = (self.h<self.params.h_min)|(~self.terrain.mask_valid)
        Vx[dry]=0.; Vy[dry]=0.
        return Vx, Vy

    def run(self) -> "SimulationResult":
        backend = "Numba JIT" if _NUMBA_AVAILABLE else "NumPy"
        log.info(f"LIA + O'Brien [{backend}]")
        log.info(self.params.summary())
        for s in self.source.inflow_sources:
            log.info(f"  '{s.name}': {s.n_cells} cells, Q_max={s.q_total.max():.1f} m³/s")

        self._initialise()

        h = self.h; Qx = self.Qx; Qy = self.Qy
        z = self._z.copy(); mv = self._mv          # bed is mutable (entrainment)
        dx = self.terrain.dx; dy = self.terrain.dy
        h_min  = self.params.h_min
        n2     = self._n2
        Ke     = self._K_eta_8gm_field.copy(); ty = self._tau_gm_field.copy()
        Cv     = self.Cv.copy()
        cx     = self._cos_x; cy = self._cos_y
        cd1    = self._cos_d1; cd2 = self._cos_d2
        Qd1    = self.Qd1; Qd2 = self.Qd2; dd = self._dd; diag = self._eight_conn
        wc     = self._wc; wd = self._wd
        entrain = self._entrain; self._entrained_vol = 0.0
        two_phase = self._two_phase; lam0 = self._lambda0
        P = (lam0 * h.copy()) if two_phase else self.P
        cfl    = self.config.cfl_number; dmin = self._dx_min
        dt_max = self.config.dt_max; dt_min = self.config.dt_min
        t_end  = self.config.t_end; out_iv = self.config.output_interval
        cell_area = dx * dy

        t_sim = 0.; t_last = 0.; ever_wet = False; t_wall = time.time()

        while t_sim < t_end:

            dt_cfl, V, wet = _cfl_vol_wet(h, mv, cfl, dmin, dx, dy, h_min)
            dt = min(dt_cfl, dt_max, t_end - t_sim)
            if dt < dt_min:
                raise RuntimeError(f"CFL step {dt:.6f} s < dt_min={dt_min} s")

            for src in self.source.inflow_sources:
                q_t = src.q_at(t_sim)
                if q_t > 0.:
                    rate = q_t / (src.n_cells * cell_area)
                    self._total_inflow_vol += q_t * dt
                    if _NUMBA_AVAILABLE:
                        _add_inflow(h, src.rows.astype(np.int64),
                                    src.cols.astype(np.int64), rate, dt)
                    else:
                        h[src.rows, src.cols] += rate * dt
                    if two_phase:
                        P[src.rows, src.cols] += lam0 * rate * dt

            if two_phase:
                lam = np.clip(P / np.maximum(h, h_min), 0.0, 1.0)
                ty_eff = ty * (1.0 - lam)        # pore pressure cuts the yield stress
            else:
                ty_eff = ty
            h, Qx, Qy, Qd1, Qd2 = _lia_step(h, Qx, Qy, Qd1, Qd2, z, mv, dx, dy, dd, dt,
                                            h_min, n2, Ke, ty_eff, cx, cy, cd1, cd2, diag, wc, wd)
            if entrain:
                h, z, Cv, ty, Ke, dV = self._entrain_step(h, z, Cv, Qx, Qy, dt)
                self._entrained_vol += dV
            if two_phase:
                P = self._pore_pressure_step(P, lam, Qx, Qy, h, dt, wc)
            _update_max(h, Qx, Qy, self.h_max, self.V_max, self.t_arrival, h_min, t_sim)

            t_sim += dt; self.n_steps += 1
            self.volume_history.append((t_sim, V))

            if not ever_wet and wet > 0:
                ever_wet = True

            if t_sim - t_last >= out_iv:
                self.h[:]=h; self.Qx[:]=Qx; self.Qy[:]=Qy
                t_last = t_sim
                if self.config.progress_callback:
                    self.config.progress_callback(t_sim, t_end)
                log.debug(f"t={t_sim:.1f}s dt={dt:.3f}s wet={wet} "
                          f"V={V:.0f}m³ h_max={self.h_max.max():.2f}m")

            if ever_wet and wet == 0:
                log.info(f"Flow stopped t={t_sim:.1f} s"); break

        self.h[:]=h; self.Qx[:]=Qx; self.Qy[:]=Qy; self.t=t_sim
        self.Qd1[:]=Qd1; self.Qd2[:]=Qd2
        self.Cv[:] = Cv
        bed_change = np.zeros_like(z)
        bed_change[mv] = z[mv] - self._z[mv]                 # final − initial bed
        cv_final = np.where((self.h_max > h_min) & mv, Cv, 0.0)
        self.P[:] = P
        lam_fld = np.clip(P / np.maximum(h, h_min), 0.0, 1.0)
        lambda_final = np.where((self.h_max > h_min) & mv, lam_fld, 0.0)
        log.info(f"Completed [{backend}]: {self.n_steps} steps, "
                 f"{t_sim:.1f} s, {time.time()-t_wall:.2f} s CPU")
        if entrain:
            log.info(f"  entrained volume: {self._entrained_vol:.1f} m³, "
                     f"Cv range [{Cv[self.h_max>h_min].min():.3f}, "
                     f"{Cv[self.h_max>h_min].max():.3f}]" if (self.h_max>h_min).any()
                     else "  entrained volume: 0 m³")

        return SimulationResult(
            h_max=self.h_max, V_max=self.V_max,
            h_final=self.h, t_arrival=self.t_arrival,
            volume_history=self.volume_history,
            t_simulated=self.t, n_steps=self.n_steps,
            params=self.params, terrain=self.terrain,
            total_inflow_vol=self._total_inflow_vol,
            initial_volume=self.volume_initial,
            cv_final=cv_final, bed_change=bed_change,
            total_entrained_vol=self._entrained_vol,
            lambda_final=lambda_final,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    h_max:            np.ndarray
    V_max:            np.ndarray
    h_final:          np.ndarray
    t_arrival:        np.ndarray
    volume_history:   list
    t_simulated:      float
    n_steps:          int
    params:           OBrienParameters
    terrain:          TerrainGrid
    total_inflow_vol: float = 0.0
    initial_volume:   float = 0.0
    cv_final:            Optional[np.ndarray] = None   # Phase 2: final Cv field
    bed_change:          Optional[np.ndarray] = None   # Phase 2: z_final − z_initial [m]
    total_entrained_vol: float = 0.0                   # Phase 2: net bed→flow volume
    lambda_final:        Optional[np.ndarray] = None   # Phase 4: final pore-pressure ratio λ

    def max_inundation_area(self, h_threshold: float = 0.01) -> float:
        return np.sum(self.h_max > h_threshold) * self.terrain.dx * self.terrain.dy

    def volume_balance(self) -> dict:
        if not self.volume_history:
            return {}
        V_final = self.volume_history[-1][1]
        V_input = (self.initial_volume + self.total_inflow_vol
                   + self.total_entrained_vol)
        if V_input < 1e-3:
            return {"V_initial_m3":   self.initial_volume,
                    "V_inflow_m3":    self.total_inflow_vol,
                    "V_entrained_m3": self.total_entrained_vol,
                    "V_final_m3":     V_final,
                    "V_loss_pct":     float("nan")}
        return {"V_initial_m3":   self.initial_volume,
                "V_inflow_m3":    self.total_inflow_vol,
                "V_entrained_m3": self.total_entrained_vol,
                "V_input_total":  V_input,
                "V_final_m3":     V_final,
                "V_loss_pct":     100. * (V_input - V_final) / V_input}
