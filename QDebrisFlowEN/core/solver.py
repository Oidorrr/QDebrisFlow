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


def boundary_cells(mask_valid: np.ndarray) -> np.ndarray:
    """
    Valid cells on the domain boundary: the edge of the grid or next to a
    nodata cell. All boundary faces are closed walls (no outflow), so flow
    that reaches these cells accumulates there instead of leaving.
    """
    m = np.asarray(mask_valid, dtype=bool)
    b = np.zeros_like(m)
    b[0, :] = b[-1, :] = True
    b[:, 0] = b[:, -1] = True
    inv = ~m
    b[1:, :] |= inv[:-1, :]
    b[:-1, :] |= inv[1:, :]
    b[:, 1:] |= inv[:, :-1]
    b[:, :-1] |= inv[:, 1:]
    return b & m


# ─────────────────────────────────────────────────────────────────────────────
# InflowSource + SourceCondition + SimulationConfig
# ─────────────────────────────────────────────────────────────────────────────

def hydrograph_volume(times, q, t0: float, t1: float) -> float:
    """
    Exact volume [m³] of the piecewise-linear hydrograph Q(t) between t0 and t1.
    Q = 0 outside [times[0], times[-1]] — the same convention as InflowSource.q_at.
    """
    times = np.asarray(times, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if times.size < 2 or t1 <= t0:
        return 0.0
    a = max(t0, times[0])
    b = min(t1, times[-1])
    if b <= a:
        return 0.0
    inner = times[np.searchsorted(times, a, side="right"):np.searchsorted(times, b, side="left")]
    tt = np.concatenate(([a], inner, [b]))
    qq = np.interp(tt, times, q)
    return float(np.sum(0.5 * (qq[1:] + qq[:-1]) * np.diff(tt)))


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

    def volume_between(self, t0: float, t1: float) -> float:
        return hydrograph_volume(self.times, self.q_total, t0, t1)

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
    progress_callback: Optional[Callable[[float, float], None]] = None
    # Face-depth limiter hf ≤ ½(hL+hR). Earlier attributed to Almeida et al. (2012), eq. 2,
    # but this formula is not found in the published restatements of that paper; its
    # source is unconfirmed. It slows wetting fronts (test fan: arrival +7…16 %) and
    # amplifies channel pulsations. ON by default = behaviour of the published runs.
    face_depth_limiter: bool = True
    # θ-weighting of the discharge in the momentum update (de Almeida et al. 2012;
    # Cozzolino et al. 2021, eq. 19): θ·q + (1−θ)/2·(q_prev + q_next). Damps the LIA
    # oscillations in steep low-friction channels. 1.0 = original Bates et al. (2010)
    # scheme = behaviour of the published runs; Landlab uses 0.8 by default.
    theta: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Computational kernels
# ─────────────────────────────────────────────────────────────────────────────

if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, nogil=True)
    def _lia_step(h, Qx, Qy, z, mask, dx, dy, dt, h_min, n2, K_eta_8gm, tau_gm, lim, theta):
        G = 9.81
        ny, nx = h.shape[0], h.shape[1]
        nxm, nym = nx - 1, ny - 1

        # X-faces
        # Trial discharge F (inertia + pressure) and semi-implicit denominator d.
        # Manning (quadratic) and viscosity (linear) — semi-implicit (Bates 2010) with the
        # COUPLED friction: |q| includes the transverse component averaged from the four
        # surrounding faces (Cozzolino et al. 2021, eqs. 19-21; de Almeida & Bates 2013)
        Fx = np.zeros_like(Qx); Dx = np.zeros_like(Qx)
        for i in range(ny):
            for j in range(nxm):
                if not (mask[i, j] and mask[i, j+1]):
                    continue
                eL = z[i, j]   + h[i, j]
                eR = z[i, j+1] + h[i, j+1]
                hf = (eL if eL > eR else eR) - (z[i,j] if z[i,j] > z[i,j+1] else z[i,j+1])
                if hf != hf or hf <= h_min:
                    continue
                if lim: hf = min(hf, 0.5*(h[i, j] + h[i, j+1]))   # optional limiter hf ≤ ½(hL+hR), see SimulationConfig.face_depth_limiter
                if hf <= h_min:
                    continue
                q  = Qx[i, j]
                qm = q   # θ-weighting (de Almeida et al. 2012); missing/nodata neighbours are dropped
                if theta < 1.:
                    nb = 0; s = 0.
                    if j > 0 and mask[i, j-1]:     s += Qx[i, j-1]; nb += 1
                    if j < nxm-1 and mask[i, j+2]: s += Qx[i, j+1]; nb += 1
                    if nb == 2:   qm = theta*q + (1.-theta)*0.5*s
                    elif nb == 1: qm = theta*q + (1.-theta)*s
                qt = 0.
                if i > 0:   qt += Qy[i-1, j] + Qy[i-1, j+1]
                if i < nym: qt += Qy[i, j]   + Qy[i, j+1]
                qt *= 0.25
                vm = (q*q + qt*qt)**0.5 / hf
                sf = n2 * vm / hf**(4./3.) + K_eta_8gm / hf**2
                d  = 1. + G * dt * sf
                if d < 1.: d = 1.
                Fx[i, j] = qm - G * hf * dt * (eR - eL) / dx
                Dx[i, j] = d

        # Y-faces
        Fy = np.zeros_like(Qy); Dy = np.zeros_like(Qy)
        for i in range(nym):
            for j in range(nx):
                if not (mask[i, j] and mask[i+1, j]):
                    continue
                eU = z[i, j]   + h[i, j]
                eD = z[i+1, j] + h[i+1, j]
                hf = (eU if eU > eD else eD) - (z[i,j] if z[i,j] > z[i+1,j] else z[i+1,j])
                if hf != hf or hf <= h_min:
                    continue
                if lim: hf = min(hf, 0.5*(h[i, j] + h[i+1, j]))
                if hf <= h_min:
                    continue
                q  = Qy[i, j]
                qm = q
                if theta < 1.:
                    nb = 0; s = 0.
                    if i > 0 and mask[i-1, j]:     s += Qy[i-1, j]; nb += 1
                    if i < nym-1 and mask[i+2, j]: s += Qy[i+1, j]; nb += 1
                    if nb == 2:   qm = theta*q + (1.-theta)*0.5*s
                    elif nb == 1: qm = theta*q + (1.-theta)*s
                qt = 0.
                if j > 0:   qt += Qx[i, j-1] + Qx[i+1, j-1]
                if j < nxm: qt += Qx[i, j]   + Qx[i+1, j]
                qt *= 0.25
                vm = (q*q + qt*qt)**0.5 / hf
                sf = n2 * vm / hf**(4./3.) + K_eta_8gm / hf**2
                d  = 1. + G * dt * sf
                if d < 1.: d = 1.
                Fy[i, j] = qm - G * hf * dt * (eD - eU) / dy
                Dy[i, j] = d

        # Yield stress τy/(γm·h) of the FLO-2D Sf acts against the trial discharge VECTOR
        # (isotropic): rest if |F| ≤ g·Δt·τy/γm, otherwise |F| is reduced by this impulse
        # BEFORE dividing by d → motion starts at hf·|∇η| > τy/γm (= rheology.h_stop)
        # for any flow direction, Δt, η and K
        yi = G * dt * tau_gm
        Qx_new = np.zeros_like(Qx)
        for i in range(ny):
            for j in range(nxm):
                d = Dx[i, j]
                if d <= 0.: continue
                F  = Fx[i, j]
                ft = 0.
                if i > 0:   ft += Fy[i-1, j] + Fy[i-1, j+1]
                if i < nym: ft += Fy[i, j]   + Fy[i, j+1]
                ft *= 0.25
                aF = (F*F + ft*ft)**0.5
                if aF > yi:
                    Qx_new[i, j] = F * (1. - yi / aF) / d
        Qy_new = np.zeros_like(Qy)
        for i in range(nym):
            for j in range(nx):
                d = Dy[i, j]
                if d <= 0.: continue
                F  = Fy[i, j]
                ft = 0.
                if j > 0:   ft += Fx[i, j-1] + Fx[i+1, j-1]
                if j < nxm: ft += Fx[i, j]   + Fx[i+1, j]
                ft *= 0.25
                aF = (F*F + ft*ft)**0.5
                if aF > yi:
                    Qy_new[i, j] = F * (1. - yi / aF) / d

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
                tot = ox * dt / dx + oy * dt / dy
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

        # Continuity
        h_new = np.empty_like(h)
        for i in range(ny):
            for j in range(nx):
                if not mask[i, j]: h_new[i, j] = 0.0; continue
                div = 0.
                if j < nxm: div -= Qx_new[i, j]   / dx
                if j > 0:   div += Qx_new[i, j-1] / dx
                if i < nym: div -= Qy_new[i, j]   / dy
                if i > 0:   div += Qy_new[i-1, j] / dy
                v = h[i, j] + dt * div
                h_new[i, j] = v if v > 0. else 0.
        return h_new, Qx_new, Qy_new

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
    def _update_max(h, Qx, Qy, h_max, V_max, t_arrival, h_min, t, HV2_max, Fr_hv2):
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
                hv2 = hv * vm * vm          # instantaneous h·v² [m³/s²]
                if hv2 > HV2_max[i, j]:
                    HV2_max[i, j] = hv2
                    Fr_hv2[i, j] = vm / (9.81 * hd)**0.5   # Froude number V/√(g·h) at the moment of peak h·v²
                if t_arrival[i, j] > 1e30: t_arrival[i, j] = t

else:
    # ── NumPy fallback ────────────────────────────────────────────────────────

    def _theta_mean(Q, valid, theta, axis):
        """θ-weighted discharge θ·q + (1−θ)/2·(q_prev + q_next) (de Almeida et al. 2012);
        missing or nodata neighbours are dropped (weights θ, 1−θ with one neighbour)."""
        if theta >= 1.0:
            return Q
        prev = np.zeros_like(Q); nxt = np.zeros_like(Q)
        vp = np.zeros(Q.shape, dtype=bool); vn = np.zeros(Q.shape, dtype=bool)
        if axis == 1:
            prev[:, 1:] = Q[:, :-1]; vp[:, 1:] = valid[:, :-1]
            nxt[:, :-1] = Q[:, 1:];  vn[:, :-1] = valid[:, 1:]
        else:
            prev[1:, :] = Q[:-1, :]; vp[1:, :] = valid[:-1, :]
            nxt[:-1, :] = Q[1:, :];  vn[:-1, :] = valid[1:, :]
        n = vp.astype(np.int64) + vn.astype(np.int64)
        tot = np.where(vp, prev, 0.) + np.where(vn, nxt, 0.)
        return np.where(n == 2, theta*Q + (1.-theta)*0.5*tot,
                        np.where(n == 1, theta*Q + (1.-theta)*tot, Q))

    def _avg4_y_to_x(A):
        """Average of the four y-face values around every x-face (missing faces = 0)."""
        P = np.zeros((A.shape[0] + 2, A.shape[1])); P[1:-1, :] = A
        return 0.25 * (P[:-1, :-1] + P[:-1, 1:] + P[1:, :-1] + P[1:, 1:])

    def _avg4_x_to_y(A):
        """Average of the four x-face values around every y-face (missing faces = 0)."""
        P = np.zeros((A.shape[0], A.shape[1] + 2)); P[:, 1:-1] = A
        return 0.25 * (P[:-1, :-1] + P[:-1, 1:] + P[1:, :-1] + P[1:, 1:])

    def _lia_step(h, Qx, Qy, z, mask, dx, dy, dt, h_min, n2, K_eta_8gm, tau_gm, lim, theta):
        G  = 9.81
        mv = mask
        eta = z + h

        # X-faces
        eL, eR = eta[:, :-1], eta[:, 1:]
        hf_x = np.maximum(np.nan_to_num(
            np.maximum(eL, eR) - np.maximum(z[:,:-1], z[:,1:]),
            nan=0., posinf=0.), 0.)
        if lim: hf_x = np.minimum(hf_x, 0.5*(h[:,:-1]+h[:,1:]))   # optional limiter, see SimulationConfig.face_depth_limiter
        wet_x = (hf_x > h_min) & mv[:, :-1] & mv[:, 1:]

        # Y-faces
        eU, eD = eta[:-1, :], eta[1:, :]
        hf_y = np.maximum(np.nan_to_num(
            np.maximum(eU, eD) - np.maximum(z[:-1,:], z[1:,:]),
            nan=0., posinf=0.), 0.)
        if lim: hf_y = np.minimum(hf_y, 0.5*(h[:-1,:]+h[1:,:]))
        wet_y = (hf_y > h_min) & mv[:-1,:] & mv[1:,:]

        # Coupled friction (Cozzolino et al. 2021, eqs. 19-21): |q| includes the transverse
        # component averaged from the four surrounding faces
        hfx = np.maximum(hf_x, 1e-12); hfy = np.maximum(hf_y, 1e-12)
        sf_x = (n2 * np.sqrt(Qx**2 + _avg4_y_to_x(Qy)**2) / hfx / hfx**(4./3.)
                + K_eta_8gm / hfx**2)
        sf_y = (n2 * np.sqrt(Qy**2 + _avg4_x_to_y(Qx)**2) / hfy / hfy**(4./3.)
                + K_eta_8gm / hfy**2)
        denom_x = np.maximum(1. + G*dt*sf_x, 1.)
        denom_y = np.maximum(1. + G*dt*sf_y, 1.)
        qm_x = _theta_mean(Qx, mv[:, :-1] & mv[:, 1:], theta, 1)
        qm_y = _theta_mean(Qy, mv[:-1, :] & mv[1:, :], theta, 0)
        F_x = np.where(wet_x, qm_x - G*hf_x*dt*np.nan_to_num((eR-eL)/dx), 0.)
        F_y = np.where(wet_y, qm_y - G*hf_y*dt*np.nan_to_num((eD-eU)/dy), 0.)

        # τy acts against the trial discharge VECTOR (isotropic): rest if |F| ≤ g·Δt·τy/γm,
        # otherwise |F| is reduced by the yield impulse before dividing by the denominator
        yield_impulse = G * dt * tau_gm
        Fm_x = np.sqrt(F_x**2 + _avg4_y_to_x(F_y)**2)
        Fm_y = np.sqrt(F_y**2 + _avg4_x_to_y(F_x)**2)
        Qx_new = np.where(wet_x & (Fm_x > yield_impulse),
                          F_x * (1. - yield_impulse / np.maximum(Fm_x, 1e-300)) / denom_x, 0.)
        Qy_new = np.where(wet_y & (Fm_y > yield_impulse),
                          F_y * (1. - yield_impulse / np.maximum(Fm_y, 1e-300)) / denom_y, 0.)

        # Outflow limiter
        ox = np.zeros_like(h)
        ox[:,:-1] += np.maximum(Qx_new,0.); ox[:,1:] += np.maximum(-Qx_new,0.)
        oy = np.zeros_like(h)
        oy[:-1,:] += np.maximum(Qy_new,0.); oy[1:,:] += np.maximum(-Qy_new,0.)
        tot = ox*dt/dx + oy*dt/dy
        sc = np.where(tot > h, 0.99*h/(tot+1e-12), 1.)
        Qx_new *= np.minimum(sc[:,:-1], sc[:,1:])
        Qy_new *= np.minimum(sc[:-1,:], sc[1:,:])

        # Continuity
        div = np.zeros_like(h)
        div[:,:-1] -= Qx_new/dx; div[:,1:]  += Qx_new/dx
        div[:-1,:] -= Qy_new/dy; div[1:,:]  += Qy_new/dy
        h_new = np.maximum(h + dt*div, 0.)
        h_new[~mv] = 0.
        return h_new, Qx_new, Qy_new

    def _cfl_vol_wet(h, mask, cfl, dx_min, dx, dy, h_min):
        hv    = h[mask]
        pos   = hv[hv > 0.]
        max_c = (9.81*pos.max())**0.5 if pos.size > 0 else 0.
        dt    = (cfl*dx_min/max_c) if max_c > 1e-10 else 1e9
        return dt, pos.sum()*dx*dy, int((hv >= h_min).sum())

    def _add_inflow(h, rows, cols, rate, dt):
        h[rows, cols] += rate * dt

    def _update_max(h, Qx, Qy, h_max, V_max, t_arrival, h_min, t, HV2_max, Fr_hv2):
        np.maximum(h_max, h, out=h_max)
        ny, nx = h.shape
        hd = np.maximum(h, h_min)
        Qxc = np.zeros((ny, nx))
        Qxc[:,1:-1]=0.5*(Qx[:,:-1]+Qx[:,1:]); Qxc[:,0]=Qx[:,0]; Qxc[:,-1]=Qx[:,-1]
        Qyc = np.zeros((ny, nx))
        Qyc[1:-1,:]=0.5*(Qy[:-1,:]+Qy[1:,:]); Qyc[0,:]=Qy[0,:]; Qyc[-1,:]=Qy[-1,:]
        vm = np.where(h >= h_min, np.minimum(np.sqrt(Qxc**2+Qyc**2)/hd, 30.), 0.)   # dry cells (h < h_min) get no velocity — same as the Numba kernel
        np.maximum(V_max, vm, out=V_max)
        hv2 = np.where(h >= h_min, h * vm**2, 0.)     # instantaneous h·v² [m³/s²]
        up = hv2 > HV2_max   # Froude number V/√(g·h) at the moment of peak h·v²
        Fr_hv2[up] = (vm / np.sqrt(9.81 * hd))[up]
        np.maximum(HV2_max, hv2, out=HV2_max)
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
    hv = np.zeros((4,4),dtype=np.float64)
    fr = np.zeros((4,4),dtype=np.float64)
    r  = np.zeros(2, dtype=np.int64)
    c  = np.arange(2, dtype=np.int64)
    _lia_step(h,Qx,Qy,z,mv,10.,10.,1.,h_min,0.0036,1e-4,0.001,True,0.8)
    _cfl_vol_wet(h,mv,0.4,10.,10.,10.,h_min)
    _update_max(h,Qx,Qy,hm,vm,ta,h_min,0.,hv,fr)
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
        self.HV2_max   = np.zeros((ny, nx), dtype=np.float64)
        self.Fr_hv2    = np.zeros((ny, nx), dtype=np.float64)

        self.t               = 0.0
        self.n_steps         = 0
        self.volume_initial  = 0.0
        self.volume_history: list = []
        self._total_inflow_vol = 0.0

        # Precomputed rheological constants for the kernel
        self._n2        = params.manning_n**2
        self._K_eta_8gm = params.K_visc * params.eta / (8.*params.gamma_mixture)
        self._tau_gm    = params.tau_yield / params.gamma_mixture

        self._z      = np.ascontiguousarray(terrain.z_bed, dtype=np.float64)
        self._mv     = np.ascontiguousarray(terrain.mask_valid)
        self._dx_min = min(terrain.dx, terrain.dy)

    def _initialise(self) -> None:
        global _JIT_READY
        if self.source.h0_array is not None:
            self.h[:] = np.maximum(self.source.h0_array, 0.)
        self.h[~self.terrain.mask_valid] = 0.
        np.copyto(self.h_max, self.h)
        self._h_init = self.h.copy()
        self.t_arrival[self.h > self.params.h_min] = 0.
        # source cells on nodata would silently destroy the injected volume → drop them
        # and spread the inflow over the remaining cells of the same source
        self._sources = []
        for s in self.source.inflow_sources:
            keep = self.terrain.mask_valid[s.rows, s.cols]
            if not keep.any():
                log.warning(f"Source '{s.name}' lies entirely on nodata cells — ignored")
                continue
            if not keep.all():
                log.warning(f"Source '{s.name}': {int((~keep).sum())} of {s.n_cells} cells are "
                            "nodata — the inflow is spread over the remaining cells")
            self._sources.append(InflowSource(
                rows=np.asarray(s.rows)[keep], cols=np.asarray(s.cols)[keep],
                times=s.times, q_total=s.q_total, name=s.name))
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
        z = self._z; mv = self._mv
        dx = self.terrain.dx; dy = self.terrain.dy
        h_min  = self.params.h_min
        n2     = self._n2; Ke = self._K_eta_8gm; ty = self._tau_gm
        cfl    = self.config.cfl_number; dmin = self._dx_min
        lim    = bool(self.config.face_depth_limiter)
        theta  = float(self.config.theta)
        dt_max = self.config.dt_max; dt_min = self.config.dt_min
        t_end  = self.config.t_end; out_iv = self.config.output_interval
        cell_area = dx * dy

        t_sim = 0.; t_last = 0.; ever_wet = False; t_wall = time.time()

        while t_sim < t_end:

            dt_cfl, V, wet = _cfl_vol_wet(h, mv, cfl, dmin, dx, dy, h_min)
            dt = min(dt_cfl, dt_max)
            # CFL must also hold for the depth the inflow adds during this step (on a dry
            # domain dt_cfl = 1e9, and the first step would otherwise run with dt_max)
            for src in self._sources:
                v_in = src.volume_between(t_sim, t_sim + dt)
                if v_in > 0.:
                    h_src = (float(h[src.rows, src.cols].max())
                             + v_in / (src.n_cells * cell_area))
                    dt = min(dt, cfl * dmin / (GRAVITY * h_src)**0.5)
            if dt < dt_min:
                raise RuntimeError(f"CFL step {dt:.6f} s < dt_min={dt_min} s")
            # dt_min guards the stability-limited step only; the last step may be
            # shorter than dt_min to land exactly on t_end
            dt = min(dt, t_end - t_sim)

            # inflow: exact volume of the piecewise-linear Q(t) over [t, t+dt] (not Q(t)·dt)
            for src in self._sources:
                v_in = src.volume_between(t_sim, t_sim + dt)
                if v_in > 0.:
                    rate = v_in / (dt * src.n_cells * cell_area)
                    self._total_inflow_vol += v_in
                    if _NUMBA_AVAILABLE:
                        _add_inflow(h, src.rows.astype(np.int64),
                                    src.cols.astype(np.int64), rate, dt)
                    else:
                        h[src.rows, src.cols] += rate * dt

            h, Qx, Qy = _lia_step(h, Qx, Qy, z, mv, dx, dy, dt, h_min, n2, Ke, ty, lim, theta)
            _update_max(h, Qx, Qy, self.h_max, self.V_max, self.t_arrival, h_min, t_sim + dt,
                        self.HV2_max, self.Fr_hv2)

            # volume at the START of the step (before inflow) → labelled t_sim; stored every
            # output_interval only (one entry per step could take GBs on long runs)
            if not self.volume_history or t_sim - self.volume_history[-1][0] >= out_iv:
                self.volume_history.append((t_sim, V))
            t_sim += dt; self.n_steps += 1

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
        # final volume (after the last step) — used by volume_balance()
        self.volume_history.append((t_sim, float(np.sum(h[mv])) * cell_area))
        # boundary cells actually reached by the flow (depth grew by ≥ h_min over the initial h)
        bnd = boundary_cells(self.terrain.mask_valid) & (self.h_max - self._h_init >= h_min)
        edge_h_max = float(self.h_max[bnd].max()) if bnd.any() else 0.0
        if edge_h_max >= h_min:
            log.warning(f"Flow reached the closed domain boundary: h_max at the edge = "
                        f"{edge_h_max:.3f} m — the fluid accumulates there instead of leaving")
        n_vcap = int(np.sum(self.V_max >= 30.0 - 1e-9))   # V is clipped at 30 m/s in _update_max
        if n_vcap:
            log.warning(f"Velocity reached the 30 m/s cap in {n_vcap} cells — a symptom of LIA "
                        f"oscillations (see theta) or of cells emptied in one step")
        log.info(f"Completed [{backend}]: {self.n_steps} steps, "
                 f"{t_sim:.1f} s, {time.time()-t_wall:.2f} s CPU")

        return SimulationResult(
            h_max=self.h_max, V_max=self.V_max,
            h_final=self.h, t_arrival=self.t_arrival,
            volume_history=self.volume_history,
            t_simulated=self.t, n_steps=self.n_steps,
            params=self.params, terrain=self.terrain,
            total_inflow_vol=self._total_inflow_vol,
            initial_volume=self.volume_initial,
            HV2_max=self.HV2_max,
            Fr_hv2=self.Fr_hv2,
            edge_h_max=edge_h_max,
            n_vcap_cells=n_vcap,
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
    # max over time of the INSTANTANEOUS product h(t)·v(t)² [m³/s²] (momentum flux per
    # unit width / ρm). Always ≤ h_max·V_max², which combines two separate envelopes
    # (e.g. front velocity with body depth) and is only an upper bound.
    HV2_max:          Optional[np.ndarray] = None
    # Froude number V/√(g·h) at the moment of the peak momentum flux h·v² in each cell.
    # The LIA agrees well with the full shallow-water equations only for Fr < 0.5, errors
    # grow above (de Almeida & Bates 2013). A plain max_t(Fr) is useless as a diagnostic:
    # it is set by thin films at the front (h ≈ h_min), where V/√(g·h) is huge; sampling
    # Fr when h·v² peaks characterises the main body of the flow.
    Fr_hv2:           Optional[np.ndarray] = None
    # max h_max over boundary cells (grid edge / next to nodata); > h_min → flow hit the wall
    edge_h_max:       float = 0.0
    # number of cells whose velocity hit the 30 m/s cap (> 0 → V_max/HV2_max there are unreliable)
    n_vcap_cells:     int = 0

    def max_inundation_area(self, h_threshold: float = 0.01) -> float:
        return np.sum(self.h_max > h_threshold) * self.terrain.dx * self.terrain.dy

    def volume_balance(self) -> dict:
        if not self.volume_history:
            return {}
        V_final = self.volume_history[-1][1]
        V_input = self.initial_volume + self.total_inflow_vol
        if V_input < 1e-3:
            return {"V_initial_m3": self.initial_volume,
                    "V_inflow_m3":  self.total_inflow_vol,
                    "V_final_m3":   V_final,
                    "V_loss_pct":   float("nan")}
        return {"V_initial_m3":  self.initial_volume,
                "V_inflow_m3":   self.total_inflow_vol,
                "V_input_total": V_input,
                "V_final_m3":    V_final,
                "V_loss_pct":    100. * (V_input - V_final) / V_input}
