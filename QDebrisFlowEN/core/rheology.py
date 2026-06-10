"""
O'Brien & Julien (1985, 1988) rheological model — quadratic.

Friction slope (FLO-2D):
    Sf = τy/(γm·h)  +  K·η·V/(8·γm·h²)  +  n²·V²/h^(4/3)

Two modes for specifying τy and η
─────────────────────────────────
Mode 1 — DIRECT INPUT (recommended):
    tau_yield_direct [Pa]   — equivalent of "Yield Stress" in HEC-RAS
    eta_direct       [Pa·s] — equivalent of "Mixture Viscosity" in HEC-RAS

Mode 2 — EXPONENTIAL COEFFICIENTS (O'Brien & Julien 1988):
    τy = alpha1 × exp(beta1 × Cv)
    η  = alpha2 × exp(beta2 × Cv)

Switching: if tau_yield_direct > 0 → Mode 1, otherwise → Mode 2.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

RHO_WATER:    float = 1000.0
RHO_SEDIMENT: float = 2650.0
GRAVITY:      float = 9.81


@dataclass
class OBrienParameters:
    # ── Concentration ────────────────────────────────────────────────────────
    Cv:           float = 0.40
    Cv_max:       float = 0.615

    # ── Component densities [kg/m³] ──────────────────────────────────────────
    rho_sediment: float = RHO_SEDIMENT   # density of the solid sediment
    rho_water:    float = RHO_WATER      # density of the liquid phase

    # ── Mode 1: direct input of τy and η ─────────────────────────────────────
    tau_yield_direct: float = 0.0
    """Yield stress [Pa]. If > 0 — used directly (Mode 1)."""

    eta_direct: float = 0.0
    """Mixture viscosity [Pa·s]. Used together with tau_yield_direct."""

    # ── Mode 2: exponential coefficients O'Brien 1988 ────────────────────────
    alpha1: float = 0.00423
    beta1:  float = 13.11
    alpha2: float = 0.0311
    beta2:  float = 16.81

    # ── Turbulent term ─────────────────────────────────────────────────────────
    manning_n: float = 0.06
    K_visc:    float = 24.0  # shape factor for the viscous term (24 = wide channel)

    h_min: float = 0.005  # minimum depth of a wet cell [m]

    # ── Derived ────────────────────────────────────────────────────────────────
    rho_mixture:   float = field(init=False)
    gamma_mixture: float = field(init=False)
    tau_yield:     float = field(init=False)
    eta:           float = field(init=False)
    _direct_mode:  bool  = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.Cv < self.Cv_max):
            raise ValueError(f"Cv={self.Cv} must be in (0, {self.Cv_max})")
        if self.manning_n <= 0:
            raise ValueError(f"Manning n must be > 0")
        if self.tau_yield_direct < 0 or self.eta_direct < 0:
            raise ValueError("tau_yield_direct and eta_direct cannot be negative")
        if self.tau_yield_direct > 0 and self.eta_direct <= 0:
            raise ValueError("When tau_yield_direct > 0, eta_direct > 0 must be specified")

        self.rho_mixture   = self.rho_sediment * self.Cv + self.rho_water * (1.0 - self.Cv)
        self.gamma_mixture = self.rho_mixture * GRAVITY

        self._direct_mode = self.tau_yield_direct > 0.0
        if self._direct_mode:
            self.tau_yield = self.tau_yield_direct
            self.eta       = self.eta_direct
        else:
            self.tau_yield = self.alpha1 * np.exp(self.beta1 * self.Cv)
            self.eta       = self.alpha2 * np.exp(self.beta2 * self.Cv)

    @property
    def mode_name(self) -> str:
        return "Direct input (HEC-RAS)" if self._direct_mode else "Exponential coeff."

    def h_stop(self, slope: float) -> float:
        """Minimum depth for motion: h_stop = τy / (γm × S)."""
        if slope <= 1e-10:
            return np.inf
        return self.tau_yield / (self.gamma_mixture * slope)

    def summary(self) -> str:
        return "\n".join([
            f"── O'Brien [{self.mode_name}] ──",
            f"  τy = {self.tau_yield:.3f} Pa",
            f"  η  = {self.eta:.4f} Pa·s",
            f"  Cv = {self.Cv:.3f},  ρm = {self.rho_mixture:.1f} kg/m³",
            f"  ρs = {self.rho_sediment:.0f} kg/m³,  ρw = {self.rho_water:.0f} kg/m³",
            f"  n  = {self.manning_n:.3f}",
            f"  h_stop (10%) = {self.h_stop(0.10):.4f} m",
        ])
