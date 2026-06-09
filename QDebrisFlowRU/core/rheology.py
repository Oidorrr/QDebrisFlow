"""
Реологическая модель O'Brien & Julien (1985, 1988) — квадратичная.

Уклон трения (FLO-2D):
    Sf = τy/(γm·h)  +  K·η·V/(8·γm·h²)  +  n²·V²/h^(4/3)

Два режима задания τy и η
─────────────────────────
Режим 1 — ПРЯМОЙ ВВОД (рекомендуется):
    tau_yield_direct [Па]   — аналог "Yield Stress" в HEC-RAS
    eta_direct       [Па·с] — аналог "Mixture Viscosity" в HEC-RAS

Режим 2 — ЭКСПОНЕНЦИАЛЬНЫЕ КОЭФФИЦИЕНТЫ (O'Brien & Julien 1988):
    τy = alpha1 × exp(beta1 × Cv)
    η  = alpha2 × exp(beta2 × Cv)

Переключение: если tau_yield_direct > 0 → Режим 1, иначе → Режим 2.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

RHO_WATER:    float = 1000.0
RHO_SEDIMENT: float = 2650.0
GRAVITY:      float = 9.81


@dataclass
class OBrienParameters:
    # ── Концентрация ──────────────────────────────────────────────────────────
    Cv:           float = 0.40
    Cv_max:       float = 0.615

    # ── Плотности компонентов [кг/м³] ────────────────────────────────────────
    rho_sediment: float = RHO_SEDIMENT   # плотность твёрдого осадка
    rho_water:    float = RHO_WATER      # плотность жидкой фазы

    # ── Режим 1: прямой ввод τy и η ──────────────────────────────────────────
    tau_yield_direct: float = 0.0
    """Предел текучести [Па]. Если > 0 — используется напрямую (Режим 1)."""

    eta_direct: float = 0.0
    """Вязкость смеси [Па·с]. Используется вместе с tau_yield_direct."""

    # ── Режим 2: экспоненциальные коэффициенты O'Brien 1988 ──────────────────
    alpha1: float = 0.00423
    beta1:  float = 13.11
    alpha2: float = 0.0311
    beta2:  float = 16.81

    # ── Турбулентный член ─────────────────────────────────────────────────────
    manning_n: float = 0.06
    K_visc:    float = 24.0  # коэф. формы для вязкого члена (24 = широкий канал)

    h_min: float = 0.005  # минимальная глубина мокрой ячейки [м]

    # ── Производные ───────────────────────────────────────────────────────────
    rho_mixture:   float = field(init=False)
    gamma_mixture: float = field(init=False)
    tau_yield:     float = field(init=False)
    eta:           float = field(init=False)
    _direct_mode:  bool  = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.Cv < self.Cv_max):
            raise ValueError(f"Cv={self.Cv} должен быть в (0, {self.Cv_max})")
        if self.manning_n <= 0:
            raise ValueError(f"Manning n должен быть > 0")
        if self.tau_yield_direct < 0 or self.eta_direct < 0:
            raise ValueError("tau_yield_direct и eta_direct не могут быть отрицательными")
        if self.tau_yield_direct > 0 and self.eta_direct <= 0:
            raise ValueError("При tau_yield_direct > 0 необходимо задать eta_direct > 0")

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
        return "Прямой ввод (HEC-RAS)" if self._direct_mode else "Экспоненциальные коэф."

    def h_stop(self, slope: float) -> float:
        """Минимальная глубина для движения: h_stop = τy / (γm × S)."""
        if slope <= 1e-10:
            return np.inf
        return self.tau_yield / (self.gamma_mixture * slope)

    def summary(self) -> str:
        return "\n".join([
            f"── O'Brien [{self.mode_name}] ──",
            f"  τy = {self.tau_yield:.3f} Па",
            f"  η  = {self.eta:.4f} Па·с",
            f"  Cv = {self.Cv:.3f},  ρm = {self.rho_mixture:.1f} кг/м³",
            f"  ρs = {self.rho_sediment:.0f} кг/м³,  ρw = {self.rho_water:.0f} кг/м³",
            f"  n  = {self.manning_n:.3f}",
            f"  h_stop (10%) = {self.h_stop(0.10):.4f} м",
        ])
