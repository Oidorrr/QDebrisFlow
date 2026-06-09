"""
core/sweep_runner.py
====================
Модуль-исполнитель одиночного LHS-прогона.

Намеренно НЕ импортирует PyQt5, QGIS и dialog.py —
это позволяет запускать функцию в дочерних процессах
(multiprocessing.Pool) без инициализации QGIS.

Публичный API:
    run_one_sample(args: dict) -> dict
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# ── константа минимальной глубины ─────────────────────────────────────────────
_H_MIN: float = 0.005   # м


# ──────────────────────────────────────────────────────────────────────────────
# Метрики качества (Barnhart et al. 2021, Section 6)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_sweep_metrics(
        h_max_sim:  np.ndarray,
        obs_mask:   np.ndarray,
        obs_rows:   np.ndarray,
        obs_cols:   np.ndarray,
        obs_depths: np.ndarray,
        threshold:  float = 0.5,
) -> dict:
    """
    Вычисляет все метрики для одного прогона.

    TP, FP, FN — доли от объединения (TP+FP+FN = 1):
        TP = истинный охват   ∈ [0, 1]
        FP = переохват        ∈ [0, 1]
        FN = недоохват        ∈ [0, 1]

    ΩT  = (TP−FP−FN)/(TP+FP+FN)          [Heiser 2017, eq.2]
    ΩTm = (1−ΩT)/2                        0=лучший, 1=худший
    Δo  = Σ|ri|[ri<0] / Σ|d_obs|          переоценка глубины
    Δu  = Σ|ri|[ri>0] / Σ|d_obs|          недооценка глубины
    Cm  = (2·ΩTm + Δo,c + Δu) / 4        0=лучший, 1=худший
    """
    sim_wet  = h_max_sim >= threshold
    obs_bool = obs_mask.astype(bool)

    n_TP = int(np.sum( sim_wet &  obs_bool))
    n_FP = int(np.sum( sim_wet & ~obs_bool))
    n_FN = int(np.sum(~sim_wet &  obs_bool))

    denom_e = n_TP + n_FP + n_FN
    if denom_e > 0:
        TP_frac  = n_TP / denom_e
        FP_frac  = n_FP / denom_e
        FN_frac  = n_FN / denom_e
        omega_T  = (n_TP - n_FP - n_FN) / denom_e
        omega_Tm = (1.0 - omega_T) / 2.0
    else:
        TP_frac = FP_frac = FN_frac = 0.0
        omega_T, omega_Tm = -1.0, 1.0

    if len(obs_rows) > 0:
        d_mod = h_max_sim[obs_rows, obs_cols]
        r     = obs_depths - d_mod          # ri = d_obs − d_mod

        denom_obs = np.sum(np.abs(obs_depths))
        if denom_obs > 1e-12:
            delta_o = float(np.sum(np.abs(r[r < 0])) / denom_obs)
            delta_u = float(np.sum(np.abs(r[r > 0])) / denom_obs)
        else:
            delta_o = delta_u = 0.0

        delta_o_c    = min(delta_o, 1.0)
        depth_bias_m = float(np.mean(-r))
        Cm = (2.0 * omega_Tm + delta_o_c + delta_u) / 4.0
    else:
        delta_o = delta_u = delta_o_c = depth_bias_m = float("nan")
        Cm = omega_Tm

    def _r4(v):
        return round(float(v), 4) if v == v else None   # NaN → None

    return dict(
        TP          = _r4(TP_frac),
        FP          = _r4(FP_frac),
        FN          = _r4(FN_frac),
        omega_Tm    = _r4(omega_Tm),
        delta_o_c   = _r4(delta_o_c),
        delta_u     = _r4(delta_u),
        Cm          = _r4(Cm),
        depth_bias_m= _r4(depth_bias_m),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Воркер одиночного прогона — вызывается в дочернем процессе
# ──────────────────────────────────────────────────────────────────────────────

def run_one_sample(args: dict) -> dict:
    """
    Выполняет один LHS-прогон. Вызывается в дочернем процессе
    (multiprocessing.Pool) или напрямую (последовательный режим).

    Параметры args (все — сериализуемые типы):
        plugin_dir      str         — путь к директории плагина
        run_id          int
        sample          dict        — {vol?, tau_yield?, eta?, manning_n?, K_visc?}
        dem             np.ndarray
        geo_info        dict        — dx, dy, nodata
        src_cells       list[dict]  — rows, cols, times, base_q_vals, name
        base_vol        float
        obs_mask        np.ndarray  — bool
        obs_rows        np.ndarray  — int
        obs_cols        np.ndarray  — int
        obs_depths      np.ndarray  — float
        Cv              float
        Cv_max          float
        param_ranges    dict
        t_end           float
        dt_max          float
        cfl_number      float
        depth_threshold float

    Возвращает row_data dict (включая h_max для отслеживания лучшего прогона).
    """
    # ── Инициализация пути (нужна в дочернем процессе) ────────────────────────
    plugin_dir = args.get("plugin_dir", "")
    if plugin_dir and plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    try:
        from core.rheology import OBrienParameters
        from core.solver   import (TerrainGrid, SimulationConfig,
                                    SourceCondition, InflowSource,
                                    DebrisFlowSolver2D)
    except ImportError as exc:
        return _error_row(args.get("run_id", -1), exc)

    run_id = args["run_id"]
    t0     = time.time()
    try:
        sample     = args["sample"]
        base_vol   = args["base_vol"]
        geo        = args["geo_info"]
        p_ranges   = args["param_ranges"]

        # ── Масштаб объёма ──────────────────────────────────────────────────
        target_vol = float(sample.get("vol", base_vol))
        scale      = target_vol / base_vol

        # ── Параметры реологии ──────────────────────────────────────────────
        def _get(key, fallback):
            v = sample.get(key)
            if v is not None:
                return float(v)
            lo, hi = p_ranges.get(key, (fallback, fallback))
            return float(lo)

        tau_y = _get("tau_yield", 600.)
        eta   = _get("eta",       10.)
        n_man = _get("manning_n", 0.06)
        k_val = _get("K_visc",   24.)

        obrien = OBrienParameters(
            Cv               = args["Cv"],
            Cv_max           = args["Cv_max"],
            rho_sediment     = args.get("rho_s", 2650.0),
            rho_water        = args.get("rho_w", 1000.0),
            tau_yield_direct = tau_y,
            eta_direct       = eta,
            manning_n        = n_man,
            K_visc           = k_val,
            h_min            = _H_MIN,
        )

        terrain = TerrainGrid(
            dem    = args["dem"],
            dx     = geo["dx"],
            dy     = geo["dy"],
            nodata = geo["nodata"],
        )

        inflow_srcs = [
            InflowSource(
                rows    = sc["rows"],
                cols    = sc["cols"],
                times   = sc["times"],
                q_total = sc["base_q_vals"] * scale,
                name    = sc["name"],
            )
            for sc in args["src_cells"]
        ]

        config = SimulationConfig(
            t_end      = args["t_end"],
            dt_max     = args["dt_max"],
            cfl_number = args["cfl_number"],
        )

        result = DebrisFlowSolver2D(
            terrain, obrien,
            SourceCondition.from_sources(inflow_srcs),
            config,
        ).run()

        metrics = _compute_sweep_metrics(
            result.h_max,
            args["obs_mask"],
            args["obs_rows"],
            args["obs_cols"],
            args["obs_depths"],
            threshold = args["depth_threshold"],
        )

        wall_s = time.time() - t0

        return dict(
            run_id       = run_id,
            status       = "ok",
            vol_m3       = round(target_vol, 0),
            tau_yield_Pa = round(tau_y,  1),
            eta_Pas      = round(eta,    2),
            manning_n    = round(n_man,  4),
            K_visc       = round(k_val,  1),
            h_max_m      = round(float(result.h_max.max()), 3),
            h_max        = result.h_max,        # для определения лучшего прогона
            wall_s       = round(wall_s, 1),
            **metrics,
        )

    except Exception as exc:
        return _error_row(run_id, exc)


def _error_row(run_id: int, exc: Exception) -> dict:
    """Формирует строку с ошибкой (все числовые поля = None)."""
    null_keys = [
        "vol_m3", "tau_yield_Pa", "eta_Pas", "manning_n", "K_visc",
        "h_max_m", "wall_s",
        "TP", "FP", "FN", "omega_Tm", "delta_o_c", "delta_u",
        "Cm", "depth_bias_m",
    ]
    return dict(
        run_id = run_id,
        status = f"error: {exc}",
        h_max  = None,
        **{k: None for k in null_keys},
    )
