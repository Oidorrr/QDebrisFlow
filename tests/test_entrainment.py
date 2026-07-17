"""
Phase 2 (A) — bed entrainment + evolving Cv field.

Checks:
1. Mass closure: with no inflow on a closed domain, the flow's volume gain
   equals the volume entrained from the bed, and the bed loses exactly that
   (flow + bed conserved to machine precision).
2. Bulking + auto-stiffening: on a steep erodible plane the flow grows and its
   concentration Cv rises toward the Takahashi equilibrium (which drives τy, η
   up through the exp(Cv) closure).
3. OFF == baseline: with entrainment disabled there is no bed change, no
   entrained volume, and volume is plainly conserved.
"""

from __future__ import annotations

import numpy as np

import _pathsetup  # noqa: F401
import synthetic


def _steep_erodible_run(entrainment, Cv0=0.30, coef=0.3, slope=0.35):
    dem, dx, dy = synthetic.tilted_plane(ny=60, nx=30, slope=slope)
    h0 = synthetic.central_blob(60, 30, depth=3.0, radius=4)
    res = synthetic.run_scenario(
        dem, dx, dy, h0=h0,
        params_kw=dict(Cv=Cv0, manning_n=0.06, entrain_coef=coef,
                       bed_friction_angle_deg=37.0),
        config_kw=dict(t_end=300.0, dt_max=8.0, cfl_number=0.5,
                       entrainment=entrainment),
    )
    return res, dx * dy


def test_entrainment_conserves_mass():
    res, area = _steep_erodible_run(entrainment=True)
    V_init = res.initial_volume
    V_final_flow = float(res.h_final.sum()) * area
    entrained = res.total_entrained_vol
    bed = float(res.bed_change.sum()) * area
    V_input = V_init + entrained
    print(f"  V_init={V_init:.1f}  V_final={V_final_flow:.1f}  entrained={entrained:.1f}")
    print(f"  flow_gain-entrained={V_final_flow - V_init - entrained:.3e}  "
          f"bed+entrained={bed + entrained:.3e}")
    assert abs(V_final_flow - V_init - entrained) < 1e-6 * V_input, "flow mass not closed"
    assert abs(bed + entrained) < 1e-6 * V_input, "bed/flow exchange not closed"


def test_entrainment_bulks_and_stiffens():
    res, _ = _steep_erodible_run(entrainment=True, Cv0=0.30)
    bulk = float(res.h_final.sum()) / max(res.initial_volume / (res.terrain.dx * res.terrain.dy), 1e-9)
    cv_wet = res.cv_final[res.cv_final > 0]
    cv_mean = float(cv_wet.mean())
    print(f"  bulking V_final/V_init={bulk:.2f}  mean Cv_wet={cv_mean:.3f} (init 0.30)")
    assert bulk > 1.5, "flow did not bulk up on an erodible steep plane"
    assert cv_mean > 0.35, "Cv did not rise toward the equilibrium concentration"


def test_entrainment_off_is_inert():
    res, area = _steep_erodible_run(entrainment=False)
    V_init = res.initial_volume
    V_final_flow = float(res.h_final.sum()) * area
    print(f"  OFF: entrained={res.total_entrained_vol:.3e}  "
          f"max|bed_change|={np.max(np.abs(res.bed_change)):.3e}  "
          f"V_final/V_init={V_final_flow / V_init:.4f}")
    assert res.total_entrained_vol == 0.0
    assert np.max(np.abs(res.bed_change)) == 0.0
    assert abs(V_final_flow - V_init) < 1e-6 * V_init, "closed domain leaked volume"


if __name__ == "__main__":
    test_entrainment_conserves_mass()
    test_entrainment_bulks_and_stiffens()
    test_entrainment_off_is_inert()
    print("test_entrainment: PASS")
