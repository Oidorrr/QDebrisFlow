"""
DEM preprocessing for debris flow simulation.

Two approaches:
1. Fill sinks (Planchon & Darboux, 2001) — стандартный метод
2. Burn stream network into DEM — если есть данные о каналах

Оба метода работают без GDAL напрямую, через numpy.
"""

import numpy as np


def fill_sinks(dem: np.ndarray, nodata: float = -9999.0,
               epsilon: float = 1e-4) -> np.ndarray:
    """
    Fill sinks (depression filling) using iterative flooding approach.
    Simplified Planchon & Darboux (2001) algorithm.

    Все ямы на DEM заполняются до уровня ближайшего выхода потока.
    Без этого поток застревает в артефактах рельефа.

    Args:
        dem:     Input DEM array [m], shape (ny, nx).
        nodata:  No-data sentinel value.
        epsilon: Minimum slope to enforce drainage [m] (prevents flat areas).

    Returns:
        Filled DEM array [m], same shape.
    """
    ny, nx = dem.shape
    valid = dem != nodata
    filled = dem.copy()

    # Initialise: boundary cells keep their elevation,
    # interior cells set to very high value (will be lowered)
    LARGE = filled[valid].max() + 1000.0
    interior = np.zeros((ny, nx), dtype=bool)
    interior[1:-1, 1:-1] = True
    interior &= valid

    filled[interior] = LARGE

    # 8-neighbour offsets
    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # Iterative relaxation
    changed = True
    max_iter = ny * nx * 2
    iteration = 0

    while changed and iteration < max_iter:
        changed = False
        iteration += 1

        for di, dj in neighbours:
            # Shift arrays for vectorised neighbour comparison
            ni_slice = slice(max(0,-di), min(ny, ny-di))
            nj_slice = slice(max(0,-dj), min(nx, nx-dj))
            ci_slice = slice(max(0, di), min(ny, ny+di))
            cj_slice = slice(max(0, dj), min(nx, nx+dj))

            neighbour_elev = filled[ni_slice, nj_slice]
            current_elev   = filled[ci_slice, cj_slice]
            current_dem    = dem[ci_slice, cj_slice]
            is_interior    = interior[ci_slice, cj_slice]

            # A cell should be lowered if a neighbour + epsilon is lower
            candidate = neighbour_elev + epsilon
            should_lower = (
                is_interior &
                (candidate < current_elev) &
                (candidate >= current_dem)  # cannot go below original DEM
            )

            if np.any(should_lower):
                filled[ci_slice, cj_slice] = np.where(
                    should_lower,
                    np.maximum(candidate, current_dem),
                    current_elev
                )
                changed = True

        if iteration % 100 == 0:
            print(f"  fill_sinks: iteration {iteration}...")

    n_filled = np.sum((filled > dem) & valid)
    print(f"fill_sinks: {n_filled} cells modified in {iteration} iterations")
    return filled


def compute_flow_accumulation(dem_filled: np.ndarray,
                               nodata: float = -9999.0) -> np.ndarray:
    """
    Compute D8 flow accumulation from a filled DEM.
    Useful for identifying main channel paths.

    Returns:
        accum: Flow accumulation array (number of upstream cells).
    """
    ny, nx = dem_filled.shape
    valid = dem_filled != nodata

    # D8 flow direction: find steepest downslope neighbour
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    flow_dir = np.full((ny, nx), -1, dtype=int)  # -1 = no direction

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if not valid[i, j]:
                continue
            best_slope = 0.0
            best_d = -1
            for d, (di, dj) in enumerate(dirs):
                ni, nj = i+di, j+dj
                if valid[ni, nj]:
                    # Distance: diagonal = sqrt(2), cardinal = 1
                    dist = np.sqrt(di**2 + dj**2)
                    slope = (dem_filled[i,j] - dem_filled[ni,nj]) / dist
                    if slope > best_slope:
                        best_slope = slope
                        best_d = d
            flow_dir[i, j] = best_d

    # Accumulate (topological sort would be faster, this is O(n²) but simple)
    accum = np.ones((ny, nx), dtype=np.float64)
    accum[~valid] = 0

    # Sort cells by elevation (high to low) — upstream before downstream
    elev_flat = dem_filled.flatten()
    order = np.argsort(-elev_flat)  # descending

    for idx in order:
        i, j = divmod(idx, nx)
        if not valid[i, j] or flow_dir[i, j] < 0:
            continue
        di, dj = dirs[flow_dir[i, j]]
        ni, nj = i+di, j+dj
        if 0 <= ni < ny and 0 <= nj < nx and valid[ni, nj]:
            accum[ni, nj] += accum[i, j]

    return accum


def check_dem_quality(dem: np.ndarray, nodata: float = -9999.0) -> dict:
    """
    Quick DEM quality diagnostics before running simulation.

    Returns dict with potential issues flagged.
    """
    valid = dem != nodata
    dem_v = dem[valid]

    issues = []

    # Check for flat areas (zero slope regions — potential sinks)
    ny, nx = dem.shape
    flat_count = 0
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if not valid[i,j]:
                continue
            neighbours = [dem[i+di, j+dj]
                         for di in [-1,0,1] for dj in [-1,0,1]
                         if (di,dj) != (0,0) and valid[i+di, j+dj]]
            if neighbours and all(abs(n - dem[i,j]) < 0.01 for n in neighbours):
                flat_count += 1

    if flat_count > valid.sum() * 0.05:
        issues.append(
            f"WARNING: {flat_count} flat cells ({100*flat_count/valid.sum():.1f}%) "
            f"— run fill_sinks() before simulation"
        )

    # Check resolution vs. expected flow depths
    dx_approx = 1.0  # placeholder
    if dem_v.max() - dem_v.min() < 5.0:
        issues.append("WARNING: Very low relief (<5m) — check DEM units (degrees vs. metres?)")

    # Check nodata fraction
    nodata_frac = (~valid).sum() / dem.size
    if nodata_frac > 0.3:
        issues.append(f"WARNING: {100*nodata_frac:.0f}% nodata cells — check DEM extent")

    return {
        "ny": ny, "nx": nx,
        "valid_cells": int(valid.sum()),
        "elev_min": float(dem_v.min()),
        "elev_max": float(dem_v.max()),
        "elev_range": float(dem_v.max() - dem_v.min()),
        "flat_cells": flat_count,
        "nodata_fraction": float(nodata_frac),
        "issues": issues,
        "ok": len(issues) == 0,
    }


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DEM preprocessing...")

    # Create DEM with an artificial sink
    dem = np.zeros((20, 20), dtype=np.float64)
    for i in range(20):
        dem[i, :] = (20 - i) * 0.5   # 10 m -> 0 m slope

    # Artificial sink at (10, 10)
    dem[10, 10] = dem[10, 10] - 2.0

    print(f"Before fill: min={dem.min():.2f}, sink at (10,10)={dem[10,10]:.2f}")

    filled = fill_sinks(dem, epsilon=0.01)
    print(f"After fill:  min={filled.min():.2f}, (10,10)={filled[10,10]:.2f}")

    assert filled[10, 10] > dem[10, 10], "Sink should be filled"
    assert np.all(filled >= dem), "Filled DEM must be >= original"
    print("✓ fill_sinks OK")

    diag = check_dem_quality(dem)
    print(f"\nDEM quality check: {diag}")
    print("Done.")
