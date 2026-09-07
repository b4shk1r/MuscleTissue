#!/usr/bin/env python3
"""
Crop MuscleInit_Complete_Final.piff to a smaller sub-window, add a synthetic
Wall ring around the new perimeter, and report the resulting fiber-cluster
count plus the SSC/macrophage/fibroblast seed counts it implies.

Preserves the source PIFF's cell grouping: pixels that belong to the same cell
in the full image stay one (possibly clipped) cell in the crop. An earlier
version wrote every pixel as its own cell, which destroyed the fiber structure
and made the quarter model both wrong and *slower* than the full one.

Standalone, stdlib-only. Run from the repo root:
    python3 make_quarter_piff.py [--force]

Edit CROP_* / OUTPUT_PIFF below for a different window (e.g. a 1/9-scale crop).
"""
import sys

# --- Config: edit these to produce a different crop ---
INPUT_PIFF = "MuscleInit_Complete_Final.piff"
OUTPUT_PIFF = "MuscleInit_Quarter.piff"
CROP_X0, CROP_X1 = 80, 240   # half-open, matches numpy slicing convention
CROP_Y0, CROP_Y1 = 104, 312
WALL_BORDER_PX = 2           # thickness of synthetic Wall ring added at the new edges

# Must match MuscleRegenSteppables.py's sscToInitialFiber / macToInitialFiber / fibroblastToInitialFiber
SSC_TO_FIBER = 4
MAC_TO_FIBER = 0.2
FIBRO_TO_FIBER = 2

FORCE = "--force" in sys.argv  # bypass the zero-seed abort if explicitly requested


def load_piff(path):
    """dict[(x,y)] = (cluster_id, cell_id, type) for every pixel of the source."""
    grid = {}
    with open(path) as f:
        f.readline()  # "Include Clusters" header
        for line in f:
            parts = line.split()
            if len(parts) != 9:
                continue  # skip the one malformed trailing row in the shipped file
            c_cluster, c_cell, ctype, x0, x1, y0, y1, z0, z1 = parts
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            assert x0 == x1 and y0 == y1, f"expected single-pixel rows, got {line!r}"
            grid[(x0, y0)] = (int(c_cluster), int(c_cell), ctype)
    return grid


def crop_and_shift(grid, x0, x1, y0, y1):
    """Slice [x0:x1) x [y0:y1) and shift so the crop starts at (0,0)."""
    out = {}
    for x in range(x0, x1):
        for y in range(y0, y1):
            v = grid.get((x, y))
            if v is not None:
                out[(x - x0, y - y0)] = v
    return out


def add_wall_border(cropped, w, h, thickness, wall_id):
    """Overwrite the perimeter ring with one frozen Wall cell (like the source's
    outer wall). One cell, not one-per-pixel, keeps the Potts cell count low."""
    for x in range(w):
        for y in range(h):
            if x < thickness or x >= w - thickness or y < thickness or y >= h - thickness:
                cropped[(x, y)] = (wall_id, wall_id, "Wall")
    return cropped


def count_fiber_clusters(cropped):
    """4-connected union-find over Fiber pixels -> approximates FiberSteppable's
    numFiberCells (its per-cluster grouping of touching fiber cells)."""
    fiber_coords = [xy for xy, v in cropped.items() if v[2] == "Fiber"]
    fiber_set = set(fiber_coords)
    parent = {c: c for c in fiber_coords}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (x, y) in fiber_coords:
        for dx, dy in ((1, 0), (0, 1)):
            n = (x + dx, y + dy)
            if n in fiber_set:
                union((x, y), n)

    clusters = len(set(find(c) for c in fiber_coords))
    return len(fiber_coords), clusters


def write_piff(cropped, path):
    """Remap surviving ids to compact ints, keeping same source cell -> one cell."""
    cluster_map, cell_map = {}, {}

    def rc(i):
        return cluster_map.setdefault(i, len(cluster_map))

    def rk(key):
        return cell_map.setdefault(key, len(cell_map))

    with open(path, "w") as f:
        f.write("Include Clusters\n")
        for (x, y), (cluster_id, cell_id, ctype) in sorted(cropped.items()):
            f.write(f"{rc(cluster_id)} {rk((cluster_id, cell_id))} {ctype} "
                    f"{x} {x} {y} {y} 0 0\n")
    return len(cell_map)


def main():
    w, h = CROP_X1 - CROP_X0, CROP_Y1 - CROP_Y0

    grid = load_piff(INPUT_PIFF)
    cropped = crop_and_shift(grid, CROP_X0, CROP_X1, CROP_Y0, CROP_Y1)
    max_src_id = max(max(c, k) for c, k, _ in cropped.values())
    cropped = add_wall_border(cropped, w, h, WALL_BORDER_PX, wall_id=max_src_id + 1)

    fiber_px, num_fiber_cells = count_fiber_clusters(cropped)
    ssc = int(num_fiber_cells // SSC_TO_FIBER)
    mac = int(num_fiber_cells * MAC_TO_FIBER)
    fibro = int(num_fiber_cells // FIBRO_TO_FIBER)

    print(f"Crop window: x[{CROP_X0}:{CROP_X1}] y[{CROP_Y0}:{CROP_Y1}] -> {w}x{h}")
    print(f"Fiber pixels: {fiber_px}, fiber clusters (numFiberCells): {num_fiber_cells}")
    print(f"Implied seed counts -> SSC: {ssc}, resident macrophage: {mac}, fibroblast: {fibro}")

    zero_types = [name for name, n in (("SSC", ssc), ("resident macrophage", mac), ("fibroblast", fibro)) if n <= 0]
    if zero_types and not FORCE:
        print(f"ABORTING: this crop would seed 0 {', '.join(zero_types)} cell(s). "
              f"Pick a denser crop window, or rerun with --force to write the PIFF anyway.")
        sys.exit(1)
    elif zero_types:
        print(f"WARNING: proceeding with 0 {', '.join(zero_types)} cell(s) due to --force.")

    n_cells = write_piff(cropped, OUTPUT_PIFF)
    print(f"Wrote {OUTPUT_PIFF} ({w * h} pixels, {n_cells} cells)")


if __name__ == "__main__":
    main()
