#!/usr/bin/env python3
"""
Crop MuscleInit_Complete_Final.piff to a smaller sub-window, add a synthetic
Wall ring around the new perimeter, and report the resulting fiber-cluster
count plus the SSC/macrophage/fibroblast seed counts it implies.

Standalone, stdlib-only (no cc3d, no scipy). Run from the repo root:
    python3 make_quarter_piff.py [--force]

To reuse for a different crop (e.g. a 1/9-scale version), edit the
CROP_X0/X1/Y0/Y1 (and OUTPUT_PIFF) constants below.
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
    """Returns dict[(x,y)] = type_string for every pixel; asserts one row per pixel."""
    grid = {}
    with open(path) as f:
        f.readline()  # "Include Clusters" header
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cell_id, cluster_id, ctype, x0, x1, y0, y1, z0, z1 = parts
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            assert x0 == x1 and y0 == y1, f"expected single-pixel rows, got {line!r}"
            grid[(x0, y0)] = ctype
    return grid


def crop_and_remap(grid, x0, x1, y0, y1):
    """Slice [x0:x1) x [y0:y1) and shift coordinates so the crop starts at (0,0)."""
    cropped = {}
    for x in range(x0, x1):
        for y in range(y0, y1):
            ctype = grid.get((x, y))
            if ctype is None:
                continue  # shouldn't happen -- the full piff covers every pixel
            cropped[(x - x0, y - y0)] = ctype
    return cropped


def add_wall_border(cropped, w, h, thickness):
    for x in range(w):
        for y in range(h):
            if x < thickness or x >= w - thickness or y < thickness or y >= h - thickness:
                cropped[(x, y)] = "Wall"
    return cropped


def count_fiber_clusters(cropped):
    """4-connected union-find over Fiber-type pixels (NeighborOrder=1, no scipy)."""
    fiber_coords = [xy for xy, t in cropped.items() if t == "Fiber"]
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


def write_piff(cropped, w, h, path):
    with open(path, "w") as f:
        f.write("Include Clusters\n")
        cell_id = 0
        for x in range(w):
            for y in range(h):
                ctype = cropped[(x, y)]
                f.write(f"{cell_id} {cell_id} {ctype} {x} {x} {y} {y} 0 0\n")
                cell_id += 1


def main():
    w, h = CROP_X1 - CROP_X0, CROP_Y1 - CROP_Y0

    grid = load_piff(INPUT_PIFF)
    cropped = crop_and_remap(grid, CROP_X0, CROP_X1, CROP_Y0, CROP_Y1)
    cropped = add_wall_border(cropped, w, h, WALL_BORDER_PX)

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

    write_piff(cropped, w, h, OUTPUT_PIFF)
    print(f"Wrote {OUTPUT_PIFF} ({w * h} pixels)")


if __name__ == "__main__":
    main()
