"""
select_replant_pixels.py

Given a raster where each pixel value is the dollar change in AC spending
from converting that 1m x 1m pixel from barren land to tree cover (more
negative = more savings = "better"), select the best N pixels to replant
under three different constraints:

  1. UNCONSTRAINED  - the N most negative pixels anywhere on the raster.
  2. RECTANGLE      - the single contiguous rectangular block (h x w >= N)
                       with the lowest total sum. Fast, exact for the
                       rectangle-shape assumption.
  3. BLOB (greedy)  - a single contiguous, arbitrary-shaped region of
                       exactly N pixels, grown greedily from the best
                       seed pixel outward, always adding whichever
                       boundary pixel is most negative next.

Notes on optimality
--------------------
(1) and (2) are exact/optimal for their respective definitions of the
problem. (3) is NOT guaranteed to be the true minimum-sum connected
region of size N -- that problem (minimum-weight connected subgraph of
fixed cardinality) is NP-hard in general. The greedy region-growing
heuristic here is the standard practical approach and tends to work
well when the cost surface is spatially autocorrelated (as spending /
cost surfaces derived from environmental variables usually are).

If you need a provably optimal contiguous blob, you'd formulate it as
a mixed-integer program with flow-based contiguity constraints (see
the "reserve design" / systematic conservation planning literature,
e.g. tools like Marxan use simulated annealing for the same reason
exact solvers don't scale). That's a much heavier lift computationally
and is not implemented here -- happy to sketch it if you need it.

Usage
-----
    python select_replant_pixels.py input.tif --n 1000 --mode blob \
        --out selected.tif --polygon selected.shp

    python select_replant_pixels.py input.tif --n 100 --mode rectangle
    python select_replant_pixels.py input.tif --n 100 --mode unconstrained
"""

import argparse
import heapq
import sys

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_raster(path, band=1):
    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"Could not open {path}")
    band_obj = ds.GetRasterBand(band)
    array = band_obj.ReadAsArray().astype(np.float64)
    nodata = band_obj.GetNoDataValue()
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None
    return array, nodata, geotransform, projection


def write_mask_raster(path, mask, geotransform, projection):
    """Write a 0/1 uint8 raster of selected pixels."""
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = mask.shape
    out_ds = driver.Create(path, cols, rows, 1, gdal.GDT_Byte,
                            options=["COMPRESS=LZW"])
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(mask.astype(np.uint8))
    out_band.SetNoDataValue(0)
    out_band.FlushCache()
    out_ds = None


def polygonize_mask(mask_path, out_vector_path, layer_name="selected"):
    """Turn the selected-pixel raster into polygon(s), e.g. for a shapefile
    you can hand to a planting crew or drop into other GIS software."""
    src_ds = gdal.Open(mask_path)
    src_band = src_ds.GetRasterBand(1)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("ESRI Shapefile driver not available")
    if out_vector_path.endswith(".shp"):
        import os
        if os.path.exists(out_vector_path):
            driver.DeleteDataSource(out_vector_path)
    out_ds = driver.CreateDataSource(out_vector_path)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjection())
    layer = out_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn("value", ogr.OFTInteger))

    gdal.Polygonize(src_band, src_band, layer, 0, [], callback=None)

    # Drop the "not selected" (value=0) polygon(s), keep only value=1
    layer.SetAttributeFilter("value = 0")
    for feat in layer:
        layer.DeleteFeature(feat.GetFID())
    layer.SetAttributeFilter(None)

    out_ds = None
    src_ds = None


# ---------------------------------------------------------------------------
# Selection algorithms
# ---------------------------------------------------------------------------

def select_top_n_unconstrained(array, n, mask=None):
    """The N most negative pixels, no spatial constraint at all."""
    valid = np.isfinite(array)
    if mask is not None:
        valid &= mask
    flat_idx = np.flatnonzero(valid.ravel())
    values = array.ravel()[flat_idx]
    if n >= len(values):
        chosen = flat_idx
    else:
        part = np.argpartition(values, n)[:n]
        chosen = flat_idx[part]
    selected = np.zeros(array.size, dtype=np.uint8)
    selected[chosen] = 1
    return selected.reshape(array.shape), float(array.ravel()[chosen].sum())


def _integral_image(a):
    return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def select_best_rectangle(array, n, mask=None, penalty=1e9):
    """Best contiguous rectangle with area >= n (area is rounded up to the
    nearest h*w >= n via divisor search; the extra cells beyond n are the
    cost of forcing a clean rectangle shape). Invalid/masked-out cells are
    given a large positive penalty so rectangles avoid them where possible.

    Returns (mask, total_sum_over_selected_cells_only, (r0, c0, h, w)).
    Note the returned total_sum excludes any masked-out pixels inside the
    rectangle (they're reported but not counted), while the h*w search
    itself is driven by the penalized sum to steer away from invalid area.
    """
    arr = array.astype(np.float64).copy()
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask
    penalized = arr.copy()
    penalized[~valid] = penalty

    integral = _integral_image(penalized)
    rows, cols = arr.shape
    best = None
    max_h = int(np.ceil(np.sqrt(n * 4))) + 1
    for h in range(1, min(max_h, rows) + 1):
        w = int(np.ceil(n / h))
        if w > cols:
            continue
        sums = (integral[h:, w:] - integral[:-h, w:]
                - integral[h:, :-w] + integral[:-h, :-w])
        idx = np.unravel_index(np.argmin(sums), sums.shape)
        s = sums[idx]
        if best is None or s < best[0]:
            best = (s, idx[0], idx[1], h, w)

    _, r0, c0, h, w = best
    out_mask = np.zeros_like(arr, dtype=np.uint8)
    out_mask[r0:r0 + h, c0:c0 + w] = 1
    real_sum = float(arr[r0:r0 + h, c0:c0 + w][valid[r0:r0 + h, c0:c0 + w]].sum())
    return out_mask, real_sum, (r0, c0, h, w)


def valid_component_sizes(mask, connectivity=4):
    """Diagnostic: label connected components of the valid-pixel mask and
    report their sizes, largest first. Use this to check whether a
    contiguous region of n pixels is even geometrically possible before
    calling select_best_connected_region.
    """
    structure = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                 if connectivity == 4 else np.ones((3, 3)))
    labeled, num = ndimage.label(mask, structure=structure)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    order = np.argsort(sizes)[::-1]
    return labeled, sizes[order], (order + 1)  # sizes and their component ids, largest first


def select_best_connected_region(array, n, mask=None, connectivity=4, seed=None):
    """Greedy best-first region growing: start at the best (most negative)
    valid pixel, repeatedly absorb whichever adjacent boundary pixel is
    most negative, until the region has exactly n pixels.

    This is a heuristic, not an exact solver -- see module docstring.
    """
    arr = array.astype(np.float64)
    rows, cols = arr.shape
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask
    if not valid.any():
        raise ValueError("No valid pixels to select from")
    if valid.sum() < n:
        raise ValueError(f"Only {int(valid.sum())} valid pixels available, need {n}")

    if seed is None:
        # Don't just seed at the global best pixel -- it might sit in a tiny
        # isolated component. Seed inside the best pixel of the smallest
        # component that's still large enough to hold n pixels, since a
        # smaller-but-sufficient component keeps the search space tight and
        # tends to yield a more concentrated (less "stringy") region.
        labeled, sizes, comp_ids = valid_component_sizes(valid, connectivity=connectivity)
        candidates = comp_ids[sizes >= n]
        if len(candidates) == 0:
            top = ", ".join(f"{int(s)}" for s in sizes[:5])
            raise ValueError(
                f"No single connected component of valid pixels has >= {n} "
                f"pixels. Largest components (pixel count): {top}"
                f"{', ...' if len(sizes) > 5 else ''}. "
                "The valid area is too fragmented for one contiguous blob "
                "this size -- see select_best_multi_patch() for a "
                "multi-cluster alternative, or reduce n."
            )
        best_comp = candidates[-1]  # smallest component that's still big enough
        comp_mask = labeled == best_comp
        masked = np.where(comp_mask, arr, np.inf)
        seed = np.unravel_index(np.argmin(masked), arr.shape)

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    in_region = np.zeros((rows, cols), dtype=bool)
    in_heap = np.zeros((rows, cols), dtype=bool)
    heap = []

    def push(r, c):
        if 0 <= r < rows and 0 <= c < cols and valid[r, c] \
                and not in_region[r, c] and not in_heap[r, c]:
            heapq.heappush(heap, (arr[r, c], r, c))
            in_heap[r, c] = True

    push(*seed)
    total = 0.0
    count = 0
    while heap and count < n:
        val, r, c = heapq.heappop(heap)
        if in_region[r, c]:
            continue
        in_region[r, c] = True
        total += val
        count += 1
        for dr, dc in neighbors:
            push(r + dr, c + dc)

    if count < n:
        raise ValueError(
            f"Region growing stalled at {count}/{n} pixels -- the valid "
            "area may be smaller/more fragmented than n, or split by "
            "nodata. Try select_top_n_unconstrained or a smaller n."
        )

    return in_region.astype(np.uint8), total


def select_best_multi_patch(array, n, mask=None, connectivity=4, max_patches=None):
    """When no single connected component is large enough to hold n
    contiguous pixels (or you're fine with several separate patches),
    this fills n pixels using multiple contiguous clusters: it walks
    components largest-first, greedily growing a contiguous region in
    each one, until n total pixels are selected.

    Note this is a coverage-first strategy, not a purely value-optimal
    one: it doesn't compare "one more small great patch elsewhere" against
    "grow the current patch a bit more" -- it fills each chosen component
    as fully as needed before moving to the next. That's usually fine in
    practice (real replanting plans naturally end up as a handful of
    parcels rather than one perfect blob), but flag it if you need
    something more rigorous.

    Returns (mask, total_sum, patch_info) where patch_info is a list of
    (component_id, n_pixels_taken, patch_sum) tuples, in the order filled.
    """
    arr = array.astype(np.float64)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask
    if valid.sum() < n:
        raise ValueError(f"Only {int(valid.sum())} valid pixels available, need {n}")

    labeled, sizes, comp_ids = valid_component_sizes(valid, connectivity=connectivity)

    out_mask = np.zeros(arr.shape, dtype=np.uint8)
    total = 0.0
    remaining = n
    patch_info = []

    for size, comp_id in zip(sizes, comp_ids):
        if remaining <= 0:
            break
        if max_patches is not None and len(patch_info) >= max_patches:
            break
        comp_mask = labeled == comp_id
        take = min(int(size), remaining)
        patch_mask, patch_sum = select_best_connected_region(
            arr, take, mask=comp_mask, connectivity=connectivity)
        out_mask |= patch_mask
        total += patch_sum
        remaining -= take
        patch_info.append((int(comp_id), take, patch_sum))

    if remaining > 0:
        raise ValueError(
            f"Filled {n - remaining}/{n} pixels across all available "
            "components -- ran out of valid area."
        )

    return out_mask, total, patch_info


def _disk_kernel(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2 <= radius ** 2).astype(np.float64)


def select_best_within_radius(array, n, radius, mask=None, top_k_candidates=25):
    """Select the best n pixels such that all of them lie within `radius`
    (in pixels) of a single common center point -- e.g. "everything needs
    to be reachable from one access point/road within R meters." This is
    NOT contiguity: the n selected pixels don't need to touch each other
    or even form a solid blob, just fit inside one circle of that radius.

    This is a looser constraint than the rectangle mode (any shape is
    fine, not just a box), and a different constraint from the blob mode
    (no adjacency/touching required, just proximity to a common center).

    Two-stage approach for efficiency on large rasters:
      1. Use an FFT convolution with a disk-shaped kernel to cheaply
         compute, for every possible center pixel, the sum over all
         valid pixels within radius -- this ranks candidate centers fast.
      2. For the top_k_candidates best centers, do an exact refinement:
         gather all valid pixels actually within radius of that center,
         and pick the true best n among them (which may be fewer than
         the full disk if the disk contains more than n valid pixels).

    Returns (mask, total_sum, center) or raises ValueError if no
    candidate center has >= n valid pixels within radius.
    """
    arr = array.astype(np.float64)
    valid = np.isfinite(arr)
    if mask is not None:
        valid &= mask
    if valid.sum() < n:
        raise ValueError(f"Only {int(valid.sum())} valid pixels available, need {n}")

    kernel = _disk_kernel(radius)
    penalized = np.where(valid, arr, 0.0)
    disk_sum = fftconvolve(penalized, kernel, mode="same")
    disk_count = fftconvolve(valid.astype(np.float64), kernel, mode="same")
    disk_sum = np.where(disk_count >= n - 0.5, disk_sum, np.inf)  # tolerate fft float noise

    if not np.isfinite(disk_sum).any():
        raise ValueError(
            f"No location has >= {n} valid pixels within radius {radius}. "
            "Try a larger radius or smaller n."
        )

    k = min(top_k_candidates, int(np.isfinite(disk_sum).sum()))
    flat_idx = np.argpartition(disk_sum.ravel(), k - 1)[:k]
    centers = [np.unravel_index(i, arr.shape) for i in flat_idx]

    rows, cols = arr.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    best = None
    for (cr, cc) in centers:
        dist2 = (yy - cr) ** 2 + (xx - cc) ** 2
        in_disk = (dist2 <= radius ** 2) & valid
        idx = np.flatnonzero(in_disk.ravel())
        if len(idx) < n:
            continue
        vals = arr.ravel()[idx]
        part = np.argpartition(vals, n)[:n]
        chosen = idx[part]
        total = float(vals[part].sum())
        if best is None or total < best[0]:
            best = (total, cr, cc, chosen)

    if best is None:
        raise ValueError(
            "Candidate centers from the coarse search didn't hold up under "
            "exact refinement -- try increasing top_k_candidates."
        )

    total, cr, cc, chosen = best
    out_mask = np.zeros(arr.size, dtype=np.uint8)
    out_mask[chosen] = 1
    return out_mask.reshape(arr.shape), total, (cr, cc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to input raster (dollar-change per pixel)")
    p.add_argument("--n", type=int, required=True, help="Number of pixels to select")
    p.add_argument("--mode",
                    choices=["unconstrained", "rectangle", "blob", "multi_patch", "radius"],
                    default="blob")
    p.add_argument("--radius", type=float, default=None,
                    help="Radius in pixels, required for --mode radius")
    p.add_argument("--max-patches", type=int, default=None,
                    help="Only used for --mode multi_patch")
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=4,
                    help="Only used for --mode blob")
    p.add_argument("--out", default=None, help="Output raster path (mask, GTiff)")
    p.add_argument("--polygon", default=None,
                    help="Optional output shapefile path for the selected area "
                         "(only meaningful for rectangle/blob modes)")
    args = p.parse_args()

    array, nodata, geotransform, projection = read_raster(args.input, args.band)
    mask = None
    if nodata is not None:
        mask = ~np.isclose(array, nodata)

    if args.mode == "unconstrained":
        selected, total = select_top_n_unconstrained(array, args.n, mask=mask)
        print(f"Selected {int(selected.sum())} pixels, total = {total:,.2f}")
    elif args.mode == "rectangle":
        selected, total, geom = select_best_rectangle(array, args.n, mask=mask)
        r0, c0, h, w = geom
        print(f"Best rectangle: {h}x{w} at row {r0}, col {c0} "
              f"({h*w} cells, valid-pixel total = {total:,.2f})")
    elif args.mode == "blob":
        selected, total = select_best_connected_region(
            array, args.n, mask=mask, connectivity=args.connectivity)
        print(f"Selected {int(selected.sum())} contiguous pixels, total = {total:,.2f}")
    elif args.mode == "multi_patch":
        selected, total, patch_info = select_best_multi_patch(
            array, args.n, mask=mask, connectivity=args.connectivity,
            max_patches=args.max_patches)
        print(f"Selected {int(selected.sum())} pixels across {len(patch_info)} "
              f"patch(es), total = {total:,.2f}")
        for comp_id, count, patch_sum in patch_info:
            print(f"  component {comp_id}: {count} px, sum = {patch_sum:,.2f}")
    else:  # radius
        if args.radius is None:
            p.error("--mode radius requires --radius")
        selected, total, center = select_best_within_radius(
            array, args.n, args.radius, mask=mask)
        print(f"Selected {int(selected.sum())} pixels within radius "
              f"{args.radius} of center row={center[0]}, col={center[1]}, "
              f"total = {total:,.2f}")

    if args.out:
        write_mask_raster(args.out, selected, geotransform, projection)
        print(f"Wrote mask raster to {args.out}")
        if args.polygon:
            polygonize_mask(args.out, args.polygon)
            print(f"Wrote polygon to {args.polygon}")
    elif args.polygon:
        print("Note: --polygon requires --out (polygonize reads from the "
              "written mask raster); skipping.", file=sys.stderr)


if __name__ == "__main__":
    main()
