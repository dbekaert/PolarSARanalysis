#!/usr/bin/env python3
"""Fetch cropped NISAR GSLC amplitude images for calving sites.

Reads the catalog JSON produced by nisar_catalog.py, then for each
auto-selected date opens the GSLC .h5 file *remotely* via earthaccess
+ h5py — only the pixels inside the site AOI are transferred.  The
complex SLC values are multi-looked (intensity averaged) and converted
to amplitude, following the same processing as S1 GRD:
  1. Compute intensity = |z|²
  2. Average over N×M pixel window (multi-look)
  3. Store sqrt(mean intensity) as amplitude GeoTIFF crop.

No full NISAR frame is ever downloaded.

Prerequisites:
    pip install earthaccess h5py rasterio numpy pyproj
    A valid ~/.netrc with Earthdata Login credentials.

Usage:
    python nisar_fetch.py jakobshavn
    python nisar_fetch.py jakobshavn --dates 2025-12-15 2025-12-27
    python nisar_fetch.py jakobshavn --min-coverage 50
    python nisar_fetch.py all
"""

import argparse
import json
import os
import sys
import time

import earthaccess
import h5py
import numpy as np
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer

from calving_sites import SITES

# Layer path inside NISAR GSLC .h5 files (CF-compliant netCDF structure)
GSLC_LAYER = "/science/LSAR/GSLC/grids/frequencyA/{pol}"


def _site_proj_bbox(site_key):
    """Return the projected bounding box (xmin, ymin, xmax, ymax) in the
    site's native CRS, matching the AOI used for S1 calving rendering."""
    cfg = SITES[site_key]
    transformer = Transformer.from_crs(
        "EPSG:4326", cfg["crs"], always_xy=True)
    cx, cy = transformer.transform(cfg["center_lon"], cfg["center_lat"])
    hw = cfg["half_width_km"] * 1000
    hh = cfg.get("half_height_km", cfg["half_width_km"]) * 1000
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _fetch_crop(h5_url, proj_bbox, site_crs, out_path,
                pol="HH", out_res=100):
    """Open a GSLC .h5 file remotely, crop to *proj_bbox*, compute
    amplitude, and save as a GeoTIFF.

    Uses earthaccess + h5py to stream only the HDF5 chunks that
    overlap the AOI.  Reads contiguous strips aligned to the HDF5
    chunk grid and downsamples locally.

    Parameters
    ----------
    h5_url : str
        HTTPS URL to the .h5 file (ASF DAAC).
    proj_bbox : tuple
        (xmin, ymin, xmax, ymax) in the site's native CRS.
    site_crs : str
        e.g. "EPSG:3413"
    out_path : str
        Output GeoTIFF path.
    pol : str
        Polarisation layer to read (default "HH").
    out_res : float
        Target output resolution in metres (default 100 m).
    """
    xmin, ymin, xmax, ymax = proj_bbox
    layer = GSLC_LAYER.format(pol=pol)

    t0 = time.time()
    fh = earthaccess.open([h5_url])[0]
    h5f = h5py.File(fh, "r")
    ds = h5f[layer]

    # Read geotransform from the rasterio-probed metadata.
    # NISAR GSLC grids use a regular grid in the projected CRS.
    # We get the transform from the dataset dimensions and
    # the x/y coordinate arrays.
    try:
        grp = h5f["/science/LSAR/GSLC/grids/frequencyA"]
        x_coords = grp["xCoordinates"][:]
        y_coords = grp["yCoordinates"][:]
        origin_x = float(x_coords[0])
        origin_y = float(y_coords[0])
        xres_native = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 2.5
        yres_native = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else -5.0
    except (KeyError, IndexError):
        # Fallback values matching Greenland frames (EPSG:3413)
        origin_x, origin_y = -340560.0, -2106720.0
        xres_native, yres_native = 2.5, -5.0

    abs_xres = abs(xres_native)
    abs_yres = abs(yres_native)

    # Pixel offsets for the AOI
    col_start = max(0, int((xmin - origin_x) / abs_xres))
    col_end = min(ds.shape[1], int((xmax - origin_x) / abs_xres))
    if yres_native < 0:  # y decreasing (north-up)
        row_start = max(0, int((origin_y - ymax) / abs_yres))
        row_end = min(ds.shape[0], int((origin_y - ymin) / abs_yres))
    else:  # y increasing
        row_start = max(0, int((ymin - origin_y) / abs_yres))
        row_end = min(ds.shape[0], int((ymax - origin_y) / abs_yres))

    if col_end <= col_start or row_end <= row_start:
        print(f"    ⚠ AOI does not overlap this granule — skipping")
        h5f.close()
        fh.close()
        return False

    # Multi-look window (matches S1 GRD: average intensity then sqrt)
    step_x = max(1, int(out_res / abs_xres))
    step_y = max(1, int(out_res / abs_yres))
    n_looks = step_x * step_y

    # Trim AOI to exact multiples of multi-look window
    n_rows = row_end - row_start
    n_cols = col_end - col_start
    n_rows_trim = (n_rows // step_y) * step_y
    n_cols_trim = (n_cols // step_x) * step_x
    out_h = n_rows_trim // step_y
    out_w = n_cols_trim // step_x

    if out_h == 0 or out_w == 0:
        print(f"    ⚠ AOI too small for multi-look window — skipping")
        h5f.close()
        fh.close()
        return False

    # Read in strips aligned to HDF5 chunks for efficient I/O
    chunk_h = ds.chunks[0] if ds.chunks else 512
    n_strips = (n_rows_trim + chunk_h - 1) // chunk_h

    t1 = time.time()
    ml_intensity = np.zeros((out_h, out_w), dtype=np.float64)
    rows_read = 0
    for i in range(n_strips):
        r0 = row_start + i * chunk_h
        r1 = min(r0 + chunk_h, row_start + n_rows_trim)
        strip = ds[r0:r1, col_start:col_start + n_cols_trim]
        # Multi-look: accumulate mean intensity row by row
        for r in range(strip.shape[0]):
            out_row = rows_read // step_y
            if out_row >= out_h:
                break
            intensity = np.abs(strip[r].astype(np.complex64)) ** 2
            ml_intensity[out_row] += (
                intensity.reshape(out_w, step_x).mean(axis=1) / step_y
            )
            rows_read += 1

    amp = np.sqrt(ml_intensity).astype(np.float32)
    t2 = time.time()

    h5f.close()
    fh.close()

    if amp.size == 0:
        print(f"    ⚠ Empty crop — skipping")
        return False

    # Build output geotransform
    crop_origin_x = origin_x + col_start * abs_xres
    if yres_native < 0:
        crop_origin_y = origin_y - row_start * abs_yres
    else:
        crop_origin_y = origin_y + row_start * abs_yres
    out_transform = Affine(out_res, 0, crop_origin_x,
                           0, -out_res, crop_origin_y)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=amp.shape[0], width=amp.shape[1],
        count=1, dtype="float32",
        crs=site_crs,
        transform=out_transform,
        compress="deflate",
    ) as dst:
        dst.write(amp, 1)

    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"    ✓ {amp.shape[1]}×{amp.shape[0]} px, "
          f"{sz_mb:.1f} MB, {t2 - t0:.0f}s "
          f"(open {t1 - t0:.0f}s + read {t2 - t1:.0f}s)")
    return True


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def fetch_site(site_key, dates=None, min_coverage=0, resolution=100):
    """Fetch cropped GSLC amplitude images for a calving site.

    Parameters
    ----------
    site_key : str
        Site name (must exist in SITES dict and have a catalog JSON).
    dates : list[str] or None
        If given, only fetch these dates. Otherwise fetch all selected.
    min_coverage : float
        Skip dates with coverage below this percentage.
    resolution : float
        Output resolution in metres (default 100 m).
    """
    catalog_path = os.path.join("output", "nisar_s1", "catalogs",
                                f"{site_key}_nisar_catalog.json")
    if not os.path.exists(catalog_path):
        print(f"[{site_key}] No catalog found at {catalog_path}")
        print(f"  Run: python nisar_catalog.py {site_key}")
        return

    with open(catalog_path) as f:
        catalog = json.load(f)

    chosen = catalog.get("chosen_mode", {})
    chosen_pol = chosen.get("polarization", "HH")
    selected = catalog.get("selected_tracks", {})
    granules = catalog.get("granules", [])
    day_summaries = catalog.get("day_summaries", [])
    label = catalog.get("label", site_key)
    site_crs = SITES[site_key]["crs"]
    proj_bbox = _site_proj_bbox(site_key)

    # Filter to requested dates
    if dates:
        selected = {d: v for d, v in selected.items() if d in dates}

    print(f"\n[{label}] Fetching GSLC amplitude crops")
    print(f"  Mode: {chosen.get('bw_mhz', '?')} MHz {chosen_pol}")
    print(f"  CRS: {site_crs}")
    print(f"  AOI bbox: {proj_bbox}")
    print(f"  Output resolution: {resolution} m")
    print(f"  Dates to fetch: {len(selected)}")
    if min_coverage > 0:
        print(f"  Min coverage: {min_coverage}%")
    print()

    out_dir = os.path.join("output", "nisar_s1", "cache", site_key)
    os.makedirs(out_dir, exist_ok=True)

    n_ok = 0
    n_skip = 0
    for date_key in sorted(selected):
        sel = selected[date_key]
        sel_path = sel["path"]
        sel_dir = sel["direction"]

        # Find the day summary to check coverage
        ds = next((x for x in day_summaries
                   if x["date"] == date_key
                   and x["path"] == sel_path
                   and x["direction"] == sel_dir), None)
        if not ds:
            continue
        cov = ds.get("coverage_pct", 0)
        if cov < min_coverage:
            print(f"  {date_key}  path {sel_path}  cov {cov:.0f}%  "
                  f"— skip (< {min_coverage}%)")
            n_skip += 1
            continue

        # Find matching granules (same date + path + direction)
        matching_granules = [
            g for g in granules
            if g["date"] == date_key
            and g["path"] == sel_path
            and g["direction"] == sel_dir
            and g.get("url")
        ]

        if not matching_granules:
            print(f"  {date_key}  path {sel_path}  — no granule URLs found")
            n_skip += 1
            continue

        # Process each frame (they can be mosaicked later)
        for g in matching_granules:
            frame = g["frame"]
            out_name = (f"{site_key}_{date_key}_P{sel_path}_"
                        f"F{frame}_{chosen_pol}_amp.tif")
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path):
                print(f"  {date_key}  P{sel_path} F{frame}  — exists, skip")
                n_ok += 1
                continue

            print(f"  {date_key}  P{sel_path} F{frame}  "
                  f"cov {cov:.0f}%  → {out_name}")

            try:
                ok = _fetch_crop(g["url"], proj_bbox, site_crs,
                                 out_path, pol=chosen_pol,
                                 out_res=resolution)
                if ok:
                    n_ok += 1
                else:
                    n_skip += 1
            except Exception as e:
                print(f"    ✗ Error: {e}")
                n_skip += 1

    print(f"\n[{label}] Done: {n_ok} crops saved, {n_skip} skipped")
    print(f"  Output: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch cropped NISAR GSLC amplitude for calving sites")
    parser.add_argument(
        "sites", nargs="+",
        help="Site key(s) or 'all'")
    parser.add_argument(
        "--dates", nargs="*", default=None,
        help="Only fetch specific dates (YYYY-MM-DD)")
    parser.add_argument(
        "--min-coverage", type=float, default=0,
        help="Skip dates with coverage below this %% (default: 0)")
    parser.add_argument(
        "--resolution", type=float, default=100,
        help="Output resolution in metres (default: 100)")
    args = parser.parse_args()

    auth = earthaccess.login()
    if not auth.authenticated:
        print("ERROR: Earthdata Login failed. Check ~/.netrc")
        sys.exit(1)
    print(f"Earthdata Login: authenticated")

    sites = list(SITES.keys()) if "all" in args.sites else args.sites
    for s in sites:
        if s not in SITES:
            print(f"Unknown site '{s}'. Available: {', '.join(SITES.keys())}")
            continue
        fetch_site(s, dates=args.dates,
                   min_coverage=args.min_coverage,
                   resolution=args.resolution)


if __name__ == "__main__":
    main()
