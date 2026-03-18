#!/usr/bin/env python3
"""Fetch paired S1 GRD + NISAR GSLC amplitude images for mutual dates.

For each date where both Sentinel-1 and NISAR acquired over the AOI,
fetches co-registered 100 m amplitude arrays (dB scale) and caches them
as .npy files with a comprehensive metadata manifest.

NISAR:  Remote GSLC crop via earthaccess + h5py, multi-looked to 100 m.
S1:     GRD from Planetary Computer STAC via odc-stac, loaded at 100 m.

Usage:
    python fetch_paired.py jakobshavn
    python fetch_paired.py jakobshavn --dates 2025-12-15
    python fetch_paired.py jakobshavn --min-nisar-coverage 25
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

import earthaccess
import h5py
import numpy as np
import odc.stac
import planetary_computer  # noqa: F401  (needed by odc.stac signing)
from odc.geo.geobox import GeoBox
from pyproj import Transformer

from calving_sites import SITES
from stac_utils import get_catalog, search_s1_grd, get_copol_band

# NISAR GSLC layer path
GSLC_LAYER = "/science/LSAR/GSLC/grids/frequencyA/{pol}"

# Minimum per-frame coverage to keep a NISAR track/frame
MIN_FRAME_COV = 20  # percent


# ---------------------------------------------------------------------------
# Site geometry helpers (shared with calving_sites.py / nisar_fetch.py)
# ---------------------------------------------------------------------------

def _site_proj_bbox(site_key):
    cfg = SITES[site_key]
    tr = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True)
    cx, cy = tr.transform(cfg["center_lon"], cfg["center_lat"])
    hw = cfg["half_width_km"] * 1000
    hh = cfg.get("half_height_km", cfg["half_width_km"]) * 1000
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _site_geobox(site_key):
    cfg = SITES[site_key]
    tr = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True)
    cx, cy = tr.transform(cfg["center_lon"], cfg["center_lat"])
    hw = cfg["half_width_km"] * 1000
    hh = cfg.get("half_height_km", cfg["half_width_km"]) * 1000
    bbox = (cx - hw, cy - hh, cx + hw, cy + hh)
    return GeoBox.from_bbox(bbox=bbox, crs=cfg["crs"],
                            resolution=cfg["resolution"]), bbox


def _search_bbox_lonlat(site_key):
    cfg = SITES[site_key]
    pad_lat = cfg["half_width_km"] / 111.0 * 1.5
    pad_lon = pad_lat / max(np.cos(np.deg2rad(cfg["center_lat"])), 0.1)
    return [cfg["center_lon"] - pad_lon, cfg["center_lat"] - pad_lat,
            cfg["center_lon"] + pad_lon, cfg["center_lat"] + pad_lat]


# ---------------------------------------------------------------------------
# Identify mutual dates
# ---------------------------------------------------------------------------

def find_mutual_dates(site_key, min_nisar_cov=MIN_FRAME_COV):
    """Return sorted list of dicts for dates with both NISAR and S1 data.

    Each dict: {date, nisar_path, nisar_dir, nisar_cov, nisar_granules,
                s1_n, s1_orbits, s1_dir}
    """
    cat_path = os.path.join("output", "nisar_s1", "catalogs",
                            f"{site_key}_nisar_catalog.json")
    cov_path = os.path.join("output", "nisar_s1", "catalogs",
                            f"{site_key}_coverage_report.json")

    with open(cat_path) as f:
        cat = json.load(f)
    with open(cov_path) as f:
        cov_report = json.load(f)

    # --- S1 dates from coverage report ---
    s1_dates = {}
    for d in cov_report["days"]:
        if d["s1"] == "\u2713":  # ✓
            s1_dates[d["date"]] = {
                "s1_n": d["s1_n"], "s1_orbits": d["s1_orbits"],
                "s1_dir": d["s1_dir"]}

    # --- NISAR: keep only frames ≥ min coverage ---
    kept_pf = set()
    for g in cat["granules"]:
        if (str(g["range_bw_mhz"]) == "77"
                and g["polarization"] == "HH"
                and g["coverage_pct"] >= min_nisar_cov):
            kept_pf.add((g["path"], g["frame"]))

    # Group kept granules by date → track
    day_tracks = defaultdict(lambda: defaultdict(list))
    for g in cat["granules"]:
        if (str(g["range_bw_mhz"]) != "77"
                or g["polarization"] != "HH"):
            continue
        if (g["path"], g["frame"]) not in kept_pf:
            continue
        day_tracks[g["date"]][g["path"]].append(g)

    # For each date, pick the best track (highest total coverage)
    nisar_sel = {}
    for date, tracks in day_tracks.items():
        best_path = max(tracks, key=lambda p: sum(
            g["coverage_pct"] for g in tracks[p]))
        granules = tracks[best_path]
        nisar_sel[date] = {
            "path": best_path,
            "dir": granules[0]["direction"],
            "cov": sum(g["coverage_pct"] for g in granules),
            "granules": granules,
        }

    # --- Mutual dates ---
    mutual = []
    for date in sorted(set(nisar_sel) & set(s1_dates)):
        ni = nisar_sel[date]
        si = s1_dates[date]
        mutual.append({
            "date": date,
            "nisar_path": ni["path"],
            "nisar_dir": ni["dir"],
            "nisar_cov": ni["cov"],
            "nisar_granules": ni["granules"],
            **si,
        })

    return mutual


# ---------------------------------------------------------------------------
# NISAR fetch (multi-looked GSLC → amplitude)
# ---------------------------------------------------------------------------

def _fetch_nisar_crop(h5_url, proj_bbox, site_crs, pol="HH", out_res=100):
    """Fetch + multilook a NISAR GSLC crop. Returns (amp_array, transform)
    or (None, None) on failure."""
    xmin, ymin, xmax, ymax = proj_bbox
    layer = GSLC_LAYER.format(pol=pol)

    fh = earthaccess.open([h5_url])[0]
    h5f = h5py.File(fh, "r")
    ds = h5f[layer]

    grp = h5f["/science/LSAR/GSLC/grids/frequencyA"]
    x_coords = grp["xCoordinates"][:]
    y_coords = grp["yCoordinates"][:]
    origin_x = float(x_coords[0])
    origin_y = float(y_coords[0])
    xres_native = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 2.5
    yres_native = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else -5.0
    abs_xres = abs(xres_native)
    abs_yres = abs(yres_native)

    col_start = max(0, int((xmin - origin_x) / abs_xres))
    col_end = min(ds.shape[1], int((xmax - origin_x) / abs_xres))
    if yres_native < 0:
        row_start = max(0, int((origin_y - ymax) / abs_yres))
        row_end = min(ds.shape[0], int((origin_y - ymin) / abs_yres))
    else:
        row_start = max(0, int((ymin - origin_y) / abs_yres))
        row_end = min(ds.shape[0], int((ymax - origin_y) / abs_yres))

    if col_end <= col_start or row_end <= row_start:
        h5f.close(); fh.close()
        return None, None

    step_x = max(1, int(out_res / abs_xres))
    step_y = max(1, int(out_res / abs_yres))
    n_rows = row_end - row_start
    n_cols = col_end - col_start
    n_rows_trim = (n_rows // step_y) * step_y
    n_cols_trim = (n_cols // step_x) * step_x
    out_h = n_rows_trim // step_y
    out_w = n_cols_trim // step_x

    if out_h == 0 or out_w == 0:
        h5f.close(); fh.close()
        return None, None

    chunk_h = ds.chunks[0] if ds.chunks else 512
    n_strips = (n_rows_trim + chunk_h - 1) // chunk_h

    ml_intensity = np.zeros((out_h, out_w), dtype=np.float64)
    rows_read = 0
    for i in range(n_strips):
        r0 = row_start + i * chunk_h
        r1 = min(r0 + chunk_h, row_start + n_rows_trim)
        strip = ds[r0:r1, col_start:col_start + n_cols_trim]
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
    h5f.close()
    fh.close()

    from rasterio.transform import Affine
    crop_ox = origin_x + col_start * abs_xres
    crop_oy = (origin_y - row_start * abs_yres if yres_native < 0
               else origin_y + row_start * abs_yres)
    transform = Affine(out_res, 0, crop_ox, 0, -out_res, crop_oy)

    return amp, transform


# ---------------------------------------------------------------------------
# S1 GRD fetch (odc-stac → dB amplitude)
# ---------------------------------------------------------------------------

def _dn_to_db(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 10.0 * np.log10(
            np.where(arr > 0, arr.astype(np.float64), np.nan))


def _aoi_polygon_lonlat(site_key):
    """Return AOI as shapely polygon in lon/lat (EPSG:4326)."""
    from shapely.geometry import box
    cfg = SITES[site_key]
    # Convert from projected to geographic for intersection with STAC items
    half_w_deg = cfg["half_width_km"] / (111.32 * np.cos(np.deg2rad(cfg["center_lat"])))
    half_h_deg = cfg.get("half_height_km", cfg["half_width_km"]) / 111.32
    return box(cfg["center_lon"] - half_w_deg, cfg["center_lat"] - half_h_deg,
               cfg["center_lon"] + half_w_deg, cfg["center_lat"] + half_h_deg)


def _filter_s1_items_by_aoi(items, aoi_polygon, min_coverage_pct=1.0):
    """Filter S1 items to only those with ≥min_coverage_pct of AOI.
    
    Returns filtered items list with additional coverage metadata.
    """
    from shapely.geometry import shape
    filtered = []
    aoi_area = aoi_polygon.area
    for it in items:
        geom = shape(it.geometry)
        intersection = geom.intersection(aoi_polygon)
        coverage_pct = (intersection.area / aoi_area) * 100 if aoi_area > 0 else 0
        if coverage_pct >= min_coverage_pct:
            it._aoi_coverage_pct = coverage_pct  # store for later use
            filtered.append(it)
    return filtered


def _select_single_geometry_s1(items, nisar_utc_hour=None):
    """Select a single geometry (track/direction) from S1 items.
    
    If multiple tracks have coverage, pick the one closest in time to NISAR.
    If nisar_utc_hour is None, pick the track with most items.
    """
    if not items:
        return items
    
    # Group by (orbit, direction)
    from collections import defaultdict
    tracks = defaultdict(list)
    for it in items:
        orbit = it.properties.get("sat:relative_orbit", 0)
        direction = it.properties.get("sat:orbit_state", "")
        tracks[(orbit, direction)].append(it)
    
    if len(tracks) <= 1:
        return items  # Already single geometry
    
    # Multiple geometries - need to select one
    # Get representative UTC hour for each track
    track_info = []
    for (orbit, direction), track_items in tracks.items():
        # Get UTC hour from first item's datetime
        dt_str = track_items[0].properties.get("datetime", "")
        utc_hour = 12.0  # default
        if "T" in dt_str:
            try:
                time_part = dt_str.split("T")[1][:5]
                utc_hour = int(time_part[:2]) + int(time_part[3:5]) / 60.0
            except (ValueError, IndexError):
                pass
        track_info.append({
            "key": (orbit, direction),
            "items": track_items,
            "utc_hour": utc_hour,
            "n_items": len(track_items),
        })
    
    # Select based on closeness to NISAR time
    if nisar_utc_hour is not None:
        # Pick track closest in time (accounting for day wrap)
        def time_diff(h):
            d = abs(h - nisar_utc_hour)
            return min(d, 24 - d)  # handle wrap around midnight
        best = min(track_info, key=lambda t: time_diff(t["utc_hour"]))
    else:
        # Pick track with most items
        best = max(track_info, key=lambda t: t["n_items"])
    
    return best["items"]


def _fetch_s1_day(site_key, date_str, geobox, catalog, nisar_utc_hour=None):
    """Fetch S1 GRD for a single day, return (arr_db, metadata) or (None, {}).
    
    Args:
        nisar_utc_hour: If provided, selects S1 track closest in time to NISAR.
                        This ensures single-geometry output (no ASC/DESC mixing).
    """
    cfg = SITES[site_key]
    bbox_ll = _search_bbox_lonlat(site_key)
    dt_range = f"{date_str}/{date_str}"
    mode = cfg.get("s1_mode")

    items = search_s1_grd(catalog, bbox_ll, dt_range,
                          max_items=50, mode=mode)
    if not items:
        return None, {}

    # Pick co-pol band from first item
    band = get_copol_band(items[0])
    if not band:
        return None, {}

    # Filter to items with that band
    items = [it for it in items if band in it.assets]
    if not items:
        return None, {}

    # Filter to items that actually cover the AOI (not just search bbox)
    aoi_polygon = _aoi_polygon_lonlat(site_key)
    items = _filter_s1_items_by_aoi(items, aoi_polygon, min_coverage_pct=1.0)
    if not items:
        return None, {}
    
    # Select single geometry (avoid mixing ASC/DESC)
    items = _select_single_geometry_s1(items, nisar_utc_hour=nisar_utc_hour)
    if not items:
        return None, {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = odc.stac.load(items, bands=[band], geobox=geobox,
                           groupby="solar_day", resampling="nearest")
        arr = ds[band].median(dim="time").values.astype(np.float64)
        del ds
        gc.collect()

    arr_db = _dn_to_db(arr)
    valid_frac = np.isfinite(arr_db).sum() / arr_db.size

    # Metadata (now reflects only items actually used)
    platforms = sorted({it.properties.get("platform", "").upper()
                        .replace("SENTINEL-1", "S1") for it in items})
    orbits = sorted({it.properties.get("sat:relative_orbit", "")
                     for it in items})
    directions = sorted({it.properties.get("sat:orbit_state", "")
                         for it in items})
    # Extract UTC times from items
    datetimes = []
    for it in items:
        dt_str = it.properties.get("datetime", "")
        if dt_str:
            datetimes.append(dt_str)
    datetimes = sorted(set(datetimes))

    meta = {
        "band": band,
        "mode": mode,
        "n_items": len(items),
        "platforms": platforms,
        "relative_orbits": orbits,
        "directions": directions,
        "datetimes": datetimes,
        "valid_fraction": round(float(valid_frac), 4),
    }
    return arr_db.astype(np.float32), meta


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch_paired(site_key, dates=None, min_nisar_cov=MIN_FRAME_COV):
    """Fetch all mutual S1+NISAR dates for a site."""
    cfg = SITES[site_key]
    label = cfg["label"]

    mutual = find_mutual_dates(site_key, min_nisar_cov=min_nisar_cov)
    if dates:
        mutual = [m for m in mutual if m["date"] in dates]

    print(f"\n{'='*60}")
    print(f"[{label}] Fetching paired S1 + NISAR data")
    print(f"  Mutual dates: {len(mutual)}")
    print(f"  NISAR min coverage: {min_nisar_cov}%")
    print(f"  Output resolution: {cfg['resolution']} m")
    print(f"{'='*60}\n")

    if not mutual:
        print("  No mutual dates found.")
        return

    cache_dir = os.path.join("output", "nisar_s1", "cache", site_key)
    os.makedirs(cache_dir, exist_ok=True)

    # Load or create manifest
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "site": site_key,
            "label": label,
            "crs": cfg["crs"],
            "resolution_m": cfg["resolution"],
            "aoi_bbox_proj": list(_site_proj_bbox(site_key)),
            "nisar_mode": "77 MHz HH",
            "nisar_min_frame_coverage": min_nisar_cov,
            "s1_mode": cfg.get("s1_mode", "IW"),
            "pairs": [],
        }

    # Index existing pairs by date
    existing = {p["date"]: p for p in manifest["pairs"]}

    proj_bbox = _site_proj_bbox(site_key)
    site_crs = cfg["crs"]
    geobox, _ = _site_geobox(site_key)
    stac_cat = get_catalog()

    # Authenticate NISAR
    auth = earthaccess.login()
    if not auth.authenticated:
        print("ERROR: Earthdata Login failed. Check ~/.netrc")
        sys.exit(1)

    n_ok = 0
    n_skip = 0

    for mi, m in enumerate(mutual, 1):
        date = m["date"]
        nisar_path = m["nisar_path"]
        nisar_dir = m["nisar_dir"]
        nisar_cov = m["nisar_cov"]

        print(f"[{mi:>2}/{len(mutual)}] {date}  "
              f"NISAR P{nisar_path} {nisar_dir[:3]} {nisar_cov:.0f}%  |  "
              f"S1 ×{m['s1_n']}")

        nisar_file = f"{site_key}_{date}_nisar.npy"
        s1_file = f"{site_key}_{date}_s1.npy"

        nisar_path_full = os.path.join(cache_dir, nisar_file)
        s1_path_full = os.path.join(cache_dir, s1_file)

        pair_meta = existing.get(date, {})
        need_nisar = not os.path.exists(nisar_path_full)
        need_s1 = not os.path.exists(s1_path_full)

        if not need_nisar and not need_s1:
            print(f"  Both cached — skip")
            n_ok += 1
            continue

        # --- Fetch NISAR ---
        nisar_meta = pair_meta.get("nisar", {})
        if need_nisar:
            print(f"  NISAR: fetching {len(m['nisar_granules'])} frame(s)...")
            t0 = time.time()
            # Fetch each frame and mosaic
            amp_parts = []
            for g in m["nisar_granules"]:
                url = g.get("url")
                if not url:
                    continue
                try:
                    amp, transform = _fetch_nisar_crop(
                        url, proj_bbox, site_crs, pol="HH",
                        out_res=cfg["resolution"])
                    if amp is not None:
                        amp_parts.append(amp)
                except Exception as e:
                    print(f"    ✗ Frame {g['frame']}: {e}")

            if not amp_parts:
                print(f"  NISAR: no data — skip date")
                n_skip += 1
                continue

            # If multiple frames, combine (max of non-zero for overlaps)
            if len(amp_parts) == 1:
                nisar_amp = amp_parts[0]
            else:
                # Pad to common shape
                max_h = max(a.shape[0] for a in amp_parts)
                max_w = max(a.shape[1] for a in amp_parts)
                combined = np.zeros((max_h, max_w), dtype=np.float32)
                for a in amp_parts:
                    h, w = a.shape
                    mask = a > 0
                    combined[:h, :w] = np.where(
                        mask, np.maximum(combined[:h, :w], a), combined[:h, :w])
                nisar_amp = combined

            # Convert to dB
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nisar_db = (20.0 * np.log10(
                    np.where(nisar_amp > 0, nisar_amp, np.nan)
                )).astype(np.float32)

            np.save(nisar_path_full, nisar_db)
            elapsed = time.time() - t0
            valid_frac = np.isfinite(nisar_db).sum() / nisar_db.size
            print(f"    ✓ {nisar_db.shape[1]}×{nisar_db.shape[0]} px, "
                  f"{valid_frac:.0%} valid, {elapsed:.0f}s")

            nisar_meta = {
                "file": nisar_file,
                "shape": list(nisar_db.shape),
                "path": nisar_path,
                "direction": nisar_dir,
                "coverage_pct": round(nisar_cov, 1),
                "n_frames": len(m["nisar_granules"]),
                "frames": [g["frame"] for g in m["nisar_granules"]],
                "valid_fraction": round(float(valid_frac), 4),
                "urls": [g["url"] for g in m["nisar_granules"] if g.get("url")],
                "fetch_time_s": round(elapsed, 1),
            }

        # --- Fetch S1 ---
        s1_meta = pair_meta.get("s1", {})
        if need_s1:
            print(f"  S1: fetching for {date}...")
            t0 = time.time()
            
            # Extract NISAR UTC hour to select S1 track closest in time
            nisar_utc_hour = None
            nisar_url = nisar_meta.get("urls", [""])[0] if nisar_meta else ""
            import re
            m_time = re.search(r'_(\d{8}T\d{6})_', nisar_url)
            if m_time:
                ts = m_time.group(1)  # e.g., 20251028T235201
                nisar_utc_hour = int(ts[9:11]) + int(ts[11:13]) / 60.0
            
            try:
                s1_db, s1_meta_new = _fetch_s1_day(
                    site_key, date, geobox, stac_cat, nisar_utc_hour=nisar_utc_hour)
            except Exception as e:
                print(f"    ✗ S1 error: {e}")
                s1_db = None
                s1_meta_new = {}

            if s1_db is None:
                print(f"  S1: no data — skip date")
                # Remove NISAR file if just created (keep pairs complete)
                if os.path.exists(nisar_path_full) and need_nisar:
                    os.remove(nisar_path_full)
                n_skip += 1
                continue

            np.save(s1_path_full, s1_db)
            elapsed = time.time() - t0
            print(f"    ✓ {s1_db.shape[1]}×{s1_db.shape[0]} px, "
                  f"{s1_meta_new.get('valid_fraction', 0):.0%} valid, "
                  f"{elapsed:.0f}s")

            s1_meta = {
                "file": s1_file,
                "shape": list(s1_db.shape),
                "fetch_time_s": round(elapsed, 1),
                **s1_meta_new,
            }

        # --- Update manifest ---
        pair_entry = {
            "date": date,
            "nisar": nisar_meta,
            "s1": s1_meta,
        }
        existing[date] = pair_entry
        n_ok += 1

        # Save manifest after each pair (incremental)
        manifest["pairs"] = [existing[d] for d in sorted(existing)]
        manifest["last_updated"] = datetime.now(tz=None).astimezone().isoformat()
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\n[{label}] Done: {n_ok} pairs OK, {n_skip} skipped")
    print(f"  Cache: {cache_dir}/")
    print(f"  Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch paired S1 + NISAR amplitude for mutual dates")
    parser.add_argument("sites", nargs="+", help="Site key(s) or 'all'")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="Only fetch specific dates (YYYY-MM-DD)")
    parser.add_argument("--min-nisar-coverage", type=float,
                        default=MIN_FRAME_COV,
                        help=f"Min NISAR frame coverage %% (default: {MIN_FRAME_COV})")
    args = parser.parse_args()

    sites = list(SITES.keys()) if "all" in args.sites else args.sites
    date_set = set(args.dates) if args.dates else None

    for s in sites:
        if s not in SITES:
            print(f"Unknown site '{s}'. Available: {', '.join(SITES.keys())}")
            continue
        fetch_paired(s, dates=date_set,
                     min_nisar_cov=args.min_nisar_coverage)


if __name__ == "__main__":
    main()
