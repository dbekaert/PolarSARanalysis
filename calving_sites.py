#!/usr/bin/env python3
"""
Calving-site time-lapse: monthly Sentinel-1 snapshots of major
glacier calving fronts around the world.

Strategy for fast iteration:
  - Monthly cadence, but only pull **one day** of data per month
    (searching within a ±3 day window around the 1st to find data).
  - Moderate resolution (200 m) for crisp glacier detail.
  - Each site has its own local-area bounding box and projection.

Usage:
    # Fetch + render all sites
    python calving_sites.py

    # Fetch only
    python calving_sites.py --fetch

    # Render only (from cache)
    python calving_sites.py --render

    # Single site
    python calving_sites.py --site jakobshavn

    # Custom date range
    python calving_sites.py --start_date 2020-01-01 --end_date 2025-12-31
"""

import argparse
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import odc.stac
import planetary_computer as pc
from odc.geo.geobox import GeoBox
from PIL import Image
from pyproj import Transformer

from scipy.ndimage import uniform_filter

from stac_utils import get_catalog, search_s1_grd


# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------
# Each site: centre (lat, lon), half-width in km, CRS, S1 mode, label.
# Arctic sites use EPSG:3413 (NSIDC North Polar Stereo).
# Antarctic sites use EPSG:3031 (Antarctic Polar Stereo).
# IW mode preferred for higher resolution (~10 m native, VV pol).
# Resolution is 100 m for glacier-scale detail.

SITES = {
    # --- Greenland ---
    "jakobshavn": {
        "label": "Jakobshavn Glacier",
        "center_lat": 69.17, "center_lon": -50.2,
        "half_width_km": 50,
        "half_height_km": 30,
        "crs": "EPSG:3413",
        "resolution": 100,
        "s1_mode": "IW",
        "preferred_orbit": "descending",
        "preferred_relative_orbit": 25,
        "cartopy_proj": ccrs.NorthPolarStereo(
            central_longitude=-45, true_scale_latitude=70),
        # Layout
        "scale_bar_side": "right",
        "globe_size": 0.20,
        "globe_satellite_height": 1500000,
        "globe_center": (72.0, -42.0),
        "globe_offset_x": -0.01,   # rightward from default
        "globe_offset_y": -0.12,   # downward from title
    },
    "petermann": {
        "label": "Petermann Glacier",
        "center_lat": 81.06, "center_lon": -60.87,
        "half_width_km": 50,
        "half_height_km": 50,
        "crs": "EPSG:3413",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.NorthPolarStereo(
            central_longitude=-45, true_scale_latitude=70),
        # Layout
        "scale_bar_side": "left",
        "scale_bar_km": 20,
        "globe_size": 0.20,
        "globe_satellite_height": 1500000,
        "globe_center": (72.0, -42.0),
        "globe_offset_x": 0.04,
        "globe_offset_y": -0.10,
        "meta_offset_x": -0.005,   # shift metadata text leftward
    },
    "79north": {
        "label": "79° North Glacier",
        "center_lat": 79.683, "center_lon": -19.317,
        "half_width_km": 80,
        "crs": "EPSG:3413",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.NorthPolarStereo(
            central_longitude=-45, true_scale_latitude=70),
        "scale_bar_side": "right",
        "scale_bar_km": 20,
        "globe_size": 0.20,
        "globe_satellite_height": 1500000,
        "globe_center": (72.0, -42.0),
        "preferred_orbit": "descending",
    },
    # --- Canadian Arctic ---
    "ellesmere": {
        "label": "Ellesmere Island Ice Shelves",
        "center_lat": 82.5, "center_lon": -82.0,
        "half_width_km": 120,
        "crs": "EPSG:3413",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.NorthPolarStereo(
            central_longitude=-45, true_scale_latitude=70),
    },
    # --- Antarctica ---
    "larsen_c": {
        "label": "Larsen C Ice Shelf",
        "center_lat": -67.5, "center_lon": -62.0,
        "half_width_km": 150,
        "crs": "EPSG:3031",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.SouthPolarStereo(
            central_longitude=0, true_scale_latitude=-71),
    },
    "pine_island": {
        "label": "Pine Island Glacier",
        "center_lat": -75.067, "center_lon": -101.791,
        "half_width_km": 50,
        "crs": "EPSG:3031",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.SouthPolarStereo(
            central_longitude=0, true_scale_latitude=-71),
        "scale_bar_side": "right",
        "scale_bar_km": 20,
        "globe_size": 0.20,
        "globe_satellite_height": 2500000,
        "globe_center": (-90.0, 0.0),
    },
    "thwaites": {
        "label": "Thwaites Glacier",
        "center_lat": -75.5, "center_lon": -107.0,
        "half_width_km": 80,
        "crs": "EPSG:3031",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.SouthPolarStereo(
            central_longitude=0, true_scale_latitude=-71),
        "scale_bar_side": "right",
        "scale_bar_km": 20,
        "globe_size": 0.20,
        "globe_satellite_height": 2500000,
        "globe_center": (-90.0, 0.0),
    },
    "ross": {
        "label": "Ross Ice Shelf Front",
        "center_lat": -78.0, "center_lon": 175.0,
        "half_width_km": 200,
        "crs": "EPSG:3031",
        "resolution": 100,
        "s1_mode": "IW",
        "cartopy_proj": ccrs.SouthPolarStereo(
            central_longitude=0, true_scale_latitude=-71),
    },
}

DEFAULT_START = "2019-01-01"
DEFAULT_END   = "2026-03-01"
# Adaptive search: try progressively wider windows until coverage is found
_SEARCH_WINDOWS = [3, 7, 14]   # days ± around the 1st of the month
MIN_COVERAGE = 0.02             # minimum valid-pixel fraction to keep
IMG_SIZE = 1024
DEFAULT_RENDER_RES = 200        # metres — coarser than fetch (100 m) to save memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _site_geobox(site_key):
    """Build a GeoBox for the given site."""
    cfg = SITES[site_key]
    transformer = Transformer.from_crs(
        "EPSG:4326", cfg["crs"], always_xy=True)
    cx, cy = transformer.transform(cfg["center_lon"], cfg["center_lat"])
    hw = cfg["half_width_km"] * 1000  # metres
    hh = cfg.get("half_height_km", cfg["half_width_km"]) * 1000
    grid_bbox = (cx - hw, cy - hh, cx + hw, cy + hh)
    return GeoBox.from_bbox(
        bbox=grid_bbox,
        crs=cfg["crs"],
        resolution=cfg["resolution"],
    ), grid_bbox


def _downsample(arr, factor):
    """Decimate a 2-D array by an integer factor (area-mean)."""
    if factor <= 1:
        return arr
    # Trim to exact multiple of factor
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    trimmed = arr[:h2, :w2]
    # Reshape and take mean (NaN-aware, suppress empty-slice warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(
            trimmed.reshape(h2 // factor, factor, w2 // factor, factor),
            axis=(1, 3),
        )


def _dn_to_db(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 10.0 * np.log10(np.where(arr > 0, arr.astype(np.float64), np.nan))


def _fill_seam_gaps(arr_db):
    """Fill narrow NaN gaps between abutting STAC items.

    When two GRD frames from the same pass abut with a 1-2 pixel gap,
    those pixels have no data in any item and appear as a thin black
    line.  This function detects interior NaN pixels (NaN pixels whose
    3x3 neighbourhood contains at least 5 valid values) and replaces
    them with the local mean of valid neighbours.
    """
    from scipy.ndimage import binary_dilation
    nan_mask = ~np.isfinite(arr_db)
    if not nan_mask.any():
        return arr_db
    valid_mask = np.isfinite(arr_db)
    # Interior NaN: adjacent to valid data on multiple sides
    nan_near_valid = nan_mask & binary_dilation(valid_mask, iterations=1)
    rows, cols = np.where(nan_near_valid)
    if len(rows) == 0:
        return arr_db
    filled = arr_db.copy()
    count = 0
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - 1), min(arr_db.shape[0], r + 2)
        c0, c1 = max(0, c - 1), min(arr_db.shape[1], c + 2)
        patch = arr_db[r0:r1, c0:c1]
        vals = patch[np.isfinite(patch)]
        if len(vals) >= 5:  # well-surrounded gap pixel
            filled[r, c] = np.mean(vals)
            count += 1
    if count:
        print(f"      gap-fill: {count} seam pixels interpolated")
    return filled


def _search_bbox_lonlat(site_key):
    """Return a lon/lat bounding box for STAC search (slightly padded)."""
    cfg = SITES[site_key]
    # Approximate padding in degrees (generous for polar distortion)
    pad_lat = cfg["half_width_km"] / 111.0 * 1.5
    pad_lon = pad_lat / max(np.cos(np.deg2rad(cfg["center_lat"])), 0.1)
    return [
        cfg["center_lon"] - pad_lon,
        cfg["center_lat"] - pad_lat,
        cfg["center_lon"] + pad_lon,
        cfg["center_lat"] + pad_lat,
    ]


def _dedup_items(items):
    """Remove duplicate STAC items that share the same acquisition datetime.

    When both NRT-3h and Fast-24h products exist for the same pass,
    keep only one (preferring Fast-24h).  Also deduplicates exact-ID
    duplicates that appear in Planetary Computer results.
    """
    seen = {}  # datetime-str -> item
    for it in items:
        dt = it.properties.get("datetime", "")[:19]
        tl = it.properties.get("s1:product_timeliness", "")
        prev = seen.get(dt)
        if prev is None:
            seen[dt] = it
        else:
            # Prefer Fast-24h over NRT-3h
            prev_tl = prev.properties.get("s1:product_timeliness", "")
            if tl == "Fast-24h" and prev_tl != "Fast-24h":
                seen[dt] = it
    return list(seen.values())


def _split_by_orbit(items):
    """Split items into descending and ascending lists (after dedup)."""
    desc = [it for it in items
            if it.properties.get("sat:orbit_state") == "descending"]
    asc = [it for it in items
           if it.properties.get("sat:orbit_state") == "ascending"]
    return desc, asc


def _group_by_day(items):
    """Group items by calendar day.  Returns dict[str, list]."""
    from collections import defaultdict
    by_day = defaultdict(list)
    for it in items:
        day = it.properties.get("datetime", "")[:10]
        by_day[day].append(it)
    return dict(by_day)


def _ranked_days(items, preferred_relative_orbit=None):
    """Return item lists for each day, ordered best-first.

    Ranking: days whose items match *preferred_relative_orbit* come
    first; then most items; ties broken by proximity to day 1.
    """
    by_day = _group_by_day(items)
    if not by_day:
        return [items] if items else []

    def _sort_key(d):
        day_items = by_day[d]
        # Prefer days where at least one item uses the preferred track
        if preferred_relative_orbit is not None:
            has_pref = any(
                it.properties.get("sat:relative_orbit") == preferred_relative_orbit
                for it in day_items
            )
        else:
            has_pref = True  # no preference → treat all equally
        return (not has_pref, -len(day_items), abs(int(d[8:10]) - 1))

    ranked = sorted(by_day.keys(), key=_sort_key)
    return [by_day[d] for d in ranked]


def _filter_items(items, prefer_orbit="descending"):
    """Apply all quality filters to STAC search results.

    1. Deduplicate (prefer Fast-24h over NRT-3h).
    2. Pick a single orbit direction (preferred first, fallback if empty).
    """
    items = _dedup_items(items)
    desc, asc = _split_by_orbit(items)
    if prefer_orbit == "descending":
        return desc if desc else asc
    else:
        return asc if asc else desc


def _month_dates(start_date, end_date):
    """Generate (year, month) tuples between start and end."""
    dt = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    months = []
    while dt <= dt_end:
        months.append((dt.year, dt.month))
        # Advance to next month
        if dt.month == 12:
            dt = dt.replace(year=dt.year + 1, month=1)
        else:
            dt = dt.replace(month=dt.month + 1)
    return months


# ---------------------------------------------------------------------------
# FETCH: one-day snapshot per month
# ---------------------------------------------------------------------------

def fetch_site(site_key, start_date, end_date, out_dir):
    """Fetch monthly 1-day snapshots for a single site.

    If a catalog JSON exists (from coverage_catalog.py), use its
    per-month selected date + track.  Otherwise fall back to the
    original heuristic search.
    """
    cfg = SITES[site_key]
    label = cfg["label"]
    geobox, grid_bbox = _site_geobox(site_key)
    bbox_ll = _search_bbox_lonlat(site_key)

    cache = Path(out_dir) / "calving" / "cache" / site_key
    cache.mkdir(parents=True, exist_ok=True)

    # Load or create manifest
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "site": site_key, "label": label,
            "crs": cfg["crs"], "resolution": cfg["resolution"],
            "grid_bbox": list(grid_bbox),
            "snapshots": [],
        }

    # Load catalog selections if available
    catalog_path = Path(out_dir) / "calving" / "catalogs" / f"{site_key}_catalog.json"
    catalog_selections = {}
    if catalog_path.exists():
        cat = json.loads(catalog_path.read_text())
        catalog_selections = cat.get("selected_days", {})
        print(f"  Using catalog selections from {catalog_path.name} "
              f"({len(catalog_selections)} months)")

    cached_months = {s["month_key"] for s in manifest["snapshots"]}
    months = _month_dates(start_date, end_date)

    stac_catalog = get_catalog()
    new_count = 0
    skip_count = 0

    print(f"\n  [{label}] Fetching {len(months)} months "
          f"({start_date} → {end_date})")
    print(f"    Mode: {cfg['s1_mode']}  Resolution: {cfg['resolution']}m")

    for year, month in months:
        month_key = f"{year:04d}-{month:02d}"

        if month_key in cached_months:
            skip_count += 1
            continue

        # Get catalog selection for this month (date + track)
        cat_sel = catalog_selections.get(month_key)

        # Search with widening windows
        d0 = datetime(year, month, 1)
        items = []
        used_window = 0

        for window_days in _SEARCH_WINDOWS:
            d_start = d0 - timedelta(days=window_days)
            d_end = d0 + timedelta(days=window_days)
            date_range = (f"{d_start.strftime('%Y-%m-%d')}/"
                          f"{d_end.strftime('%Y-%m-%d')}")

            found = search_s1_grd(
                stac_catalog, bbox_ll, date_range,
                max_items=200, mode=cfg["s1_mode"])

            if found:
                items = found
                used_window = window_days

        try:
            if not items:
                print(f"    {month_key}: no data")
                continue

            deduped = _dedup_items(items)

            # Auto-detect copol band from the items (hh or vv)
            band = "vv" if "vv" in deduped[0].assets else "hh"

            if cat_sel:
                # --- Catalog-guided selection ---
                target_date = cat_sel["date"]
                target_track = str(cat_sel["track"])

                # Filter to items on the selected date + track
                cand = [it for it in deduped
                        if it.datetime.strftime("%Y-%m-%d") == target_date
                        and str(it.properties.get("sat:relative_orbit", "")) == target_track
                        and band in it.assets]

                if not cand:
                    # Fallback: just the date, any track
                    cand = [it for it in deduped
                            if it.datetime.strftime("%Y-%m-%d") == target_date
                            and band in it.assets]
                    if cand:
                        print(f"    {month_key}: catalog track {target_track} "
                              f"not found on {target_date}, using available tracks")

                if not cand:
                    print(f"    {month_key}: catalog date {target_date} not found "
                          f"in STAC results, falling back to heuristic")
                    cat_sel = None  # fall through to heuristic below

            if not cat_sel:
                # --- Original heuristic selection ---
                desc, asc = _split_by_orbit(deduped)
                prefer_orbit = cfg.get("preferred_orbit", "descending")
                if prefer_orbit == "descending":
                    orbit_candidates = [desc, asc] if desc else [asc]
                else:
                    orbit_candidates = [asc, desc] if asc else [desc]

                best_arr_db = None
                best_frac = 0.0
                best_n = 0
                best_items = []

                for orbit_items in orbit_candidates:
                    if not orbit_items:
                        continue
                    pref_relorb = cfg.get("preferred_relative_orbit")
                    day_groups = _ranked_days(
                        orbit_items,
                        preferred_relative_orbit=pref_relorb)
                    found_good = False

                    for day_items in day_groups:
                        cand = [it for it in day_items if band in it.assets]
                        if not cand:
                            continue

                        ds = odc.stac.load(cand, bands=[band], geobox=geobox,
                                           groupby="solar_day",
                                           resampling="nearest")
                        arr = ds[band].median(dim="time").values.astype(
                            np.float64)
                        del ds
                        gc.collect()

                        arr_db = _dn_to_db(arr)
                        arr_db = _fill_seam_gaps(arr_db)

                        frac = np.isfinite(arr_db).sum() / arr_db.size
                        if frac > best_frac:
                            best_arr_db = arr_db
                            best_frac = frac
                            best_n = len(cand)
                            best_items = cand
                        if frac >= MIN_COVERAGE:
                            found_good = True
                            break
                    if found_good:
                        break

                if best_arr_db is None or best_frac < MIN_COVERAGE:
                    print(f"    {month_key}: too little coverage "
                          f"({best_frac:.0%})")
                    continue

                arr_db = best_arr_db
                valid_frac = best_frac
                cand = best_items
                # Proceed to save below

            if cat_sel and cand:
                # Load the catalog-selected items
                ds = odc.stac.load(cand, bands=[band], geobox=geobox,
                                   groupby="solar_day",
                                   resampling="nearest")
                arr = ds[band].median(dim="time").values.astype(np.float64)
                del ds
                gc.collect()

                arr_db = _dn_to_db(arr)
                arr_db = _fill_seam_gaps(arr_db)

                valid_frac = np.isfinite(arr_db).sum() / arr_db.size

                if valid_frac < MIN_COVERAGE:
                    print(f"    {month_key}: catalog selection too little "
                          f"coverage ({valid_frac:.0%})")
                    continue

            fname = f"{site_key}_{month_key}.npy"
            np.save(str(cache / fname), arr_db)

            # Extract metadata from the items actually used
            platforms = sorted({it.properties.get("platform", "")
                                .upper().replace("SENTINEL-1", "S1")
                                for it in cand})
            orbit_dir = cand[0].properties.get(
                "sat:orbit_state", "") if cand else ""
            rel_orbit = cand[0].properties.get(
                "sat:relative_orbit", "") if cand else ""
            acq_date = cand[0].datetime.strftime("%Y-%m-%d") if cand else ""

            manifest["snapshots"].append({
                "month_key": month_key,
                "file": fname,
                "acquisition_date": acq_date,
                "n_items": len(cand),
                "window_days": used_window,
                "valid_frac": round(float(valid_frac), 3),
                "shape": list(arr_db.shape),
                "platforms": platforms,
                "orbit_state": orbit_dir,
                "relative_orbit": rel_orbit,
                "catalog_guided": cat_sel is not None,
            })
            manifest["snapshots"].sort(key=lambda s: s["month_key"])
            manifest_path.write_text(json.dumps(manifest, indent=2))
            new_count += 1
            track_note = f" T{rel_orbit}" if rel_orbit else ""
            window_note = f"  (±{used_window}d)" if used_window > 3 else ""
            cat_note = " [catalog]" if cat_sel else " [heuristic]"
            print(f"    {month_key}: {acq_date}{track_note}, "
                  f"{len(cand)} items, "
                  f"{valid_frac:.0%} coverage{window_note}{cat_note}  ✓")

        except Exception as exc:
            print(f"    {month_key}: FAILED — {exc}")

    manifest["snapshots"].sort(key=lambda s: s["month_key"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = new_count + skip_count
    print(f"  [{label}] {new_count} new + {skip_count} cached = {total} months")


# ---------------------------------------------------------------------------
# RENDER: helpers & animation from cached snapshots
# ---------------------------------------------------------------------------

def _format_meta(meta):
    """Format metadata dict into (platform_str, orbit_str) for display."""
    if not meta:
        return "", ""
    plats = meta.get("platforms", [])
    orbit_st = meta.get("orbit_state", "")
    rel_orb = meta.get("relative_orbit", "")
    suffixes = [p.replace("S1", "") for p in plats if p.startswith("S1")]
    if suffixes:
        plat_str = "Sentinel-1" + "/".join(suffixes)
    else:
        plat_str = ", ".join(plats) if plats else ""
    if orbit_st:
        dir_str = "Asc" if orbit_st == "ascending" else "Desc"
        orb_str = f"{dir_str} {rel_orb:>03}" if rel_orb else dir_str
    else:
        orb_str = ""
    return plat_str, orb_str


def _close_ctx(ctx):
    """Close a reusable figure context."""
    if ctx and 'fig' in ctx:
        plt.close(ctx['fig'])


def _add_attribution(img):
    """Append an attribution strip to the bottom of a PIL Image."""
    from PIL import ImageDraw, ImageFont
    attr_text = "Image analysis by David Bekaert"
    strip_h = max(28, img.height // 60)
    strip = Image.new("RGB", (img.width, strip_h), (0, 0, 0))
    draw = ImageDraw.Draw(strip)
    fontsize = max(12, strip_h - 10)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), attr_text, font=font)
    tw = bbox[2] - bbox[0]
    x = img.width - tw - 10
    y = (strip_h - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), attr_text, fill=(160, 160, 160), font=font)
    combined = Image.new("RGB", (img.width, img.height + strip_h), (0, 0, 0))
    combined.paste(img, (0, 0))
    combined.paste(strip, (0, img.height))
    return combined


def _compress_gif(gif_path, lossy=80):
    """Compress a GIF in-place with gifsicle if available."""
    if shutil.which("gifsicle") is None:
        return
    tmp = str(gif_path) + ".tmp.gif"
    try:
        subprocess.run(
            ["gifsicle", f"--lossy={lossy}", "-O3", "--colors", "256",
             str(gif_path), "-o", tmp],
            check=True, capture_output=True,
        )
        os.replace(tmp, str(gif_path))
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(tmp):
            os.remove(tmp)


def _assemble_gif_from_pngs(png_paths, gif_path, fps, global_palette=False):
    """Create a GIF by streaming PNGs from disk (low memory).

    Only two frames are in memory at a time instead of all frames.

    Parameters
    ----------
    global_palette : bool
        If True, build a shared 256-colour palette from a sample of
        frames and quantize every frame against it with *no* dithering.
        This eliminates the per-frame colour-shift flicker and the
        Floyd-Steinberg speckle around text that plague large composite
        GIFs.
    """
    if not png_paths:
        return

    if global_palette:
        # Sample up to 8 evenly-spaced frames to build the palette
        n = len(png_paths)
        sample_idx = [int(i * (n - 1) / min(7, n - 1))
                      for i in range(min(8, n))]
        strips = []
        for idx in sample_idx:
            img = Image.open(str(png_paths[idx])).convert("RGB")
            # Take a vertical strip (1/8 width) to keep memory low
            w = img.width
            strip = img.crop((0, 0, max(1, w // 8), img.height))
            strips.append(strip)
            img.close()
        # Concatenate strips into one tall image and quantize
        total_h = sum(s.height for s in strips)
        combined = Image.new("RGB", (strips[0].width, total_h))
        y = 0
        for s in strips:
            combined.paste(s, (0, y))
            y += s.height
            s.close()
        palette_img = combined.quantize(colors=256,
                                        dither=Image.Dither.NONE)
        combined.close()

        def _open(path):
            img = Image.open(str(path)).convert("RGB")
            return img.quantize(palette=palette_img,
                                dither=Image.Dither.NONE)
    else:
        palette_img = None

        def _open(path):
            return Image.open(str(path)).convert("RGB")

    first = _open(png_paths[0])

    def _frames():
        for p in png_paths[1:]:
            yield _open(p)

    first.save(str(gif_path), save_all=True, append_images=_frames(),
               duration=int(1000 / fps), loop=0)
    first.close()
    if palette_img is not None:
        palette_img.close()


def _render_site_frame(arr_db, site_key, grid_bbox, title,
                       all_dates=None, current_idx=0,
                       meta=None,
                       cmap_name="gray", vis_min=15, vis_max=35,
                       show_timeline=True, _ctx=None):
    """Render one site frame as a PIL Image.

    Parameters
    ----------
    show_timeline : bool
        If False, render only the map panel (no timeline strip).
        Used by the combined-panel renderer which adds its own timeline.
    _ctx : dict or None
        Reusable figure context.  Pass None on the first call to create
        the full figure; pass the returned ctx on subsequent calls for a
        large speedup (avoids recreating cartopy/globe/coastlines).

    Returns
    -------
    img : PIL.Image
    ctx : dict
        Pass this back as ``_ctx`` on the next call.
    """
    cfg = SITES[site_key]

    # --- Fast path: reuse existing figure ---
    if _ctx is not None:
        fig = _ctx['fig']
        _ctx['im_obj'].set_data(np.ma.masked_invalid(arr_db))
        plat_str, orb_str = _format_meta(meta)
        _ctx['meta_plat_text'].set_text(plat_str)
        _ctx['meta_orb_text'].set_text(orb_str)
        if show_timeline and 'tl_d_num' in _ctx:
            d_num = _ctx['tl_d_num']
            cur_dn = d_num[current_idx]
            _ctx['tl_progress'].set_xdata([d_num[0], cur_dn])
            _ctx['tl_dot'].set_xdata([cur_dn])
            tl_ax_pos = _ctx['tl_ax_pos']
            frac_tl = (cur_dn - d_num[0]) / (d_num[-1] - d_num[0])
            fig_x = tl_ax_pos.x0 + frac_tl * tl_ax_pos.width
            if fig_x < 0.10:
                fig_x, ha_lbl = 0.10, "left"
            elif fig_x > 0.90:
                fig_x, ha_lbl = 0.90, "right"
            else:
                ha_lbl = "center"
            _ctx['tl_date_text'].set_x(fig_x)
            _ctx['tl_date_text'].set_ha(ha_lbl)
            _ctx['tl_date_text'].set_text(all_dates[current_idx])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                    edgecolor="none")
        buf.seek(0)
        return Image.open(buf).convert("RGB"), _ctx

    proj = cfg["cartopy_proj"]

    img_extent = [grid_bbox[0], grid_bbox[2],
                  grid_bbox[1], grid_bbox[3]]

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")

    # Compute aspect ratio from the bounding box
    bbox_w = grid_bbox[2] - grid_bbox[0]
    bbox_h = grid_bbox[3] - grid_bbox[1]
    aspect = bbox_h / bbox_w  # height / width of map area

    # Layout
    map_content_w = 10          # inches – the map stays this wide
    side_pad = 0.6              # extra inches each side (black border)
    fig_w = map_content_w + 2 * side_pad
    map_h = map_content_w * aspect

    if show_timeline:
        timeline_h = 0.7        # inches for the timeline strip (compact)
        top_pad = 0.55          # extra inches above timeline for date labels
        fig_h = map_h + timeline_h + top_pad
    else:
        timeline_h = 0.0
        top_pad = 0.15          # small top margin for title clearance
        fig_h = map_h + top_pad

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=round(IMG_SIZE / map_content_w),
                     facecolor="black")
    lr_frac = side_pad / fig_w  # left/right margin as fraction

    if show_timeline:
        # gridspec: thin timeline row + main SAR image
        tl_ratio = timeline_h / fig_h
        gs = fig.add_gridspec(2, 1, height_ratios=[tl_ratio, 1 - tl_ratio],
                              hspace=0.04, left=lr_frac, right=1 - lr_frac,
                              top=1 - top_pad / fig_h, bottom=0.01)

        # --- Timeline bar ---
        ax_tl = fig.add_subplot(gs[0])
        ax_tl.set_facecolor("black")
        for spine in ax_tl.spines.values():
            spine.set_visible(False)
        ax_tl.set_yticks([])

        if all_dates and len(all_dates) > 1:
            from matplotlib.dates import date2num
            import matplotlib.dates as mdates
            date_objs = [datetime.strptime(d, "%Y-%m") for d in all_dates]
            d_min, d_max = date_objs[0], date_objs[-1]
            d_num = [date2num(d) for d in date_objs]

            # Horizontal track line
            ax_tl.set_xlim(d_num[0], d_num[-1])
            ax_tl.set_ylim(-0.5, 0.5)
            ax_tl.axhline(0, color="#666666", linewidth=2, zorder=1)

            # Tick marks for all dates (upward only)
            for dn in d_num:
                ax_tl.plot([dn, dn], [0, 0.25], color="#777777",
                           linewidth=0.8, zorder=2)

            # Year labels along track
            years_shown = set()
            for d_obj, dn in zip(date_objs, d_num):
                if d_obj.year not in years_shown and d_obj.month <= 2:
                    ax_tl.text(dn, -0.2, str(d_obj.year), color="white",
                               fontsize=20, ha="center", va="top")
                    years_shown.add(d_obj.year)

            # Progress bar (filled portion) — use plot() for updatability
            cur_dn = d_num[current_idx]
            tl_progress, = ax_tl.plot([d_num[0], cur_dn], [0, 0],
                                      color="cyan", linewidth=2.5,
                                      zorder=3)

            # Current position marker
            tl_dot, = ax_tl.plot([cur_dn], [0], marker="o", color="cyan",
                                 markersize=7, zorder=5)
            # Current date label — use fig.text in figure coords so it
            # never clips off the left/right edge of the image.
            ax_pos = ax_tl.get_position()
            frac = (cur_dn - d_num[0]) / (d_num[-1] - d_num[0])
            fig_x = ax_pos.x0 + frac * ax_pos.width
            min_x, max_x = 0.10, 0.90
            if fig_x < min_x:
                fig_x = min_x
                ha_lbl = "left"
            elif fig_x > max_x:
                fig_x = max_x
                ha_lbl = "right"
            else:
                ha_lbl = "center"
            fig_y = ax_pos.y1 + 0.01
            tl_date_text = fig.text(fig_x, fig_y,
                                     all_dates[current_idx],
                                     color="white", fontsize=30,
                                     fontweight="bold",
                                     ha=ha_lbl, va="bottom")

        ax_tl.set_xticks([])

        # Map axes in row 1
        ax = fig.add_subplot(gs[1], projection=proj)
    else:
        # No timeline — single axes filling figure
        ax = fig.add_axes([lr_frac, 0.01, 1 - 2 * lr_frac,
                           1 - top_pad / fig_h - 0.01],
                          projection=proj)
    ax.set_xlim(grid_bbox[0], grid_bbox[2])
    ax.set_ylim(grid_bbox[1], grid_bbox[3])

    im_obj = ax.imshow(np.ma.masked_invalid(arr_db), origin="upper",
                       extent=img_extent, transform=proj,
                       cmap=cmap, vmin=vis_min, vmax=vis_max,
                       interpolation="nearest")

    gl = ax.gridlines(draw_labels=False, linewidth=0.3,
                      color="white", alpha=0.3, linestyle=":")

    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.6)

    # --- Title inside figure, top-right ---
    ax.text(0.97, 0.97, title, fontsize=26, fontweight="bold",
            color="white", ha="right", va="top",
            transform=ax.transAxes, zorder=10)

    # --- Scale bar ---
    sb_side = cfg.get("scale_bar_side", "right")
    map_width_km = (grid_bbox[2] - grid_bbox[0]) / 1000
    sb_km = cfg.get("scale_bar_km")
    if sb_km is None:
        for sb_km in [5, 10, 20, 25, 50, 100]:
            if sb_km >= map_width_km * 0.15 and sb_km <= map_width_km * 0.35:
                break
    sb_m = sb_km * 1000
    y0 = grid_bbox[1] + (grid_bbox[3] - grid_bbox[1]) * 0.04
    if sb_side == "left":
        x0 = grid_bbox[0] + (grid_bbox[2] - grid_bbox[0]) * 0.05
        x1 = x0 + sb_m
    else:
        x1 = grid_bbox[2] - (grid_bbox[2] - grid_bbox[0]) * 0.05
        x0 = x1 - sb_m
    ax.plot([x0, x1], [y0, y0], color="white",
            linewidth=2.5, transform=proj, solid_capstyle="butt", zorder=8)
    tick_h = (grid_bbox[3] - grid_bbox[1]) * 0.012
    for xp in [x0, x1]:
        ax.plot([xp, xp], [y0 - tick_h, y0 + tick_h], color="white",
                linewidth=1.5, transform=proj, zorder=8)
    ax.text(x0 + sb_m / 2, y0 + tick_h * 1.5, f"{sb_km} km",
            color="white", fontsize=22, ha="center", va="bottom",
            transform=proj, zorder=8)

    # --- 3D globe inset (below title, top-right, equal margin) ---
    # Use axes fraction: 3% margin from right & top of main axes
    margin = 0.03
    globe_size = cfg.get("globe_size", 0.22)  # fraction of axes width
    globe_sat_h = cfg.get("globe_satellite_height")
    globe_ctr = cfg.get("globe_center")  # (lat, lon) override
    g_lat = globe_ctr[0] if globe_ctr else cfg["center_lat"]
    g_lon = globe_ctr[1] if globe_ctr else cfg["center_lon"]
    if globe_sat_h:
        globe_proj = ccrs.NearsidePerspective(
            central_longitude=g_lon,
            central_latitude=g_lat,
            satellite_height=globe_sat_h)
    else:
        globe_proj = ccrs.Orthographic(
            central_longitude=g_lon,
            central_latitude=g_lat)
    # Position globe below the title text (~88% from bottom to leave room)
    # Convert axes fraction to figure fraction
    ax_pos = ax.get_position()
    g_w = globe_size * ax_pos.width
    g_h = g_w * fig_w / fig_h        # keep globe square in display pixels
    g_off_x = cfg.get("globe_offset_x", 0.0)   # positive = leftward
    g_off_y = cfg.get("globe_offset_y", -0.12)  # negative = downward
    g_x = ax_pos.x1 - g_w - (margin + g_off_x) * ax_pos.width
    g_y = ax_pos.y1 - g_h + g_off_y * ax_pos.height
    ax_globe = fig.add_axes([g_x, g_y, g_w, g_h],
                            projection=globe_proj)
    globe_extent = cfg.get("globe_extent")
    if globe_extent:
        ax_globe.set_extent(globe_extent, crs=ccrs.PlateCarree())
    else:
        ax_globe.set_global()
    ax_globe.add_feature(cfeature.LAND, facecolor="#444444",
                         edgecolor="none")
    ax_globe.add_feature(cfeature.OCEAN, facecolor="#111111")
    ax_globe.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'coastline', '110m',
        edgecolor='#666666', facecolor='none', linewidth=0.4))
    ax_globe.plot(cfg["center_lon"], cfg["center_lat"],
                  marker="o", color="cyan", markersize=6,
                  markeredgecolor="white", markeredgewidth=0.8,
                  transform=ccrs.PlateCarree(), zorder=10)
    ax_globe.spines['geo'].set_edgecolor('white')
    ax_globe.spines['geo'].set_linewidth(1.0)
    ax_globe.set_facecolor("#111111")
    ax_globe.patch.set_alpha(1.0)

    # --- Metadata text below globe (always create for reuse) ---
    plat_str, orb_str = _format_meta(meta)
    meta_off_x = cfg.get("meta_offset_x", 0.0)
    globe_cx = g_x + g_w / 2 + meta_off_x
    globe_bot = g_y - 0.008
    meta_plat_text = fig.text(globe_cx, globe_bot, plat_str,
                              color="white", fontsize=27,
                              ha="center", va="top")
    meta_orb_text = fig.text(globe_cx, globe_bot - 0.045, orb_str,
                             color="white", fontsize=27,
                             ha="center", va="top")

    # Build reusable context
    ctx = {
        'fig': fig,
        'im_obj': im_obj,
        'meta_plat_text': meta_plat_text,
        'meta_orb_text': meta_orb_text,
    }
    if show_timeline and all_dates and len(all_dates) > 1:
        ctx.update({
            'tl_d_num': d_num,
            'tl_progress': tl_progress,
            'tl_dot': tl_dot,
            'tl_date_text': tl_date_text,
            'tl_ax_pos': ax_tl.get_position(),
        })

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                edgecolor="none")
    buf.seek(0)
    return Image.open(buf).convert("RGB"), ctx


def _render_timeline_bar(width_px, all_dates, current_idx, _ctx=None):
    """Render a standalone timeline bar as a PIL Image.

    Parameters
    ----------
    _ctx : dict or None
        Reusable figure context.  Pass None on the first call; pass the
        returned ctx on subsequent calls for efficient reuse.

    Returns
    -------
    img : PIL.Image
    ctx : dict
    """
    from matplotlib.dates import date2num

    # --- Fast path: reuse existing figure ---
    if _ctx is not None:
        d_num = _ctx['d_num']
        cur_dn = d_num[current_idx]
        _ctx['progress'].set_xdata([d_num[0], cur_dn])
        _ctx['dot'].set_xdata([cur_dn])
        side_frac = _ctx['side_frac']
        frac = (cur_dn - d_num[0]) / (d_num[-1] - d_num[0])
        fig_x = side_frac + frac * (1 - 2 * side_frac)
        if fig_x < 0.12:
            fig_x, ha_label = 0.12, "left"
        elif fig_x > 0.88:
            fig_x, ha_label = 0.88, "right"
        else:
            ha_label = "center"
        _ctx['date_text'].set_x(fig_x)
        _ctx['date_text'].set_ha(ha_label)
        _ctx['date_text'].set_text(all_dates[current_idx])
        buf = io.BytesIO()
        _ctx['fig'].savefig(buf, format="png", facecolor="black",
                            edgecolor="none")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        tw = _ctx['target_width']
        if img.width != tw:
            img = img.resize((tw, int(img.height * tw / img.width)),
                             Image.LANCZOS)
        return img, _ctx

    # --- Full creation path ---
    timeline_h_in = 0.7
    top_pad_in = 0.85
    fig_h_in = timeline_h_in + top_pad_in
    dpi = 100  # will be rescaled to width_px
    fig_w_in = width_px / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi,
                     facecolor="black")
    side_frac = 0.15
    ax = fig.add_axes([side_frac, 0.01,
                       1 - 2 * side_frac,
                       timeline_h_in / fig_h_in])
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    date_objs = [datetime.strptime(d, "%Y-%m") for d in all_dates]
    d_num = [date2num(d) for d in date_objs]

    ax.set_xlim(d_num[0], d_num[-1])
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(0, color="#666666", linewidth=2, zorder=1)

    for dn in d_num:
        ax.plot([dn, dn], [0, 0.25], color="#777777",
                linewidth=0.8, zorder=2)

    years_shown = set()
    for d_obj, dn in zip(date_objs, d_num):
        if d_obj.year not in years_shown and d_obj.month <= 2:
            ax.text(dn, -0.2, str(d_obj.year), color="white",
                    fontsize=20, ha="center", va="top")
            years_shown.add(d_obj.year)

    cur_dn = d_num[current_idx]
    progress, = ax.plot([d_num[0], cur_dn], [0, 0],
                        color="cyan", linewidth=2.5, zorder=3)
    dot, = ax.plot([cur_dn], [0], marker="o", color="cyan",
                   markersize=7, zorder=5)

    # Place date label in *figure* coordinates so it never clips
    frac = (cur_dn - d_num[0]) / (d_num[-1] - d_num[0])  # 0..1
    fig_x = side_frac + frac * (1 - 2 * side_frac)
    if fig_x < 0.12:
        fig_x, ha_label = 0.12, "left"
    elif fig_x > 0.88:
        fig_x, ha_label = 0.88, "right"
    else:
        ha_label = "center"
    date_text = fig.text(fig_x, 0.82, all_dates[current_idx],
                         color="white", fontsize=30, fontweight="bold",
                         ha=ha_label, va="top")
    ax.set_xticks([])

    ctx = {
        'fig': fig, 'd_num': d_num, 'side_frac': side_frac,
        'progress': progress, 'dot': dot, 'date_text': date_text,
        'target_width': width_px,
    }

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", edgecolor="none")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # Resize to exact target width
    if img.width != width_px:
        img = img.resize((width_px, int(img.height * width_px / img.width)),
                         Image.LANCZOS)
    return img, ctx


def _load_site_data(site_key, out_dir, min_coverage=0.0):
    """Load cached snapshots for a site.  Returns (grid_bbox, snap_lookup, ref_shape)."""
    cfg = SITES[site_key]
    cache = Path(out_dir) / "calving" / "cache" / site_key
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        return None, {}, None

    manifest = json.loads(manifest_path.read_text())
    snapshots = manifest.get("snapshots", [])
    if not snapshots:
        return None, {}, None

    cached_bbox = tuple(manifest["grid_bbox"])
    _geobox, current_bbox = _site_geobox(site_key)
    needs_crop = (cached_bbox != current_bbox)

    if needs_crop:
        res = cfg["resolution"]
        c_x0, c_y0, c_x1, c_y1 = cached_bbox
        n_x0, n_y0, n_x1, n_y1 = current_bbox
        col0 = int(round((n_x0 - c_x0) / res))
        col1 = int(round((n_x1 - c_x0) / res))
        row0 = int(round((c_y1 - n_y1) / res))
        row1 = int(round((c_y1 - n_y0) / res))
        grid_bbox = current_bbox
    else:
        grid_bbox = cached_bbox

    snap_lookup = {}
    for snap in snapshots:
        npy_path = cache / snap["file"]
        if not npy_path.exists():
            continue
        arr_db = np.load(str(npy_path))
        if needs_crop:
            arr_db = arr_db[row0:row1, col0:col1]
        frac = np.isfinite(arr_db).sum() / arr_db.size
        if frac >= min_coverage:
            snap_lookup[snap["month_key"]] = (arr_db, snap)

    ref_shape = next(iter(snap_lookup.values()))[0].shape if snap_lookup else None
    return grid_bbox, snap_lookup, ref_shape


def render_combined(site_keys, out_dir, cmap_name="gray", fps=4,
                    vis_min=15, vis_max=35, min_coverage=0.0,
                    render_res=None):
    """Render a combined grid GIF with shared timeline.

    Layout (when all 5 default sites are present):
        Row 0 : timeline bar
        Row 1 : jakobshavn (centred, full width)
        Row 2 : petermann (left) | 79north (right)
        Row 3 : pine_island (left) | thwaites (right)

    Falls back to a single horizontal row for arbitrary site lists.

    Optimised for low memory:
    - Two-pass approach so only ONE matplotlib figure is alive at a time.
    - GIF is assembled by streaming saved PNGs from disk.
    """
    gap_px = 6  # black gap between panels
    if render_res is None:
        render_res = DEFAULT_RENDER_RES

    # --- Load manifests only (no arrays) ------------------------------
    site_info = {}
    for sk in site_keys:
        cfg = SITES[sk]
        cache = Path(out_dir) / "calving" / "cache" / sk
        manifest_path = cache / "manifest.json"
        if not manifest_path.exists():
            print(f"  [{cfg['label']}] No cached data — skipping.")
            continue
        manifest = json.loads(manifest_path.read_text())
        snapshots = manifest.get("snapshots", [])
        if not snapshots:
            print(f"  [{cfg['label']}] No snapshots — skipping.")
            continue

        cached_bbox = tuple(manifest["grid_bbox"])
        _geobox, current_bbox = _site_geobox(sk)
        needs_crop = (cached_bbox != current_bbox)
        crop_params = None
        if needs_crop:
            res = cfg["resolution"]
            c_x0, c_y0, c_x1, c_y1 = cached_bbox
            n_x0, n_y0, n_x1, n_y1 = current_bbox
            crop_params = (
                int(round((c_y1 - n_y1) / res)),
                int(round((c_y1 - n_y0) / res)),
                int(round((n_x0 - c_x0) / res)),
                int(round((n_x1 - c_x0) / res)),
            )
            grid_bbox = current_bbox
        else:
            grid_bbox = cached_bbox

        # Downsample factor for rendering
        ds_factor = max(1, int(round(render_res / cfg["resolution"])))

        # Metadata lookup (no arrays loaded yet)
        snap_meta = {}
        ref_shape = None
        for snap in snapshots:
            npy_path = cache / snap["file"]
            if npy_path.exists():
                snap_meta[snap["month_key"]] = snap
                if ref_shape is None:
                    arr = np.load(str(npy_path))
                    if needs_crop:
                        r0, r1, c0, c1 = crop_params
                        arr = arr[r0:r1, c0:c1]
                    arr = _downsample(arr, ds_factor)
                    ref_shape = arr.shape
                    del arr

        site_info[sk] = {
            'grid_bbox': grid_bbox,
            'snap_meta': snap_meta,
            'ref_shape': ref_shape,
            'needs_crop': needs_crop,
            'crop_params': crop_params,
            'cache': cache,
            'ds_factor': ds_factor,
        }

    if len(site_info) < 2:
        print("  [Combined] Need at least 2 sites with cached data.")
        return

    # --- Shared timeline (union of all months) ------------------------
    all_months = set()
    for info in site_info.values():
        all_months.update(info['snap_meta'].keys())
    all_dates = sorted(all_months)
    n_months = len(all_dates)

    print(f"\n  [Combined] {len(site_info)} sites, "
          f"{n_months} months ({all_dates[0]} → {all_dates[-1]})")

    # --- Compute panel sizes ------------------------------------------
    panel_sizes = {}
    for sk in site_info:
        cfg = SITES[sk]
        gb = site_info[sk]['grid_bbox']
        bbox_w = gb[2] - gb[0]
        bbox_h = gb[3] - gb[1]
        aspect = bbox_h / bbox_w
        map_w_in = 10
        side_pad = 0.6
        fig_w_in = map_w_in + 2 * side_pad
        top_pad_in = 0.15
        fig_h_in = map_w_in * aspect + top_pad_in
        dpi = round(IMG_SIZE / map_w_in)
        panel_sizes[sk] = (round(fig_w_in * dpi), round(fig_h_in * dpi))

    # --- Build grid layout -------------------------------------------
    # Default 5-site grid: jakobshavn on top row, then paired rows
    _default_grid = [
        ["jakobshavn"],
        ["petermann", "79north"],
        ["pine_island", "thwaites"],
    ]
    available = set(site_info.keys())
    # Use the grid layout if the expected sites are present
    if all(sk in available for row in _default_grid for sk in row):
        grid = [[sk for sk in row if sk in available] for row in _default_grid]
        # Append any extra sites not in the default grid
        used = {sk for row in grid for sk in row}
        extras = [sk for sk in site_info if sk not in used]
        if extras:
            # Pair them up
            for j in range(0, len(extras), 2):
                grid.append(extras[j:j+2])
    else:
        # Fallback: single row (old behaviour)
        grid = [list(site_info.keys())]

    # Compute row heights and total width
    # For paired rows, scale panels so both in a row have the same width
    # (= half the total width minus gap).  For single-panel rows the
    # panel is centred at its natural size (capped at total_w).
    # First, determine total_w from the widest row.
    def _row_natural_w(row):
        return sum(panel_sizes[sk][0] for sk in row) + gap_px * max(0, len(row) - 1)

    total_w = max(_row_natural_w(row) for row in grid)

    # For each row, compute the actual (w, h) of every panel after scaling
    row_panel_sizes = []  # list of list of (w, h) per panel
    row_heights = []
    for row in grid:
        if len(row) == 1:
            sk = row[0]
            pw, ph = panel_sizes[sk]
            # Don't upscale single panels wider than natural
            if pw > total_w:
                scale = total_w / pw
                pw, ph = total_w, int(ph * scale)
            row_panel_sizes.append([(pw, ph)])
            row_heights.append(ph)
        else:
            # Scale each panel to equal share of total_w
            share_w = (total_w - gap_px * (len(row) - 1)) // len(row)
            sized = []
            for sk in row:
                pw, ph = panel_sizes[sk]
                scale = share_w / pw
                sized.append((share_w, int(ph * scale)))
            row_panel_sizes.append(sized)
            row_heights.append(max(h for (_, h) in sized))

    frames_dir = Path(out_dir) / "calving" / "frames" / "combined"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # Pass 1: render each site's panels independently (one fig at a time)
    # =================================================================
    site_panel_dirs = {}
    for sk in site_info:
        info = site_info[sk]
        cfg = SITES[sk]
        snap_meta = info['snap_meta']
        panel_dir = frames_dir / f"_panels_{sk}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        site_panel_dirs[sk] = panel_dir

        site_ctx = None
        print(f"    Pass 1 [{cfg['label']}]: rendering {n_months} panels …")

        for i, month_key in enumerate(all_dates):
            if month_key in snap_meta:
                snap = snap_meta[month_key]
                npy_path = info['cache'] / snap["file"]
                arr_db = np.load(str(npy_path))
                if info['needs_crop']:
                    r0, r1, c0, c1 = info['crop_params']
                    arr_db = arr_db[r0:r1, c0:c1]
                arr_db = _downsample(arr_db, info['ds_factor'])
                frac = np.isfinite(arr_db).sum() / arr_db.size
                if frac < min_coverage:
                    arr_db = np.full(info['ref_shape'], np.nan)
                    meta = None
                else:
                    meta = {
                        "platforms": snap.get("platforms", []),
                        "orbit_state": snap.get("orbit_state", ""),
                        "relative_orbit": snap.get("relative_orbit", ""),
                    }
            elif info['ref_shape'] is not None:
                arr_db = np.full(info['ref_shape'], np.nan)
                meta = None
            else:
                continue

            panel_img, site_ctx = _render_site_frame(
                arr_db, sk, info['grid_bbox'], cfg["label"],
                all_dates=all_dates, current_idx=i,
                meta=meta, cmap_name=cmap_name,
                vis_min=vis_min, vis_max=vis_max,
                show_timeline=False,
                _ctx=site_ctx)
            del arr_db

            panel_img.save(str(panel_dir / f"{month_key}.png"))
            panel_img.close()

            if (i + 1) % 24 == 0 or i == n_months - 1:
                print(f"      {i + 1}/{n_months}")

        # Close this site's figure before moving to the next site
        _close_ctx(site_ctx)
        gc.collect()

    # =================================================================
    # Pass 2: composite per-site panels into grid + timeline
    # =================================================================
    grid_h = sum(row_heights) + gap_px * (len(grid) - 1)
    print(f"    Pass 2: compositing {n_months} combined frames "
          f"({total_w}×{grid_h} grid) …")
    tl_ctx = None
    rendered_paths = []

    for i, month_key in enumerate(all_dates):
        # Load panel images for this month
        site_panels = {}
        for sk in site_info:
            panel_path = site_panel_dirs[sk] / f"{month_key}.png"
            if panel_path.exists():
                site_panels[sk] = Image.open(str(panel_path))

        # Build grid composite
        composite = Image.new("RGB", (total_w, grid_h), (0, 0, 0))
        y_offset = 0
        for row_idx, row in enumerate(grid):
            rh = row_heights[row_idx]
            sized = row_panel_sizes[row_idx]

            if len(row) == 1:
                # Centre single panel
                sk = row[0]
                tw, th = sized[0]
                x_offset = (total_w - tw) // 2
                if sk in site_panels:
                    panel = site_panels[sk].resize((tw, th), Image.LANCZOS)
                    composite.paste(panel, (x_offset, y_offset))
                    panel.close()
            else:
                # Place panels left to right, top-aligned within row
                x_offset = 0
                for j, sk in enumerate(row):
                    tw, th = sized[j]
                    if sk in site_panels:
                        panel = site_panels[sk].resize((tw, th),
                                                        Image.LANCZOS)
                        composite.paste(panel, (x_offset, y_offset))
                        panel.close()
                    x_offset += tw + gap_px

            y_offset += rh + gap_px

        # Close source panels
        for p in site_panels.values():
            p.close()

        # Render shared timeline bar (reuse figure — tiny)
        tl_img, tl_ctx = _render_timeline_bar(total_w, all_dates, i,
                                              _ctx=tl_ctx)

        # Stack: timeline on top, grid below
        combined = Image.new("RGB",
                             (total_w, tl_img.height + grid_h),
                             (0, 0, 0))
        combined.paste(tl_img, (0, 0))
        combined.paste(composite, (0, tl_img.height))

        # Attribution strip
        combined = _add_attribution(combined)

        frame_path = frames_dir / f"combined_{month_key}.png"
        combined.save(str(frame_path))
        rendered_paths.append(frame_path)

        composite.close()
        tl_img.close()
        combined.close()

        if (i + 1) % 12 == 0 or i == n_months - 1:
            print(f"      {i + 1}/{n_months}")

    _close_ctx(tl_ctx)

    # Clean up temporary panel directories
    for panel_dir in site_panel_dirs.values():
        shutil.rmtree(panel_dir, ignore_errors=True)

    if not rendered_paths:
        print("  [Combined] No frames rendered.")
        return

    # Assemble GIF by streaming PNGs from disk (low memory)
    gif_dir = Path(out_dir) / "calving" / "animations"
    gif_dir.mkdir(parents=True, exist_ok=True)
    site_tag = "_".join(site_info.keys())
    gif_path = gif_dir / f"combined_{site_tag}_timelapse.gif"
    _assemble_gif_from_pngs(rendered_paths, gif_path, fps,
                            global_palette=True)
    _compress_gif(gif_path, lossy=30)
    print(f"  [GIF] {len(rendered_paths)} frames @ {fps} fps → {gif_path}")


def render_site(site_key, out_dir, cmap_name="gray", fps=4,
                vis_min=15, vis_max=35, min_coverage=0.0,
                sync_months=None, render_res=None):
    """Render animation for one site from cached snapshots.

    Parameters
    ----------
    sync_months : list[str] or None
        If given, the GIF will contain exactly these months in order.
        Gap months (no cached data) are rendered as a black map with
        title, scale bar, globe, and timeline but no sensor/orbit info.

    Optimised: reuses the matplotlib figure across frames and streams
    the GIF from disk PNGs instead of holding all frames in memory.
    """
    if render_res is None:
        render_res = DEFAULT_RENDER_RES
    cfg = SITES[site_key]
    label = cfg["label"]
    cache = Path(out_dir) / "calving" / "cache" / site_key
    manifest_path = cache / "manifest.json"

    if not manifest_path.exists():
        print(f"  [{label}] No cached data. Run with --fetch first.")
        return

    manifest = json.loads(manifest_path.read_text())
    snapshots = manifest.get("snapshots", [])
    if not snapshots:
        print(f"  [{label}] No snapshots in cache.")
        return

    cached_bbox = tuple(manifest["grid_bbox"])

    # Use current site config extent (may differ from cached extent)
    _geobox, current_bbox = _site_geobox(site_key)
    needs_crop = (cached_bbox != current_bbox)
    crop_params = None

    if needs_crop:
        res = cfg["resolution"]
        c_x0, c_y0, c_x1, c_y1 = cached_bbox
        n_x0, n_y0, n_x1, n_y1 = current_bbox
        crop_params = (
            int(round((c_y1 - n_y1) / res)),
            int(round((c_y1 - n_y0) / res)),
            int(round((n_x0 - c_x0) / res)),
            int(round((n_x1 - c_x0) / res)),
        )
        print(f"  Cropping cached {cached_bbox} → {current_bbox}")
        print(f"    rows [{crop_params[0]}:{crop_params[1]}], "
              f"cols [{crop_params[2]}:{crop_params[3]}]")
        grid_bbox = current_bbox
    else:
        grid_bbox = cached_bbox

    print(f"\n  [{label}] Rendering {len(snapshots)} monthly snapshots …")

    frames_dir = Path(out_dir) / "calving" / "frames" / site_key
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Downsample factor for rendering
    ds_factor = max(1, int(round(render_res / cfg["resolution"])))
    if ds_factor > 1:
        print(f"    Render resolution: {render_res}m "
              f"(downsample {ds_factor}x from {cfg['resolution']}m)")

    # Build metadata lookup (no arrays loaded) and determine ref_shape
    snap_meta = {}
    ref_shape = None
    for snap in snapshots:
        npy_path = cache / snap["file"]
        if not npy_path.exists():
            continue
        # Check coverage without keeping array in memory
        arr_db = np.load(str(npy_path))
        if needs_crop:
            r0, r1, c0, c1 = crop_params
            arr_db = arr_db[r0:r1, c0:c1]
        arr_db = _downsample(arr_db, ds_factor)
        frac = np.isfinite(arr_db).sum() / arr_db.size
        if ref_shape is None:
            ref_shape = arr_db.shape
        if frac >= min_coverage:
            snap_meta[snap["month_key"]] = snap
        del arr_db

    # Determine the timeline
    if sync_months:
        all_dates = list(sync_months)
    else:
        all_dates = sorted(snap_meta.keys())

    if not all_dates:
        print(f"  [{label}] No renderable frames.")
        return

    print(f"    Timeline: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} months)")
    if sync_months:
        n_gaps = sum(1 for m in all_dates if m not in snap_meta)
        print(f"    Sync mode: {n_gaps} gap months will show black frame")

    # Render frames (reuse figure context, load arrays on demand)
    site_ctx = None
    rendered_paths = []
    frame_dates = []

    for i, month_key in enumerate(all_dates):
        if month_key in snap_meta:
            snap = snap_meta[month_key]
            npy_path = cache / snap["file"]
            arr_db = np.load(str(npy_path))
            if needs_crop:
                r0, r1, c0, c1 = crop_params
                arr_db = arr_db[r0:r1, c0:c1]
            arr_db = _downsample(arr_db, ds_factor)
            meta = {
                "platforms": snap.get("platforms", []),
                "orbit_state": snap.get("orbit_state", ""),
                "relative_orbit": snap.get("relative_orbit", ""),
            }
            is_gap = False
        elif ref_shape is not None:
            arr_db = np.full(ref_shape, np.nan)
            meta = None
            is_gap = True
        else:
            continue

        img, site_ctx = _render_site_frame(
            arr_db, site_key, grid_bbox, label,
            all_dates=all_dates,
            current_idx=i,
            meta=meta,
            cmap_name=cmap_name,
            vis_min=vis_min, vis_max=vis_max,
            _ctx=site_ctx)
        del arr_db

        img = _add_attribution(img)

        frame_path = frames_dir / f"{site_key}_{month_key}.png"
        img.save(str(frame_path))
        rendered_paths.append(frame_path)
        frame_dates.append(month_key)
        img.close()

        tag = " (gap)" if is_gap else ""
        rendered = len(rendered_paths)
        if rendered % 12 == 0 or i == len(all_dates) - 1:
            print(f"    Rendered {rendered}/{len(all_dates)}{tag}")

    _close_ctx(site_ctx)

    if not rendered_paths:
        print(f"  [{label}] No renderable frames.")
        return

    # Animated GIF (streamed from disk PNGs)
    gif_dir = Path(out_dir) / "calving" / "animations"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{site_key}_timelapse.gif"
    _assemble_gif_from_pngs(rendered_paths, gif_path, fps)
    _compress_gif(gif_path, lossy=80)  # moderate compression for individual sites
    print(f"  [GIF] {len(rendered_paths)} frames @ {fps} fps → {gif_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Monthly Sentinel-1 time-lapse of major calving sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fetch", action="store_true",
                      help="Fetch data only (skip render).")
    mode.add_argument("--render", action="store_true",
                      help="Render only (from cache).")

    p.add_argument("--site", nargs="+",
                   choices=list(SITES.keys()) + ["all"],
                   default=["all"],
                   help="Which site(s) to process (default: all).")
    p.add_argument("--start_date", default=DEFAULT_START,
                   help=f"Start date (default: {DEFAULT_START}).")
    p.add_argument("--end_date", default=DEFAULT_END,
                   help=f"End date (default: {DEFAULT_END}).")
    p.add_argument("--cmap", default="gray",
                   help="Colour map (default: gray).")
    p.add_argument("--fps", type=int, default=4,
                   help="GIF frames per second (default: 4).")
    p.add_argument("--vis_min", type=float, default=15,
                   help="Colour-bar min dB (default: 15).")
    p.add_argument("--vis_max", type=float, default=35,
                   help="Colour-bar max dB (default: 35).")
    p.add_argument("--sync", action="store_true",
                   help="Sync GIFs: all sites share the same month timeline.")
    p.add_argument("--combined", action="store_true",
                   help="Also produce a combined side-by-side GIF "
                        "with shared timeline (requires >=2 sites).")
    p.add_argument("--render_res", type=int, default=DEFAULT_RENDER_RES,
                   help=f"Render resolution in metres (default: "
                        f"{DEFAULT_RENDER_RES}). Higher = faster & less "
                        f"memory; set to site resolution (e.g. 100) for "
                        f"full detail.")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: ./output).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    do_fetch  = args.fetch or (not args.fetch and not args.render)
    do_render = args.render or (not args.fetch and not args.render)

    sites = list(SITES.keys()) if "all" in args.site else args.site

    print("=" * 60)
    print("  Calving-site time-lapse (monthly Sentinel-1 snapshots)")
    print(f"  Sites: {', '.join(sites)}")
    if do_fetch:
        print(f"  Date range: {args.start_date} → {args.end_date}")
    print("=" * 60)

    # Compute shared timeline for --sync mode
    sync_months = None
    if args.sync and do_render and len(sites) > 1:
        all_site_months = set()
        for sk in sites:
            mp = Path(out_dir) / "calving" / "cache" / sk / "manifest.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                for s in m.get("snapshots", []):
                    all_site_months.add(s["month_key"])
        sync_months = sorted(all_site_months)
        print(f"  Sync timeline: {sync_months[0]} → {sync_months[-1]} "
              f"({len(sync_months)} months)")

    for sk in sites:
        if do_fetch:
            fetch_site(sk, args.start_date, args.end_date, out_dir)
        if do_render:
            render_site(sk, out_dir,
                        cmap_name=args.cmap, fps=args.fps,
                        vis_min=args.vis_min, vis_max=args.vis_max,
                        sync_months=sync_months,
                        render_res=args.render_res)

    # Combined side-by-side GIF
    if args.combined and do_render and len(sites) >= 2:
        render_combined(sites, out_dir,
                        cmap_name=args.cmap, fps=args.fps,
                        vis_min=args.vis_min, vis_max=args.vis_max,
                        render_res=args.render_res)

    print(f"\n[DONE] Outputs → {out_dir}")


if __name__ == "__main__":
    main()
