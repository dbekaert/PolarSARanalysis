#!/usr/bin/env python3
"""
Sentinel-1 GRD Weekly Polar Mosaic Animation (Planetary Computer).

Two-phase workflow so you can iterate on visuals without re-downloading:

  1. FETCH  – download weekly composites → cached .npy files  (slow, once)
  2. RENDER – load cache → animated GIF                       (fast, repeat)

Usage examples:

    # Full pipeline: fetch data + render animation
    python animate_polar_weekly.py --region arctic

    # Fetch only – save weekly composites to cache (interrupt-safe)
    python animate_polar_weekly.py --region arctic --fetch

    # Render only – iterate on cmap, fps, boundaries, date format …
    python animate_polar_weekly.py --region arctic --render --cmap inferno --fps 6
    python animate_polar_weekly.py --region arctic --render --cmap gray --no_boundaries
    python animate_polar_weekly.py --region arctic --render --fps 2 --vis_min 10 --vis_max 30

Cache is stored under  <out_dir>/cache/<region>/  as individual .npy files
per week, plus a manifest.json with metadata.  Subsequent --fetch runs skip
weeks that are already cached.
"""

import argparse
import io
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gc
import odc.stac
import planetary_computer as pc
from odc.geo.geobox import GeoBox
from PIL import Image
from pyproj import Transformer

from stac_utils import get_catalog, search_s1_grd, get_copol_band


# ---------------------------------------------------------------------------
# Region configuration
# ---------------------------------------------------------------------------

REGION_CONFIG = {
    "arctic": {
        "bbox": [-180, 60, 180, 90],
        "crs": "EPSG:3413",
        "resolution": 5000,
        # Fixed output grid (meters in EPSG:3413).  All weeks snap to
        # this exact pixel grid — eliminates frame-to-frame geocoding shift.
        "grid_bbox": (-3850000, -3850000, 3850000, 3850000),
        # EPSG:3413 = NSIDC Sea Ice Polar Stereo North
        #   central_longitude = -45°,  true_scale_latitude = 70°
        "cartopy_proj": ccrs.NorthPolarStereo(
            central_longitude=-45, true_scale_latitude=70),
        "label": "Arctic",
    },
    "antarctic": {
        "bbox": [-180, -90, 180, -60],
        "crs": "EPSG:3031",
        "resolution": 5000,
        "grid_bbox": (-3850000, -3850000, 3850000, 3850000),
        # EPSG:3031 = Antarctic Polar Stereographic
        #   central_longitude = 0°,  true_scale_latitude = -71°
        "cartopy_proj": ccrs.SouthPolarStereo(
            central_longitude=0, true_scale_latitude=-71),
        "label": "Antarctic",
    },
}

LON_CHUNK = 60  # degrees – STAC search longitude chunk width

# Default date range: last 12 months
DEFAULT_START = "2025-03-12"
DEFAULT_END   = "2026-03-12"

IMG_SIZE = 1024

# Sentinel-1 approximate geographic heading (degrees CW from north)
_S1_HEADING_ASC  = 348.0
_S1_HEADING_DESC = 192.0


# ===================================================================
#  PHASE 1 — FETCH : download weekly composites → cache
# ===================================================================

def dn_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert raw DN to dB: 10·log10(DN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 10.0 * np.log10(np.where(arr > 0, arr.astype(np.float64), np.nan))


def _bbox_chunks(bbox: list[float]) -> list[list[float]]:
    west, south, east, north = bbox
    chunks = []
    lon = west
    while lon < east:
        chunks.append([lon, south, min(lon + LON_CHUNK, east), north])
        lon += LON_CHUNK
    return chunks


def _get_geobox(region_key: str) -> GeoBox:
    """Return a fixed-size GeoBox for the region.

    Using a fixed grid ensures every weekly composite has the exact same
    pixel grid (shape, origin, resolution) — eliminating frame-to-frame
    geocoding shifts in the animation.
    """
    cfg = REGION_CONFIG[region_key]
    return GeoBox.from_bbox(
        bbox=cfg["grid_bbox"],
        crs=cfg["crs"],
        resolution=cfg["resolution"],
    )


def _load_weekly_mosaic(region_key: str, start: str, end: str,
                        s1_mode: str = "EW") -> np.ndarray | None:
    """Download a 7-day median composite.  Returns dB array.

    Parameters
    ----------
    region_key : str
    start, end : date strings (YYYY-MM-DD)
    s1_mode : str
        Sentinel-1 acquisition mode.  Default 'EW' (Extra-Wide, HH/HV)
        provides the most uniform polar coverage.  Use 'IW' for land-only,
        or None/'' for both (not recommended — causes calibration seams).
    """
    cfg = REGION_CONFIG[region_key]
    catalog = get_catalog()
    geobox = _get_geobox(region_key)

    # Determine mode filter
    mode_filter = s1_mode if s1_mode else None

    all_items = []
    for chunk in _bbox_chunks(cfg["bbox"]):
        all_items.extend(
            search_s1_grd(catalog, chunk, f"{start}/{end}",
                          max_items=600, mode=mode_filter))

    seen = set()
    unique = [it for it in all_items
              if it.id not in seen and not seen.add(it.id)]

    if not unique:
        return None

    # Pick band based on mode:  EW → hh,  IW → vv
    if s1_mode == "EW":
        band = "hh"
        items = [it for it in unique if "hh" in it.assets]
    elif s1_mode == "IW":
        band = "vv"
        items = [it for it in unique if "vv" in it.assets]
    else:
        band = "hh"
        items = [it for it in unique if "hh" in it.assets]

    mode_label = s1_mode or "ALL"
    print(f"      {mode_label} mode: {len(items)} scenes with {band.upper()}")

    if not items:
        return None

    # Load onto fixed geobox — every week gets the exact same grid
    ds = odc.stac.load(items, bands=[band],
                       geobox=geobox,
                       chunks={"x": 2048, "y": 2048},
                       groupby="solar_day")
    arr = ds[band].median(dim="time").compute().values.astype(np.float64)

    return dn_to_db(arr)


def _cross_track_unit_vector(item, transformer):
    """Return the cross-track unit vector on the output map grid.

    Sentinel-1 is right-looking: cross-track points 90° clockwise from
    the flight direction.
    """
    coords = item.geometry["coordinates"][0]
    clon = float(np.mean([c[0] for c in coords]))
    clat = float(np.mean([c[1] for c in coords]))

    orbit = item.properties.get("sat:orbit_state", "ascending")
    heading_rad = np.deg2rad(
        _S1_HEADING_ASC if orbit == "ascending" else _S1_HEADING_DESC)
    delta = 0.5

    lat1 = clat - delta * np.cos(heading_rad)
    lon1 = clon - delta * np.sin(heading_rad) / max(np.cos(np.deg2rad(clat)), 0.1)
    lat2 = clat + delta * np.cos(heading_rad)
    lon2 = clon + delta * np.sin(heading_rad) / max(np.cos(np.deg2rad(clat)), 0.1)

    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)

    along_x, along_y = float(x2 - x1), float(y2 - y1)
    norm = np.hypot(along_x, along_y)
    if norm < 1.0:
        return 1.0, 0.0
    return along_y / norm, -along_x / norm


def _swath_center_weight(item, transformer, grid_x, grid_y, taper_fraction=0.3):
    """Compute per-pixel weight based on distance from swath centre.

    Returns a weight array (same shape as grid_x) with values in [0, 1]:
      - 1.0 in the central portion of the swath
      - tapers to 0 at the near/far range edges (outermost *taper_fraction*)
      - 0 outside the swath footprint

    Parameters
    ----------
    taper_fraction : float
        Fraction of the swath width (each side) over which to taper.
        0.3 means the outer 30 % on each side is tapered, keeping the
        central 40 % at full weight.
    """
    cx, cy = _cross_track_unit_vector(item, transformer)

    # Project footprint vertices to find swath extent in cross-track
    coords = item.geometry["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    xs, ys = transformer.transform(lons, lats)
    fp_ct = np.array([float(x) * cx + float(y) * cy for x, y in zip(xs, ys)])
    ct_min, ct_max = float(fp_ct.min()), float(fp_ct.max())
    ct_width = ct_max - ct_min
    if ct_width < 1000:
        return np.ones_like(grid_x)

    # Cross-track position for every output pixel
    ct = grid_x * cx + grid_y * cy

    # Normalise to [0, 1] across the swath (0 = near-range edge, 1 = far-range)
    ct_norm = (ct - ct_min) / ct_width

    # Symmetric distance from centre: 0 at centre, 0.5 at edges
    dist = np.abs(ct_norm - 0.5)

    # Build weight: 1.0 in centre, cosine taper at edges
    edge_start = 0.5 - taper_fraction
    weight = np.where(
        dist <= edge_start,
        1.0,
        0.5 * (1.0 + np.cos(np.pi * (dist - edge_start) / taper_fraction))
    )
    # Zero out pixels outside the swath
    weight[(ct_norm < 0) | (ct_norm > 1)] = 0.0
    return weight.astype(np.float32)


def _load_weekly_mosaic_weighted(region_key: str, start: str, end: str,
                                 s1_mode: str = "EW") -> np.ndarray | None:
    """Download a 7-day weighted composite.  Returns dB array.

    Each item is loaded individually and weighted by its distance from
    swath centre (centre = full weight, edges taper to 0).  This
    effectively suppresses the cross-track antenna gain ramp without
    needing the calibration LUTs.
    """
    cfg = REGION_CONFIG[region_key]
    catalog = get_catalog()
    geobox = _get_geobox(region_key)

    all_items = []
    for chunk in _bbox_chunks(cfg["bbox"]):
        all_items.extend(
            search_s1_grd(catalog, chunk, f"{start}/{end}",
                          max_items=600, mode=s1_mode or None))

    seen = set()
    unique = [it for it in all_items
              if it.id not in seen and not seen.add(it.id)]
    if not unique:
        return None

    band = "hh" if s1_mode == "EW" else "vv"
    items = [it for it in unique if band in it.assets]
    print(f"      {s1_mode or 'ALL'} mode: {len(items)} scenes with {band.upper()}"
          f"  [weighted mosaic]")
    if not items:
        return None

    shape = (geobox.shape[0], geobox.shape[1])
    wt_sum = np.zeros(shape, dtype=np.float64)
    dn_sum = np.zeros(shape, dtype=np.float64)

    # Precompute grid coordinates
    tf = geobox.affine
    cols, rows = np.meshgrid(
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[0], dtype=np.float32))
    grid_x = (tf.a * cols + tf.c).astype(np.float32)
    grid_y = (tf.e * rows + tf.f).astype(np.float32)
    del cols, rows

    crs_transformer = Transformer.from_crs(
        "EPSG:4326", cfg["crs"], always_xy=True)

    n_ok = 0
    for i, item in enumerate(items):
        try:
            signed = pc.sign(item)
            ds = odc.stac.load([signed], bands=[band], geobox=geobox)
            dn = ds[band].isel(time=0).values.astype(np.float32)
            del ds

            valid = dn > 0
            if valid.sum() < 100:
                continue

            w = _swath_center_weight(item, crs_transformer, grid_x, grid_y)
            mask = valid & (w > 0)

            dn_sum[mask] += dn[mask].astype(np.float64) * w[mask]
            wt_sum[mask] += w[mask]

            del dn, w, mask, valid
            n_ok += 1

            if (i + 1) % 50 == 0:
                print(f"        {i+1}/{len(items)} items loaded …")
                gc.collect()

        except Exception:
            continue

    if n_ok == 0:
        return None

    print(f"      {n_ok}/{len(items)} items used")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dn_mean = np.where(wt_sum > 0, dn_sum / wt_sum, np.nan)

    return dn_to_db(dn_mean)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_dir(out_dir: str, region_key: str) -> Path:
    return Path(out_dir) / "cache" / region_key


def _manifest_path(out_dir: str, region_key: str) -> Path:
    return _cache_dir(out_dir, region_key) / "manifest.json"


def _load_manifest(out_dir: str, region_key: str) -> dict:
    mp = _manifest_path(out_dir, region_key)
    if mp.exists():
        return json.loads(mp.read_text())
    return {"region": region_key,
            "crs": REGION_CONFIG[region_key]["crs"],
            "resolution": REGION_CONFIG[region_key]["resolution"],
            "grid_bbox": list(REGION_CONFIG[region_key]["grid_bbox"]),
            "s1_mode": "EW",
            "weeks": []}


def _save_manifest(out_dir: str, region_key: str, manifest: dict):
    mp = _manifest_path(out_dir, region_key)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2))


def _week_filename(week_idx: int, start: str) -> str:
    return f"week_{week_idx:03d}_{start}.npy"


# ---------------------------------------------------------------------------
# fetch_composites — download & cache
# ---------------------------------------------------------------------------

def fetch_composites(region_key: str, start_date: str, end_date: str,
                     out_dir: str, s1_mode: str = "EW",
                     weighted: bool = False):
    """
    Download weekly composites for *region_key* and cache them.

    Already-cached weeks are skipped, so interrupted runs can be resumed.
    When *weighted* is True, items are mosaicked with swath-centre
    weighting to suppress the cross-track brightness ramp.
    """
    cfg = REGION_CONFIG[region_key]
    label = cfg["label"]
    cache = _cache_dir(out_dir, region_key)
    cache.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(out_dir, region_key)
    # Store mode in manifest so render knows what was fetched
    manifest["s1_mode"] = s1_mode
    manifest["weighted"] = weighted
    cached_starts = {w["start"] for w in manifest["weeks"]}

    dt = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    week_idx = 0
    new_count = 0
    skip_count = 0

    while dt < dt_end:
        week_idx += 1
        d1 = min(dt + timedelta(days=7), dt_end)
        s0 = dt.strftime("%Y-%m-%d")
        s1 = d1.strftime("%Y-%m-%d")

        if s0 in cached_starts:
            print(f"  [{label}] Week {week_idx}: {s0} → {s1}  [CACHED — skip]")
            skip_count += 1
            dt = d1
            continue

        print(f"  [{label}] Week {week_idx}: {s0} → {s1}  [fetching …]")
        try:
            if weighted:
                arr = _load_weekly_mosaic_weighted(
                    region_key, s0, s1, s1_mode=s1_mode)
            else:
                arr = _load_weekly_mosaic(region_key, s0, s1, s1_mode=s1_mode)
            if arr is None:
                raise RuntimeError("no STAC items found")

            fname = _week_filename(week_idx, s0)
            np.save(str(cache / fname), arr)

            manifest["weeks"].append({
                "index": week_idx,
                "start": s0,
                "end":   s1,
                "file":  fname,
                "shape": list(arr.shape),
            })
            # Save manifest after each week → interrupt-safe
            _save_manifest(out_dir, region_key, manifest)
            new_count += 1

        except Exception as exc:
            print(f"    [SKIP] {exc}")

        dt = d1

    # Sort manifest weeks by start date
    manifest["weeks"].sort(key=lambda w: w["start"])
    _save_manifest(out_dir, region_key, manifest)

    total = new_count + skip_count
    print(f"\n  [FETCH DONE] {label}: {new_count} new + {skip_count} cached"
          f" = {total} weeks  →  {cache}")


# ===================================================================
#  PHASE 2 — RENDER : load cache → animation
# ===================================================================

def _deramp_mosaic_db(arr_db: np.ndarray, kernel_size: int = 151) -> np.ndarray:
    """Remove large-scale brightness gradient from a dB mosaic.

    Estimates the smooth background via a large box filter, subtracts it,
    and re-centres on the global median.  Keeps fine-scale geophysical
    signal while removing the systematic instrument-gain ramp.
    """
    valid = np.isfinite(arr_db)
    if valid.sum() < 1000:
        return arr_db.copy()
    med = float(np.nanmedian(arr_db))
    filled = np.where(valid, arr_db, med).astype(np.float64)
    bg = uniform_filter(filled, size=kernel_size, mode="nearest")
    weight = uniform_filter(valid.astype(np.float64),
                            size=kernel_size, mode="nearest")
    reliable = weight > 0.3
    deramped = arr_db.copy()
    ok = valid & reliable
    deramped[ok] = arr_db[ok] - bg[ok] + med
    return deramped


def _compute_ramp_field(weeks: list, cache: Path) -> np.ndarray:
    """Compute a stable ramp-correction field from the temporal mean."""
    stack = [np.load(str(cache / wk["file"])) for wk in weeks]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temporal_mean = np.nanmean(stack, axis=0)
    bg = _deramp_mosaic_db(temporal_mean)
    ramp_field = temporal_mean - bg
    return ramp_field


def render_frame(arr_db: np.ndarray, region_key: str, title: str,
                 cmap_name: str = "plasma", show_boundaries: bool = True,
                 vis_min: float = 15, vis_max: float = 35) -> Image.Image:
    """Render one frame as a PIL Image with cartopy projection."""
    cfg = REGION_CONFIG[region_key]
    proj = cfg["cartopy_proj"]

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")
    masked = np.ma.masked_invalid(arr_db)

    # Data extent in projection-native metres (must match the geobox)
    data_extent = cfg["grid_bbox"]  # (left, bottom, right, top)
    img_extent = [data_extent[0], data_extent[2],
                  data_extent[1], data_extent[3]]  # [xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=(10, 10), dpi=round(IMG_SIZE / 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    # Set axes limits to the data extent in native projection coords
    ax.set_xlim(data_extent[0], data_extent[2])
    ax.set_ylim(data_extent[1], data_extent[3])

    ax.imshow(masked, origin="upper",
              extent=img_extent,
              transform=proj, cmap=cmap, vmin=vis_min, vmax=vis_max,
              interpolation="nearest")

    if show_boundaries:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5, edgecolor="white",
                       linestyle="--")
        ax.add_feature(cfeature.LAND,
                       facecolor=(0.15, 0.15, 0.15, 0.3), edgecolor="none")
        gl = ax.gridlines(draw_labels=False, linewidth=0.3,
                          color="white", alpha=0.4, linestyle=":")
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, 10))

    ax.set_title(title, fontsize=14, fontweight="bold", color="white",
                 pad=12, loc="left")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.6)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _format_title(region_key: str, start: str, end: str,
                  date_fmt: str = "%Y-%m-%d") -> str:
    """Build a frame title.  *date_fmt* controls how dates appear."""
    label = REGION_CONFIG[region_key]["label"]
    ds = datetime.strptime(start, "%Y-%m-%d").strftime(date_fmt)
    de = datetime.strptime(end,   "%Y-%m-%d").strftime(date_fmt)
    return f"{label}  |  {ds}  →  {de}"


def render_animation(region_key: str, out_dir: str,
                     cmap_name: str = "plasma",
                     show_boundaries: bool = True,
                     fps: int = 4,
                     vis_min: float = 15,
                     vis_max: float = 35,
                     date_fmt: str = "%Y-%m-%d",
                     deramp: bool = False):
    """
    Render an animated GIF from cached weekly composites.

    This is fast (seconds–minutes) because no data is downloaded.
    """
    manifest = _load_manifest(out_dir, region_key)
    weeks = manifest.get("weeks", [])
    if not weeks:
        print(f"  [ERROR] No cached data for {region_key}. Run with --fetch first.")
        return

    cache = _cache_dir(out_dir, region_key)
    label = REGION_CONFIG[region_key]["label"]

    print(f"  [{label}] Rendering {len(weeks)} cached weeks …")
    print(f"  [{label}]   cmap={cmap_name}  fps={fps}  "
          f"vis=[{vis_min}, {vis_max}]  boundaries={'ON' if show_boundaries else 'OFF'}"
          f"  deramp={'ON' if deramp else 'OFF'}")

    ramp_field = None
    if deramp:
        print(f"  [{label}] Computing ramp correction from temporal mean …")
        ramp_field = _compute_ramp_field(weeks, cache)

    frames: list[Image.Image] = []
    frame_dates: list[str] = []
    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, wk in enumerate(weeks, 1):
        npy_path = cache / wk["file"]
        if not npy_path.exists():
            print(f"    [SKIP] {wk['file']} not found")
            continue

        arr = np.load(str(npy_path))
        if ramp_field is not None:
            med_orig = np.nanmedian(arr)
            arr = arr - ramp_field
            arr += med_orig - np.nanmedian(arr)  # re-centre
        title = _format_title(region_key, wk["start"], wk["end"], date_fmt)

        img = render_frame(arr, region_key, title,
                           cmap_name=cmap_name,
                           show_boundaries=show_boundaries,
                           vis_min=vis_min, vis_max=vis_max)
        frames.append(img)
        frame_dates.append(wk["start"])

        # Save individual frame
        img.save(str(frames_dir /
                     f"{region_key}_week_{wk['index']:03d}_{wk['start']}.png"))

        if i % 10 == 0 or i == len(weeks):
            print(f"    Rendered {i}/{len(weeks)}")

    if not frames:
        print(f"  [WARN] No renderable frames for {label}")
        return

    # --- Animated GIF ---
    gif_path = os.path.join(out_dir, f"{region_key}_weekly_amplitude.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"  [GIF] {len(frames)} frames @ {fps} fps → {gif_path}")

    # --- Highlight snapshots ---
    for tag, idx in {"first": 0, "mid": len(frames) // 2,
                     "last": len(frames) - 1}.items():
        p = os.path.join(out_dir,
                         f"{region_key}_highlight_{tag}_{frame_dates[idx]}.png")
        frames[idx].save(p)
        print(f"  [IMG] {p}")


# ---------------------------------------------------------------------------
# Colour-bar legend
# ---------------------------------------------------------------------------

def save_colorbar(out_dir: str, cmap_name: str = "plasma",
                  vis_min: float = 15, vis_max: float = 35):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vis_min, vmax=vis_max)
    fig, ax = plt.subplots(figsize=(6, 0.6))
    fig.subplots_adjust(bottom=0.55)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation="horizontal")
    cb.set_label("Sentinel-1 Backscatter – 10·log₁₀(DN)", fontsize=11)
    path = os.path.join(out_dir, "colorbar_backscatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Colour bar → {path}")


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Weekly Sentinel-1 polar animation (Planetary Computer).\n\n"
                    "Run --fetch once to cache data, then --render repeatedly "
                    "to iterate on visual settings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Mode ---
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fetch", action="store_true",
                      help="Download weekly composites → cache (skip render).")
    mode.add_argument("--render", action="store_true",
                      help="Render animation from cache (skip download).")

    # --- Region & dates ---
    p.add_argument("--region", choices=["arctic", "antarctic", "both"],
                   default="both")
    p.add_argument("--start_date", default=DEFAULT_START,
                   help=f"Start date YYYY-MM-DD (default: {DEFAULT_START}).")
    p.add_argument("--end_date", default=DEFAULT_END,
                   help=f"End date YYYY-MM-DD (default: {DEFAULT_END}).")
    p.add_argument("--s1_mode", default="EW",
                   choices=["EW", "IW"],
                   help="Sentinel-1 acquisition mode (default: EW). "
                        "EW provides wall-to-wall polar ocean/ice coverage "
                        "(HH band). IW covers land masses (VV band).")
    p.add_argument("--weighted", action="store_true",
                   help="Use swath-centre weighted mosaic during fetch. "
                        "Down-weights swath edges where the cross-track "
                        "brightness ramp is strongest. Slower but produces "
                        "more uniform mosaics.")

    # --- Render settings (used by render / default mode) ---
    p.add_argument("--cmap", default="plasma",
                   help="Matplotlib colour-map (default: plasma). "
                        "Examples: gray, inferno, viridis, cividis, magma, "
                        "coolwarm, Greys_r, bone.")
    p.add_argument("--fps", type=int, default=4,
                   help="GIF frames per second (default: 4).")
    p.add_argument("--vis_min", type=float, default=15,
                   help="Colour-bar minimum dB (default: 15).")
    p.add_argument("--vis_max", type=float, default=35,
                   help="Colour-bar maximum dB (default: 35).")
    p.add_argument("--no_boundaries", action="store_true",
                   help="Disable coastline & border overlay.")
    p.add_argument("--deramp", action="store_true",
                   help="Remove cross-track brightness ramp from mosaics.")
    p.add_argument("--date_format", default="%Y-%m-%d",
                   help="strftime format for dates on frames "
                        "(default: %%Y-%%m-%%d). Try: '%%b %%d, %%Y'.")

    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: ./output).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    do_fetch  = args.fetch or (not args.fetch and not args.render)
    do_render = args.render or (not args.fetch and not args.render)

    regions = (["arctic", "antarctic"] if args.region == "both"
               else [args.region])

    print("[INFO] Data source  : Microsoft Planetary Computer (no auth)")
    if do_fetch:
        print(f"[INFO] S1 mode      : {args.s1_mode}")
        print(f"[INFO] Weighted     : {'ON' if args.weighted else 'OFF'}")
        print(f"[INFO] Date range   : {args.start_date} → {args.end_date}")
    if do_render:
        print(f"[INFO] Colour map   : {args.cmap}")
        print(f"[INFO] FPS          : {args.fps}")
        print(f"[INFO] dB range     : [{args.vis_min}, {args.vis_max}]")
        print(f"[INFO] Boundaries   : {'OFF' if args.no_boundaries else 'ON'}")
        print(f"[INFO] Deramp       : {'ON' if args.deramp else 'OFF'}")
        print(f"[INFO] Date format  : {args.date_format}")

    for rk in regions:
        label = REGION_CONFIG[rk]["label"]
        print(f"\n{'=' * 60}")
        print(f"  {label.upper()}")
        print(f"{'=' * 60}\n")

        if do_fetch:
            fetch_composites(rk, args.start_date, args.end_date, out_dir,
                             s1_mode=args.s1_mode,
                             weighted=args.weighted)

        if do_render:
            render_animation(rk, out_dir,
                             cmap_name=args.cmap,
                             show_boundaries=not args.no_boundaries,
                             fps=args.fps,
                             vis_min=args.vis_min,
                             vis_max=args.vis_max,
                             date_fmt=args.date_format,
                             deramp=args.deramp)

    if do_render:
        save_colorbar(out_dir, cmap_name=args.cmap,
                      vis_min=args.vis_min, vis_max=args.vis_max)

    print(f"\n[DONE] Outputs → {out_dir}")
    if do_fetch and not do_render:
        print("[TIP]  Re-run with --render to build the animation from cache:")
        print(f"       python animate_polar_weekly.py --region {args.region} "
              f"--render --cmap plasma --fps 4")


if __name__ == "__main__":
    main()
