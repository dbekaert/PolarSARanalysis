#!/usr/bin/env python3
"""
Sentinel-1 GRD Polar Ice-Extent Time-Series (Planetary Computer).

Builds a monthly ice-extent time series for Arctic and Antarctic regions
using Sentinel-1 GRD backscatter.  Uses **all** acquisition modes
(EW + IW) to maximise coverage, applying simple dB-threshold ice
classification to the co-pol band (HH or VV).

Outputs:
  - Seasonal composite maps (winter / summer) with coastlines
  - Time-series plot of estimated ice-extent area
  - Combined overview figure

Usage:
    python polar_ice_change.py [--region arctic|antarctic|both]
                               [--start_date 2015-01-01]
                               [--end_date   2026-03-12]
                               [--cmap Blues_r]
                               [--no_boundaries]
                               [--out_dir ./output]

Author : Auto-generated for polar-cap analysis
Date   : 2026-03-12
"""

import argparse
import io
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import odc.stac
from PIL import Image

from stac_utils import get_catalog, search_s1_grd, get_copol_band


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ice classification thresholds (10·log10(DN) scale)
ICE_THRESHOLD_HH = 27.0   # HH dB above this → ice
ICE_THRESHOLD_VV = 25.0   # VV tends ~2 dB lower over ice
ICE_THRESHOLD_HV = 17.0   # Cross-pol (informational, not used here)

LON_CHUNK = 60  # longitude chunk width for STAC search

REGION_CONFIG = {
    "arctic": {
        "bbox": [-180, 65, 180, 90],
        "crs": "EPSG:3413",
        "resolution": 2000,          # 2 km for ice classification
        "cartopy_proj": ccrs.NorthPolarStereo(),
        "extent": [-180, 180, 65, 90],
        "label": "Arctic",
        "seasons": {
            "winter": {"months": [12, 1, 2, 3], "label": "Winter (Dec–Mar)"},
            "summer": {"months": [6, 7, 8, 9], "label": "Summer (Jun–Sep)"},
        },
    },
    "antarctic": {
        "bbox": [-180, -90, 180, -60],
        "crs": "EPSG:3031",
        "resolution": 2000,
        "cartopy_proj": ccrs.SouthPolarStereo(),
        "extent": [-180, 180, -90, -60],
        "label": "Antarctic",
        "seasons": {
            "winter": {"months": [6, 7, 8, 9], "label": "Winter (Jun–Sep)"},
            "summer": {"months": [12, 1, 2, 3], "label": "Summer (Dec–Mar)"},
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dn_to_db(arr: np.ndarray) -> np.ndarray:
    """Convert raw DN to log scale: 10·log10(DN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 10.0 * np.log10(np.where(arr > 0, arr.astype(np.float64), np.nan))


def classify_ice(arr_db: np.ndarray, threshold: float) -> np.ndarray:
    """Return boolean ice mask:  True where arr_db ≥ threshold."""
    return arr_db >= threshold


def _bbox_chunks(bbox: list[float]) -> list[list[float]]:
    """Split a wide bbox into longitude chunks."""
    west, south, east, north = bbox
    chunks = []
    lon = west
    while lon < east:
        lon_end = min(lon + LON_CHUNK, east)
        chunks.append([lon, south, lon_end, north])
        lon = lon_end
    return chunks


# ---------------------------------------------------------------------------
# Monthly composite and ice extent
# ---------------------------------------------------------------------------

def monthly_ice_extent(region_key: str, year: int, month: int):
    """
    Load a monthly composite for one region, classify ice, return
    (ice_fraction, total_valid_pixels, arr_db).
    """
    cfg = REGION_CONFIG[region_key]
    catalog = get_catalog()

    d1 = datetime(year, month, 1)
    if month == 12:
        d2 = datetime(year + 1, 1, 1)
    else:
        d2 = datetime(year, month + 1, 1)
    date_range = f"{d1:%Y-%m-%d}/{d2:%Y-%m-%d}"

    all_items = []
    for chunk in _bbox_chunks(cfg["bbox"]):
        items = search_s1_grd(catalog, chunk, date_range, max_items=400)
        all_items.extend(items)

    # De-duplicate
    seen = set()
    unique = []
    for it in all_items:
        if it.id not in seen:
            seen.add(it.id)
            unique.append(it)

    if not unique:
        return None, None, None

    # Split by co-pol band
    hh_items = [it for it in unique if "hh" in it.assets]
    vv_items = [it for it in unique if "vv" in it.assets and "hh" not in it.assets]

    print(f"      {year}-{month:02d}: {len(hh_items)} HH + {len(vv_items)} VV")

    composites = []
    thresholds = []

    for items, band, thr in [
        (hh_items, "hh", ICE_THRESHOLD_HH),
        (vv_items, "vv", ICE_THRESHOLD_VV),
    ]:
        if not items:
            continue
        ds = odc.stac.load(
            items,
            bands=[band],
            crs=cfg["crs"],
            resolution=cfg["resolution"],
            chunks={"x": 2048, "y": 2048},
            groupby="solar_day",
        )
        med = ds[band].median(dim="time").compute().values.astype(np.float64)
        composites.append(med)
        thresholds.append(thr)

    if not composites:
        return None, None, None

    # Merge composites: prefer HH where available
    merged_db = dn_to_db(composites[0])
    merged_thr = thresholds[0]  # HH threshold
    if len(composites) > 1:
        vv_db = dn_to_db(composites[1])
        if vv_db.shape == merged_db.shape:
            mask = np.isnan(merged_db)
            merged_db[mask] = vv_db[mask]

    valid = ~np.isnan(merged_db)
    if np.sum(valid) == 0:
        return None, None, merged_db

    ice = classify_ice(merged_db, merged_thr)
    ice_frac = np.sum(ice & valid) / np.sum(valid)
    total_valid = int(np.sum(valid))

    return ice_frac, total_valid, merged_db


# ---------------------------------------------------------------------------
# Time-series builder
# ---------------------------------------------------------------------------

def build_ice_extent_timeseries(region_key: str,
                                 start_date: str,
                                 end_date: str) -> pd.DataFrame:
    """Build monthly ice-fraction time series for a region."""
    dt = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    records = []
    while dt < dt_end:
        print(f"  [{REGION_CONFIG[region_key]['label']}] {dt:%Y-%m}")
        try:
            frac, n_valid, _ = monthly_ice_extent(region_key, dt.year, dt.month)
            if frac is not None:
                # Convert fraction to area estimate (km²)
                resolution_km = REGION_CONFIG[region_key]["resolution"] / 1000
                pixel_area_km2 = resolution_km * resolution_km
                area_km2 = frac * n_valid * pixel_area_km2
                records.append({
                    "date": dt,
                    "ice_fraction": frac,
                    "ice_area_km2": area_km2,
                    "n_valid_pixels": n_valid,
                })
        except Exception as exc:
            print(f"    [SKIP] {exc}")

        # Next month
        if dt.month == 12:
            dt = datetime(dt.year + 1, 1, 1)
        else:
            dt = datetime(dt.year, dt.month + 1, 1)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting: time series
# ---------------------------------------------------------------------------

def plot_timeseries(df: pd.DataFrame, region_key: str,
                    out_dir: str):
    """Plot ice-extent (area) over time."""
    if df.empty:
        print("  [WARN] Empty time series, skipping.")
        return

    label = REGION_CONFIG[region_key]["label"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.fill_between(df["date"], df["ice_area_km2"] / 1e6, alpha=0.3,
                     color="steelblue")
    ax1.plot(df["date"], df["ice_area_km2"] / 1e6, "o-", ms=4,
             color="steelblue", lw=1.2)
    ax1.set_ylabel("Estimated ice area (million km²)")
    ax1.set_xlabel("Date")
    ax1.set_title(f"{label} — Sentinel-1 monthly ice extent estimate",
                  fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{region_key}_ice_timeseries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [INFO] Saved → {path}")


# ---------------------------------------------------------------------------
# Plotting: seasonal composite maps
# ---------------------------------------------------------------------------

def plot_seasonal_composite(region_key: str, season_key: str,
                            year: int, cmap_name: str,
                            show_boundaries: bool,
                            out_dir: str):
    """
    Build & render a seasonal (3-month) composite map with optional
    coastlines and boundaries.
    """
    cfg = REGION_CONFIG[region_key]
    season = cfg["seasons"][season_key]
    months = season["months"]
    label = f"{cfg['label']} {season['label']} {year}"

    # Gather data over the season months
    catalog = get_catalog()
    all_items = []
    for m in months:
        yr = year if m >= 6 else year  # handle Dec-Mar spanning years
        if season_key == "winter" and region_key == "arctic" and m in (1, 2, 3):
            yr = year  # Jan-Mar of the same winter
        d1 = datetime(yr, m, 1)
        if m == 12:
            d2 = datetime(yr + 1, 1, 1)
        else:
            d2 = datetime(yr, m + 1, 1)
        date_range = f"{d1:%Y-%m-%d}/{d2:%Y-%m-%d}"
        for chunk in _bbox_chunks(cfg["bbox"]):
            items = search_s1_grd(catalog, chunk, date_range, max_items=200)
            all_items.extend(items)

    # De-dup
    seen = set()
    unique = []
    for it in all_items:
        if it.id not in seen:
            seen.add(it.id)
            unique.append(it)

    if not unique:
        print(f"  [WARN] No data for {label}")
        return

    hh_items = [it for it in unique if "hh" in it.assets]
    vv_items = [it for it in unique if "vv" in it.assets and "hh" not in it.assets]
    print(f"      {label}: {len(hh_items)} HH + {len(vv_items)} VV scenes")

    composites = []
    for items, band in [(hh_items, "hh"), (vv_items, "vv")]:
        if not items:
            continue
        ds = odc.stac.load(
            items, bands=[band],
            crs=cfg["crs"], resolution=cfg["resolution"],
            chunks={"x": 2048, "y": 2048},
            groupby="solar_day",
        )
        med = ds[band].median(dim="time").compute().values.astype(np.float64)
        composites.append(med)

    if not composites:
        return

    merged = composites[0].copy()
    if len(composites) > 1 and composites[1].shape == merged.shape:
        mask = (merged == 0) | np.isnan(merged)
        merged[mask] = composites[1][mask]

    arr_db = dn_to_db(merged)

    # Render with cartopy
    proj = cfg["cartopy_proj"]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")

    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(cfg["extent"], crs=ccrs.PlateCarree())

    masked = np.ma.masked_invalid(arr_db)
    ax.imshow(
        masked, origin="upper", extent=ax.get_extent(), transform=proj,
        cmap=cmap, vmin=15, vmax=35, interpolation="nearest",
    )

    if show_boundaries:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white",
                       linestyle="--")
        ax.add_feature(cfeature.LAND, facecolor=(0.15, 0.15, 0.15, 0.3),
                       edgecolor="none")
        gl = ax.gridlines(draw_labels=False, linewidth=0.3,
                          color="white", alpha=0.4, linestyle=":")
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))

    ax.set_title(label, fontsize=14, fontweight="bold", color="white",
                 pad=12, loc="left")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.6)
    fig.tight_layout(pad=0.5)

    path = os.path.join(out_dir, f"{region_key}_{season_key}_{year}.png")
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  [INFO] Saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    today = datetime.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(
        description="Sentinel-1 polar ice-extent time series & seasonal composites.",
    )
    p.add_argument("--region", choices=["arctic", "antarctic", "both"],
                   default="both",
                   help="Region (default: both).")
    p.add_argument("--start_date", default="2024-01-01",
                   help="Start date YYYY-MM-DD (default: 2024-01-01).")
    p.add_argument("--end_date", default=today,
                   help=f"End date YYYY-MM-DD (default: {today}).")
    p.add_argument("--cmap", default="Blues_r",
                   help="Matplotlib colour-map for seasonal maps (default: Blues_r).")
    p.add_argument("--no_boundaries", action="store_true",
                   help="Disable coastline & boundary overlay.")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: ./output).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    show_bounds = not args.no_boundaries
    regions = (["arctic", "antarctic"] if args.region == "both"
               else [args.region])

    print("[INFO] Data source: Microsoft Planetary Computer (no auth needed)")
    print(f"[INFO] Colour map : {args.cmap}")
    print(f"[INFO] Boundaries : {'ON' if show_bounds else 'OFF'}")

    for rk in regions:
        cfg = REGION_CONFIG[rk]
        print(f"\n{'=' * 60}")
        print(f"  ICE EXTENT : {cfg['label'].upper()}")
        print(f"  Period     : {args.start_date} → {args.end_date}")
        print(f"{'=' * 60}\n")

        # --- Time series ---
        df = build_ice_extent_timeseries(rk, args.start_date, args.end_date)
        if not df.empty:
            csv_path = os.path.join(out_dir, f"{rk}_ice_timeseries.csv")
            df.to_csv(csv_path, index=False)
            print(f"  [INFO] Saved CSV → {csv_path}")
            plot_timeseries(df, rk, out_dir)

        # --- Seasonal composites (latest year that has full data) ---
        latest_year = datetime.strptime(args.end_date, "%Y-%m-%d").year - 1
        for season_key in cfg["seasons"]:
            plot_seasonal_composite(rk, season_key, latest_year,
                                    args.cmap, show_bounds, out_dir)

    print(f"\n[DONE] All outputs → {out_dir}")


if __name__ == "__main__":
    main()
