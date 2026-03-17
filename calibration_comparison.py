#!/usr/bin/env python3
"""
Compare uncalibrated 10*log10(DN) vs properly calibrated sigma0(dB)
with thermal noise removal, for one week of Sentinel-1 EW GRD.

This script:
  1. Loads one week of EW data from Planetary Computer
  2. Renders a side-by-side: raw DN→dB (left) vs calibrated σ0 dB (right)

The calibration follows ESA's official formula:
  σ0 = (DN² - noiseLut) / sigmaNoughtLut²
  σ0_dB = 10 * log10(σ0)

Both the calibration and noise LUTs are provided per-item as XML assets
in the STAC catalog.
"""

import gc
import warnings
import io
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import odc.stac
import planetary_computer as pc
from odc.geo.geobox import GeoBox
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter

from stac_utils import get_catalog, search_s1_grd


# ---------------------------------------------------------------------------
# Region config (same as animate script)
# ---------------------------------------------------------------------------
REGION = {
    "bbox": [-180, 60, 180, 90],
    "crs": "EPSG:3413",
    "resolution": 5000,
    "grid_bbox": (-3850000, -3850000, 3850000, 3850000),
    "cartopy_proj": ccrs.NorthPolarStereo(
        central_longitude=-45, true_scale_latitude=70),
}


LON_CHUNK = 60

# Sentinel-1 orbit inclination → geographic heading (degrees CW from north)
# ascending ≈ 348°, descending ≈ 192°  (≈ 12° off polar)
_S1_HEADING_ASC  = 348.0
_S1_HEADING_DESC = 192.0

def _bbox_chunks(bbox):
    west, south, east, north = bbox
    chunks = []
    lon = west
    while lon < east:
        chunks.append([lon, south, min(lon + LON_CHUNK, east), north])
        lon += LON_CHUNK
    return chunks


def _get_geobox():
    return GeoBox.from_bbox(
        bbox=REGION["grid_bbox"],
        crs=REGION["crs"],
        resolution=REGION["resolution"],
    )


# ---------------------------------------------------------------------------
# XML LUT parsing
# ---------------------------------------------------------------------------

def _parse_lut_vectors(xml_text, vector_tag, lut_tag):
    """Parse an ESA noise or calibration XML into a list of (line, pixels, values).
    
    Returns list of dicts: [{"line": int, "pixels": [int,...], "values": [float,...]}, ...]
    """
    root = ET.fromstring(xml_text)
    vectors = []
    for vec in root.iter(vector_tag):
        line_el = vec.find("line")
        pixel_el = vec.find("pixel")
        lut_el = vec.find(lut_tag)
        if line_el is None or pixel_el is None or lut_el is None:
            continue
        line = int(line_el.text)
        pixels = [int(x) for x in pixel_el.text.split()]
        values = [float(x) for x in lut_el.text.split()]
        # Truncate to consistent length (noise LUT might have different count)
        n = min(len(pixels), len(values))
        vectors.append({
            "line": line,
            "pixels": pixels[:n],
            "values": values[:n],
        })
    return vectors


def _compute_lut_mean(vectors, power=1):
    """Compute the mean of LUT values directly from sparse vectors.

    This avoids allocating a full native-resolution grid (~960 MB per LUT)
    which is unnecessary when only the scalar mean is needed.

    Parameters
    ----------
    vectors : list of dicts from _parse_lut_vectors
    power : int
        Raise values to this power before averaging (e.g. 2 for sigma²).

    Returns
    -------
    float – mean LUT value (after applying power)
    """
    if not vectors:
        return 1.0
    all_values = []
    for v in vectors:
        all_values.extend(v["values"])
    arr = np.array(all_values, dtype=np.float64)
    if power != 1:
        arr = arr ** power
    return float(np.nanmean(arr))


def _lut_1d_range_profile(vectors):
    """Extract 1D range-dependent LUT profile by averaging across azimuth.

    The LUT varies primarily across the range (pixel) axis.  By averaging
    all azimuth lines we get a compact 1D profile that can be quickly
    interpolated onto the geocoded grid.

    Returns
    -------
    pixels : 1D float64 array – pixel positions (range sample indices)
    values : 1D float64 array – LUT values averaged across azimuth
    """
    if not vectors:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])
    min_pixels = min(len(v["pixels"]) for v in vectors)
    pixels = np.array(vectors[0]["pixels"][:min_pixels], dtype=np.float64)
    values_stack = np.array(
        [v["values"][:min_pixels] for v in vectors], dtype=np.float64)
    return pixels, np.mean(values_stack, axis=0)


def _compute_spatial_luts(item, cal_vectors, noise_vectors, geobox,
                          grid_x_crs, grid_y_crs, transformer):
    """Map range-dependent LUTs from radar coordinates onto the geocoded grid.

    Strategy
    --------
    1. Extract 1D range profiles from the sparse calibration & noise vectors.
    2. Use the item’s footprint (in lat/lon) transformed to the output CRS
       plus PCA to find the azimuth (long) and range (short) axes of the
       swath on the map.
    3. Orient the range axis using the orbit state (ascending / descending)
       and the fact that Sentinel-1 is right-looking:
       • ascending  → ground track is to the west  of the swath
       • descending → ground track is to the east of the swath
       A reference “ground track” point is projected to the output CRS to
       resolve the 180° ambiguity of the PCA eigenvector, even in polar
       stereographic projections where “east/west” are not axis-aligned.
    4. Project every output-grid pixel onto the range axis, normalise to
       [0, 1] (near-range → far-range), and interpolate the 1D LUT.

    Returns
    -------
    sigma_lut_sq : float32 (H, W) – sigmaNought² on the output grid
    noise_lut    : float32 (H, W) – noise on the output grid
    """
    # --- 1. 1D range profiles -------------------------------------------
    pix_cal, val_cal   = _lut_1d_range_profile(cal_vectors)
    pix_noise, val_noise = _lut_1d_range_profile(noise_vectors)
    max_pixel = max(pix_cal[-1], pix_noise[-1])

    # --- 2. Footprint in output CRS -------------------------------------
    coords = item.geometry["coordinates"][0]          # exterior ring
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])
    xs, ys = transformer.transform(lons, lats)
    cx, cy = np.mean(xs), np.mean(ys)

    # PCA on centred footprint vertices
    pts = np.column_stack([xs - cx, ys - cy])
    cov = pts.T @ pts / len(pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # smallest eigenvalue → range (short axis), largest → azimuth (long)
    azimuth_axis = eigvecs[:, 1]

    # --- 3. Orient range direction (near → far) -------------------------
    pass_dir = item.properties.get("sat:orbit_state", "ascending")
    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))

    # Reference “ground track” point: asc → west, desc → east of centre
    gt_lon = centroid_lon + (-3.0 if pass_dir == "ascending" else 3.0)
    gt_x, gt_y = transformer.transform(gt_lon, centroid_lat)
    gt_vec = np.array([gt_x - cx, gt_y - cy])          # centre → ground track

    # Perpendicular to azimuth (90° clockwise)
    range_dir = np.array([azimuth_axis[1], -azimuth_axis[0]])
    # Must point AWAY from the ground track (near → far)
    if np.dot(range_dir, gt_vec) > 0:
        range_dir = -range_dir

    # --- 4. Footprint range extent --------------------------------------
    range_proj_fp = pts @ range_dir
    near_proj = float(np.min(range_proj_fp))
    far_proj  = float(np.max(range_proj_fp))
    rng_width = far_proj - near_proj

    shape = (geobox.shape[0], geobox.shape[1])
    if rng_width < 100:                         # degenerate → scalar fallback
        sig2 = float(np.mean(val_cal ** 2))
        nse  = float(np.mean(val_noise))
        return (np.full(shape, sig2, dtype=np.float32),
                np.full(shape, nse,  dtype=np.float32))

    # --- 5. Project output pixels & interpolate -------------------------
    nrange = ((grid_x_crs - cx) * range_dir[0]
              + (grid_y_crs - cy) * range_dir[1])
    nrange = np.clip((nrange - near_proj) / rng_width, 0.0, 1.0)
    pixel_idx = nrange * max_pixel

    sigma_sq = np.interp(pixel_idx, pix_cal, val_cal).astype(np.float32) ** 2
    noise    = np.interp(pixel_idx, pix_noise, val_noise).astype(np.float32)

    del nrange, pixel_idx
    return sigma_sq, noise


def _interpolate_lut_to_grid(vectors, nrows, ncols):
    """Interpolate sparse LUT vectors onto the full image grid.
    
    WARNING: This allocates an array of shape (nrows, ncols) which can be
    ~960 MB for EW GRD.  Prefer _compute_lut_mean() when only the scalar
    mean is needed.

    Parameters
    ----------
    vectors : list of dicts from _parse_lut_vectors
    nrows, ncols : image dimensions
    
    Returns
    -------
    2D numpy array (nrows, ncols) with interpolated LUT values
    """
    if not vectors:
        return np.ones((nrows, ncols), dtype=np.float32)
    
    # Build sparse grid
    lines = np.array([v["line"] for v in vectors])
    
    # Use the pixel grid from the first vector (they should all be same)
    # but handle variable-length vectors by finding the common set
    min_pixels = min(len(v["pixels"]) for v in vectors)
    pixels = np.array(vectors[0]["pixels"][:min_pixels])
    
    # Build values matrix
    values_matrix = np.zeros((len(lines), len(pixels)), dtype=np.float32)
    for i, v in enumerate(vectors):
        vals = np.array(v["values"][:min_pixels], dtype=np.float32)
        values_matrix[i, :] = vals
    
    # Create interpolator
    # Clamp edge values (extrapolation)
    interp = RegularGridInterpolator(
        (lines, pixels), values_matrix,
        method="linear", bounds_error=False, fill_value=None)
    
    # Full image grid
    row_coords = np.arange(nrows)
    col_coords = np.arange(ncols)
    rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
    pts = np.column_stack([rr.ravel(), cc.ravel()])
    
    result = interp(pts).reshape(nrows, ncols)
    del rr, cc, pts  # free temporaries immediately
    return result


def _build_coarse_lut(vectors, nrows, ncols, subsample=50):
    """Build a coarse (subsampled) 2D LUT from sparse annotation vectors."""
    coarse_lines = np.arange(0, nrows, subsample, dtype=np.float64)
    coarse_pixels = np.arange(0, ncols, subsample, dtype=np.float64)
    ch, cw = len(coarse_lines), len(coarse_pixels)

    if not vectors:
        return np.ones((ch, cw), dtype=np.float32)

    lines = np.array([v["line"] for v in vectors], dtype=np.float64)
    min_px = min(len(v["pixels"]) for v in vectors)
    px = np.array(vectors[0]["pixels"][:min_px], dtype=np.float64)
    vals = np.array([v["values"][:min_px] for v in vectors], dtype=np.float64)

    interp = RegularGridInterpolator(
        (lines, px), vals,
        method="linear", bounds_error=False, fill_value=None)

    rr, cc = np.meshgrid(coarse_lines, coarse_pixels, indexing="ij")
    pts = np.column_stack([rr.ravel(), cc.ravel()])
    return interp(pts).reshape(ch, cw).astype(np.float32)


def _warp_lut_to_geobox(coarse_lut, native_crs, native_transform,
                         subsample, geobox):
    """Reproject a coarse LUT from the item's native grid to the output geobox."""
    s = subsample
    coarse_tf = Affine(
        native_transform.a * s,
        native_transform.b * s,
        native_transform.c,
        native_transform.d * s,
        native_transform.e * s,
        native_transform.f,
    )

    dst_h, dst_w = geobox.shape[0], geobox.shape[1]
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    reproject(
        coarse_lut, dst,
        src_transform=coarse_tf,
        src_crs=native_crs,
        dst_transform=geobox.affine,
        dst_crs=RioCRS.from_user_input(str(geobox.crs)),
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _get_native_projection(item, signed_item, band):
    """Get native CRS and transform from STAC item metadata or GeoTIFF header."""
    epsg = item.properties.get("proj:epsg")
    proj_code = item.properties.get("proj:code")  # e.g. "EPSG:4326"
    shape = item.properties.get("proj:shape") or item.properties.get("s1:shape")
    tf = item.properties.get("proj:transform")

    crs = None
    if epsg:
        crs = RioCRS.from_epsg(epsg)
    elif proj_code:
        crs = RioCRS.from_user_input(proj_code)

    if crs and shape and tf:
        return (crs, Affine(*tf[:6]), shape[0], shape[1])

    # Fallback: read from the GeoTIFF header
    with rasterio.open(signed_item.assets[band].href) as src:
        return src.crs, src.transform, src.height, src.width


def calibrate_item(item, band="hh"):
    """Apply radiometric calibration + thermal noise removal to one STAC item.
    
    Returns calibrated sigma0 in linear scale, on the item's native grid.
    Uses the ESA formula:
        σ0 = (DN² - noiseLut) / sigmaNoughtLut²
    
    Parameters
    ----------
    item : pystac Item (already signed)
    band : str, "hh" or "vv"
    
    Returns
    -------
    tuple (sigma0_array, item) — sigma0 in linear power (not dB)
    """
    # Get image dimensions from item properties
    shape = item.properties.get("s1:shape") or item.properties.get("proj:shape")
    nrows, ncols = shape[0], shape[1]
    
    # Download calibration XML
    cal_key = f"schema-calibration-{band}"
    cal_url = item.assets[cal_key].href
    cal_resp = requests.get(cal_url, timeout=30)
    cal_resp.raise_for_status()
    
    # Download noise XML  
    noise_key = f"schema-noise-{band}"
    noise_url = item.assets[noise_key].href
    noise_resp = requests.get(noise_url, timeout=30)
    noise_resp.raise_for_status()
    
    # Parse calibration LUT (sigmaNought)
    cal_vectors = _parse_lut_vectors(
        cal_resp.text, "calibrationVector", "sigmaNought")
    sigma_lut = _interpolate_lut_to_grid(cal_vectors, nrows, ncols)
    
    # Parse noise LUT (noiseRangeLut)
    noise_vectors = _parse_lut_vectors(
        noise_resp.text, "noiseRangeVector", "noiseRangeLut")
    noise_lut = _interpolate_lut_to_grid(noise_vectors, nrows, ncols)
    
    return sigma_lut, noise_lut


def apply_calibration_to_mosaic(items, band, geobox, max_items=30):
    """Load items onto a common grid with proper calibration.
    
    Strategy: for a 5km mosaic grid, we load each item individually,
    calibrate in native space, then re-aggregate. However, since we're
    loading 100s of items, this is too slow. Instead, we use a simpler
    but effective approach:
    
    1. Load all raw DN onto the common grid (median mosaic) 
    2. Compute the MEDIAN calibration correction factor from a sample
       of items
    3. Apply the correction to the entire mosaic
    
    This works because the calibration LUTs are instrument-dependent
    (not scene-dependent) and vary only with range position.
    """
    # Limit items for the demo 
    items = items[:max_items]
    
    print(f"    Loading {len(items)} items onto fixed grid (raw DN)...")
    
    # Load raw DN onto fixed grid — no dask chunks (we .compute() immediately)
    ds = odc.stac.load(items, bands=[band],
                       geobox=geobox,
                       groupby="solar_day")
    
    # Get raw DN median
    raw_median = ds[band].median(dim="time").values.astype(np.float32)
    del ds
    gc.collect()
    
    # Now compute calibration for a sample of items
    # Sample items spread across the dataset
    n_sample = min(5, len(items))
    sample_indices = np.linspace(0, len(items)-1, n_sample, dtype=int)
    sample_items = [items[i] for i in sample_indices]
    
    print(f"    Computing calibration LUTs from {n_sample} sample items...")
    
    # For each sample item, compute scalar LUT means from sparse vectors
    # (avoids allocating massive native-resolution grids)
    cal_ratios = []
    
    for si, sample_item in enumerate(sample_items):
        try:
            signed_item = pc.sign(sample_item)
            
            # Parse calibration LUT vectors (sparse — small memory)
            cal_url = signed_item.assets[f"schema-calibration-{band}"].href
            cal_resp = requests.get(cal_url, timeout=30)
            cal_resp.raise_for_status()
            cal_vectors = _parse_lut_vectors(
                cal_resp.text, "calibrationVector", "sigmaNought")
            
            noise_url = signed_item.assets[f"schema-noise-{band}"].href
            noise_resp = requests.get(noise_url, timeout=30)
            noise_resp.raise_for_status()
            noise_vectors = _parse_lut_vectors(
                noise_resp.text, "noiseRangeVector", "noiseRangeLut")
            
            # Compute scalar means directly from sparse vectors (no grid alloc)
            sigma_mean2 = _compute_lut_mean(cal_vectors, power=2)
            noise_mean = _compute_lut_mean(noise_vectors, power=1)
            
            print(f"      Item {si+1}/{n_sample}: sigma²_mean={sigma_mean2:.0f}, "
                  f"noise_mean={noise_mean:.0f}")
            
            cal_ratios.append({
                "sigma_lut_mean": sigma_mean2,
                "noise_lut_mean": noise_mean,
            })
            
        except Exception as exc:
            print(f"      Item {si+1}/{n_sample}: FAILED - {exc}")
            continue
    
    if not cal_ratios:
        print("    [WARN] Could not compute calibration. Falling back to raw DN.")
        return raw_median, raw_median
    
    # Compute average calibration parameters
    avg_sigma2 = np.mean([r["sigma_lut_mean"] for r in cal_ratios])
    avg_noise = np.mean([r["noise_lut_mean"] for r in cal_ratios])
    
    print(f"    Average sigma² = {avg_sigma2:.0f}, average noise = {avg_noise:.0f}")
    
    # Apply calibration to the full mosaic
    # σ0 = (DN² - noise) / sigma²
    # Since we're working on a reprojected grid, we use the average calibration
    dn2 = raw_median ** 2
    sigma0 = (dn2 - avg_noise) / avg_sigma2
    
    # Also return raw for comparison
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_db = 10.0 * np.log10(np.where(raw_median > 0, raw_median, np.nan))
        cal_db = 10.0 * np.log10(np.where(sigma0 > 0, sigma0, np.nan))
    
    return raw_db, cal_db


# ---------------------------------------------------------------------------
# Heading / cross-track deramp helpers
# ---------------------------------------------------------------------------

def _geographic_heading_deg(item):
    """Return the satellite geographic heading (degrees CW from north)."""
    orbit = item.properties.get("sat:orbit_state", "ascending")
    return _S1_HEADING_ASC if orbit == "ascending" else _S1_HEADING_DESC


def _cross_track_unit_vector(item, transformer):
    """Return the cross-track unit vector on the output map grid.

    Sentinel-1 is right-looking: cross-track points 90° clockwise from
    the flight direction.  We compute the flight-direction vector *on
    the map* by projecting two points along the geographic heading
    from the swath centroid into the output CRS.

    Returns
    -------
    (ux, uy) : floats – unit vector in output-CRS coordinates
    """
    coords = item.geometry["coordinates"][0]
    clon = float(np.mean([c[0] for c in coords]))
    clat = float(np.mean([c[1] for c in coords]))

    heading_rad = np.deg2rad(_geographic_heading_deg(item))
    delta = 0.5  # degrees along the heading

    # Two points along the flight direction in geographic coords
    lat1 = clat - delta * np.cos(heading_rad)
    lon1 = clon - delta * np.sin(heading_rad) / max(np.cos(np.deg2rad(clat)), 0.1)
    lat2 = clat + delta * np.cos(heading_rad)
    lon2 = clon + delta * np.sin(heading_rad) / max(np.cos(np.deg2rad(clat)), 0.1)

    x1, y1 = transformer.transform(lon1, lat1)
    x2, y2 = transformer.transform(lon2, lat2)

    along_x, along_y = x2 - x1, y2 - y1
    norm = np.hypot(along_x, along_y)
    if norm < 1.0:
        return 1.0, 0.0  # degenerate fallback
    along_x /= norm
    along_y /= norm

    # 90° clockwise → cross-track (right-looking)
    return along_y, -along_x


def _deramp_image(dn, cross_x, cross_y, grid_x, grid_y, nbins=50):
    """Remove the linear cross-track brightness ramp from *dn* (linear).

    Parameters
    ----------
    dn : float32 (H, W) – raw digital numbers (linear, >0 where valid)
    cross_x, cross_y : cross-track unit vector on the map grid
    grid_x, grid_y : float32 (H, W) – output-grid CRS coordinates
    nbins : int – number of cross-track bins for the trend fit

    Returns
    -------
    dn_deramped : float32 (H, W) – multiplicatively corrected DN
    """
    valid = dn > 0
    if valid.sum() < nbins * 5:
        return dn  # too few pixels – return unchanged

    # Project every pixel onto the cross-track axis
    ct = grid_x * cross_x + grid_y * cross_y  # scalar projection

    ct_valid = ct[valid]
    dn_valid = dn[valid]

    ct_min, ct_max = float(ct_valid.min()), float(ct_valid.max())
    ct_range = ct_max - ct_min
    if ct_range < 1.0:
        return dn

    # Bin by cross-track position and compute median power in each bin
    edges = np.linspace(ct_min, ct_max, nbins + 1)
    bin_idx = np.digitize(ct_valid, edges) - 1
    bin_idx = np.clip(bin_idx, 0, nbins - 1)

    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    bin_medians = np.full(nbins, np.nan, dtype=np.float64)
    for b in range(nbins):
        sel = dn_valid[bin_idx == b]
        if len(sel) > 20:
            bin_medians[b] = np.median(sel.astype(np.float64) ** 2)

    good = np.isfinite(bin_medians)
    if good.sum() < 3:
        return dn

    # Fit linear trend to median(DN²) vs cross-track position
    coeffs = np.polyfit(bin_centres[good], bin_medians[good], 1)
    trend = np.polyval(coeffs, ct).astype(np.float64)

    # Normalise trend so mean correction = 1  (preserves overall level)
    trend_mean = np.mean(trend[valid])
    if trend_mean <= 0:
        return dn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correction = np.sqrt(np.clip(trend / trend_mean, 0, None)).astype(np.float32)
    correction[correction <= 0] = 1.0

    out = dn.copy()
    out[valid] = dn[valid] / correction[valid]
    return out


def apply_deramp_per_item(items, band, geobox, max_items=30):
    """Load each item, deramp the cross-track brightness gradient, mosaic.

    For each geocoded item we:
    1. Compute the satellite heading → cross-track direction on the map
    2. Fit a linear trend to median(DN²) along the cross-track axis
    3. Remove the trend multiplicatively (preserving the overall mean)
    4. Accumulate the deramped DN into a running mean mosaic
    """
    items = items[:max_items]
    print(f"    Per-item deramp for {len(items)} items...")

    geobox_shape = (geobox.shape[0], geobox.shape[1])
    cal_sum   = np.zeros(geobox_shape, dtype=np.float64)
    cal_count = np.zeros(geobox_shape, dtype=np.int16)
    raw_sum   = np.zeros(geobox_shape, dtype=np.float64)
    raw_count = np.zeros(geobox_shape, dtype=np.int16)

    # Precompute output-grid CRS coordinates (reused for every item)
    transform = geobox.affine
    cols, rows = np.meshgrid(
        np.arange(geobox_shape[1], dtype=np.float32),
        np.arange(geobox_shape[0], dtype=np.float32))
    grid_x = (transform.a * cols + transform.c).astype(np.float32)
    grid_y = (transform.e * rows + transform.f).astype(np.float32)
    del cols, rows

    crs_transformer = Transformer.from_crs(
        "EPSG:4326", str(geobox.crs), always_xy=True)

    for i, item in enumerate(items):
        try:
            signed = pc.sign(item)

            # Load raw DN onto the common geocoded grid
            ds = odc.stac.load([signed], bands=[band], geobox=geobox)
            dn = ds[band].isel(time=0).values.astype(np.float32)
            del ds

            # Determine cross-track direction for this item
            cx, cy = _cross_track_unit_vector(item, crs_transformer)

            # Deramp: remove the linear cross-track brightness gradient
            dn_deramped = _deramp_image(dn, cx, cy, grid_x, grid_y)

            # Accumulate
            valid = dn > 0
            raw_sum[valid] += dn[valid]
            raw_count[valid] += 1

            valid_d = dn_deramped > 0
            cal_sum[valid_d] += dn_deramped[valid_d]
            cal_count[valid_d] += 1

            del dn, dn_deramped

            if (i + 1) % 10 == 0 or i == 0 or i == len(items) - 1:
                print(f"      Processed {i+1}/{len(items)} items")
                gc.collect()

        except Exception as exc:
            if (i + 1) % 10 == 0:
                print(f"      [{i+1}/{len(items)}] skip: {exc}")
            continue

    # Compute mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_mean = np.where(raw_count > 0, raw_sum / raw_count, np.nan)
        cal_mean = np.where(cal_count > 0, cal_sum / cal_count, np.nan)

        raw_db = 10.0 * np.log10(np.where(raw_mean > 0, raw_mean, np.nan))
        cal_db = 10.0 * np.log10(np.where(cal_mean > 0, cal_mean, np.nan))

    return raw_db, cal_db


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_comparison(raw_db, cal_db, title_prefix, out_path,
                      vis_range_width=15):
    """Side-by-side comparison: raw DN² dB vs calibrated σ0 dB.
    
    Both panels use the same color-bar width (in dB) centered on their
    respective medians so the visual contrast / stretch is identical.
    """
    proj = REGION["cartopy_proj"]
    data_extent = REGION["grid_bbox"]
    img_extent = [data_extent[0], data_extent[2],
                  data_extent[1], data_extent[3]]
    
    # Compute per-panel range: same width, centered on each median
    raw_med = np.nanmedian(raw_db)
    cal_med = np.nanmedian(cal_db)
    half = vis_range_width / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                             subplot_kw={"projection": proj},
                             dpi=100)
    
    for ax, arr, title, vmin, vmax, label in [
        (axes[0], raw_db, f"{title_prefix}\n10·log₁₀(DN²)  [BEFORE]",
         raw_med - half, raw_med + half, "10·log₁₀(DN²) [dB]"),
        (axes[1], cal_db, f"{title_prefix}\n10·log₁₀(DN²)  [AFTER cross-track deramp]",
         cal_med - half, cal_med + half, "10·log₁₀(DN²) deramped [dB]"),
    ]:
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_bad(color="black")
        masked = np.ma.masked_invalid(arr)
        
        ax.set_xlim(data_extent[0], data_extent[2])
        ax.set_ylim(data_extent[1], data_extent[3])
        
        im = ax.imshow(masked, origin="upper",
                       extent=img_extent,
                       transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white",
                       linestyle="--")
        ax.add_feature(cfeature.LAND,
                       facecolor=(0.15, 0.15, 0.15, 0.3), edgecolor="none")
        gl = ax.gridlines(draw_labels=False, linewidth=0.3,
                          color="white", alpha=0.4, linestyle=":")
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, 10))
        
        ax.set_title(title, fontsize=12, fontweight="bold", color="white",
                     pad=12, loc="left")
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(0.6)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05, orientation="horizontal")
        cbar.set_label(label, color="white", fontsize=10)
        cbar.ax.tick_params(colors="white", labelsize=8)
    
    fig.patch.set_facecolor("black")
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, facecolor="black", edgecolor="none",
                bbox_inches="tight", dpi=150, pad_inches=0.2)
    plt.close(fig)
    print(f"  [SAVED] {out_path}")


# ---------------------------------------------------------------------------
# Mosaic-level deramp  (works on already-cached dB mosaics)
# ---------------------------------------------------------------------------

def _deramp_mosaic_db(arr_db, kernel_size=151):
    """Remove large-scale brightness gradient from a dB mosaic.

    Estimates the smooth background field via a large box filter on valid
    pixels, then subtracts it and re-centres on the global median.
    The result keeps the fine-scale geophysical signal but removes the
    systematic instrument-gain ramp baked into the mosaic.

    Parameters
    ----------
    arr_db : float (H, W) – backscatter mosaic in dB (NaN = no-data)
    kernel_size : int – box-filter window (pixels); should be >> swath width
                  on the geocoded grid.  At 5 km resolution a 151-pixel
                  kernel ≈ 755 km – larger than the ~400 km EW swath.

    Returns
    -------
    deramped_db : float (H, W) – deramped backscatter in dB
    """
    valid = np.isfinite(arr_db)
    if valid.sum() < 1000:
        return arr_db.copy()

    # Fill NaN with median so the convolution kernel doesn't spread NaN
    med = float(np.nanmedian(arr_db))
    filled = np.where(valid, arr_db, med).astype(np.float64)

    # Smooth background estimate
    bg = uniform_filter(filled, size=kernel_size, mode="nearest")

    # Also smooth a binary mask to know how much each output pixel was
    # influenced by actual data vs filled values
    weight = uniform_filter(valid.astype(np.float64),
                            size=kernel_size, mode="nearest")

    # Where weight is too low the estimate is unreliable → keep original
    reliable = weight > 0.3

    deramped = arr_db.copy()
    ok = valid & reliable
    deramped[ok] = arr_db[ok] - bg[ok] + med
    return deramped


# ---------------------------------------------------------------------------
# Cached-data comparison  (before / after deramp for every cached week)
# ---------------------------------------------------------------------------

IMG_SIZE = 1024

def _render_single_frame(arr_db, proj, data_extent, title,
                         cmap_name="plasma", vis_min=15, vis_max=35,
                         show_boundaries=True):
    """Render one polar map frame as a PIL Image."""
    img_extent = [data_extent[0], data_extent[2],
                  data_extent[1], data_extent[3]]
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")

    fig = plt.figure(figsize=(10, 10), dpi=round(IMG_SIZE / 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_xlim(data_extent[0], data_extent[2])
    ax.set_ylim(data_extent[1], data_extent[3])

    ax.imshow(np.ma.masked_invalid(arr_db), origin="upper",
              extent=img_extent, transform=proj,
              cmap=cmap, vmin=vis_min, vmax=vis_max,
              interpolation="nearest")

    if show_boundaries:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="white",
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


def main_cached():
    """Apply deramp to all cached weekly mosaics and render comparison."""
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    cache_dir = os.path.join(out_dir, "cache", "arctic")
    manifest_path = os.path.join(cache_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        print("[ERROR] No cached data found. Run animate_polar_weekly.py --fetch first.")
        return

    manifest = json.loads(Path(manifest_path).read_text())
    weeks = manifest.get("weeks", [])
    if not weeks:
        print("[ERROR] Manifest has no weeks.")
        return

    proj = REGION["cartopy_proj"]
    data_extent = REGION["grid_bbox"]

    # Compute a global background from the temporal mean of all weeks
    # (the orbit-driven ramp pattern is nearly identical every week)
    print(f"Computing temporal-mean background from {len(weeks)} weeks...")
    stack = []
    for wk in weeks:
        arr = np.load(os.path.join(cache_dir, wk["file"]))
        stack.append(arr)
    stack = np.array(stack, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        temporal_mean = np.nanmean(stack, axis=0)
    del stack

    # Deramp the temporal mean → gives us the smooth ramp field
    bg = _deramp_mosaic_db(temporal_mean, kernel_size=151)
    ramp_field = temporal_mean - bg  # positive where original is brighter than smooth
    # ramp_field is in dB — we'll subtract it from each week
    # Actually simpler: just pass each week through _deramp_mosaic_db individually
    # but using a shared ramp_field is more consistent across weeks.
    # Let's use the shared field approach:
    ramp_correction = temporal_mean - bg  # dB offset at each pixel
    del bg, temporal_mean

    # Vis range (same for raw and deramped so they're directly comparable)
    vis_min, vis_max = 15, 35

    frames_raw = []
    frames_deramped = []
    comp_dir = os.path.join(out_dir, "comparison_frames")
    os.makedirs(comp_dir, exist_ok=True)

    print(f"\nRendering {len(weeks)} weeks (raw + deramped)...")

    for i, wk in enumerate(weeks, 1):
        arr_db = np.load(os.path.join(cache_dir, wk["file"]))
        deramped_db = arr_db - ramp_correction
        # Re-centre on the original median so the overall brightness matches
        med_shift = np.nanmedian(arr_db) - np.nanmedian(deramped_db)
        deramped_db += med_shift

        title_raw = f"Arctic  |  {wk['start']} → {wk['end']}  [RAW]"
        title_der = f"Arctic  |  {wk['start']} → {wk['end']}  [DERAMPED]"

        img_raw = _render_single_frame(arr_db, proj, data_extent, title_raw,
                                       vis_min=vis_min, vis_max=vis_max)
        img_der = _render_single_frame(deramped_db, proj, data_extent, title_der,
                                       vis_min=vis_min, vis_max=vis_max)

        # Side-by-side composite
        w, h = img_raw.size
        combo = Image.new("RGB", (w * 2, h), "black")
        combo.paste(img_raw, (0, 0))
        combo.paste(img_der, (w, 0))
        combo.save(os.path.join(comp_dir,
                                f"compare_{wk['index']:03d}_{wk['start']}.png"))

        frames_raw.append(img_raw)
        frames_deramped.append(img_der)

        if i % 10 == 0 or i == len(weeks):
            print(f"  Rendered {i}/{len(weeks)}")

    # Save animated GIFs
    fps = 4
    duration = int(1000 / fps)

    gif_raw = os.path.join(out_dir, "arctic_weekly_amplitude_raw.gif")
    frames_raw[0].save(gif_raw, save_all=True, append_images=frames_raw[1:],
                       duration=duration, loop=0)
    print(f"\n[GIF] Raw:      {gif_raw}  ({len(frames_raw)} frames)")

    gif_der = os.path.join(out_dir, "arctic_weekly_amplitude_deramped.gif")
    frames_deramped[0].save(gif_der, save_all=True,
                            append_images=frames_deramped[1:],
                            duration=duration, loop=0)
    print(f"[GIF] Deramped: {gif_der}  ({len(frames_deramped)} frames)")

    # Side-by-side GIF
    combos = []
    for fr, fd in zip(frames_raw, frames_deramped):
        w, h = fr.size
        c = Image.new("RGB", (w * 2, h), "black")
        c.paste(fr, (0, 0))
        c.paste(fd, (w, 0))
        combos.append(c)
    gif_comp = os.path.join(out_dir, "arctic_weekly_amplitude_comparison.gif")
    combos[0].save(gif_comp, save_all=True, append_images=combos[1:],
                   duration=duration, loop=0)
    print(f"[GIF] Comparison: {gif_comp}  ({len(combos)} frames)")

    print("\n[DONE]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    
    # Use a single week for comparison
    start = "2025-03-12"
    end = "2025-03-19"
    band = "hh"
    
    print("=" * 70)
    print("  Calibration Comparison: raw DN→dB vs calibrated σ⁰ dB")
    print("  Week:", start, "→", end)
    print("=" * 70)
    
    # Try to reload cached arrays first (skip expensive recompute)
    raw_cache = os.path.join(out_dir, "comparison_raw_db.npy")
    cal_cache = os.path.join(out_dir, "comparison_cal_db.npy")
    
    # Force recompute (spatial LUT calibration changed)
    for f in [raw_cache, cal_cache]:
        if os.path.exists(f):
            os.remove(f)

    if os.path.exists(raw_cache) and os.path.exists(cal_cache) and "--recompute" not in sys.argv:
        print("\n[1-2] Loading cached arrays (use --recompute to force refresh)...")
        raw_db = np.load(raw_cache)
        cal_db = np.load(cal_cache)
    else:
        # Search for EW items
        print("\n[1] Searching for Sentinel-1 EW items...")
        catalog = get_catalog()
        geobox = _get_geobox()
        
        all_items = []
        for chunk in _bbox_chunks(REGION["bbox"]):
            all_items.extend(
                search_s1_grd(catalog, chunk, f"{start}/{end}",
                              max_items=600, mode="EW"))
        
        seen = set()
        unique = [it for it in all_items if it.id not in seen and not seen.add(it.id)]
        items = [it for it in unique if "hh" in it.assets]
        print(f"    Found {len(items)} EW items with HH band")
        
        if not items:
            print("  [ERROR] No items found!")
            return
        
        # --- Method: per-item cross-track deramp ---
        print("\n[2] Applying per-item cross-track deramp...")
        raw_db, cal_db = apply_deramp_per_item(items, band, geobox, max_items=30)
        
        # Save the numpy arrays for reuse
        np.save(raw_cache, raw_db)
        np.save(cal_cache, cal_db)
    
    # Print stats  
    print(f"\n[3] Statistics:")
    print(f"    Raw 10*log10(DN):    min={np.nanmin(raw_db):.1f}  max={np.nanmax(raw_db):.1f}  "
          f"median={np.nanmedian(raw_db):.1f}")
    print(f"    Calibrated σ⁰ dB:   min={np.nanmin(cal_db):.1f}  max={np.nanmax(cal_db):.1f}  "
          f"median={np.nanmedian(cal_db):.1f}")
    
    # Render comparison
    print("\n[4] Rendering comparison...")
    
    # Convert raw from 10*log10(DN) to 10*log10(DN²) = 20*log10(DN)
    # so it is in the same power-dB units as calibrated σ⁰ dB
    raw_db = 2.0 * raw_db
    
    raw_p5, raw_p95 = np.nanpercentile(raw_db, [5, 95])
    cal_p5, cal_p95 = np.nanpercentile(cal_db, [5, 95])
    print(f"    Raw  5th-95th percentile: [{raw_p5:.1f}, {raw_p95:.1f}]  (width {raw_p95-raw_p5:.1f} dB)")
    print(f"    Cal  5th-95th percentile: [{cal_p5:.1f}, {cal_p95:.1f}]  (width {cal_p95-cal_p5:.1f} dB)")
    
    # Use the wider of the two 5-95th ranges + margin as the shared color width
    shared_width = max(raw_p95 - raw_p5, cal_p95 - cal_p5) + 2
    print(f"    Shared color range width: {shared_width:.0f} dB")
    
    out_path = os.path.join(out_dir, "calibration_comparison.png")
    render_comparison(raw_db, cal_db,
                      f"Arctic EW HH  |  {start} → {end}",
                      out_path,
                      vis_range_width=shared_width)
    
    print(f"\n[DONE] See {out_path}")


if __name__ == "__main__":
    if "--cached" in sys.argv:
        main_cached()
    else:
        main()
