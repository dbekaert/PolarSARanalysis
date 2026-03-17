#!/usr/bin/env bash
# ==========================================================================
#  Calving-site time-lapse pipeline
# ==========================================================================
#
#  End-to-end workflow for producing monthly Sentinel-1 SAR animations
#  of glacier calving fronts.  Each step is independent and can be re-run
#  in isolation.  The pipeline uses two main scripts:
#
#    calving_sites.py     – fetch SAR data and render animations
#    coverage_catalog.py  – build per-track, per-day coverage reports
#
#  Prerequisites:
#    • Python 3.12 venv at .venv/ with deps from requirements.txt
#    • Microsoft Planetary Computer STAC access (no key needed)
#    • gifsicle (optional) – for post-assembly GIF compression
#        sudo apt install gifsicle   # or brew install gifsicle
#
# --------------------------------------------------------------------------
#  HOW TO USE
#    1. Activate the venv:   source .venv/bin/activate
#    2. Run individual steps below, or execute this entire script:
#         bash run_calving_pipeline.sh
#
#  SITES
#    Greenland  : jakobshavn, petermann, 79north
#    Antarctica : pine_island, thwaites
#    (Others defined but not yet animated: ellesmere, larsen_c, ross)
#
#  OUTPUT STRUCTURE
#    output/
#      <site>_catalog.json      – machine-readable coverage catalog
#      <site>_catalog.txt       – human-readable table (◄ = selected)
#      calving_cache/<site>/    – cached .npy SAR arrays + manifest
#      calving_frames/<site>/   – per-site rendered PNG frames
#      calving_frames/combined/ – composite PNG frames (all sites)
#      calving_animations/      – final animated GIFs
#
#  RENDERING DETAILS
#    • SAR data fetched at native 100 m, rendered at 200 m (area-mean
#      downsampling in 2×2 blocks) to cut memory ~4×
#    • Figure context is reused across frames (avoids re-creating
#      cartopy projections/coastlines each frame)
#    • GIFs assembled by streaming PNGs from disk (two frames in
#      memory at a time)
#    • Individual GIFs compressed with gifsicle --lossy=80
#    • Combined GIF uses a global 256-colour palette (no dithering)
#      and light compression (gifsicle --lossy=30) to keep text sharp
#    • Each frame includes "Image analysis by David Bekaert"
#      attribution strip at the bottom
#
#  COMBINED GRID LAYOUT (5 sites)
#    ┌───────────────────────┐
#    │      Jakobshavn       │
#    ├───────────┬───────────┤
#    │ Petermann │ 79° North │
#    ├───────────┼───────────┤
#    │Pine Island│ Thwaites  │
#    └───────────┴───────────┘
#      + shared timeline bar on top
#      + attribution strip on bottom
# --------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

SITES="jakobshavn petermann 79north pine_island thwaites"
START="2015-01-01"
END="2026-03-31"

# Suppress noisy RuntimeWarnings from nanmean on all-NaN slices
export PYTHONWARNINGS="ignore::RuntimeWarning"

# ==========================================================================
#  STEP 1 — Define / tune the spatial crop for each site
# ==========================================================================
#  Each site's extent is defined in calving_sites.py → SITES dict:
#    center_lon, center_lat    – geographic center of the AOI
#    half_width_km              – half-width of the bounding box (km)
#    half_height_km             – half-height (defaults to half_width_km)
#
#  To tune a crop:
#    • Edit SITES[<key>] in calving_sites.py
#    • Fetch + render a single month to inspect the frame:
#        python calving_sites.py --site jakobshavn \
#            --start_date 2020-06-01 --end_date 2020-06-30
#    • Check output/calving_frames/<site>/<site>_2020-06.png
#    • Repeat until the crop captures the calving front + context
#
#  Current extents:
#    Jakobshavn  : 100 × 60 km   (center -50.20°E, 69.17°N)
#    Petermann   : 100 × 100 km  (center -60.87°E, 81.06°N)
#    79° North   : 160 × 160 km  (center -19.32°E, 79.68°N)
#    Pine Island :  100 × 100 km (center -101.79°E, -75.07°N)
#    Thwaites    : 160 × 160 km  (center -107.00°E, -75.50°N)

# ==========================================================================
#  STEP 2 — Build day-by-day coverage catalog for each site
# ==========================================================================
#  Scans the Planetary Computer STAC for every Sentinel-1 IW GRD pass
#  over each site, for every month in the date range.  Records per-track,
#  per-day metadata: coverage %, platforms, orbit direction, relative
#  orbit (track number), timeliness, and polarisation.
#
#  Outputs per site:
#    output/<site>_catalog.json  – machine-readable (used by fetch)
#    output/<site>_catalog.txt   – human-readable table with ◄ markers

echo ""
echo "===== STEP 2: Building coverage catalogs ====="
for site in $SITES; do
    echo "  → $site"
    python coverage_catalog.py "$site" \
        --start "$START" --end "$END"
done

# ==========================================================================
#  STEP 3 — Identify the optimal acquisition per month
# ==========================================================================
#  This is performed automatically inside coverage_catalog.py (Step 2).
#  The selection algorithm picks one day + track per month using:
#
#    Priority (highest → lowest):
#      1. Full-coverage single-track days
#      2. Preferred orbit direction (descending for Greenland sites)
#      3. Preferred relative orbit / track number
#      4. Fast-24h timeliness over NRT-3h
#      5. Highest spatial coverage %
#
#  The selected date+track are stored in the catalog JSON and are
#  consumed by the fetch step to ensure reproducible data pulls.
#
#  To inspect selections:
#    cat output/<site>_catalog.txt | grep '◄'

# ==========================================================================
#  STEP 4 — Fetch SAR data & render individual site GIFs
# ==========================================================================
#  Fetches the catalog-selected acquisition for each month, loads via
#  odc.stac, converts to calibrated dB, and caches as .npy arrays.
#  Then renders each site into an animated GIF.
#
#  Key parameters:
#    --render_res 200   (default) render at 200 m via 2× area-mean
#    --fps 4            frames per second in the GIF
#    --vis_min 15       dB floor for grayscale mapping
#    --vis_max 35       dB ceiling for grayscale mapping
#    --sync             align all sites to a shared timeline
#
#  Outputs:
#    output/calving_cache/<site>/manifest.json   – snapshot metadata
#    output/calving_cache/<site>/<month>.npy      – cached SAR arrays
#    output/calving_frames/<site>/<site>_<month>.png
#    output/calving_animations/<site>_timelapse.gif (compressed)

echo ""
echo "===== STEP 4: Fetching data & rendering individual GIFs ====="
for site in $SITES; do
    echo "  → Fetching $site"
    python calving_sites.py --fetch --site "$site" \
        --start_date "$START" --end_date "$END"
done

echo ""
echo "  → Rendering individual GIFs (synced timeline, fps=4)"
python calving_sites.py --render --sync --site $SITES --fps 4

# ==========================================================================
#  STEP 5 — Render combined multi-panel GIF
# ==========================================================================
#  Produces a single GIF with all sites arranged in a grid layout
#  sharing one timeline bar at the top.
#
#  Layout for 5 sites (hard-coded in calving_sites.py → _default_grid):
#    Row 1: Jakobshavn (centered, full width)
#    Row 2: Petermann | 79° North
#    Row 3: Pine Island | Thwaites
#
#  Each panel preserves its own spatial scale (m/pixel), globe inset,
#  title, scale bar, and sensor/orbit metadata.  Shorter panels are
#  bottom-padded with black to equalise row heights.
#
#  Two-pass rendering:
#    Pass 1 – renders each site's panels as individual PNGs
#    Pass 2 – composites panels into combined frames with timeline
#             and attribution strip, saves to disk
#  GIF assembly streams PNGs from disk to keep memory low.
#  Uses a global palette (no dithering) for clean text in the GIF.
#
#  Output:
#    output/calving_frames/combined/combined_<month>.png
#    output/calving_animations/combined_<sites>_timelapse.gif

echo ""
echo "===== STEP 5: Rendering combined multi-panel GIF ====="
python calving_sites.py --render --combined --site $SITES --fps 4

echo ""
echo "===== PIPELINE COMPLETE ====="
echo "Outputs in: output/calving_animations/"
ls -lh output/calving_animations/*.gif 2>/dev/null || true
