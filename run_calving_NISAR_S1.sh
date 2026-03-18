#!/usr/bin/env bash
# ==========================================================================
#  NISAR GSLC Calving-Site Metadata Catalog Pipeline
# ==========================================================================
#
#  Queries the ASF DAAC for NISAR L2 GSLC product metadata over the same
#  calving-site bounding boxes used by the Sentinel-1 pipeline.  No data
#  is downloaded — only metadata is collected and analysed.
#
#  This pipeline builds day-by-day coverage catalogs tracking:
#    - Path (track) and frame numbers
#    - Flight direction (ascending / descending)
#    - Range bandwidth (MHz)  — e.g. 5 MHz (diagnostic) vs 77 MHz (science)
#    - CRID version (Composite Release ID)
#    - Frame coverage (Full / Partial)
#    - PGE software version
#    - Main-band polarisation
#    - Production configuration and collection version
#
#  Prerequisites:
#    • Python 3.12 venv at .venv/ with deps from requirements-lock.txt
#    • asf_search (pip install asf_search)
#    • No authentication required for metadata queries
#
#  Usage:
#    source .venv/bin/activate
#    bash run_calving_NISAR_S1.sh
#
#  SITES (same bounding boxes as Sentinel-1 pipeline):
#    Greenland  : jakobshavn, petermann, 79north
#    Antarctica : pine_island, thwaites
#
#  OUTPUT STRUCTURE:
#    output/nisar_s1/
#      catalogs/<site>_nisar_catalog.json  — machine-readable metadata catalog
#      catalogs/<site>_nisar_catalog.txt   — human-readable day-by-day table
#      catalogs/<site>_coverage_report.*   — S1+NISAR coverage comparison
#      cache/<site>/                       — cached S1+NISAR .npy arrays
#      animations/<site>/                  — final comparison GIFs
#
# --------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# Start with Jakobshavn as the primary test site
SITES="jakobshavn"
START="2024-06-01"
END="2026-03-31"

# ==========================================================================
#  STEP 1 — Build NISAR GSLC day-by-day coverage catalog
# ==========================================================================
#  Queries ASF DAAC via asf_search for all NISAR L2 GSLC products that
#  intersect each site's bounding box within the date range.
#
#  For each granule, extracts metadata from the ASF search results
#  (no HDF5 files are downloaded).  The catalog groups granules by
#  (date, path, direction) to show which track/frame combinations are
#  available on each day.

echo ""
echo "===== Building NISAR GSLC coverage catalogs ====="
for site in $SITES; do
    echo "  → $site"
    python nisar_catalog.py "$site" \
        --start "$START" --end "$END"
done

echo ""
echo "===== CATALOG COMPLETE ====="
echo ""

# ==========================================================================
#  STEP 2 — S1 GRD + NISAR GSLC day-by-day coverage comparison
# ==========================================================================
#  For the NISAR date range, queries Planetary Computer for Sentinel-1 GRD
#  products and builds a combined day-by-day availability table.
#
#  This does NOT modify any calving pipeline outputs.

echo "===== Building S1 + NISAR coverage reports ====="
for site in $SITES; do
    echo "  → $site"
    python coverage_nisar_s1.py "$site"
done

# ==========================================================================
#  STEP 3 — Fetch paired S1 + NISAR amplitude data
# ==========================================================================
#  For days with both S1 GRD and NISAR GSLC coverage (≥20% frame overlap),
#  fetches:
#    - NISAR: Multi-looked amplitude from remote HDF5 via earthaccess
#    - S1:    Amplitude from Planetary Computer STAC
#
#  Both resampled to 100 m, stored as dB in .npy format.
#  Metadata (URLs, orbits, coverage, etc.) recorded in manifest.json.

echo ""
echo "===== Fetching paired S1 + NISAR data ====="
for site in $SITES; do
    echo "  → $site"
    python fetch_paired.py "$site"
done

# ==========================================================================
#  STEP 4 — Create S1 vs NISAR comparison animations
# ==========================================================================
#  4a) Side-by-side timeline animation:
#      - Left:   Sentinel-1 (C-band) with scale bar + colorbar
#      - Right:  NISAR (L-band) with colorbar
#      - Center: 3D globe with location marker (NearsidePerspective)
#      - Bottom: Timeline slider with month labels
#      - Top:    Glacier name title
#      Layout: fig 14×5.8 in, globe 10% smaller, centered
#
#  4b) Slider reveal animation (single date):
#      - Interactive slider revealing S1 ↔ NISAR
#
#  4c) Combined visualization:
#      - RGB composite, difference highlighting, rapid flicker
#
#  All compressed with gifsicle (lossy=30-40).
#
#  Default date for slider/combined: latest available in manifest

echo ""
echo "===== Creating S1/NISAR comparison animations ====="
for site in $SITES; do
    echo "  → $site (timeline)"
    python animate_paired.py --site "$site" --fps 1
    
    echo "  → $site (slider reveal - latest date)"
    python animate_slider.py --site "$site"
    
    echo "  → $site (combined viz - latest date)"
    python animate_combined.py --site "$site"
done

echo ""
echo "===== ALL COMPLETE ====="
echo ""
echo "Outputs:"
echo "  Catalogs:    output/nisar_s1/catalogs/*.txt, *.json"
echo "  Paired data: output/nisar_s1/cache/<site>/*.npy, manifest.json"
echo "  Animations:  output/nisar_s1/cache/<site>/*_comparison.gif"
echo "               output/nisar_s1/cache/<site>/*_slider.gif"
echo "               output/nisar_s1/cache/<site>/*_combined.gif"
echo ""
ls -lh output/nisar_s1/catalogs/*.txt output/nisar_s1/cache/*/manifest.json output/nisar_s1/cache/*/*.gif 2>/dev/null || true
