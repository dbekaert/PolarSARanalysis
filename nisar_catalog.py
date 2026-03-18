#!/usr/bin/env python3
"""Day-by-day NISAR GSLC coverage catalog for calving sites.

Queries the ASF DAAC via asf_search for NISAR L2 GSLC products over the
same bounding boxes used by the Sentinel-1 calving pipeline.  No data is
downloaded — only metadata is collected and analysed.

For each site the script builds a day-level catalog tracking:
  - Path (track) and frame numbers
  - Flight direction (ascending / descending)
  - Range bandwidth (MHz)
  - CRID version (Composite Release ID)
  - Frame coverage (Full / Partial)
  - PGE software version
  - Main-band polarisation
  - Production configuration (PR, UR, …)
  - Collection name (Beta / Provisional / Validated)

Outputs per site:
  output/nisar/<site>_nisar_catalog.json  — machine-readable
  output/nisar/<site>_nisar_catalog.txt   — human-readable table

Usage:
    python nisar_catalog.py jakobshavn
    python nisar_catalog.py jakobshavn --start 2025-10-01 --end 2026-03-31
    python nisar_catalog.py all

Requires:
    pip install asf_search shapely
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import asf_search as asf
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from calving_sites import SITES, _search_bbox_lonlat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_START = "2024-06-01"
DEFAULT_END = "2026-12-31"


def _site_wkt(site_key):
    """Return a WKT polygon for the site's lon/lat search bounding box."""
    w, s, e, n = _search_bbox_lonlat(site_key)
    return f"POLYGON(({w} {n},{w} {s},{e} {s},{e} {n},{w} {n}))"


def _site_aoi(site_key):
    """Return a Shapely polygon for the site AOI in lon/lat."""
    w, s, e, n = _search_bbox_lonlat(site_key)
    return Polygon([(w, n), (w, s), (e, s), (e, n)])


def _parse_results(results, aoi):
    """Parse ASF search results into flat record dicts.

    Each record corresponds to one GSLC granule (one frame on one track
    on one date).
    """
    aoi_area = aoi.area
    records = []
    for r in results:
        props = r.geojson()["properties"]
        geom = r.geojson().get("geometry")

        # Compute spatial overlap with AOI
        coverage_pct = 0.0
        if geom:
            try:
                footprint = shape(geom)
                overlap = footprint.intersection(aoi).area
                coverage_pct = round(overlap / aoi_area * 100, 1) if aoi_area else 0
            except Exception:
                pass

        bw = props.get("rangeBandwidth")
        bw_str = ",".join(str(b) for b in bw) if isinstance(bw, list) else str(bw or "?")

        # Polarization: mainBandPolarization for science (77 MHz),
        # sideBandPolarization for diagnostic (5 MHz)
        pol = props.get("mainBandPolarization")
        if not pol or pol == [None]:
            pol = props.get("sideBandPolarization")
        pol_str = ",".join(str(p) for p in pol) if isinstance(pol, list) and pol else "?"

        # Approximate resolution from range bandwidth
        # 77 MHz → ~3 m range res (science mode)
        #  5 MHz → ~30 m range res (diagnostic/calibration)
        bw_val = int(bw[0]) if isinstance(bw, list) and bw and bw[0].isdigit() else 0
        range_res_m = round(3e8 / (2 * bw_val * 1e6), 0) if bw_val > 0 else None

        records.append({
            "date": props["startTime"][:10],
            "start_time": props["startTime"],
            "path": props.get("pathNumber"),
            "frame": props.get("frameNumber"),
            "direction": props.get("flightDirection", "?"),
            "crid": props.get("crid", "?"),
            "range_bw_mhz": bw_str,
            "range_res_m": range_res_m,
            "polarization": pol_str,
            "frame_coverage": props.get("frameCoverage", "?"),
            "pge_version": props.get("pgeVersion", "?"),
            "collection": props.get("collectionName", "?"),
            "config": props.get("productionConfiguration", "?"),
            "sensor": props.get("sensor", "?"),
            "orbit": props.get("orbit"),
            "scene_name": props.get("sceneName", ""),
            "url": props.get("url", ""),
            "coverage_pct": coverage_pct,
        })
    return records


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def build_nisar_catalog(site_key, start_date=DEFAULT_START,
                        end_date=DEFAULT_END):
    """Query ASF for NISAR GSLC products over *site_key* and build catalog."""
    if site_key not in SITES:
        raise ValueError(f"Unknown site '{site_key}'. "
                         f"Available: {', '.join(SITES.keys())}")

    cfg = SITES[site_key]
    label = cfg["label"]
    wkt = _site_wkt(site_key)
    aoi = _site_aoi(site_key)

    print(f"\n[{label}] Querying ASF for NISAR GSLC products …")
    print(f"  Date range : {start_date} → {end_date}")
    print(f"  AOI (WKT)  : {wkt[:80]}…")

    results = asf.geo_search(
        dataset="NISAR",
        processingLevel="GSLC",
        intersectsWith=wkt,
        start=datetime.strptime(start_date, "%Y-%m-%d"),
        end=datetime.strptime(end_date, "%Y-%m-%d"),
    )

    print(f"  Found {len(results)} GSLC granules")

    if not results:
        print("  No data found.")
        return None

    records = _parse_results(results, aoi)

    # Sort by date, then path, then frame
    records.sort(key=lambda r: (r["date"], r["path"], r["frame"]))

    # ---- Aggregate statistics ----
    dates = sorted(set(r["date"] for r in records))
    paths = sorted(set(r["path"] for r in records))
    crids = sorted(set(r["crid"] for r in records))
    bws = sorted(set(r["range_bw_mhz"] for r in records))

    # Per-day aggregation: group frames by (date, path, direction)
    day_track_groups = defaultdict(list)
    for r in records:
        key = (r["date"], r["path"], r["direction"])
        day_track_groups[key].append(r)

    # Build day-level summary records (one per date+path+direction)
    day_summaries = []
    for (date, path, direction), frames in sorted(day_track_groups.items()):
        frame_nums = sorted(set(f["frame"] for f in frames))
        crids_day = sorted(set(f["crid"] for f in frames))
        bws_day = sorted(set(f["range_bw_mhz"] for f in frames))
        pols_day = sorted(set(f["polarization"] for f in frames))
        res_vals = [f["range_res_m"] for f in frames if f["range_res_m"]]

        # Aggregate spatial coverage (sum of frame overlaps — frames can be
        # stitched so multi-frame coverage is additive)
        coverage_pct = min(round(sum(f["coverage_pct"] for f in frames), 1),
                           100.0)

        day_summaries.append({
            "date": date,
            "path": path,
            "direction": direction,
            "n_frames": len(frames),
            "frames": frame_nums,
            "crids": crids_day,
            "range_bw_mhz": bws_day,
            "range_res_m": sorted(set(res_vals)) if res_vals else [],
            "polarization": pols_day,
            "coverage_pct": coverage_pct,
            "pge_versions": sorted(set(f["pge_version"] for f in frames)),
            "configs": sorted(set(f["config"] for f in frames)),
            "collections": sorted(set(f["collection"] for f in frames)),
        })

    # ---- Group by month ----
    month_groups = defaultdict(list)
    for ds in day_summaries:
        month_key = ds["date"][:7]  # YYYY-MM
        month_groups[month_key].append(ds)

    # ---- Path frequency (track revisit analysis) ----
    # Only count science-mode (77 MHz) for dominant track selection
    path_dates_sci = defaultdict(set)
    path_dates_all = defaultdict(set)
    for r in records:
        path_dates_all[r["path"]].add(r["date"])
        if r["range_bw_mhz"] != "5":
            path_dates_sci[r["path"]].add(r["date"])
    path_freq = {p: len(d) for p, d in sorted(path_dates_all.items())}
    path_freq_sci = {p: len(d) for p, d in sorted(path_dates_sci.items())}

    # Dominant track: most revisits at science bandwidth
    dominant_path = max(path_freq_sci, key=path_freq_sci.get) if path_freq_sci else None

    # ---- Choose a single BW + polarization mode ----
    # Count (bw, pol) combos across all day-summaries to pick the dominant
    # mode.  We never mix bandwidths or polarisations in the selection.
    mode_dates = defaultdict(set)  # (bw, pol) -> set of dates
    for ds in day_summaries:
        # Only consider single-mode entries (no mixed BW/pol within a track-day)
        if len(ds["range_bw_mhz"]) == 1 and len(ds["polarization"]) == 1:
            mode_dates[(ds["range_bw_mhz"][0], ds["polarization"][0])].add(ds["date"])
    if mode_dates:
        # Prefer science BW (non-5 MHz), then most dates
        chosen_mode = max(mode_dates,
                          key=lambda m: (0 if m[0] == "5" else 1,
                                         len(mode_dates[m])))
    else:
        chosen_mode = None
    chosen_bw, chosen_pol = chosen_mode if chosen_mode else (None, None)

    # ---- Auto-select best track per day ----
    # Only consider day-summaries matching the chosen BW+pol mode.
    # Among those, pick highest coverage, then dominant track.
    date_groups = defaultdict(list)
    for ds in day_summaries:
        date_groups[ds["date"]].append(ds)

    selected = {}  # date -> best track summary
    for date_key in sorted(date_groups):
        options = date_groups[date_key]
        if not options:
            continue

        # Filter to chosen mode (single BW + single pol, matching)
        if chosen_bw and chosen_pol:
            matching = [d for d in options
                        if len(d["range_bw_mhz"]) == 1
                        and d["range_bw_mhz"][0] == chosen_bw
                        and len(d["polarization"]) == 1
                        and d["polarization"][0] == chosen_pol]
        else:
            matching = options

        if not matching:
            continue  # no track on this date matches the chosen mode

        def _score(d):
            is_dominant = 1 if d["path"] == dominant_path else 0
            return (d["coverage_pct"], is_dominant)

        best = max(matching, key=_score)
        selected[date_key] = {
            "date": best["date"],
            "path": best["path"],
            "direction": best["direction"],
        }

    # ---- Output ----
    out_dir = os.path.join("output", "nisar_s1", "catalogs")
    os.makedirs(out_dir, exist_ok=True)

    catalog_data = {
        "site": site_key,
        "label": label,
        "sensor": "NISAR L-SAR",
        "product_level": "L2 GSLC",
        "date_range": f"{start_date} / {end_date}",
        "aoi_bbox_lonlat": list(_search_bbox_lonlat(site_key)),
        "n_granules": len(records),
        "n_unique_dates": len(dates),
        "date_range_actual": f"{dates[0]} / {dates[-1]}" if dates else "",
        "unique_paths": paths,
        "unique_crids": crids,
        "unique_range_bw_mhz": bws,
        "dominant_path": dominant_path,
        "chosen_mode": {"bw_mhz": chosen_bw, "polarization": chosen_pol},
        "path_frequency": path_freq,
        "path_frequency_science": path_freq_sci,
        "selected_tracks": selected,
        "day_summaries": day_summaries,
        "granules": records,
    }

    json_path = os.path.join(out_dir, f"{site_key}_nisar_catalog.json")
    with open(json_path, "w") as f:
        json.dump(catalog_data, f, indent=2)

    # ---- Human-readable text ----
    txt_path = os.path.join(out_dir, f"{site_key}_nisar_catalog.txt")
    with open(txt_path, "w") as f:
        def pr(s=""):
            print(s)
            f.write(s + "\n")

        pr(f"{label} — Comprehensive Day-Level Metadata Catalog")
        pr(f"Product: NISAR L2 GSLC  |  Sensor: L-SAR")
        pr(f"Date range: {start_date} → {end_date}")
        bbox = _search_bbox_lonlat(site_key)
        pr(f"AOI: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}] "
           f"(lon/lat)")
        pr(f"Dominant track (science): {dominant_path}")
        bw_label = f"{chosen_bw} MHz" if chosen_bw else "?"
        pr(f"Selected mode: {bw_label} {chosen_pol or '?'}  "
           f"({len(selected)} of {len(dates)} dates)")
        pr(f"Total granules: {len(records)}  |  Unique dates: {len(dates)}")
        if dates:
            pr(f"Data available: {dates[0]} → {dates[-1]}")
        pr()

        pr("Bandwidth → Resolution:")
        pr("  77 MHz  →  ~2 m range resolution  (science mode, HH)")
        pr("   5 MHz  →  ~30 m range resolution (diagnostic/calibration, VV)")
        pr()

        # Track frequency table
        pr("Track (Path) Revisit Frequency:")
        pr(f"  {'Path':>6}  {'#Dates':>7}  {'#Sci':>5}  {'Direction':>12}")
        pr(f"  {'─'*6}  {'─'*7}  {'─'*5}  {'─'*12}")
        for path in sorted(path_freq, key=lambda p: -path_freq[p]):
            dirs = set()
            for r in records:
                if r["path"] == path:
                    dirs.add(r["direction"])
            sci = path_freq_sci.get(path, 0)
            pr(f"  {path:>6}  {path_freq[path]:>7}  {sci:>5}  "
               f"{'/'.join(sorted(dirs)):>12}")
        pr()

        # Day-by-day table grouped by month (S1-style)
        hdr = (f"{'Month':>7}  {'Date':>10}  {'Cov%':>5}  {'Path':>5}  "
               f"{'Dir':>5}  {'#Fr':>3}  {'Frames':>10}  "
               f"{'BW':>4}  {'Pol':>3}  {'CRID':>8}  {'Sel':>3}")
        pr(hdr)
        pr("-" * len(hdr))

        for month_key in sorted(month_groups):
            days = month_groups[month_key]
            if not days:
                pr(f"{month_key:>7}  {'(no data)':>10}")
                pr()
                continue
            for d in sorted(days, key=lambda x: (x["date"], x["path"])):
                sel = selected.get(d["date"], {})
                is_sel = (" ◄" if d["path"] == sel.get("path")
                          and d["direction"] == sel.get("direction")
                          else "")
                frames_str = _compact_frames(d["frames"])
                pr(f"{month_key:>7}  {d['date']:>10}  "
                   f"{d['coverage_pct']:>5.0f}  {d['path']:>5}  "
                   f"{d['direction'][:5]:>5}  "
                   f"{d['n_frames']:>3}  {frames_str:>10}  "
                   f"{','.join(d['range_bw_mhz']):>4}  "
                   f"{','.join(d['polarization']):>3}  "
                   f"{','.join(d['crids']):>8}{is_sel}")
            pr()

        # Auto-selected summary (day-by-day)
        pr()
        pr("=" * 70)
        pr(f"AUTO-SELECTED: Best track per day  "
           f"(mode: {chosen_bw} MHz {chosen_pol})")
        pr(f"  (Ranked by: coverage, then track={dominant_path}.  "
           f"Only {chosen_bw} MHz {chosen_pol} tracks considered.)")
        pr()

        # Coverage threshold summary
        sel_coverages = []
        for dk in sorted(selected):
            sel = selected[dk]
            d = next(x for x in day_summaries
                     if x["date"] == sel["date"]
                     and x["path"] == sel["path"]
                     and x["direction"] == sel["direction"])
            sel_coverages.append(d["coverage_pct"])

        n_sel = len(sel_coverages)
        n_gt75 = sum(1 for c in sel_coverages if c >= 75)
        n_gt50 = sum(1 for c in sel_coverages if c >= 50)
        n_gt25 = sum(1 for c in sel_coverages if c >= 25)
        pr(f"Coverage summary ({n_sel} selected dates):")
        pr(f"  ≥ 75%: {n_gt75:>3} dates")
        pr(f"  ≥ 50%: {n_gt50:>3} dates")
        pr(f"  ≥ 25%: {n_gt25:>3} dates")
        pr(f"  < 25%: {n_sel - n_gt25:>3} dates")
        pr()

        pr(f"{'Date':>10}  {'Cov%':>5}  {'Path':>5}  "
           f"{'Dir':>5}  {'#Fr':>3}  {'CRID':>8}")
        pr("-" * 48)
        for dk, cov in zip(sorted(selected), sel_coverages):
            sel = selected[dk]
            d = next(x for x in day_summaries
                     if x["date"] == sel["date"]
                     and x["path"] == sel["path"]
                     and x["direction"] == sel["direction"])
            pr(f"{d['date']:>10}  {d['coverage_pct']:>5.0f}  "
               f"{d['path']:>5}  {d['direction'][:5]:>5}  "
               f"{d['n_frames']:>3}  "
               f"{','.join(d['crids']):>8}")
        pr()

        # CRID versions
        pr("CRID Versions:")
        crid_counts = Counter(r["crid"] for r in records)
        for crid, cnt in crid_counts.most_common():
            pr(f"  {crid}: {cnt} granules")
        pr()

        # Range bandwidth
        pr("Range Bandwidth:")
        bw_counts = Counter(r["range_bw_mhz"] for r in records)
        for bw, cnt in bw_counts.most_common():
            res = "~2 m" if bw == "77" else "~30 m" if bw == "5" else "?"
            pr(f"  {bw} MHz ({res} range res): {cnt} granules")
        pr()

        # Collection versions
        pr("Collection Versions:")
        col_counts = Counter(r["collection"] for r in records)
        for col, cnt in col_counts.most_common():
            pr(f"  {col}: {cnt} granules")

    print(f"\nSaved → {json_path}")
    print(f"Saved → {txt_path}")
    return catalog_data


def _compact_frames(frame_list):
    """Compact a list of frame numbers into range notation, e.g. '35-39'."""
    if not frame_list:
        return ""
    nums = sorted(frame_list)
    if len(nums) == 1:
        return str(nums[0])
    # Check if consecutive
    if nums[-1] - nums[0] == len(nums) - 1:
        return f"{nums[0]}-{nums[-1]}"
    return ",".join(str(n) for n in nums)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build NISAR GSLC day-by-day coverage catalog for "
                    "calving sites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "site", nargs="+",
        help="Site key(s) from calving_sites.SITES, or 'all'.")
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START}).")
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END}).")
    args = parser.parse_args()

    sites = list(SITES.keys()) if "all" in args.site else args.site
    for s in sites:
        if s not in SITES:
            print(f"Unknown site: {s}. Available: {', '.join(SITES.keys())}")
            sys.exit(1)

    for s in sites:
        build_nisar_catalog(s, start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()
