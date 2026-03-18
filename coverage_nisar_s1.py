#!/usr/bin/env python3
"""Day-by-day S1 GRD + NISAR GSLC coverage comparison for calving sites.

For the period covered by the NISAR GSLC catalog, queries the Planetary
Computer STAC API for Sentinel-1 GRD products over the same AOI and
builds a combined day-by-day availability report.

This is a standalone analysis script — it does NOT modify any outputs
from the existing calving pipeline (calving_sites.py / run_calving_pipeline.sh).

Inputs:
  output/nisar/<site>_nisar_catalog.json  — from nisar_catalog.py

Outputs:
  output/nisar/<site>_coverage_report.txt  — human-readable table
  output/nisar/<site>_coverage_report.json — machine-readable summary

Usage:
    python coverage_nisar_s1.py jakobshavn
    python coverage_nisar_s1.py all

Requires:
    pip install pystac-client planetary-computer asf_search
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

from calving_sites import SITES, _search_bbox_lonlat
from stac_utils import get_catalog, search_s1_grd


# ---------------------------------------------------------------------------
# S1 GRD day-by-day search
# ---------------------------------------------------------------------------

def _query_s1_days(site_key, start_date, end_date):
    """Query Planetary Computer for S1 GRD items and group by day.

    Returns dict[date_str, list[dict]] where each dict has:
      date, orbit_state, relative_orbit, mode, platform, timeliness
    """
    bbox = _search_bbox_lonlat(site_key)
    cfg = SITES[site_key]
    s1_mode = cfg.get("s1_mode", "IW")

    catalog = get_catalog()
    dt_range = f"{start_date}/{end_date}"
    items = search_s1_grd(catalog, bbox, dt_range,
                          max_items=2000, mode=s1_mode)

    # Parse into lightweight records grouped by day
    by_day = defaultdict(list)
    for it in items:
        p = it.properties
        day = p.get("datetime", "")[:10]
        by_day[day].append({
            "date": day,
            "orbit_state": p.get("sat:orbit_state", "?"),
            "relative_orbit": p.get("sat:relative_orbit"),
            "mode": p.get("sar:instrument_mode", "?"),
            "platform": p.get("platform", "?"),
            "timeliness": p.get("s1:product_timeliness", "?"),
        })
    return dict(by_day)


# ---------------------------------------------------------------------------
# NISAR catalog loader
# ---------------------------------------------------------------------------

def _load_nisar_catalog(site_key):
    """Load the NISAR catalog JSON and return (date_range, day_summary).

    Returns:
        start, end: date strings bounding the NISAR catalog
        nisar_days: dict[date_str, dict] with keys:
            selected, path, direction, coverage_pct, bw_mhz, pol, n_frames
    """
    cat_path = os.path.join("output", "nisar_s1", "catalogs",
                            f"{site_key}_nisar_catalog.json")
    if not os.path.exists(cat_path):
        print(f"  ✗ No NISAR catalog at {cat_path}")
        print(f"    Run: python nisar_catalog.py {site_key}")
        return None, None, None

    with open(cat_path) as f:
        cat = json.load(f)

    granules = cat.get("granules", [])
    selected = cat.get("selected_tracks", {})
    chosen = cat.get("chosen_mode", {})
    day_summaries = cat.get("day_summaries", [])

    # Get all unique dates from granules
    all_dates = sorted(set(g["date"] for g in granules))
    if not all_dates:
        return None, None, None

    start, end = all_dates[0], all_dates[-1]

    # Build per-day summary
    nisar_days = {}
    for d in all_dates:
        # Get day summaries for this date
        ds_list = [ds for ds in day_summaries if ds["date"] == d]
        sel = selected.get(d)

        # Best coverage from any track/frame combo on this date
        best_cov = max((ds.get("coverage_pct", 0) for ds in ds_list),
                       default=0)
        n_frames = sum(1 for g in granules if g["date"] == d)

        # Granule-level info
        granules_today = [g for g in granules if g["date"] == d]
        bws = set(g.get("range_bw_mhz", "?") for g in granules_today)
        pols = set(g.get("polarization", "?") for g in granules_today)

        nisar_days[d] = {
            "selected": sel is not None,
            "path": sel["path"] if sel else None,
            "direction": sel["direction"] if sel else None,
            "coverage_pct": best_cov,
            "bw_mhz": ",".join(sorted(bws)),
            "pol": ",".join(sorted(pols)),
            "n_frames": n_frames,
        }

    return start, end, nisar_days


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(site_key):
    """Build and write the combined S1 + NISAR day-by-day coverage report."""
    cfg = SITES[site_key]
    label = cfg["label"]

    print(f"\n{'='*70}")
    print(f"  {label} — S1 GRD + NISAR GSLC Coverage Comparison")
    print(f"{'='*70}")

    # 1. Load NISAR catalog
    start, end, nisar_days = _load_nisar_catalog(site_key)
    if nisar_days is None:
        return

    print(f"  NISAR period: {start} → {end}")
    print(f"  NISAR dates with data: {len(nisar_days)}")

    # 2. Query S1 GRD for the same period
    print(f"  Querying S1 GRD ({cfg.get('s1_mode', 'IW')} mode) ...")
    s1_days = _query_s1_days(site_key, start, end)
    print(f"  S1 dates with data: {len(s1_days)}")

    # 3. Build day-by-day calendar
    d_start = datetime.strptime(start, "%Y-%m-%d")
    d_end = datetime.strptime(end, "%Y-%m-%d")
    all_days = []
    d = d_start
    while d <= d_end:
        all_days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    # 4. Compile table
    rows = []
    for day in all_days:
        ni = nisar_days.get(day)
        s1 = s1_days.get(day)

        nisar_flag = "✓" if ni else "·"
        nisar_sel = "◄" if (ni and ni["selected"]) else " "
        nisar_cov = f"{ni['coverage_pct']:5.0f}%" if ni else "     -"
        nisar_nf = f"{ni['n_frames']}" if ni else "-"

        s1_flag = "✓" if s1 else "·"
        s1_n = len(s1) if s1 else 0
        # Summarise S1 orbits for the day
        if s1:
            orbits = sorted(set(r["relative_orbit"] for r in s1
                                if r["relative_orbit"]))
            orb_str = ",".join(str(o) for o in orbits)
            dirs = set(r["orbit_state"][0].upper() for r in s1
                       if r["orbit_state"] and r["orbit_state"] != "?")
            dir_str = "/".join(sorted(dirs))
        else:
            orb_str = "-"
            dir_str = "-"

        rows.append({
            "date": day,
            "nisar": nisar_flag,
            "nisar_sel": nisar_sel,
            "nisar_cov": nisar_cov,
            "nisar_frames": nisar_nf,
            "s1": s1_flag,
            "s1_n": s1_n,
            "s1_orbits": orb_str,
            "s1_dir": dir_str,
            "has_both": ni is not None and s1 is not None,
        })

    # 5. Summary statistics
    n_total = len(all_days)
    n_nisar = sum(1 for r in rows if r["nisar"] == "✓")
    n_nisar_sel = sum(1 for r in rows if r["nisar_sel"] == "◄")
    n_s1 = sum(1 for r in rows if r["s1"] == "✓")
    n_both = sum(1 for r in rows if r["has_both"])
    n_either = sum(1 for r in rows
                   if r["nisar"] == "✓" or r["s1"] == "✓")
    n_nisar_only = sum(1 for r in rows
                       if r["nisar"] == "✓" and r["s1"] != "✓")
    n_s1_only = sum(1 for r in rows
                    if r["s1"] == "✓" and r["nisar"] != "✓")
    n_neither = n_total - n_either

    # NISAR chosen mode
    cat_path = os.path.join("output", "nisar_s1", "catalogs",
                            f"{site_key}_nisar_catalog.json")
    with open(cat_path) as f:
        chosen = json.load(f).get("chosen_mode", {})
    nisar_mode_str = f"{chosen.get('bw_mhz', '?')} MHz {chosen.get('polarization', '?')}"

    # 6. Write text report
    out_dir = os.path.join("output", "nisar_s1", "catalogs")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"{site_key}_coverage_report.txt")

    lines = []
    lines.append(f"{'='*74}")
    lines.append(f"  {label} — S1 GRD + NISAR GSLC Day-by-Day Coverage")
    lines.append(f"{'='*74}")
    lines.append(f"  Period: {start} → {end}  ({n_total} days)")
    lines.append(f"  NISAR mode: {nisar_mode_str}  (L-band)")
    lines.append(f"  S1 mode:    {cfg.get('s1_mode', 'IW')}  (C-band)")
    lines.append("")
    lines.append(f"  SUMMARY")
    lines.append(f"  {'—'*40}")
    lines.append(f"  NISAR dates:       {n_nisar:3d} / {n_total}")
    lines.append(f"  NISAR selected:    {n_nisar_sel:3d} / {n_total}"
                 f"  (auto-selected best track)")
    lines.append(f"  S1 GRD dates:      {n_s1:3d} / {n_total}")
    lines.append(f"  Both sensors:      {n_both:3d} / {n_total}")
    lines.append(f"  Either sensor:     {n_either:3d} / {n_total}")
    lines.append(f"  NISAR only:        {n_nisar_only:3d}")
    lines.append(f"  S1 only:           {n_s1_only:3d}")
    lines.append(f"  Neither:           {n_neither:3d}")
    lines.append("")

    # Revisit frequency
    if n_nisar > 1:
        nisar_dates_sorted = sorted(d for d in all_days
                                    if nisar_days.get(d))
        gaps = [(datetime.strptime(nisar_dates_sorted[i+1], "%Y-%m-%d")
                 - datetime.strptime(nisar_dates_sorted[i], "%Y-%m-%d")).days
                for i in range(len(nisar_dates_sorted)-1)]
        lines.append(f"  NISAR revisit:  median {sorted(gaps)[len(gaps)//2]}d, "
                     f"mean {sum(gaps)/len(gaps):.1f}d, "
                     f"max {max(gaps)}d")
    if n_s1 > 1:
        s1_dates_sorted = sorted(d for d in all_days if s1_days.get(d))
        gaps = [(datetime.strptime(s1_dates_sorted[i+1], "%Y-%m-%d")
                 - datetime.strptime(s1_dates_sorted[i], "%Y-%m-%d")).days
                for i in range(len(s1_dates_sorted)-1)]
        lines.append(f"  S1 revisit:     median {sorted(gaps)[len(gaps)//2]}d, "
                     f"mean {sum(gaps)/len(gaps):.1f}d, "
                     f"max {max(gaps)}d")
    if n_both > 1:
        both_dates = sorted(d for d in all_days
                            if nisar_days.get(d) and s1_days.get(d))
        gaps = [(datetime.strptime(both_dates[i+1], "%Y-%m-%d")
                 - datetime.strptime(both_dates[i], "%Y-%m-%d")).days
                for i in range(len(both_dates)-1)]
        lines.append(f"  Both revisit:   median {sorted(gaps)[len(gaps)//2]}d, "
                     f"mean {sum(gaps)/len(gaps):.1f}d, "
                     f"max {max(gaps)}d")

    lines.append("")
    lines.append(f"  DAY-BY-DAY TABLE")
    lines.append(f"  {'—'*68}")
    hdr = (f"  {'Date':12s} {'NISAR':5s} {'Sel':3s} {'Cov':>6s} "
           f"{'#F':>2s}  {'S1':4s} {'#':>2s} {'Orbits':>10s} {'Dir':>3s}  "
           f"{'Both':4s}")
    lines.append(hdr)
    lines.append(f"  {'—'*68}")

    for r in rows:
        both_mark = "★" if r["has_both"] else " "
        line = (f"  {r['date']:12s} {r['nisar']:5s} {r['nisar_sel']:3s} "
                f"{r['nisar_cov']:>6s} {r['nisar_frames']:>2s}  "
                f"{r['s1']:4s} {r['s1_n']:>2d} {r['s1_orbits']:>10s} "
                f"{r['s1_dir']:>3s}  {both_mark}")
        lines.append(line)

    lines.append(f"  {'—'*68}")
    lines.append(f"  ✓ = data available  ◄ = NISAR auto-selected  ★ = both sensors")
    lines.append(f"  Cov = NISAR AOI coverage %   #F = NISAR frames   # = S1 items")
    lines.append("")

    report_text = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(report_text)

    # 7. Write JSON summary
    json_path = os.path.join(out_dir, f"{site_key}_coverage_report.json")
    report_json = {
        "site": site_key,
        "label": label,
        "period": {"start": start, "end": end, "total_days": n_total},
        "nisar_mode": nisar_mode_str,
        "s1_mode": cfg.get("s1_mode", "IW"),
        "summary": {
            "nisar_dates": n_nisar,
            "nisar_selected": n_nisar_sel,
            "s1_dates": n_s1,
            "both_sensors": n_both,
            "either_sensor": n_either,
            "nisar_only": n_nisar_only,
            "s1_only": n_s1_only,
            "neither": n_neither,
        },
        "days": rows,
    }
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)

    # Print report to stdout
    print(report_text)
    print(f"  Saved: {txt_path}")
    print(f"  Saved: {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="S1 GRD + NISAR GSLC day-by-day coverage comparison")
    parser.add_argument(
        "sites", nargs="+",
        help="Site key(s) from calving_sites.SITES, or 'all'")
    args = parser.parse_args()

    site_list = list(SITES.keys()) if "all" in args.sites else args.sites
    for s in site_list:
        if s not in SITES:
            print(f"Unknown site: {s}")
            continue
        build_report(s)


if __name__ == "__main__":
    main()
