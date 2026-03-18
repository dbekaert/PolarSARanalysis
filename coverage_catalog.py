#!/usr/bin/env python3
"""Comprehensive metadata catalog of all Sentinel-1 candidate days per site.

Queries STAC once and stores EVERY candidate day with full metadata in a
JSON file per site.  From that catalog you can down-select later without
re-querying.

Outputs per site:
  output/<site>_catalog.json   — machine-readable, every day + metadata
  output/<site>_catalog.txt    — human-readable table + auto-selected summary

Usage:
    python coverage_catalog.py jakobshavn
    python coverage_catalog.py petermann
    python coverage_catalog.py all
"""
import json, os, sys
from collections import Counter
from datetime import datetime, timedelta

from shapely.geometry import shape, box
from shapely.ops import unary_union

from stac_utils import get_catalog, search_s1_grd
from calving_sites import (SITES, _dedup_items, _split_by_orbit,
                           _group_by_day, _search_bbox_lonlat,
                           _site_geobox, _month_dates, _SEARCH_WINDOWS)


def _items_metadata(items, aoi, aoi_area):
    """Extract metadata dict from a list of STAC items."""
    footprints = []
    platforms = set()
    timeliness_set = set()
    orbit_dirs = set()
    rel_orbits = set()
    polarisations = set()

    for it in items:
        try:
            footprints.append(shape(it.to_dict()["geometry"]))
        except Exception:
            pass
        plat = (it.properties.get("platform", "?")
                .upper()
                .replace("SENTINEL-1A", "S1A")
                .replace("SENTINEL-1B", "S1B")
                .replace("SENTINEL-1C", "S1C"))
        platforms.add(plat)
        timeliness_set.add(it.properties.get("s1:product_timeliness", "?"))
        orbit_dirs.add(it.properties.get("sat:orbit_state", "?"))
        rel_orbits.add(it.properties.get("sat:relative_orbit", "?"))
        pol = it.properties.get("s1:polarization", [])
        if isinstance(pol, list):
            polarisations.update(pol)
        else:
            polarisations.add(str(pol))

    if footprints:
        union = unary_union(footprints)
        cov = union.intersection(aoi).area / aoi_area * 100
    else:
        cov = 0.0

    return {
        "n_items": len(items),
        "coverage_pct": round(cov, 1),
        "platforms": sorted(platforms),
        "orbit_direction": sorted(orbit_dirs),
        "relative_orbits": sorted(str(r) for r in rel_orbits),
        "timeliness": sorted(timeliness_set),
        "polarisations": sorted(polarisations),
    }


def _split_day_by_track(day_items):
    """Split a day's items into per-track groups (by relative_orbit)."""
    by_track = {}
    for it in day_items:
        track = str(it.properties.get("sat:relative_orbit", "?"))
        by_track.setdefault(track, []).append(it)
    return by_track


def build_catalog(site_key, start_date="2015-01-01", end_date="2026-03-14"):
    """Query STAC and build comprehensive day-level catalog."""
    cfg = SITES[site_key]
    label = cfg["label"]
    bbox_ll = _search_bbox_lonlat(site_key)
    aoi = box(*bbox_ll)
    aoi_area = aoi.area
    _gb, grid_bbox = _site_geobox(site_key)
    w_km = (grid_bbox[2] - grid_bbox[0]) / 1000
    h_km = (grid_bbox[3] - grid_bbox[1]) / 1000

    months = _month_dates(start_date, end_date)
    catalog_api = get_catalog()

    # ---- Collect all candidate days ----
    all_days = []       # flat list of day records
    month_groups = {}   # month_key -> list of day records

    print(f"\n[{label}] Scanning {len(months)} months for candidate days …")

    for year, month in months:
        month_key = f"{year:04d}-{month:02d}"
        d0 = datetime(year, month, 1)
        month_days = []

        # Use widest window that returns data (same logic as fetch)
        items = []
        used_window = 0
        for window_days in _SEARCH_WINDOWS:
            d_start = d0 - timedelta(days=window_days)
            d_end = d0 + timedelta(days=window_days)
            dr = f"{d_start.strftime('%Y-%m-%d')}/{d_end.strftime('%Y-%m-%d')}"
            found = search_s1_grd(catalog_api, bbox_ll, dr,
                                  max_items=200, mode=cfg["s1_mode"])
            if found:
                items = found
                used_window = window_days

        if not items:
            month_groups[month_key] = []
            print(f"  {month_key}: no data")
            continue

        deduped = _dedup_items(items)

        # Process BOTH orbit directions separately, then split by track
        desc, asc = _split_by_orbit(deduped)
        for orbit_label, orbit_items in [("descending", desc),
                                         ("ascending", asc)]:
            if not orbit_items:
                continue
            by_day = _group_by_day(orbit_items)
            for day in sorted(by_day):
                # Split this day's items by track
                by_track = _split_day_by_track(by_day[day])
                for track, track_items in sorted(by_track.items()):
                    meta = _items_metadata(track_items, aoi, aoi_area)
                    rec = {
                        "month_key": month_key,
                        "date": day,
                        "search_window_days": used_window,
                        **meta,
                    }
                    month_days.append(rec)
                    all_days.append(rec)

        month_groups[month_key] = month_days
        n = len(month_days)
        best = max((d["coverage_pct"] for d in month_days), default=0)
        print(f"  {month_key}: {n} candidate days  (best coverage: {best:.0f}%)")

    # ---- Auto-select best day per month ----
    # Strategy: pick the day with highest coverage among those using the
    # most-common relative orbit (track consistency), preferring Fast-24h.
    # Determine dominant track across all data
    track_counts = Counter()
    for d in all_days:
        for t in d["relative_orbits"]:
            track_counts[t] += 1
    dominant_track = track_counts.most_common(1)[0][0] if track_counts else None

    prefer_orbit = cfg.get("preferred_orbit", "descending")
    pref_track = cfg.get("preferred_relative_orbit")
    if pref_track is not None:
        dominant_track = str(pref_track)

    selected = {}
    for month_key in sorted(month_groups):
        days = month_groups[month_key]
        if not days:
            continue

        def _score(d):
            # Each record is one track on one day.
            # Priority:
            #   1. Full coverage (>=95%) — a single track that covers the AOI
            #   2. Preferred orbit direction
            #   3. Preferred track
            #   4. Fast-24h over NRT
            #   5. Higher coverage
            full = 1 if d["coverage_pct"] >= 95 else 0
            orb_match = 1 if prefer_orbit in d["orbit_direction"] else 0
            track_match = 1 if dominant_track in d["relative_orbits"] else 0
            is_fast = 1 if all(t == "Fast-24h" for t in d["timeliness"]) else 0
            return (full, orb_match, track_match, is_fast, d["coverage_pct"])

        best = max(days, key=_score)
        selected[month_key] = {
            "date": best["date"],
            "track": best["relative_orbits"][0] if best["relative_orbits"] else "?",
        }

    # ---- Build output ----
    catalog_data = {
        "site": site_key,
        "label": label,
        "grid_km": f"{w_km:.0f}x{h_km:.0f}",
        "crs": cfg["crs"],
        "resolution": cfg["resolution"],
        "aoi_bbox_lonlat": bbox_ll,
        "grid_bbox_projected": list(grid_bbox),
        "mode": cfg["s1_mode"],
        "date_range": f"{start_date} / {end_date}",
        "dominant_track": dominant_track,
        "preferred_orbit": prefer_orbit,
        "n_months": len(months),
        "n_candidate_days": len(all_days),
        "candidate_days": all_days,
        "selected_days": selected,
    }

    os.makedirs("output/calving/catalogs", exist_ok=True)
    json_path = f"output/calving/catalogs/{site_key}_catalog.json"
    with open(json_path, "w") as f:
        json.dump(catalog_data, f, indent=2)

    # ---- Human-readable text ----
    txt_path = f"output/calving/catalogs/{site_key}_catalog.txt"
    with open(txt_path, "w") as f:
        def pr(s=""):
            print(s)
            f.write(s + "\n")

        pr(f"{label} — Comprehensive Day-Level Metadata Catalog")
        pr(f"Grid: {w_km:.0f}×{h_km:.0f} km  |  CRS: {cfg['crs']}  |  Res: {cfg['resolution']}m")
        pr(f"Mode: {cfg['s1_mode']}  |  Date range: {start_date} → {end_date}")
        pr(f"Dominant track: {dominant_track}  |  Preferred orbit: {prefer_orbit}")
        pr(f"Total candidate days: {len(all_days)}")
        pr()

        hdr = (f"{'Month':>7}  {'Date':>10}  {'Cov%':>5}  {'Sats':>8}  "
               f"{'Orbit':>5}  {'Track':>6}  {'Timeliness':>12}  "
               f"{'Pol':>6}  {'N':>3}  {'Sel':>3}")
        pr(hdr)
        pr("-" * len(hdr))

        for month_key in sorted(month_groups):
            days = month_groups[month_key]
            if not days:
                pr(f"{month_key:>7}  {'(no data)':>10}")
                continue
            sel = selected.get(month_key, {})
            sel_date = sel.get("date", "")
            sel_track = sel.get("track", "")
            for d in sorted(days, key=lambda x: (x["date"], x["relative_orbits"])):
                d_track = d["relative_orbits"][0] if d["relative_orbits"] else ""
                is_sel = " ◄" if (d["date"] == sel_date and d_track == sel_track) else ""
                pr(f"{month_key:>7}  {d['date']:>10}  {d['coverage_pct']:>5.0f}  "
                   f"{','.join(d['platforms']):>8}  "
                   f"{','.join(d['orbit_direction']):>5}  "
                   f"{','.join(d['relative_orbits']):>6}  "
                   f"{','.join(d['timeliness']):>12}  "
                   f"{','.join(d['polarisations']):>6}  "
                   f"{d['n_items']:>3}{is_sel}")
            pr()

        # Summary: selected days
        pr()
        pr("=" * 70)
        pr("AUTO-SELECTED: Best day per month")
        pr(f"  (Ranked by: 100% single-track first, then orbit={prefer_orbit}, "
           f"track={dominant_track}, Fast-24h, coverage)")
        pr()
        pr(f"{'Month':>7}  {'Date':>10}  {'Cov%':>5}  {'Sats':>8}  "
           f"{'Track':>6}  {'Timeliness':>12}")
        pr("-" * 60)
        for mk in sorted(selected):
            sel = selected[mk]
            d = next(x for x in all_days
                     if x["month_key"] == mk
                     and x["date"] == sel["date"]
                     and sel["track"] in x["relative_orbits"])
            pr(f"{mk:>7}  {d['date']:>10}  {d['coverage_pct']:>5.0f}  "
               f"{','.join(d['platforms']):>8}  "
               f"{','.join(d['relative_orbits']):>6}  "
               f"{','.join(d['timeliness']):>12}")

    print(f"\nSaved → {json_path}")
    print(f"Saved → {txt_path}")
    return catalog_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python coverage_catalog.py <site|all>")
        sys.exit(1)

    target = sys.argv[1]
    if target == "all":
        for sk in SITES:
            build_catalog(sk)
    elif target in SITES:
        build_catalog(target)
    else:
        print(f"Unknown site: {target}. Available: {', '.join(SITES.keys())}")
        sys.exit(1)
