#!/usr/bin/env python3
"""
Create side-by-side S1/NISAR animation for Jakobshavn calving front.

Layout:
  - Top: time slider showing all dates with month labels
  - Left: Sentinel-1 (C-band) with scale bar, colorbar on left edge
  - Right: NISAR (L-band) with colorbar on right edge
  - Center: 3D globe showing location (overlaps top corners of both panels)
  - Bottom: "Image analysis by David Bekaert" attribution

Usage:
    python animate_paired.py [--site jakobshavn] [--fps 1] [--output FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num

# Site config (extend for other sites)
SITE_CFG = {
    "jakobshavn": {
        "title": "Jakobshavn Glacier",
        "center_lat": 69.17,
        "center_lon": -50.2,
        "globe_center": (72.0, -42.0),
        "globe_satellite_height": 1_500_000,
        "scale_bar_km": 20,
        "out_res": 100,  # metres
    },
}


def _compress_gif(gif_path: str, lossy: int = 30) -> None:
    """Compress with gifsicle if available."""
    if shutil.which("gifsicle") is None:
        print("  gifsicle not found — skipping compression")
        return
    try:
        subprocess.run(
            ["gifsicle", f"--lossy={lossy}", "-O3", "--colors", "256",
             gif_path, "-o", gif_path],
            check=True, capture_output=True
        )
        print(f"  Compressed with gifsicle (lossy={lossy})")
    except subprocess.CalledProcessError as e:
        print(f"  gifsicle failed: {e}")


def _extract_utc_time_nisar(url: str) -> str:
    """Extract UTC time from NISAR filename in URL (e.g. 20251028T235201 → 23:52 UTC)."""
    m = re.search(r'_(\d{8}T\d{6})_', url)
    if m:
        ts = m.group(1)  # 20251028T235201
        return f"{ts[9:11]}:{ts[11:13]} UTC"
    return ""


def _extract_utc_time_s1(datetimes: list, directions: list) -> str:
    """Extract UTC time(s) from S1 datetime strings.
    
    When multiple passes are combined, show time range or multiple times.
    Also handles misalignment between datetimes and directions lists.
    """
    if not datetimes:
        return ""
    
    # Parse all times
    times = []
    for dt_str in datetimes:
        try:
            if 'T' in dt_str:
                time_part = dt_str.split('T')[1][:5]  # HH:MM
                times.append(time_part)
        except Exception:
            pass
    
    if not times:
        return ""
    
    # Deduplicate (consecutive swaths have ~same time)
    unique_times = []
    for t in times:
        if not unique_times or abs(int(t[:2]) - int(unique_times[-1][:2])) > 1:
            unique_times.append(t)
    
    # If multiple distinct times (likely different passes), show both
    if len(unique_times) > 1:
        return f"{unique_times[0]} & {unique_times[-1]} UTC"
    else:
        return f"{unique_times[0]} UTC"


def _format_s1_track_info(s1_meta: dict) -> str:
    """Format S1 platform + track + direction for display.
    
    Handles multiple passes correctly by showing all combinations.
    """
    platforms = s1_meta.get('platforms', ['S1'])
    orbits = s1_meta.get('relative_orbits', [])
    directions = s1_meta.get('directions', [])
    datetimes = s1_meta.get('datetimes', [])
    
    plat_str = "/".join(sorted(set(platforms)))
    
    # If single direction, simple case
    if len(directions) == 1 and len(orbits) == 1:
        dir_str = directions[0][:3].upper()
        return f"{plat_str}  T{orbits[0]} {dir_str}"
    
    # Multiple passes - need to associate orbits with directions
    # Heuristic for Greenland: morning (~09:xx) = descending, evening (~20:xx) = ascending
    pass_info = []
    for orb in sorted(set(orbits)):
        # Try to infer direction from orbit number patterns
        # For Jakobshavn: T127 = descending, T46/T90 = ascending
        pass_info.append(f"T{orb}")
    
    if len(directions) > 1:
        dir_str = "/".join(d[:3].upper() for d in sorted(directions))
    else:
        dir_str = directions[0][:3].upper() if directions else ""
    
    return f"{plat_str}  {'+'.join(pass_info)} {dir_str}"


def render_frame(
    fig, axes, ax_globe, ax_tl,
    s1_arr, nisar_arr, pair_meta, current_idx, all_dates,
    s1_vmin, s1_vmax, ni_vmin, ni_vmax, cfg,
    tl_d_num, tl_progress, tl_dot, tl_date_text, tl_ax_pos,
    im_s1, im_ni, cbar_s1, cbar_ni,
    globe_marker, s1_meta_texts, ni_meta_texts
):
    """Update frame data for animation."""
    ax_s1, ax_ni = axes
    s1_meta_line1, s1_meta_line2 = s1_meta_texts
    ni_meta_line1, ni_meta_line2 = ni_meta_texts

    # Update image data
    s1_disp = np.where(np.isfinite(s1_arr) & (s1_arr != 0), s1_arr, np.nan)
    ni_disp = np.where(np.isfinite(nisar_arr) & (nisar_arr != 0), nisar_arr, np.nan)
    im_s1.set_data(s1_disp)
    im_ni.set_data(ni_disp)

    # Update metadata text
    s1_meta = pair_meta['s1']
    ni_meta = pair_meta['nisar']

    # S1: Line 1 = platform + track(s) + direction(s), Line 2 = UTC time(s)
    s1_line1 = _format_s1_track_info(s1_meta)
    s1_utc = _extract_utc_time_s1(s1_meta.get('datetimes', []), s1_meta.get('directions', []))
    s1_meta_line1.set_text(s1_line1)
    s1_meta_line2.set_text(s1_utc)

    # NISAR: Line 1 = NISAR + track, Line 2 = UTC time
    ni_path = ni_meta.get('path', 0)
    ni_dir = ni_meta.get('direction', '')[:3].upper()
    ni_line1 = f"NISAR  T{ni_path} {ni_dir}"
    ni_utc = ""
    if ni_meta.get('urls'):
        ni_utc = _extract_utc_time_nisar(ni_meta['urls'][0])
    ni_meta_line1.set_text(ni_line1)
    ni_meta_line2.set_text(ni_utc)

    # Update timeline
    cur_dn = tl_d_num[current_idx]
    tl_progress.set_xdata([tl_d_num[0], cur_dn])
    tl_dot.set_data([cur_dn], [0])

    # Update date label position
    frac = (cur_dn - tl_d_num[0]) / (tl_d_num[-1] - tl_d_num[0])
    fig_x = tl_ax_pos.x0 + frac * tl_ax_pos.width
    min_x, max_x = 0.12, 0.88
    if fig_x < min_x:
        fig_x, ha = min_x, "left"
    elif fig_x > max_x:
        fig_x, ha = max_x, "right"
    else:
        ha = "center"
    tl_date_text.set_position((fig_x, tl_ax_pos.y1 + 0.008))
    tl_date_text.set_horizontalalignment(ha)
    tl_date_text.set_text(all_dates[current_idx])


def create_animation(
    site_key: str = "jakobshavn",
    fps: int = 1,
    output_path: str | None = None,
):
    """Create S1 vs NISAR side-by-side animation."""
    cfg = SITE_CFG[site_key]
    cache_dir = Path(f"output/nisar_s1/cache/{site_key}")
    animations_dir = Path(f"output/nisar_s1/animations/{site_key}")
    animations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    manifest = json.load(open(manifest_path))
    pairs = manifest['pairs']
    
    # Filter out pairs with NISAR valid_fraction < 20%
    MIN_VALID_FRAC = 0.20
    pairs = [p for p in pairs if p['nisar'].get('valid_fraction', 0) >= MIN_VALID_FRAC]
    
    n_frames = len(pairs)
    print(f"Creating animation for {site_key}: {n_frames} frames (after filtering ≥{MIN_VALID_FRAC:.0%} valid)")

    # Compute global percentile stretches per sensor
    all_nisar, all_s1 = [], []
    for p in pairs:
        ni = np.load(cache_dir / p['nisar']['file'])
        s1 = np.load(cache_dir / p['s1']['file'])
        nv = ni[np.isfinite(ni) & (ni != 0)]
        sv = s1[np.isfinite(s1) & (s1 != 0)]
        all_nisar.append(nv)
        all_s1.append(sv)
    all_nisar = np.concatenate(all_nisar)
    all_s1 = np.concatenate(all_s1)
    ni_vmin, ni_vmax = np.percentile(all_nisar, 2), np.percentile(all_nisar, 98)
    s1_vmin, s1_vmax = np.percentile(all_s1, 2), np.percentile(all_s1, 98)
    print(f"  NISAR stretch: {ni_vmin:.1f} – {ni_vmax:.1f} dB")
    print(f"  S1 stretch:    {s1_vmin:.1f} – {s1_vmax:.1f} dB")

    # Get date list
    all_dates = [p['date'] for p in pairs]
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates]
    tl_d_num = [date2num(d) for d in date_objs]

    # Figure setup
    # Two panels side by side + colorbars + timeline + globe
    fig_w, fig_h = 14, 5.8  # reduced height to eliminate excess space
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    # Layout fractions
    tl_height = 0.10     # timeline height
    top_pad = 0.13       # increased to shift panels down
    bot_pad = 0.02       # bottom for attribution
    tl_pad = 0.01        # small gap between timeline and panels
    cbar_w = 0.025       # colorbar width
    gap = 0.03           # gap between panels
    side_pad = 0.06      # left/right margin

    # Timeline at bottom, panels above
    tl_bottom = bot_pad
    map_bottom = tl_bottom + tl_height + tl_pad
    map_height = 1 - top_pad - map_bottom
    panel_w = (1 - 2*side_pad - 2*cbar_w - gap) / 2

    # Axes positions: [left, bottom, width, height]
    s1_left = side_pad + cbar_w
    ni_left = s1_left + panel_w + gap

    # Sentinel-1 axes (left)
    ax_s1 = fig.add_axes([s1_left, map_bottom, panel_w, map_height])
    ax_s1.set_facecolor("black")
    ax_s1.set_xticks([])
    ax_s1.set_yticks([])
    for spine in ax_s1.spines.values():
        spine.set_color('#444444')

    # NISAR axes (right)
    ax_ni = fig.add_axes([ni_left, map_bottom, panel_w, map_height])
    ax_ni.set_facecolor("black")
    ax_ni.set_xticks([])
    ax_ni.set_yticks([])
    for spine in ax_ni.spines.values():
        spine.set_color('#444444')

    # Timeline axes (below panels)
    ax_tl = fig.add_axes([side_pad, tl_bottom, 1 - 2*side_pad, tl_height])
    ax_tl.set_facecolor("black")
    for spine in ax_tl.spines.values():
        spine.set_visible(False)
    ax_tl.set_yticks([])
    ax_tl.set_xticks([])

    # Timeline track - add padding to avoid right tick cropping
    date_range = tl_d_num[-1] - tl_d_num[0]
    ax_tl.set_xlim(tl_d_num[0] - date_range * 0.02, tl_d_num[-1] + date_range * 0.02)
    ax_tl.set_ylim(-0.6, 0.15)  # standard range
    ax_tl.axhline(0, color="#555555", linewidth=2, zorder=1)

    # Tick marks
    for dn in tl_d_num:
        ax_tl.plot([dn, dn], [-0.1, 0.1], color="#666666", linewidth=0.8, zorder=2)

    # Generate month labels for full date range
    from dateutil.relativedelta import relativedelta
    import calendar
    d_start = date_objs[0].replace(day=1)
    d_end = date_objs[-1].replace(day=1) + relativedelta(months=1)
    month_labels = []
    cur = d_start
    while cur <= d_end:
        month_labels.append(cur)
        cur += relativedelta(months=1)

    for m_dt in month_labels:
        m_dn = date2num(m_dt)
        if tl_d_num[0] <= m_dn <= tl_d_num[-1]:
            # Format as "Nov 2025"
            ax_tl.text(m_dn, -0.25, m_dt.strftime("%b %Y"), color="#888888",
                       fontsize=9, ha="center", va="top")

    # Progress bar
    tl_progress, = ax_tl.plot([tl_d_num[0], tl_d_num[0]], [0, 0],
                               color="cyan", linewidth=3, zorder=3)
    tl_dot, = ax_tl.plot([tl_d_num[0]], [0], marker="o", color="cyan",
                          markersize=8, zorder=5)

    # Date label (figure coords) - above timeline
    tl_ax_pos = ax_tl.get_position()
    tl_date_text = fig.text(tl_ax_pos.x0, tl_ax_pos.y1 + 0.008,
                            all_dates[0], color="white", fontsize=13,
                            fontweight="bold", ha="left", va="bottom")

    # Load first frame
    p0 = pairs[0]
    s1_arr = np.load(cache_dir / p0['s1']['file'])
    nisar_arr = np.load(cache_dir / p0['nisar']['file'])

    s1_disp = np.where(np.isfinite(s1_arr) & (s1_arr != 0), s1_arr, np.nan)
    ni_disp = np.where(np.isfinite(nisar_arr) & (nisar_arr != 0), nisar_arr, np.nan)

    im_s1 = ax_s1.imshow(s1_disp, cmap='gray', vmin=s1_vmin, vmax=s1_vmax,
                          aspect='equal', origin='upper')
    im_ni = ax_ni.imshow(ni_disp, cmap='gray', vmin=ni_vmin, vmax=ni_vmax,
                          aspect='equal', origin='upper')

    # Titles with band labels
    ax_s1.set_title("Sentinel-1 (C-band)", color="white", fontsize=14,
                    fontweight="bold", pad=8)
    ax_ni.set_title("NISAR (L-band)", color="white", fontsize=14,
                    fontweight="bold", pad=8)

    # Colorbars
    cbar_s1_ax = fig.add_axes([side_pad, map_bottom + 0.12, cbar_w*0.6, map_height - 0.24])
    cbar_s1 = fig.colorbar(im_s1, cax=cbar_s1_ax)
    cbar_s1.ax.yaxis.set_ticks_position('left')
    cbar_s1.ax.yaxis.set_label_position('left')
    cbar_s1.set_label('dB', color='white', fontsize=10)
    cbar_s1.ax.tick_params(colors='white', labelsize=8)
    # White outline around colorbar
    for spine in cbar_s1.ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1)

    cbar_ni_ax = fig.add_axes([1 - side_pad - cbar_w*0.6, map_bottom + 0.12, cbar_w*0.6, map_height - 0.24])
    cbar_ni = fig.colorbar(im_ni, cax=cbar_ni_ax)
    cbar_ni.set_label('dB', color='white', fontsize=10)
    cbar_ni.ax.tick_params(colors='white', labelsize=8)
    # White outline around colorbar
    for spine in cbar_ni.ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(1)

    # Scale bar (S1 only, bottom-left)
    sb_km = cfg.get("scale_bar_km", 20)
    out_res = cfg.get("out_res", 100)
    sb_px = sb_km * 1000 / out_res
    y0 = s1_arr.shape[0] - s1_arr.shape[0] * 0.06
    x0 = s1_arr.shape[1] * 0.05
    x1 = x0 + sb_px
    ax_s1.plot([x0, x1], [y0, y0], color="white", linewidth=2.5, solid_capstyle="butt", zorder=8)
    tick_h = s1_arr.shape[0] * 0.015
    ax_s1.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=8)
    ax_s1.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=8)
    ax_s1.text((x0 + x1) / 2, y0 - tick_h * 3, f"{sb_km} km", color="white",
               fontsize=11, ha="center", va="top", zorder=8)

    # Metadata text (S1 upper-left, NISAR upper-right)
    # S1: Line 1 = mission + track, Line 2 = UTC time
    s1_meta_line1 = ax_s1.text(0.03, 0.97, "", color="white", fontsize=10,
                               ha="left", va="top", transform=ax_s1.transAxes,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    s1_meta_line2 = ax_s1.text(0.03, 0.90, "", color="#cccccc", fontsize=9,
                               ha="left", va="top", transform=ax_s1.transAxes,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # NISAR: Line 1 = mission + track, Line 2 = UTC time
    ni_meta_line1 = ax_ni.text(0.97, 0.97, "", color="white", fontsize=10,
                               ha="right", va="top", transform=ax_ni.transAxes,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
    ni_meta_line2 = ax_ni.text(0.97, 0.90, "", color="#cccccc", fontsize=9,
                               ha="right", va="top", transform=ax_ni.transAxes,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # 3D Globe - smaller, centered
    globe_size = 0.117  # 10% smaller
    globe_x = 0.5 - globe_size / 2  # centered
    globe_y = map_bottom + map_height - globe_size * 2.5  # shifted down

    globe_proj = ccrs.NearsidePerspective(
        central_longitude=cfg["globe_center"][1],
        central_latitude=cfg["globe_center"][0],
        satellite_height=cfg["globe_satellite_height"]
    )
    ax_globe = fig.add_axes([globe_x, globe_y, globe_size, globe_size * fig_w / fig_h],
                            projection=globe_proj)
    ax_globe.set_global()
    ax_globe.add_feature(cfeature.LAND, facecolor="#333333", edgecolor="none")
    ax_globe.add_feature(cfeature.OCEAN, facecolor="#111111")
    ax_globe.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'coastline', '110m',
        edgecolor='#555555', facecolor='none', linewidth=0.4))
    globe_marker, = ax_globe.plot(
        cfg["center_lon"], cfg["center_lat"],
        marker="o", color="cyan", markersize=5,
        markeredgecolor="white", markeredgewidth=0.6,
        transform=ccrs.PlateCarree(), zorder=10
    )
    ax_globe.spines['geo'].set_edgecolor('#666666')
    ax_globe.spines['geo'].set_linewidth(1.0)
    ax_globe.patch.set_alpha(0.9)

    # Attribution at bottom center (moved up, smaller)
    fig.text(0.5, 0.012, "Image analysis by David Bekaert", color="#888888", fontsize=9,
             ha="center", va="bottom", style="italic")

    # Glacier name (shifted down with space above)
    fig.text(0.5, 0.935, cfg["title"], color="white", fontsize=18,
             fontweight="bold", ha="center", va="top")

    # Create frames directory
    frames_dir = cache_dir / "animation_frames"
    frames_dir.mkdir(exist_ok=True)

    # Generate frames
    print(f"Rendering {n_frames} frames...")
    for i, p in enumerate(pairs):
        s1_arr = np.load(cache_dir / p['s1']['file'])
        nisar_arr = np.load(cache_dir / p['nisar']['file'])

        render_frame(
            fig, (ax_s1, ax_ni), ax_globe, ax_tl,
            s1_arr, nisar_arr, p, i, all_dates,
            s1_vmin, s1_vmax, ni_vmin, ni_vmax, cfg,
            tl_d_num, tl_progress, tl_dot, tl_date_text, tl_ax_pos,
            im_s1, im_ni, cbar_s1, cbar_ni,
            globe_marker, (s1_meta_line1, s1_meta_line2), (ni_meta_line1, ni_meta_line2)
        )

        frame_path = frames_dir / f"frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=120, facecolor="black",
                    bbox_inches='tight', pad_inches=0.02)
        print(f"  [{i+1}/{n_frames}] {all_dates[i]}")

    plt.close(fig)

    # Compile GIF
    if output_path is None:
        output_path = str(animations_dir / f"{site_key}_s1_nisar_comparison.gif")

    delay = int(100 / fps)  # centiseconds per frame
    frame_pattern = str(frames_dir / "frame_*.png")

    print(f"Compiling GIF: {output_path}")

    # Use ImageMagick convert or PIL
    try:
        # Try ImageMagick first
        subprocess.run(
            ["convert", "-delay", str(delay), "-loop", "0",
             str(frames_dir / "frame_*.png"), output_path],
            shell=False, check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to PIL
        from PIL import Image
        frames = sorted(frames_dir.glob("frame_*.png"))
        images = [Image.open(f) for f in frames]
        images[0].save(
            output_path, save_all=True, append_images=images[1:],
            duration=1000 // fps, loop=0
        )

    # Get size before compression
    size_before = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Raw GIF: {size_before:.1f} MB")

    # Compress
    _compress_gif(output_path, lossy=30)

    size_after = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final: {size_after:.1f} MB")
    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create S1/NISAR comparison animation")
    parser.add_argument("--site", default="jakobshavn", help="Site key")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second (default: 1)")
    parser.add_argument("--output", help="Output GIF path")
    args = parser.parse_args()

    create_animation(
        site_key=args.site,
        fps=args.fps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
