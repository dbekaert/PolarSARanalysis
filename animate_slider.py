#!/usr/bin/env python3
"""
Create a slider reveal animation comparing S1 and NISAR for a single date.

The animation shows:
- S1 (left half) and NISAR (right half) with a sliding divider
- Starts in middle, slides right (full S1), then slides left (full NISAR)
- Mission/track/UTC text visible in corners (covered when slider passes)

Usage:
    python animate_slider.py [--site jakobshavn] [--date 2025-11-21] [--output FILE]
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
from matplotlib.patches import Rectangle


# Site config
SITE_CFG = {
    "jakobshavn": {
        "title": "Jakobshavn Glacier",
        "center_lat": 69.17,
        "center_lon": -50.2,
        "globe_center": (72.0, -42.0),
        "globe_satellite_height": 1_500_000,
        "scale_bar_km": 20,
        "out_res": 100,
    },
}


def _compress_gif(gif_path: str, lossy: int = 30) -> None:
    """Compress with gifsicle if available."""
    if shutil.which("gifsicle") is None:
        return
    try:
        subprocess.run(
            ["gifsicle", f"--lossy={lossy}", "-O3", "--colors", "256",
             gif_path, "-o", gif_path],
            check=True, capture_output=True
        )
        print(f"  Compressed with gifsicle (lossy={lossy})")
    except subprocess.CalledProcessError:
        pass


def _extract_utc_time_nisar(url: str) -> str:
    """Extract UTC time from NISAR filename."""
    m = re.search(r'_(\d{8}T\d{6})_', url)
    if m:
        ts = m.group(1)
        return f"{ts[9:11]}:{ts[11:13]} UTC"
    return ""


def _extract_utc_time_s1(datetimes: list, directions: list = None) -> str:
    """Extract UTC time(s) from S1 datetime strings.
    
    When multiple passes are combined, show time range.
    """
    if not datetimes:
        return ""
    
    times = []
    for dt_str in datetimes:
        try:
            if 'T' in dt_str:
                times.append(dt_str.split('T')[1][:5])
        except Exception:
            pass
    
    if not times:
        return ""
    
    # Deduplicate (consecutive swaths have ~same time)
    unique_times = []
    for t in times:
        if not unique_times or abs(int(t[:2]) - int(unique_times[-1][:2])) > 1:
            unique_times.append(t)
    
    if len(unique_times) > 1:
        return f"{unique_times[0]} & {unique_times[-1]} UTC"
    else:
        return f"{unique_times[0]} UTC"


def _format_s1_track_info(s1_meta: dict) -> str:
    """Format S1 platform + track + direction for display."""
    platforms = s1_meta.get('platforms', ['S1'])
    orbits = s1_meta.get('relative_orbits', [])
    directions = s1_meta.get('directions', [])
    
    plat_str = "/".join(sorted(set(platforms)))
    
    if len(directions) == 1 and len(orbits) == 1:
        dir_str = directions[0][:3].upper()
        return f"{plat_str}  T{orbits[0]} {dir_str}"
    
    pass_info = [f"T{orb}" for orb in sorted(set(orbits))]
    dir_str = "/".join(d[:3].upper() for d in sorted(directions)) if directions else ""
    
    return f"{plat_str}  {'+'.join(pass_info)} {dir_str}"


def create_slider_animation(
    site_key: str = "jakobshavn",
    target_date: str = "2025-11-21",
    output_path: str | None = None,
    n_frames: int = 60,
):
    """Create slider reveal animation for a single date."""
    cfg = SITE_CFG[site_key]
    cache_dir = Path(f"output/nisar_s1/cache/{site_key}")
    animations_dir = Path(f"output/nisar_s1/animations/{site_key}")
    animations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    manifest = json.load(open(manifest_path))
    pairs = manifest['pairs']

    # Find the target date or closest available
    pair = None
    for p in pairs:
        if p['date'] == target_date:
            pair = p
            break
    
    if pair is None:
        # Find closest date
        dates = [p['date'] for p in pairs]
        closest = min(dates, key=lambda d: abs(
            datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")
        ))
        print(f"  Target date {target_date} not found, using closest: {closest}")
        pair = next(p for p in pairs if p['date'] == closest)
        target_date = closest

    print(f"Creating slider animation for {site_key} / {target_date}")

    # Load data
    s1_arr = np.load(cache_dir / pair['s1']['file'])
    nisar_arr = np.load(cache_dir / pair['nisar']['file'])

    # Compute stretch per sensor
    s1_valid = s1_arr[np.isfinite(s1_arr) & (s1_arr != 0)]
    ni_valid = nisar_arr[np.isfinite(nisar_arr) & (nisar_arr != 0)]
    s1_vmin, s1_vmax = np.percentile(s1_valid, 2), np.percentile(s1_valid, 98)
    ni_vmin, ni_vmax = np.percentile(ni_valid, 2), np.percentile(ni_valid, 98)

    # Prepare display arrays (normalized to 0-1)
    s1_disp = np.where(np.isfinite(s1_arr) & (s1_arr != 0), s1_arr, np.nan)
    ni_disp = np.where(np.isfinite(nisar_arr) & (nisar_arr != 0), nisar_arr, np.nan)

    # Normalize to 0-1 range for blending
    s1_norm = (s1_disp - s1_vmin) / (s1_vmax - s1_vmin)
    s1_norm = np.clip(s1_norm, 0, 1)
    ni_norm = (ni_disp - ni_vmin) / (ni_vmax - ni_vmin)
    ni_norm = np.clip(ni_norm, 0, 1)

    # Make arrays same size (crop to minimum)
    h = min(s1_norm.shape[0], ni_norm.shape[0])
    w = min(s1_norm.shape[1], ni_norm.shape[1])
    s1_norm = s1_norm[:h, :w]
    ni_norm = ni_norm[:h, :w]

    # Extract metadata
    s1_meta = pair['s1']
    ni_meta = pair['nisar']

    s1_line1 = _format_s1_track_info(s1_meta)
    s1_utc = _extract_utc_time_s1(s1_meta.get('datetimes', []), s1_meta.get('directions', []))

    ni_path = ni_meta.get('path', 0)
    ni_dir = ni_meta.get('direction', '')[:3].upper()
    ni_line1 = f"NISAR  T{ni_path} {ni_dir}"
    ni_utc = ""
    if ni_meta.get('urls'):
        ni_utc = _extract_utc_time_nisar(ni_meta['urls'][0])

    # Figure setup
    fig_w, fig_h = 10, 8
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="black")

    # Main image axis (full width)
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.82])
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#444444')

    # Title
    fig.text(0.5, 0.96, f"{cfg['title']} — {target_date}", color="white",
             fontsize=16, fontweight="bold", ha="center", va="top")

    # Band labels
    fig.text(0.15, 0.92, "Sentinel-1 (C-band)", color="cyan", fontsize=12,
             ha="center", va="bottom", fontweight="bold")
    fig.text(0.85, 0.92, "NISAR (L-band)", color="orange", fontsize=12,
             ha="center", va="bottom", fontweight="bold")

    # Attribution
    fig.text(0.5, 0.015, "Image analysis by David Bekaert", color="#888888",
             fontsize=10, ha="center", va="bottom", style="italic")

    # Scale bar (bottom-left)
    sb_km = cfg.get("scale_bar_km", 20)
    out_res = cfg.get("out_res", 100)
    sb_px = sb_km * 1000 / out_res
    y0 = h - h * 0.06
    x0 = w * 0.05
    x1 = x0 + sb_px
    ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.5, solid_capstyle="butt", zorder=20)
    tick_h = h * 0.015
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=20)
    ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=20)
    ax.text((x0 + x1) / 2, y0 - tick_h * 3, f"{sb_km} km", color="white",
            fontsize=10, ha="center", va="top", zorder=20)

    # Create frames directory
    frames_dir = cache_dir / "slider_frames"
    frames_dir.mkdir(exist_ok=True)

    # Generate slider positions: middle → right → middle → left → middle
    # Total n_frames split into phases
    phase_frames = n_frames // 4
    positions = []
    
    # Middle to right (reveal full S1)
    for i in range(phase_frames):
        pos = 0.5 + 0.5 * (i / phase_frames)
        positions.append(pos)
    
    # Pause at right
    for _ in range(phase_frames // 2):
        positions.append(1.0)
    
    # Right to left (reveal full NISAR)
    for i in range(phase_frames * 2):
        pos = 1.0 - (i / (phase_frames * 2))
        positions.append(pos)
    
    # Pause at left
    for _ in range(phase_frames // 2):
        positions.append(0.0)
    
    # Left to middle
    for i in range(phase_frames):
        pos = 0.5 * (i / phase_frames)
        positions.append(pos)

    print(f"Rendering {len(positions)} frames...")

    for frame_idx, slider_pos in enumerate(positions):
        # Create composite image
        # slider_pos: 0 = full NISAR, 1 = full S1, 0.5 = split
        divider_col = int(slider_pos * w)

        # Composite: S1 on left side, NISAR on right side
        composite = np.zeros((h, w))
        if divider_col > 0:
            composite[:, :divider_col] = s1_norm[:, :divider_col]
        if divider_col < w:
            composite[:, divider_col:] = ni_norm[:, divider_col:]

        # Handle NaN for display
        composite = np.where(np.isfinite(composite), composite, 0.5)

        ax.clear()
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        # Show composite
        ax.imshow(composite, cmap='gray', vmin=0, vmax=1, aspect='equal', origin='upper')

        # Draw vertical slider line
        if 0 < divider_col < w:
            ax.axvline(divider_col, color='cyan', linewidth=2, zorder=15)
            # Slider handle
            handle_y = h / 2
            ax.plot([divider_col], [handle_y], 'o', color='cyan', markersize=12,
                    markeredgecolor='white', markeredgewidth=2, zorder=16)

        # Metadata text (left side for S1, right side for NISAR)
        # Only show if that side is visible
        if slider_pos > 0.1:  # S1 visible
            ax.text(0.03, 0.97, s1_line1, color="white", fontsize=10,
                    ha="left", va="top", transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                    zorder=18)
            if s1_utc:
                ax.text(0.03, 0.91, s1_utc, color="#cccccc", fontsize=9,
                        ha="left", va="top", transform=ax.transAxes,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
                        zorder=18)

        if slider_pos < 0.9:  # NISAR visible
            ax.text(0.97, 0.97, ni_line1, color="white", fontsize=10,
                    ha="right", va="top", transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                    zorder=18)
            if ni_utc:
                ax.text(0.97, 0.91, ni_utc, color="#cccccc", fontsize=9,
                        ha="right", va="top", transform=ax.transAxes,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
                        zorder=18)

        # Re-draw scale bar
        ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.5, solid_capstyle="butt", zorder=20)
        ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=20)
        ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=20)
        ax.text((x0 + x1) / 2, y0 - tick_h * 3, f"{sb_km} km", color="white",
                fontsize=10, ha="center", va="top", zorder=20)

        frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=120, facecolor="black",
                    bbox_inches='tight', pad_inches=0.02)

        if (frame_idx + 1) % 20 == 0:
            print(f"  [{frame_idx + 1}/{len(positions)}]")

    plt.close(fig)

    # Compile GIF
    if output_path is None:
        output_path = str(animations_dir / f"{site_key}_{target_date}_slider.gif")

    print(f"Compiling GIF: {output_path}")
    
    try:
        from PIL import Image
        frames = sorted(frames_dir.glob("frame_*.png"))
        images = [Image.open(f) for f in frames]
        # 50ms per frame = 20fps for smooth slider
        images[0].save(
            output_path, save_all=True, append_images=images[1:],
            duration=50, loop=0
        )
    except Exception as e:
        print(f"  Error compiling GIF: {e}")
        return None

    size_before = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Raw GIF: {size_before:.1f} MB")

    _compress_gif(output_path, lossy=40)

    size_after = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final: {size_after:.1f} MB")
    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create slider reveal animation")
    parser.add_argument("--site", default="jakobshavn", help="Site key")
    parser.add_argument("--date", default="2025-11-21", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames")
    parser.add_argument("--output", help="Output GIF path")
    args = parser.parse_args()

    create_slider_animation(
        site_key=args.site,
        target_date=args.date,
        output_path=args.output,
        n_frames=args.frames,
    )


if __name__ == "__main__":
    main()
