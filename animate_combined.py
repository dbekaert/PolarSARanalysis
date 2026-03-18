#!/usr/bin/env python3
"""
Create creative visualizations showing S1/NISAR differences and combinations.

Visualizations:
1. RGB Composite: R=NISAR, G=difference, B=S1 (highlights sensor-specific features)
2. Difference Animation: Rapid flicker between sensors to highlight changes
3. "Combined Intelligence": Weighted fusion showing best of both sensors

Usage:
    python animate_combined.py [--site jakobshavn] [--date 2025-11-21] [--output FILE]
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Circle


# Site config
SITE_CFG = {
    "jakobshavn": {
        "title": "Jakobshavn Glacier",
        "center_lat": 69.17,
        "center_lon": -50.2,
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
    m = re.search(r'_(\d{8}T\d{6})_', url)
    if m:
        ts = m.group(1)
        return f"{ts[9:11]}:{ts[11:13]} UTC"
    return ""


def _extract_utc_time_s1(datetimes: list, directions: list = None) -> str:
    """Extract UTC time(s) from S1 datetime strings."""
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
        return f"{plat_str} T{orbits[0]} {dir_str}"
    
    pass_info = [f"T{orb}" for orb in sorted(set(orbits))]
    dir_str = "/".join(d[:3].upper() for d in sorted(directions)) if directions else ""
    
    return f"{plat_str} {'+'.join(pass_info)} {dir_str}"


def create_combined_visualization(
    site_key: str = "jakobshavn",
    target_date: str = "2025-11-21",
    output_path: str | None = None,
):
    """Create creative visualization comparing S1 and NISAR."""
    cfg = SITE_CFG[site_key]
    cache_dir = Path(f"output/nisar_s1/cache/{site_key}")
    animations_dir = Path(f"output/nisar_s1/animations/{site_key}")
    animations_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")

    manifest = json.load(open(manifest_path))
    pairs = manifest['pairs']

    # Find target date
    pair = None
    for p in pairs:
        if p['date'] == target_date:
            pair = p
            break

    if pair is None:
        dates = [p['date'] for p in pairs]
        closest = min(dates, key=lambda d: abs(
            datetime.strptime(d, "%Y-%m-%d") - datetime.strptime(target_date, "%Y-%m-%d")
        ))
        print(f"  Target date {target_date} not found, using closest: {closest}")
        pair = next(p for p in pairs if p['date'] == closest)
        target_date = closest

    print(f"Creating combined visualization for {site_key} / {target_date}")

    # Load data
    s1_arr = np.load(cache_dir / pair['s1']['file'])
    nisar_arr = np.load(cache_dir / pair['nisar']['file'])

    # Compute stretch
    s1_valid = s1_arr[np.isfinite(s1_arr) & (s1_arr != 0)]
    ni_valid = nisar_arr[np.isfinite(nisar_arr) & (nisar_arr != 0)]
    s1_vmin, s1_vmax = np.percentile(s1_valid, 2), np.percentile(s1_valid, 98)
    ni_vmin, ni_vmax = np.percentile(ni_valid, 2), np.percentile(ni_valid, 98)

    # Normalize to 0-1
    s1_disp = np.where(np.isfinite(s1_arr) & (s1_arr != 0), s1_arr, np.nan)
    ni_disp = np.where(np.isfinite(nisar_arr) & (nisar_arr != 0), nisar_arr, np.nan)

    s1_norm = np.clip((s1_disp - s1_vmin) / (s1_vmax - s1_vmin), 0, 1)
    ni_norm = np.clip((ni_disp - ni_vmin) / (ni_vmax - ni_vmin), 0, 1)

    # Match sizes
    h = min(s1_norm.shape[0], ni_norm.shape[0])
    w = min(s1_norm.shape[1], ni_norm.shape[1])
    s1_norm = s1_norm[:h, :w]
    ni_norm = ni_norm[:h, :w]

    # Metadata
    s1_meta = pair['s1']
    ni_meta = pair['nisar']
    s1_line1 = _format_s1_track_info(s1_meta)
    s1_utc = _extract_utc_time_s1(s1_meta.get('datetimes', []), s1_meta.get('directions', []))

    ni_path = ni_meta.get('path', 0)
    ni_dir = ni_meta.get('direction', '')[:3].upper()
    ni_line1 = f"NISAR T{ni_path} {ni_dir}"
    ni_utc = _extract_utc_time_nisar(ni_meta.get('urls', [''])[0]) if ni_meta.get('urls') else ""

    # --- Create multi-panel visualization ---
    frames_dir = cache_dir / "combined_frames"
    frames_dir.mkdir(exist_ok=True)

    # Create RGB composite: R=NISAR (L-band), G=average, B=S1 (C-band)
    # This shows: cyan = S1-only features, red/yellow = NISAR-only, green/white = both
    
    # Handle NaN
    s1_clean = np.nan_to_num(s1_norm, nan=0.5)
    ni_clean = np.nan_to_num(ni_norm, nan=0.5)

    # Compute difference
    diff = ni_clean - s1_clean  # Positive = NISAR brighter, Negative = S1 brighter

    # RGB composite
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = ni_clean                         # Red = NISAR (L-band)
    rgb[:, :, 1] = (s1_clean + ni_clean) / 2        # Green = average
    rgb[:, :, 2] = s1_clean                         # Blue = S1 (C-band)

    # Alternative: color-coded difference
    # Orange = NISAR sees more, Cyan = S1 sees more, Gray = similar
    diff_rgb = np.zeros((h, w, 3))
    base_gray = (s1_clean + ni_clean) / 2
    
    # Scale difference for visibility
    diff_scaled = np.clip(diff * 3, -1, 1)  # Amplify difference
    
    # Where NISAR is brighter (diff > 0): add orange tint
    mask_nisar = diff_scaled > 0.1
    diff_rgb[:, :, 0] = np.where(mask_nisar, base_gray + diff_scaled * 0.5, base_gray)
    diff_rgb[:, :, 1] = np.where(mask_nisar, base_gray - diff_scaled * 0.2, base_gray)
    diff_rgb[:, :, 2] = np.where(mask_nisar, base_gray - diff_scaled * 0.3, base_gray)
    
    # Where S1 is brighter (diff < 0): add cyan tint
    mask_s1 = diff_scaled < -0.1
    diff_rgb[:, :, 0] = np.where(mask_s1, base_gray + diff_scaled * 0.3, diff_rgb[:, :, 0])
    diff_rgb[:, :, 1] = np.where(mask_s1, base_gray - diff_scaled * 0.3, diff_rgb[:, :, 1])
    diff_rgb[:, :, 2] = np.where(mask_s1, base_gray - diff_scaled * 0.5, diff_rgb[:, :, 2])
    
    # Neither mask: keep grayscale
    mask_similar = ~mask_nisar & ~mask_s1
    diff_rgb[:, :, 0] = np.where(mask_similar, base_gray, diff_rgb[:, :, 0])
    diff_rgb[:, :, 1] = np.where(mask_similar, base_gray, diff_rgb[:, :, 1])
    diff_rgb[:, :, 2] = np.where(mask_similar, base_gray, diff_rgb[:, :, 2])
    
    diff_rgb = np.clip(diff_rgb, 0, 1)

    # Animation: cycle through views
    # Frame 0-19: S1 only
    # Frame 20-39: NISAR only
    # Frame 40-59: RGB composite
    # Frame 60-79: Difference highlight
    # Frame 80-99: Rapid flicker (attention-grabbing)

    n_frames = 100
    views = []
    
    # Build sequence
    for i in range(20):
        views.append(('s1', None))
    for i in range(20):
        views.append(('nisar', None))
    for i in range(20):
        views.append(('rgb', None))
    for i in range(20):
        views.append(('diff', None))
    for i in range(20):
        # Rapid flicker
        views.append(('s1' if i % 2 == 0 else 'nisar', 'flicker'))

    print(f"Rendering {len(views)} frames...")

    for frame_idx, (view_type, mode) in enumerate(views):
        fig = plt.figure(figsize=(12, 9), facecolor="black")
        ax = fig.add_axes([0.05, 0.08, 0.9, 0.78])
        ax.set_facecolor("black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        if view_type == 's1':
            ax.imshow(s1_clean, cmap='gray', vmin=0, vmax=1, aspect='equal')
            label = "Sentinel-1 (C-band)"
            label_color = "cyan"
            meta_text = f"{s1_line1}  {s1_utc}"
        elif view_type == 'nisar':
            ax.imshow(ni_clean, cmap='gray', vmin=0, vmax=1, aspect='equal')
            label = "NISAR (L-band)"
            label_color = "orange"
            meta_text = f"{ni_line1}  {ni_utc}"
        elif view_type == 'rgb':
            ax.imshow(rgb, aspect='equal')
            label = "RGB Composite (R=L-band, G=avg, B=C-band)"
            label_color = "white"
            meta_text = "Cyan=C-band dominant | Yellow=L-band dominant | White=both"
        elif view_type == 'diff':
            ax.imshow(diff_rgb, aspect='equal')
            label = "Wavelength Sensitivity Differences"
            label_color = "white"
            meta_text = "Orange=L-band sees more | Cyan=C-band sees more | Gray=similar"

        # Title
        fig.text(0.5, 0.96, f"{cfg['title']} — {target_date}", color="white",
                 fontsize=16, fontweight="bold", ha="center", va="top")

        # View label
        fig.text(0.5, 0.90, label, color=label_color, fontsize=14,
                 fontweight="bold", ha="center", va="top")

        # Metadata
        fig.text(0.5, 0.04, meta_text, color="#aaaaaa", fontsize=10,
                 ha="center", va="bottom")

        # Attribution
        fig.text(0.5, 0.01, "Image analysis by David Bekaert", color="#666666",
                 fontsize=9, ha="center", va="bottom", style="italic")

        # Flicker indicator
        if mode == 'flicker':
            fig.text(0.95, 0.96, "⚡", fontsize=20, ha="right", va="top", color="yellow")

        # Scale bar
        sb_km = cfg.get("scale_bar_km", 20)
        sb_px = sb_km * 1000 / cfg.get("out_res", 100)
        y0 = h - h * 0.06
        x0 = w * 0.05
        x1 = x0 + sb_px
        tick_h = h * 0.015
        ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.5, zorder=20)
        ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=20)
        ax.plot([x1, x1], [y0 - tick_h, y0 + tick_h], color="white", linewidth=1.5, zorder=20)
        ax.text((x0 + x1) / 2, y0 - tick_h * 3, f"{sb_km} km", color="white",
                fontsize=10, ha="center", va="top", zorder=20)

        frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=100, facecolor="black",
                    bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        if (frame_idx + 1) % 25 == 0:
            print(f"  [{frame_idx + 1}/{len(views)}]")

    # Compile GIF
    if output_path is None:
        output_path = str(animations_dir / f"{site_key}_{target_date}_combined.gif")

    print(f"Compiling GIF: {output_path}")

    try:
        from PIL import Image
        frames = sorted(frames_dir.glob("frame_*.png"))
        images = [Image.open(f) for f in frames]
        # 500ms for main views, 100ms for flicker
        durations = [500] * 80 + [100] * 20
        images[0].save(
            output_path, save_all=True, append_images=images[1:],
            duration=durations, loop=0
        )
    except Exception as e:
        print(f"  Error: {e}")
        return None

    size_before = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Raw GIF: {size_before:.1f} MB")

    _compress_gif(output_path, lossy=40)

    size_after = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final: {size_after:.1f} MB")
    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create combined S1/NISAR visualization")
    parser.add_argument("--site", default="jakobshavn", help="Site key")
    parser.add_argument("--date", default="2025-11-21", help="Target date")
    parser.add_argument("--output", help="Output GIF path")
    args = parser.parse_args()

    create_combined_visualization(
        site_key=args.site,
        target_date=args.date,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
