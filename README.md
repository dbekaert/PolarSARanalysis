# Polar Caps – Sentinel-1 SAR Ice Monitoring

Monitoring polar ice-cap changes and glacier calving fronts using
**Sentinel-1 C-band SAR GRD** imagery via the **Microsoft Planetary Computer**
STAC API.

**No account, no authentication, no API key required.**

---

## Repository Structure

```
GEE_polarcaps/
├── README.md                    ← you are here
├── requirements.txt             ← pip dependencies (flexible)
├── requirements-lock.txt        ← pinned environment (exact replication)
├── pyproject.toml               ← project metadata
├── .gitignore
│
├── calving_sites.py             ← calving-front time-lapse pipeline
├── coverage_catalog.py          ← per-track per-day coverage catalog
├── run_calving_pipeline.sh      ← end-to-end calving pipeline script
│
├── polar_ice_change.py          ← ice-extent time-series analysis
├── animate_polar_weekly.py      ← weekly amplitude animation (GIF)
├── stac_utils.py                ← shared Planetary Computer / STAC helpers
│
├── calibration_comparison.py    ← S-1 calibration comparison tool
└── output/                      ← created at runtime (git-ignored)
```

---

## 1  Quick Start

```bash
cd ~/Software/GEE_polarcaps

# Create a virtual environment (Python 3.12 recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (exact versions)
pip install -r requirements-lock.txt

# --- Calving-front time-lapse (main workflow) ---
# Run the full pipeline (catalog → fetch → render → compress):
bash run_calving_pipeline.sh

# Or step by step for a single site:
python coverage_catalog.py jakobshavn --start 2015-01-01 --end 2026-03-31
python calving_sites.py --fetch --site jakobshavn
python calving_sites.py --render --site jakobshavn

# --- Polar-wide analyses ---
python polar_ice_change.py
python animate_polar_weekly.py
```

That's it — no sign-up, no tokens, no auth.

---

## 2  How It Works

All Sentinel-1 GRD data is accessed through the
[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
STAC API.  The scripts use **all** Sentinel-1 acquisition modes:

| Mode | Swath | Polarisation | Primary area |
|------|-------|-------------|--------------|
| **EW** (Extra-Wide) | ~400 km | HH + HV | Sea-ice, open ocean |
| **IW** (Interferometric Wide) | ~250 km | VV + VH | Land, coastal areas |

Both modes contribute to polar mosaics, maximising spatial coverage.

| What | How |
|---|---|
| Catalog discovery | `pystac-client` queries the STAC API |
| URL signing | `planetary-computer` adds SAS tokens (transparent, no account) |
| Raster loading | `odc-stac` streams COGs into `xarray` + `dask` arrays |
| Compositing | `xarray.median(dim="time")` for temporal mosaics |
| Coastlines & borders | `cartopy` renders boundaries on polar projections |
| Animation | `Pillow` assembles annotated frames into GIF |

---

## 3  Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.12 | Tested with 3.12.3 |
| Internet connection | — | To reach the Planetary Computer API |
| No account needed | ✓ | Data access is anonymous |
| gifsicle | optional | GIF compression (`sudo apt install gifsicle`) |

### Install dependencies

```bash
# Exact replication (recommended)
pip install -r requirements-lock.txt

# Or flexible version ranges
pip install -r requirements.txt
```

Key packages: `pystac-client`, `planetary-computer`, `odc-stac`, `xarray`,
`rioxarray`, `dask`, `numpy`, `pandas`, `matplotlib`, `Pillow`, `cartopy`,
`scipy`, `rasterio`.

---

## 4  Scripts

### 4.1  Calving-Front Time-Lapse Pipeline

Produces monthly animated GIFs of Sentinel-1 SAR backscatter over major
glacier calving fronts.  The pipeline covers five sites across both poles:

| Site | Region | Extent |
|---|---|---|
| **Jakobshavn** | Greenland (W) | 100 × 60 km |
| **Petermann** | Greenland (NW) | 100 × 100 km |
| **79° North** | Greenland (NE) | 160 × 160 km |
| **Pine Island** | Antarctica (W) | 100 × 100 km |
| **Thwaites** | Antarctica (W) | 160 × 160 km |

The pipeline has four steps (see `run_calving_pipeline.sh` for details):

1. **Catalog** — `coverage_catalog.py` scans the STAC API and selects the
   best acquisition per month per site (by coverage, orbit, timeliness)
2. **Fetch** — `calving_sites.py --fetch` downloads and caches SAR data
   as calibrated dB `.npy` arrays
3. **Render** — `calving_sites.py --render` produces per-site GIFs with
   coastlines, scale bars, globe insets, and sensor metadata
4. **Combined** — `calving_sites.py --render --combined` composites all
   sites into a single grid GIF with a shared timeline

```bash
# Full pipeline
bash run_calving_pipeline.sh

# Single site fetch + render
python calving_sites.py --site jakobshavn --start_date 2015-01-01

# Render only (from cache), synced timeline, custom resolution
python calving_sites.py --render --site jakobshavn petermann --sync --render_res 100
```

| Flag | Default | Description |
|---|---|---|
| `--fetch` | — | Fetch data only (skip render) |
| `--render` | — | Render only (from cache) |
| `--site` | `all` | Site(s) to process |
| `--start_date` | `2015-01-01` | Start date (YYYY-MM-DD) |
| `--end_date` | `2026-03-31` | End date |
| `--cmap` | `gray` | matplotlib colour-map |
| `--fps` | `4` | GIF frames per second |
| `--vis_min` / `--vis_max` | `15` / `35` | dB range for grayscale mapping |
| `--sync` | off | Align all sites to shared month timeline |
| `--combined` | off | Produce combined multi-panel GIF |
| `--render_res` | `200` | Render resolution in metres |

**Rendering optimisations:**
- SAR data fetched at native 100 m, rendered at 200 m (2×2 area-mean
  downsampling) to reduce memory ~4×
- Figure context reused across frames (avoids re-creating cartopy
  projections each frame)
- GIFs assembled by streaming PNGs from disk (two frames in memory)
- Individual GIFs compressed with `gifsicle --lossy=80`
- Combined GIF uses a global 256-colour palette (no dithering) for
  sharp text, compressed with `gifsicle --lossy=30`

**Combined grid layout:**
```
┌───────────────────────┐
│      Jakobshavn       │
├───────────┬───────────┤
│ Petermann │ 79° North │
├───────────┼───────────┤
│Pine Island│ Thwaites  │
└───────────┴───────────┘
  + shared timeline bar    + attribution strip
```


### 4.2  `animate_polar_weekly.py` – Weekly Polar Amplitude Animation

Generates animated GIFs of Sentinel-1 backscatter amplitude at weekly
intervals.  Each frame is a 7-day median mosaic in polar-stereographic
projection overlaid with coastlines and country boundaries.

```bash
# Full year, both poles (default)
python animate_polar_weekly.py

# Arctic melt season with viridis colour-map
python animate_polar_weekly.py --region arctic \
    --start_date 2024-05-01 --end_date 2024-10-01 --cmap viridis

# No coastline overlay, slower animation
python animate_polar_weekly.py --no_boundaries --fps 2
```

| Flag | Default | Description |
|---|---|---|
| `--region` | `both` | `arctic`, `antarctic`, or `both` |
| `--start_date` | 12 months ago | Start date (YYYY-MM-DD) |
| `--end_date` | today | End date |
| `--cmap` | `plasma` | Any matplotlib colour-map (`inferno`, `viridis`, `gray`, `cividis`, `Greys_r`, `bone`, `coolwarm`, …) |
| `--no_boundaries` | off | Disable coastline & country boundary overlay |
| `--fps` | `4` | Frames per second |
| `--out_dir` | `./output` | Output directory |

**Outputs:**

| File | Description |
|---|---|
| `arctic_weekly_amplitude.gif` | Animated GIF – Arctic |
| `antarctic_weekly_amplitude.gif` | Animated GIF – Antarctic |
| `frames/*.png` | Individual annotated weekly frames |
| `*_highlight_{first,mid,last}_*.png` | Three static snapshots per pole |
| `colorbar_backscatter.png` | Colour-bar legend |


### 4.3  `polar_ice_change.py` – Ice Extent Time-Series

Builds monthly composites from all Sentinel-1 modes, classifies ice vs.
open water using a backscatter threshold, and tracks ice extent over time.

```bash
# Both poles, 2024 to present
python polar_ice_change.py

# Arctic only, custom date range, custom colour-map
python polar_ice_change.py --region arctic \
    --start_date 2020-01-01 --end_date 2025-12-31 --cmap Blues_r
```

| Flag | Default | Description |
|---|---|---|
| `--region` | `both` | `arctic`, `antarctic`, or `both` |
| `--start_date` | `2024-01-01` | Start date |
| `--end_date` | today | End date |
| `--cmap` | `Blues_r` | Colour-map for seasonal composites |
| `--no_boundaries` | off | Disable coastline overlay |
| `--out_dir` | `./output` | Output directory |

**Outputs:**

| File | Description |
|---|---|
| `*_ice_timeseries.csv` | Monthly ice-extent area (km²) |
| `*_ice_timeseries.png` | Trend plot |
| `*_winter_*.png` / `*_summer_*.png` | Seasonal composite maps with coastlines |


### 4.4  `stac_utils.py` – Shared STAC Module

Thin wrapper around `pystac-client` + `planetary-computer`:

- `get_catalog()` — returns a signed STAC client
- `search_s1_grd()` — search all modes (or filter by EW/IW)
- `get_copol_band()` — returns `'hh'` or `'vv'` for a given item

---

## 5  Methodology

### Sentinel-1 Data

- **Collection**: `sentinel-1-grd` on Planetary Computer
- **Format**: Cloud-Optimized GeoTIFF (COG)
- **Modes**: EW (sea-ice, HH/HV) + IW (land, VV/VH) — both included
- **Values**: Digital Numbers (DN), converted via $10 \cdot \log_{10}(\text{DN})$
- **Orbit**: Ascending + descending combined

### Ice Classification

Co-pol backscatter threshold on $10 \cdot \log_{10}(\text{DN})$ scale:

- **HH ≥ 27 dB** → ice  (EW mode)
- **VV ≥ 25 dB** → ice  (IW mode)
- Otherwise → open water / no data

> First-order heuristic for trend detection.  For operational ice charting,
> use ML classifiers or fuse with passive microwave (AMSR-2).

### Coastlines & Boundaries

Maps use **cartopy** to overlay:
- Coastlines (solid white)
- Country borders (dashed white)
- Latitude/longitude gridlines (dotted, 30° spacing)

### Seasonal Composites

| Hemisphere | Summer | Winter |
|---|---|---|
| Arctic | Jun – Sep | Dec – Mar |
| Antarctic | Dec – Mar | Jun – Sep |

### Inclination Gap

Sentinel-1's sun-synchronous orbit (98.18° inclination) leaves a small
~3°-diameter hole at each geographic pole that is never imaged.

---

## 6  Comparison: Planetary Computer vs. Google Earth Engine

| | Planetary Computer | GEE |
|---|---|---|
| **Auth required** | No | Yes (Google account + OAuth) |
| **Data format** | COGs on Azure Blob | GEE ImageCollection |
| **Processing** | Local (xarray/dask) | Server-side (GEE compute) |
| **Cost** | Free | Free (non-commercial) |
| **Python API** | Standard STAC ecosystem | Proprietary `ee` API |
| **Offline capable** | Yes (once data cached) | No |

---

## 7  Limitations & Future Work

- **Threshold sensitivity** — fixed dB thresholds may misclassify thin ice or wind-roughened water
- **Cross-mode calibration** — HH and VV are merged naively; a more rigorous approach would apply offset corrections
- **Coverage gaps** — S-1 EW acquisitions became consistent from ~2017 onward
- **Memory** — full-pole composites at high resolution need significant RAM; the scripts use Dask for lazy loading
- **Ice type** — binary mask; does not distinguish first-year / multi-year ice

---

## 8  References

- Planetary Computer S1 GRD: <https://planetarycomputer.microsoft.com/dataset/sentinel-1-grd>
- ESA Sentinel-1 User Guide: <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar>
- NSIDC Sea Ice Index: <https://nsidc.org/data/seaice_index>
- cartopy: <https://scitools.org.uk/cartopy/>
- odc-stac docs: <https://odc-stac.readthedocs.io/>
- pystac-client: <https://pystac-client.readthedocs.io/>

---

## License

This code is provided as-is for research and educational purposes.
