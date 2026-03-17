#!/usr/bin/env python3
"""
stac_utils.py – Shared utilities for Sentinel-1 GRD access via
Microsoft Planetary Computer STAC API.

No authentication is required.

Supports EW (Extra-Wide, ~400 km swath, HH/HV) and IW (Interferometric
Wide, ~250 km, VV/VH) modes.  EW is the default for polar monitoring —
it provides the most consistent wall-to-wall coverage above 60° latitude.
IW is used mainly over land (Greenland, Scandinavia, etc.).
"""

import pystac_client
import planetary_computer


# ---------------------------------------------------------------------------
# STAC catalog
# ---------------------------------------------------------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-grd"


def get_catalog() -> pystac_client.Client:
    """Return a signed Planetary Computer STAC client."""
    return pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_s1_grd(catalog: pystac_client.Client,
                   bbox: list[float],
                   datetime_range: str,
                   max_items: int = 500,
                   mode: str | None = None) -> list:
    """
    Search for Sentinel-1 GRD items in any mode.

    Parameters
    ----------
    catalog : pystac_client.Client
    bbox : [west, south, east, north]
    datetime_range : e.g. "2024-01-01/2024-01-07"
    max_items : int
    mode : str, optional
        If set (e.g. 'EW' or 'IW'), restrict to that mode.
        If None, return all modes.

    Returns
    -------
    list of pystac.Item
    """
    query = {}
    if mode:
        query["sar:instrument_mode"] = {"eq": mode}

    results = catalog.search(
        collections=[S1_COLLECTION],
        bbox=bbox,
        datetime=datetime_range,
        query=query if query else None,
        max_items=max_items,
    )
    return list(results.items())


def get_copol_band(item) -> str | None:
    """
    Determine the co-pol band name available in a STAC item.

    EW mode → 'hh';  IW mode → 'vv' (fallback to 'hh' if present).
    Returns None if no suitable band found.
    """
    if "hh" in item.assets:
        return "hh"
    if "vv" in item.assets:
        return "vv"
    return None
