#!/usr/bin/env python3
"""Download a few NARVAL spectra via PolarBase SSA."""

import re
from pathlib import Path

import astropy.units as u
import pyvo
import requests
from astropy.coordinates import SkyCoord

POLARBASE_SSA_URL = "https://www.polarbase.ovgso.fr/download/ssa_polarbase?"


def _resolve_to_coord(identifier: str) -> SkyCoord:
    s = identifier.strip()
    try:
        return SkyCoord.from_name(s, frame="icrs")
    except Exception as e:
        raise ValueError(f"Could not resolve '{identifier}'.") from e


def _narval_mask(tab) -> list[bool]:
    candidate_cols = [
        "instrument_name", "instrument", "instr", "obs_collection",
        "facility_name", "collection", "title", "target_name"
    ]
    text_cols = [c for c in candidate_cols if c in tab.colnames]
    if not text_cols:
        return [True] * len(tab)
    mask = []
    for row in tab:
        blob = " ".join(str(row[c]) for c in text_cols).lower()
        mask.append("narval" in blob)
    return mask


def _get_access_url(row) -> str | None:
    for key in ("access_url", "acref", "accessURL", "url"):
        if key in row.colnames:
            v = str(row[key])
            if v and v.lower() != "nan":
                return v
    return None


def download_narval_spectra(
    identifier: str,
    outdir: str | Path = "narval_data",
    radius_arcsec: float = 5.0,
    max_spectra: int | None = 5,
    overwrite: bool = False,
    timeout: int = 60,
) -> list[Path]:
    """Download PolarBase NARVAL spectra for a given identifier."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    coord = _resolve_to_coord(identifier)
    svc = pyvo.dal.SSAService(POLARBASE_SSA_URL)
    size_deg = (radius_arcsec * u.arcsec).to(u.deg).value
    results = svc.search(pos=(coord.ra.deg, coord.dec.deg), size=size_deg)

    if len(results) == 0:
        raise RuntimeError(
            f"No PolarBase spectra found within {radius_arcsec}\" of {identifier}."
        )

    tab = results.to_table()
    narval_mask = _narval_mask(tab)
    tab = tab[narval_mask]

    if len(tab) == 0:
        raise RuntimeError(
            f"PolarBase returned spectra near {identifier}, but none were NARVAL."
        )

    downloaded: list[Path] = []
    session = requests.Session()
    rows = list(tab)
    if max_spectra:
        rows = rows[:max_spectra]

    for row in rows:
        url = _get_access_url(row)
        if not url:
            continue
        fname = url.split("/")[-1]
        outpath = outdir / fname
        if outpath.exists() and not overwrite:
            downloaded.append(outpath)
            continue
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        outpath.write_bytes(r.content)
        downloaded.append(outpath)

    return downloaded


if __name__ == "__main__":
    paths = download_narval_spectra("HD195564", max_spectra=5)
    print(f"Downloaded {len(paths)} NARVAL spectra:")
    for p in paths:
        print(f"  {p}")
