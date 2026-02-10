"""Downloader for ESO archive spectra via TAP."""

from __future__ import annotations
from typing import Any, Dict

from astropy.coordinates import Angle
from astropy.io import fits as astro_fits
from io import BytesIO
import numpy as np
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

from spectra_download.models import SpectrumRecord
from spectra_download.http_client import DownloadError, download_bytes
from spectra_download.sources.base import _extract_datalink_this_url, _looks_like_votable
from spectra_download.sources.keys import DataKeys, ObservedFrame, SpectrumKeys
from spectra_download.sources.tap import TapSpectraSource


class EsoTapSource(TapSpectraSource):
    """ESO TAP spectra source for a specific instrument."""

    tap_url = "https://archive.eso.org/tap_obs/sync"
    table = "ivoa.obscore"

    _ESO_TOKEN_URL = "https://www.eso.org/sso/oidc/token"
    _ESO_OIDC_CLIENT_ID = "clientid"
    _ESO_OIDC_CLIENT_SECRET = "clientSecret"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._eso_id_token: str | None = None
        self._eso_id_token_obtained_at: float | None = None

    def _load_eso_credentials(self) -> tuple[str, str] | None:
        """Best-effort: read ESO credentials from env or local `eso.env`.

        This is intended for local notebook usage.
        """

        username = os.getenv("ESO_USERNAME")
        password = os.getenv("ESO_PASSWORD")
        if username and password:
            return username, password

        def _parse_env_file(p: Path) -> Dict[str, str]:
            out: Dict[str, str] = {}
            try:
                for raw in p.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
            except Exception:
                return {}
            return out

        # Common locations in this repo.
        candidates = [
            Path.cwd() / "eso.env",
            Path(__file__).resolve().parents[2] / "eso.env",
        ]
        for p in candidates:
            if p.exists():
                d = _parse_env_file(p)
                u = d.get("ESO_USERNAME")
                pw = d.get("ESO_PASSWORD")
                if u and pw:
                    return u, pw
        return None

    def _get_eso_id_token(self) -> str | None:
        """Get (and cache) an ESO id_token for DataPortal access."""

        # Tokens are valid for ~8h; refresh conservatively.
        if self._eso_id_token and self._eso_id_token_obtained_at:
            if (time.time() - self._eso_id_token_obtained_at) < 7.5 * 3600:
                return self._eso_id_token

        creds = self._load_eso_credentials()
        if not creds:
            return None
        username, password = creds

        # Avoid logging credentials: call the token URL directly without using
        # the shared http_client (which logs failed URLs).
        params = {
            "response_type": "id_token token",
            "grant_type": "password",
            "client_id": self._ESO_OIDC_CLIENT_ID,
            "client_secret": self._ESO_OIDC_CLIENT_SECRET,
            "username": username,
            "password": password,
        }
        url = f"{self._ESO_TOKEN_URL}?{urlencode(params)}"
        try:
            from urllib import request as urlrequest  # local import

            with urlrequest.urlopen(url, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8", errors="replace"))
            token = payload.get("id_token")
            if isinstance(token, str) and token.strip():
                self._eso_id_token = token.strip()
                self._eso_id_token_obtained_at = time.time()
                return self._eso_id_token
        except Exception:
            return None
        return None

    def _download_eso_bytes(self, url: str) -> bytes:
        """Download bytes, retrying with an ESO Bearer token on 401."""

        try:
            return download_bytes(url, timeout=self.timeout, max_retries=self.max_retries)
        except DownloadError as exc:
            if exc.status_code != 401:
                raise
            token = self._get_eso_id_token()
            if not token:
                raise
            return download_bytes(
                url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                headers={"Authorization": f"Bearer {token}"},
            )

    def fetch_fits_payload(
        self,
        *,
        access_url: str,
        spectrum: SpectrumRecord,
    ) -> tuple[bytes, Dict[str, Any]]:
        """ESO DataPortal supports token auth for proprietary files.

        Override the base implementation so resolved DataLink product URLs that
        return 401 can be retried with a Bearer token (from `ESO_USERNAME` /
        `ESO_PASSWORD`).
        """

        fits_bytes = self._download_eso_bytes(access_url)

        metadata_updates: Dict[str, Any] = {}
        datalink_hops = 0
        datalink_url = access_url
        while _looks_like_votable(fits_bytes) and datalink_hops < 3:
            resolved = _extract_datalink_this_url(fits_bytes)
            if not resolved:
                break
            metadata_updates["datalink_url"] = datalink_url
            metadata_updates["access_url"] = resolved
            datalink_url = resolved
            fits_bytes = self._download_eso_bytes(resolved)
            datalink_hops += 1

        return fits_bytes, metadata_updates


class EsoHarpsSource(EsoTapSource):
    """ESO HARPS spectra downloads."""

    name = "eso_harps"
    extra_conditions = ("instrument_name='HARPS'",)
    
    def extract_spectrum_arrays_from_fits_payload(
        self, *, fits_bytes: bytes, spectrum: SpectrumRecord
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract Zarr-ready arrays for ESO HARPS spectra from FITS bytes."""
        fits = astro_fits.open(BytesIO(fits_bytes))
        if len(fits) != 2:
            raise ValueError(f"ESO HARPS FITS files should have 2 extensions. This fits file has {len(fits)} extensions.")
        hdr = fits[0].header
        arrays: Dict[str, Any] = {
            SpectrumKeys.wavelengths.value: np.atleast_1d(np.array(fits[1].data['WAVE'], dtype=np.float64)),
            SpectrumKeys.intensity.value: np.atleast_1d(np.array(fits[1].data['FLUX'], dtype=np.float64)),
            SpectrumKeys.error.value: np.atleast_1d(np.array(fits[1].data['ERR'], dtype=np.float64))
        }
        info: Dict[str, Any] = {
            DataKeys.exptime.value: hdr['TEXPTIME'],
            DataKeys.ra.value: hdr['RA'],
            DataKeys.dec.value: hdr['DEC'],
            DataKeys.reference_frame.value: hdr['RADECSYS'],
            DataKeys.reference_frame_epoch.value: hdr['EQUINOX'],
            DataKeys.date.value: hdr['DATE-OBS'],
            DataKeys.mjd.value: hdr['MJD-OBS'],
            DataKeys.object.value: hdr['OBJECT'],
            DataKeys.frame.value: ObservedFrame.barycentric.value,
            DataKeys.normalized.value: False,
            DataKeys.snr.value: hdr['SNR']
        }
        # NOTE: HARPS S1D products store their main arrays in extension 1 as columns
        # (WAVE/FLUX/ERR). The primary HDU typically has no data payload.
        return arrays, info


class EsoNirpsSource(EsoTapSource):
    """ESO NIRPS spectra downloads."""

    name = "eso_nirps"
    extra_conditions = ("instrument_name='NIRPS'",)


class EsoUvesSource(EsoTapSource):
    """ESO UVES spectra downloads."""

    name = "eso_uves"
    extra_conditions = ("instrument_name='UVES'",)


class EsoFerosSource(EsoTapSource):
    """ESO FEROS spectra downloads."""

    name = "eso_feros"
    extra_conditions = ("instrument_name='FEROS'",)

    def extract_spectrum_arrays_from_fits_payload(
        self, *, fits_bytes: bytes, spectrum: SpectrumRecord
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract Zarr-ready arrays for ESO FEROS spectra from FITS bytes.

        FEROS spectra products commonly expose a binary table with columns:
        WAVE, FLUX, ERR (similar to HARPS).
        """

        fits = astro_fits.open(BytesIO(fits_bytes))
        hdr = fits[0].header

        arrays: Dict[str, Any] = {}
        info: Dict[str, Any] = {}

        # Prefer a bintable with WAVE/FLUX/(ERR) columns.
        if len(fits) >= 2 and getattr(fits[1], "data", None) is not None:
            data = fits[1].data
            names = set(getattr(data, "names", []) or [])
            if {"WAVE", "FLUX"}.issubset(names):
                arrays[SpectrumKeys.wavelengths.value] = np.atleast_1d(np.array(data["WAVE"], dtype=np.float64))
                arrays[SpectrumKeys.intensity.value] = np.atleast_1d(np.array(data["FLUX"], dtype=np.float64))
                if "ERR" in names:
                    arrays[SpectrumKeys.error.value] = np.atleast_1d(np.array(data["ERR"], dtype=np.float64))

        # Minimal metadata, best-effort (header keyword coverage differs across products).
        for k_src, k_dst in [
            ("TEXPTIME", DataKeys.exptime.value),
            ("RA", DataKeys.ra.value),
            ("DEC", DataKeys.dec.value),
            ("DATE-OBS", DataKeys.date.value),
            ("MJD-OBS", DataKeys.mjd.value),
            ("OBJECT", DataKeys.object.value),
            ("SNR", DataKeys.snr.value),
        ]:
            if k_src in hdr:
                info[k_dst] = hdr[k_src]
        info.setdefault(DataKeys.normalized.value, False)

        return arrays, info
