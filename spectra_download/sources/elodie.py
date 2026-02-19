"""Downloader for the ELODIE archive via the atlas HTML interface.

The ELODIE atlas does not expose a stable JSON API for the products we want.
Instead, we follow the same flow used in `elodie.ipynb`:

- Object search page (by object name) -> spectrum FITS links
- Object search page -> "search_ccf" pages -> "get_ccf" FITS links

This source returns `Spectrum` records whose `metadata` contains direct download
URLs (we keep `peaks` empty because we do not parse FITS content here).
"""

from __future__ import annotations

import logging
import re
import time
from astropy.coordinates import Angle
import astropy.units as u
import numpy as np
from html.parser import HTMLParser
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error, request
from urllib.parse import parse_qs, quote, urljoin, urlparse
from astropy.io import fits as astro_fits
from io import BytesIO

from spectra_download.models import CCFRecord, SpectrumRecord
from spectra_download.sources.base import SpectraSource
from spectra_download.sources.keys import DataKeys, ObservedFrame, SpectrumKeys

logger = logging.getLogger(__name__)

_URL_WHITESPACE_RE = re.compile(r"[\t\r\n ]+")


def _sanitize_url(url: str) -> str:
    """Best-effort URL sanitizer for HTML-derived hrefs.

    ELODIE pages sometimes contain literal spaces in query strings (e.g. `... =[x ='y']`).
    `urllib` refuses URLs with control characters, so we percent-encode whitespace.
    """

    url = (url or "").strip()
    if not url:
        return url
    return _URL_WHITESPACE_RE.sub("%20", url)


def _normalize_base_url(base_url: str) -> str:
    """Normalize base_url to an absolute HTTP(S) URL with a trailing slash."""

    base_url = (base_url or "").strip()
    if not base_url:
        return base_url

    parsed = urlparse(base_url)
    if not parsed.scheme:
        # Common notebook usage is to provide `atlas.obs-hp.fr/elodie/` without scheme.
        base_url = f"http://{base_url.lstrip('/')}"

    if not base_url.endswith("/"):
        base_url = f"{base_url}/"
    return base_url


def _safe_angle_deg(value: Any, *, unit: u.UnitBase) -> float | None:
    """Best-effort Angle -> degree conversion (return None on parse failure)."""

    if value is None:
        return None
    try:
        return Angle(value, unit=unit).degree
    except Exception:  # noqa: BLE001 - best-effort parsing for inconsistent headers
        return None


def _split_zarr_paths(zarr_paths: Any) -> tuple[Any | None, Any | None]:
    if not zarr_paths:
        return None, None
    if isinstance(zarr_paths, (list, tuple)) and len(zarr_paths) >= 2:
        return zarr_paths[0], zarr_paths[1]
    if isinstance(zarr_paths, (str, os.PathLike)):
        path_str = str(zarr_paths)
        if path_str.endswith(".zarr"):
            return zarr_paths, f"{path_str[:-5]}_ccf.zarr"
        return zarr_paths, f"{path_str}_ccf"
    return None, None


class _AnchorHTMLParser(HTMLParser):
    """Minimal anchor extractor (stdlib-only) for ELODIE HTML pages."""

    def __init__(self) -> None:
        super().__init__()
        self._in_a = False
        self._href: Optional[str] = None
        self._text_parts: List[str] = []
        self.anchors: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:  # noqa: D401
        if tag.lower() != "a":
            return
        href: Optional[str] = None
        for key, value in attrs:
            if key.lower() == "href":
                href = value
                break
        if not href:
            return
        self._in_a = True
        self._href = href
        self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._in_a or not self._href:
            return
        text = "".join(self._text_parts).strip()
        self.anchors.append((self._href, text))
        self._in_a = False
        self._href = None
        self._text_parts = []


def _unique_in_order(urls: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _extract_anchors(html: str) -> List[Tuple[str, str]]:
    parser = _AnchorHTMLParser()
    parser.feed(html)
    return parser.anchors


def _spectrum_id_from_url(url: str, *, fallback: str) -> str:
    """Create a stable-ish ID for an ELODIE product URL."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if "o" in params and params["o"]:
            return params["o"][0]
        if parsed.path:
            tail = parsed.path.rsplit("/", 1)[-1]
            return tail or fallback
    except Exception:  # noqa: BLE001 - best-effort ID extraction
        pass
    return fallback


def _extract_spectrum_fits_urls(html: str, *, base_url: str) -> List[str]:
    """Extract spectrum FITS URLs from the object search page HTML."""
    fits_urls: List[str] = []
    for href, text in _extract_anchors(html):
        full = _sanitize_url(urljoin(base_url, href))
        haystack = f"{href} {text}".lower()
        # In practice the atlas uses `a=mime:application/fits` and link text often
        # includes `get_spec` (as in the notebook).
        is_fits = "mime:application/fits" in haystack or full.lower().endswith(".fits")
        is_spec = "get_spec" in haystack or ("spec" in text.lower() and "get_ccf" not in haystack)
        is_ccf = "get_ccf" in haystack or "ccf" in text.lower()
        if is_fits and is_spec and not is_ccf:
            fits_urls.append(full)
    return _unique_in_order(fits_urls)


def _extract_search_ccf_page_urls(html: str, *, base_url: str) -> List[str]:
    """Extract 'search_ccf' page URLs from the object search page HTML."""
    pages: List[str] = []
    for href, text in _extract_anchors(html):
        haystack = f"{href} {text}".lower()
        if "search_ccf" in haystack:
            pages.append(_sanitize_url(urljoin(base_url, href)))
    return _unique_in_order(pages)


def _extract_ccf_fits_urls(search_ccf_html: str, *, base_url: str) -> List[str]:
    """Extract 'get_ccf' FITS URLs from a search_ccf page HTML."""
    fits_urls: List[str] = []
    for href, text in _extract_anchors(search_ccf_html):
        haystack = f"{href} {text}".lower()
        if "get_ccf" not in haystack:
            continue
        # Matches notebook logic: exclude HTML pages, keep the data download links.
        if "html" in href.lower():
            continue
        full = _sanitize_url(urljoin(base_url, href))
        fits_urls.append(full)
    return _unique_in_order(fits_urls)


class ElodieSource(SpectraSource):
    """ELODIE spectra downloads via the atlas HTML pages (spectrum + optional CCF)."""

    name = "elodie"

    # Matches `elodie.ipynb`.
    base_url = "http://atlas.obs-hp.fr/elodie/"
    
    def extract_ccf_arrays_from_fits_payload(self, *, fits_bytes: bytes, spectrum: SpectrumRecord):
        fits = astro_fits.open(BytesIO(fits_bytes))
        return {"ccf": np.atleast_1d(np.array(fits[0].data, dtype=np.float64))}, {}
    
    def extract_spectrum_arrays_from_fits_payload(
        self, *, fits_bytes: bytes, spectrum: SpectrumRecord
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract Zarr-ready arrays for ELODIE spectra from FITS bytes."""
        fits = astro_fits.open(BytesIO(fits_bytes))
        if len(fits) != 3:
            raise ValueError(f"ELODIE FITS files should have 3 extensions. This fits file has {len(fits)} extensions.")
        hdr = fits[0].header
        n = hdr["NAXIS1"]
        crval = hdr["CRVAL1"]      # in units of 0.1 nm
        cdelt = hdr["CDELT1"]
        crpix = hdr["CRPIX1"]
        arrays: Dict[str, Any] = {
            SpectrumKeys.wavelengths.value: (crval + (np.arange(n) + 1 - crpix) * cdelt),
            SpectrumKeys.intensity.value: np.atleast_1d(np.array(fits[0].data, dtype=np.float64)),
            SpectrumKeys.error.value: np.atleast_1d(np.array(fits[2].data, dtype=np.float64))
        }
        info: Dict[str, Any] = {
            DataKeys.exptime.value: hdr["EXPTIME"],
            DataKeys.date.value: hdr["DATE-OBS"],
            DataKeys.mjd.value: hdr["MJD-OBS"],
            DataKeys.airmass.value: hdr["AIRMASS"],
            DataKeys.object.value: hdr["OBJECT"],
            DataKeys.berv.value: hdr["BERV"],
            DataKeys.frame.value: ObservedFrame.observer.value,
            DataKeys.normalized.value: False,
        }
        ra_deg = _safe_angle_deg(hdr.get("ALPHA"), unit=u.hourangle)
        if ra_deg is not None:
            info[DataKeys.ra.value] = ra_deg
        dec_deg = _safe_angle_deg(hdr.get("DELTA"), unit=u.deg)
        if dec_deg is not None:
            info[DataKeys.dec.value] = dec_deg
        return arrays, info

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:
        base_url = _normalize_base_url(str(extra_params.get("base_url") or self.base_url))
        obj_safe = quote(identifier)
        return f"{base_url}fE.cgi?ob=objname,dataset,imanum&c=o&o={obj_safe}"

    def _fetch_html(self, url: str) -> str:
        url = _sanitize_url(url)
        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Fetching HTML",
                    extra={"source": self.name, "url": url, "attempt": attempt, "timeout": self.timeout},
                )
                with request.urlopen(url, timeout=self.timeout) as response:
                    # ELODIE pages are simple HTML; utf-8 works in practice.
                    html = response.read().decode("utf-8", errors="replace")
                    logger.debug(
                        "HTML fetched",
                        extra={"source": self.name, "url": url, "attempt": attempt, "chars": len(html)},
                    )
                    return html
            except error.URLError as exc:
                last_error = str(exc)
                logger.warning(
                    "Download attempt failed",
                    extra={"source": self.name, "url": url, "attempt": attempt, "error": last_error},
                )
                time.sleep(0.5 * attempt)
        raise RuntimeError(f"Failed to download {url}: {last_error}")

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[SpectrumRecord]:
        html = payload.get("html")
        if not isinstance(html, str):
            raise TypeError("ELODIE parse_response expects payload['html'] as a string")

        # When `download()` is called with an overridden base_url, we must preserve
        # that during URL-joining; otherwise downstream saving may fetch from the
        # wrong host.
        base_url = payload.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            base_url = self.base_url
        base_url = _normalize_base_url(base_url)

        spectra_urls = _extract_spectrum_fits_urls(html, base_url=base_url)
        logger.info(
            "ELODIE spectra links extracted",
            extra={"source": self.name, "identifier": identifier, "count": len(spectra_urls)},
        )
        spectra: List[SpectrumRecord] = []
        for idx, url in enumerate(spectra_urls, start=1):
            spectrum_id = _spectrum_id_from_url(url, fallback=f"{identifier}:spectrum:{idx}")
            spectra.append(
                SpectrumRecord(
                    spectrum_id=spectrum_id,
                    source=self.name,
                    metadata={
                        "identifier": identifier,
                        "product": "spectrum",
                        "access_url": url,
                    },
                )
            )
        return spectra

    def download(
        self,
        identifier: str,
        extra_params: Optional[Dict[str, Any]] = None,
        *,
        # Convenience kwargs for consistency with other sources and notebooks.
        raw_save_path: str | os.PathLike[str] | None = None,
        zarr_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
        not_found_path: str | os.PathLike[str] | None = None,
        error_path: str | os.PathLike[str] | None = None,
    ) -> List[SpectrumRecord]:
        """Download ELODIE spectra (and optionally CCF) for an object identifier.

        Optional persistence (same semantics as the base class post-processing):
            - raw_save_path: directory to write raw FITS bytes
            - zarr_paths: Zarr store path (or list of store paths) to write structured outputs

        Note:
            This source overrides `download()` because it scrapes HTML rather than using
            a JSON API, but it still delegates persistence to `postprocess_downloaded_spectra()`.
        """

        extra_params = self._normalize_download_extra_params(
            extra_params,
            raw_save_path=raw_save_path,
            zarr_paths=zarr_paths,
            not_found_path=not_found_path,
            error_path=error_path,
        )
        # Default to identifier-based naming for persisted outputs unless the caller
        # explicitly selects a strategy.
        if (
            "filename_strategy" not in extra_params
            and (
                extra_params.get("raw_save_path")
                or extra_params.get("save_dir")
                or extra_params.get("save_path")
                or extra_params.get("zarr_paths")
            )
        ):
            extra_params["filename_strategy"] = "identifier"

        try:
            base_url = _normalize_base_url(str(extra_params.get("base_url") or self.base_url))
            include_ccf = bool(extra_params.get("include_ccf", True))

            # Preserve any extra_params in case subclasses extend build_request_url behavior.
            url = self.build_request_url(identifier, {**extra_params, "base_url": base_url})
            logger.info(
                "Downloading ELODIE object page",
                extra={"source": self.name, "identifier": identifier, "url": url, "include_ccf": include_ccf},
            )
            html = self._fetch_html(url)

            # Spectra from the object page.
            spectra = self.parse_response({"html": html, "base_url": base_url}, identifier)

            if not include_ccf:
                spectra = self.postprocess_downloaded_spectra(spectra, extra_params=extra_params)
                logger.info(
                    "Download complete",
                    extra={"source": self.name, "identifier": identifier, "count": len(spectra)},
                )
                return spectra

            # CCF products (object page -> search_ccf pages -> get_ccf FITS links)
            search_ccf_pages = _extract_search_ccf_page_urls(html, base_url=base_url)
            logger.info(
                "ELODIE search_ccf pages extracted",
                extra={"source": self.name, "identifier": identifier, "count": len(search_ccf_pages)},
            )
            ccf_urls: List[str] = []
            for page_url in search_ccf_pages:
                logger.debug(
                    "Fetching search_ccf page",
                    extra={"source": self.name, "identifier": identifier, "url": page_url},
                )
                page_html = self._fetch_html(page_url)
                ccf_urls.extend(_extract_ccf_fits_urls(page_html, base_url=base_url))
            ccf_urls = _unique_in_order(ccf_urls)
            logger.info(
                "ELODIE CCF links extracted",
                extra={"source": self.name, "identifier": identifier, "count": len(ccf_urls)},
            )

            for idx, ccf_url in enumerate(ccf_urls, start=1):
                ccf_id = _spectrum_id_from_url(ccf_url, fallback=f"{identifier}:ccf:{idx}")
                spectra.append(
                    CCFRecord(
                        spectrum_id=ccf_id,
                        source=self.name,
                        metadata={
                            "identifier": identifier,
                            "product": "ccf",
                            "access_url": ccf_url,
                        },
                    )
                )

            logger.info(
                "Download complete",
                extra={"source": self.name, "identifier": identifier, "count": len(spectra)},
            )
            if not spectra:
                self._record_not_found(identifier, extra_params=extra_params, reason="no_records")

            zarr_paths = extra_params.get("zarr_paths")
            spec_zarr, ccf_zarr = _split_zarr_paths(zarr_paths)
            if include_ccf and spec_zarr and ccf_zarr:
                spectra_only = [s for s in spectra if not isinstance(s, CCFRecord)]
                ccf_only = [s for s in spectra if isinstance(s, CCFRecord)]

                saved: List[SpectrumRecord] = []
                if spectra_only:
                    spec_params = dict(extra_params, zarr_paths=spec_zarr, process_ccf=False)
                    saved.extend(self.postprocess_downloaded_spectra(spectra_only, extra_params=spec_params))
                if ccf_only:
                    ccf_params = dict(extra_params, zarr_paths=ccf_zarr, process_ccf=True)
                    saved.extend(self.postprocess_downloaded_spectra(ccf_only, extra_params=ccf_params))
                return saved

            return self.postprocess_downloaded_spectra(spectra, extra_params=extra_params)
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            self._record_error(identifier, extra_params=extra_params, stage="download", error=exc)
            raise
