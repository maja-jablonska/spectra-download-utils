"""Downloader for CAFE spectra from the Calar Alto Archive (CALTO).

CAFE data is hosted by the Calar Alto Archive at CAB (INTA-CSIC). The public
web UI is a POST form (`/calto/jsp/searchform.jsp`) that returns an HTML results
page containing per-dataset download links.

Those download links (`/calto/servlet/FetchSci?...`) return a ZIP file even for
single datasets. This source therefore:

- Scrapes the HTML results page to obtain FetchSci URLs
- Overrides `fetch_fits_payload()` to download the ZIP and extract the FITS file

This allows the standard `SpectraSource` persistence flow (raw_save_path, zarr)
to operate on the extracted FITS bytes.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from http.cookiejar import CookieJar
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import logging

from spectra_download.http_client import download_bytes
from spectra_download.models import SpectrumRecord
from spectra_download.sources.base import SpectraSource
from spectra_download.sources.tap import _resolve_name_to_icrs_degrees


class CafeSource(SpectraSource):
    """CAFE spectra downloads via the CALTO HTML interface."""

    name = "cafe"
    # Main search form endpoint (POST).
    search_url = "https://caha.sdc.cab.inta-csic.es/calto/jsp/searchform.jsp"

    _HD_RE = re.compile(r"^\s*HD\s*0*([0-9]+)\s*$", re.IGNORECASE)
    _HIP_RE = re.compile(r"^\s*HIP\s*0*([0-9]+)\s*$", re.IGNORECASE)

    _log = logging.getLogger(__name__)

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:  # type: ignore[override]
        # Not used (we override `download()`), but keep for completeness.
        return self.search_url

    def _fetch_results_html(self, *, form: Any, debug_request_path: str | None = None) -> str:
        """POST the search form and return HTML."""

        self._log.info(
            "CALTO search",
            extra={
                "source": self.name,
                "objID": form.get("objID") if isinstance(form, dict) else None,
                "page": form.get("pag") if isinstance(form, dict) else None,
                "limit": form.get("result") if isinstance(form, dict) else None,
            },
        )
        if isinstance(form, dict):
            payload = {k: v for k, v in form.items() if v is not None}
        else:
            payload = form
        encoded = urlencode(payload, doseq=True)
        if debug_request_path:
            try:
                p = Path(str(debug_request_path))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(encoded, encoding="utf-8")
            except Exception:
                pass
        data = encoded.encode("utf-8")
        req = urlrequest.Request(
            self.search_url,
            data=data,
            method="POST",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": self.search_url,
                "Origin": "https://caha.sdc.cab.inta-csic.es",
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        last: Optional[str] = None
        opener = urlrequest.build_opener(urlrequest.HTTPCookieProcessor(CookieJar()))
        for attempt in range(1, self.max_retries + 1):
            try:
                # CALTO expects a session cookie set by the search form page.
                try:
                    opener.open(self.search_url, timeout=self.timeout)
                except Exception:
                    pass
                with opener.open(req, timeout=self.timeout) as resp:
                    self._log.info(
                        "CALTO response",
                        extra={"source": self.name, "status": resp.getcode(), "url": resp.geturl()},
                    )
                    html = resp.read().decode("utf-8", errors="replace")
                    self._log.debug(
                        "CALTO HTML fetched",
                        extra={"source": self.name, "objID": form.get("objID"), "chars": len(html)},
                    )
                    return html
            except urlerror.URLError as exc:
                last = str(exc)
        raise RuntimeError(f"Failed to fetch CALTO HTML: {last}")

    @staticmethod
    def _parse_fetchsci_links(html: str, *, base_url: str, tipe: str) -> List[Tuple[str, str]]:
        """Return list of (dataset_id, absolute_fetch_url) for a given tipe (raw/red)."""

        # Example:
        #   href="/calto/servlet/FetchSci?id=148663&tipe=red&t=web"
        # Accept single/double quotes and additional params; parse query to extract id/tipe.
        pat = re.compile(r"/calto/servlet/FetchSci\?[^\"'>\s]+", re.I)
        out: List[Tuple[str, str]] = []
        for href in pat.findall(html):
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            did = (params.get("id") or [None])[0]
            tipe_val = (params.get("tipe") or [None])[0]
            if not did or not tipe_val:
                continue
            if str(tipe_val).lower() != tipe.lower():
                continue
            out.append((str(did), urljoin(base_url, href)))

        # De-dup by dataset_id, preserve order.
        seen: set[str] = set()
        uniq: List[Tuple[str, str]] = []
        for did, u in out:
            if did in seen:
                continue
            seen.add(did)
            uniq.append((did, u))
        return uniq

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[SpectrumRecord]:  # type: ignore[override]
        html = payload.get("html")
        if not isinstance(html, str):
            raise TypeError("CafeSource.parse_response expects payload['html'] as a string")
        tipe = str(payload.get("tipe") or "red")
        base_url = str(payload.get("base_url") or self.search_url)

        total_fetchsci = len(re.findall(r"/calto/servlet/FetchSci\?", html, flags=re.I))
        if total_fetchsci == 0:
            self._log.warning(
                "CALTO response contains no FetchSci links",
                extra={"source": self.name, "identifier": identifier, "tipe": tipe},
            )

        spectra: List[SpectrumRecord] = []
        for did, fetch_url in self._parse_fetchsci_links(html, base_url=base_url, tipe=tipe):
            spectra.append(
                SpectrumRecord(
                    spectrum_id=str(did),
                    source=self.name,
                    metadata={
                        "identifier": identifier,
                        "product": "spectrum",
                        "tipe": tipe,
                        "access_url": fetch_url,
                    },
                )
            )
        return spectra

    def download(
        self,
        identifier: str,
        extra_params: Optional[Dict[str, Any]] = None,
        *,
        raw_save_path=None,
        zarr_paths=None,
    ) -> List[SpectrumRecord]:
        """Download CAFE spectra for an identifier via CALTO, with optional persistence."""

        extra_params = dict(extra_params or {})
        if raw_save_path is not None and "raw_save_path" not in extra_params and "save_dir" not in extra_params:
            extra_params["raw_save_path"] = raw_save_path
        if zarr_paths is not None and "zarr_paths" not in extra_params and "save_path" not in extra_params:
            extra_params["zarr_paths"] = zarr_paths

        use_coords = bool(extra_params.get("use_coords", True))
        # Form defaults.
        tipe = "red" if bool(extra_params.get("reduced", True)) else "raw"
        force_mark_all = bool(extra_params.get("force_mark_all", True))
        # Radius in decimal degrees in CALTO form.
        radius_deg = float(extra_params.get("radius_deg", 0.05) or 0.05)
        limit = int(extra_params.get("limit", 10) or 10)
        page = int(extra_params.get("page", 1) or 1)
        order = str(extra_params.get("order", "obs_date") or "obs_date")

        # Date range is required to actually execute the query.
        date_ini = extra_params.get("date_ini", (1, 1, 2008))  # (d, m, y)
        date_end = extra_params.get("date_end", (31, 12, 2025))
        di_d, di_m, di_y = date_ini
        de_d, de_m, de_y = date_end

        def _identifier_variants(value: str) -> List[str]:
            ident = (value or "").strip()
            if not ident:
                return [value]
            out = [ident]
            m = self._HD_RE.match(ident)
            if m:
                n = int(m.group(1))
                out.append(f"HD {n}")
                out.append(f"HD  {n}")
            m = self._HIP_RE.match(ident)
            if m:
                n = int(m.group(1))
                out.append(f"HIP {n}")
                out.append(f"HIP  {n}")
            seen = set()
            uniq: List[str] = []
            for s in out:
                if s not in seen:
                    uniq.append(s)
                    seen.add(s)
            return uniq

        last: List[SpectrumRecord] = []
        def _format_debug_path(base: str, *, attempt: int, obj_id: str) -> Path:
            if "{objID}" in base or "{attempt}" in base:
                return Path(base.format(objID=obj_id, attempt=attempt))
            p = Path(base)
            suffix = p.suffix or ".html"
            stem = p.name[: -len(suffix)]
            return p.with_name(f"{stem}_{attempt}{suffix}")

        def _base_form(obj_id: str) -> Dict[str, Any]:
            return {
                "objID": obj_id or "",
                "size": f"{radius_deg}",
                "dateini_d": str(int(di_d)),
                "dateini_m": f"{int(di_m):02d}",
                "dateini_y": str(int(di_y)),
                "dateend_d": str(int(de_d)),
                "dateend_m": f"{int(de_m):02d}",
                "dateend_y": str(int(de_y)),
                "cafe_red": "1" if tipe == "red" else None,
                "cafe_raw": "1" if tipe == "raw" else None,
                "result": str(limit),
                "pag": str(page),
                "order": order,
                "submit": "Submit Query",
            }

        for attempt, ident in enumerate(_identifier_variants(identifier), start=1):
            form: Dict[str, Any] = _base_form(ident)
            if force_mark_all:
                form["markAll"] = ["on", "on"]

            debug_path = extra_params.get("debug_html_path")
            req_debug = None
            if debug_path:
                req_debug = str(_format_debug_path(str(debug_path), attempt=attempt, obj_id=f"{ident}_request"))
            html = self._fetch_results_html(form=form, debug_request_path=req_debug)
            if debug_path:
                try:
                    p = _format_debug_path(str(debug_path), attempt=attempt, obj_id=ident)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(html, encoding="utf-8")
                    self._log.info(
                        "CALTO HTML saved for debugging",
                        extra={"source": self.name, "objID": ident, "path": str(p)},
                    )
                except Exception as exc:  # noqa: BLE001
                    self._log.warning(
                        "Failed to save CALTO HTML",
                        extra={"source": self.name, "objID": ident, "error": str(exc)},
                    )
            spectra = self.parse_response({"html": html, "tipe": tipe, "base_url": self.search_url}, identifier)
            self._log.info(
                "CALTO results parsed (count=%s)",
                len(spectra),
                extra={"source": self.name, "objID": ident, "identifier": identifier},
            )
            if spectra:
                return self.postprocess_downloaded_spectra(spectra, extra_params=extra_params)
            last = spectra

        if use_coords:
            ra_deg = extra_params.get("query_ra_deg")
            dec_deg = extra_params.get("query_dec_deg")
            if ra_deg is None or dec_deg is None:
                try:
                    ra_deg, dec_deg = _resolve_name_to_icrs_degrees(identifier)
                except Exception:
                    self._log.warning(
                        "CALTO coordinate resolution failed",
                        extra={"source": self.name, "identifier": identifier},
                    )
                    ra_deg = dec_deg = None
            if ra_deg is not None and dec_deg is not None:
                coord_ident = f"{float(ra_deg):.6f} {float(dec_deg):.6f}"
                self._log.info(
                    "CALTO coords lookup",
                    extra={"source": self.name, "identifier": identifier, "objID": coord_ident},
                )
                form = _base_form(coord_ident)
                if force_mark_all:
                    form["markAll"] = ["on", "on"]
                req_debug = None
                if debug_path:
                    req_debug = str(_format_debug_path(str(debug_path), attempt=attempt + 1, obj_id=f"{coord_ident}_request"))
                html = self._fetch_results_html(form=form, debug_request_path=req_debug)
                debug_path = extra_params.get("debug_html_path")
                if debug_path:
                    try:
                        p = _format_debug_path(str(debug_path), attempt=attempt + 1, obj_id=coord_ident)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(html, encoding="utf-8")
                        self._log.info(
                            "CALTO HTML saved for debugging",
                            extra={"source": self.name, "objID": coord_ident, "path": str(p)},
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._log.warning(
                            "Failed to save CALTO HTML",
                            extra={"source": self.name, "objID": coord_ident, "error": str(exc)},
                        )
                spectra = self.parse_response({"html": html, "tipe": tipe, "base_url": self.search_url}, identifier)
                self._log.info(
                    "CALTO results parsed (coords count=%s)",
                    len(spectra),
                    extra={"source": self.name, "objID": coord_ident, "identifier": identifier},
                )
                if spectra:
                    return self.postprocess_downloaded_spectra(spectra, extra_params=extra_params)
                last = spectra

        return last

    def fetch_fits_payload(
        self,
        *,
        access_url: str,
        spectrum: SpectrumRecord,
    ) -> tuple[bytes, Dict[str, Any]]:
        """Download a CALTO FetchSci ZIP and extract the FITS bytes."""

        zip_bytes = download_bytes(access_url, timeout=self.timeout, max_retries=self.max_retries)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Prefer a FITS-ish member; otherwise, take the first file.
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                raise ValueError("Empty ZIP payload from CALTO FetchSci")
            fits_candidates = [n for n in members if n.lower().endswith((".fits", ".fit", ".fts", ".fits.gz"))]
            chosen = fits_candidates[0] if fits_candidates else members[0]
            data = zf.read(chosen)

        # If it's gzipped FITS, leave as-is (astropy can read .fits.gz bytes only if decompressed,
        # but persistence is raw bytes; callers can override extraction if needed).
        return data, {"caha_zip_member": chosen, "caha_fetch_url": access_url}

