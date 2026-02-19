"""Shared PolarBase TAP source logic for ESPaDOnS and NARVAL."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List
from urllib.parse import urlencode

from spectra_download.models import SpectrumRecord
from spectra_download.sources.base import SpectraSource
from spectra_download.sources.tap import _cone_where, _resolve_name_to_icrs_degrees, _tap_records

logger = logging.getLogger(__name__)

_HD_RE = re.compile(r"^\s*HD\s*0*([0-9]+)\s*$", re.IGNORECASE)


def _adql_string(value: str) -> str:
    """ADQL single-quoted string with escaped quotes."""

    v = (value or "").replace("'", "''")
    return f"'{v}'"


def _polarbase_identifier_variants(identifier: str) -> List[str]:
    """Generate a small set of likely `Object` variants for PolarBase."""

    ident = (identifier or "").strip()
    if not ident:
        return [identifier]
    out = [ident]

    m = _HD_RE.match(ident)
    if m:
        n = int(m.group(1))
        out.append(f"HD  {n:5d}")  # matches VizieR-like spacing: 'HD  18474'
        out.append(f"HD {n}")      # common single-space variant

    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _escape_like(value: str) -> str:
    # Basic escaping for ADQL LIKE. (TAPVizieR uses SQL-like semantics.)
    return (value or "").replace("%", "\\%").replace("_", "\\_").replace("'", "''")


def _relabel_primary_identifier(
    spectra: List[SpectrumRecord],
    *,
    primary: str,
    query_identifier: str,
) -> List[SpectrumRecord]:
    out: List[SpectrumRecord] = []
    for s in spectra:
        md = dict(s.metadata)
        md.setdefault("query_identifier", query_identifier)
        md["identifier"] = primary
        out.append(SpectrumRecord(spectrum_id=s.spectrum_id, source=s.source, metadata=md))
    return out


class PolarBaseSource(SpectraSource):
    """Base source for PolarBase instruments using TAPVizieR."""

    tap_url = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
    table = '"B/polarbase/polarbase"'
    _inst_value: str

    def _apply_coordinate_corrections(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        extra_params: Dict[str, Any],
        identifier: str,
    ) -> tuple[float, float]:
        # Subclasses can override (e.g. ESPaDOnS proper-motion correction).
        return ra_deg, dec_deg

    def download(  # type: ignore[override]
        self,
        identifier: str,
        extra_params: Dict[str, Any] | None = None,
        *,
        raw_save_path=None,
        zarr_paths=None,
    ) -> List[SpectrumRecord]:
        """Download PolarBase spectra with identifier, coordinate, and LIKE fallbacks."""

        extra_params = self._normalize_download_extra_params(
            extra_params,
            raw_save_path=raw_save_path,
            zarr_paths=zarr_paths,
        )
        use_like = bool(extra_params.pop("use_like", True))
        use_coords = bool(extra_params.pop("use_coords", True))
        radius_arcsec = float(extra_params.pop("search_radius_arcsec", 5.0) or 5.0)
        # Avoid writing not-found for intermediate fallback attempts.
        not_found_path = extra_params.pop("not_found_path", None)

        logger.info(
            "%s download start",
            self.name,
            extra={
                "source": self.name,
                "identifier": identifier,
                "use_like": use_like,
                "use_coords": use_coords,
                "radius_arcsec": radius_arcsec,
            },
        )

        # First: exact matches on likely Object variants.
        candidates = _polarbase_identifier_variants(identifier)
        last: List[SpectrumRecord] = []
        for cand in candidates:
            spectra = super().download(
                cand,
                dict(extra_params),
                raw_save_path=raw_save_path,
                zarr_paths=zarr_paths,
            )
            if spectra:
                return _relabel_primary_identifier(spectra, primary=identifier, query_identifier=cand)
            last = spectra

        if use_coords:
            # Second: coordinate cone search (name -> ICRS coords).
            ra_deg = extra_params.pop("query_ra_deg", None)
            dec_deg = extra_params.pop("query_dec_deg", None)
            if ra_deg is None or dec_deg is None:
                try:
                    ra_deg, dec_deg = _resolve_name_to_icrs_degrees(identifier)
                except Exception:
                    ra_deg = dec_deg = None  # type: ignore[assignment]

            if ra_deg is not None and dec_deg is not None:
                ra_deg, dec_deg = self._apply_coordinate_corrections(
                    ra_deg=float(ra_deg),
                    dec_deg=float(dec_deg),
                    extra_params=extra_params,
                    identifier=identifier,
                )
                cone = _cone_where(
                    ra_deg=float(ra_deg),
                    dec_deg=float(dec_deg),
                    radius_arcsec=radius_arcsec,
                    ra_field="RA_ICRS",
                    dec_field="DE_ICRS",
                )
                params = dict(extra_params)
                params["identifier_field"] = None
                existing_where = params.get("where")
                params["where"] = f"({existing_where}) AND ({cone})" if existing_where else cone
                spectra = super().download(
                    identifier,
                    params,
                    raw_save_path=raw_save_path,
                    zarr_paths=zarr_paths,
                )
                if spectra:
                    return _relabel_primary_identifier(
                        spectra,
                        primary=identifier,
                        query_identifier=f"COORDS:{float(ra_deg):.6f},{float(dec_deg):.6f},r={radius_arcsec}arcsec",
                    )
                last = spectra

        if use_like:
            # Third: LIKE match fallback.
            params = dict(extra_params)
            params["identifier_field"] = None
            params["where"] = f"Object LIKE '%{_escape_like(identifier)}%'"
            spectra = super().download(
                identifier,
                params,
                raw_save_path=raw_save_path,
                zarr_paths=zarr_paths,
            )
            if spectra:
                return _relabel_primary_identifier(spectra, primary=identifier, query_identifier=f"LIKE:%{identifier}%")
            last = spectra

        if not last and not_found_path is not None:
            record_params = dict(extra_params)
            record_params["not_found_path"] = not_found_path
            self._record_not_found(identifier, extra_params=record_params, reason="no_records")

        return last

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:  # type: ignore[override]
        """Build a TAP request URL for PolarBase spectra."""

        tap_url = str(extra_params.get("tap_url") or self.tap_url)
        table = str(extra_params.get("table") or self.table)
        if "identifier_field" in extra_params:
            identifier_field = extra_params.get("identifier_field")
        else:
            identifier_field = "Object"
        select_fields = extra_params.get("select_fields") or ["*"]
        limit = extra_params.get("limit")

        conditions: List[str] = [f"Inst='{self._inst_value}'"]
        if identifier_field:
            conditions.append(f'{identifier_field}={_adql_string(identifier)}')

        stokes = extra_params.get("stokes")
        if stokes:
            conditions.append(f"stokes={_adql_string(str(stokes))}")
        normalized = extra_params.get("normalized")
        if normalized is not None:
            conditions.append(f"normalized={1 if bool(normalized) else 0}")

        extra_where = extra_params.get("where")
        if extra_where:
            conditions.append(str(extra_where))
        extra_conditions = extra_params.get("conditions") or []
        for c in extra_conditions:
            if c:
                conditions.append(str(c))

        top_clause = f"TOP {int(limit)} " if limit else ""
        select_clause = ", ".join([str(f) for f in select_fields])
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT {top_clause}{select_clause} FROM {table} WHERE {where_clause}"

        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": query,
        }
        return f"{tap_url}?{urlencode(params)}"

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[SpectrumRecord]:  # type: ignore[override]
        """Map TAP records to `SpectrumRecord`s with `access_url` set."""

        records = _tap_records(payload)
        spectra: List[SpectrumRecord] = []
        for i, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            metadata = dict(record)
            access_url = metadata.get("url")
            if access_url and "access_url" not in metadata:
                metadata["access_url"] = access_url
            metadata.setdefault("identifier", identifier)

            spectrum_id = metadata.get("ID") or metadata.get("recno") or f"{identifier}_{i}"
            spectra.append(
                SpectrumRecord(
                    spectrum_id=str(spectrum_id),
                    source=self.name,
                    metadata=metadata,
                )
            )
        return spectra
