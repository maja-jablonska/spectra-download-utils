"""Downloader for NARVAL spectra (PolarBase) via TAP.

PolarBase (Petit et al.) hosts high-resolution spectropolarimetric observations
from ESPaDOnS and NARVAL. The metadata is exposed via CDS TAPVizieR as the
`"B/polarbase/polarbase"` table; the `Inst` column distinguishes instruments.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlencode

from spectra_download.models import SpectrumRecord
from spectra_download.sources.base import SpectraSource
from spectra_download.sources.tap import _cone_where, _resolve_name_to_icrs_degrees, _tap_records


class NarvalSource(SpectraSource):
    """NARVAL spectra downloads via CDS TAPVizieR (PolarBase)."""

    name = "narval"
    tap_url = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
    table = '"B/polarbase/polarbase"'
    _inst_value = "narval"

    def download(  # type: ignore[override]
        self,
        identifier: str,
        extra_params: Dict[str, Any] | None = None,
        *,
        raw_save_path=None,
        zarr_paths=None,
    ) -> List[SpectrumRecord]:
        """Download NARVAL spectra with best-effort identifier fallbacks.

        See `EspadonsSource.download()` for rationale.
        """

        extra_params = dict(extra_params or {})
        use_like = bool(extra_params.pop("use_like", True))
        use_coords = bool(extra_params.pop("use_coords", True))
        radius_arcsec = float(extra_params.pop("search_radius_arcsec", 5.0) or 5.0)

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
            ra_deg = extra_params.pop("query_ra_deg", None)
            dec_deg = extra_params.pop("query_dec_deg", None)
            if ra_deg is None or dec_deg is None:
                try:
                    ra_deg, dec_deg = _resolve_name_to_icrs_degrees(identifier)
                except Exception:
                    ra_deg = dec_deg = None  # type: ignore[assignment]

            if ra_deg is not None and dec_deg is not None:
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

        return last

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:  # type: ignore[override]
        """Build a TAP request URL for PolarBase NARVAL spectra.

        Identifier matching:
            - By default we match on `Object='<identifier>'` exactly as stored in the table.
              (Examples include 'HD  18474', 'Betelgeuse', etc.)

        Optional filters via extra_params:
            - limit: TOP N
            - select_fields: list of column names to select (default '*')
            - where: additional ADQL WHERE fragment appended with AND
            - stokes: restrict to a given Stokes product (table column `stokes`)
            - normalized: restrict normalized flag (table column `normalized`)
        """

        tap_url = str(extra_params.get("tap_url") or self.tap_url)
        table = str(extra_params.get("table") or self.table)
        identifier_field = str(extra_params.get("identifier_field") or "Object")
        select_fields = extra_params.get("select_fields") or ["*"]
        limit = extra_params.get("limit")

        # Always restrict to NARVAL within PolarBase.
        conditions: List[str] = [f"Inst='{self._inst_value}'"]
        if identifier_field:
            # Note: TAPVizieR table columns are case-sensitive; use exact names like `Object`.
            conditions.append(f'{identifier_field}={_adql_string(identifier)}')

        # Common optional filters.
        stokes = extra_params.get("stokes")
        if stokes:
            conditions.append(f"stokes={_adql_string(str(stokes))}")
        normalized = extra_params.get("normalized")
        if normalized is not None:
            # Table stores 0/1-like values; accept bool-ish.
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
        """Map TAP records to `SpectrumRecord`s with `access_url` set.

        PolarBase table provides a direct FITS download link in column `url`.
        """

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


def _adql_string(value: str) -> str:
    """ADQL single-quoted string with escaped quotes."""

    v = (value or "").replace("'", "''")
    return f"'{v}'"


_HD_RE = re.compile(r"^\s*HD\s*0*([0-9]+)\s*$", re.IGNORECASE)


def _polarbase_identifier_variants(identifier: str) -> List[str]:
    ident = (identifier or "").strip()
    if not ident:
        return [identifier]
    out = [ident]
    m = _HD_RE.match(ident)
    if m:
        n = int(m.group(1))
        out.append(f"HD  {n:5d}")
        out.append(f"HD {n}")
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _escape_like(value: str) -> str:
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
