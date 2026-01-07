"""Shared TAP/ADQL helper source for astronomy archives."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import urlencode

from spectra_download.models import Spectrum
from spectra_download.sources.base import SpectraSource

logger = logging.getLogger(__name__)


def _tap_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]

    rows = payload.get("data") or payload.get("records") or payload.get("rows") or payload.get("result") or []
    if rows and isinstance(rows[0], dict):
        return rows

    metadata = payload.get("metadata") or payload.get("fields") or payload.get("columns") or []
    names = [entry.get("name") for entry in metadata if isinstance(entry, dict) and entry.get("name")]
    if not names:
        return []

    records: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Iterable):
            continue
        records.append(dict(zip(names, row)))
    return records


class TapSpectraSource(SpectraSource):
    """Base TAP/ADQL spectra source for astronomy archives."""

    tap_url: str
    table: str = "ivoa.obscore"
    identifier_field: str = "obs_id"
    select_fields: Sequence[str] = ()
    extra_conditions: Sequence[str] = ()

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:
        tap_url = extra_params.get("tap_url", self.tap_url)
        table = extra_params.get("table", self.table)
        identifier_field = extra_params.get("identifier_field", self.identifier_field)
        select_fields = extra_params.get("select_fields") or list(self.select_fields) or ["*"]
        limit = extra_params.get("limit")

        conditions = [f"{identifier_field}='{identifier}'"]
        conditions.extend(self.extra_conditions)
        extra_where = extra_params.get("where")
        if extra_where:
            conditions.append(extra_where)
        extra_conditions = extra_params.get("conditions")
        if extra_conditions:
            conditions.extend(extra_conditions)

        top_clause = f"TOP {int(limit)} " if limit else ""
        select_clause = ", ".join(select_fields)
        where_clause = " AND ".join(conditions)
        query = f"SELECT {top_clause}{select_clause} FROM {table} WHERE {where_clause}"

        params = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "json",
            "QUERY": query,
        }
        return f"{tap_url}?{urlencode(params)}"

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[Spectrum]:
        records = _tap_records(payload)
        if not records:
            logger.warning("No spectra found", extra={"source": self.name, "identifier": identifier})
        spectra: List[Spectrum] = []
        for record in records:
            peaks = record.get("peaks", []) if isinstance(record, dict) else []
            metadata = {key: value for key, value in record.items() if key != "peaks"}
            spectrum_id = (
                record.get("spectrum_id")
                or record.get("obs_id")
                or record.get("obs_publisher_did")
                or identifier
            )
            spectra.append(
                Spectrum(
                    spectrum_id=spectrum_id,
                    source=self.name,
                    peaks=peaks,
                    metadata=metadata,
                )
            )
        return spectra
