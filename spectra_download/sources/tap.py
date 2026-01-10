"""Shared TAP/ADQL helper source for astronomy archives."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlencode

from spectra_download.models import Spectrum
from spectra_download.sources.base import SpectraSource

logger = logging.getLogger(__name__)


def _cone_where(
    *,
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    ra_field: str = "s_ra",
    dec_field: str = "s_dec",
) -> str:
    """Return an ADQL WHERE fragment for a cone search around (ra, dec).

    Uses ObsCore canonical fields `s_ra` and `s_dec` (ICRS degrees) by default.
    """

    rad_deg = float(radius_arcsec) / 3600.0
    # ADQL geometry functions expect degrees for CIRCLE radius.
    return (
        "1 = CONTAINS("
        f"POINT('ICRS', {ra_field}, {dec_field}), "
        f"CIRCLE('ICRS', {float(ra_deg)}, {float(dec_deg)}, {rad_deg})"
        ")"
    )


def _resolve_name_to_icrs_degrees(name: str) -> Tuple[float, float]:
    """Resolve an object name to (ra_deg, dec_deg) using astropy (Sesame).

    This requires network access and the `astropy` package.
    """

    try:
        from astropy.coordinates import SkyCoord  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Name resolution requires `astropy` (SkyCoord.from_name).") from exc

    coord = SkyCoord.from_name(name)
    return float(coord.icrs.ra.deg), float(coord.icrs.dec.deg)


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

    def download(self, identifier: str, extra_params: Dict[str, Any] | None = None) -> List[Spectrum]:  # type: ignore[override]
        """Download using TAP, optionally trying fallback identifiers.

        If `extra_params["fallback_identifiers"]` is provided, candidates are tried
        in order until a non-empty result is returned.

        Intended usage for FGK notebooks:
            - identifier: primary star name
            - fallback_identifiers: [star_alt1, star_alt2, ...]
            - identifier_field: "target_name" (ESO ObsCore)

        Saving/naming uses the *primary* identifier; successful queries annotate
        `metadata["query_identifier"]` with the candidate that matched.
        """

        extra_params = dict(extra_params or {})
        fallbacks = extra_params.get("fallback_identifiers") or []
        use_coords = bool(extra_params.get("use_coords", False))
        radius_arcsec = float(extra_params.get("search_radius_arcsec", 5.0) or 5.0)
        ra_field = str(extra_params.get("ra_field", "s_ra"))
        dec_field = str(extra_params.get("dec_field", "s_dec"))

        # Avoid spamming not-found entries for intermediate attempts.
        not_found_path = extra_params.pop("not_found_path", None)

        # Default to target_name for name-based lookups if the caller didn't specify.
        extra_params.setdefault("identifier_field", "target_name")

        candidates: List[str] = [identifier]
        for c in fallbacks:
            if c and isinstance(c, str):
                candidates.append(c)

        last: List[Spectrum] = []
        for cand in candidates:
            params = dict(extra_params)
            # Remove fallbacks so we don't recurse/duplicate tries through super().
            params.pop("fallback_identifiers", None)

            # Coordinate-based query (preferred): resolve name -> cone search.
            if use_coords:
                try:
                    ra_deg, dec_deg = _resolve_name_to_icrs_degrees(cand)
                except Exception:
                    ra_deg = dec_deg = None  # type: ignore[assignment]
                if ra_deg is not None and dec_deg is not None:
                    cone = _cone_where(
                        ra_deg=ra_deg,
                        dec_deg=dec_deg,
                        radius_arcsec=radius_arcsec,
                        ra_field=ra_field,
                        dec_field=dec_field,
                    )
                    existing_where = params.get("where")
                    params["where"] = f"({existing_where}) AND ({cone})" if existing_where else cone
                    # Disable identifier equality condition; rely on cone.
                    params["identifier_field"] = None
                    # Try this candidate; if it yields results, relabel to primary.
                    try:
                        spectra = super().download(identifier, params)
                    except Exception:
                        continue
                    if spectra:
                        relabeled: List[Spectrum] = []
                        for s in spectra:
                            md = dict(s.metadata)
                            md.setdefault("query_identifier", cand)
                            md.setdefault("query_ra_deg", ra_deg)
                            md.setdefault("query_dec_deg", dec_deg)
                            md.setdefault("search_radius_arcsec", radius_arcsec)
                            md["identifier"] = identifier
                            relabeled.append(
                                Spectrum(
                                    spectrum_id=s.spectrum_id,
                                    source=s.source,
                                    intensity=s.intensity,
                                    wavelength=s.wavelength,
                                    normalized=s.normalized,
                                    metadata=md,
                                )
                            )
                        return relabeled
                    last = spectra
                    continue

            try:
                spectra = super().download(cand, params)
            except Exception:
                # Errors are already recorded via error_path in the base class if enabled.
                continue
            if spectra:
                # Re-label to the primary identifier for consistent saving.
                relabeled: List[Spectrum] = []
                for s in spectra:
                    md = dict(s.metadata)
                    md.setdefault("query_identifier", cand)
                    md["identifier"] = identifier
                    relabeled.append(
                        Spectrum(
                            spectrum_id=s.spectrum_id,
                            source=s.source,
                            intensity=s.intensity,
                            wavelength=s.wavelength,
                            normalized=s.normalized,
                            metadata=md,
                        )
                    )
                return relabeled
            last = spectra

        # None matched: record not-found for the primary identifier once.
        if not_found_path is not None:
            extra_params["not_found_path"] = not_found_path
        self._record_not_found(identifier, extra_params=extra_params, reason="no_records")
        return last

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:
        tap_url = extra_params.get("tap_url", self.tap_url)
        table = extra_params.get("table", self.table)
        identifier_field = extra_params.get("identifier_field", self.identifier_field)
        select_fields = extra_params.get("select_fields") or list(self.select_fields) or ["*"]
        limit = extra_params.get("limit")

        conditions: List[str] = []
        if identifier_field:
            conditions.append(f"{identifier_field}='{identifier}'")
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
            if peaks:
                # Preserve any "peaks" payloads from archives in metadata.
                metadata["peaks"] = peaks
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
                    intensity=[],
                    wavelength=[],
                    normalized=False,
                    metadata={**metadata, "identifier": metadata.get("identifier") or identifier},
                )
            )
        return spectra
