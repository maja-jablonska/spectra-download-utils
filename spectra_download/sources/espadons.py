"""Downloader for ESPaDOnS spectra (PolarBase) via TAP.

PolarBase (Petit et al.) hosts high-resolution spectropolarimetric observations
from ESPaDOnS and NARVAL. We access the metadata via CDS TAPVizieR and download
the referenced FITS file using the standard `SpectraSource` persistence flow.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List
from urllib.parse import urlencode

from spectra_download.models import SpectrumRecord
from spectra_download.sources.base import SpectraSource
from spectra_download.sources.tap import _cone_where, _resolve_name_to_icrs_degrees, _tap_records

logger = logging.getLogger(__name__)


class EspadonsSource(SpectraSource):
    """ESPaDOnS spectra downloads via CDS TAPVizieR (PolarBase)."""

    name = "espadons"
    tap_url = "http://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
    table = '"B/polarbase/polarbase"'
    _inst_value = "espadons"

    def _apply_proper_motion(
        self,
        *,
        ra_deg: float,
        dec_deg: float,
        extra_params: Dict[str, Any],
    ) -> tuple[float, float]:
        """Apply proper motion correction when parameters are provided.

        Expected extra_params keys:
        - pm_ra_mas_yr / pm_dec_mas_yr (or pm_ra_cosdec_mas_yr / pm_dec_mas_yr)
        - ref_epoch_year (default 2000.0)
        - obs_epoch_year (target epoch; default ref_epoch_year)
        """

        pm_ra = (
            extra_params.get("pm_ra_cosdec_mas_yr")
            if extra_params.get("pm_ra_cosdec_mas_yr") is not None
            else extra_params.get("pm_ra_mas_yr")
        )
        pm_dec = extra_params.get("pm_dec_mas_yr")
        if pm_ra is None or pm_dec is None:
            return ra_deg, dec_deg

        ref_epoch = float(extra_params.get("ref_epoch_year", 2000.0))
        obs_epoch = float(extra_params.get("obs_epoch_year", ref_epoch))
        if obs_epoch == ref_epoch:
            return ra_deg, dec_deg

        try:
            from astropy.coordinates import SkyCoord  # type: ignore
            import astropy.units as u  # type: ignore
            from astropy.time import Time  # type: ignore
        except Exception:
            logger.warning(
                "Proper motion correction requires astropy",
                extra={"source": self.name, "identifier": extra_params.get("identifier")},
            )
            return ra_deg, dec_deg

        try:
            c = SkyCoord(
                ra=ra_deg * u.deg,
                dec=dec_deg * u.deg,
                pm_ra_cosdec=float(pm_ra) * u.mas / u.yr,
                pm_dec=float(pm_dec) * u.mas / u.yr,
                obstime=Time(ref_epoch, format="jyear"),
                frame="icrs",
            )
            c2 = c.apply_space_motion(new_obstime=Time(obs_epoch, format="jyear"))
            logger.info(
                "ESPaDOnS proper motion applied",
                extra={
                    "source": self.name,
                    "ref_epoch_year": ref_epoch,
                    "obs_epoch_year": obs_epoch,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "ra_corr_deg": float(c2.ra.deg),
                    "dec_corr_deg": float(c2.dec.deg),
                },
            )
            return float(c2.ra.deg), float(c2.dec.deg)
        except Exception:
            logger.warning(
                "Proper motion correction failed",
                extra={"source": self.name, "ref_epoch_year": ref_epoch, "obs_epoch_year": obs_epoch},
            )
            return ra_deg, dec_deg

    def download(  # type: ignore[override]
        self,
        identifier: str,
        extra_params: Dict[str, Any] | None = None,
        *,
        raw_save_path=None,
        zarr_paths=None,
    ) -> List[SpectrumRecord]:
        """Download ESPaDOnS spectra with best-effort identifier fallbacks.

        PolarBase's `Object` field is not normalized (e.g. `HD  18474` with
        multiple spaces). This method attempts:
        - exact match on Object
        - HD number normalization (e.g. `HD18474` -> `HD  18474`)
        - optional LIKE match (disabled by setting `extra_params["use_like"]=False`)

        If a fallback identifier matches, we set `metadata["query_identifier"]`
        on each returned record, while keeping `metadata["identifier"]` as the
        original input.
        """

        extra_params = dict(extra_params or {})
        use_like = bool(extra_params.pop("use_like", True))
        use_coords = bool(extra_params.pop("use_coords", True))
        radius_arcsec = float(extra_params.pop("search_radius_arcsec", 5.0) or 5.0)
        logger.info(
            "ESPaDOnS download start",
            extra={
                "source": self.name,
                "identifier": identifier,
                "use_like": use_like,
                "use_coords": use_coords,
                "radius_arcsec": radius_arcsec,
            },
        )

        # First: try exact matches on a small set of candidate identifier variants.
        candidates = _polarbase_identifier_variants(identifier)
        logger.debug(
            "ESPaDOnS identifier candidates",
            extra={"source": self.name, "identifier": identifier, "candidates": candidates},
        )
        last: List[SpectrumRecord] = []
        for cand in candidates:
            logger.info(
                "ESPaDOnS candidate lookup",
                extra={"source": self.name, "identifier": identifier, "candidate": cand},
            )
            spectra = super().download(
                cand,
                dict(extra_params),
                raw_save_path=raw_save_path,
                zarr_paths=zarr_paths,
            )
            if spectra:
                logger.info(
                    "ESPaDOnS candidate matched",
                    extra={
                        "source": self.name,
                        "identifier": identifier,
                        "candidate": cand,
                        "count": len(spectra),
                    },
                )
                return _relabel_primary_identifier(spectra, primary=identifier, query_identifier=cand)
            last = spectra

        if use_coords:
            # Third: coordinate-based cone search (name -> ICRS coords).
            ra_deg = extra_params.pop("query_ra_deg", None)
            dec_deg = extra_params.pop("query_dec_deg", None)
            if ra_deg is None or dec_deg is None:
                try:
                    ra_deg, dec_deg = _resolve_name_to_icrs_degrees(identifier)
                    logger.info(
                        "ESPaDOnS name resolved",
                        extra={
                            "source": self.name,
                            "identifier": identifier,
                            "ra_deg": ra_deg,
                            "dec_deg": dec_deg,
                        },
                    )
                except Exception:
                    logger.warning(
                        "ESPaDOnS name resolution failed",
                        extra={"source": self.name, "identifier": identifier},
                    )
                    ra_deg = dec_deg = None  # type: ignore[assignment]

            if ra_deg is not None and dec_deg is not None:
                ra_deg, dec_deg = self._apply_proper_motion(
                    ra_deg=float(ra_deg),
                    dec_deg=float(dec_deg),
                    extra_params={**extra_params, "identifier": identifier},
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
                logger.info(
                    "ESPaDOnS coords lookup",
                    extra={
                        "source": self.name,
                        "identifier": identifier,
                        "ra_deg": ra_deg,
                        "dec_deg": dec_deg,
                        "radius_arcsec": radius_arcsec,
                        "where": params["where"],
                    },
                )
                spectra = super().download(
                    identifier,
                    params,
                    raw_save_path=raw_save_path,
                    zarr_paths=zarr_paths,
                )
                if spectra:
                    logger.info(
                        "ESPaDOnS coords matched",
                        extra={
                            "source": self.name,
                            "identifier": identifier,
                            "count": len(spectra),
                        },
                    )
                    return _relabel_primary_identifier(
                        spectra,
                        primary=identifier,
                        query_identifier=f"COORDS:{float(ra_deg):.6f},{float(dec_deg):.6f},r={radius_arcsec}arcsec",
                    )
                last = spectra

        if use_like:
            # Fourth: LIKE match. We disable identifier_field equality and add a WHERE fragment.
            params = dict(extra_params)
            params["identifier_field"] = None
            params["where"] = f"Object LIKE '%{_escape_like(identifier)}%'"
            logger.info(
                "ESPaDOnS LIKE lookup",
                extra={
                    "source": self.name,
                    "identifier": identifier,
                    "where": params["where"],
                },
            )
            spectra = super().download(
                identifier,
                params,
                raw_save_path=raw_save_path,
                zarr_paths=zarr_paths,
            )
            if spectra:
                logger.info(
                    "ESPaDOnS LIKE matched",
                    extra={
                        "source": self.name,
                        "identifier": identifier,
                        "count": len(spectra),
                    },
                )
                return _relabel_primary_identifier(spectra, primary=identifier, query_identifier=f"LIKE:%{identifier}%")
            last = spectra
        logger.info(
            "ESPaDOnS download complete (no match)",
            extra={"source": self.name, "identifier": identifier, "count": len(last)},
        )
        return last

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:  # type: ignore[override]
        """Build a TAP request URL for PolarBase ESPaDOnS spectra.

        Identifier matching:
            - By default we match on `Object='<identifier>'` exactly as stored in the table.

        Optional filters via extra_params:
            - limit: TOP N
            - select_fields: list of column names to select (default '*')
            - where: additional ADQL WHERE fragment appended with AND
            - stokes: restrict to a given Stokes product (table column `stokes`)
            - normalized: restrict normalized flag (table column `normalized`)
        """

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
        url = f"{tap_url}?{urlencode(params)}"
        logger.debug(
            "ESPaDOnS TAP URL built: %s",
            url,
            extra={
                "source": self.name,
                "identifier": identifier,
                "tap_url": tap_url,
                "table": table,
                "query": query,
            },
        )
        return url

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[SpectrumRecord]:  # type: ignore[override]
        """Map TAP records to `SpectrumRecord`s with `access_url` set."""

        records = _tap_records(payload)
        logger.info(
            "ESPaDOnS TAP records parsed (count=%s)",
            len(records),
            extra={"source": self.name, "identifier": identifier},
        )
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
        if spectra:
            sample = spectra[0].metadata.get("url") or spectra[0].metadata.get("access_url")
        else:
            sample = None
        logger.debug(
            "ESPaDOnS spectra records built (count=%s)",
            len(spectra),
            extra={"source": self.name, "identifier": identifier, "sample_url": sample},
        )
        return spectra


def _adql_string(value: str) -> str:
    """ADQL single-quoted string with escaped quotes."""

    v = (value or "").replace("'", "''")
    return f"'{v}'"


_HD_RE = re.compile(r"^\s*HD\s*0*([0-9]+)\s*$", re.IGNORECASE)


def _polarbase_identifier_variants(identifier: str) -> List[str]:
    """Generate a small set of likely `Object` variants for PolarBase."""

    ident = (identifier or "").strip()
    if not ident:
        return [identifier]
    out = [ident]

    m = _HD_RE.match(ident)
    if m:
        n = int(m.group(1))
        out.append(f"HD  {n:5d}")  # matches VizieR-like spacing: 'HD␠␠18474'
        out.append(f"HD {n}")      # common single-space variant

    # De-duplicate while preserving order.
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

