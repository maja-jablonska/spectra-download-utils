"""Downloader for the MassBank spectra archive."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

from spectra_download.models import Spectrum
from spectra_download.sources.base import SpectraSource

logger = logging.getLogger(__name__)


class MassBankSource(SpectraSource):
    """MassBank spectra downloads using the search endpoint."""

    name = "massbank"
    base_url = "https://massbank.eu/MassBank/api/spectra"

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:
        # MassBank uses accession or compound name searches; preserve caller params.
        params = {
            "accession": identifier,
            "format": "json",
            **extra_params,
        }
        return f"{self.base_url}?{urlencode(params)}"

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[Spectrum]:
        spectra_records = payload.get("records", [])
        if not spectra_records:
            logger.warning("No spectra found", extra={"source": self.name, "identifier": identifier})
        spectra: List[Spectrum] = []
        for record in spectra_records:
            spectra.append(
                Spectrum(
                    spectrum_id=record.get("accession", identifier),
                    source=self.name,
                    peaks=record.get("peaks", []),
                    metadata={
                        "compound": record.get("compound_name"),
                        "ion_mode": record.get("ion_mode"),
                    },
                )
            )
        return spectra
