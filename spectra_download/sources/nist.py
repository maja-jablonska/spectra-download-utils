"""Downloader for the NIST Mass Spectral Library."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

from spectra_download.models import Spectrum
from spectra_download.sources.base import SpectraSource

logger = logging.getLogger(__name__)


class NistSource(SpectraSource):
    """NIST spectra downloads via the public REST API."""

    name = "nist"
    base_url = "https://chemdata.nist.gov/api/spectra"

    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:
        params = {
            "id": identifier,
            "format": "json",
            **extra_params,
        }
        return f"{self.base_url}?{urlencode(params)}"

    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[Spectrum]:
        spectra_records = payload.get("spectra", [])
        if not spectra_records:
            logger.warning("No spectra found", extra={"source": self.name, "identifier": identifier})
        spectra: List[Spectrum] = []
        for record in spectra_records:
            spectra.append(
                Spectrum(
                    spectrum_id=record.get("id", identifier),
                    source=self.name,
                    peaks=record.get("peaks", []),
                    metadata={
                        "compound": record.get("compound"),
                        "instrument": record.get("instrument"),
                    },
                )
            )
        return spectra
