"""Base classes for spectra download sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from spectra_download.http_client import download_json
from spectra_download.models import Spectrum

logger = logging.getLogger(__name__)


class SpectraSource(ABC):
    """Base class for spectra data sources.

    Each source should implement how to build URLs and parse responses. The base
    class centralizes logging, retries, and general download flow.
    """

    name: str

    def __init__(self, *, timeout: int = 30, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def build_request_url(self, identifier: str, extra_params: Dict[str, Any]) -> str:
        """Build the request URL for the spectra identifier."""

    @abstractmethod
    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[Spectrum]:
        """Parse response payload into spectra records."""

    def download(self, identifier: str, extra_params: Dict[str, Any]) -> List[Spectrum]:
        """Download spectra for a single identifier using the unified flow."""

        url = self.build_request_url(identifier, extra_params)
        logger.info(
            "Downloading spectra",
            extra={"source": self.name, "identifier": identifier, "url": url},
        )
        payload = download_json(url, timeout=self.timeout, max_retries=self.max_retries)
        spectra = self.parse_response(payload, identifier)
        logger.info(
            "Download complete",
            extra={"source": self.name, "identifier": identifier, "count": len(spectra)},
        )
        return spectra
