"""Shared data models for spectra downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SpectrumRecord:
    """Represents a single spectrum returned by a source."""

    spectrum_id: str
    source: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CCFRecord:
    """Represents a cross-correlation function (CCF) product.

    This is primarily used for persistence/conversion (e.g., writing to Zarr)
    when CCFs require different parsing than regular spectra.
    """

    spectrum_id: str
    source: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class SpectraRequest:
    """Input record for a single spectrum download request."""

    source: str
    identifier: str
    extra_params: Optional[Dict[str, Any]] = None


@dataclass
class SpectraResult:
    """Outcome of a bulk download for a single request."""

    request: SpectraRequest
    spectra: List[SpectrumRecord]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True when spectra were downloaded successfully."""

        return self.error is None
