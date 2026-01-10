"""Shared data models for spectra downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Spectrum:
    """Represents a single spectrum returned by a source."""

    spectrum_id: str
    source: str
    intensity: ArrayLike
    wavelength: ArrayLike
    normalized: bool
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CCF:
    """Represents a cross-correlation function (CCF) product.

    This is primarily used for persistence/conversion (e.g., writing to Zarr)
    when CCFs require different parsing than regular spectra.
    """

    ccf_id: str
    source: str
    velocity: ArrayLike
    ccf: ArrayLike
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
    spectra: List[Spectrum]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True when spectra were downloaded successfully."""

        return self.error is None
