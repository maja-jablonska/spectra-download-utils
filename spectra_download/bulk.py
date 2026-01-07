"""Bulk download orchestration for spectra requests."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List

from spectra_download.models import SpectraRequest, SpectraResult
from spectra_download.sources.base import SpectraSource

logger = logging.getLogger(__name__)


def bulk_download(
    requests: Iterable[SpectraRequest],
    sources: Dict[str, SpectraSource],
) -> List[SpectraResult]:
    """Run a single, unified flow for bulk spectra downloads.

    The caller provides per-source downloaders so each source can maintain its
    own request format. We unify everything else (logging, retries, and error
    handling) through this orchestration step.
    """

    grouped: Dict[str, List[SpectraRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.source].append(request)

    results: List[SpectraResult] = []
    for source_name, source_requests in grouped.items():
        source = sources.get(source_name)
        if not source:
            logger.error("Unknown source", extra={"source": source_name})
            for request in source_requests:
                results.append(
                    SpectraResult(
                        request=request,
                        spectra=[],
                        error=f"Unknown source: {source_name}",
                    )
                )
            continue

        logger.info(
            "Processing source batch",
            extra={"source": source_name, "count": len(source_requests)},
        )
        for request in source_requests:
            extra_params = request.extra_params or {}
            try:
                spectra = source.download(request.identifier, extra_params)
                results.append(SpectraResult(request=request, spectra=spectra))
            except Exception as exc:  # noqa: BLE001 - keep broad to continue bulk flow
                logger.exception(
                    "Download failed",
                    extra={"source": source_name, "identifier": request.identifier},
                )
                results.append(
                    SpectraResult(
                        request=request,
                        spectra=[],
                        error=str(exc),
                    )
                )

    return results
