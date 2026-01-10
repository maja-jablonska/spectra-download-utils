"""Bulk download orchestration for spectra requests."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from spectra_download.models import SpectraRequest, SpectraResult
from spectra_download.sources.base import SpectraSource

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS_CAP = 32


def _default_max_workers() -> int:
    # HPC-friendly: honor common scheduler env vars when present.
    for key in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        val = os.environ.get(key)
        if val:
            try:
                n = int(val)
                if n > 0:
                    return max(1, min(n, _DEFAULT_MAX_WORKERS_CAP))
            except Exception:
                pass
    cpu = os.cpu_count() or 4
    return max(1, min(int(cpu), _DEFAULT_MAX_WORKERS_CAP))


def bulk_download(
    requests: Iterable[SpectraRequest],
    sources: Dict[str, SpectraSource],
    *,
    max_workers: Optional[int] = None,
    per_source_max_workers: Optional[int] = None,
    on_result: Optional[Callable[[SpectraResult, int, int], None]] = None,
) -> List[SpectraResult]:
    """Run a single, unified flow for bulk spectra downloads.

    The caller provides per-source downloaders so each source can maintain its
    own request format. We unify everything else (logging, retries, and error
    handling) through this orchestration step.
    """

    # Preserve input order in the output.
    req_list: List[SpectraRequest] = list(requests)
    indexed: List[Tuple[int, SpectraRequest]] = list(enumerate(req_list))

    grouped: Dict[str, List[Tuple[int, SpectraRequest]]] = defaultdict(list)
    for idx, req in indexed:
        grouped[req.source].append((idx, req))

    # Pre-fill unknown sources synchronously.
    out_by_index: Dict[int, SpectraResult] = {}
    for source_name, items in grouped.items():
        if source_name in sources:
            continue
        logger.error("Unknown source", extra={"source": source_name})
        for idx, req in items:
            result = SpectraResult(request=req, spectra=[], error=f"Unknown source: {source_name}")
            out_by_index[idx] = result
            if on_result is not None:
                try:
                    on_result(result, len(out_by_index), len(req_list))
                except Exception:
                    logger.debug("on_result callback failed", exc_info=True)

    # Concurrency controls.
    if max_workers is None:
        max_workers = _default_max_workers()
    if per_source_max_workers is not None and per_source_max_workers < 1:
        per_source_max_workers = None

    semaphores: Dict[str, Semaphore] = {}
    if per_source_max_workers is not None:
        for source_name in grouped.keys():
            if source_name in sources:
                semaphores[source_name] = Semaphore(per_source_max_workers)

    def _run_one(idx: int, req: SpectraRequest) -> SpectraResult:
        source = sources[req.source]
        sem = semaphores.get(req.source)
        if sem is None:
            return _run_one_no_sem(idx, req, source)
        with sem:
            return _run_one_no_sem(idx, req, source)

    def _run_one_no_sem(idx: int, req: SpectraRequest, source: SpectraSource) -> SpectraResult:
        extra_params = req.extra_params or {}
        logger.info(
            "Bulk download start",
            extra={"source": req.source, "index": idx + 1, "total": len(req_list), "identifier": req.identifier},
        )
        try:
            spectra = source.download(req.identifier, extra_params)
            return SpectraResult(request=req, spectra=spectra)
        except Exception as exc:  # noqa: BLE001 - keep broad to continue bulk flow
            logger.exception("Download failed", extra={"source": req.source, "identifier": req.identifier})
            return SpectraResult(request=req, spectra=[], error=str(exc))

    # Execute in parallel (threads: HPC-friendly, avoids pickling sources).
    futures: List[Future[Tuple[int, SpectraResult]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, req in indexed:
            if idx in out_by_index:
                continue
            futures.append(executor.submit(lambda i=idx, r=req: (i, _run_one(i, r))))

        for fut in as_completed(futures):
            idx, result = fut.result()
            out_by_index[idx] = result
            if on_result is not None:
                try:
                    on_result(result, len(out_by_index), len(req_list))
                except Exception:
                    logger.debug("on_result callback failed", exc_info=True)

    return [out_by_index[i] for i in range(len(req_list))]
