"""Base classes for spectra download sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import os
import re
import json
import time
import traceback
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from spectra_download.http_client import download_bytes, download_file, download_json
from spectra_download.fits_utils import (
    extract_1d_wavelength_intensity_from_fits_bytes,
    extract_ccf_from_fits_bytes,
    extract_data_keys_from_fits_bytes,
)
from spectra_download.sources.keys import DataKeys
from spectra_download.models import CCFRecord, SpectrumRecord

logger = logging.getLogger(__name__)

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_JSONL_LOCKS: Dict[str, threading.Lock] = {}
_ZARR_OPEN_LOCKS: Dict[str, threading.Lock] = {}


def _jsonl_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    lock = _JSONL_LOCKS.get(key)
    if lock is None:
        lock = threading.Lock()
        _JSONL_LOCKS[key] = lock
    return lock


def _zarr_open_lock(save_path: str | os.PathLike[str]) -> threading.Lock:
    key = str(Path(save_path).resolve())
    lock = _ZARR_OPEN_LOCKS.get(key)
    if lock is None:
        lock = threading.Lock()
        _ZARR_OPEN_LOCKS[key] = lock
    return lock


def _safe_filename(value: str, *, default: str = "spectrum") -> str:
    value = (value or "").strip()
    if not value:
        return default
    value = value.replace("/", "_").replace("\\", "_")
    value = _FILENAME_SAFE_RE.sub("_", value)
    value = value.strip("._-")
    return value or default


def _filename_from_access_url(access_url: str) -> str | None:
    try:
        path = urlparse(access_url).path
        if not path:
            return None
        tail = path.rsplit("/", 1)[-1]
        return tail or None
    except Exception:  # noqa: BLE001 - best-effort filename extraction
        return None


def _product_tag(spectrum: SpectrumRecord) -> str | None:
    product = spectrum.metadata.get("product")
    return str(product).strip().lower() if product is not None else None


def _json_attr_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    return _json_attr_value(value)


def _looks_like_votable(payload: bytes) -> bool:
    head = payload[:512].lstrip()
    if head.startswith(b"<?xml"):
        return b"VOTABLE" in head.upper()
    return b"<VOTABLE" in head.upper()


def _extract_datalink_this_url(votable_xml: bytes) -> str | None:
    """Extract the '#this' FITS access_url from an IVOA DataLink VOTable."""

    try:
        root = ET.fromstring(votable_xml)
    except Exception:
        return None

    # Namespace-agnostic tag handling
    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].upper()

    # Find FIELD names to map TD indices.
    field_names: List[str] = []
    for el in root.iter():
        if _local(el.tag) == "FIELD":
            name = el.attrib.get("name") or el.attrib.get("ID") or ""
            field_names.append(name)
        if _local(el.tag) == "TABLEDATA":
            break

    def _idx(name: str) -> int | None:
        name_low = name.lower()
        for i, n in enumerate(field_names):
            if (n or "").lower() == name_low:
                return i
        return None

    access_i = _idx("access_url")
    sem_i = _idx("semantics")
    ctype_i = _idx("content_type")
    if access_i is None or sem_i is None:
        return None

    # Iterate table rows.
    for tr in root.iter():
        if _local(tr.tag) != "TR":
            continue
        tds = [td.text or "" for td in tr if _local(td.tag) == "TD"]
        if len(tds) <= max(access_i, sem_i):
            continue
        semantics = (tds[sem_i] or "").strip()
        if semantics != "#this":
            continue
        access_url = (tds[access_i] or "").strip()
        if not access_url:
            continue
        if ctype_i is not None and len(tds) > ctype_i:
            ctype = (tds[ctype_i] or "").lower()
            # Prefer FITS-like payloads (but don't hard fail; some use x-fits-bintable).
            if "fits" not in ctype:
                continue
        return access_url

    # Fallback: any #this row, even without content_type match.
    for tr in root.iter():
        if _local(tr.tag) != "TR":
            continue
        tds = [td.text or "" for td in tr if _local(td.tag) == "TD"]
        if len(tds) <= max(access_i, sem_i):
            continue
        if (tds[sem_i] or "").strip() != "#this":
            continue
        access_url = (tds[access_i] or "").strip()
        if access_url:
            return access_url

    return None


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
    def parse_response(self, payload: Dict[str, Any], identifier: str) -> List[SpectrumRecord]:
        """Parse response payload into spectra records."""

    def _normalize_download_extra_params(
        self,
        extra_params: Optional[Dict[str, Any]],
        *,
        raw_save_path: str | os.PathLike[str] | None = None,
        zarr_paths: str | os.PathLike[str] | List[str | os.PathLike[str]] | None = None,
        not_found_path: str | os.PathLike[str] | None = None,
        error_path: str | os.PathLike[str] | None = None,
    ) -> Dict[str, Any]:
        """Merge convenience download kwargs into `extra_params`."""

        params = dict(extra_params or {})
        if raw_save_path is not None and "raw_save_path" not in params and "save_dir" not in params:
            params["raw_save_path"] = raw_save_path
        if zarr_paths is not None and "zarr_paths" not in params and "save_path" not in params:
            params["zarr_paths"] = zarr_paths
        if not_found_path is not None and "not_found_path" not in params:
            params["not_found_path"] = not_found_path
        if error_path is not None and "error_path" not in params:
            params["error_path"] = error_path
        return params

    def download(
        self,
        identifier: str,
        extra_params: Optional[Dict[str, Any]] = None,
        *,
        # Convenience kwargs (match ElodieSource.download) so notebooks/callers can
        # pass persistence paths without constructing extra_params by hand.
        raw_save_path: str | os.PathLike[str] | None = None,
        zarr_paths: str | os.PathLike[str] | List[str | os.PathLike[str]] | None = None,
    ) -> List[SpectrumRecord]:
        """Download spectra for a single identifier using the unified flow.

        Optional saving:
            - If `extra_params["raw_save_path"]` is provided, each returned Spectrum
              with `metadata["access_url"]` will be downloaded (binary) and written
              to that directory as a `.fits` file (exact raw bytes).
            - If `extra_params["save_path"]` (or `extra_params["zarr_paths"]`) is provided,
              a Zarr store (or multiple stores) will be created/expanded and each spectrum
              will be written via `self.write_fits_to_zarr(...)`.

              `zarr_paths` may be a single string path or a list/tuple of paths.
            - If `extra_params["save_path"]` is provided, a Zarr store at that path
              will be created/expanded and each spectrum will be written via
              `self.write_fits_to_zarr(...)`.

        Backwards compatibility:
            - `save_dir` behaves like `raw_save_path` and sets `metadata["local_path"]`.
        """

        extra_params = self._normalize_download_extra_params(
            extra_params,
            raw_save_path=raw_save_path,
            zarr_paths=zarr_paths,
        )
        try:
            url = self.build_request_url(identifier, extra_params)
            logger.info(
                "Downloading spectra",
                extra={"source": self.name, "identifier": identifier, "url": url},
            )
            payload = download_json(url, timeout=self.timeout, max_retries=self.max_retries)
            spectra = self.parse_response(payload, identifier)
            logger.info(
                "Parsed spectra response",
                extra={"source": self.name, "identifier": identifier, "count": len(spectra)},
            )

            if not spectra:
                self._record_not_found(identifier, extra_params=extra_params, reason="no_records")

            spectra = self.postprocess_downloaded_spectra(spectra, extra_params=extra_params)

            logger.info(
                "Download complete",
                extra={"source": self.name, "identifier": identifier, "count": len(spectra)},
            )
            return spectra
        except Exception as exc:  # noqa: BLE001 - optional recording, then re-raise
            self._record_error(identifier, extra_params=extra_params, stage="download", error=exc)
            raise

    def _record_not_found(self, identifier: str, *, extra_params: Dict[str, Any], reason: str) -> None:
        """Optionally append not-found identifiers to a JSONL file.

        Enable by passing `extra_params["not_found_path"]`.
        """

        path = extra_params.get("not_found_path")
        if not path:
            return
        try:
            p = Path(str(path))
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": time.time(),
                "source": self.name,
                "identifier": identifier,
                "reason": reason,
            }
            with _jsonl_lock(p):
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            logger.warning(
                "Identifier not found",
                extra={"source": self.name, "identifier": identifier, "reason": reason, "not_found_path": str(p)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to record not-found identifier",
                extra={"source": self.name, "identifier": identifier, "reason": reason, "error": str(exc)},
            )

    def _record_error(self, identifier: str, *, extra_params: Dict[str, Any], stage: str, error: Exception) -> None:
        """Optionally append download failures to a JSONL file.

        Enable by passing `extra_params["error_path"]`.
        """

        path = extra_params.get("error_path")
        if not path:
            return
        try:
            p = Path(str(path))
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": time.time(),
                "source": self.name,
                "identifier": identifier,
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            with _jsonl_lock(p):
                with p.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
            logger.warning(
                "Recorded download error",
                extra={
                    "source": self.name,
                    "identifier": identifier,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "error_path": str(p),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to record download error",
                extra={"source": self.name, "identifier": identifier, "stage": stage, "error": str(exc)},
            )

    def postprocess_downloaded_spectra(
        self,
        spectra: List[SpectrumRecord],
        *,
        extra_params: Dict[str, Any],
        process_ccf: bool | None = None,
    ) -> List[SpectrumRecord]:
        """Optional saving/transforms applied after source-specific download logic."""

        raw_save_path = extra_params.get("raw_save_path") or extra_params.get("save_dir")
        save_path = extra_params.get("save_path")
        zarr_paths = extra_params.get("zarr_paths")
        overwrite = bool(extra_params.get("overwrite", False))
        progress_every = int(extra_params.get("progress_every", 10) or 10)
        # Default to identifier-based naming when saving, falling back to spectrum_id
        # if the source did not include `metadata["identifier"]`.
        filename_strategy = str(extra_params.get("filename_strategy", "identifier"))
        set_local_path = "save_dir" in extra_params
        if process_ccf is None:
            process_ccf = bool(extra_params.get("process_ccf", True))

        save_paths: List[str] = []
        if zarr_paths:
            if isinstance(zarr_paths, (list, tuple)):
                save_paths = [str(p) for p in zarr_paths if p]
            else:
                save_paths = [str(zarr_paths)]
        elif save_path:
            save_paths = [str(save_path)]

        if not raw_save_path and not save_paths:
            return spectra

        logger.info(
            "Post-processing spectra",
            extra={
                "source": self.name,
                "count": len(spectra),
                "raw_save_path": str(raw_save_path) if raw_save_path else None,
                "save_path": str(save_paths[0]) if save_paths else None,
                "zarr_paths": save_paths if save_paths else None,
                "overwrite": overwrite,
                "filename_strategy": filename_strategy,
            },
        )

        zarr_roots: List[tuple[str, Any]] = []
        for sp in save_paths:
            zarr_roots.append((sp, self._open_zarr_root(sp)))

        return self._persist_spectra_outputs(
            spectra,
            raw_save_path=raw_save_path,
            zarr_roots=zarr_roots,
            overwrite=overwrite,
            filename_strategy=filename_strategy,
            set_local_path=set_local_path,
            process_ccf=process_ccf,
            progress_every=progress_every,
        )

    def fetch_fits_payload(
        self,
        *,
        access_url: str,
        spectrum: SpectrumRecord,
    ) -> tuple[bytes, Dict[str, Any]]:
        """Fetch FITS bytes for a spectrum.

        Override this in source subclasses to customize how the FITS payload is
        resolved, downloaded, or transformed before persistence.

        Returns:
            (fits_bytes, metadata_updates)
        """

        fits_bytes = download_bytes(access_url, timeout=self.timeout, max_retries=self.max_retries)

        metadata_updates: Dict[str, Any] = {}
        datalink_hops = 0
        datalink_url = access_url
        while _looks_like_votable(fits_bytes) and datalink_hops < 3:
            resolved = _extract_datalink_this_url(fits_bytes)
            if not resolved:
                logger.warning(
                    "DataLink VOTable detected but could not resolve #this URL",
                    extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "url": datalink_url},
                )
                break
            logger.info(
                "Resolved DataLink VOTable to product URL",
                extra={
                    "source": self.name,
                    "spectrum_id": spectrum.spectrum_id,
                    "datalink_url": datalink_url,
                    "resolved_url": resolved,
                    "hop": datalink_hops + 1,
                },
            )
            metadata_updates["datalink_url"] = datalink_url
            metadata_updates["access_url"] = resolved
            datalink_url = resolved
            fits_bytes = download_bytes(resolved, timeout=self.timeout, max_retries=self.max_retries)
            datalink_hops += 1

        return fits_bytes, metadata_updates

    def _persist_spectra_outputs(
        self,
        spectra: List[SpectrumRecord],
        *,
        raw_save_path: str | os.PathLike[str] | None,
        zarr_roots: List[tuple[str, Any]],
        overwrite: bool,
        filename_strategy: str,
        set_local_path: bool,
        process_ccf: bool,
        progress_every: int = 10,
    ) -> List[SpectrumRecord]:
        raw_dir: Path | None = None
        if raw_save_path:
            raw_dir = Path(raw_save_path)
            raw_dir.mkdir(parents=True, exist_ok=True)

        # Precompute per-identifier/product counts for naming strategies that need it.
        group_counts: Dict[tuple[str, str], int] = {}
        if filename_strategy == "identifier":
            for s in spectra:
                ident = s.metadata.get("identifier")
                ident_str = str(ident).strip() if ident is not None else ""
                if not ident_str:
                    ident_str = str(s.spectrum_id or "").strip() or self.name
                prod = _product_tag(s) or "spectrum"
                group_counts[(ident_str, prod)] = group_counts.get((ident_str, prod), 0) + 1
        group_seen: Dict[tuple[str, str], int] = {}

        saved: List[SpectrumRecord] = []
        total = len(spectra)
        for index, spectrum in enumerate(spectra, start=1):
            access_url = spectrum.metadata.get("access_url")
            if not access_url or not isinstance(access_url, str):
                logger.debug(
                    "Skipping persistence (no access_url)",
                    extra={
                        "source": self.name,
                        "spectrum_id": spectrum.spectrum_id,
                        "index": index,
                    },
                )
                saved.append(spectrum)
                continue

        # Build a persistence name (used for raw FITS filename and Zarr key).
            if filename_strategy == "access_url":
                candidate = _filename_from_access_url(access_url) or spectrum.spectrum_id
                base_name = _safe_filename(str(candidate), default=f"{self.name}_{index}")
                if base_name.lower().endswith(".fits"):
                    base_name = base_name[: -len(".fits")]
                raw_base_name = base_name
                zarr_base_name = base_name
                ccf_seen = 1
                ccf_total = 1
            elif filename_strategy == "identifier":
                ident = spectrum.metadata.get("identifier")
                ident_str = str(ident).strip() if ident is not None else ""
                if not ident_str:
                    ident_str = str(spectrum.spectrum_id or "").strip() or f"{self.name}_{index}"
                prod = _product_tag(spectrum) or "spectrum"

                total = group_counts.get((ident_str, prod), 1)
                seen = group_seen.get((ident_str, prod), 0) + 1
                group_seen[(ident_str, prod)] = seen

                # Raw filenames keep the old convention:
                #   - CCF -> {identifier}_ccf[_N].fits
                #   - spectrum -> {identifier}[_N].fits
                raw_name = ident_str
                if prod == "ccf":
                    raw_name = f"{raw_name}_ccf"
                if total > 1:
                    raw_name = f"{raw_name}_{seen}"
                raw_base_name = _safe_filename(raw_name, default=f"{self.name}_{index}")

                # Zarr keys:
                #   - spectra -> store under the object identifier group (no _N suffix),
                #     and disambiguate via dataset name when multiple spectra exist.
                #   - CCF -> store under the object identifier group (no _ccf suffix),
                #     and disambiguate via dataset name when multiple CCFs exist.
                if prod == "ccf":
                    zarr_base_name = _safe_filename(ident_str, default=f"{self.name}_{index}")
                else:
                    zarr_base_name = _safe_filename(ident_str, default=f"{self.name}_{index}")

                ccf_seen = seen
                ccf_total = total
            else:
                candidate = spectrum.spectrum_id or f"{self.name}_{index}"
                base_name = _safe_filename(str(candidate), default=f"{self.name}_{index}")
                raw_base_name = base_name
                zarr_base_name = base_name
                ccf_seen = 1
                ccf_total = 1

            filename = f"{raw_base_name}.fits"
            dest: Path | None = (raw_dir / filename) if raw_dir is not None else None

            if total:
                if index == 1 or index == total or (progress_every > 0 and index % progress_every == 0):
                    logger.info(
                        "Persistence progress",
                        extra={
                            "source": self.name,
                            "index": index,
                            "total": total,
                            "spectrum_id": spectrum.spectrum_id,
                            "identifier": spectrum.metadata.get("identifier"),
                            "product": _product_tag(spectrum),
                            # NOTE: `filename` is reserved in Python's logging LogRecord.
                            "output_filename": filename,
                            "raw_save_path": str(raw_dir) if raw_dir else None,
                            "zarr_paths": [sp for sp, _ in zarr_roots] if zarr_roots else None,
                        },
                    )

            # Caching: if the raw FITS file already exists, re-use its bytes instead of
            # re-downloading. This avoids redundant network calls when re-running
            # notebooks or bulk jobs.
            fits_bytes: bytes | None = None
            metadata_updates: Dict[str, Any] = {}
            if dest is not None and dest.exists() and not overwrite:
                try:
                    head = dest.open("rb").read(512)
                    if _looks_like_votable(head):
                        # If we previously saved a DataLink VOTable under a .fits name,
                        # do not treat it as a cache hit.
                        logger.warning(
                            "Existing raw file is a VOTable; ignoring cache and re-downloading",
                            extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "path": str(dest)},
                        )
                    else:
                        fits_bytes = dest.read_bytes()
                        logger.info(
                            "Using cached raw FITS (skip download)",
                            extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "path": str(dest)},
                        )
                except Exception as exc:  # noqa: BLE001 - best-effort cache read
                    logger.warning(
                        "Failed to read cached raw FITS; re-downloading",
                        extra={
                            "source": self.name,
                            "spectrum_id": spectrum.spectrum_id,
                            "path": str(dest),
                            "error": str(exc),
                        },
                    )
                    fits_bytes = None

            logger.debug(
                "Fetching FITS bytes for persistence",
                extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "url": access_url},
            )
            new_metadata = dict(spectrum.metadata)
            if fits_bytes is None:
                fits_bytes, metadata_updates = self.fetch_fits_payload(access_url=access_url, spectrum=spectrum)
            if metadata_updates:
                new_metadata.update(metadata_updates)
            # Provide stable persistence naming for downstream consumers.
            new_metadata["save_name"] = filename
            if filename_strategy == "identifier":
                prod = _product_tag(spectrum)
                if prod == "ccf":
                    new_metadata["_ccf_index"] = ccf_seen
                    new_metadata["_ccf_total"] = ccf_total
                else:
                    new_metadata["_spec_index"] = ccf_seen
                    new_metadata["_spec_total"] = ccf_total

            if raw_dir is not None:
                logger.info(
                    "Attempting to save raw FITS",
                    extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "path": str(raw_dir / filename)},
                )
                should_write = overwrite or not dest.exists()
                if not should_write and dest.exists():
                    # If we previously saved a DataLink VOTable under a .fits name, fix it automatically.
                    try:
                        head = dest.open("rb").read(512)
                        if _looks_like_votable(head):
                            logger.warning(
                                "Existing raw file is a VOTable; overwriting with resolved FITS",
                                extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "path": str(dest)},
                            )
                            should_write = True
                    except Exception:
                        pass

                if not should_write:
                    logger.info(
                        "Raw FITS exists; skipping write",
                        extra={"source": self.name, "spectrum_id": spectrum.spectrum_id, "path": str(dest)},
                    )
                else:
                    logger.info(
                        "Writing raw FITS",
                        extra={
                            "source": self.name,
                            "spectrum_id": spectrum.spectrum_id,
                            "path": str(dest),
                            "bytes": len(fits_bytes),
                            "overwrite": overwrite,
                        },
                    )
                    dest.write_bytes(fits_bytes)
                new_metadata["raw_path"] = str(dest)
                if set_local_path:
                    new_metadata["local_path"] = str(dest)

            if zarr_roots:
                zarr_writes: List[Dict[str, str]] = []
                for store_path, root in zarr_roots:
                    logger.info(
                        "Writing to Zarr (progress)",
                        extra={
                            "source": self.name,
                            "index": index,
                            "total": total,
                            "spectrum_id": spectrum.spectrum_id,
                            "identifier": spectrum.metadata.get("identifier"),
                            "product": _product_tag(spectrum),
                            "zarr_store": store_path,
                            "zarr_group": f"spectra/{_safe_filename(zarr_base_name, default='spectrum')}",
                        },
                    )
                    logger.info(
                        "Writing spectrum to Zarr",
                        extra={
                            "source": self.name,
                            "spectrum_id": spectrum.spectrum_id,
                            "save_path": store_path,
                            "bytes": len(fits_bytes),
                            "overwrite": overwrite,
                        },
                    )
                    zarr_key = self.write_fits_to_zarr(
                        fits_bytes=fits_bytes,
                        spectrum=SpectrumRecord(
                            spectrum_id=spectrum.spectrum_id,
                            source=spectrum.source,
                            metadata=new_metadata
                        ),
                        root=root,
                        zarr_name=zarr_base_name,
                        process_ccf=process_ccf,
                        overwrite=overwrite,
                    )
                    logger.info(
                        "Zarr write complete",
                        extra={
                            "source": self.name,
                            "spectrum_id": spectrum.spectrum_id,
                            "save_path": store_path,
                            "zarr_key": zarr_key,
                        },
                    )
                    zarr_writes.append({"store": store_path, "key": zarr_key})

                # Backwards compatible single-store keys:
                if len(zarr_writes) == 1:
                    new_metadata["zarr_store"] = zarr_writes[0]["store"]
                    new_metadata["zarr_key"] = zarr_writes[0]["key"]
                else:
                    new_metadata["zarr_writes"] = zarr_writes
                    new_metadata["zarr_stores"] = [w["store"] for w in zarr_writes]
                    new_metadata["zarr_keys"] = [w["key"] for w in zarr_writes]

            saved.append(
                SpectrumRecord(
                    spectrum_id=spectrum.spectrum_id,
                    source=spectrum.source,
                    metadata=new_metadata
                )
            )

        return saved

    def _open_zarr_root(self, save_path: str | os.PathLike[str]) -> Any:
        """Open a Zarr group in append mode (create if needed)."""

        try:
            import zarr  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Zarr support requires the `zarr` package. Install it to use save_path."
            ) from exc

        save_path_str = str(save_path)
        logger.info("Opening Zarr store", extra={"source": self.name, "save_path": save_path_str})

        # Zarr v3 can throw `ContainsGroupError` if multiple threads race during first open.
        # Serialize opens per path to make concurrent bulk downloads reliable on HPC.
        with _zarr_open_lock(save_path):
            exists = Path(save_path_str).exists()
            mode = "r+" if exists else "a"
            root = zarr.open_group(save_path_str, mode=mode)

        logger.info("Zarr store ready", extra={"source": self.name, "save_path": save_path_str})
        return root

    def write_fits_to_zarr(
        self,
        *,
        fits_bytes: bytes,
        spectrum: SpectrumRecord,
        root: Any,
        zarr_name: str | None = None,
        process_ccf: bool = True,
        overwrite: bool = False,
    ) -> str:
        """Write a spectrum to Zarr using a consistent, overridable conversion hook.

        Sources should typically override `spectrum_to_zarr_components()` to store
        structured arrays (e.g., wavelength/flux) and attach Spectrum metadata as
        Zarr attributes on the per-spectrum group.

        Returns:
            The Zarr key (group path) written for this spectrum.
        """

        import numpy as np  # type: ignore

        if process_ccf and _product_tag(spectrum) == "ccf":
            arrays, attrs = self.ccf_to_zarr_components(fits_bytes=fits_bytes, spectrum=spectrum)
        else:
            arrays, attrs = self.spectrum_to_zarr_components(fits_bytes=fits_bytes, spectrum=spectrum)

        spectra_root = root.require_group("spectra")
        key = _safe_filename(zarr_name or spectrum.spectrum_id, default="spectrum")
        g = spectra_root.require_group(key)

        # Datasets we always store as 2D (n_records, n_points), even when n_records == 1.
        # Note: ELODIE uses `wavelengths` (plural) for its wavelength grid.
        always_2d = {"wavelength", "wavelengths", "intensity", "error", "ccf"}

        # Write arrays as datasets under the spectrum group.
        for name, array in arrays.items():
            if array is None:
                continue
            arr = np.asarray(array)

            def _stack_params() -> tuple[int | None, int | None]:
                # For identifier-based naming, we annotate indices during persistence.
                prod = _product_tag(spectrum)
                if prod == "ccf" and name == "ccf":
                    idx = spectrum.metadata.get("_ccf_index")
                    total = spectrum.metadata.get("_ccf_total")
                # Stack per-spectrum datasets across multiple spectra for a given identifier.
                # Support both singular/plural naming conventions used by different sources.
                elif name in {"intensity", "wavelength", "wavelengths", "error"}:
                    idx = spectrum.metadata.get("_spec_index")
                    total = spectrum.metadata.get("_spec_total")
                else:
                    return None, None

                try:
                    idx_i = int(idx) if idx is not None else None
                    total_i = int(total) if total is not None else None
                except Exception:
                    return None, None
                return idx_i, total_i

            idx_i, total_i = _stack_params()
            # Ensure at-least-2D layout for key datasets, even for single-record outputs.
            wants_2d = name in always_2d
            if wants_2d:
                if total_i is None:
                    total_i = 1
                if idx_i is None:
                    idx_i = 1

            # For stacking, normalize 2D (1, N) to 1D row vector.
            row_vec = None
            if wants_2d:
                if arr.ndim == 1:
                    row_vec = arr
                elif arr.ndim == 2 and arr.shape[0] == 1:
                    row_vec = arr[0]

            should_stack = (
                wants_2d
                and idx_i is not None
                and total_i is not None
                and row_vec is not None
            )

            if should_stack:
                # Store as (n, n_points) with rows indexed by idx_i (1-based).
                row = int(idx_i) - 1
                n_points = int(row_vec.shape[0])

                # Use float + NaN padding so variable-length rows can coexist.
                arr2 = np.asarray(row_vec, dtype=float)

                if name in g:
                    ds = g[name]
                    if getattr(ds, "ndim", 0) != 2:
                        if overwrite:
                            del g[name]
                            ds = None
                        else:
                            raise TypeError(f"Existing Zarr dataset {name} is not 2D; set overwrite=True to replace")
                    if ds is not None:
                        if ds.shape[0] != total_i:
                            if overwrite:
                                del g[name]
                                ds = None
                            else:
                                raise TypeError(f"Incompatible shape ({ds.shape} vs {(total_i, ds.shape[1])})")
                        if ds.shape[1] < n_points:
                            ds.resize((total_i, n_points))
                else:
                    ds = None

                if ds is None:
                    ds = g.require_array(
                        name,
                        shape=(total_i, n_points),
                        dtype="f8",
                        chunks=(1, n_points),
                        fill_value=np.nan,
                        overwrite=overwrite,
                    )

                # Write row and pad remainder (if any) with NaN.
                ds[row, :n_points] = arr2
                if ds.shape[1] > n_points:
                    ds[row, n_points:] = np.nan

                # Intentionally do not store per-row point-count datasets in Zarr.
            else:
                # If the caller provided a 1D vector for a dataset that should be 2D,
                # store it as (1, N) for consistency.
                if wants_2d and arr.ndim == 1:
                    arr = np.asarray(arr, dtype=float)[None, :]
                ds = g.require_array(
                    name,
                    shape=arr.shape,
                    dtype=arr.dtype,
                    overwrite=overwrite,
                )
                ds[...] = arr

        # Attach attributes.
        prod = _product_tag(spectrum)
        record_idx = spectrum.metadata.get("_ccf_index" if prod == "ccf" else "_spec_index")
        record_total = spectrum.metadata.get("_ccf_total" if prod == "ccf" else "_spec_total")
        if record_idx is not None:
            records = dict(getattr(g.attrs, "get", lambda k, d=None: d)("records", {}) or {})
            try:
                record_key = str(int(record_idx))
            except Exception:
                record_key = str(spectrum.spectrum_id)
            records[record_key] = _json_sanitize(
                {
                    "index": record_idx,
                    "total": record_total,
                    "spectrum_id": spectrum.spectrum_id,
                    "source": spectrum.source,
                    "product": prod,
                    "metadata": attrs.get("metadata"),
                    "data": attrs.get("data"),
                    "extraction": attrs.get("extraction"),
                }
            )
            g.attrs["records"] = records

        for k, v in attrs.items():
            try:
                g.attrs[k] = _json_sanitize(v)
            except Exception:
                # Zarr attrs require JSON-serializable values; fall back to strings.
                g.attrs[k] = str(v)

        return f"spectra/{key}"

    def extract_arrays_from_fits_payload(
        self,
        *,
        fits_bytes: bytes,
        spectrum: SpectrumRecord,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Backward-compatible dispatcher for FITS->arrays extraction.

        Prefer overriding:
        - `extract_spectrum_arrays_from_fits_payload(...)` for spectra
        - `extract_ccf_arrays_from_fits_payload(...)` for CCF
        """

        if _product_tag(spectrum) == "ccf":
            return self.extract_ccf_arrays_from_fits_payload(fits_bytes=fits_bytes, spectrum=spectrum)
        return self.extract_spectrum_arrays_from_fits_payload(fits_bytes=fits_bytes, spectrum=spectrum)

    def extract_spectrum_arrays_from_fits_payload(
        self,
        *,
        fits_bytes: bytes,
        spectrum: SpectrumRecord,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract Zarr arrays for a spectrum product from FITS bytes.

        This is the *main per-source hook* for spectra: override this in each source.
        Default behavior parses from FITS via `extract_1d_wavelength_intensity_from_fits_bytes`.
        """

        import numpy as np  # type: ignore

        arrays: Dict[str, Any] = {}
        info: Dict[str, Any] = {}
        wavelength = None
        intensity = None
        try:
            wl2, inten2, s_info = extract_1d_wavelength_intensity_from_fits_bytes(fits_bytes)
            info = s_info or {}
            if wl2 is not None:
                wavelength = np.asarray(wl2)
            if inten2 is not None:
                intensity = np.asarray(inten2)
        except Exception:
            info = {"extraction": "none"}

        if wavelength is not None:
            arrays["wavelength"] = wavelength
        if intensity is not None:
            arrays["intensity"] = intensity
        return arrays, info

    def extract_ccf_arrays_from_fits_payload(
        self,
        *,
        fits_bytes: bytes,
        spectrum: SpectrumRecord,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract Zarr arrays for a CCF product from FITS bytes.

        This is the *main per-source hook* for CCF: override this in each source.
        Default behavior parses `ccf` from FITS using `extract_ccf_from_fits_bytes`.
        """

        import numpy as np  # type: ignore

        arrays: Dict[str, Any] = {}
        info: Dict[str, Any] = {}

        try:
            _, ccf_vals, ccf_info = extract_ccf_from_fits_bytes(fits_bytes)
            info = ccf_info or {}
            if ccf_vals is not None:
                arrays["ccf"] = np.asarray(ccf_vals)
        except Exception:
            info = {"extraction": "none"}
        return arrays, info

    def ccf_to_zarr_components(
        self,
        *,
        fits_bytes: bytes,
        spectrum: SpectrumRecord,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a CCF product to Zarr components (arrays + attributes).

        Sources should override this for source-specific CCF layouts.
        The default tries to parse (velocity, ccf) from FITS bytes (requires astropy).
        """

        arrays, extraction_info = self.extract_ccf_arrays_from_fits_payload(fits_bytes=fits_bytes, spectrum=spectrum)
        data_keys = extract_data_keys_from_fits_bytes(fits_bytes)
        # Prefer per-source extraction info for known DataKeys.
        for key in [k.value for k in DataKeys]:
            if key in extraction_info and extraction_info[key] is not None:
                data_keys[key] = extraction_info[key]
        attrs: Dict[str, Any] = {
            "schema": "spectra_download.ccf.v1",
            "product": "ccf",
            "spectrum_id": spectrum.spectrum_id,
            "source": spectrum.source,
            "metadata": {k: str(v) for k, v in spectrum.metadata.items()},
        }
        if data_keys:
            attrs["data"] = {k: _json_attr_value(v) for k, v in data_keys.items()}
        if extraction_info:
            attrs["extraction"] = extraction_info

        # Construct the typed dataclass for callers that want it (primarily for debugging/extension).
        _ = CCFRecord(
            spectrum_id=str(spectrum.spectrum_id),
            source=str(spectrum.source),
            metadata=dict(spectrum.metadata),
        )

        return arrays, attrs

    def spectrum_to_zarr_components(
        self,
        *,
        fits_bytes: bytes,
        spectrum: SpectrumRecord,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a spectrum to Zarr components (arrays + attributes).

        This is the main hook for sources to override when they want a structured
        Zarr representation.

        Returns:
            arrays: Mapping of dataset name -> array-like (NumPy array, list, etc.)
            attrs: Mapping of attribute name -> JSON-serializable value (or best-effort).
        """

        arrays, extraction_info = self.extract_spectrum_arrays_from_fits_payload(fits_bytes=fits_bytes, spectrum=spectrum)
        data_keys = extract_data_keys_from_fits_bytes(fits_bytes)
        # Prefer per-source extraction info for known DataKeys.
        for key in [k.value for k in DataKeys]:
            if key in extraction_info and extraction_info[key] is not None:
                data_keys[key] = extraction_info[key]

        attrs: Dict[str, Any] = {
            "schema": "spectra_download.v1",
            "spectrum_id": spectrum.spectrum_id,
            "source": spectrum.source,
            "normalized": bool(getattr(spectrum, "normalized", False)),
            "metadata": {k: str(v) for k, v in spectrum.metadata.items()},
        }
        if data_keys:
            attrs["data"] = {k: _json_attr_value(v) for k, v in data_keys.items()}
        if extraction_info:
            attrs["extraction"] = extraction_info

        return arrays, attrs

    def _save_spectra_files(
        self,
        spectra: List[SpectrumRecord],
        *,
        save_dir: str | os.PathLike[str],
        overwrite: bool = False,
        filename_strategy: str = "spectrum_id",
    ) -> List[SpectrumRecord]:
        """Download each Spectrum's access_url to `save_dir` and annotate local_path."""

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: List[SpectrumRecord] = []
        for index, spectrum in enumerate(spectra, start=1):
            access_url = spectrum.metadata.get("access_url")
            if not access_url or not isinstance(access_url, str):
                saved.append(spectrum)
                continue

            if filename_strategy == "access_url":
                candidate = _filename_from_access_url(access_url) or spectrum.spectrum_id
            else:
                candidate = spectrum.spectrum_id or f"{self.name}_{index}"

            filename = _safe_filename(str(candidate), default=f"{self.name}_{index}")
            if not filename.lower().endswith(".fits"):
                filename = f"{filename}.fits"

            dest = out_dir / filename
            if dest.exists() and not overwrite:
                local_path = dest
            else:
                local_path = download_file(
                    access_url,
                    dest,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )

            new_metadata = dict(spectrum.metadata)
            new_metadata["local_path"] = str(local_path)
            saved.append(
                SpectrumRecord(
                    spectrum_id=spectrum.spectrum_id,
                    source=spectrum.source,
                    metadata=new_metadata
                )
            )

        return saved
