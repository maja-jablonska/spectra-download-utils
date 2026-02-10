"""Shared HTTP helpers for spectra downloads."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a download request fails after retries."""

    def __init__(self, message: str, *, url: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code


def download_json(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    headers: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Download JSON payloads with retries.

    Args:
        url: Fully qualified URL to fetch.
        timeout: Timeout in seconds for the HTTP request.
        max_retries: Number of retry attempts before failing.

    Returns:
        Parsed JSON payload.
    """

    last_error: Optional[str] = None
    last_status: int | None = None
    for attempt in range(1, max_retries + 1):
        logger.debug("Fetching JSON", extra={"url": url, "attempt": attempt})
        try:
            req = request.Request(url, headers=headers or {})
            with request.urlopen(req, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
                logger.debug(
                    "Download successful",
                    extra={"url": url, "attempt": attempt, "bytes": len(payload)},
                )
                return json.loads(payload)
        except error.HTTPError as exc:
            last_error = str(exc)
            last_status = int(getattr(exc, "code", None) or 0) or None
            logger.warning(
                "Download attempt failed",
                extra={"url": url, "attempt": attempt, "error": last_error},
            )
            time.sleep(0.5 * attempt)
        except (error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            logger.warning(
                "Download attempt failed",
                extra={"url": url, "attempt": attempt, "error": last_error},
            )
            time.sleep(0.5 * attempt)

    raise DownloadError(f"Failed to download {url}: {last_error}", url=url, status_code=last_status)


def download_bytes(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    headers: Dict[str, str] | None = None,
) -> bytes:
    """Download a binary payload with retries."""

    last_error: Optional[str] = None
    last_status: int | None = None
    for attempt in range(1, max_retries + 1):
        logger.debug("Fetching bytes", extra={"url": url, "attempt": attempt})
        try:
            req = request.Request(url, headers=headers or {})
            with request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
                logger.debug(
                    "Download successful",
                    extra={"url": url, "attempt": attempt, "bytes": len(payload)},
                )
                return payload
        except error.HTTPError as exc:
            last_error = str(exc)
            last_status = int(getattr(exc, "code", None) or 0) or None
            logger.warning(
                "Download attempt failed",
                extra={"url": url, "attempt": attempt, "error": last_error},
            )
            time.sleep(0.5 * attempt)
        except error.URLError as exc:
            last_error = str(exc)
            logger.warning(
                "Download attempt failed",
                extra={"url": url, "attempt": attempt, "error": last_error},
            )
            time.sleep(0.5 * attempt)

    raise DownloadError(f"Failed to download {url}: {last_error}", url=url, status_code=last_status)


def download_file(
    url: str,
    destination: str | os.PathLike[str],
    *,
    timeout: int = 30,
    max_retries: int = 3,
    headers: Dict[str, str] | None = None,
) -> Path:
    """Download a URL and write it to `destination` (binary)."""

    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading file", extra={"url": url, "destination": str(dest)})
    payload = download_bytes(url, timeout=timeout, max_retries=max_retries, headers=headers)
    dest.write_bytes(payload)
    logger.info("File saved", extra={"destination": str(dest), "bytes": len(payload)})
    return dest
