"""Shared HTTP helpers for spectra downloads."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional
from urllib import error, request

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """Raised when a download request fails after retries."""


def download_json(url: str, *, timeout: int = 30, max_retries: int = 3) -> Dict[str, Any]:
    """Download JSON payloads with retries.

    Args:
        url: Fully qualified URL to fetch.
        timeout: Timeout in seconds for the HTTP request.
        max_retries: Number of retry attempts before failing.

    Returns:
        Parsed JSON payload.
    """

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        logger.debug("Fetching JSON", extra={"url": url, "attempt": attempt})
        try:
            with request.urlopen(url, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload)
        except (error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            logger.warning(
                "Download attempt failed",
                extra={"url": url, "attempt": attempt, "error": last_error},
            )
            time.sleep(0.5 * attempt)

    raise DownloadError(f"Failed to download {url}: {last_error}")
