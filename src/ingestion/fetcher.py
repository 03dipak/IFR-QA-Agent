"""Polite HTTP fetcher with retry/backoff and FetchAttempt records.

Implements lld_ingestion_pdf.txt sections 3, 6, 5.6: 1 req/s etiquette (D3),
resume-able, retryable vs permanent error classification, idempotent
acquisition, FetchAttempt records separate from documents.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

import requests

from .models import FetchAttempt

# Permanent HTTP status codes (not retryable)
PERMANENT_STATUSES = {400, 401, 403, 404, 410, 451}
# Retryable HTTP status codes
RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB hard cap
MIN_PDF_BYTES = 1  # zero-byte = corrupt


class FetchError(Exception):
    """Raised when a fetch cannot succeed (permanent or exhausted retries)."""
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class Fetcher:
    """Polite fetcher. Etiquette: min 1 second between requests to the
    same host (D3)."""

    def __init__(self, min_interval_seconds: float = 1.0,
                 max_retries: int = 5, backoff_base: float = 2.0,
                 timeout: float = 60.0, user_agent: str = "IFR-QA-Agent/0.1"):
        self.min_interval = min_interval_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self._last_request_by_host: dict[str, float] = {}

    def _throttle(self, host: str) -> None:
        now = time.monotonic()
        last = self._last_request_by_host.get(host, 0.0)
        wait = self.min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        self._last_request_by_host[host] = time.monotonic()

    def fetch(self, url: str, candidate_id: str) -> tuple[bytes, FetchAttempt]:
        """Fetch a URL, returning (bytes, FetchAttempt). Raises FetchError
        on permanent failure or exhausted retries."""
        host = urlparse(url).netloc
        attempt_no = 0
        last_attempt: FetchAttempt | None = None
        while attempt_no < self.max_retries:
            attempt_no += 1
            self._throttle(host)
            attempt_id = str(uuid.uuid4())
            started = datetime.now(UTC)
            try:
                resp = requests.get(url, headers=self.headers,
                                    timeout=self.timeout, stream=True)
                downloaded = b""
                for chunk in resp.iter_content(chunk_size=65536):
                    downloaded += chunk
                    if len(downloaded) > MAX_PDF_BYTES:
                        raise FetchError("file_too_large",
                                         f"exceeds {MAX_PDF_BYTES} bytes",
                                         retryable=False)
                completed = datetime.now(UTC)
                attempt = FetchAttempt(
                    fetch_attempt_id=attempt_id,
                    candidate_id=candidate_id,
                    source_url=url,
                    started_at=started,
                    completed_at=completed,
                    http_status=resp.status_code,
                    content_length=int(resp.headers.get("Content-Length", 0)) or None,
                    downloaded_bytes=len(downloaded),
                    attempt_number=attempt_no,
                )
                last_attempt = attempt

                # Content-Length mismatch -> broken partial
                cl = attempt.content_length
                if cl is not None and cl > 0 and cl != len(downloaded):
                    attempt.error_code = "content_length_mismatch"
                    attempt.retryable = True
                    if attempt_no >= self.max_retries:
                        raise FetchError("content_length_mismatch",
                                         "partial download", retryable=True)
                    time.sleep(self.backoff_base ** attempt_no)
                    continue

                if resp.status_code in PERMANENT_STATUSES:
                    attempt.error_code = f"http_{resp.status_code}"
                    attempt.retryable = False
                    raise FetchError(f"http_{resp.status_code}",
                                     f"HTTP {resp.status_code}", retryable=False)

                if resp.status_code in RETRYABLE_STATUSES:
                    attempt.error_code = f"http_{resp.status_code}"
                    attempt.retryable = True
                    if attempt_no >= self.max_retries:
                        raise FetchError(f"http_{resp.status_code}",
                                         f"HTTP {resp.status_code} exhausted",
                                         retryable=True)
                    time.sleep(self.backoff_base ** attempt_no)
                    continue

                if resp.status_code != 200:
                    attempt.error_code = f"http_{resp.status_code}"
                    attempt.retryable = False
                    raise FetchError(f"http_{resp.status_code}",
                                     f"HTTP {resp.status_code}", retryable=False)

                # Validate PDF header (first bytes)
                if len(downloaded) < MIN_PDF_BYTES or not downloaded.startswith(b"%PDF"):
                    attempt.error_code = "invalid_pdf_header"
                    attempt.retryable = False
                    raise FetchError("invalid_pdf_header",
                                     "not a PDF (HTML-as-.pdf?)",
                                     retryable=False)

                # Zero-byte / truncated handled by size checks above
                return downloaded, attempt

            except requests.RequestException as exc:
                completed = datetime.now(UTC)
                attempt = FetchAttempt(
                    fetch_attempt_id=attempt_id,
                    candidate_id=candidate_id,
                    source_url=url,
                    started_at=started,
                    completed_at=completed,
                    http_status=None,
                    downloaded_bytes=0,
                    attempt_number=attempt_no,
                    error_code="network_error",
                    retryable=True,
                )
                last_attempt = attempt
                if attempt_no >= self.max_retries:
                    raise FetchError("network_error", str(exc), retryable=True)
                time.sleep(self.backoff_base ** attempt_no)
                continue

        # Should not reach here
        raise FetchError("exhausted", "retries exhausted",
                         retryable=getattr(last_attempt, "retryable", False))
