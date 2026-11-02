"""Polite HTTP fetcher with retry/backoff and FetchAttempt records.

Implements lld_ingestion_pdf.txt sections 3, 6, 5.6: 1 req/s etiquette (D3),
resume-able, retryable vs permanent error classification, idempotent
acquisition, FetchAttempt records separate from documents.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests

from .models import FetchAttempt

# Permanent HTTP status codes (not retryable)
PERMANENT_STATUSES = {400, 401, 403, 404, 410, 451}
# Retryable HTTP status codes
RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}

MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB hard cap
MIN_PDF_BYTES = 1  # zero-byte = corrupt
MAX_BACKOFF_SECONDS = 60.0  # documented cap for exponential backoff


class FetchError(Exception):
    """Raised when a fetch cannot succeed (permanent or exhausted retries)."""
    def __init__(self, code: str, message: str, retryable: bool = False,
                 attempt: FetchAttempt | None = None,
                 attempts: list[FetchAttempt] | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.attempt = attempt
        self.attempts = list(attempts) if attempts is not None else (
            [attempt] if attempt is not None else []
        )


def _content_length(resp: requests.Response) -> int | None:
    raw = resp.headers.get("Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _looks_like_html(body: bytes, content_type: str | None) -> bool:
    if content_type:
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type in {"text/html", "application/xhtml+xml"}:
            return True
    head = body.lstrip()[:512].lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or b"<html" in head[:128]
    )


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Return Retry-After as non-negative seconds, or None when unusable.

    Accepts delta-seconds and HTTP-date forms (RFC 9110). A date in the past
    means "retry now" (0), not "no header".
    """
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - now).total_seconds())


class Fetcher:
    """Polite fetcher. Etiquette: min 1 second between requests to the
    same host (D3). Sleep/clock/now are injectable so retry behaviour is
    testable without real waiting or a real wall clock."""

    def __init__(self, min_interval_seconds: float = 1.0,
                 max_retries: int = 5, backoff_base: float = 2.0,
                 backoff_cap_seconds: float = MAX_BACKOFF_SECONDS,
                 timeout: float = 60.0, user_agent: str = "IFR-QA-Agent/0.1",
                 sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None,
                 now: Callable[[], datetime] | None = None):
        self.min_interval = min_interval_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap_seconds = backoff_cap_seconds
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._now = now or (lambda: datetime.now(UTC))
        self._last_request_by_host: dict[str, float] = {}

    def _throttle(self, host: str) -> None:
        now = self._clock()
        last = self._last_request_by_host.get(host, 0.0)
        wait = self.min_interval - (now - last)
        if wait > 0:
            self._sleep(wait)
        self._last_request_by_host[host] = self._clock()

    def _backoff_delay(self, attempt_no: int) -> float:
        return min(self.backoff_base ** attempt_no, self.backoff_cap_seconds)

    def _retry_delay(self, attempt_no: int, resp: requests.Response) -> float:
        retry_after = parse_retry_after(resp.headers.get("Retry-After"), self._now())
        if retry_after is not None:
            return retry_after
        return self._backoff_delay(attempt_no)

    def fetch(self, url: str, candidate_id: str) -> tuple[bytes, FetchAttempt]:
        """Fetch a URL, returning (bytes, FetchAttempt). Raises FetchError
        on permanent failure or exhausted retries."""
        host = urlparse(url).netloc
        attempt_no = 0
        last_attempt: FetchAttempt | None = None
        attempts: list[FetchAttempt] = []
        while attempt_no < self.max_retries:
            attempt_no += 1
            self._throttle(host)
            attempt_id = str(uuid.uuid4())
            started = self._now()
            try:
                resp = requests.get(url, headers=self.headers,
                                    timeout=self.timeout, stream=True)
            except requests.RequestException as exc:
                completed = self._now()
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
                attempts.append(attempt)
                last_attempt = attempt
                if attempt_no >= self.max_retries:
                    raise FetchError("network_error", str(exc), retryable=True,
                                     attempt=attempt, attempts=attempts) from exc
                self._sleep(self._backoff_delay(attempt_no))
                continue

            try:
                downloaded = b""
                too_large = False
                for chunk in resp.iter_content(chunk_size=65536):
                    downloaded += chunk
                    if len(downloaded) > MAX_PDF_BYTES:
                        too_large = True
                        break
                completed = self._now()
                attempt = FetchAttempt(
                    fetch_attempt_id=attempt_id,
                    candidate_id=candidate_id,
                    source_url=url,
                    started_at=started,
                    completed_at=completed,
                    http_status=resp.status_code,
                    content_length=_content_length(resp),
                    downloaded_bytes=len(downloaded),
                    attempt_number=attempt_no,
                )
                attempts.append(attempt)
                last_attempt = attempt

                if resp.status_code in PERMANENT_STATUSES:
                    attempt.error_code = f"http_{resp.status_code}"
                    attempt.retryable = False
                    raise FetchError(f"http_{resp.status_code}",
                                     f"HTTP {resp.status_code}", retryable=False,
                                     attempt=attempt, attempts=attempts)

                if resp.status_code in RETRYABLE_STATUSES:
                    attempt.error_code = f"http_{resp.status_code}"
                    attempt.retryable = True
                    if attempt_no >= self.max_retries:
                        raise FetchError(f"http_{resp.status_code}",
                                         f"HTTP {resp.status_code} exhausted",
                                         retryable=True, attempt=attempt, attempts=attempts)
                    self._sleep(self._retry_delay(attempt_no, resp))
                    continue

                if resp.status_code != 200:
                    attempt.error_code = f"http_{resp.status_code}"
                    attempt.retryable = False
                    raise FetchError(f"http_{resp.status_code}",
                                     f"HTTP {resp.status_code}", retryable=False,
                                     attempt=attempt, attempts=attempts)

                if too_large:
                    attempt.error_code = "file_too_large"
                    attempt.retryable = False
                    raise FetchError("file_too_large",
                                     f"exceeds {MAX_PDF_BYTES} bytes",
                                     retryable=False, attempt=attempt, attempts=attempts)

                # An HTML/plain error page served at a .pdf URL is a document
                # exclusion (html_only), not a generic network failure.
                if _looks_like_html(downloaded, resp.headers.get("Content-Type")):
                    attempt.error_code = "html_only"
                    attempt.retryable = False
                    raise FetchError("html_only", "HTML response at a PDF URL",
                                     retryable=False, attempt=attempt, attempts=attempts)

                # Short Content-Length / zero bytes = broken partial. Discard
                # every byte and retry; never append or store it.
                cl = attempt.content_length
                short = cl is not None and cl > len(downloaded)
                if short or len(downloaded) == 0:
                    attempt.error_code = "truncated"
                    attempt.retryable = True
                    if attempt_no >= self.max_retries:
                        raise FetchError("corrupt_pdf",
                                         "truncated or zero-byte download",
                                         retryable=True, attempt=attempt, attempts=attempts)
                    self._sleep(self._backoff_delay(attempt_no))
                    continue

                # Validate PDF header (first bytes) after excluding the two
                # distinct HTML and truncation causes above.
                if not downloaded.startswith(b"%PDF"):
                    attempt.error_code = "invalid_pdf_header"
                    attempt.retryable = False
                    raise FetchError("invalid_pdf_header",
                                     "not a PDF",
                                     retryable=False, attempt=attempt, attempts=attempts)

                return downloaded, attempt

            except requests.RequestException as exc:
                completed = self._now()
                expected = _content_length(resp)
                partial = len(downloaded) > 0 or (
                    expected is not None and len(downloaded) < expected
                )
                attempt = FetchAttempt(
                    fetch_attempt_id=attempt_id,
                    candidate_id=candidate_id,
                    source_url=url,
                    started_at=started,
                    completed_at=completed,
                    http_status=resp.status_code,
                    content_length=expected,
                    downloaded_bytes=len(downloaded),
                    attempt_number=attempt_no,
                    error_code="truncated" if partial else "network_error",
                    retryable=True,
                )
                attempts.append(attempt)
                last_attempt = attempt
                if attempt_no >= self.max_retries:
                    code = "corrupt_pdf" if partial else "network_error"
                    raise FetchError(code, str(exc), retryable=True,
                                     attempt=attempt, attempts=attempts) from exc
                self._sleep(self._backoff_delay(attempt_no))
                continue
            finally:
                resp.close()

        # Should not reach here
        raise FetchError("exhausted", "retries exhausted",
                         retryable=getattr(last_attempt, "retryable", False),
                         attempt=last_attempt, attempts=attempts)
