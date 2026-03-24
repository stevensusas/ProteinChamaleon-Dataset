"""Base HTTP client with rate limiting, retries, and structured error handling."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Raised when an API call fails after all retries."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class BaseClient:
    """
    Thread-safe, rate-limited HTTP client.

    Implements:
    - Per-client configurable rate limiting via a monotonic clock (lock-protected)
    - Optional concurrency semaphore to cap parallel requests per API
    - Automatic retry with exponential backoff for 5xx and network errors
    - Respect for Retry-After headers on 429 responses
    """

    def __init__(
        self,
        base_url: str,
        rate_limit_delay: float = 0.5,
        timeout: float = 30.0,
        max_retries: int = 3,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._last_request_time: float = 0.0
        self._throttle_lock = threading.Lock()
        self._semaphore: Optional[threading.Semaphore] = (
            threading.Semaphore(max_concurrency) if max_concurrency else None
        )
        self.session = httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": (
                    "ProteinChameleonDataset/1.0 "
                    "(research; github.com/ProteinChameleon)"
                ),
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_time
            wait = self.rate_limit_delay - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request_time = time.monotonic()

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self._semaphore is not None:
            self._semaphore.acquire()
        try:
            return self._request_inner(method, url, **kwargs)
        finally:
            if self._semaphore is not None:
                self._semaphore.release()

    def _request_inner(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception = APIError("No attempts made")
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp: httpx.Response = getattr(self.session, method)(url, **kwargs)

                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1) * 5))
                    logger.warning("Rate limited by %s. Waiting %.1fs (attempt %d)", url, wait, attempt + 1)
                    time.sleep(wait)
                    with self._throttle_lock:
                        self._last_request_time = 0.0
                    continue

                resp.raise_for_status()
                return resp

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (500, 502, 503, 504):
                    wait = 2.0 ** attempt
                    logger.warning(
                        "Server error %d from %s. Retrying in %.1fs",
                        exc.response.status_code, url, wait,
                    )
                    time.sleep(wait)
                    last_error = APIError(str(exc), exc.response.status_code)
                else:
                    raise APIError(
                        f"HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                        exc.response.status_code,
                    ) from exc

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                wait = 2.0 ** attempt
                logger.warning("Network error on %s. Retrying in %.1fs: %s", url, wait, exc)
                time.sleep(wait)
                last_error = APIError(str(exc))

        raise last_error

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._request("get", url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._request("post", url, **kwargs)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "BaseClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
