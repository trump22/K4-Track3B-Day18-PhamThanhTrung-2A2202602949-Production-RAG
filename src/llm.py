"""Gemini client shared by generation and enrichment."""
import asyncio
import json
import threading
import time

from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_REQUEST_INTERVAL

_request_lock = threading.Lock()
_next_request = 0.0
_request_interval = GEMINI_REQUEST_INTERVAL


def _reserve_request_slot():
    global _next_request
    with _request_lock:
        now = time.monotonic()
        delay = max(0.0, _next_request - now)
        if delay == 0:
            _next_request = now + _request_interval
    return delay


def _pace_request(request):
    while True:
        delay = _reserve_request_slot()
        if delay <= 0:
            return
        time.sleep(delay)


async def _pace_async_request(request):
    while True:
        delay = _reserve_request_slot()
        if delay <= 0:
            return
        await asyncio.sleep(delay)


def _update_quota(response):
    global _next_request, _request_interval
    if response.status_code != 429:
        return
    try:
        payload = json.loads(response.content)
        if isinstance(payload, list):
            payload = payload[0]
        details = payload.get("error", {}).get("details", [])
        if any("PerDay" in violation.get("quotaId", "")
               for detail in details for violation in detail.get("violations", [])):
            # A daily quota cannot be recovered by spacing requests; do not
            # turn Google's many-hour retryDelay into a blocked HTTP hook.
            return
        retry_delay = 2.0
        with _request_lock:
            for detail in details:
                if "retryDelay" in detail:
                    retry_delay = max(retry_delay, float(detail["retryDelay"].rstrip("s")))
                for violation in detail.get("violations", []):
                    quota_id = violation.get("quotaId", "")
                    if "PerMinute" in quota_id and "Requests" in quota_id:
                        limit = int(violation.get("quotaValue", 0))
                        if limit > 0:
                            interval = 60 / limit + 0.2
                            if interval > _request_interval:
                                _request_interval = interval
                                print(f"  Gemini quota: {limit} requests/min; pacing {_request_interval:.1f}s.", flush=True)
            _next_request = max(_next_request, time.monotonic() + retry_delay)
    except (ValueError, TypeError, IndexError):
        pass


def _quota_response(response):
    if response.status_code == 429:
        response.read()
        _update_quota(response)


async def _quota_async_response(response):
    if response.status_code == 429:
        await response.aread()
        _update_quota(response)


def gemini_http_client():
    import httpx
    return httpx.Client(event_hooks={"request": [_pace_request], "response": [_quota_response]})


def gemini_async_http_client():
    import httpx
    return httpx.AsyncClient(event_hooks={"request": [_pace_async_request], "response": [_quota_async_response]})


def has_gemini_key() -> bool:
    return bool(GEMINI_API_KEY and GEMINI_API_KEY.lower() not in {
        "your_gemini_api_key", "your_gemini_api_key_here", "sk-...", "...",
    })


def create_gemini_client():
    if not has_gemini_key():
        raise ValueError("Set GEMINI_API_KEY in the project .env file")
    from openai import OpenAI

    return OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, timeout=30, max_retries=3,
                  http_client=gemini_http_client())
