"""Overpass API HTTP client."""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://overpass-api.de/api/interpreter"


def fetch_overpass(query: str, url: str = _DEFAULT_URL, http_timeout: int = 180) -> dict:
    logger.debug("Overpass query:\n%s", query)
    resp = requests.post(url, data={"data": query}, timeout=http_timeout)
    resp.raise_for_status()
    data = resp.json()
    n_elements = len(data.get("elements", []))
    logger.info("Overpass returned %d elements", n_elements)
    return data
