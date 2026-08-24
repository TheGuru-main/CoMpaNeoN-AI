"""
Content-based cache for crawled pages.
Uses SHA‑256 signature to detect unchanged content.
"""

import hashlib
import json
from datetime import datetime, timedelta

class PageCache:
    def __init__(self, default_ttl_minutes=120):
        self.cache = {}
        self.default_ttl = timedelta(minutes=default_ttl_minutes)

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, url: str):
        entry = self.cache.get(url)
        if entry and datetime.utcnow() < entry["expires"]:
            return entry["data"], entry["hash"]
        return None, None

    def set(self, url: str, content_hash: str, data, ttl_minutes=None):
        ttl = timedelta(minutes=ttl_minutes) if ttl_minutes else self.default_ttl
        self.cache[url] = {
            "hash": content_hash,
            "data": data,
            "expires": datetime.utcnow() + ttl,
        }

    def has_changed(self, url: str, new_content: str) -> bool:
        _, old_hash = self.get(url)
        new_hash = self._hash(new_content)
        return old_hash is None or old_hash != new_hash

    def update_content_hash(self, url: str, new_content: str):
        new_hash = self._hash(new_content)
        entry = self.cache.get(url)
        if entry:
            entry["hash"] = new_hash
        else:
            self.cache[url] = {"hash": new_hash, "data": None, "expires": datetime.utcnow()}