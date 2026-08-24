from datetime import datetime
import hashlib

class MemoryCache:
    def __init__(self):
        self.cache = {}

    def _key(self, query: str) -> str:
        q = query.strip().lower()
        return hashlib.sha256(q.encode()).hexdigest()

    def get(self, query: str):
        key = self._key(query)
        return self.cache.get(key)

    def set(self, query: str, data):
        key = self._key(query)
        self.cache[key] = data

    def clear(self):
        self.cache.clear()