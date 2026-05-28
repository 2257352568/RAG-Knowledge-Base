import hashlib
import json
import time
from collections import OrderedDict


class QueryCache:
    """TTL-based LRU cache for RAG query results."""

    def __init__(self, max_size: int = 256, ttl: int = 3600):
        self._max_size = max_size
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def _hash(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def get(self, query: str) -> dict | None:
        key = self._hash(query)
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return json.loads(value)

    def set(self, query: str, result: dict):
        key = self._hash(query)
        self._store[key] = (time.time(), json.dumps(result, ensure_ascii=False, default=str))
        self._store.move_to_end(key)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def invalidate_by_source(self, source_path: str):
        to_remove = []
        for key, (_, value) in self._store.items():
            try:
                data = json.loads(value)
                for src in data.get("sources", []):
                    if src.get("metadata", {}).get("source_path") == source_path:
                        to_remove.append(key)
                        break
            except (json.JSONDecodeError, TypeError):
                pass
        for key in to_remove:
            del self._store[key]

    def clear(self):
        self._store.clear()
