"""Versioned local cache for deterministic primitive-audit intermediates."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np

T = TypeVar("T")

# Bump this whenever the serialized intermediate semantics change. Cache files
# are derived local artifacts; they are never source data or annotation truth.
CACHE_SCHEMA = "primitive-audit-v1"


def stable_key(*parts: object) -> str:
    payload = json.dumps(
        (CACHE_SCHEMA, *parts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
    }


def array_identity(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


class PrimitiveAuditCache:
    """Read/write trusted, derived artifacts below one explicitly chosen root."""

    def __init__(self, root: Path):
        self.root = root
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}

    def _record(self, bucket: dict[str, int], namespace: str) -> None:
        bucket[namespace] = bucket.get(namespace, 0) + 1

    def _path(self, namespace: str, key: str, suffix: str) -> Path:
        return self.root / CACHE_SCHEMA / namespace / f"{key}{suffix}"

    def text(self, namespace: str, key: str, produce: Callable[[], str]) -> str:
        path = self._path(namespace, key, ".txt")
        if path.exists():
            self._record(self.hits, namespace)
            return path.read_text(encoding="utf-8")
        self._record(self.misses, namespace)
        value = produce()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return value

    def arrays(
        self,
        namespace: str,
        key: str,
        produce: Callable[[], tuple[np.ndarray, ...]],
    ) -> tuple[np.ndarray, ...]:
        path = self._path(namespace, key, ".npz")
        if path.exists():
            self._record(self.hits, namespace)
            with np.load(path, allow_pickle=False) as payload:
                return tuple(payload[name] for name in sorted(payload.files))
        self._record(self.misses, namespace)
        values = produce()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, **{f"a{index}": value for index, value in enumerate(values)}
        )
        return values

    def object(self, namespace: str, key: str, produce: Callable[[], T]) -> T:
        """Cache internal Python records; only this private cache is trusted."""
        path = self._path(namespace, key, ".pickle")
        if path.exists():
            self._record(self.hits, namespace)
            with path.open("rb") as stream:
                return pickle.load(stream)
        self._record(self.misses, namespace)
        value = produce()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return value

    def stats(self) -> dict[str, dict[str, int]]:
        return {"hits": dict(self.hits), "misses": dict(self.misses)}
