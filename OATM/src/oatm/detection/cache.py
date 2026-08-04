"""Phase 4: content-hash detector cache. A cache hit requires every one of
image hash, model name, weights hash, image size, confidence floor, and
package versions to match exactly -- reused directly from the pattern
validated in Assignments 3-4 (see reuse_audit.md precedent)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(image_hash: str, model_name: str, weights_hash: str, imgsz: int,
              confidence_floor: float, package_versions: dict) -> str:
    payload = json.dumps({
        "image_hash": image_hash, "model_name": model_name, "weights_hash": weights_hash,
        "imgsz": imgsz, "confidence_floor": confidence_floor, "package_versions": package_versions,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DetectionCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        if cache_path.is_file():
            with open(cache_path, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, detections: list[dict], inference_time_ms: float) -> None:
        self._data[key] = {"detections": detections, "inference_time_ms": inference_time_ms}

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def __len__(self) -> int:
        return len(self._data)
