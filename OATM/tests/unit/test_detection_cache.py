"""Required Task 5 test: a cache hit must not re-run inference, and every key
field must match exactly for a hit to occur."""
from oatm.detection.cache import DetectionCache, cache_key, sha256_of_file


def test_sha256_of_file_is_stable_for_identical_content(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"same content")
    f2.write_bytes(b"same content")
    assert sha256_of_file(f1) == sha256_of_file(f2)


def test_sha256_of_file_differs_for_different_content(tmp_path):
    f1 = tmp_path / "a.bin"
    f2 = tmp_path / "b.bin"
    f1.write_bytes(b"content A")
    f2.write_bytes(b"content B")
    assert sha256_of_file(f1) != sha256_of_file(f2)


def test_cache_key_changes_if_any_single_field_changes():
    base = dict(image_hash="img1", model_name="m", weights_hash="w1", imgsz=640,
                confidence_floor=0.05, package_versions={"a": "1"})
    baseline = cache_key(**base)

    for field, new_value in [
        ("image_hash", "img2"), ("model_name", "m2"), ("weights_hash", "w2"),
        ("imgsz", 480), ("confidence_floor", 0.1), ("package_versions", {"a": "2"}),
    ]:
        variant = dict(base)
        variant[field] = new_value
        assert cache_key(**variant) != baseline, f"changing {field} should change the cache key"


def test_cache_round_trips_detections(tmp_path):
    cache = DetectionCache(tmp_path / "cache.json")
    key = cache_key("img", "model", "weights", 640, 0.05, {"pkg": "1"})
    detections = [{"class": "car", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]

    assert cache.get(key) is None
    cache.set(key, detections, inference_time_ms=42.0)
    cache.save()

    reloaded = DetectionCache(tmp_path / "cache.json")
    cached = reloaded.get(key)
    assert cached is not None
    assert cached["detections"] == detections
    assert cached["inference_time_ms"] == 42.0


def test_a_cache_hit_means_the_expensive_function_is_never_called(tmp_path):
    """Simulates the real script's hit/miss logic with a fake, call-counting
    'inference' function -- proves a hit skips it entirely."""
    cache = DetectionCache(tmp_path / "cache.json")
    key = cache_key("img", "model", "weights", 640, 0.05, {"pkg": "1"})

    call_count = {"n": 0}

    def fake_expensive_inference():
        call_count["n"] += 1
        return [{"class": "car", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10}], 100.0

    def get_or_run(cache, key):
        cached = cache.get(key)
        if cached is not None:
            return cached["detections"], cached["inference_time_ms"]
        dets, ms = fake_expensive_inference()
        cache.set(key, dets, ms)
        return dets, ms

    get_or_run(cache, key)
    assert call_count["n"] == 1
    get_or_run(cache, key)  # second call, same key -- must be a cache hit
    assert call_count["n"] == 1, "the second call with an identical key must not re-run inference"
