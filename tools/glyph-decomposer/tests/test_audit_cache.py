import numpy as np

from glyph_decomposer.audit_cache import PrimitiveAuditCache, stable_key


def test_cache_reuses_text_arrays_and_objects(tmp_path):
    cache = PrimitiveAuditCache(tmp_path)
    calls = {"text": 0, "arrays": 0, "object": 0}

    def text_value():
        calls["text"] += 1
        return "page"

    def array_value():
        calls["arrays"] += 1
        return (np.asarray([[True, False]]), np.asarray([[0.0, 1.0]]))

    def object_value():
        calls["object"] += 1
        return ((1, 2), {"score": 3.0})

    assert cache.text("page-svg", "one", text_value) == "page"
    assert cache.text("page-svg", "one", text_value) == "page"
    first_arrays = cache.arrays("skeleton", "two", array_value)
    second_arrays = cache.arrays("skeleton", "two", array_value)
    assert all(
        np.array_equal(first, second)
        for first, second in zip(first_arrays, second_arrays)
    )
    assert cache.object("leaf-pool", "three", object_value) == (
        (1, 2),
        {"score": 3.0},
    )
    assert cache.object("leaf-pool", "three", object_value) == (
        (1, 2),
        {"score": 3.0},
    )
    assert calls == {"text": 1, "arrays": 1, "object": 1}
    assert cache.stats() == {
        "hits": {"page-svg": 1, "skeleton": 1, "leaf-pool": 1},
        "misses": {"page-svg": 1, "skeleton": 1, "leaf-pool": 1},
    }


def test_stable_key_is_order_independent_for_mappings():
    assert stable_key({"page": 2, "size": 256}) == stable_key({"size": 256, "page": 2})
