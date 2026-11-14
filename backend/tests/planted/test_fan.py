import numpy as np
from aml.detect.fan import detect_fans, FanConfig
from aml.graph.build import build_graph
from tests_helpers import frame


def test_fan_exact_edges_parallel_ties_self_loop_and_merge() -> None:
    g = build_graph(frame([(0, 1, 0, 10, 1), (0, 2, 1, 10, 1), (0, 1, 1, 10, 1),
                           (0, 3, 2, 10, 1), (0, 0, 2, 10, 1), (0, 4, 4, 10, 1)]))
    hits = detect_fans(g, FanConfig(window_seconds=3, k=3))
    out = [h for h in hits if h.typology == "fan_out"]
    assert len(out) == 1
    assert out[0].accounts == [0, 1, 2, 3, 4]
    assert out[0].edge_ids == [0, 1, 2, 3, 5]
    assert out[0].meta["hub"] == 0
    assert detect_fans(g, FanConfig(window_seconds=1, k=4)) == []
    assert out[0].model_dump() == [h for h in detect_fans(g, FanConfig(window_seconds=3, k=3)) if h.typology == "fan_out"][0].model_dump()


def test_fan_in_and_k_minus_one_negative() -> None:
    g = build_graph(frame([(1, 0, 0, 10, 1), (2, 0, 1, 10, 1), (3, 0, 2, 10, 1)]))
    hits = detect_fans(g, FanConfig(window_seconds=2, k=3))
    assert len(hits) == 1 and hits[0].typology == "fan_in"
    assert hits[0].edge_ids == [0, 1, 2]
    assert detect_fans(g, FanConfig(window_seconds=2, k=4)) == []
