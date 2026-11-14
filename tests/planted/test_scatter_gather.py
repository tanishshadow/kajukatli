from aml.detect.scatter_gather import detect_scatter_gather, ScatterConfig
from aml.graph.build import build_graph
from tests_helpers import frame


def config() -> ScatterConfig:
    return ScatterConfig(window_seconds=10, m=2, m_in=2, m_out=2, rho_lo=.8, eps=.1)


def test_direct_scatter_fuzzy_stages_and_amount_negative() -> None:
    data = frame([(0, 1, 0, 10, 1), (1, 3, 1, 10, 1), (0, 2, 2, 10, 1), (2, 3, 3, 10, 1)])
    hits = detect_scatter_gather(build_graph(data), config())
    hit = next(h for h in hits if h.typology == "scatter_gather")
    assert hit.accounts == [0, 1, 2, 3] and hit.edge_ids == [0, 1, 2, 3]
    data.loc[3, "amt_out"] = 1
    assert not any(h.typology == "scatter_gather" for h in detect_scatter_gather(build_graph(data), config()))


def test_gather_scatter_aggregate_amount_and_time() -> None:
    data = frame([(0, 2, 0, 10, 1), (1, 2, 1, 10, 1), (2, 3, 2, 12, 1), (2, 4, 3, 8, 1)])
    hits = detect_scatter_gather(build_graph(data), config())
    hit = next(h for h in hits if h.typology == "gather_scatter")
    assert hit.edge_ids == [0, 1, 2, 3]
    data.ts = [3, 4, 0, 1]
    assert detect_scatter_gather(build_graph(data), config()) == []


def test_ties_parallel_edges_and_self_loops_do_not_inflate_middle_count() -> None:
    data = frame([(0, 1, 0, 10, 1), (0, 1, 0, 10, 1), (1, 3, 0, 10, 1), (0, 0, 0, 10, 1)])
    assert detect_scatter_gather(build_graph(data), config()) == []
