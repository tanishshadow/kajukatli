from aml.detect.passthrough import PassConfig, detect_passthrough
from aml.graph.build import build_graph
from tests_helpers import frame


def config(**changes: object) -> PassConfig:
    return PassConfig(**({"dwell_seconds": 5, "rho_lo": .8, "eps": .1, "top_m": 2,
                         "min_links": 2, "max_paths_per_component": 100} | changes))


def test_chain_maximal_only_received_amount_and_mules() -> None:
    data = frame([(0, 1, 0, 20, 1), (1, 2, 1, 10, 1), (2, 3, 2, 10, 1), (3, 4, 3, 10, 1)])
    data.loc[0, "amt_in"] = 10
    hits = detect_passthrough(build_graph(data), config())
    assert len(hits) == 1 and hits[0].edge_ids == [0, 1, 2, 3]
    assert hits[0].meta["likely_mules"] == [1, 2, 3]


def test_dwell_amount_negative_parallel_ties_and_self_loop() -> None:
    data = frame([(0, 1, 0, 10, 1), (1, 2, 0, 10, 1), (2, 3, 0, 10, 1), (2, 2, 0, 10, 1)])
    assert detect_passthrough(build_graph(data), config())[0].edge_ids == [0, 1, 2]
    data.loc[1, "amt_out"] = 1
    assert detect_passthrough(build_graph(data), config()) == []
    data.loc[1, "amt_out"] = 10
    data.loc[2, "ts"] = 20
    assert detect_passthrough(build_graph(data), config()) == []


def test_branch_paths_top_m_and_component_cap_log(caplog) -> None:
    data = frame([(0, 1, 0, 10, 1), (1, 2, 1, 10, 1), (1, 2, 1, 10, 1),
                  (2, 3, 2, 10, 1), (2, 4, 2, 10, 1)])
    hits = detect_passthrough(build_graph(data), config(max_paths_per_component=1))
    assert len(hits) == 1 and "paths_per_component" in caplog.text
    caplog.clear()
    detect_passthrough(build_graph(data), config(top_m=1))
    assert "top_m" in caplog.text
