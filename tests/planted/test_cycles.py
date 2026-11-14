from aml.detect.cycles import CycleConfig, detect_cycles
from aml.graph.build import build_graph
from tests_helpers import frame


def config(**changes: object) -> CycleConfig:
    return CycleConfig(**({"window_seconds": 10, "max_length": 5, "rho_lo": .8, "eps": .1,
                          "deg_cap": 10, "expansion_budget": 1000} | changes))


def test_cycle_exact_unique_ties_and_amount_pruning() -> None:
    data = frame([(0, 1, 0, 10, 1), (1, 2, 0, 10, 1), (2, 0, 0, 10, 1), (0, 0, 0, 10, 1)])
    hits = detect_cycles(build_graph(data), config())
    assert len(hits) == 1 and hits[0].edge_ids == [0, 1, 2]
    data.loc[1, "amt_out"] = 1
    assert detect_cycles(build_graph(data), config()) == []


def test_cycle_wrong_order_large_span_and_budget_log(caplog) -> None:
    data = frame([(0, 1, 0, 10, 1), (1, 2, 2, 10, 1), (2, 0, 1, 10, 1)])
    assert detect_cycles(build_graph(data), config()) == []
    data.ts = [0, 1, 20]
    assert detect_cycles(build_graph(data), config()) == []
    data.ts = [0, 1, 2]
    detect_cycles(build_graph(data), config(expansion_budget=1))
    assert "budget" in caplog.text


def test_cycle_degree_cap_excluded_positive_logging(caplog) -> None:
    g = build_graph(frame([(0, 1, 0, 10, 1), (1, 2, 1, 10, 1), (2, 0, 2, 10, 1), (1, 3, 1, 10, 1)]))
    assert detect_cycles(g, config(deg_cap=1)) == []
    assert "deg_cap" in caplog.text and "positives" in caplog.text
