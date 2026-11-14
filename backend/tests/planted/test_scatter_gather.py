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


def test_gather_overlapping_stages_respect_flow_causality() -> None:
    data = frame([(0, 2, 0, 10, 1), (2, 3, 1, 8, 1), (1, 2, 2, 10, 1), (2, 4, 3, 12, 1)])
    assert any(h.typology == "gather_scatter" for h in detect_scatter_gather(build_graph(data), config()))
    data.loc[1, "amt_out"] = 15
    data.loc[3, "amt_out"] = 5
    assert not any(h.typology == "gather_scatter" for h in detect_scatter_gather(build_graph(data), config()))


def test_scatter_delayed_stage_outside_window_and_missing_fx_logged(caplog) -> None:
    from aml.data.fx import FXModel
    data = frame([(0, 1, 0, 10, 1), (0, 2, 0, 10, 1), (1, 3, 20, 10, 1), (2, 3, 20, 10, 1)])
    assert detect_scatter_gather(build_graph(data), config()) == []
    data.ts = [0, 0, 1, 1]
    data.loc[0, ["cur_in", "cur_out"]] = "UNKNOWN"
    cfg = config().model_copy(update={"fx": FXModel(base_currency="USD", factors={"USD": 1.}, fit_max_ts=0)})
    hits = detect_scatter_gather(build_graph(data), cfg)
    assert hits and hits[0].meta["fx_missing"] and "fx_missing" in caplog.text


def test_rolling_gather_matches_brute_force_windows() -> None:
    import numpy as np
    from aml.detect.scatter_gather import _gather_paths
    rng = np.random.default_rng(42)
    edges = [(int(rng.integers(1, 5)), 0, i, float(rng.integers(1, 20)), 0) if i % 2 == 0
             else (0, int(rng.integers(5, 9)), i, float(rng.integers(1, 20)), 0) for i in range(100)]
    g = build_graph(frame(edges))
    paid, received = g.amt_out.astype(float), g.amt_in.astype(float)
    matches, _ = _gather_paths(np.array([0]), g.src, g.dst, g.ts, g.in_idx.indptr, g.in_idx.edge_ids,
        g.out_idx.indptr, g.out_idx.edge_ids, paid, received, 15, 2, 2, .8, .1)
    actual = {(tuple(ins), tuple(outs)) for _, ins, outs, _ in matches}
    expected = set()
    for end in g.out_idx[0]:
        ins = g.in_idx[0][(g.in_idx[0] < end) & (g.ts[g.in_idx[0]] >= g.ts[end] - 15)]
        if not len(ins):
            continue
        outs = g.out_idx[0][(g.out_idx[0] <= end) & (g.out_idx[0] > ins[0]) & (g.ts[g.out_idx[0]] >= g.ts[end] - 15)]
        if len(set(g.src[ins])) < 2 or len(set(g.dst[outs])) < 2:
            continue
        if not .8 <= paid[outs].sum() / received[ins].sum() <= 1.1:
            continue
        if any(paid[outs[:j + 1]].sum() > received[ins[ins < out]].sum() * 1.1 for j, out in enumerate(outs)):
            continue
        expected.add((tuple(ins), tuple(outs)))
    assert actual == expected
