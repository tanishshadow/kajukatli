"""Temporal multigraph preservation and CSR correctness against small oracles."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml.graph.build import build_graph, load_graph


def fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "tx_id": [40, 10, 30, 20, 50], "ts": [200, 100, 100, 100, 300],
        "src": [2, 0, 0, 1, 2], "dst": [0, 1, 1, 2, 2],
        "amt_out": [4., 1., 3., 2., 5.], "amt_in": [8., 2., 6., 4., 10.],
        "cur_out": ["USD", "EUR", "USD", "EUR", "JPY"],
        "cur_in": ["EUR", "USD", "USD", "JPY", "JPY"],
        "fmt": ["ACH"] * 5, "label": [1, 0, 1, 1, 1],
        "is_self_loop": [False, False, False, False, True],
    })


def test_total_order_preserves_edge_attributes_parallel_edges_and_self_loop() -> None:
    original = fixture()
    graph = build_graph(original, node_count=4)
    assert graph.tx_id.tolist() == [10, 20, 30, 40, 50]
    assert graph.ts.tolist() == [100, 100, 100, 200, 300]
    assert graph.src.tolist() == [0, 1, 0, 2, 2]
    assert graph.dst.tolist() == [1, 2, 1, 0, 2]
    assert graph.amt_out.tolist() == [1., 2., 3., 4., 5.]
    assert graph.amt_in.tolist() == [2., 4., 6., 8., 10.]
    assert graph.label.tolist() == [0, 1, 1, 1, 1]
    assert graph.is_self_loop.tolist() == [False, False, False, False, True]
    assert graph.currency_names == ("EUR", "JPY", "USD")
    assert [graph.currency_names[i] for i in graph.cur_in] == ["USD", "JPY", "USD", "EUR", "JPY"]
    assert original.equals(fixture())
    assert graph.out_idx[0].tolist() == [0, 2]
    assert graph.in_idx[2].tolist() == [1, 4]
    assert graph.out_idx[3].tolist() == graph.in_idx[3].tolist() == []


def test_transaction_ids_and_graph_positions_are_explicitly_distinct() -> None:
    graph = build_graph(fixture())
    assert graph.positions_for_tx_ids(np.array([50, 10, 30, 10])).tolist() == [4, 0, 2, 0]
    assert graph.tx_id[graph.out_idx[0]].tolist() == [10, 30]
    with pytest.raises(KeyError):
        graph.positions_for_tx_ids(np.array([11]))
    with pytest.raises(IndexError):
        _ = graph.out_idx[-1]


def test_csr_matches_brute_force_and_build_is_input_order_independent() -> None:
    rng = np.random.default_rng(42)
    frame = pd.concat([fixture()] * 40, ignore_index=True)
    frame.tx_id = rng.permutation(len(frame)) * 3
    frame.ts = rng.integers(0, 5, len(frame))
    frame.src = rng.integers(0, 12, len(frame))
    frame.dst = rng.integers(0, 12, len(frame))
    frame.is_self_loop = frame.src == frame.dst
    graph = build_graph(frame, node_count=13)
    other = build_graph(frame.sample(frac=1, random_state=7), node_count=13)
    for name in graph.edge_array_names():
        np.testing.assert_array_equal(getattr(graph, name), getattr(other, name))
    for node in range(13):
        np.testing.assert_array_equal(graph.out_idx[node], np.flatnonzero(graph.src == node))
        np.testing.assert_array_equal(graph.in_idx[node], np.flatnonzero(graph.dst == node))


def test_arrays_are_compact_contiguous_and_read_only() -> None:
    graph = build_graph(fixture())
    for name in graph.edge_array_names():
        array = getattr(graph, name)
        assert array.flags.c_contiguous and not array.flags.writeable
    assert graph.src.dtype == np.int32 and graph.ts.dtype == np.int64
    assert graph.amt_out.dtype == np.float32 and graph.cur_in.dtype == np.int32
    with pytest.raises(ValueError):
        graph.src[0] = 99
    with pytest.raises(ValueError):
        graph.out_idx.edge_ids[0] = 99


@pytest.mark.parametrize("column,value", [
    ("tx_id", [1, 1, 2, 3, 4]), ("src", [-1, 0, 0, 1, 2]),
    ("dst", [2**32, 1, 1, 2, 2]), ("ts", [1.5, 2., 3., 4., 5.]),
    ("amt_in", [float("nan"), 1, 1, 1, 1]), ("label", [2, 0, 1, 1, 1]),
    ("is_self_loop", [False] * 5), ("cur_in", [None, "USD", "USD", "USD", "USD"]),
])
def test_invalid_normalized_input_is_rejected(column: str, value: list[object]) -> None:
    frame = fixture()
    frame[column] = value
    with pytest.raises(ValueError):
        build_graph(frame)


def test_empty_graph_and_isolated_nodes() -> None:
    graph = build_graph(fixture().iloc[:0], node_count=3)
    assert graph.edge_count == 0 and graph.node_count == 3
    assert graph.out_idx.indptr.tolist() == [0, 0, 0, 0]
    assert graph.positions_for_tx_ids(np.array([], dtype=np.int64)).size == 0
    with pytest.raises(KeyError):
        graph.positions_for_tx_ids(np.array([0]))


def test_parquet_entry_point_preserves_all_nodes(tmp_path: Path) -> None:
    path = tmp_path / "transactions.parquet"
    accounts = tmp_path / "accounts.parquet"
    fixture().to_parquet(path, index=False)
    pd.DataFrame({"account_id": [0, 1, 2, 3]}).to_parquet(accounts, index=False)
    graph = load_graph(path, accounts)
    assert graph.node_count == 4 and graph.edge_count == 5
    with pytest.raises(ValueError):
        build_graph(fixture(), node_count=2)
