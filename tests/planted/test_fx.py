import pytest
from aml.graph.build import build_graph
from aml.data.fx import fit_fx
from tests_helpers import frame


def test_fx_train_only_median_chaining_and_missing_currency() -> None:
    data = frame([(0, 1, 0, 10, 0), (1, 2, 1, 10, 0), (2, 3, 2, 10, 0), (3, 4, 9, 10, 0)])
    data.cur_out = ["USD", "USD", "EUR", "XXX"]
    data.cur_in = ["EUR", "USD", "JPY", "USD"]
    data.amt_in = [20., 10., 30., 100000.]
    g = build_graph(data)
    fx = fit_fx(g, train_end=3)
    assert fx.base_currency == "USD"
    assert fx.factors["EUR"] == .5
    assert fx.factors["JPY"] == pytest.approx(1 / 6)
    assert "XXX" not in fx.factors
    assert fx.fit_max_ts == 2
