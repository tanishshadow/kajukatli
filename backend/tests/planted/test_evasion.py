import pandas as pd
from aml.data.patterns_parser import Attempt
from aml.eval.evasion import _perturb, perturb_transactions, validate_perturbation


def table():
    return pd.DataFrame({"tx_id":[0,1,2],"ts":[1,2,3],"src":[0,1,8],"dst":[1,2,9],
        "amt_out":[10.,10.,5.],"amt_in":[10.,10.,5.],"cur_out":["USD"]*3,
        "cur_in":["USD"]*3,"fmt":["ACH"]*3,"label":[1,1,0],"is_self_loop":[False]*3})


def test_real_amount_split_changes_edges_preserves_value_labels_deterministically():
    original = table()
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0,1], accounts=[0,1,2], t_start=1, t_end=2)
    first = perturb_transactions(original, [attempt], .5, "amount_split", 42)
    second = perturb_transactions(original, [attempt], .5, "amount_split", 42)
    validate_perturbation(original, first, .5, "amount_split", [attempt])
    pd.testing.assert_frame_equal(first, second)
    assert len(first) > len(original)
    assert set(first.loc[first.label == 1, "label"]) == {1}


def test_zero_strength_reproduces_table():
    original = table()
    attempt = Attempt(attempt_id="P", typology="CYCLE", description="", edge_ids=[0], accounts=[0,1], t_start=1, t_end=1)
    changed = perturb_transactions(original, [attempt], 0, "combined", 1)
    validate_perturbation(original, changed, 0, "combined", [attempt])


import pytest


@pytest.mark.parametrize("kind", ["amount_split", "delays", "decoys", "combined"])
def test_each_perturbation_changes_table_deterministically_and_keeps_labels(kind):
    original = table()
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0, 1],
                      accounts=[0, 1, 2], t_start=1, t_end=2)
    first, altered_attempts = _perturb(original, [attempt], .5, kind, 42)
    second, _ = _perturb(original, [attempt], .5, kind, 42)
    validate_perturbation(original, first, .5, kind, [attempt])
    pd.testing.assert_frame_equal(first, second)
    assert set(first.label.unique()) <= {0, 1}
    if kind in ("amount_split", "combined"):
        assert len(altered_attempts[0].edge_ids) > len(attempt.edge_ids)
        assert first.loc[first.tx_id.isin(altered_attempts[0].edge_ids), "label"].eq(1).all()
    if kind in ("delays", "combined"):
        path = first.set_index("tx_id").loc[altered_attempts[0].edge_ids].sort_values(["ts", "tx_id"])
        assert path.ts.is_monotonic_increasing
    if kind in ("decoys", "combined"):
        original_ids = set(original.tx_id)
        assert first.loc[~first.tx_id.isin(original_ids), "label"].eq(0).any()
