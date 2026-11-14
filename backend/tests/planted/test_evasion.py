import numpy as np
import pandas as pd
from aml.data.patterns_parser import Attempt
from aml.cluster.merge import EvidenceCluster
from aml.eval.evasion import (PerturbationTrace, _perturb, _structural_attempt_recall,
                              perturb_transactions, validate_perturbation)


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


def test_delay_strength_one_never_exceeds_seven_day_maximum():
    original = table()
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0, 1],
                      accounts=[0, 1, 2], t_start=1, t_end=2)
    changed = perturb_transactions(original, [attempt], 1.0, "delays", 123)
    old_times = original.set_index("tx_id").ts
    new_times = changed.set_index("tx_id").loc[old_times.index].ts
    offsets = new_times.loc[[0, 1]].to_numpy() - old_times.loc[[0, 1]].to_numpy()
    assert offsets.min() >= 0
    assert offsets.max() <= 7 * 86400
    assert np.all(np.diff(offsets) >= 0)


def test_combined_perturbation_delays_the_generated_split_paths():
    original = table()
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0, 1],
                      accounts=[0, 1, 2], t_start=1, t_end=2)
    changed, altered_attempts = _perturb(original, [attempt], 1.0, "combined", 123)
    generated = changed.loc[changed.tx_id.isin(altered_attempts[0].edge_ids)]
    assert len(generated) > len(attempt.edge_ids)
    assert generated.ts.max() - generated.ts.min() > original.ts.max() - original.ts.min()


def test_decoys_reuse_observed_benign_transaction_values():
    original = table()
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0, 1],
                      accounts=[0, 1, 2], t_start=1, t_end=2)
    changed = perturb_transactions(original, [attempt], 1.0, "decoys", 123)
    new = changed.loc[~changed.tx_id.isin(original.tx_id)]
    benign = original.loc[original.label == 0]
    observed = set(zip(benign.amt_out, benign.amt_in, benign.cur_out, benign.cur_in, benign.fmt))
    assert len(new) > 0
    assert all((row.amt_out, row.amt_in, row.cur_out, row.cur_in, row.fmt) in observed
               for row in new.itertuples())
    assert new.label.eq(0).all()


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("kind", ["amount_split", "delays", "decoys", "combined"])
def test_all_strengths_are_deterministic_valid_and_keep_original_provenance(kind, strength):
    original = table()
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0, 1],
                      accounts=[0, 1, 2], t_start=1, t_end=2)
    changed, changed_attempts = _perturb(original, [attempt], strength, kind, 77)
    repeated, repeated_attempts = _perturb(original, [attempt], strength, kind, 77)
    validate_perturbation(original, changed, strength, kind, [attempt])
    pd.testing.assert_frame_equal(changed, repeated)
    assert changed_attempts == repeated_attempts


def test_structural_recovery_requires_complete_split_path_in_same_cluster():
    attempt = Attempt(attempt_id="P", typology="STACK", description="", edge_ids=[0],
                      accounts=[0, 1], t_start=1, t_end=2)
    trace = PerturbationTrace(paths_by_attempt={"P": {0: [(10, 11), (12, 13), (14, 15), (16, 17)]}})
    partial_paths = [EvidenceCluster(accounts=[0, 1], edge_ids=[10, 11, 12, 14], hit_ids=[0], t_start=1, t_end=2)]
    assert _structural_attempt_recall([attempt], partial_paths, trace, tau=0.5) == 0.0
    complete_path = [EvidenceCluster(accounts=[0, 1], edge_ids=[10, 11, 12, 13], hit_ids=[0], t_start=1, t_end=2)]
    assert _structural_attempt_recall([attempt], complete_path, trace, tau=0.5) == 1.0
