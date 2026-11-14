"""M1 planted data: identity, preservation, parsing and temporal isolation."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from aml.data.loader import RAW_COLUMNS, convert_csv
from aml.data.patterns_parser import match_attempts, parse_patterns
from aml.data.split import make_split, select_period, train_attempts
from aml.data.checklist import derive_parameters
from aml.data.checklist import prepare
from aml.data.labels import account_labels
from aml.config import Settings


def row(day: int, source: str = "A", target: str = "B", label: str = "1",
        amount: str = "10.00", bank: str = "001") -> list[str]:
    return [f"2022/09/{day:02d} 00:00", bank, source, "002", target,
            amount, "USD", amount, "USD", "ACH", label]


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(RAW_COLUMNS)
        writer.writerows(rows)


def write_patterns(path: Path, rows: list[list[str]], name: str = "FAN-OUT") -> None:
    path.write_text(f"BEGIN LAUNDERING ATTEMPT - {name}: fixture\n"
                    + "\n".join(",".join(r) for r in rows)
                    + f"\nEND LAUNDERING ATTEMPT - {name}\n", encoding="utf-8")


def test_compact_loading_preserves_identity_parallel_edges_and_ties(tmp_path: Path) -> None:
    rows = [row(2), row(1), row(1), row(3, bank="01")]
    rows.append(["2022/09/03 00:00", "001", "A", "001", "A",
                 "0.00", "USD", "0.00", "USD", "ACH", "1"])
    raw = tmp_path / "HI-Small_Trans.csv"
    write_csv(raw, rows)
    before = raw.read_bytes()
    result = convert_csv(raw, tmp_path / "processed", chunk_rows=2)
    table = pq.read_table(result.transactions_path)
    frame = table.to_pandas()
    assert len(frame) == 5
    assert frame.tx_id.tolist() == [0, 1, 2, 3, 4]
    assert frame.src.iloc[0] == frame.src.iloc[1] == frame.src.iloc[2]
    assert frame.src.iloc[3] != frame.src.iloc[0]  # bank identity preserved
    assert frame.is_self_loop.tolist() == [False, False, False, False, True]
    for field, dtype in [("src", "int32"), ("dst", "int32"), ("ts", "int64"),
                         ("amt_in", "float32"), ("amt_out", "float32")]:
        assert str(frame[field].dtype) == dtype
    assert str(frame.cur_in.dtype) == str(frame.fmt.dtype) == "category"
    mapping = pq.read_table(result.accounts_path).to_pandas()
    assert {"001", "01"}.issubset(set(mapping.bank))
    assert raw.read_bytes() == before


def test_factorization_is_chunk_size_independent(tmp_path: Path) -> None:
    raw = tmp_path / "HI-Small_Trans.csv"
    write_csv(raw, [row(1), row(2, "C", "A"), row(3, "A", "C")])
    a = convert_csv(raw, tmp_path / "a", chunk_rows=1)
    b = convert_csv(raw, tmp_path / "b", chunk_rows=3)
    assert pq.read_table(a.transactions_path).equals(pq.read_table(b.transactions_path))
    assert pq.read_table(a.accounts_path).equals(pq.read_table(b.accounts_path))


@pytest.mark.parametrize("bad", ["-1", "nan", "inf"])
def test_invalid_amount_rejected(tmp_path: Path, bad: str) -> None:
    raw = tmp_path / "HI-Small_Trans.csv"
    write_csv(raw, [row(1, amount=bad)])
    with pytest.raises(ValueError, match="amount"):
        convert_csv(raw, tmp_path / "out", chunk_rows=2)


def test_large_variant_rejected_before_open(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="variant"):
        convert_csv(tmp_path / "HI-Large_Trans.csv", tmp_path / "out", chunk_rows=2)


def test_parser_matches_exact_rows_before_float32_rounding(tmp_path: Path) -> None:
    rows = [row(2, amount="100000000.01"), row(1, amount="100000000.02")]
    raw, patterns = tmp_path / "HI-Small_Trans.csv", tmp_path / "patterns.txt"
    write_csv(raw, rows)
    write_patterns(patterns, rows)
    parsed = parse_patterns(patterns)
    result = convert_csv(raw, tmp_path / "out", chunk_rows=1, patterns=parsed)
    attempts = match_attempts(parsed, result.matches)
    assert len(attempts) == 1
    assert attempts[0].edge_ids == [1, 0]  # temporal order, original row ids
    assert attempts[0].t_start < attempts[0].t_end
    assert attempts[0].typology == "FAN-OUT"


def test_duplicate_rows_within_one_attempt_keep_both_edges(tmp_path: Path) -> None:
    rows = [row(1), row(1)]
    raw, patterns = tmp_path / "raw.csv", tmp_path / "patterns.txt"
    write_csv(raw, rows)
    write_patterns(patterns, rows)
    parsed = parse_patterns(patterns)
    loaded = convert_csv(raw, tmp_path / "out", chunk_rows=1, patterns=parsed)
    assert match_attempts(parsed, loaded.matches)[0].edge_ids == [0, 1]


@pytest.mark.parametrize("text", [
    "BEGIN LAUNDERING ATTEMPT - CYCLE\n",
    "END LAUNDERING ATTEMPT - CYCLE\n",
    "BEGIN LAUNDERING ATTEMPT - CYCLE\nEND LAUNDERING ATTEMPT - FAN-OUT\n",
    "BEGIN LAUNDERING ATTEMPT - CYCLE\nEND LAUNDERING ATTEMPT - CYCLE\n",
])
def test_malformed_pattern_blocks_fail(tmp_path: Path, text: str) -> None:
    path = tmp_path / "patterns.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_patterns(path)


def test_unmatched_or_ambiguous_membership_fails(tmp_path: Path) -> None:
    raw, patterns = tmp_path / "raw.csv", tmp_path / "patterns.txt"
    write_patterns(patterns, [row(1)])
    parsed = parse_patterns(patterns)
    for rows in [[row(2)], [row(1), row(1)]]:
        write_csv(raw, rows)
        loaded = convert_csv(raw, tmp_path / "out", chunk_rows=1, patterns=parsed)
        with pytest.raises(ValueError, match="match"):
            match_attempts(parsed, loaded.matches)


def test_whole_day_split_period_local_labels_and_straddling_attempt(tmp_path: Path) -> None:
    raw, patterns = tmp_path / "raw.csv", tmp_path / "patterns.txt"
    rows = [row(d, label="1" if d in (6, 7) else "0") for d in range(1, 11)]
    write_csv(raw, rows)
    write_patterns(patterns, [rows[5], rows[6]])
    parsed = parse_patterns(patterns)
    loaded = convert_csv(raw, tmp_path / "out", chunk_rows=3, patterns=parsed)
    frame = pq.read_table(loaded.transactions_path).to_pandas()
    split = make_split(frame.ts.to_numpy(), (0.6, 0.2, 0.2))
    assert [len(select_period(frame, split, p)) for p in ("train", "validation", "test")] == [6, 2, 2]
    assert [int(select_period(frame, split, p).label.sum()) for p in ("train", "validation", "test")] == [1, 1, 0]
    attempt = match_attempts(parsed, loaded.matches)[0]
    assert split.period_of(attempt.t_start) == "train"
    assert split.period_of(attempt.t_end) == "validation"
    assert train_attempts([attempt], split) == []  # no look-ahead in fitted durations
    with pytest.raises(AssertionError, match="test"):
        select_period(frame, split, "test", for_tuning=True)


def test_short_span_uses_distinct_timestamp_boundaries() -> None:
    split = make_split(np.arange(10, dtype=np.int64), (0.6, 0.2, 0.2))
    assert split.method == "timestamp_quantiles"
    assert split.train_end == 6 and split.validation_end == 8
    with pytest.raises(ValueError, match="distinct"):
        make_split(np.array([0, 0]), (0.6, 0.2, 0.2))


def test_quantile_split_handles_sparse_tail_and_preserves_timestamp_ties() -> None:
    ts = np.repeat(np.arange(10, dtype=np.int64) * 86400, 100)
    ts = np.append(ts, 18 * 86400)
    split = make_split(ts, (0.6, 0.2, 0.2), method="timestamp_quantiles")
    assert split.train_end == 6 * 86400
    assert split.validation_end == 8 * 86400
    assert split.period_of(int(ts[-1])) == "test"
    assert sum(ts < split.train_end) == 600


def test_parameter_derivation_rejects_test_rows(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    write_csv(raw, [row(d) for d in range(1, 11)])
    loaded = convert_csv(raw, tmp_path / "out", chunk_rows=3)
    frame = pq.read_table(loaded.transactions_path).to_pandas()
    split = make_split(frame.ts.to_numpy(), (0.6, 0.2, 0.2))
    settings = Settings()
    with pytest.raises(AssertionError, match="train"):
        derive_parameters(frame, [], split, settings)
    train = select_period(frame, split, "train", for_tuning=True)
    first = derive_parameters(train, [], split, settings)
    assert first == derive_parameters(train.copy(), [], split, settings)
    assert first["fan"]["window_seconds"] <= split.train_end - split.start


def test_account_population_and_labels_are_period_local() -> None:
    frame = pd.DataFrame({"src": [0, 1, 3], "dst": [1, 2, 3], "label": [0, 1, 0]})
    accounts, labels = account_labels(frame)
    assert accounts.tolist() == [0, 1, 2, 3]
    assert labels.tolist() == [False, True, True, False]


def test_prepare_tiny_fixture_reports_exact_counts_and_train_parameters(tmp_path: Path) -> None:
    raw, patterns = tmp_path / "HI-Small_Trans.csv", tmp_path / "HI-Small_Patterns.txt"
    benign = [row(day, f"normal-{i}", f"normal-{i+1}", label="0")
              for day in range(1, 11) for i in range(30)]
    fan = [row(day, "hub", f"leaf-{i}") for i, day in enumerate([1, 2, 3])]
    write_csv(raw, benign + fan)
    write_patterns(patterns, fan)
    settings = Settings.model_validate({"data": {"transactions": raw, "patterns": patterns,
        "processed_dir": tmp_path / "data" / "processed", "chunk_rows": 31},
        "output_dir": tmp_path / "outputs"})
    report = prepare(settings)
    assert report["transactions"] == 303
    assert report["positive_transactions"] == 3
    assert report["matched_pattern_transactions"] == 3
    assert report["positive_transactions_outside_patterns"] == 0
    assert report["initial_parameters"]["fan"] == {"window_seconds": 172800, "k": 3}
    assert report["periods"]["validation"]["positive_transactions"] == 0
    assert report["periods"]["test"]["positive_transactions"] == 0
    assert (tmp_path / "outputs" / "m1" / "checklist.md").is_file()
    assert Settings.from_yaml(tmp_path / "outputs" / "m1" / "measured_config.yaml").detectors == report["initial_parameters"]
    # Re-auditing a prepared dataset must not open either raw data file.
    from unittest.mock import patch
    with patch("aml.data.checklist.convert_csv", side_effect=AssertionError("raw CSV reread")), \
         patch("aml.data.checklist.parse_patterns", side_effect=AssertionError("raw patterns reread")):
        repeated = prepare(settings, reuse=True)
    assert repeated["initial_parameters"] == report["initial_parameters"]
    assert repeated["periods"] == report["periods"]


def test_missing_patterns_fail_before_transaction_access(tmp_path: Path) -> None:
    settings = Settings.model_validate({"data": {
        "transactions": tmp_path / "HI-Large_Trans.csv", "patterns": tmp_path / "missing.txt"}})
    with pytest.raises(FileNotFoundError, match="patterns"):
        prepare(settings)
