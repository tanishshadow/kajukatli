"""M1 descriptive audit and train-only initialization of detector parameters."""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from aml.config import Settings
from aml.data.labels import TYPOLOGY_MAPPING, period_counts
from aml.data.loader import RAW_COLUMNS, LoadResult, convert_csv
from aml.data.patterns_parser import Attempt, match_attempts, parse_patterns
from aml.data.split import TemporalSplit, make_split, select_period, train_attempts


def iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def distribution(values: list[int] | np.ndarray, quantiles: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(array), "quantiles": {
        str(q): float(np.quantile(array, q)) if len(array) else None for q in quantiles}}


def pattern_statistics(train: pd.DataFrame, attempts: list[Attempt]) -> dict[str, list[int]]:
    """Topology and link delays only from known train attempts, before FX fitting."""
    indexed = train.set_index("tx_id", drop=False)
    stats: dict[str, list[int]] = {key: [] for key in (
        "fan_sizes", "fan_durations", "scatter_sizes", "gather_in_sizes", "gather_out_sizes",
        "two_stage_durations", "cycle_durations", "chain_dwells", "all_durations")}
    for attempt in attempts:
        edges = indexed.loc[attempt.edge_ids]
        edges = edges.loc[~edges.is_self_loop]
        duration = attempt.t_end - attempt.t_start
        stats["all_durations"].append(duration)
        outgoing = edges.groupby("src").dst.nunique()
        incoming = edges.groupby("dst").src.nunique()
        if attempt.typology in ("FAN-OUT", "FAN-IN") and len(edges):
            sizes = outgoing if attempt.typology == "FAN-OUT" else incoming
            stats["fan_sizes"].append(int(sizes.max()))
            stats["fan_durations"].append(duration)
        elif attempt.typology == "SCATTER-GATHER" and len(edges):
            source, sink = int(outgoing.idxmax()), int(incoming.idxmax())
            middle = set(edges.loc[edges.src == source, "dst"]) & set(edges.loc[edges.dst == sink, "src"])
            stats["scatter_sizes"].append(len(middle - {source, sink}))
            stats["two_stage_durations"].append(duration)
        elif attempt.typology == "GATHER-SCATTER" and len(edges):
            hubs = sorted(set(outgoing.index) & set(incoming.index))
            if hubs:
                hub = max(hubs, key=lambda node: (int(incoming[node]) + int(outgoing[node]), -node))
                stats["gather_in_sizes"].append(int(incoming[hub]))
                stats["gather_out_sizes"].append(int(outgoing[hub]))
            stats["two_stage_durations"].append(duration)
        elif attempt.typology == "CYCLE":
            stats["cycle_durations"].append(duration)
        elif attempt.typology == "STACK":
            # Only temporally ordered adjacent edges; no claim that all STACKs
            # satisfy the future amount or minimal-path-length constraints.
            for edge in edges.itertuples(index=False):
                following = edges.loc[(edges.src == edge.dst) & (
                    (edges.ts > edge.ts) | ((edges.ts == edge.ts) & (edges.tx_id > edge.tx_id)))]
                stats["chain_dwells"].extend(int(t - edge.ts) for t in following.ts)
    return stats


def derive_parameters(train: pd.DataFrame, attempts: list[Attempt], split: TemporalSplit,
                      settings: Settings) -> dict[str, Any]:
    assert len(train) and train.ts.ge(split.start).all() and train.ts.lt(split.train_end).all(), \
        "parameter fitting accepts train rows only"
    assert all(split.start <= a.t_start <= a.t_end < split.train_end for a in attempts), \
        "parameter fitting accepts fully contained train attempts only"
    policy = settings.derivation
    stats = pattern_statistics(train, attempts)
    span = max(policy.minimum_window_seconds, int(train.ts.max() - train.ts.min()))
    gaps = np.diff(np.sort(train.ts.unique()))
    resolution = int(np.gcd.reduce(gaps)) if len(gaps) else policy.minimum_window_seconds
    minimum = max(policy.minimum_window_seconds, resolution)

    def window(values: list[int]) -> int:
        candidate = float(np.quantile(values, policy.duration_quantile)) if values else span * policy.fallback_window_fraction
        return min(span, max(minimum, math.ceil(candidate / minimum) * minimum))

    def size(values: list[int]) -> int:
        valid = [v for v in values if v >= policy.minimum_counterparties]
        return max(policy.minimum_counterparties, math.floor(float(np.quantile(valid, policy.size_quantile)))) if valid else policy.minimum_counterparties

    fan_window = window(stats["fan_durations"])
    return {
        "fan": {"window_seconds": fan_window, "k": size(stats["fan_sizes"])},
        "scatter_gather": {"window_seconds": window(stats["two_stage_durations"]),
                           "m": size(stats["scatter_sizes"]), "m_in": size(stats["gather_in_sizes"]),
                           "m_out": size(stats["gather_out_sizes"])},
        "cycle": {"window_seconds": window(stats["cycle_durations"])},
        "pass_through": {"dwell_seconds": window(stats["chain_dwells"])},
        "merge": {"window_seconds": max(minimum, math.ceil(window(stats["all_durations"]) * policy.merge_window_fraction / minimum) * minimum)},
        "baseline": {"window_seconds": fan_window},
        "dampening": {"regularity_window_seconds": span, "bins": policy.regularity_bins,
                      "active_bin_fraction": policy.active_bin_fraction, "maximum_count_cv": policy.maximum_count_cv},
    }


def degree_statistics(train: pd.DataFrame, attempts: list[Attempt], settings: Settings) -> dict[str, Any]:
    accounts = np.union1d(train.src, train.dst)
    positive = train.loc[train.label == 1]
    positives = np.union1d(positive.src, positive.dst)
    normal = accounts[~np.isin(accounts, positives)]
    no_loops = train.loc[~train.is_self_loop]
    out_degree = no_loops.groupby("src").dst.nunique().reindex(accounts, fill_value=0)
    in_degree = no_loops.groupby("dst").src.nunique().reindex(accounts, fill_value=0)
    indexed = train.set_index("tx_id")
    hubs: dict[str, set[int]] = {"FAN-OUT": set(), "FAN-IN": set()}
    for attempt in attempts:
        if attempt.typology in hubs:
            edges = indexed.loc[attempt.edge_ids]
            edges = edges.loc[~edges.is_self_loop]
            degrees = edges.groupby("src").dst.nunique() if attempt.typology == "FAN-OUT" else edges.groupby("dst").src.nunique()
            if len(degrees):
                hubs[attempt.typology].add(int(degrees.idxmax()))
    q = settings.derivation.degree_quantiles
    return {"scope": "distinct counterparties over train, excluding self-loops; not rolling-window degrees",
            "normal_definition": "accounts touching no labeled train transaction",
            "normal_out": distribution(out_degree.loc[normal].to_numpy(), q),
            "normal_in": distribution(in_degree.loc[normal].to_numpy(), q),
            "fan_out_hubs_out": distribution(out_degree.loc[sorted(hubs["FAN-OUT"])].to_numpy(), q),
            "fan_in_hubs_in": distribution(in_degree.loc[sorted(hubs["FAN-IN"])].to_numpy(), q)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(settings: Settings, *, reuse: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    # Fail before opening transactions when ground-truth attempts are absent.
    if not settings.data.patterns.is_file():
        raise FileNotFoundError("HI-Small patterns file is missing; attempt recall requires it")
    if not settings.data.transactions.is_file():
        raise FileNotFoundError("HI-Small transactions missing; use planted fixtures, no real-data results exist")
    if reuse:
        previous = json.loads((settings.output_dir / "m1" / "checklist.json").read_text(encoding="utf-8"))
        for path in (settings.data.transactions, settings.data.patterns):
            if previous["files_bytes"].get(str(path)) != path.stat().st_size:
                raise ValueError("source size/path changed; refusing stale prepared data")
        loaded = LoadResult(transactions_path=settings.data.processed_dir / "transactions.parquet",
                            accounts_path=settings.data.processed_dir / "accounts.parquet",
                            row_count=previous["transactions"], account_count=previous["accounts"],
                            sampled_peak_rss_bytes=previous["memory"]["loader_sampled_peak_rss_bytes"],
                            process_peak_rss_bytes=previous["memory"]["loader_process_peak_rss_bytes"], matches={})
        attempts = [Attempt.model_validate({key: row[key] for key in Attempt.model_fields})
                    for row in pq.read_table(settings.data.processed_dir / "attempts.parquet").to_pylist()]
        load_seconds = previous["timing_seconds"]["load_and_pattern_parse"]
    else:
        patterns = parse_patterns(settings.data.patterns)
        loaded = convert_csv(settings.data.transactions, settings.data.processed_dir,
                             chunk_rows=settings.data.chunk_rows, patterns=patterns)
        load_seconds = time.perf_counter() - started
        attempts = match_attempts(patterns, loaded.matches)
    # The raw CSV is never used again after compact conversion.
    frame = pq.read_table(loaded.transactions_path).to_pandas()
    split = make_split(frame.ts.to_numpy(), settings.split_fractions, method=settings.split_method)
    train = select_period(frame, split, "train", for_tuning=True)
    fitting_attempts = train_attempts(attempts, split)
    parameters = derive_parameters(train, fitting_attempts, split, settings)
    output = settings.output_dir / "m1"
    output.mkdir(parents=True, exist_ok=True)
    attempt_records = [{**a.model_dump(), "period": split.period_of(a.t_start),
                        "straddles": split.period_of(a.t_start) != split.period_of(a.t_end)} for a in attempts]
    attempts_path = settings.data.processed_dir / "attempts.parquet"
    pq.write_table(pa.Table.from_pylist(attempt_records), attempts_path, compression="zstd")
    write_json(settings.data.processed_dir / "split.json", split.model_dump())
    measured = settings.model_dump(mode="json")
    measured["detectors"] = parameters
    (output / "measured_config.yaml").write_text(yaml.safe_dump(measured, sort_keys=False), encoding="utf-8")
    positives = frame.loc[frame.label == 1]
    loops = frame.is_self_loop
    positive_loops = int(frame.loc[loops, "label"].sum())
    unique_times = np.sort(frame.ts.unique())
    gaps = np.diff(unique_times)
    cross_currency = frame.cur_out.astype(str) != frame.cur_in.astype(str)
    member_ids = {edge for a in attempts for edge in a.edge_ids}
    train_start_attempts = [a for a in attempts if split.period_of(a.t_start) == "train"]
    file_paths = [settings.data.transactions, settings.data.patterns, loaded.transactions_path,
                  loaded.accounts_path, attempts_path]
    report: dict[str, Any] = {
        "command": "python -m aml prepare --config configs/default.yaml" + (" --reuse" if reuse else ""),
        "reused_parquet": reuse,
        "source_columns": RAW_COLUMNS,
        "account_identity": "(bank string, account string); leading zeros preserved; src/dst factorized int32",
        "timestamp_encoding": "naive dataset clock encoded as UTC epoch seconds; source timezone unspecified",
        "tx_id": "zero-based CSV data-row number; Parquet retains source order; M2 graph sorts (ts, tx_id)",
        "transactions": len(frame), "accounts": loaded.account_count,
        "positive_transactions": len(positives), "positive_rate": len(positives) / len(frame),
        "first_timestamp": iso(int(unique_times[0])), "last_timestamp": iso(int(unique_times[-1])),
        "span_seconds": int(unique_times[-1] - unique_times[0]),
        "calendar_days": int(unique_times[-1] // 86400 - unique_times[0] // 86400 + 1),
        "timestamp_grid_seconds": int(np.gcd.reduce(gaps)), "minimum_timestamp_gap_seconds": int(gaps.min()),
        "split": {**split.model_dump(), "iso_boundaries": {p: [iso(x) for x in split.bounds(p)]
                   for p in ("train", "validation", "test")}, "intervals": "[start, end)"},
        "periods": {p: period_counts(frame, attempts, split, p) for p in ("train", "validation", "test")},
        "daily_counts": [{"day": iso(int(day) * 86400)[:10], "transactions": int(values["count"]),
                          "positive_transactions": int(values["sum"])}
                         for day, values in frame.groupby(frame.ts // 86400).label.agg(["count", "sum"]).iterrows()],
        "attempts": len(attempts), "matched_pattern_transactions": len(member_ids),
        "positive_transactions_outside_patterns": int((~positives.tx_id.isin(member_ids)).sum()),
        "typology_mapping": {name: TYPOLOGY_MAPPING.get(name, {"detector": None, "coverage": "not covered"})
                             for name in sorted({a.typology for a in attempts})},
        "straddling_attempts": [{"attempt_id": a.attempt_id, "typology": a.typology,
                                 "start_period": split.period_of(a.t_start), "end_period": split.period_of(a.t_end),
                                 "start": iso(a.t_start), "end": iso(a.t_end)}
                                for a in attempts if split.period_of(a.t_start) != split.period_of(a.t_end)],
        "parameter_provenance": {"scope": "train transactions and fully contained train attempts only",
                                 "train_attempts_used": len(fitting_attempts),
                                 "train_straddlers_excluded": len(train_start_attempts) - len(fitting_attempts),
                                 "policy": settings.derivation.model_dump(),
                                 "train_pattern_statistics": {name: distribution(values, settings.derivation.degree_quantiles)
                                     for name, values in pattern_statistics(train, fitting_attempts).items()},
                                 "all_attempt_durations_by_typology_descriptive_only": {
                                     name: distribution([a.t_end - a.t_start for a in attempts if a.typology == name],
                                                        settings.derivation.degree_quantiles)
                                     for name in sorted({a.typology for a in attempts})}},
        "initial_parameters": parameters,
        "degree_statistics_train": degree_statistics(train, fitting_attempts, settings),
        "currency": {"paid": {str(k): int(v) for k, v in frame.cur_out.value_counts().items()},
                     "received": {str(k): int(v) for k, v in frame.cur_in.value_counts().items()},
                     "cross_currency_transactions": int(cross_currency.sum()),
                     "cross_currency_positives": int((cross_currency & (frame.label == 1)).sum()),
                     "base_currency_train": str(pd.concat([train.cur_out.astype(str), train.cur_in.astype(str)]).value_counts().sort_index().idxmax()),
                     "base_currency_policy": "most frequent across paid+received train currency observations; lexical ties"},
        "self_loops": {"transactions": int(loops.sum()), "positive_transactions": positive_loops,
                       "edge_recall_ceiling_if_excluded": 1 - positive_loops / len(positives) if len(positives) else None,
                       "ceiling_scope": "conditional: detectors excluding self-loops; loops retained in normalized data"},
        "deg_cap_recall_ceiling": "not measured; no degree cap applied in M1",
        "files_bytes": {str(path): path.stat().st_size for path in file_paths},
        "memory": {"loader_sampled_peak_rss_bytes": loaded.sampled_peak_rss_bytes,
                   "loader_process_peak_rss_bytes": loaded.process_peak_rss_bytes,
                   "loader_peak_method": "Windows process lifetime peak working set when available; otherwise sampled RSS",
                   "checklist_process_peak_rss_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss),
                   "host_ram_bytes": psutil.virtual_memory().total,
                   "normalized_frame_bytes": int(frame.memory_usage(deep=True).sum())},
        "timing_seconds": {"load_and_pattern_parse": load_seconds, "prepare": time.perf_counter() - started},
        "timing_note": "load timing and loading peak are from original conversion; prepare timing is this invocation",
        "notes": ["Synthetic data; pattern-targeted detectors are favored. No real-world performance claim.",
                  "Full-period counts are descriptive only, not a test evaluation. No detector or scorer has run.",
                  "Degree summaries cover the train span, not rolling windows. Initial k comes from planted train fan sizes.",
                  "Window sizes are initial train-derived values, not tuned detector performance results.",
                  "Timestamp quantiles avoid assigning almost all transactions to train because of the sparse late tail; equal timestamps stay in one period.",
                  "The patterns file does not cover every labeled transaction. Attempt recall is limited to enumerated attempts; edge/account evaluation uses all labels.",
                  "Initial fan k may be permissive; no precision or runtime claim is made before detector validation and tuning.",
                  "Dampening uses occupied activity bins and transaction-count coefficient of variation within train span; no 30-day assumption.",
                  "STACK is only partially covered; BIPARTITE and RANDOM have no dedicated detector.",
                  "Missing FX handling is deferred to the shared utility before M6."]}
    write_json(output / "checklist.json", report)
    (output / "checklist.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report: dict[str, Any]) -> str:
    lines = ["# M1 data checklist", "", "Command: `" + report["command"] + "`", "",
             "No detector metrics or final test results exist yet.", "",
             f"Transactions: {report['transactions']:,}; positive: {report['positive_transactions']:,} "
             f"({report['positive_rate']:.6%}); accounts: {report['accounts']:,}.",
             f"Timeline: {report['first_timestamp']} to {report['last_timestamp']}; "
             f"{report['calendar_days']} calendar days; grid {report['timestamp_grid_seconds']} seconds.", "",
             "## Period-local descriptive counts", "",
             "| Period | Start (inclusive) | End (exclusive) | Transactions | Positive edges | Positive accounts | Attempts |",
             "|---|---|---|---:|---:|---:|---:|"]
    for period, values in report["periods"].items():
        start, end = report["split"]["iso_boundaries"][period]
        lines.append(f"| {period} | {start} | {end} | {values['transactions']} | {values['positive_transactions']} | {values['positive_accounts']} | {values['attempts']} |")
    lines.extend(["", "## Attempt mapping and start-period counts", "",
                  "| Typology | Coverage | Train | Validation | Test |", "|---|---|---:|---:|---:|"])
    for name, mapping in report["typology_mapping"].items():
        counts = [report["periods"][p]["attempts_by_typology"][name] for p in ("train", "validation", "test")]
        lines.append(f"| {name} | {mapping['detector']}: {mapping['coverage']} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines.extend(["", "## Daily volume (split-design audit)", "",
                  "| Day | Transactions | Positive edges |", "|---|---:|---:|"])
    for day in report["daily_counts"]:
        lines.append(f"| {day['day']} | {day['transactions']} | {day['positive_transactions']} |")
    for heading, key in [("Train-derived initial configuration", "initial_parameters"),
                         ("Derivation and attempt durations", "parameter_provenance"),
                         ("Train degree distributions", "degree_statistics_train"),
                         ("Currency mix", "currency"), ("Self-loop ceiling", "self_loops"),
                         ("Files (bytes)", "files_bytes"), ("Memory (bytes)", "memory"),
                         ("Timing (seconds)", "timing_seconds")]:
        lines.extend(["", "## " + heading, "", "```json", json.dumps(report[key], indent=2), "```"])
    lines.extend(["", "## Straddling attempts (scored in start period only)", "",
                  "| Attempt | Typology | Start period | End period | Start | End |",
                  "|---|---|---|---|---|---|"])
    for attempt in report["straddling_attempts"]:
        lines.append("| " + " | ".join(str(attempt[key]) for key in
                     ("attempt_id", "typology", "start_period", "end_period", "start", "end")) + " |")
    lines.extend(["", "## Interpretation", ""] + ["- " + note for note in report["notes"]])
    lines.extend(["", "Timestamp convention: " + report["timestamp_encoding"],
                  "", "Timing convention: " + report["timing_note"],
                  "", "Transaction identity: " + report["tx_id"],
                  "", f"Matched pattern transactions: {report['matched_pattern_transactions']}; "
                  f"positive transactions outside patterns: {report['positive_transactions_outside_patterns']}.",
                  "", "Degree-cap ceiling: " + report["deg_cap_recall_ceiling"], ""])
    return "\n".join(lines)
