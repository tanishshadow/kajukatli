"""Validation-only, transaction-level evasion experiment with frozen systems."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from xgboost import XGBClassifier

from aml.baseline.threshold import BaselineConfig, baseline
from aml.cluster.merge import EvidenceCluster, MergeConfig, merge_hits
from aml.config import Settings
from aml.data.fx import FXModel, fit_fx
from aml.data.patterns_parser import Attempt
from aml.data.split import TemporalSplit
from aml.detect.common import Hit
from aml.detect.fan import FanConfig, detect_fans
from aml.eval.frozen import frozen_hash
from aml.graph.build import TemporalGraph, build_graph, load_graph
from aml.score.features import FeatureConfig, features_for_groups
from aml.score.model import ModelConfig, probabilities

Perturbation = Literal["amount_split", "delays", "decoys", "combined"]
KINDS: tuple[Perturbation, ...] = ("amount_split", "delays", "decoys", "combined")


def _perturb(frame: pd.DataFrame, attempts: list[Attempt], strength: float, kind: Perturbation,
             seed: int) -> tuple[pd.DataFrame, list[Attempt]]:
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    rng = np.random.default_rng(seed)
    original = frame.sort_values(["ts", "tx_id"], kind="stable").reset_index(drop=True)
    result = original.copy(deep=True)
    next_account = int(max(result.src.max(), result.dst.max())) + 1 if len(result) else 0
    next_tx = int(result.tx_id.max()) + 1 if len(result) else 0
    append: list[dict] = []
    attempt_updates: dict[str, Attempt] = {a.attempt_id: a.model_copy(deep=True) for a in attempts}
    split_edges: dict[int, list[int]] = {}
    if strength and kind in ("amount_split", "combined"):
        paths = max(2, int(np.ceil(2 + 8 * strength)))
        for attempt in attempts:
            updated_ids: list[int] = []
            for edge_id in attempt.edge_ids:
                matches = result.index[result.tx_id == edge_id]
                if len(matches) != 1:
                    continue
                row = result.loc[int(matches[0])].to_dict()
                result = result.drop(index=int(matches[0]))
                generated: list[int] = []
                for _ in range(paths):
                    intermediary = next_account
                    next_account += 1
                    first_id, second_id = next_tx, next_tx + 1
                    next_tx += 2
                    common = {k: v for k, v in row.items() if k not in {"tx_id", "src", "dst", "amt_out", "amt_in", "label", "is_self_loop"}}
                    amount_out = float(row["amt_out"]) / paths
                    amount_in = float(row["amt_in"]) / paths
                    append.append({**common, "tx_id": first_id, "src": int(row["src"]), "dst": intermediary,
                        "amt_out": amount_out, "amt_in": amount_in, "label": 1, "is_self_loop": False})
                    append.append({**common, "tx_id": second_id, "src": intermediary, "dst": int(row["dst"]),
                        "amt_out": amount_out, "amt_in": amount_in, "label": 1, "is_self_loop": False})
                    generated.extend((first_id, second_id))
                split_edges[edge_id] = generated
                updated_ids.extend(generated)
            old = attempt_updates[attempt.attempt_id]
            attempt_updates[attempt.attempt_id] = old.model_copy(update={"edge_ids": updated_ids,
                "accounts": sorted(set(old.accounts) | {int(v) for ids in split_edges.values() for v in []})})
            # Include all newly created intermediaries in this attempt.
            newly_created = [int(row["dst"]) for row in append if int(row["src"]) in attempt.accounts or int(row["dst"]) in attempt.accounts]
            attempt_updates[attempt.attempt_id] = attempt_updates[attempt.attempt_id].model_copy(
                update={"accounts": sorted(set(attempt.accounts) | set(newly_created))})

    if strength and kind in ("delays", "combined"):
        max_delay = int(round(strength * 7 * 86400))
        for attempt in attempts:
            ids = [e for e in attempt.edge_ids if e not in split_edges]
            positions = {int(tx): int(ts) for tx, ts in zip(result.tx_id, result.ts, strict=True)}
            ordered = sorted(ids, key=lambda edge: (positions.get(edge, attempt.t_start), edge))
            delay = 0
            prior = attempt.t_start
            for edge_id in ordered:
                extra = int(rng.integers(0, max_delay + 1)) if max_delay else 0
                current = max(positions.get(edge_id, prior), prior) + extra + delay
                result.loc[result.tx_id == edge_id, "ts"] = current
                delay += extra
                prior = current

    if strength and kind in ("decoys", "combined"):
        ring = sorted({int(account) for attempt in attempts for account in attempt.accounts})
        all_accounts = np.union1d(result.src, result.dst)
        normal = np.setdiff1d(all_accounts, np.asarray(ring, dtype=np.int64))
        count = int(np.ceil(strength * max(1, sum(len(a.edge_ids) for a in attempts))))
        if len(ring) and len(normal):
            for _ in range(count):
                source, target = int(rng.choice(ring)), int(rng.choice(normal))
                base = result.iloc[int(rng.integers(0, len(result)))].to_dict()
                common = {k: v for k, v in base.items() if k not in {"tx_id", "src", "dst", "amt_out", "amt_in", "label", "is_self_loop"}}
                noise = float(rng.uniform(1, 1000))
                append.append({**common, "tx_id": next_tx, "ts": int(rng.choice(result.ts)), "src": source, "dst": target,
                    "amt_out": noise, "amt_in": noise, "label": 0, "is_self_loop": False})
                next_tx += 1
                append.append({**common, "tx_id": next_tx, "ts": int(rng.choice(result.ts)), "src": target, "dst": source,
                    "amt_out": noise, "amt_in": noise, "label": 0, "is_self_loop": False})
                next_tx += 1
    if append:
        result = pd.concat([result, pd.DataFrame(append)], ignore_index=True)
    result = result.sort_values(["ts", "tx_id"], kind="stable").reset_index(drop=True)
    altered_attempts = list(attempt_updates.values())
    for i, attempt in enumerate(altered_attempts):
        positions = result.set_index("tx_id").ts
        valid = [edge for edge in attempt.edge_ids if edge in positions.index]
        if valid:
            altered_attempts[i] = attempt.model_copy(update={"t_start": int(positions.loc[valid].min()),
                "t_end": int(positions.loc[valid].max())})
    return result, altered_attempts


def perturb_transactions(frame: pd.DataFrame, attempts: list[Attempt], strength: float,
                         kind: Perturbation, seed: int) -> pd.DataFrame:
    return _perturb(frame, attempts, strength, kind, seed)[0]


def validate_perturbation(original: pd.DataFrame, altered: pd.DataFrame, strength: float,
                          kind: Perturbation, attempts: list[Attempt]) -> None:
    if strength == 0:
        pd.testing.assert_frame_equal(original.sort_values(["ts", "tx_id"]).reset_index(drop=True), altered)
        return
    assert len(altered) > len(original) or not np.array_equal(altered.src, original.src)
    assert set(altered.label.unique()) <= {0, 1}
    if kind in ("amount_split", "combined"):
        member_ids = {edge for attempt in attempts for edge in attempt.edge_ids}
        source_rows = original.loc[original.tx_id.isin(member_ids)]
        expected = source_rows["amt_out"].sum()
        original_sources = set(source_rows.src.astype(int))
        split = altered.loc[(altered.label == 1) & ~altered.tx_id.isin(original.tx_id) &
                            altered.src.isin(original_sources), "amt_out"].sum()
        assert np.isclose(expected, split, rtol=1e-6, atol=1e-5), (expected, split)
    if kind in ("delays", "combined"):
        assert altered["ts"].is_monotonic_increasing
    if kind == "decoys":
        new = altered.loc[~altered.tx_id.isin(original.tx_id)]
        assert len(new) > 0 and (new.label == 0).all()
    assert not altered.tx_id.duplicated().any()
    assert altered[["ts", "tx_id"]].values.tolist() == altered.sort_values(["ts", "tx_id"])[["ts", "tx_id"]].values.tolist()


def _detect_all(graph: TemporalGraph, settings: Settings, fx: FXModel) -> list[Hit]:
    hits = detect_fans(graph, FanConfig(**settings.detectors["fan"], fx=fx))
    from aml.detect.scatter_gather import ScatterConfig, detect_scatter_gather
    from aml.detect.cycles import CycleConfig, detect_cycles
    from aml.detect.passthrough import PassConfig, detect_passthrough
    hits.extend(detect_scatter_gather(graph, ScatterConfig(**settings.detectors["scatter_gather"], fx=fx)))
    hits.extend(detect_cycles(graph, CycleConfig(**settings.detectors["cycle"], fx=fx)))
    hits.extend(detect_passthrough(graph, PassConfig(**settings.detectors["pass_through"], fx=fx)))
    return hits


def _recall(attempts: list[Attempt], clusters: list[EvidenceCluster]) -> tuple[float, float]:
    if not attempts:
        return 0.0, 0.0
    evidence_edges = {edge for cluster in clusters for edge in cluster.edge_ids}
    accounts = {account for cluster in clusters for account in cluster.accounts}
    edge_hits = [len(evidence_edges.intersection(a.edge_ids)) / len(a.edge_ids) >= 0.5 for a in attempts if a.edge_ids]
    account_hits = [len(set(a.accounts).intersection(accounts)) / len(a.accounts) >= 0.5 for a in attempts if a.accounts]
    return (float(np.mean(edge_hits)) if edge_hits else 0.0, float(np.mean(account_hits)) if account_hits else 0.0)


def _system_recall(frame: pd.DataFrame, attempts: list[Attempt], settings: Settings, fx: FXModel,
                   banks: dict[int, str], model_path: Path, m8_model_path: Path, cutoff: float,
                   baseline_config: BaselineConfig) -> dict[str, float]:
    node_count = int(max(frame.src.max(), frame.dst.max())) + 1
    graph = build_graph(frame, node_count=node_count)
    hits = _detect_all(graph, settings, fx)
    merge_cfg = MergeConfig(**settings.detectors["merge"])
    merged = merge_hits(graph, hits, merge_cfg)
    attempt_accounts = {account for attempt in attempts for account in attempt.accounts}
    relevant = [cluster for cluster in merged if attempt_accounts.intersection(cluster.accounts)]
    features_cfg = FeatureConfig(**settings.scoring["features"])
    groups = [c.edge_ids for c in relevant]
    hit_groups = [[hits[i] for i in c.hit_ids] for c in relevant]
    if groups:
        features = features_for_groups(graph, groups, hit_groups, banks, features_cfg, fx)
        m8_model = XGBClassifier()
        m8_model.load_model(m8_model_path)
        features["model_probability"] = probabilities(m8_model, features)
        model = XGBClassifier()
        model.load_model(model_path)
        scores = probabilities(model, features)
        selected = [cluster for cluster, score in zip(relevant, scores, strict=True) if float(score) >= cutoff]
    else:
        selected = []
    baseline_result = baseline(graph, baseline_config)
    return dict(zip(("main_attempt_recall", "main_account_recall"), _recall(attempts, selected), strict=True)) | \
        dict(zip(("baseline_attempt_recall", "baseline_account_recall"), _recall(attempts, baseline_result.clusters), strict=True))


def run(settings: Settings) -> dict:
    split = TemporalSplit.model_validate_json((settings.data.processed_dir / "split.json").read_text())
    digest = frozen_hash(settings)
    frozen_path = settings.output_dir / "m9" / "frozen.yaml"
    frozen = json.loads(frozen_path.read_text())
    cutoff = float(frozen["cutoff"])
    model_path = settings.output_dir / "m9" / f"model_{frozen['model']['n_estimators']}_{frozen['model']['max_depth']}.json"
    if not model_path.exists():
        model_path = settings.output_dir / "m8" / "cluster_model.json"
    m8_model_path = settings.output_dir / "m8" / "cluster_model.json"
    baseline_config = BaselineConfig(**json.loads((settings.output_dir / "m3" / "baseline.json").read_text())["selected"])
    # FX is fitted from the prefix ending at validation start; no validation labels or test rows fit it.
    train_frame = pq.read_table(settings.data.processed_dir / "transactions.parquet",
        filters=[("ts", "<", split.train_end)]).to_pandas()
    fx = fit_fx(build_graph(train_frame), split.train_end)
    del train_frame
    frame = pq.read_table(settings.data.processed_dir / "transactions.parquet",
        filters=[("ts", ">=", split.train_end), ("ts", "<", split.validation_end)]).to_pandas()
    attempts = [Attempt.model_validate({k: row[k] for k in Attempt.model_fields})
        for row in pq.read_table(settings.data.processed_dir / "attempts.parquet").to_pylist()
        if split.train_end <= row["t_start"] < split.validation_end]
    accounts_path = settings.data.processed_dir / "accounts.parquet"
    account_table = pq.read_table(accounts_path).to_pandas()
    banks = dict(zip(account_table.account_id.astype(int), account_table.bank.astype(str), strict=True))
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows: list[dict] = []
    unperturbed_result: dict[str, float] | None = None
    for kind in ("amount_split", "delays", "decoys"):
        for level, strength in enumerate(strengths):
            changed, changed_attempts = _perturb(frame, attempts, float(strength), kind, settings.seed + level)
            validate_perturbation(frame, changed, float(strength), kind, attempts)
            if float(strength) == 0 and unperturbed_result is not None:
                values = unperturbed_result.copy()
            else:
                values = _system_recall(changed, changed_attempts, settings, fx, banks, model_path,
                                        m8_model_path, cutoff, baseline_config)
                if float(strength) == 0:
                    unperturbed_result = values.copy()
            rows.append({"kind": kind, "strength": float(strength), **values,
                "input_edges": len(frame), "perturbed_edges": len(changed), "attempts": len(attempts),
                "seed": settings.seed + level, "frozen_hash": digest})
    if frozen_hash(settings) != digest:
        raise RuntimeError("frozen config or model artifacts changed during evasion evaluation")
    output = settings.output_dir / "m10"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(settings.output_dir / "evasion.csv", index=False)
    (settings.output_dir / "evasion.json").write_text(json.dumps({"command": "aml evasion", "period": "validation",
        "validation_only": True, "frozen_hash": digest, "baseline": baseline_config.model_dump(),
        "model": str(model_path), "cutoff": cutoff, "rows": rows}, indent=2) + "\n")
    return {"command": "aml evasion", "validation_only": True, "frozen_hash": digest,
        "rows": len(rows), "complete": False,
        "scope": "amount splitting, delays, and benign decoys across configured strengths; combined perturbation not run",
        "csv": str(settings.output_dir / "evasion.csv"), "json": str(settings.output_dir / "evasion.json")}
