# Graph-Based Detection of Layered Money Laundering Networks
### Final architecture and build spec (hackathon edition, v2)

Problem statement: Build Beyond'26, PS 2.1.
Items marked **[VERIFY]** are assumptions to check against the real data in the first hour. Do not build on them until checked.

---

## 1. Goal and pitch

Rule-based AML checks one transaction at a time. Laundering rings only show up in the *connections* between accounts. We model transactions as a temporal, directed, weighted multigraph, detect the three required typologies, rank suspicious clusters by risk, label each with its typology, explain the money trail in plain language, and prove it with precision / recall / F1 against a tuned threshold baseline.

**Pitch line:** "We flag the ring, not the transaction."

Deliverables (1:1 with the problem statement):

1. Temporal, directed, weighted multigraph.
2. Detect fan-in / fan-out dispersion, cyclic round-tripping, rapid pass-through (mule) chains.
3. Rank clusters by risk; label each alert with its laundering typology.
4. Plain-language summary of the money trail.
5. Precision, recall, F1 vs a threshold-based baseline on a labeled dataset.

## 2. Design principles

- **Detectors decide, the LLM only narrates.** Flags and typology labels come from deterministic, testable code.
- **Every claim is traceable.** Each alert carries an evidence list (accounts, edges, amounts, timestamps). Narrative numbers are rendered from the evidence by code, never typed by the LLM.
- **Time-respecting everywhere.** A cycle or chain counts only if each hop happens after the previous one.
- **Patterns + learned classifier, not hand-tuned points.** Pattern counts feed a gradient-boosted model. A transparent score is kept for explanation. (Published IBM work reports pattern features plus gradient boosting beating standard GNNs on minority-class F1.)
- **Evaluate honestly.** Temporal split, tuned baseline, three evaluation units, confidence intervals, test set touched once.
- **Vertical slice first.** A working end-to-end demo exists before every detector is finished.

### Non-goals
GNN as a main model, real-time serving, multi-bank privacy, case management. A GNN (for example the open-source IBM Multi-GNN code) may be added later as an optional comparison.

### Reference inspirations
- Altman et al., *Realistic Synthetic Financial Transactions for AML Models* (NeurIPS 2023): dataset and benchmark.
- IBM *Graph Feature Preprocessor*: pattern counts in a time window plus gradient boosting.
- Egressy et al., *Provably Powerful GNNs for Directed Multigraphs* (AAAI 2024): optional GNN comparison.
- BlazingAML (arXiv 2604.12241): fuzzy, stage-based pattern definitions.
- Paranjape, Benson, Leskovec, *Motifs in Temporal Networks*: delta-temporal motif definition.

Write our own implementation. Credit all sources in the README and check their licenses.

---

## 3. Tech stack

- Python 3.11
- `numpy`, `pandas`; **`numba`** for the cycle and chain search (pure Python DFS will not finish)
- `networkx` only for small per-cluster subgraphs
- `xgboost` or `lightgbm` for the learned scorer
- `scikit-learn` for metrics
- `pydantic`, `pyyaml`, `typer`
- `pytest`
- `streamlit` + `pyvis` (or Cytoscape.js in a custom component) for the demo
- `LLMClient` interface with one provider and an offline stub

## 4. Repository layout

```
aml-graph/
  ARCHITECTURE.md  AGENTS.md  README.md  pyproject.toml
  configs/default.yaml          # all thresholds and weights
  data/                         # gitignored
  src/aml/
    data/      loader.py split.py labels.py patterns_parser.py
    graph/     build.py                      # TemporalGraph + CSR indexes
    detect/    fan.py scatter_gather.py cycles.py passthrough.py common.py
    cluster/   merge.py
    score/     features.py model.py risk.py  # features, XGBoost, transparent score
    typology/  label.py
    baseline/  threshold.py
    eval/      metrics.py tune.py bootstrap.py report.py evasion.py
    narrative/ evidence.py template.py llm.py render.py
    api/       schemas.py
    cli.py
  app/streamlit_app.py
  tests/planted/  test_*.py
```

---

## 5. Data

### 5.1 Dataset
IBM synthetic AML data (AMLworld), **HI-Small** for development and the headline results.
Expected files **[VERIFY]**: transactions CSV with a laundering flag; a **patterns file** (text format with begin/end blocks per laundering attempt and typology, not a CSV) listing member transactions.
Expected columns **[VERIFY]**: Timestamp, From Bank, From Account, To Bank, To Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering.

### 5.2 First-hour checklist (M1, before any other work)
1. Time span of the dataset and timestamp resolution (minutes?). **Set all windows and split sizes from this.**
2. Count of positives per split, and per typology from the patterns file.
3. Typology names in the patterns file and their mapping to our detectors (table in section 10).
4. Degree distribution of labeled fans vs normal accounts (to set `k`).
5. Currency mix and how many laundering edges are cross-currency.
6. Number of self-loops and how many are labeled laundering.

### 5.3 Normalized schema
One row per transaction: `tx_id`, `ts` (epoch seconds), `src`, `dst` (factorized ints, with a map back to bank+account), `amt_out`, `amt_in`, `cur_out`, `cur_in`, `fmt`, `label`.

- **Self-loops: flag, do not drop.** Report the recall ceiling if any are labeled.
- Do not collapse parallel edges (multigraph).
- **Tie-breaking:** equal timestamps are ordered by `tx_id`. Time-order tests use this total order, so ties cannot create impossible cycles or miss real ones.

### 5.4 Temporal split
Sort by `ts`. Default 60/20/20 train / validation / test, **adjusted after the checklist** if the span is short.

- **Detectors run on the full timeline.** Only evaluation is restricted to the period.
- **An attempt belongs to the period in which it starts.** Attempts that straddle a boundary are scored in their start period only and listed in the report.
- All tuning uses train + validation. The test period is used once.

### 5.5 Development subset
Slice by time window or take account neighborhoods around known attempts plus sampled normal accounts. Never random-sample transactions (it destroys the patterns).

---

## 6. Graph model

`TemporalGraph` holds edge arrays sorted by (`ts`, `tx_id`): `src, dst, ts, amt_out, amt_in, cur_out, cur_in, label, tx_id`, plus CSR indexes `out_idx[node]` and `in_idx[node]` (edge ids sorted by time).

Detectors are pure functions of (`TemporalGraph`, config) and return `Hit` objects.

```python
class Hit(BaseModel):
    typology: Literal["fan_out","fan_in","scatter_gather","gather_scatter","cycle","pass_through"]
    accounts: list[int]
    edge_ids: list[int]
    t_start: int
    t_end: int
    total_value: float
    meta: dict
```

---

## 7. Detectors

All parameters come from `configs/default.yaml` and are set from the data checklist, then tuned on train.

### 7.1 Fan-out / fan-in
Two-pointer sliding window of width `W` over each node's time-sorted out-edges (in-edges for fan-in). Hit when distinct counterparties in the window `>= k`. Merge overlapping windows per hub. O(E log E).

### 7.2 Scatter-gather / gather-scatter (direct, not built on fan hits)
Detect 2-hop structure directly: source `S`, sink `T`, and at least `m` distinct middle accounts `M` with `S->M` at time `t1` and `M->T` at `t2` such that `t1 <= t2 <= t1 + W2`, and the amounts on each `M` path are consistent. This avoids missing rings whose fan size is below `k`. Gather-scatter: a node with fan-in followed by fan-out within `W2`.

### 7.3 Time-respecting cycles (the main performance risk)
Definition: simple directed cycle `v0 -> ... -> v0`, edges ordered by (`ts`, `tx_id`), span `<= T_cycle`, length `3 <= L <= L_max`.

**Amount rule (two-sided, applied during the search, not only at closing):** `rho_lo <= amt_next / amt_prev <= 1 + eps`. This prunes far more than depth caps.

**Implementation:**
1. Prefilter: only start from edges whose endpoints lie in a non-trivial strongly connected component of the graph restricted to a time window.
2. Each cycle is found once, anchored at its earliest edge.
3. `numba`-compiled DFS with a per-start expansion budget.
4. `deg_cap` on in-window out-degree. **Log every skipped account and report how many labeled positives it excludes.** If that number is large, raise the cap or special-case hubs.
5. Canonicalize and dedupe.

**Cross-currency hops:** use implied FX normalization (section 7.5), so the amount rule still applies.

### 7.4 Rapid pass-through chains
- **Link:** inflow `a` into `v` and outflow `b` out of `v` with `0 <= ts(b)-ts(a) <= D` and `rho_lo <= amt_out(b)/amt_in(a) <= 1 + eps` (inflow's *received* amount vs outflow's *paid* amount, in a common currency).
- Per inflow keep the best `top_m` matches (closest in time and amount).
- **Chains:** links form a time-ordered DAG. Emit **maximal paths only** with at least `m` links, via DP. Cap the number of paths per component and log when it triggers.
- Interior accounts are recorded as likely mules in `meta`.

### 7.5 Currency normalization (v1)
Estimate implied FX rates per currency pair from the data (median of received/paid on cross-currency transactions) and convert to a base currency. All amount tests use the converted values.

### 7.6 Fuzzy patterns (from BlazingAML's idea)
Define patterns as stages (placement, layering, integration) with a window between stages, and allow any order of operations *inside* a stage. Use this for scatter-gather and for the evasion test: variable number of middle accounts and varying delays must still match.

---

## 8. Clustering

- Evidence graph: accounts from all hits, linked when they co-occur in a hit.
- **Merge rule:** two hits merge only if they share an account **and** their time spans overlap or are within `W_merge`, **and** the shared account is not a very-high-degree hub (excluded as a link point, threshold from config).
- Components larger than `N_max` accounts are not merged; keep their hits as separate clusters.
- Each cluster stores accounts, edge ids, contributing hits, time span, total value.

---

## 9. Scoring and ranking

### 9.1 Features (per account and per cluster)
Pattern counts per typology (binned by size, like the Graph Feature Preprocessor), number of distinct typologies, accounts, edges, total value, value per hour, max fan degree, cycle length, chain length, cross-bank and cross-currency hop counts, dwell-time statistics, hub legitimacy features (activity span, regularity), plus simple degree/volume statistics.

### 9.2 Learned scorer (in v1)
Gradient-boosted classifier (XGBoost or LightGBM) on pattern-count and statistics features, trained on the train period with a fixed seed, with class weighting for imbalance. Account-level and cluster-level versions share the feature code.

### 9.3 Transparent score (kept for explanation)
A 0-100 weighted score from typology strength, shown next to the model output, plus the top contributing features per alert (SHAP or feature contributions) so an analyst sees *why*.

### 9.4 Dampening
Reduce risk for hubs that look like legitimate merchants or payroll (long steady activity). The long-span rule is only used if the dataset span supports it (checklist item 1).

### 9.5 Ranking
Sort clusters by model risk descending. Alert cutoff chosen on validation.

---

## 10. Typology labeling

Labels come directly from the detectors that fired.

| Hit typology | Alert label |
|---|---|
| fan_out | Fan-out dispersion |
| fan_in | Fan-in aggregation |
| scatter_gather / gather_scatter | Scatter-gather / gather-scatter layering |
| cycle | Cyclic round-tripping |
| pass_through | Rapid pass-through (mule chain) |

Primary label = highest contribution to the score; others listed as secondary.

**Dataset mapping table (fill in during M1):** the benchmark's own pattern names (for example fan-out, fan-in, scatter-gather, gather-scatter, cycle, stack, bipartite, random, and any degree-style patterns **[VERIFY]**) map to our detectors. Where we have no detector, per-typology recall is reported as not covered, honestly.

---

## 11. Evaluation

### 11.1 Units (apply identically to every system)
1. **Edge level (comparable to published benchmarks):** transaction predicted positive if it belongs to an alerted cluster; ground truth is the dataset label.
2. **Account level (headline):** account is positive if it touches at least one laundering transaction in the evaluated period; predicted positive if in an alerted cluster.
3. **Cluster level:** a cluster is a true positive if at least `tau` (default 50%) of its edges are labeled laundering. Report precision@k.
4. **Attempt level (per-typology recall):** an attempt is detected if at least `tau_a` of its accounts fall in an alerted cluster, using the patterns file.

### 11.2 Metrics
Precision, recall, **F1 on the laundering class**, PR-AUC (`average_precision_score`), confusion counts, alert volume, precision@k, per-typology recall. No ROC-AUC (extreme imbalance).

### 11.3 Uncertainty
Report **bootstrap confidence intervals over attempts** for the headline F1 and recall, and positive counts per split. A single number from a short test period is not enough.

### 11.4 Tuning protocol
- Tune detector parameters, model hyperparameters and cutoff on **train**, check on **validation**, freeze.
- **Ablations run on validation, not test.**
- Test is run once, at the end. Add an assertion in tuning code that blocks any use of the test period.
- Fixed seeds; log every trial to CSV.

### 11.5 Systems compared
1. Threshold baseline
2. Detectors only (all hits flagged)
3. Full system (detectors + learned scorer + clusters + cutoff)
4. *(optional)* Full system + GNN feature

Ablations: no time-respecting constraint, no dampening, no amount rule, per-detector contribution, patterns-only vs model.

### 11.6 Evasion test (key slide)
`evasion.py` takes planted or labeled rings and perturbs them: split amounts across more accounts, add random delays, add decoy accounts and benign-looking edges. Re-run both systems. Show that the graph pipeline keeps recall while the threshold baseline collapses. Report recall vs perturbation strength as a curve.

### 11.7 Known limits (say them first on the slide)
The data is synthetic and generated with explicit laundering patterns, so detectors that target those patterns are favored. Real data is messier. Also state the self-loop and `deg_cap` recall ceilings.

---

## 12. Threshold baseline

A fair, tuned rules engine.

- Transaction rule: flag if `amount >= A`.
- Account rules over rolling window `W_b`: outflow count `>= N`, or total outflow `>= S`, or total inflow `>= S`.
- Account predicted positive if any rule fires; edge predicted positive if the transaction or its account is flagged (define once, use for all three units).
- Tune `A, N, S, W_b` by grid search on train with the same effort as the main system.

---

## 13. Narrative generation

### 13.1 Evidence bundle
`evidence.py` builds a structured bundle per alert: typology, ordered accounts (with ids `E1, E2, ...`), edges (with ids `T1, T2, ...`), amounts, currencies, timestamps, hop sequence.

### 13.2 Deterministic summary (always produced)
Per-typology templates, for example: "Between {t_start} and {t_end}, account {hub} sent funds to {k} distinct accounts totalling {amount}, which then ..." This is the fallback.

### 13.3 LLM summary (optional polish)
- The LLM receives the evidence bundle and typology guidance and must output **JSON**: a list of sentences, each with text containing placeholders like `{E1}`, `{T3.amount}`, and a list of evidence ids it relies on.
- `render.py` fills placeholders from the evidence in code. The LLM never types account ids, amounts or dates.
- Validation: every placeholder must resolve, every referenced id must exist, sentence order must match time order where it claims sequence. Any failure discards the LLM output and uses the template.
- Language rule: "shows patterns consistent with ...", never an accusation of guilt.
- Log the failure rate as a metric. Offline stub returns the template text.

---

## 14. Interfaces

### 14.1 Alert schema
```json
{
  "alert_id": "A-000123",
  "rank": 1,
  "risk_score": 87.5,
  "primary_typology": "cycle",
  "secondary_typologies": ["pass_through"],
  "accounts": ["..."],
  "edges": [{"tx_id": 1, "src": "...", "dst": "...", "ts": "...", "amount": 0.0, "currency": "..."}],
  "time_span": {"start": "...", "end": "..."},
  "total_value": 0.0,
  "top_features": [{"name": "...", "contribution": 0.0}],
  "summary": "...",
  "summary_source": "template | llm_validated",
  "evidence": {"hits": [], "features": {}}
}
```

### 14.2 CLI
```
aml prepare   # load, normalize, split, data checklist report
aml detect    # run detectors
aml cluster   # merge, score, rank
aml baseline
aml tune      # train period only
aml evaluate  # all units, CIs, ablations (validation) / final (test)
aml evasion
aml narrate
aml report
aml demo
```

### 14.3 Demo UI (this is what gets judged, so budget real time)
- Ranked alert queue (score, typology, value, accounts, time span).
- Selected alert: interactive graph with the ring highlighted, **a time scrubber that animates money moving along edges**, and the plain-language summary with its source tag.
- Side panel: top contributing features, and a confirm / dismiss button (logged for future feedback).
- Results tab: metrics table for all systems with confidence intervals, the PR curve, and the evasion curve.

---

## 15. Testing

Planted-pattern tests are the backbone. Each detector needs a positive case (exact accounts and edges), a near-miss negative (hops out of time order, fan with `k-1`, large dwell, amount drop), and an edge case (parallel edges, self-loop, timestamp ties).

Also: metrics vs hand-computed confusion matrices; split has no leakage; baseline and main system use identical units and periods; narrative validator rejects an invented amount and a wrong event order; determinism under fixed seed; a tiny end-to-end fixture (a few hundred transactions) that runs the whole pipeline in seconds.

---

## 16. Build order (vertical slice first)

| # | Milestone | Done when |
|---|---|---|
| M1 | Data checklist (section 5.2), loader, split, labels, patterns parser | checklist report written; windows and `k` set from data |
| M2 | `TemporalGraph` + indexes | tests pass |
| M3 | Evaluation harness (3 units) + threshold baseline | baseline numbers on train and validation |
| **M4** | **Thin vertical slice:** fan detector, one alert, graph view, template summary in Streamlit | demo runs end to end |
| M5 | Scatter-gather detector | planted tests pass |
| M6 | Cycle detector (numba, SCC prefilter, amount pruning) | planted tests pass; runtime documented; skipped-positives logged |
| M7 | Pass-through detector + FX normalization | planted tests pass |
| M8 | Clustering, features, learned scorer, ranking | ranked alerts JSON valid |
| M9 | Tuning on train, check on validation, ablations on validation | tuning log saved; parameters frozen |
| M10 | Evasion test | robustness curve generated |
| M11 | Final test evaluation, CIs, per-typology recall | one-command reproducible report |
| M12 | LLM narrative with JSON placeholders + validator | validator tests pass; fallback works |
| M13 | Demo polish: time scrubber, results tab, README, slides | demo and pitch rehearsed |

**If time runs short, cut in this order:** LLM narrative (keep template), learned cluster scorer (keep account-level model), implied FX, optional GNN, confirm/dismiss button. **Never cut:** baseline, temporal split, planted tests, the demo.

---

## 17. Conventions (also put in `AGENTS.md`)

- One milestone at a time; write the planted-pattern tests before the detector.
- Type hints everywhere; pydantic at module boundaries.
- No threshold or weight hard-coded; everything in `configs/default.yaml`.
- Fixed seeds; deterministic detectors and metrics.
- Never touch the test period inside tuning code; assertion enforces it.
- Do not change `eval/metrics.py` without updating its tests.
- Detectors are pure functions of (`TemporalGraph`, config).
- Log every cap and budget trigger so truncation is visible.
- Do not copy code from reference repos; cite them.

---

## 18. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cycle search too slow | numba, SCC prefilter, amount pruning, expansion budget, dev subset first |
| `deg_cap` hides hub-centered rings | log skipped accounts and excluded positives; raise cap or special-case hubs |
| Short dataset span breaks windows and split | set all windows from the M1 checklist; adjust split |
| Few test positives, noisy F1 | bootstrap CIs, report counts, attempts scored by start period |
| Low precision | time and amount constraints, learned scorer, dampening, cluster guards |
| Giant merged clusters | time-overlap merge rule, hub exclusion, `N_max` |
| Cross-currency amounts incomparable | implied FX normalization in v1 |
| Baseline looks like a straw man | same tuning effort, same units and period |
| Label leakage | temporal split, leakage test, test used once, ablations on validation |
| LLM invents facts | placeholder JSON rendered by code, validator, template fallback |
| Dataset assumptions wrong | M1 checklist before anything else |
| Synthetic-data bias | state it up front; show evasion robustness |
| Over-scoped build | vertical slice at M4; explicit cut list |

---

## 19. Open items to verify before coding
1. Exact file names, columns and time span of the IBM dataset version you download, and the patterns-file format.
2. Licenses of the dataset and any reference repos; confirm the reference papers and repos you plan to cite actually exist and say what you claim.
3. Whether an LLM provider and key are available for the demo; otherwise ship template summaries and the stub.
4. Real runtime of the cycle detector on the full small dataset, to set `deg_cap` and the expansion budget.
5. Team size and hours available, to apply the cut list.
