# AML Graph backend

Backend implementation of **Graph-Based Detection of Layered Money Laundering
Networks**, Build Beyond'26 PS 2.1. The specification is
[ARCHITECTURE1.md](ARCHITECTURE1.md), with user-approved clarifications in
[AGENTS.md](AGENTS.md).

M1 implements compact loading, exact pattern membership, period-local labels,
temporal splits, and the data checklist. M2 adds the validated temporal graph
and CSR indexes. No detectors or model evaluations exist
yet. M4 will produce a backend fan-detection slice and alerts JSON. Frontend work
and M13 are excluded.

## Environment

The user authorized the existing **Python 3.13.7** instead of installing the
architecture's Python 3.11. No interpreter was installed. The project `.venv`
uses that interpreter and existing system packages. M1 dependencies are present;
Numba is not installed and will be needed before the search-detector milestones.
Compatibility of future milestones is not yet established.

The environment was created and the local package installed without downloading
dependencies:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\aml prepare --config configs/default.yaml
# Subsequent checklist runs use the existing Parquet:
.\.venv\Scripts\aml prepare --config configs/default.yaml --reuse
```

`python -m aml prepare --config configs/default.yaml` is equivalent after
activation. Dependency declarations are in `pyproject.toml`; no full dependency
lock is claimed for unimplemented milestones.

## Data and outputs

Use the existing `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` at the project
root. `HI-Small_accounts.csv` is optional and initially unused. Raw data is
read-only. Do not download data or open Medium/Large variants. All derived
artifacts go under `data/processed/` or `outputs/`, both ignored by Git:

- `data/processed/transactions.parquet`: compact edges in source-row order;
  M2 builds graph arrays sorted by `(ts, tx_id)`.
- `data/processed/accounts.parquet`: reversible int32 ID to bank/account map.
- `data/processed/attempts.parquet`: exact transaction membership, temporal
  order, start-period assignment, and straddling flags.
- `data/processed/split.json`: half-open period boundaries.
- `outputs/m1/checklist.md` and `checklist.json`: measurements and provenance.
- `outputs/m1/measured_config.yaml`: train-derived initial detector settings.

The actual CSV has duplicate `Account` headers. They are mapped by column
position, preserving bank strings and leading zeros. `tx_id` is the zero-based
CSV data-row index. Pattern matching uses original decimal values before
float32 conversion; missing or ambiguous matches fail instead of guessing.
Naive dataset timestamps are encoded as UTC epoch seconds for arithmetic; the
source timezone is unspecified. Ties remain distinct through transaction IDs.

M1 performs descriptive full-timeline counts only. Parameter derivation uses
train rows and fully contained train attempts, excluding train-start straddlers
from fitting. Quantile policies are in `configs/default.yaml`; measured settings
are emitted under `outputs/`. Dampening will use activity-bin occupancy and
count regularity within the measured train span, replacing a 30-day rule.

## M1 measurements

Produced by `.\.venv\Scripts\python -m aml prepare --config configs/default.yaml`
and then `.\.venv\Scripts\aml prepare --config configs/default.yaml --reuse` after
selecting timestamp-quantile boundaries:

- 5,078,345 transactions; 515,088 bank-qualified accounts; 5,177 labeled
  positives (0.10194%). Timeline: September 1–18, 2022, on a 60-second grid.
- Train ends September 6 at 13:36; validation ends September 8 at 16:12.
  Half-open periods keep equal timestamps together. Calendar-span fractions
  were unsuitable because of a sparse September 11–18 tail.
- Train/validation/test positive edge counts: 2,297 / 1,082 / 1,798.
  Attempt counts by start period: 199 / 76 / 95; 155 attempts straddle boundaries.
- 370 attempts match exactly to 3,209 transactions. Another 1,968 labeled
  positives have no membership in this patterns file. Attempt recall covers
  only enumerated attempts; edge/account metrics retain all labels.
- 591,212 self-loops, including 11 positives. Excluding those positives would
  cap edge recall at 99.7875%. No degree cap has been applied or measured yet.
- Transactions CSV: 475,664,283 bytes; compact transactions Parquet:
  105,138,395 bytes. Original loading peak working set: 330,629,120 bytes;
  original conversion plus pattern parsing: 35.74 seconds. The checklist also
  reports its separate process peak. These are local-run observations.
- Measured host RAM: 25,534,783,488 bytes. CPU-only planning remains in place.

See [the generated checklist](outputs/m1/checklist.md) for exact boundaries,
per-typology counts, all straddlers, currency/degree distributions, and initial
parameters. Initial fan `k=2` is permissive and not a performance claim.
`STACK` is partially mapped to qualifying pass-through paths; `BIPARTITE` and
`RANDOM` have no dedicated detector. No final test evaluation has occurred.

The dataset is synthetic and generated using explicit laundering patterns;
detectors targeting those patterns may be favored. Results will not establish
real-world detection performance. Self-loop and degree-cap recall ceilings
remain unmeasured and must be reported after the relevant milestones.

## Evaluation decisions

Fit detector/model parameters and FX on train. Select the alert cutoff on
validation; perform validation ablations; use test once after freezing.
Edge/account labels use period-local transactions. Whole attempts belong to
their start period, including straddling attempts. Report cluster precision and
precision@k, not undefined cluster recall or true negatives. Use a fixed account
population and dependent-block bootstrap for account F1. Exact bootstrap block
and baseline cluster definitions must be resolved before M3.

`risk_score` is model probability times 100. The transparent score lives in
`evidence.features`; the section 14.1 alert contract is preserved. M12 uses
templates and an offline `LLMClient` stub; no provider key is required.

## M2 temporal graph

Build and validate the entire prepared graph without opening raw files:

```powershell
.\.venv\Scripts\python -m aml.graph.build
```

The command writes `outputs/m2/graph_report.json`. The measured full-data run
contained 515,088 nodes and 5,078,345 edges, including all 5,177 positive edges
and 591,212 self-loops. Build plus validation took 3.99 seconds, with a process
peak working set of 1,265,618,944 bytes. Persistent NumPy arrays occupy
363,725,574 bytes; this excludes transient construction buffers, Python object
overhead, and vocabulary strings. Timing and memory are local-run observations.

`aml.graph.build.load_graph(transactions_path, accounts_path)` returns a
Pydantic-validated `TemporalGraph`. All edge arrays are sorted by `(ts, tx_id)`.
`graph.out_idx[node]` and `graph.in_idx[node]` expose CSR slices sorted in the
same total order. Parallel edges and self-loops remain distinct; isolated nodes
from the account map remain addressable. Arrays are contiguous and read-only.

CSR edge IDs are **positions in the sorted arrays**, not original transaction
IDs. Use `graph.positions_for_tx_ids(ids)` to resolve pattern membership and
`graph.tx_id[positions]` to recover evidence IDs. Currency columns share one
alphabetically sorted int32 vocabulary (`currency_names`); payment formats
have a separate vocabulary (`format_names`). No FX conversion is performed yet.

Graph construction validates both complete CSR permutations, node ownership,
temporal order, transaction lookup, types, labels, amounts, and self-loop flags.
The graph is rebuilt in memory from Parquet; only its audit report is persisted.

## Dataset citation and license

Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Beni Egressy, Andreea Anghel,
and Kubilay Atasu. *Realistic Synthetic Financial Transactions for Anti-Money
Laundering Models*. NeurIPS 2023, Datasets and Benchmarks Track.
[Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html).

IBM's [dataset README](https://github.com/IBM/AML-Data/blob/main/README.md)
identifies the data license as **Community Data License Agreement – Sharing 1.0**
and links the [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
The repository's Apache-2.0 license is distinct from the data license.
[Official CDLA-Sharing-1.0 text](https://cdla.dev/sharing-1-0/).
These primary sources were checked during M1; no dataset or reference code was
downloaded. Raw and derived datasets are not committed or redistributed here.

## TO VERIFY: remaining references and licenses

The following are supplied by the request/specification, **not independently
verified citations or license findings**. Verify original sources before citing
them as established facts. No reference code has been copied.

- IBM Graph Feature Preprocessor: verify source, license, and the claim that
  pattern features plus gradient boosting outperform GNNs on minority-class F1.
- Egressy et al., *Provably Powerful GNNs for Directed Multigraphs*, AAAI 2024:
  verify citation and IBM Multi-GNN repository/license before crediting.
- BlazingAML, claimed arXiv ID `2604.12241`: verify existence, authorship,
  stage-based matching claims, and any repository license.
- Paranjape, Benson, Leskovec, *Motifs in Temporal Networks*: verify citation
  and the temporal-motif definition being credited.
- Check licenses for the selected runtime dependencies before distribution.

## Milestone status

M0: committed scaffold. M1: preparation and checklist. M2: temporal graph and
CSR indexes, with 34 passing tests across M1/M2. M3 has not started.
Planning assumes CPU-only execution and one or two people for one or two days;
the team/time values remain placeholders. Host RAM and loading peak memory are
measured in the checklist. Stop and report after each milestone.
