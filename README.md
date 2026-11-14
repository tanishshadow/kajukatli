# AML Graph backend

Backend implementation of **Graph-Based Detection of Layered Money Laundering
Networks**, Build Beyond'26 PS 2.1. The specification is
[ARCHITECTURE1.md](ARCHITECTURE1.md), with user-approved clarifications in
[AGENTS.md](AGENTS.md).

M0 provides scaffolding only. No data has been loaded, no detectors or CLI
commands exist yet, and no real-data results have been produced. M4 will produce
a backend fan-detection slice and alerts JSON. Frontend work and M13 are excluded.

## Environment

Python **3.11** is required. The current environment check found Python 3.13.7
only through the Windows Python launcher; a 3.11 interpreter is needed before
implementation validation. Dependencies are declared in `pyproject.toml`;
reproducible version locking follows installation and compatibility validation.

Once Python 3.11 is available:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest -q
```

No functional tests exist yet. `pytest -q` in M0 therefore collects no tests;
this is not a passing test suite. The M0 runner reported a nonzero exit status.

## Data and outputs

Use the existing `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` at the project
root. `HI-Small_accounts.csv` is optional and initially unused. Raw data is
read-only. Do not download data or open Medium/Large variants. All derived
artifacts go under `data/processed/` or `outputs/`, both ignored by Git.

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

## TO VERIFY: dataset citation, licenses and sources to credit

The following are supplied by the request/specification, **not independently
verified citations or license findings**. Verify original sources before citing
them as established facts. No reference code has been copied.

- Dataset: IBM AMLworld, Kaggle identifier
  `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`.
- Supplied paper: Altman et al., *Realistic Synthetic Financial Transactions
  for AML Models*, NeurIPS 2023 Datasets and Benchmarks. Verify bibliographic
  details and dataset association.
- Supplied dataset license: Community Data License Agreement – Sharing 1.0
  (CDLA-Sharing 1.0). Confirm the original dataset license and derived-data
  obligations.
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

M0: scaffold and instructions. M1 has not started. Planning assumes a CPU-only
16 GB laptop and one or two people for one or two days; these are placeholders,
not measured resource or runtime claims. Stop and report after each milestone.
