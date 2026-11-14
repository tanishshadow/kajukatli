# Backend implementation instructions

## Authority and scope

Read `ARCHITECTURE1.md` as the authoritative architecture. Keep its filename.
The user's backend request and clarifications recorded here override conflicting
architecture text. Implement backend only: `src/aml/`, configuration, tests, CLI,
documentation, JSON and Markdown outputs. No frontend or `aml demo`. Replace M4
with fan detection through one schema-valid alert and a template summary. Skip
M13; retain backend README work. M12 uses an offline stub behind `LLMClient` only.

## Milestone workflow

- Complete one milestone at a time, in section 16 order after M0 scaffolding.
- Write planted-pattern tests before implementing each detector or milestone.
- Run `pytest -q` before every milestone report and paste the actual summary.
- Report Built, Tests, Assumptions made, Numbers produced (with commands), and
  Needs from me before the next milestone. State failures and incomplete gates.
- The user authorized completing through M8 without pauses. Report and commit
  each milestone in order and continue without further confirmation. M4 uses
  an explicitly marked zero score placeholder; M8 replaces it with model probabilities.
- Make one small, reviewable Git commit per milestone. Never include raw data.
- Do not cut scope without telling the user; use section 16's cut order only.

## Engineering conventions (section 17)

- Type hints everywhere; Pydantic at module boundaries.
- No hard-coded thresholds or weights: use `configs/default.yaml`.
- Fixed seeds; deterministic detectors and metrics.
- Never access test-period inputs inside tuning code; enforce with assertions.
- Do not change `eval/metrics.py` without updating its tests.
- Detectors are pure functions of `(TemporalGraph, config)`.
- Log every cap and budget trigger so truncation is visible.
- Do not copy reference repository code. Verify references before citing them.
- Use the installed Python (currently 3.13), as explicitly authorized by the
  user after M0. Do not install another interpreter. NumPy/Pandas, Numba,
  XGBoost or LightGBM, scikit-learn,
  Pydantic, PyYAML, Typer, pytest. NetworkX only for small cluster subgraphs.

## Data protection and preparation

- Raw files are read-only: never modify, move, rename, or delete them.
- Use only `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` initially.
  `HI-Small_accounts.csv` is optional; inspect only if metadata is needed.
- Never open Medium/Large variants. Never download datasets, use Kaggle
  credentials, or use mirrors. Ask if duplicate HI-Small versions appear.
- All derived data belongs in `data/processed/` or `outputs/`.
- M1 reads CSV in chunks with explicit dtypes and converts to Parquet:
  int32 account IDs, int64 epoch seconds, float32 amounts, categorical currencies
  and payment formats. Report file sizes and peak loading memory.
- Preserve parallel edges; flag self-loops rather than dropping them. Sort edges
  by `(ts, tx_id)`. Never randomly sample transactions.
- M1 measures the full timeline for descriptive statistics. Learn label-informed
  windows, thresholds, and FX rates from train only. Do not assume span,
  positive rate, column names, timestamp resolution, or typology names.

## Resolved protocol and algorithm decisions

- Detectors run on the full timeline. Tuning must not access test-period inputs.
- Tune detector/model parameters on train, check on validation, and choose the
  alert cutoff on validation. Freeze before the one final test evaluation.
- Run ablations on validation only. Give the baseline the same tuning effort.
- Edge/account metrics use each transaction's timestamp and label in its period.
  Use a fixed account population per evaluated period.
- Attempt recall uses whole-attempt membership and start-period attribution;
  list straddling attempts. Cluster precision and precision@k are defined;
  do not invent cluster recall or true negatives. Recall is attempt-level.
- Bootstrap dependent blocks, not individual accounts. The user approved
  joining attempts sharing accounts into blocks, with remaining benign accounts
  grouped by connected component over a fixed period account population.
- Baseline clusters are account-connected components of flagged edges within
  each rolling baseline window; deduplicate identical edge sets. Preserve the
  edge/account predictions of the baseline rules independently of clustering.
- Gather-scatter: at least `m_in` distinct sources into a hub followed by at
  least `m_out` distinct destinations within `W2`. Matched total outflow/inflow
  ratio is in `[rho_lo, 1 + eps]`. Derive sizes from train patterns. Stages may
  overlap, but each matched incoming-to-outgoing path respects total time order.
- Implement shared FX before M6. Fit pairwise median received/paid rates on
  train only. The most frequent train currency is the base; chain connected
  pairs. For an unconnected currency, skip the amount test on that hop, mark
  `fx_missing` in hit metadata, and log counts.
- Keep all section 14.1 alert fields unchanged. `risk_score` is model
  probability times 100; transparent score goes in `evidence.features`.
- Adapt activity dampening to measured span; do not use an impossible long-span
  rule on a short dataset.
- Narratives always include deterministic templates, use evidence placeholders,
  and fall back on validation failure. Use “consistent with”, never accusations.

## Reporting and resource assumptions

- Plan provisionally for 16 GB RAM, no GPU, one or two people, and one or two
  days. These are user placeholders, not measured hardware or elapsed work.
- Report only measured numbers with their generating commands.
- State synthetic-data limitations and self-loop/degree-cap recall ceilings.
- Unverified citations, licenses, and performance claims belong in README's
  TO VERIFY list until checked against original sources.
