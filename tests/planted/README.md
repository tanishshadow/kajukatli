# Planted fixtures

M1 includes data normalization, pattern-parser, period-local label, temporal-split,
train-only fitting, and a 303-transaction preparation fixture. Tests were written
before implementation. M2 adds exact temporal ordering, parallel-edge/self-loop
preservation, immutable compact arrays, transaction-ID lookup, isolated nodes,
invalid input validation, Parquet loading, and CSR comparison with a brute-force
oracle on a deterministic 200-edge multigraph. Later milestones add detector positive, near-miss,
parallel-edge, self-loop, and timestamp-tie fixtures, plus evaluation,
no-leakage, baseline parity, narrative validation, determinism, and a tiny
end-to-end fixture for the complete detection pipeline.
