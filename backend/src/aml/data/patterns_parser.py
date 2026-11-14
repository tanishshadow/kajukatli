"""Strict AMLworld block parsing and lossless raw-row membership matching."""

import csv
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

Signature: TypeAlias = tuple[str, ...]


class RawAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt_id: str
    typology: str
    description: str
    rows: list[Signature]


class MatchedEdge(BaseModel):
    tx_id: int
    src: int
    dst: int
    ts: int


class Attempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt_id: str
    typology: str
    description: str
    edge_ids: list[int]
    accounts: list[int]
    t_start: int
    t_end: int


def signature(row: list[str] | tuple[str, ...]) -> Signature:
    """Canonicalize decimals before any float32 conversion; keep string IDs."""
    if len(row) != 11:
        raise ValueError(f"expected 11 transaction fields, got {len(row)}")
    fields = list(row)
    for index in (5, 7):
        amount = Decimal(fields[index])
        if not amount.is_finite() or amount < 0:
            raise ValueError("invalid pattern amount")
        fields[index] = str(amount.normalize())
    if fields[10] != "1":
        raise ValueError("pattern member must have laundering label 1")
    return tuple(fields)


def parse_patterns(path: Path) -> list[RawAttempt]:
    attempts: list[RawAttempt] = []
    active: RawAttempt | None = None
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT - "):
                if active is not None:
                    raise ValueError(f"nested attempt at line {line_number}")
                name, _, description = line.removeprefix("BEGIN LAUNDERING ATTEMPT - ").partition(":")
                active = RawAttempt(attempt_id=f"P-{len(attempts):06d}", typology=name.strip(),
                                    description=description.strip(), rows=[])
            elif line.startswith("END LAUNDERING ATTEMPT - "):
                name = line.removeprefix("END LAUNDERING ATTEMPT - ").strip()
                if active is None or name != active.typology or not active.rows:
                    raise ValueError(f"invalid/empty attempt end at line {line_number}")
                attempts.append(active)
                active = None
            else:
                if active is None:
                    raise ValueError(f"transaction outside block at line {line_number}")
                active.rows.append(signature(next(csv.reader([line]))))
    if active is not None:
        raise ValueError("unterminated laundering attempt")
    if not attempts:
        raise ValueError("patterns file has no attempts")
    return attempts


def match_attempts(patterns: list[RawAttempt], matches: dict[Signature, list[MatchedEdge]]) -> list[Attempt]:
    owners: dict[Signature, set[str]] = defaultdict(set)
    expected: Counter[Signature] = Counter()
    for pattern in patterns:
        expected.update(pattern.rows)
        for key in pattern.rows:
            owners[key].add(pattern.attempt_id)
    for key, count in expected.items():
        if len(owners[key]) != 1 or len(matches.get(key, [])) != count:
            raise ValueError("pattern match missing or ambiguous; cannot assign transaction occurrences")
    result: list[Attempt] = []
    for pattern in patterns:
        members = [edge for key in dict.fromkeys(pattern.rows) for edge in matches[key]]
        members.sort(key=lambda edge: (edge.ts, edge.tx_id))
        result.append(Attempt(attempt_id=pattern.attempt_id, typology=pattern.typology,
                              description=pattern.description,
                              edge_ids=[edge.tx_id for edge in members],
                              accounts=sorted({node for edge in members for node in (edge.src, edge.dst)}),
                              t_start=members[0].ts, t_end=members[-1].ts))
    return result
