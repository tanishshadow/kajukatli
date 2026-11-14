"""Chunked CSV conversion. tx_id is the zero-based source data-row number."""

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from aml.data.patterns_parser import MatchedEdge, RawAttempt, Signature, signature

RAW_COLUMNS = ["Timestamp", "From Bank", "Account", "To Bank", "Account",
               "Amount Received", "Receiving Currency", "Amount Paid",
               "Payment Currency", "Payment Format", "Is Laundering"]
INPUT_NAMES = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
               "received", "receiving_currency", "paid", "payment_currency", "format", "label"]
EDGE_SCHEMA = pa.schema([
    ("tx_id", pa.int64()), ("ts", pa.int64()), ("src", pa.int32()), ("dst", pa.int32()),
    ("amt_out", pa.float32()), ("amt_in", pa.float32()),
    ("cur_out", pa.dictionary(pa.int32(), pa.string())),
    ("cur_in", pa.dictionary(pa.int32(), pa.string())),
    ("fmt", pa.dictionary(pa.int32(), pa.string())), ("label", pa.int8()),
    ("is_self_loop", pa.bool_()),
])


class LoadResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    transactions_path: Path
    accounts_path: Path
    row_count: int
    account_count: int
    sampled_peak_rss_bytes: int
    process_peak_rss_bytes: int
    matches: dict[Signature, list[MatchedEdge]]


def convert_csv(path: Path, output_dir: Path, *, chunk_rows: int,
                patterns: list[RawAttempt] | None = None) -> LoadResult:
    if re.search(r"(?:HI|LI)-(?:Medium|Large)", path.name, re.IGNORECASE):
        raise ValueError("forbidden dataset variant; only HI-Small is permitted")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream))
    if header != RAW_COLUMNS:
        raise ValueError(f"unexpected CSV header: {header!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    transactions_path = output_dir / "transactions.parquet"
    accounts_path = output_dir / "accounts.parquet"
    if path.resolve() in (transactions_path.resolve(), accounts_path.resolve()):
        raise ValueError("output cannot overwrite raw data")
    account_ids: dict[tuple[str, str], int] = {}
    wanted = {key for pattern in patterns or [] for key in pattern.rows}
    matches: dict[Signature, list[MatchedEdge]] = defaultdict(list)
    process = psutil.Process()
    peak = process.memory_info().rss
    offset = 0
    with pq.ParquetWriter(transactions_path, EDGE_SCHEMA, compression="zstd") as writer:
        for chunk in pd.read_csv(path, header=0, names=INPUT_NAMES,
                                 dtype={name: "string" for name in INPUT_NAMES},
                                 chunksize=chunk_rows, keep_default_na=False):
            if (chunk == "").any(axis=None):
                raise ValueError("empty CSV field")
            if not chunk.label.isin(["0", "1"]).all():
                raise ValueError("label must be 0 or 1")
            times = pd.to_datetime(chunk.timestamp, format="%Y/%m/%d %H:%M", utc=True, errors="raise")
            # UTC is a neutral epoch encoding of naive dataset clock values, not a
            # claim that the generator's source timestamps carry a known timezone.
            ts = times.astype("int64").to_numpy() // 1_000_000_000
            try:
                paid = pd.to_numeric(chunk.paid, errors="raise").to_numpy(dtype=np.float64)
                received = pd.to_numeric(chunk.received, errors="raise").to_numpy(dtype=np.float64)
            except ValueError as error:
                raise ValueError("invalid amount in CSV") from error
            for values in (paid, received):
                if not np.isfinite(values).all() or (values < 0).any() or (values > np.finfo(np.float32).max).any():
                    raise ValueError("invalid or overflowing amount")
            src = np.empty(len(chunk), dtype=np.int32)
            dst = np.empty(len(chunk), dtype=np.int32)
            identity_rows = zip(chunk.from_bank, chunk.from_account, chunk.to_bank, chunk.to_account)
            for index, (from_bank, source, to_bank, target) in enumerate(identity_rows):
                src[index] = account_ids.setdefault((from_bank, source), len(account_ids))
                dst[index] = account_ids.setdefault((to_bank, target), len(account_ids))
            if len(account_ids) > np.iinfo(np.int32).max:
                raise ValueError("account IDs exceed int32 capacity")
            labels = chunk.label.to_numpy(dtype=np.int8)
            if wanted:
                for index in np.flatnonzero(labels):
                    key = signature(tuple(chunk.iloc[index]))
                    if key in wanted:
                        matches[key].append(MatchedEdge(tx_id=offset + int(index), src=int(src[index]),
                                                       dst=int(dst[index]), ts=int(ts[index])))
            frame = pd.DataFrame({
                "tx_id": np.arange(offset, offset + len(chunk), dtype=np.int64), "ts": ts,
                "src": src, "dst": dst, "amt_out": paid.astype(np.float32),
                "amt_in": received.astype(np.float32),
                "cur_out": pd.Categorical(chunk.payment_currency),
                "cur_in": pd.Categorical(chunk.receiving_currency), "fmt": pd.Categorical(chunk.format),
                "label": labels, "is_self_loop": src == dst,
            })
            table = pa.Table.from_pandas(frame, schema=EDGE_SCHEMA, preserve_index=False)
            writer.write_table(table)
            offset += len(chunk)
            peak = max(peak, process.memory_info().rss)
    if offset == 0:
        raise ValueError("empty transaction file")
    mapping = pa.table({"account_id": pa.array(range(len(account_ids)), type=pa.int32()),
                        "bank": [key[0] for key in account_ids], "account": [key[1] for key in account_ids]})
    pq.write_table(mapping, accounts_path, compression="zstd")
    memory = process.memory_info()
    peak = max(peak, memory.rss)
    return LoadResult(transactions_path=transactions_path, accounts_path=accounts_path,
                      row_count=offset, account_count=len(account_ids), sampled_peak_rss_bytes=peak,
                      process_peak_rss_bytes=getattr(memory, "peak_wset", peak), matches=dict(matches))
