"""Build a read-only graph from compact normalized Parquet, never raw CSV.

CSR edge_ids are positions in sorted graph arrays, NOT original tx_id values.
Resolve pattern membership using positions_for_tx_ids; export evidence via tx_id.
"""

import argparse
import json
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArrayModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")


class CSRIndex(ArrayModel):
    indptr: np.ndarray
    edge_ids: np.ndarray

    @model_validator(mode="after")
    def validate_index(self) -> "CSRIndex":
        for array in (self.indptr, self.edge_ids):
            if array.ndim != 1 or array.dtype != np.int64 or not array.flags.c_contiguous:
                raise ValueError("CSR arrays must be contiguous int64 vectors")
        if not len(self.indptr) or self.indptr[0] != 0 or self.indptr[-1] != len(self.edge_ids):
            raise ValueError("invalid CSR offsets")
        if np.any(np.diff(self.indptr) < 0):
            raise ValueError("CSR offsets must be nondecreasing")
        if len(self.edge_ids) and (self.edge_ids.min() < 0 or self.edge_ids.max() >= len(self.edge_ids)
                                  or not np.all(np.bincount(self.edge_ids, minlength=len(self.edge_ids)) == 1)):
            raise ValueError("CSR must contain each edge position exactly once")
        self.indptr.flags.writeable = False
        self.edge_ids.flags.writeable = False
        return self

    def __getitem__(self, node: int) -> np.ndarray:
        if not 0 <= node < len(self.indptr) - 1:
            raise IndexError("node outside graph")
        return self.edge_ids[self.indptr[node]:self.indptr[node + 1]]


class TemporalGraph(ArrayModel):
    node_count: int = Field(ge=0)
    tx_id: np.ndarray
    ts: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    amt_out: np.ndarray
    amt_in: np.ndarray
    cur_out: np.ndarray
    cur_in: np.ndarray
    fmt: np.ndarray
    label: np.ndarray
    is_self_loop: np.ndarray
    currency_names: tuple[str, ...]
    format_names: tuple[str, ...]
    out_idx: CSRIndex
    in_idx: CSRIndex
    tx_id_order: np.ndarray

    DTYPES: ClassVar[dict[str, str]] = {
        "tx_id": "int64", "ts": "int64", "src": "int32", "dst": "int32",
        "amt_out": "float32", "amt_in": "float32", "cur_out": "int32",
        "cur_in": "int32", "fmt": "int32", "label": "int8", "is_self_loop": "bool",
    }

    @classmethod
    def edge_array_names(cls) -> tuple[str, ...]:
        return tuple(cls.DTYPES)

    @property
    def edge_count(self) -> int:
        return len(self.tx_id)

    @property
    def array_bytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in self.DTYPES) + self.tx_id_order.nbytes + sum(
            index.indptr.nbytes + index.edge_ids.nbytes for index in (self.out_idx, self.in_idx))

    @model_validator(mode="after")
    def validate_graph(self) -> "TemporalGraph":
        for name, dtype in self.DTYPES.items():
            array = getattr(self, name)
            if array.ndim != 1 or len(array) != self.edge_count or array.dtype != np.dtype(dtype) or not array.flags.c_contiguous:
                raise ValueError(f"invalid graph array {name}")
        if len(self.tx_id_order) != self.edge_count or self.tx_id_order.dtype != np.int64 or self.tx_id_order.ndim != 1:
            raise ValueError("invalid transaction lookup index")
        if self.edge_count:
            if self.tx_id_order.min() < 0 or self.tx_id_order.max() >= self.edge_count:
                raise ValueError("transaction lookup index out of bounds")
            sorted_ids = self.tx_id[self.tx_id_order]
            if sorted_ids[0] < 0 or np.any(sorted_ids[1:] <= sorted_ids[:-1]):
                raise ValueError("tx_id must be nonnegative and unique, with a sorted lookup")
            if np.any(self.ts[1:] < self.ts[:-1]) or np.any(
                (self.ts[1:] == self.ts[:-1]) & (self.tx_id[1:] <= self.tx_id[:-1])):
                raise ValueError("graph edges must be ordered by (ts, tx_id)")
        for name in ("src", "dst"):
            values = getattr(self, name)
            if len(values) and (values.min() < 0 or values.max() >= self.node_count):
                raise ValueError("account index outside graph")
        for name in ("amt_out", "amt_in"):
            if not np.isfinite(getattr(self, name)).all() or np.any(getattr(self, name) < 0):
                raise ValueError("invalid amount")
        if not np.isin(self.label, [0, 1]).all() or not np.array_equal(self.is_self_loop, self.src == self.dst):
            raise ValueError("invalid labels or self-loop flags")
        for name, vocabulary in (("cur_out", self.currency_names), ("cur_in", self.currency_names), ("fmt", self.format_names)):
            values = getattr(self, name)
            if len(set(vocabulary)) != len(vocabulary) or any(not value for value in vocabulary):
                raise ValueError("invalid categorical vocabulary")
            if len(values) and (values.min() < 0 or values.max() >= len(vocabulary)):
                raise ValueError("categorical code outside vocabulary")
        for nodes, index in ((self.src, self.out_idx), (self.dst, self.in_idx)):
            if len(index.indptr) != self.node_count + 1 or len(index.edge_ids) != self.edge_count:
                raise ValueError("CSR dimensions do not match graph")
            if not np.array_equal(np.diff(index.indptr), np.bincount(nodes, minlength=self.node_count)):
                raise ValueError("CSR degrees disagree with graph")
            grouped = nodes[index.edge_ids]
            if np.any(grouped[1:] < grouped[:-1]) or np.any(
                (grouped[1:] == grouped[:-1]) & (index.edge_ids[1:] <= index.edge_ids[:-1])):
                raise ValueError("CSR ownership or temporal ordering is invalid")
        for name in self.DTYPES:
            getattr(self, name).flags.writeable = False
        self.tx_id_order.flags.writeable = False
        return self

    def positions_for_tx_ids(self, tx_ids: np.ndarray) -> np.ndarray:
        requested = np.asarray(tx_ids)
        if requested.ndim != 1 or requested.dtype.kind not in "iu":
            raise ValueError("transaction IDs must be an integer vector")
        ranks = np.searchsorted(self.tx_id, requested, sorter=self.tx_id_order)
        if np.any(ranks >= self.edge_count):
            raise KeyError("unknown transaction ID")
        positions = self.tx_id_order[ranks]
        if not np.array_equal(self.tx_id[positions], requested):
            raise KeyError("unknown transaction ID")
        return positions


def _csr(nodes: np.ndarray, node_count: int) -> CSRIndex:
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(np.bincount(nodes, minlength=node_count), out=offsets[1:])
    # Stable grouping retains global time/tx_id order inside each adjacency row.
    positions = np.argsort(nodes, kind="stable").astype(np.int64, copy=False)
    return CSRIndex(indptr=offsets, edge_ids=positions)


def build_graph(frame: pd.DataFrame, *, node_count: int | None = None) -> TemporalGraph:
    missing = set(TemporalGraph.DTYPES) - set(frame.columns)
    if missing or frame[list(TemporalGraph.DTYPES)].isna().any(axis=None):
        raise ValueError(f"missing graph fields or null values: {sorted(missing)}")
    arrays: dict[str, np.ndarray] = {}
    for name in ("tx_id", "ts", "src", "dst", "label"):
        column = frame[name]
        dtype = np.dtype(TemporalGraph.DTYPES[name])
        if len(column) and (column.dtype.kind not in "iu" or column.min() < np.iinfo(dtype).min or column.max() > np.iinfo(dtype).max):
            raise ValueError(f"invalid integer field {name}")
        arrays[name] = column.to_numpy(dtype=dtype, copy=True)
    maximum_node = max(int(arrays["src"].max()), int(arrays["dst"].max())) if len(frame) else -1
    count = maximum_node + 1 if node_count is None else node_count
    if count < maximum_node + 1 or count < 0 or (len(frame) and min(arrays["src"].min(), arrays["dst"].min()) < 0):
        raise ValueError("invalid node_count or account index")
    order = np.lexsort((arrays["tx_id"], arrays["ts"]))
    for name in ("amt_out", "amt_in"):
        values = frame[name].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > np.finfo(np.float32).max):
            raise ValueError("invalid amount")
        arrays[name] = values.astype(np.float32)
    if len(frame) and frame.is_self_loop.dtype != bool:
        raise ValueError("self-loop flags must be boolean")
    arrays["is_self_loop"] = frame.is_self_loop.to_numpy(dtype=bool, copy=True)
    for name in ("cur_out", "cur_in", "fmt"):
        if any(not isinstance(value, str) or not value for value in frame[name].unique()):
            raise ValueError("categories must be nonempty strings")
    currencies = tuple(sorted(set(frame.cur_out) | set(frame.cur_in)))
    formats = tuple(sorted(set(frame.fmt)))
    for name, vocabulary in (("cur_out", currencies), ("cur_in", currencies), ("fmt", formats)):
        arrays[name] = pd.Categorical(frame[name], categories=vocabulary).codes.astype(np.int32)
    arrays = {name: np.ascontiguousarray(array[order]) for name, array in arrays.items()}
    return TemporalGraph(**arrays, node_count=count, currency_names=currencies, format_names=formats,
                         out_idx=_csr(arrays["src"], count), in_idx=_csr(arrays["dst"], count),
                         tx_id_order=np.argsort(arrays["tx_id"]).astype(np.int64, copy=False))


def load_graph(transactions: Path, accounts: Path) -> TemporalGraph:
    """Read normalized Parquet; retain account IDs for isolated nodes in subsets."""
    ids = pq.read_table(accounts, columns=["account_id"])["account_id"].to_numpy()
    if not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("account map must contain contiguous unique IDs starting at zero")
    return build_graph(pq.read_table(transactions).to_pandas(), node_count=len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate M2 graph from prepared Parquet")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/m2/graph_report.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    graph = load_graph(args.processed_dir / "transactions.parquet", args.processed_dir / "accounts.parquet")
    elapsed = time.perf_counter() - start
    memory = psutil.Process().memory_info()
    report = {"nodes": graph.node_count, "edges": graph.edge_count, "positive_edges": int(graph.label.sum()),
              "self_loops": int(graph.is_self_loop.sum()), "array_bytes": graph.array_bytes,
              "currency_count": len(graph.currency_names), "format_count": len(graph.format_names),
              "build_and_validation_seconds": elapsed, "process_peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
              "peak_method": "Windows lifetime peak working set when available; otherwise current RSS",
              "validation": "all edges ordered, both CSR indexes complete and time-ordered; labels/loops/lookup validated",
              "edge_id_convention": "CSR IDs are graph positions; tx_id retains the original CSV row identity",
              "command": f"python -m aml.graph.build --processed-dir {args.processed_dir} --output {args.output}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
