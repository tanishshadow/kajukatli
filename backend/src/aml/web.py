"""Small read-only HTTP surface joining the static frontend to AML artifacts."""
from __future__ import annotations

import csv
import json
import mimetypes
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aml.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_FILES = {"index.html", "approach.html", "patterns.html", "evaluation.html", "investigation.html", "site-pages.css", "site-pages.js", "investigation.css", "investigation.js", "README.md"}

_DETECTORS = {
    "fan": ("fan", "Fan in / fan out", {"fan_in", "fan_out"}),
    "scatter": ("scatter_gather", "Scatter & gather", {"scatter_gather", "gather_scatter"}),
    "cycle": ("cycle", "Time-respecting cycles", {"cycle"}),
    "pass": ("pass_through", "Rapid pass-through", {"pass_through"}),
}


def _hit_rows(path: Path):
    """Yield individual JSON array items; hit files can be hundreds of MB."""
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip().rstrip(",")
                if not line or line in {"[", "]"}:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def _investigation(endpoint: str, query: dict[str, list[str]], settings: Settings) -> dict[str, Any] | None:
    import pyarrow.dataset as ds

    if endpoint == "investigation/account":
        raw_account = query.get("id", [""])[0]
        try:
            account_id = int(raw_account)
        except (TypeError, ValueError):
            return {"error": "invalid account id", "cases": [], "total": 0}
        matches = []
        for detector, (folder, _, typologies) in _DETECTORS.items():
            for raw_index, hit in enumerate(_hit_rows(settings.output_dir / folder / "hits.json")):
                if hit.get("typology") not in typologies:
                    continue
                if account_id not in (int(value) for value in hit.get("accounts", [])):
                    continue
                matches.append({"detector": detector, "id": raw_index, "typology": hit["typology"],
                    "edge_count": len(hit.get("edge_ids", [])), "account_count": len(hit.get("accounts", [])),
                    "t_start": hit.get("t_start"), "t_end": hit.get("t_end"),
                    "total_value": hit.get("total_value"),
                    "currency": hit.get("meta", {}).get("value_currency")})
        total = len(matches)
        return {"account": account_id, "cases": matches[:100], "total": total, "truncated": total > 100}

    if endpoint == "investigation/detectors":
        return {"detectors": [{"id": key, "label": label,
            "available": (settings.output_dir / folder / "hits.json").is_file(),
            "hit_count": int((_read_json(settings.output_dir / folder / "report.json") or {}).get("hits", 0))}
            for key, (folder, label, _) in _DETECTORS.items()]}
    if endpoint == "investigation/cases":
        detector = query.get("detector", [""])[0]
        if detector not in _DETECTORS:
            return {"error": "unknown detector", "cases": [], "total": 0}
        offset = max(0, int(query.get("offset", ["0"])[0]))
        limit = min(100, max(1, int(query.get("limit", ["30"])[0])))
        folder, _, typologies = _DETECTORS[detector]
        rows = []
        total = 0
        for raw_index, hit in enumerate(_hit_rows(settings.output_dir / folder / "hits.json")):
            if hit.get("typology") not in typologies:
                continue
            if total >= offset and len(rows) < limit:
                rows.append({"id": raw_index, "typology": hit["typology"],
                    "edge_count": len(hit.get("edge_ids", [])), "account_count": len(hit.get("accounts", [])),
                    "t_start": hit.get("t_start"), "t_end": hit.get("t_end"),
                    "total_value": hit.get("total_value"), "currency": hit.get("meta", {}).get("value_currency"),
                    "meta": hit.get("meta", {})})
            total += 1
        return {"detector": detector, "cases": rows, "offset": offset, "limit": limit, "total": total,
            "has_more": offset + len(rows) < total}
    if endpoint == "investigation/case":
        detector = query.get("detector", [""])[0]
        if detector not in _DETECTORS:
            return {"error": "unknown detector"}
        case_id = int(query.get("id", ["-1"])[0])
        folder, _, _ = _DETECTORS[detector]
        hit = next((row for index, row in enumerate(_hit_rows(settings.output_dir / folder / "hits.json")) if index == case_id), None)
        if hit is None:
            return {"error": "case not found"}
        edge_ids = [int(value) for value in hit.get("edge_ids", [])]
        max_edges = 5000
        omitted = max(0, len(edge_ids) - max_edges)
        edge_ids = edge_ids[:max_edges]
        tx_path = settings.data.processed_dir / "transactions.parquet"
        account_path = settings.data.processed_dir / "accounts.parquet"
        if not tx_path.is_file() or not account_path.is_file():
            return {"error": "processed transaction/account Parquet is missing; run aml prepare first"}
        tx = ds.dataset(tx_path, format="parquet").to_table(
            columns=["tx_id", "ts", "src", "dst", "amt_out", "cur_out", "label"],
            filter=ds.field("tx_id").isin(edge_ids)).to_pydict()
        order = sorted(range(len(tx["tx_id"])), key=lambda i: (tx["ts"][i], tx["tx_id"][i]))
        accounts = [int(value) for value in hit.get("accounts", [])]
        account_rows = ds.dataset(account_path, format="parquet").to_table(
            columns=["account_id", "bank"], filter=ds.field("account_id").isin(accounts)).to_pydict()
        banks = dict(zip(account_rows["account_id"], account_rows["bank"]))
        currencies = sorted({str(tx["cur_out"][i]) for i in order})
        actual_edges = [{"id": int(tx["tx_id"][i]), "src": int(tx["src"][i]), "dst": int(tx["dst"][i]),
            "amount": float(tx["amt_out"][i]), "currency": str(tx["cur_out"][i]),
            "ts": int(tx["ts"][i]), "label": int(tx["label"][i])} for i in order]
        config_key = {"fan": "fan", "scatter": "scatter_gather", "cycle": "cycle", "pass": "pass_through"}[detector]
        cfg = settings.detectors.get(config_key, {})
        meta = hit.get("meta", {})
        conditions: list[str] = []
        if detector == "fan":
            conditions.append(f"Hub {meta.get('hub')} connects {meta.get('distinct_counterparties')} distinct counterparties; configured minimum is {cfg.get('k')} within {meta.get('window_seconds')} seconds.")
        elif detector == "scatter":
            stages = meta.get("stages", {})
            placement, integration = stages.get("placement", []), stages.get("integration", [])
            if hit.get("typology") == "gather_scatter":
                hub = meta.get("hub")
                incoming = [edge for edge in actual_edges if edge["dst"] == hub]
                outgoing = [edge for edge in actual_edges if edge["src"] == hub]
                conditions.append(f"Hub {hub} matched {len({edge['src'] for edge in incoming})} distinct incoming sources (minimum {cfg.get('m_in')}) and {len({edge['dst'] for edge in outgoing})} distinct outgoing destinations (minimum {cfg.get('m_out')}).")
            else:
                conditions.append(f"Direct paths link source {meta.get('source')} to sink {meta.get('sink')} through {len(meta.get('middle_accounts', []))} distinct middle accounts (minimum {cfg.get('m')}).")
            conditions.append(f"Detector window is {cfg.get('window_seconds')} seconds; matched placement/integration stages contain {len(placement)}/{len(integration)} transfers. Configured FX amount ratio is [{cfg.get('rho_lo')}, {1 + float(cfg.get('eps', 0))}].")
        elif detector == "cycle":
            conditions.append(f"Time-ordered cycle length {meta.get('length', len(edge_ids))} is within the configured maximum {cfg.get('max_length')}; candidate span is {hit.get('t_end', 0) - hit.get('t_start', 0)} seconds (window {cfg.get('window_seconds')}).")
            conditions.append(f"Each comparable FX-normalized hop is checked against ratio [{cfg.get('rho_lo')}, {1 + float(cfg.get('eps', 0))}]; fx_missing={bool(meta.get('fx_missing'))}.")
        else:
            conditions.append(f"Matched path contains {meta.get('links')} links (configured minimum {cfg.get('min_links')}) with {meta.get('likely_mules', [])} as intermediate accounts; maximum dwell per hop is {cfg.get('dwell_seconds')} seconds.")
            conditions.append(f"Comparable FX-normalized hop ratios are bounded by [{cfg.get('rho_lo')}, {1 + float(cfg.get('eps', 0))}]; fx_missing={bool(meta.get('fx_missing'))}.")
        if meta.get("fx_missing"):
            conditions.append("At least one FX conversion was unavailable; the detector marks the amount comparison as skipped for that hop.")
        return {"detector": detector, "case": {"id": case_id, **hit, "omitted_edges": omitted},
            "nodes": [{"id": account, "bank": banks.get(account)} for account in accounts],
            "edges": actual_edges, "conditions": conditions, "currencies": currencies}
    return None


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, (dict, list)) else None
    except (OSError, json.JSONDecodeError):
        return None


def _compact_pattern_data(settings: Settings) -> list[dict[str, Any]]:
    files = {
        "fan": ("Fan in / fan out", ["fan"]),
        "scatter": ("Scatter & gather", ["scatter_gather"]),
        "cycle": ("Time-respecting cycles", ["cycle"]),
        "pass": ("Pass-through paths", ["pass_through"]),
    }
    result = []
    for key, (label, dirs) in files.items():
        reports = [_read_json(settings.output_dir / name / "report.json") for name in dirs]
        reports = [r for r in reports if isinstance(r, dict)]
        result.append({"id": key, "label": label,
            "hits": sum(int(report.get("hits", 0)) for report in reports),
            "ran": bool(reports), "fx_missing_hits": sum(int(report.get("fx_missing_hits", 0)) for report in reports)})
    return result


def api_data(endpoint: str, settings: Settings, query: dict[str, list[str]] | None = None) -> dict[str, Any] | None:
    """Return safe, summary-only page data; never returns raw transactions/accounts."""
    if endpoint.startswith("investigation/"):
        try:
            return _investigation(endpoint, query or {}, settings)
        except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
            return {"error": f"investigation data unavailable: {exc}"}
    checklist = _read_json(settings.output_dir / "m1" / "checklist.json")
    checklist = checklist if isinstance(checklist, dict) else {}
    final = _read_json(settings.output_dir / "m11" / "final_report.json")
    final = final if isinstance(final, dict) else None
    m8 = _read_json(settings.output_dir / "m8" / "report.json")
    m8 = m8 if isinstance(m8, dict) else {}
    split = checklist.get("split", {}).get("iso_boundaries", {})
    detectors = settings.detectors
    if endpoint == "overview":
        return {"transactions": checklist.get("transactions"), "positive_transactions": checklist.get("positive_transactions"),
            "accounts": checklist.get("accounts"), "detectors": len(_compact_pattern_data(settings)),
            "ranked_clusters": m8.get("clusters"), "test_report_available": final is not None}
    if endpoint == "approach":
        return {"train_end": split.get("train", [None, None])[1],
            "validation_end": split.get("validation", [None, None])[1],
            "fan_window_seconds": detectors.get("fan", {}).get("window_seconds"),
            "cycle_window_seconds": detectors.get("cycle", {}).get("window_seconds"),
            "pass_through_dwell_seconds": detectors.get("pass_through", {}).get("dwell_seconds"),
            "cutoff": (_read_json(settings.output_dir / "m9" / "frozen.yaml") or {}).get("cutoff")}
    if endpoint == "patterns":
        return {"patterns": _compact_pattern_data(settings)}
    if endpoint == "evaluation":
        account = final.get("account", {}) if final else {}
        baseline = _read_json(settings.output_dir / "m3" / "baseline.json")
        baseline = baseline if isinstance(baseline, dict) else {}
        validation_baseline = baseline.get("metrics", {}).get("validation", {}).get("account", {})
        evasion: list[dict[str, Any]] = []
        curve_path = settings.output_dir / "evasion.csv"
        if curve_path.is_file():
            try:
                with curve_path.open(newline="", encoding="utf-8") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("kind") in {"amount_split", "delays", "decoys"}:
                            evasion.append({"kind": row["kind"], "strength": float(row["strength"]),
                                "main_recall": float(row["main_attempt_recall"]),
                                "baseline_recall": float(row["baseline_attempt_recall"])})
            except (OSError, ValueError, KeyError):
                evasion = []
        return {"available": final is not None,
            "account_f1": account.get("f1"), "account_recall": account.get("recall"),
            "account_precision": account.get("precision"),
            "baseline_validation": {"precision": validation_baseline.get("precision"),
                "recall": validation_baseline.get("recall"), "f1": validation_baseline.get("f1"),
                "period": "validation", "note": "No baseline test metrics are saved; this is validation only."},
            "attempts": final.get("attempts_by_start_period") if final else None,
            "test_transactions": final.get("test_transactions") if final else None,
            "positive_test_transactions": final.get("positive_test_transactions") if final else None,
            "attempt_recall_by_typology": final.get("attempt_recall_edge_coverage_tau_0_5", {}) if final else {},
            "evasion": evasion, "evasion_scope": "amount splitting only" if evasion else None,
            "synthetic_data": True}
    return None


def make_handler(settings: Settings, frontend_dir: Path = PROJECT_ROOT / "frontend") -> type[BaseHTTPRequestHandler]:
    class AMLHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store" if content_type == "application/json; charset=utf-8" else "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            route = parsed.path
            if route.startswith("/api/"):
                value = api_data(route.removeprefix("/api/"), settings, urllib.parse.parse_qs(parsed.query))
                if value is None:
                    self._send(404, b'{"error":"unknown endpoint"}', "application/json; charset=utf-8")
                else:
                    self._send(200, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")
                return
            name = "index.html" if route == "/" else route.lstrip("/")
            if name not in FRONTEND_FILES:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            path = (PROJECT_ROOT / "README.md") if name == "README.md" else (frontend_dir / name)
            try:
                body = path.read_bytes()
            except OSError:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            mime = "text/markdown; charset=utf-8" if name.endswith(".md") else (mimetypes.guess_type(name)[0] or "application/octet-stream") + "; charset=utf-8"
            self._send(200, body, mime)

        def log_message(self, format: str, *args: Any) -> None:
            print("frontend", self.address_string(), format % args)

    return AMLHandler


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(settings))
    print(f"AML Graph frontend: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
