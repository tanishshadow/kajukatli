"""Small read-only HTTP surface joining the static frontend to AML artifacts."""
from __future__ import annotations

import csv
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aml.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_FILES = {"index.html", "approach.html", "patterns.html", "evaluation.html", "site-pages.css", "site-pages.js", "README.md"}


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


def api_data(endpoint: str, settings: Settings) -> dict[str, Any] | None:
    """Return safe, summary-only page data; never returns raw transactions/accounts."""
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
            route = urlparse(self.path).path
            if route.startswith("/api/"):
                value = api_data(route.removeprefix("/api/"), settings)
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
