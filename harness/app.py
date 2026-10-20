"""Local web app: run the harness and show documents that link to the code.

The server is deliberately small: an explicit file allowlist, JSON only, one
job at a time, and no client-supplied path ever reaches the filesystem.  Views
are built here from the dataclasses so the browser receives structure rather
than markdown or HTML and cannot turn model text into markup.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .citations import Citation
from .compose import Document, DocumentSet, compose_documents, render_markdown
from .config import ConfigError, load_config, load_dotenv
from .extract import Ledger, extract_claims, plan_chunks
from .facts import extract_facts
from .grounding import claim_id_key
from .injection import scan_injection
from .llm import (
    FakeLLM,
    LLMAuthError,
    LLMClient,
    LLMRequestError,
    LLMTransientError,
    make_client,
)
from .metrics import RunMetrics, build_metrics
from .models import ParseResult
from .parser import parse_text
from .verify import VerifiedLedger, verify_ledger

Client = LLMClient | FakeLLM
Pipeline = Callable[..., dict[str, Any]]

_STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
}
_API_PATHS = frozenset(
    {"/api/info", "/api/sample", "/api/status", "/api/result", "/api/run"}
)
_CSP = "default-src 'self'; base-uri 'none'; frame-ancestors 'none'"
_BODY_LIMIT = 1_500_000
_TEXT_LIMIT = 1_000_000
_DRAIN_LIMIT = 2_000_000


class StageSetter(Protocol):
    """The one callback the pipeline needs to report progress."""

    def __call__(self, stage: str, done: int = 0, total: int = 0) -> None: ...


@dataclass
class Job:
    """The single run slot; the lock makes check-and-start one operation."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _state: dict[str, object] = field(
        default_factory=lambda: {
            "state": "idle",
            "stage": "",
            "done": 0,
            "total": 0,
            "error": None,
        }
    )
    _result: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, object]:
        """Return a copy so a response cannot expose a half-updated dict."""
        with self._lock:
            return dict(self._state)

    def begin(self) -> bool:
        """Claim the slot; False means another run already owns it."""
        with self._lock:
            if self._state["state"] == "running":
                return False
            self._state = {
                "state": "running",
                "stage": "Starting",
                "done": 0,
                "total": 0,
                "error": None,
            }
            self._result = None
            return True

    def set_stage(self, stage: str, done: int = 0, total: int = 0) -> None:
        """Ignore late progress from a thread whose run has already ended."""
        with self._lock:
            if self._state["state"] != "running":
                return
            self._state.update({"stage": stage, "done": done, "total": total})

    def finish(self, result: dict[str, Any]) -> None:
        """Publish the result only after every file has been written."""
        with self._lock:
            state = dict(self._state)
            state.update({"state": "done", "stage": "Finished", "error": None})
            self._state = state
            self._result = result

    def fail(self, message: str) -> None:
        """Record a human-readable failure and keep the server alive."""
        with self._lock:
            state = dict(self._state)
            state.update({"state": "error", "stage": "Failed", "error": message})
            self._state = state
            self._result = None

    def completed(self) -> dict[str, Any] | None:
        """Return the result only when the slot is done, in one locked read."""
        with self._lock:
            if self._state["state"] != "done":
                return None
            return self._result


def _cites_view(cites: Sequence[Citation]) -> list[dict[str, int]]:
    """Keep the browser's cite shape in one place."""
    return [{"cell": cite.cell, "start": cite.start, "end": cite.end} for cite in cites]


def notebook_view(result: ParseResult, name: str | None = None) -> dict[str, Any]:
    """Every cell and raw file line, because a citation may land anywhere."""
    label = name
    if label is None:
        label = Path(result.path).name if result.path is not None else ""
    return {
        "name": label,
        "cells": [
            {
                "number": cell.number,
                "kind": cell.kind.value,
                "lines": [
                    {"rel": line.rel_no, "abs": line.abs_no, "raw": line.raw}
                    for line in cell.lines
                ],
            }
            for cell in result.cells
        ],
    }


def claims_view(verified: VerifiedLedger) -> list[dict[str, Any]]:
    """Show every claim, dropped ones included, so a reader sees rejections."""
    items = sorted(verified.items, key=lambda item: claim_id_key(item.claim.id))
    return [
        {
            "id": item.claim.id,
            "status": item.status,
            "layer": item.layer,
            "reasons": list(item.reasons),
            "judge": item.judge,
            "kind": item.claim.kind,
            "tag": item.claim.tag,
            "text": item.claim.text,
            "cites": _cites_view(item.claim.cites),
        }
        for item in items
    ]


def _doc_view(doc: Document) -> dict[str, Any]:
    """Return exactly the document contract, never a dataclass dump."""
    return {
        "audience": doc.audience,
        "title": doc.title,
        "sections": [
            {
                "heading": section.heading,
                "sentences": [
                    {
                        "text": sentence.text,
                        "ids": list(sentence.ids),
                        "cites": _cites_view(sentence.cites),
                        "source": sentence.source,
                        "flags": list(sentence.flags),
                    }
                    for sentence in section.sentences
                ],
            }
            for section in doc.sections
        ],
        "issues": list(doc.issues),
        "error": doc.error,
        "used_ids": list(doc.used_ids),
    }


def documents_view(documents: DocumentSet) -> dict[str, Any]:
    """Return the explicit three-document contract plus its run summary."""
    return {
        "manager": _doc_view(documents.manager),
        "developer": _doc_view(documents.developer),
        "agent": _doc_view(documents.agent),
        "issues": list(documents.issues),
        "summary": {
            "total": int(documents.summary.get("total", 0)),
            "verified": int(documents.summary.get("verified", 0)),
            "flagged": int(documents.summary.get("flagged", 0)),
            "dropped": int(documents.summary.get("dropped", 0)),
            "used": int(documents.summary.get("used", 0)),
        },
        "notes": list(documents.notes),
    }


def _safe_name(name: str) -> str:
    """Turn an uploaded name into one path segment that is never empty."""
    stem = name[:-3] if name.endswith(".py") else name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:40]
    return safe or "notebook"


def _write_json(path: Path, value: object) -> None:
    """Write stable, readable JSON without depending on a module's to_dict."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _save_outputs(
    result: ParseResult,
    verified: VerifiedLedger,
    documents: DocumentSet,
    metrics: RunMetrics,
    text: str,
    name: str,
    runs_dir: str | Path,
) -> Path:
    """Save the three documents and their evidence under one timestamped dir."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / f"{stamp}-{_safe_name(name)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for audience, doc in (
        ("manager", documents.manager),
        ("developer", documents.developer),
        ("agent", documents.agent),
    ):
        (run_dir / f"{audience}.md").write_text(
            render_markdown(doc, verified), encoding="utf-8"
        )
    _write_json(run_dir / "claims.json", claims_view(verified))
    _write_json(run_dir / "documents.json", documents_view(documents))
    _write_json(run_dir / "metrics.json", metrics.to_dict())
    (run_dir / "notebook.py").write_bytes(text.encode("utf-8"))
    return run_dir


def run_pipeline(
    text: str,
    name: str,
    client: Client,
    *,
    judge: bool,
    runs_dir: str | Path,
    set_stage: StageSetter,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run every stage in order and return only the explicit contract."""
    set_stage("Parsing notebook")
    result = parse_text(text, "py", name)
    facts = extract_facts(result)
    injection = scan_injection(result)

    chunks = plan_chunks(result)
    set_stage("Extracting claims", 0, len(chunks))
    repairs_before = client.repair_attempts
    ledger: Ledger = extract_claims(
        result,
        client,
        facts,
        on_progress=lambda done, total: set_stage("Extracting claims", done, total),
    )

    set_stage("Verifying claims")
    verified = verify_ledger(result, ledger, judge_client=client if judge else None)

    set_stage("Writing the three documents")
    documents = compose_documents(result, verified, facts, client)

    metrics = build_metrics(
        ledger,
        verified,
        documents,
        repairs=client.repair_attempts - repairs_before,
        strategy=client.strategy,
        injection=injection,
    )
    run_dir = _save_outputs(result, verified, documents, metrics, text, name, runs_dir)
    emit(metrics.line())
    return {
        "notebook": notebook_view(result, name),
        "documents": documents_view(documents),
        "claims": claims_view(verified),
        "metrics": metrics.to_dict(),
        "run_dir": str(run_dir),
    }


def _failure_message(exc: Exception) -> str:
    """Keep human-facing provider messages; label anything unexpected."""
    if isinstance(exc, (LLMAuthError, LLMRequestError, LLMTransientError, ConfigError)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"[:300]


def _run_job(
    job: Job,
    pipeline: Pipeline,
    text: str,
    name: str,
    client: Client,
    judge: bool,
    runs_dir: str | Path,
) -> None:
    """The thread boundary: every failure becomes job state, never an escape."""
    try:
        result = pipeline(
            text,
            name,
            client,
            judge=judge,
            runs_dir=runs_dir,
            set_stage=job.set_stage,
        )
    except Exception as exc:
        job.fail(_failure_message(exc))
    else:
        job.finish(result)


def _handler_class(
    client: Client,
    job: Job,
    runs_dir: str | Path,
    sample_path: str | Path | None,
    judge: bool,
    pipeline: Pipeline,
) -> type[BaseHTTPRequestHandler]:
    """Close over one server's dependencies instead of using class globals."""
    static_dir = Path(__file__).resolve().parent / "static"
    sample = Path(sample_path) if sample_path is not None else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "notebook-harness"
        sys_version = ""

        def log_message(self, format: str, *args: object) -> None:
            """Silence per-request logging; main prints the one useful line."""
            return

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:  # pragma: no cover - only malformed request lines reach here
            """Keep even protocol-level errors JSON and within the CSP."""
            self._send_json(code, {"error": message or "request error"})

        def __getattr__(self, name: str) -> Any:
            """Route every method this class does not implement to 405."""
            if name.startswith("do_"):
                return self._guarded_method
            raise AttributeError(name)  # pragma: no cover - normal attribute miss

        def do_GET(self) -> None:
            self._guarded(self._get)

        def do_POST(self) -> None:
            self._guarded(self._post)

        def _guarded_method(self) -> None:
            self._guarded(self._method_not_allowed)

        def _guarded(self, handler: Callable[[], None]) -> None:
            """One place for the host check and the no-traceback 500 policy."""
            self._sent = False
            try:
                if not self._host_ok():
                    self._send_json(403, {"error": "bad host"})
                    return
                handler()
            except Exception:  # pragma: no cover - last-resort handler
                if not self._sent:
                    with contextlib.suppress(OSError):
                        self._send_json(500, {"error": "internal error"})

        def _host_ok(self) -> bool:
            """Reject rebinding hosts before any route or body is touched."""
            port = int(getattr(self.server, "server_port", 0))
            host = self.headers.get("Host", "")
            return host in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            """Send every header, including security headers, exactly once."""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", _CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            if content_type.startswith("application/json"):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self._sent = True
            self.wfile.write(body)

        def _send_json(self, status: int, value: object) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def _method_not_allowed(self) -> None:
            self._send_json(405, {"error": "method not allowed"})

        def _content_length(self) -> int | None:
            """Return a safe length or None so the caller can answer 400."""
            header = self.headers.get("Content-Length")
            if header is None:
                return None
            try:
                return max(0, int(header))
            except ValueError:
                return None

        def _drain(self, length: int) -> None:
            """Read a bounded prefix so an early 413 cannot deadlock the client."""
            remaining = min(length, _DRAIN_LIMIT)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:  # pragma: no cover - client closed early
                    break
                remaining -= len(chunk)

        def _get(self) -> None:
            path = urlsplit(self.path).path
            entry = _STATIC_FILES.get(path)
            if entry is not None:
                filename, content_type = entry
                self._send(200, (static_dir / filename).read_bytes(), content_type)
                return
            if path == "/api/info":
                self._send_json(
                    200,
                    {
                        "provider": client.provider,
                        "model": client.model,
                        "strategy": client.strategy,
                        "sample": sample is not None,
                        "judge": judge,
                    },
                )
                return
            if path == "/api/sample":
                if sample is None:
                    self._send_json(404, {"error": "No sample notebook is configured."})
                    return
                try:
                    text = sample.read_bytes().decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    self._send_json(
                        404, {"error": "The sample notebook could not be read."}
                    )
                    return
                self._send_json(200, {"name": sample.name, "text": text})
                return
            if path == "/api/status":
                self._send_json(200, job.snapshot())
                return
            if path == "/api/result":
                result = job.completed()
                if result is None:
                    self._send_json(409, {"error": "No finished run."})
                else:
                    self._send_json(200, result)
                return
            if path in _API_PATHS:
                self._method_not_allowed()
                return
            self._send_json(404, {"error": "not found"})

        def _post(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/run":
                self._run()
                return
            if path in _API_PATHS:
                self._method_not_allowed()
                return
            self._send_json(404, {"error": "not found"})

        def _run(self) -> None:
            """Validate fully, then claim the job slot and answer 202."""
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                self._send_json(
                    415, {"error": "Send the notebook as application/json."}
                )
                return
            length = self._content_length()
            if length is None:
                self._send_json(400, {"error": "Content-Length is required."})
                return
            if length > _BODY_LIMIT:
                self._drain(length)
                self.close_connection = True
                self._send_json(413, {"error": "The request body is too large."})
                return
            raw = self.rfile.read(length)
            try:
                payload: object = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "The request body is not valid JSON."})
                return
            if not isinstance(payload, dict):
                self._send_json(
                    400, {"error": "The request body must be a JSON object."}
                )
                return
            name = payload.get("name")
            text = payload.get("text")
            if not isinstance(name, str) or not isinstance(text, str):
                self._send_json(
                    400, {"error": "Both 'name' and 'text' must be strings."}
                )
                return
            if len(text) > _TEXT_LIMIT:
                self._send_json(413, {"error": "The notebook text is too large."})
                return
            if not name.endswith(".py"):
                self._send_json(400, {"error": "Only .py notebooks are supported."})
                return
            try:
                parse_text(text, "py", name)
            except Exception as exc:
                self._send_json(400, {"error": str(exc)[:300]})
                return
            if not job.begin():
                self._send_json(409, {"error": "A run is already in progress."})
                return
            thread = threading.Thread(
                target=_run_job,
                args=(job, pipeline, text, name, client, judge, runs_dir),
                daemon=True,
            )
            thread.start()
            self._send_json(202, {"ok": True})

    return Handler


def create_server(
    client: Client,
    *,
    port: int = 0,
    runs_dir: str | Path = "runs",
    sample_path: str | Path | None = None,
    judge: bool = False,
    pipeline: Pipeline | None = None,
) -> ThreadingHTTPServer:
    """Bind the loopback-only server; tests pass port 0 for a free port."""
    job = Job()
    handler = _handler_class(
        client=client,
        job=job,
        runs_dir=runs_dir,
        sample_path=sample_path,
        judge=judge,
        pipeline=pipeline if pipeline is not None else run_pipeline,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Command line entry point; configuration errors are one clean line."""
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Serve the notebook harness on 127.0.0.1.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sample", default=None, help="notebook offered as a sample")
    parser.add_argument(
        "--judge", action="store_true", help="ask the LLM to judge claims"
    )
    parser.add_argument("--runs", default="runs", help="output directory")
    args = parser.parse_args(argv)

    try:
        load_dotenv()
        client = make_client(load_config())
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    server = create_server(
        client,
        port=args.port,
        runs_dir=args.runs,
        sample_path=args.sample,
        judge=args.judge,
    )
    port = int(server.server_address[1])
    print(
        f"Serving on http://127.0.0.1:{port}  "
        f"(provider={client.provider}, model={client.model}, "
        f"strategy={client.strategy})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
