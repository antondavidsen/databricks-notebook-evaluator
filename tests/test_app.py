"""HTTP and view tests for the local harness app."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from harness.app import (
    Client,
    Job,
    Pipeline,
    StageSetter,
    _failure_message,
    _safe_name,
    claims_view,
    create_server,
    documents_view,
    main,
    notebook_view,
    run_pipeline,
)
from harness.citations import Citation
from harness.compose import DocSection, DocSentence, Document, DocumentSet
from harness.extract import Claim
from harness.llm import FakeLLM, LLMAuthError
from harness.parser import parse_file, parse_text
from harness.verify import VerifiedClaim, VerifiedLedger

NOTEBOOK = """# Databricks notebook source
RATE = 0.2
# COMMAND ----------
print(RATE)
"""

EXTRACTION = json.dumps(
    {
        "claims": [
            {
                "kind": "OBSERVED",
                "tag": "business_rule",
                "text": "RATE is set to 0.2.",
                "quote": "RATE = 0.2",
                "cites": [{"cell": 1, "start": 1, "end": 1}],
            }
        ]
    }
)
MANAGER = json.dumps(
    {
        "title": "Notebook summary",
        "sections": [
            {
                "heading": "What this notebook does",
                "sentences": [{"text": "The rate is set to 0.2.", "ids": ["F1"]}],
            }
        ],
    }
)
DEVELOPER = json.dumps(
    {
        "title": "Notebook guide",
        "sections": [
            {
                "heading": "Purpose",
                "sentences": [{"text": "The rate is set to 0.2.", "ids": ["F1"]}],
            }
        ],
    }
)
AGENT = json.dumps(
    {
        "title": "Agent brief",
        "sections": [
            {
                "heading": "Invariants",
                "sentences": [{"text": "The rate is set to 0.2.", "ids": ["F1"]}],
            }
        ],
    }
)


def fake_result(name: str = "demo.py") -> dict[str, Any]:
    """A result shaped like run_pipeline's contract, with no real content."""
    return {
        "notebook": {"name": name, "cells": []},
        "documents": {
            "manager": {},
            "developer": {},
            "agent": {},
            "issues": [],
            "summary": {},
            "notes": [],
        },
        "claims": [],
        "run_dir": "nowhere",
    }


def make_pipeline(result: dict[str, Any], seen: dict[str, Any]) -> Pipeline:
    """Record the arguments the handler passes, then return a fixed result."""

    def pipeline(
        text: str,
        name: str,
        client: Client,
        *,
        judge: bool,
        runs_dir: str | Path,
        set_stage: StageSetter,
    ) -> dict[str, Any]:
        seen.update(
            {
                "text": text,
                "name": name,
                "client": client,
                "judge": judge,
                "runs_dir": str(runs_dir),
            }
        )
        set_stage("Working", 1, 2)
        return result

    return pipeline


@contextmanager
def serve(
    tmp_path: Path,
    *,
    pipeline: Pipeline | None = None,
    sample_path: Path | None = None,
    judge: bool = False,
    client: FakeLLM | None = None,
) -> Iterator[ThreadingHTTPServer]:
    """Run one loopback server for the duration of a test."""
    server = create_server(
        client if client is not None else FakeLLM([]),
        port=0,
        runs_dir=tmp_path / "runs",
        sample_path=sample_path,
        judge=judge,
        pipeline=pipeline,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def call(
    server: ThreadingHTTPServer,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Make one HTTP/1.1 request and return status, headers and body."""
    conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        return (
            response.status,
            {key.lower(): value for key, value in response.getheaders()},
            data,
        )
    finally:
        conn.close()


def decode(data: bytes) -> dict[str, Any]:
    """Parse a JSON object response, failing the test on any other shape."""
    value = json.loads(data)
    assert isinstance(value, dict)
    return value


def run_body(text: str = NOTEBOOK, name: str = "demo.py") -> bytes:
    """Encode the one request shape /api/run accepts."""
    return json.dumps({"name": name, "text": text}).encode("utf-8")


def wait_for_state(
    server: ThreadingHTTPServer, wanted: str, timeout: float = 5.0
) -> dict[str, Any]:
    """Poll /api/status until it reaches a terminal or requested state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, _, data = call(server, "GET", "/api/status")
        assert status == 200
        payload = decode(data)
        if payload["state"] == wanted:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"state never became {wanted!r}")


def test_job_begin_is_atomic_and_late_progress_is_ignored() -> None:
    job = Job()

    assert job.begin() is True
    assert job.begin() is False
    job.set_stage("Halfway", 2, 4)
    assert job.snapshot() == {
        "state": "running",
        "stage": "Halfway",
        "done": 2,
        "total": 4,
        "error": None,
    }
    assert job.completed() is None

    job.finish({"value": 1})
    assert job.completed() == {"value": 1}
    job.set_stage("Too late", 4, 4)
    assert job.snapshot()["stage"] == "Finished"

    job.fail("broken")
    assert job.completed() is None
    assert job.snapshot()["error"] == "broken"


def test_safe_name_is_one_path_segment() -> None:
    assert _safe_name("a notebook.py") == "a_notebook"
    assert _safe_name("../../etc/passwd.py") == ".._.._etc_passwd"
    assert _safe_name(".py") == "notebook"


def test_failure_message_keeps_provider_text_and_labels_the_rest() -> None:
    assert _failure_message(LLMAuthError("the key is wrong")) == "the key is wrong"
    message = _failure_message(RuntimeError("x" * 500))
    assert message.startswith("RuntimeError: ")
    assert len(message) == 300


def test_notebook_view_keeps_both_line_numbers() -> None:
    result = parse_text(NOTEBOOK, "py", "demo.py")

    view = notebook_view(result, "demo.py")

    assert view["name"] == "demo.py"
    assert [cell["number"] for cell in view["cells"]] == [1, 2]
    assert view["cells"][0]["kind"] == "python"
    assert view["cells"][0]["lines"] == [{"rel": 1, "abs": 2, "raw": "RATE = 0.2"}]
    assert notebook_view(parse_text(NOTEBOOK, "py"))["name"] == ""


def test_claims_view_sorts_ids_numerically_and_keeps_dropped() -> None:
    cite = Citation(1, 1, 1)
    first = Claim("F2", "OBSERVED", "config", "first", "RATE = 0", (cite,), 1)
    tenth = Claim("F10", "OBSERVED", "config", "tenth", "RATE = 1", (cite,), 1)
    other = Claim("note", "OBSERVED", "config", "other", "RATE = 2", (cite,), 1)
    ledger = VerifiedLedger(
        (
            VerifiedClaim(tenth, "flagged", "grounding", ("weak",), "unsupported"),
            VerifiedClaim(other, "dropped", "citation", ("bad cite",), None),
            VerifiedClaim(first, "verified", None, (), None),
        ),
        (),
        (),
        ("note",),
    )

    view = claims_view(ledger)

    assert [claim["id"] for claim in view] == ["note", "F2", "F10"]
    assert view[1]["status"] == "verified"
    assert view[1]["cites"] == [{"cell": 1, "start": 1, "end": 1}]
    assert view[1]["reasons"] == []
    assert view[1]["judge"] is None
    assert view[2]["reasons"] == ["weak"]
    assert view[2]["judge"] == "unsupported"
    assert view[0]["status"] == "dropped"
    assert view[0]["layer"] == "citation"


def test_documents_view_is_an_explicit_contract() -> None:
    cite = Citation(1, 1, 1)
    sentence = DocSentence("text", ("F1",), (cite,), "claims", ("flag",))
    section = DocSection("Heading", (sentence,))
    doc = Document("manager", "Title", (section,), ("issue",), None, ("F1",))
    documents = DocumentSet(
        doc,
        doc,
        doc,
        ("cross",),
        {"total": 1, "verified": 1, "flagged": 0, "dropped": 0, "used": 1},
        ("read this",),
    )

    view = documents_view(documents)

    assert view["manager"] == {
        "audience": "manager",
        "title": "Title",
        "sections": [
            {
                "heading": "Heading",
                "sentences": [
                    {
                        "text": "text",
                        "ids": ["F1"],
                        "cites": [{"cell": 1, "start": 1, "end": 1}],
                        "source": "claims",
                        "flags": ["flag"],
                    }
                ],
            }
        ],
        "issues": ["issue"],
        "error": None,
        "used_ids": ["F1"],
    }
    assert view["issues"] == ["cross"]
    assert view["summary"] == {
        "total": 1,
        "verified": 1,
        "flagged": 0,
        "dropped": 0,
        "used": 1,
    }
    assert view["notes"] == ["read this"]


def test_static_allowlist_headers_and_traversal(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        status, headers, body = call(server, "GET", "/")
        assert status == 200
        assert headers["content-type"] == "text/html; charset=utf-8"
        assert headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert b'id="file"' in body

        status, headers, _ = call(server, "GET", "/static/app.js")
        assert status == 200
        assert headers["content-type"] == "text/javascript; charset=utf-8"

        status, headers, _ = call(server, "GET", "/static/style.css")
        assert status == 200
        assert headers["content-type"] == "text/css; charset=utf-8"

        for path in ("/static/../app.py", "/../harness/app.py", "/app.py"):
            status, _, body = call(server, "GET", path)
            assert status == 404
            assert decode(body) == {"error": "not found"}

        status, headers, body = call(server, "GET", "/api/status")
        assert status == 200
        assert headers["cache-control"] == "no-store"
        assert decode(body)["state"] == "idle"


def test_foreign_host_is_rejected_before_any_route(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        status, _, body = call(
            server, "GET", "/api/info", headers={"Host": "attacker.example"}
        )

    assert status == 403
    assert decode(body) == {"error": "bad host"}


def test_methods_and_unknown_paths(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        status, _, body = call(server, "GET", "/api/run")
        assert status == 405
        assert decode(body) == {"error": "method not allowed"}

        status, _, _ = call(server, "POST", "/api/status", body=b"")
        assert status == 405

        status, _, _ = call(server, "PUT", "/api/status", body=b"")
        assert status == 405

        status, _, body = call(server, "POST", "/api/nope", body=b"")
        assert status == 404
        assert decode(body) == {"error": "not found"}


def test_run_rejects_bad_requests_before_starting_a_job(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    pipeline = make_pipeline(fake_result(), seen)

    with serve(tmp_path, pipeline=pipeline) as server:
        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=run_body(),
            headers={"Content-Type": "text/plain"},
        )
        assert status == 415
        assert "application/json" in decode(body)["error"]

        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            conn.putrequest("POST", "/api/run")
            conn.putheader("Content-Type", "application/json")
            conn.endheaders()
            response = conn.getresponse()
            assert response.status == 400
            assert "Content-Length" in decode(response.read())["error"]
        finally:
            conn.close()

        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        try:
            conn.putrequest("POST", "/api/run")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", "not-a-number")
            conn.endheaders()
            response = conn.getresponse()
            assert response.status == 400
            assert "Content-Length" in decode(response.read())["error"]
        finally:
            conn.close()

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=b"{",
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "valid JSON" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=b"[]",
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "JSON object" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=b'{"name": "demo.py"}',
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "must be strings" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=b'{"name": 1, "text": ""}',
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "must be strings" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=run_body(text="print(1)\n", name="demo.txt"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert ".py" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=run_body(text=""),
            headers={"Content-Type": "application/json"},
        )
        assert status == 400
        assert "empty file" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=run_body(text="x" * 1_000_001),
            headers={"Content-Type": "application/json"},
        )
        assert status == 413
        assert "notebook text" in decode(body)["error"]

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=b"x" * 1_500_001,
            headers={"Content-Type": "application/json"},
        )
        assert status == 413
        assert "request body" in decode(body)["error"]

        assert "text" not in seen
        status, _, body = call(server, "GET", "/api/status")
        assert decode(body)["state"] == "idle"


def test_successful_run_returns_202_then_the_result(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    result = fake_result()
    pipeline = make_pipeline(result, seen)

    with serve(tmp_path, pipeline=pipeline) as server:
        status, _, body = call(server, "GET", "/api/result")
        assert status == 409
        assert decode(body) == {"error": "No finished run."}

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=run_body(),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert status == 202
        assert decode(body) == {"ok": True}

        state = wait_for_state(server, "done")
        assert state["stage"] == "Finished"

        status, _, body = call(server, "GET", "/api/result")
        assert status == 200
        assert decode(body) == result

    assert seen["name"] == "demo.py"
    assert seen["text"] == NOTEBOOK
    assert seen["judge"] is False
    assert Path(seen["runs_dir"]) == tmp_path / "runs"


def test_second_run_is_rejected_while_the_first_is_running(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    result = fake_result()

    def pipeline(
        text: str,
        name: str,
        client: Client,
        *,
        judge: bool,
        runs_dir: str | Path,
        set_stage: StageSetter,
    ) -> dict[str, Any]:
        set_stage("Working", 1, 2)
        started.set()
        assert release.wait(timeout=5)
        return result

    with serve(tmp_path, pipeline=pipeline) as server:
        status, _, _ = call(
            server,
            "POST",
            "/api/run",
            body=run_body(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 202
        assert started.wait(timeout=5)

        status, _, body = call(
            server,
            "POST",
            "/api/run",
            body=run_body(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 409
        assert decode(body) == {"error": "A run is already in progress."}
        assert decode(call(server, "GET", "/api/status")[2])["state"] == "running"

        release.set()
        assert wait_for_state(server, "done")["state"] == "done"
        assert decode(call(server, "GET", "/api/result")[2]) == result


def test_pipeline_failure_becomes_job_state_not_a_crash(tmp_path: Path) -> None:
    def pipeline(
        text: str,
        name: str,
        client: Client,
        *,
        judge: bool,
        runs_dir: str | Path,
        set_stage: StageSetter,
    ) -> dict[str, Any]:
        set_stage("Working")
        raise RuntimeError("broken pipeline")

    with serve(tmp_path, pipeline=pipeline) as server:
        status, _, _ = call(
            server,
            "POST",
            "/api/run",
            body=run_body(),
            headers={"Content-Type": "application/json"},
        )
        assert status == 202

        state = wait_for_state(server, "error")
        assert state["stage"] == "Failed"
        assert state["error"] == "RuntimeError: broken pipeline"

        status, _, body = call(server, "GET", "/api/result")
        assert status == 409
        assert decode(body) == {"error": "No finished run."}


def test_info_sample_and_judge_are_reported(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(NOTEBOOK, encoding="utf-8")
    seen: dict[str, Any] = {}

    with serve(
        tmp_path,
        pipeline=make_pipeline(fake_result("sample.py"), seen),
        sample_path=sample,
        judge=True,
    ) as server:
        status, _, body = call(server, "GET", "/api/info")
        assert status == 200
        assert decode(body) == {
            "provider": "fake",
            "model": "fake",
            "strategy": "prompt",
            "sample": True,
            "judge": True,
        }

        status, _, body = call(server, "GET", "/api/sample")
        assert status == 200
        assert decode(body) == {"name": "sample.py", "text": NOTEBOOK}

        status, _, _ = call(
            server,
            "POST",
            "/api/run",
            body=run_body(name="sample.py"),
            headers={"Content-Type": "application/json"},
        )
        assert status == 202
        assert wait_for_state(server, "done")["state"] == "done"

    assert seen["judge"] is True
    assert seen["name"] == "sample.py"
    assert isinstance(seen["client"], FakeLLM)


def test_sample_is_absent_or_unreadable(tmp_path: Path) -> None:
    with serve(tmp_path) as server:
        status, _, body = call(server, "GET", "/api/sample")
        assert status == 404
        assert decode(body) == {"error": "No sample notebook is configured."}
        assert decode(call(server, "GET", "/api/info")[2])["sample"] is False

    with serve(tmp_path, sample_path=tmp_path / "missing.py") as server:
        status, _, body = call(server, "GET", "/api/sample")
        assert status == 404
        assert decode(body) == {"error": "The sample notebook could not be read."}


def test_run_pipeline_end_to_end_writes_every_output(tmp_path: Path) -> None:
    client = FakeLLM([EXTRACTION, MANAGER, DEVELOPER, AGENT])
    events: list[tuple[str, int, int]] = []

    result = run_pipeline(
        NOTEBOOK,
        "demo.py",
        client,
        judge=False,
        runs_dir=tmp_path / "runs",
        set_stage=lambda stage, done=0, total=0: events.append((stage, done, total)),
    )

    assert [event[0] for event in events[:2]] == [
        "Parsing notebook",
        "Extracting claims",
    ]
    assert ("Extracting claims", 1, 1) in events
    assert ("Writing the three documents", 0, 0) in events
    assert len(client.calls) == 4

    notebook = result["notebook"]
    assert notebook["name"] == "demo.py"
    assert notebook["cells"][0]["lines"] == [{"rel": 1, "abs": 2, "raw": "RATE = 0.2"}]

    claims = result["claims"]
    assert [claim["id"] for claim in claims] == ["F1"]
    assert claims[0]["status"] == "verified"
    assert claims[0]["text"] == "RATE is set to 0.2."
    assert claims[0]["cites"] == [{"cell": 1, "start": 1, "end": 1}]

    documents = result["documents"]
    assert documents["summary"] == {
        "total": 1,
        "verified": 1,
        "flagged": 0,
        "dropped": 0,
        "used": 1,
    }
    sentence = documents["manager"]["sections"][0]["sentences"][0]
    assert sentence["text"] == "The rate is set to 0.2."
    assert sentence["source"] == "claims"
    assert sentence["cites"] == [{"cell": 1, "start": 1, "end": 1}]
    assert sentence["flags"] == []
    assert documents["developer"]["sections"][0]["heading"] == "Purpose"
    assert "Invariants" in {
        section["heading"] for section in documents["agent"]["sections"]
    }

    run_dir = Path(result["run_dir"])
    assert run_dir.parent == tmp_path / "runs"
    assert run_dir.name.endswith("-demo")
    for filename in (
        "manager.md",
        "developer.md",
        "agent.md",
        "claims.json",
        "documents.json",
        "notebook.py",
    ):
        assert (run_dir / filename).is_file()
    assert (run_dir / "notebook.py").read_text(encoding="utf-8") == NOTEBOOK
    assert json.loads((run_dir / "claims.json").read_text(encoding="utf-8")) == claims
    assert (
        json.loads((run_dir / "documents.json").read_text(encoding="utf-8"))
        == documents
    )
    assert "The rate is set to 0.2." in (run_dir / "manager.md").read_text(
        encoding="utf-8"
    )


def test_real_notebook_anchors_survive_the_notebook_view() -> None:
    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "cash_application_engine.py"
    )
    result = parse_file(fixture)

    view = notebook_view(result)

    assert len(view["cells"]) == 47
    assert view["cells"][28]["lines"][3] == {
        "rel": 4,
        "abs": 361,
        "raw": "TOL = 5.0",
    }
    assert view["cells"][4]["lines"][0] == {
        "rel": 1,
        "abs": 44,
        "raw": "# MAGIC %run ../shared/fin_utils",
    }


def test_malformed_request_line_gets_a_json_error(tmp_path: Path) -> None:
    with (
        serve(tmp_path) as server,
        socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as sock,
    ):
        sock.settimeout(1.0)
        sock.sendall(b"NOT-AN-HTTP-REQUEST\r\n\r\n")
        chunks: list[bytes] = []
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        except TimeoutError:
            pass

    raw = b"".join(chunks)
    assert raw.startswith(b'{"error"')
    assert b"Bad request syntax" in raw


def test_module_entry_point_reexports_main() -> None:
    import importlib

    entry = importlib.import_module("harness.__main__")
    assert entry.main is main
