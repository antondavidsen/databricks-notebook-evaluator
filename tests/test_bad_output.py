"""one table drills every bad-output shape through repair and verify."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from harness.citations import Citation
from harness.compose import DocOut
from harness.extract import ChunkOutput, Claim, extract_claims
from harness.llm import (
    Completion,
    FakeLLM,
    StructuredOutputError,
    TruncatedOutput,
)
from harness.parser import parse_text
from harness.verify import verify_claim, verify_ledger

NOTEBOOK = "# Databricks notebook source\nRATE = 0.2\n"
GOOD_CLAIM = {
    "kind": "OBSERVED",
    "tag": "config",
    "text": "RATE is set to 0.2.",
    "quote": "RATE = 0.2",
    "cites": [{"cell": 1, "start": 1, "end": 1}],
}
GOOD = json.dumps({"claims": [GOOD_CLAIM]})


def _claim(overrides: dict[str, object]) -> str:
    return json.dumps({"claims": [{**GOOD_CLAIM, **overrides}]})


def _claim_without(field: str) -> str:
    body = {key: value for key, value in GOOD_CLAIM.items() if key != field}
    return json.dumps({"claims": [body]})


@dataclass(frozen=True)
class BadReply:
    """One malformed reply, the feedback it must produce, and the repairs."""

    reply: str
    hint: str
    repairs: int


REPAIRABLE = (
    pytest.param(
        BadReply('{"claims": [{"kind": "OBSERVED"}],}', "trailing comma", 1),
        id="trailing-comma",
    ),
    pytest.param(
        BadReply("{\u201cclaims\u201d: []}", "Expecting property name", 1),
        id="smart-quotes",
    ),
    pytest.param(
        BadReply('{"claims": [], "claims": []}', 'duplicate key "claims"', 1),
        id="duplicate-keys",
    ),
    pytest.param(
        BadReply(_claim_without("quote"), "quote", 1),
        id="missing-field",
    ),
    pytest.param(
        BadReply(_claim({"cites": 1}), "cites", 1),
        id="wrong-cites-type",
    ),
    pytest.param(
        BadReply(
            _claim({"kind": "OBSERVED", "tag": "risk"}),
            'tag "risk" requires kind "INFERRED"',
            1,
        ),
        id="kind-tag-mismatch",
    ),
    pytest.param(
        BadReply('{"claims": [{"kind": "OBSERVED"', "Expecting", 1),
        id="physically-truncated",
    ),
)


@pytest.mark.parametrize("case", REPAIRABLE)
def test_bad_replies_are_repaired_then_verified(case: BadReply) -> None:
    parsed = parse_text(NOTEBOOK, "py", "demo.py")
    client = FakeLLM([case.reply, GOOD])

    ledger = extract_claims(parsed, client)
    verified = verify_ledger(parsed, ledger)

    assert len(client.calls) == 1 + case.repairs
    assert client.repair_attempts == case.repairs
    assert case.hint in client.calls[-1]["messages"][-1]["content"]
    assert "invalid" in client.calls[-1]["messages"][-1]["content"]
    assert len(ledger.claims) == 1
    assert ledger.n_dropped_foreign == 0
    assert [item.status for item in verified.items] == ["verified"]


def test_good_replies_need_no_repair_and_are_recovered_from_prose() -> None:
    for reply in (
        GOOD,
        f"```json\n{GOOD}\n```",
        f"I will use {{ref}} first.\n{GOOD}",
    ):
        client = FakeLLM([reply])

        ledger = extract_claims(parse_text(NOTEBOOK, "py", "demo.py"), client)

        assert len(ledger.claims) == 1
        assert client.repair_attempts == 0
        assert len(client.calls) == 1


def test_duplicate_keys_are_never_resolved_last_wins() -> None:
    first = json.dumps(GOOD_CLAIM)
    second = json.dumps({**GOOD_CLAIM, "text": "RATE is set to 0.9."})
    reply = '{"claims": [' + first + '], "claims": [' + second + "]}"

    client = FakeLLM([reply, GOOD])

    ledger = extract_claims(parse_text(NOTEBOOK, "py", "demo.py"), client)

    assert len(ledger.claims) == 1
    assert ledger.claims[0].text == "RATE is set to 0.2."
    assert 'duplicate key "claims"' in client.calls[1]["messages"][-1]["content"]


@dataclass(frozen=True)
class TerminalReply:
    """A reply that can never validate, and the error it must raise."""

    reply: str | Completion
    repeats: int
    error: type[Exception]
    hint: str


TERMINAL = (
    pytest.param(
        TerminalReply(
            '{"claims": [],}',
            3,
            StructuredOutputError,
            "trailing comma",
        ),
        id="trailing-comma",
    ),
    pytest.param(
        TerminalReply(
            '{"claims": [], "claims": []}',
            3,
            StructuredOutputError,
            'duplicate key "claims"',
        ),
        id="duplicate-keys",
    ),
    pytest.param(
        TerminalReply(
            "{\u201cclaims\u201d: []}",
            3,
            StructuredOutputError,
            "Expecting property name",
        ),
        id="smart-quotes",
    ),
    pytest.param(
        TerminalReply(
            Completion("partial", "length", True, None, None, 0, "fake", "fake"),
            1,
            TruncatedOutput,
            "output limit",
        ),
        id="finish-reason-length",
    ),
)


@pytest.mark.parametrize("case", TERMINAL)
def test_terminal_bad_outputs_stop_after_the_bounded_retries(
    case: TerminalReply,
) -> None:
    client = FakeLLM([case.reply] * case.repeats)

    with pytest.raises(case.error) as caught:
        client.generate([{"role": "user", "content": "extract"}], ChunkOutput)

    assert case.hint in str(caught.value)
    assert len(client.calls) == case.repeats
    assert client.repair_attempts == case.repeats - 1
    if isinstance(caught.value, StructuredOutputError):
        assert caught.value.attempts == case.repeats


@pytest.mark.parametrize(
    ("text", "quote", "cell", "expected", "layer", "reason"),
    [
        pytest.param(
            "RATE is set to 0.2.",
            "RATE = 0.2",
            999,
            "dropped",
            "citation",
            "cell 999 does not exist",
            id="missing-cell",
        ),
        pytest.param(
            "RATE is set to 0.2.",
            "RATE = 0.9",
            1,
            "dropped",
            "quote",
            "quote not found",
            id="quote-mismatch",
        ),
        pytest.param(
            "RATE is set to 0.5.",
            "RATE = 0.2",
            1,
            "flagged",
            "grounding",
            "not in the cited lines: 0.5",
            id="number",
        ),
        pytest.param(
            "RATE is set to 0.2.",
            "RATE = 0.2",
            1,
            "verified",
            None,
            "",
            id="valid",
        ),
    ],
)
def test_verifier_table_reports_the_first_failing_layer(
    text: str,
    quote: str,
    cell: int,
    expected: str,
    layer: str | None,
    reason: str,
) -> None:
    claim = Claim(
        id="F1",
        kind="OBSERVED",
        tag="config",
        text=text,
        quote=quote,
        cites=(Citation(cell, 1, 1),),
        chunk=1,
    )

    verified = verify_claim(parse_text(NOTEBOOK, "py", "demo.py"), claim)

    assert verified.status == expected
    assert verified.layer == layer
    if reason:
        assert reason in verified.reasons[0]


def test_composer_schema_is_also_covered_by_the_repair_loop() -> None:
    """Same loop, different schema: proves the table is not extractor-only."""
    client = FakeLLM(["{}", '{"title": "ok", "sections": []}'])

    document = client.generate([{"role": "user", "content": "write"}], DocOut)

    assert document.title == "ok"
    assert client.repair_attempts == 1
    assert "title" in client.calls[1]["messages"][-1]["content"]
