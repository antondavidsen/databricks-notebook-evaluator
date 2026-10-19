from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from harness import (
    Citation,
    Claim,
    FakeLLM,
    ParseResult,
    StaticFacts,
    compose_document,
    compose_documents,
    extract_facts,
    parse_file,
    parse_text,
    render_markdown,
)
from harness.compose import (
    AGENT_FINAL_ORDER,
    AGENT_MODEL_HEADINGS,
    AGENT_PROMPT,
    COMMON_PROMPT,
    DEVELOPER_HEADINGS,
    DEVELOPER_PROMPT,
    LIMITS_HEADING,
    MANAGER_HEADINGS,
    MANAGER_PROMPT,
    Audience,
    DocOut,
    DocSection,
    DocSentence,
    Document,
    DocumentSet,
    SectionOut,
    SentenceOut,
    build_messages,
    check_consistency,
    check_document,
    select_claims,
)
from harness.extract import FailedRegion, Kind, Tag
from harness.facts import (
    Binding,
    ConstantFact,
    FunctionFact,
    Mutation,
    TableFact,
    UnresolvedCall,
)
from harness.llm import (
    LLMAuthError,
    LLMRequestError,
    LLMResponseError,
    LLMTransientError,
    StructuredOutputError,
    TruncatedOutput,
)
from harness.verify import Status, VerifiedClaim, VerifiedLedger

HEADER = "# Databricks notebook source"
SEP = "# COMMAND ----------"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cash_application_engine.py"

SMALL_CELLS = (
    "bal = {}\nTOL = 0.5\nLIMIT = 10\nearly_call()",
    "# MAGIC %run ../shared/utils",
    (
        'def apply(amount):\n    bal["total"] = amount\n\n'
        "def helper():\n    return 1\n\n"
        "def caller():\n    notify_ar()\n    log_run()\n\n"
        "apply(1)\ncaller()"
    ),
    (
        'spark.range(1).write.format("delta").mode("overwrite")'
        '.saveAsTable("cat.tbl")\n'
        'spark.range(1).write.format("delta").saveAsTable("cat.raw")'
    ),
    "TOL = 5.0\nprint(TOL)",
)


def notebook(*cells: str) -> ParseResult:
    body = f"\n\n{SEP}\n\n".join(cells)
    return parse_text(f"{HEADER}\n\n{body}\n", "py")


def c(cell: int, start: int = 1, end: int | None = None) -> Citation:
    return Citation(cell, start, start if end is None else end)


def mk(
    id: str,
    kind: Kind,
    tag: Tag,
    text: str,
    cites: Sequence[Citation],
) -> Claim:
    return Claim(
        id=id,
        kind=kind,
        tag=tag,
        text=text,
        quote="x",
        cites=tuple(cites),
        chunk=1,
    )


def vc(claim: Claim, status: Status = "verified") -> VerifiedClaim:
    return VerifiedClaim(claim=claim, status=status, layer=None, reasons=(), judge=None)


def ledger(
    *items: VerifiedClaim,
    failed_regions: tuple[FailedRegion, ...] = (),
    uncovered_cells: tuple[int, ...] = (),
    notes: tuple[str, ...] = ("note",),
) -> VerifiedLedger:
    return VerifiedLedger(
        items=tuple(items),
        failed_regions=failed_regions,
        uncovered_cells=uncovered_cells,
        notes=notes,
    )


def small_result() -> ParseResult:
    return notebook(*SMALL_CELLS)


def small_facts() -> StaticFacts:
    return extract_facts(small_result())


def sent(text: str, *ids: str) -> dict[str, Any]:
    return {"text": text, "ids": list(ids)}


def sec(heading: str, *sentences: dict[str, Any]) -> dict[str, Any]:
    return {"heading": heading, "sentences": list(sentences)}


def reply(title: str, *sections: dict[str, Any]) -> str:
    return json.dumps({"title": title, "sections": list(sections)})


def dsection(heading: str, *sentences: DocSentence) -> DocSection:
    return DocSection(heading, tuple(sentences))


def document(
    audience: Audience,
    *sections: DocSection,
    issues: tuple[str, ...] = (),
    error: str | None = None,
    used_ids: tuple[str, ...] = (),
) -> Document:
    return Document(
        audience=audience,
        title="Document",
        sections=tuple(sections),
        issues=issues,
        error=error,
        used_ids=used_ids,
    )


def real_result() -> ParseResult:
    return parse_file(FIXTURE)


def real_ledger() -> VerifiedLedger:
    """The six claims used by the specification's full-run scenario."""
    items = (
        vc(
            mk(
                "F1",
                "OBSERVED",
                "config",
                "TOL is set to 0.5 in the settings cell.",
                [c(4, 1)],
            )
        ),
        vc(
            mk(
                "F2",
                "OBSERVED",
                "config",
                "TOL is reassigned to 5.0 before the allocation loop.",
                [c(29, 4)],
            )
        ),
        vc(
            mk(
                "F3",
                "OBSERVED",
                "config",
                "MAX_WO is set to 250.0 and never read.",
                [c(4, 2)],
            )
        ),
        vc(
            mk(
                "F4",
                "NOT_VISIBLE",
                "external_dependency",
                "notify_ar is called but defined outside this notebook.",
                [c(43, 11)],
            )
        ),
        vc(
            mk(
                "F5",
                "OBSERVED",
                "data_flow",
                "A SQL cell counts rows in fin_cash.alloc_lines by type.",
                [c(45, 2)],
            )
        ),
        vc(
            mk(
                "F6",
                "INFERRED",
                "risk",
                "Raising TOL changes how many small differences are written off.",
                [c(29, 4)],
            )
        ),
    )
    return ledger(*items)


def manager_reply() -> str:
    return reply(
        "Manager view",
        sec(
            "What we cannot see",
            sent(
                "One step of the process is defined outside this notebook, "
                "so what it does cannot be seen here.",
                "F4",
            ),
        ),
    )


def developer_reply() -> str:
    return reply(
        "Developer guide",
        sec("Parameters", sent("TOL is set to 0.5 in the settings cell.", "F1")),
        sec("Data flow", sent("A SQL cell counts rows by type.", "F5")),
    )


def agent_reply() -> str:
    return reply(
        "Agent brief",
        sec(
            "Risky areas",
            sent(
                "Raising TOL may change how many small differences are written off.",
                "F6",
            ),
        ),
        sec(
            "Post-change checks",
            sent(
                "Compare the number of write-off lines before and after changing TOL.",
                "F6",
            ),
        ),
    )


# --- Claim selection and prompts -------------------------------------------


def test_select_claims_verified_sorted_and_truncated() -> None:
    f10 = mk("F10", "OBSERVED", "config", "Ten.", [c(5)])
    f2 = mk("F2", "OBSERVED", "config", "Two.", [c(1, 3)])
    f3 = mk("F3", "OBSERVED", "config", "Three.", [c(1, 1)])
    f4 = mk("F4", "OBSERVED", "config", "Four.", [c(1, 1)])
    no_cites = mk("FX", "OBSERVED", "config", "No cite.", [])
    flagged = mk("F5", "OBSERVED", "config", "Bad.", [c(1)])
    verified = ledger(
        vc(f2), vc(f3), vc(f4), vc(f10), vc(no_cites), vc(flagged, "flagged")
    )

    selected, omitted = select_claims(verified, "developer", 10)
    assert [item.claim.id for item in selected] == [
        "F3",
        "F4",
        "F2",
        "F10",
        "FX",
    ]
    assert omitted == 0

    selected, omitted = select_claims(verified, "developer", 2)
    assert [item.claim.id for item in selected] == ["F3", "F4"]
    assert omitted == 3

    selected, omitted = select_claims(verified, "developer", 0)
    assert selected == ()
    assert omitted == 5


def test_select_claims_manager_keeps_business_tags() -> None:
    claims = (
        mk("F1", "OBSERVED", "business_rule", "Rule.", [c(1)]),
        mk("F2", "INFERRED", "risk", "Risk.", [c(2)]),
        mk("F3", "NOT_VISIBLE", "external_dependency", "Outside.", [c(3)]),
        mk("F4", "NOT_VISIBLE", "not_visible", "Hidden.", [c(4)]),
        mk("F5", "OBSERVED", "data_flow", "Flow.", [c(5)]),
    )
    selected, omitted = select_claims(
        ledger(*(vc(claim) for claim in claims)), "manager", 120
    )
    assert [item.claim.id for item in selected] == ["F1", "F2", "F3", "F4"]
    assert omitted == 0


def test_build_messages_includes_only_stated_constants() -> None:
    verified = [vc(mk("F1", "OBSERVED", "config", "TOL is 0.5.", [c(1, 2)]))]
    for audience in ("developer", "agent"):
        messages = build_messages(audience, verified, small_facts())
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith(COMMON_PROMPT)
        user = messages[1]["content"]
        assert "F1 [OBSERVED, config] TOL is 0.5." in user
        assert "CONSTANT NOTES" in user
        assert "TOL: assigned 0.5, then 5.0" in user
        assert "C1" not in user
        assert user.endswith("Write the document now.")

    manager = build_messages("manager", verified, small_facts())
    assert "CONSTANT NOTES" not in manager[1]["content"]

    plain = build_messages("developer", verified, None)
    assert "CONSTANT NOTES" not in plain[1]["content"]

    stable = extract_facts(notebook("TOL = 0.5"))
    quiet = build_messages("agent", verified, stable)
    assert "CONSTANT NOTES" not in quiet[1]["content"]

    raw = [mk("F1", "OBSERVED", "config", "TOL is 0.5.", [c(1)])]
    mixed = build_messages("agent", raw, small_facts())
    assert "TOL is 0.5." in mixed[1]["content"]


def test_prompts_state_their_rules_and_headings() -> None:
    assert MANAGER_PROMPT.startswith(COMMON_PROMPT)
    assert DEVELOPER_PROMPT.startswith(COMMON_PROMPT)
    assert AGENT_PROMPT.startswith(COMMON_PROMPT)
    for heading in MANAGER_HEADINGS:
        assert heading in MANAGER_PROMPT
    for heading in DEVELOPER_HEADINGS:
        assert heading in DEVELOPER_PROMPT
    for heading in AGENT_MODEL_HEADINGS:
        assert heading in AGENT_PROMPT
    assert "Do not use code names" in MANAGER_PROMPT
    assert "other sections" in AGENT_PROMPT


# --- Reply assembly and deterministic checks --------------------------------


def test_headings_are_canonical_and_missing_are_empty() -> None:
    result = notebook("x = 1")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "x is 1.", [c(1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="  WHAT THIS NOTEBOOK DOES  ",
                sentences=[SentenceOut(text=" x is 1. ", ids=["F1"])],
            )
        ],
    )
    check = check_document("manager", out, [claim], result, facts_, ledger(claim))
    assert [section.heading for section in check.sections] == list(MANAGER_HEADINGS)
    assert check.sections[0].sentences == (
        DocSentence("x is 1.", ("F1",), (c(1),), "claims", ()),
    )
    assert all(not section.sentences for section in check.sections[1:])
    assert check.issues == ()
    assert check.used_ids == ("F1",)


def test_duplicate_and_unexpected_sections_are_reported() -> None:
    result = notebook("x = 1")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "x is 1.", [c(1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="Purpose",
                sentences=[SentenceOut(text="First.", ids=["F1"])],
            ),
            SectionOut(
                heading=" purpose ",
                sentences=[SentenceOut(text="Second.", ids=["F1"])],
            ),
            SectionOut(heading="Extra", sentences=[]),
        ],
    )
    check = check_document("developer", out, [claim], result, facts_, ledger(claim))
    assert check.sections[0].sentences[0].text == "First."
    assert check.issues == ("unexpected section 'Extra' ignored",)
    assert check.used_ids == ("F1",)


def test_unknown_and_missing_ids_are_unsourced() -> None:
    result = notebook("x = 1")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "x is 1.", [c(1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="Gotchas",
                sentences=[
                    SentenceOut(text="Known and unknown.", ids=["F1", "F99", "F99"]),
                    SentenceOut(text="Nothing.", ids=[]),
                    SentenceOut(text="Only unknown.", ids=["F98"]),
                ],
            )
        ],
    )
    check = check_document("developer", out, [claim], result, facts_, ledger(claim))
    first, second, third = check.sections[7].sentences
    assert first.ids == ("F1",)
    assert first.flags == ("unknown id F99",)
    assert first.source == "claims"
    assert first.cites == (c(1),)
    assert second.ids == ()
    assert second.flags == ("no claim ids",)
    assert second.source == "unsourced"
    assert third.ids == ()
    assert third.flags == ("unknown id F98", "no claim ids")
    assert third.source == "unsourced"


def test_numbers_and_identifiers_must_come_from_cited_claims() -> None:
    result = notebook("TOL = 0.5\nMAX_WO = 250.0")
    facts_ = extract_facts(result)
    first = vc(
        mk(
            "F1",
            "OBSERVED",
            "config",
            "TOL is 0.5 and MAX_WO is 250.0.",
            [c(1, 1, 2)],
        )
    )
    second = vc(mk("F2", "OBSERVED", "config", "The limit is 1000.", [c(1, 1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="Gotchas",
                sentences=[
                    SentenceOut(text="TOL is 7 and MAX_WO is used.", ids=["F1"]),
                    SentenceOut(text="MAX_WO is 250 and TOL is 0.5.", ids=["F1"]),
                    SentenceOut(
                        text="There are 2 steps, 0 gaps and 1 retry.", ids=["F1"]
                    ),
                    SentenceOut(
                        text=("DK01, camelCase, fin.cash, MAX_WO, SQL and e.g appear."),
                        ids=["F1"],
                    ),
                    SentenceOut(text="The limit is 1,000.", ids=["F1", "F2"]),
                    SentenceOut(text="DK01 and DK01 appear.", ids=["F1"]),
                ],
            )
        ],
    )
    check = check_document(
        "developer",
        out,
        [first, second],
        result,
        facts_,
        ledger(first, second),
    )
    sentences = check.sections[7].sentences
    assert sentences[0].flags == ("not in cited claims: 7",)
    assert sentences[1].flags == ()
    assert sentences[2].flags == ()
    assert sentences[3].flags == ("not in cited claims: DK01, camelCase, fin.cash",)
    assert sentences[4].flags == ()
    assert sentences[5].flags == ("not in cited claims: DK01",)


def test_manager_sees_numbers_but_not_grounded_identifiers() -> None:
    result = notebook("TOL = 0.5")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "TOL is 0.5 and MAX_WO is set.", [c(1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="What this notebook does",
                sentences=[SentenceOut(text="Everything is 7.", ids=["F1"])],
            )
        ],
    )
    check = check_document("manager", out, [claim], result, facts_, ledger(claim))
    assert check.sections[0].sentences[0].flags == ("not in cited claims: 7",)


def test_manager_code_vocabulary_is_flagged_once() -> None:
    result = notebook("TOL = 0.5")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "The value is set.", [c(1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="The rules it applies",
                sentences=[
                    SentenceOut(
                        text=(
                            "Use `x`, customer_id, camelCase, fin.cash, Spark, "
                            "DataFrame, cell 4 and line 9."
                        ),
                        ids=["F1"],
                    ),
                    SentenceOut(text="Use customer_id.", ids=[]),
                ],
            )
        ],
    )
    check = check_document("manager", out, [claim], result, facts_, ledger(claim))
    assert check.sections[1].sentences[0].flags == (
        "not in cited claims: 4, 9",
        "code identifier: `",
        "code identifier: spark",
        "code identifier: dataframe",
        "code identifier: cell 4",
        "code identifier: line 9",
        "code identifier: customer_id",
        "code identifier: camelCase",
        "code identifier: fin.cash",
    )
    assert check.sections[1].sentences[1].flags == (
        "no claim ids",
        "code identifier: customer_id",
    )


def test_manager_prose_is_not_mistaken_for_code() -> None:
    result = notebook("TOL = 0.5")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "The value is 5.0.", [c(1)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="What this notebook does",
                sentences=[
                    SentenceOut(text="For example, e.g. 5.0 is used.", ids=["F1"])
                ],
            )
        ],
    )
    check = check_document("manager", out, [claim], result, facts_, ledger(claim))
    assert check.sections[0].sentences[0].flags == ()


# --- Coverage and risk pairing ---------------------------------------------


def test_developer_lists_uncovered_and_failed_cells() -> None:
    result = notebook("x = 1", "y = 2", "z = 3")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "x is 1.", [c(1)]))
    out = DocOut(title="T", sections=[])
    check = check_document(
        "developer",
        out,
        [claim],
        result,
        facts_,
        ledger(claim, failed_regions=(FailedRegion((3,), "boom"),)),
    )
    assert check.sections[-1].heading == "Not covered"
    assert [sentence.text for sentence in check.sections[-1].sentences] == [
        "No verified claim covers cells: 2, 3.",
        "Extraction failed for cells: 3.",
    ]


def test_developer_omits_not_covered_when_every_cell_has_a_claim() -> None:
    result = notebook("x = 1", "y = 2")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "config", "x is 1 and y is 2.", [c(1), c(2)]))
    out = DocOut(title="T", sections=[])
    check = check_document("developer", out, [claim], result, facts_, ledger(claim))
    assert all(section.heading != "Not covered" for section in check.sections)


def test_agent_risk_without_matching_check_is_an_issue() -> None:
    result = small_result()
    facts_ = small_facts()
    risk = vc(mk("F6", "INFERRED", "risk", "Raising TOL is risky.", [c(1, 2)]))
    uncited_risk = vc(mk("F7", "INFERRED", "risk", "A second risk.", [c(1, 2)]))
    config = vc(mk("F1", "OBSERVED", "config", "Raising TOL is risky.", [c(1, 2)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="Risky areas",
                sentences=[SentenceOut(text="Raising TOL is risky.", ids=["F6"])],
            )
        ],
    )
    check = check_document(
        "agent",
        out,
        [risk, uncited_risk, config],
        result,
        facts_,
        ledger(risk, uncited_risk, config),
    )
    assert check.issues == ("risk F6 has no matching check",)
    assert check.problems == ("Risky areas: risk F6 has no matching check",)
    assert check.used_ids == ("F6",)


def test_agent_risk_with_a_same_claim_check_is_clean() -> None:
    result = small_result()
    facts_ = small_facts()
    risk = vc(mk("F6", "INFERRED", "risk", "Raising TOL is risky.", [c(1, 2)]))
    out = DocOut(
        title="T",
        sections=[
            SectionOut(
                heading="Risky areas",
                sentences=[SentenceOut(text="Raising TOL is risky.", ids=["F6"])],
            ),
            SectionOut(
                heading="Pre-change checks",
                sentences=[
                    SentenceOut(
                        text="Check the write-off count before changing TOL.",
                        ids=["F6"],
                    )
                ],
            ),
        ],
    )
    check = check_document("agent", out, [risk], result, facts_, ledger(risk))
    assert check.issues == ()
    assert check.problems == ()


def test_agent_static_sections_render_facts_without_the_model() -> None:
    result = small_result()
    facts_ = small_facts()
    out = DocOut(title="T", sections=[])
    check = check_document("agent", out, [], result, facts_, ledger())
    headings = [section.heading for section in check.sections]
    assert headings == [*AGENT_FINAL_ORDER, LIMITS_HEADING]
    by_heading = {section.heading: section for section in check.sections}
    assert [s.text for s in by_heading["Unused definitions"].sentences] == [
        "helper is defined and never referenced."
    ]
    assert [s.text for s in by_heading["Invisible dependencies"].sentences] == [
        "early_call is called but not defined in this notebook.",
        (
            "notify_ar is called but not defined in this notebook. "
            "It may come from a %run of ../shared/utils."
        ),
        (
            "log_run is called but not defined in this notebook. "
            "It may come from a %run of ../shared/utils."
        ),
    ]
    constants = [
        s.text for s in by_heading["Constants and where they can change"].sentences
    ]
    assert (
        "TOL is assigned more than once with different values: 0.5, then 5.0."
        in constants
    )
    assert "LIMIT is assigned 10 and never read anywhere in the notebook." in constants
    mutable = [
        s.text for s in by_heading["Mutable state (globals and tables)"].sentences
    ]
    assert (
        "The global bal is modified in place inside these functions: apply." in mutable
    )
    assert "Writes table cat.tbl with mode overwrite." in mutable
    assert "Writes table cat.raw with mode not set." in mutable
    assert all(
        sentence.source == "static"
        for section in check.sections
        for sentence in section.sentences
    )
    limits = by_heading[LIMITS_HEADING].sentences
    assert len(limits) == len(facts_.limitations)
    assert all(sentence.source == "static" for sentence in limits)


def test_static_sections_handle_unresolved_and_missing_details() -> None:
    result = notebook("known = 1")
    edge = StaticFacts(
        functions=(FunctionFact("known", c(1, 1, 1), ()),),
        constants=(
            ConstantFact(
                "A",
                (Binding("A", "assign", c(1, 1), False, None),),
                False,
                False,
                False,
            ),
            ConstantFact("B", (), True, True, True),
            ConstantFact(
                "C",
                (Binding("C", "assign", c(1, 4), True, None),),
                True,
                True,
                True,
            ),
        ),
        widgets=(),
        tables=(
            TableFact("read", "spark.read", "src", None, None, None, c(1, 2)),
            TableFact("write", "saveAsTable", None, None, None, None, c(1, 3)),
            TableFact(
                "write", "saveAsTable", None, "some_expr", None, "append", c(1, 5)
            ),
        ),
        runs=(),
        unresolved_calls=(UnresolvedCall("ghost", (), ()),),
        unused_functions=("known", "missing"),
        mutations=(
            Mutation("f", "g", "store", c(2, 1)),
            Mutation("f", "g", "store", c(2, 1)),
        ),
        parse_issues=(),
        limitations=(),
    )
    check = check_document(
        "agent", DocOut(title="T", sections=[]), [], result, edge, ledger()
    )
    by_heading = {section.heading: section for section in check.sections}
    constants = by_heading["Constants and where they can change"].sentences
    assert [sentence.text for sentence in constants] == [
        "B is assigned more than once with different values: .",
        "B is assigned ? and never read anywhere in the notebook.",
        "C is assigned more than once with different values: .",
        "C is assigned ? and never read anywhere in the notebook.",
    ]
    assert constants[1].cites == ()
    assert constants[3].cites == (c(1, 4),)
    mutable = by_heading["Mutable state (globals and tables)"].sentences
    assert [sentence.text for sentence in mutable] == [
        "The global g is modified in place inside these functions: f.",
        "Writes table an unresolved target with mode not set.",
        "Writes table some_expr with mode append.",
    ]
    assert mutable[0].cites == (c(2, 1),)
    assert [s.text for s in by_heading["Unused definitions"].sentences] == [
        "known is defined and never referenced."
    ]
    invisible = by_heading["Invisible dependencies"].sentences
    assert [sentence.text for sentence in invisible] == [
        "ghost is called but not defined in this notebook."
    ]
    assert invisible[0].cites == ()
    assert by_heading[LIMITS_HEADING].sentences == ()


# --- Cross-document consistency --------------------------------------------


def test_consistency_reports_wrong_and_partial_values() -> None:
    facts_ = StaticFacts(
        functions=(),
        constants=(
            ConstantFact(
                "TOL",
                (
                    Binding("TOL", "assign", c(1, 1), True, "0.5"),
                    Binding("TOL", "assign", c(2, 1), True, "0.5"),
                    Binding("TOL", "assign", c(3, 1), True, "5.0"),
                    Binding("TOL", "aug", c(4, 1), False, None),
                    Binding("TOL", "assign", c(5, 1), True, None),
                    Binding("TOL", "assign", c(6, 1), True, "'abc'"),
                ),
                True,
                True,
                False,
            ),
            ConstantFact(
                "NAME",
                (Binding("NAME", "assign", c(1, 1), True, "'x'"),),
                False,
                False,
                False,
            ),
            ConstantFact(
                "FLAT",
                (Binding("FLAT", "assign", c(1, 1), True, "3"),),
                False,
                False,
                False,
            ),
        ),
        widgets=(),
        tables=(),
        runs=(),
        unresolved_calls=(),
        unused_functions=(),
        mutations=(),
        parse_issues=(),
        limitations=(),
    )
    docs = (
        document(
            "manager",
            dsection(
                "The rules it applies",
                DocSentence("TOL is 7 and again 7.", (), (), "claims", ()),
            ),
        ),
        document(
            "developer",
            dsection(
                "Parameters",
                DocSentence("TOL is 0.5 and 7.", (), (), "claims", ()),
            ),
        ),
        document(
            "agent",
            dsection(
                "Constants and where they can change",
                DocSentence("TOL is 0.5 and 5.0.", (), (), "claims", ()),
            ),
        ),
        document(
            "agent",
            dsection(
                "Gotchas",
                DocSentence("FLAT is 3.", (), (), "claims", ()),
            ),
        ),
        document(
            "manager",
            dsection(
                "What this notebook does",
                DocSentence("Nothing here.", (), (), "claims", ()),
            ),
        ),
    )
    assert check_consistency(docs, facts_) == (
        "manager: TOL stated as 7, bound values are 0.5, 5.0",
        "manager: TOL mentions only no numbers; also bound: 0.5, 5.0",
        "developer: TOL stated as 7, bound values are 0.5, 5.0",
        "developer: TOL mentions only 0.5; also bound: 5.0",
    )
    assert check_consistency((), facts_) == ()
    assert check_consistency(docs, extract_facts(notebook("x = 1"))) == ()


# --- Rendering and serialisation -------------------------------------------


def test_to_dict_is_json_safe() -> None:
    sentence = DocSentence("Text.", ("F1",), (c(1), c(2, 1, 3)), "claims", ("flag",))
    doc = document(
        "developer",
        dsection("Purpose", sentence),
        issues=("issue",),
        error=None,
        used_ids=("F1",),
    )
    assert doc.to_dict() == {
        "audience": "developer",
        "title": "Document",
        "sections": [
            {
                "heading": "Purpose",
                "sentences": [
                    {
                        "text": "Text.",
                        "ids": ["F1"],
                        "cites": [
                            {"cell": 1, "start": 1, "end": 1},
                            {"cell": 2, "start": 1, "end": 3},
                        ],
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
    assert json.loads(json.dumps(doc.to_dict())) == doc.to_dict()

    docs = DocumentSet(
        manager=doc,
        developer=doc,
        agent=doc,
        issues=("issue",),
        summary={
            "total": 2,
            "verified": 1,
            "flagged": 1,
            "dropped": 0,
            "used": 1,
        },
        notes=("note",),
    )
    raw = docs.to_dict()
    assert raw["summary"] == docs.summary
    assert raw["notes"] == ["note"]
    assert json.loads(json.dumps(raw)) == raw


def test_render_markdown_keeps_everything_visible() -> None:
    doc = document(
        "developer",
        dsection("Purpose"),
        dsection(
            "Gotchas",
            DocSentence("Plain text.", ("F1",), (c(1),), "claims", ()),
            DocSentence("Flagged text.", (), (), "unsourced", ("no claim ids",)),
            DocSentence("Bare.", (), (), "static", ()),
            DocSentence(
                "Many chips.",
                ("F1",),
                tuple(c(number) for number in range(1, 7)),
                "claims",
                (),
            ),
        ),
        issues=("one issue",),
        error="boom",
        used_ids=("F1",),
    )
    claim = vc(mk("F1", "OBSERVED", "config", "Plain text.", [c(1)]))
    rendered = render_markdown(
        doc,
        ledger(
            claim,
            vc(mk("F2", "OBSERVED", "config", "Other.", [c(1)]), "flagged"),
            vc(mk("F3", "OBSERVED", "config", "Gone.", [c(1)]), "dropped"),
        ),
    )
    assert rendered.startswith("# Document\n\n")
    assert "> Document generation failed: boom" in rendered
    assert "## Purpose\n\n_No verified claims for this section._" in rendered
    assert "Plain text. [C1:L1]" in rendered
    assert "[flagged: no claim ids] Flagged text." in rendered
    assert "\nBare.\n" in rendered
    assert "Many chips. [C1:L1] [C2:L1] [C3:L1] [C4:L1] +2 more" in rendered
    assert "## Issues\n\n- one issue\n" in rendered
    assert (
        "Built from 1 verified claims; 2 claims were flagged or dropped and not used."
    ) in rendered
    assert rendered.endswith("\n")


def test_render_markdown_without_ledger_has_no_footer() -> None:
    doc = document("manager")
    rendered = render_markdown(doc)
    assert rendered == "# Document\n\n---\n"


# --- One retry, failures, and the three-document set -----------------------


def _single_claim() -> tuple[ParseResult, VerifiedLedger, StaticFacts, VerifiedClaim]:
    result = notebook("TOL = 0.5")
    facts_ = extract_facts(result)
    claim = vc(mk("F1", "OBSERVED", "business_rule", "TOL is 0.5.", [c(1)]))
    return result, ledger(claim), facts_, claim


def _manager_doc(*sentences: dict[str, Any], title: str = "T") -> str:
    return reply(title, sec("What this notebook does", *sentences))


def test_clean_reply_costs_one_call_and_reports_omissions() -> None:
    result, verified, facts_, _ = _single_claim()
    extra = vc(mk("F2", "OBSERVED", "business_rule", "A limit exists.", [c(1, 3)]))
    verified = ledger(*verified.items, extra)
    llm = FakeLLM([_manager_doc(sent("TOL is 0.5.", "F1"))])
    doc = compose_document("manager", result, verified, facts_, llm, max_claims=1)
    assert len(llm.calls) == 1
    assert doc.error is None
    assert doc.used_ids == ("F1",)
    assert doc.issues == ("1 claims omitted (limit 1)",)
    assert doc.sections[0].sentences[0].text == "TOL is 0.5."


def test_flagged_reply_is_repaired_once() -> None:
    result, verified, facts_, _ = _single_claim()
    llm = FakeLLM(
        [
            _manager_doc(sent("TOL is 9.", "F1")),
            _manager_doc(sent("TOL is 0.5.", "F1"), title="Fixed"),
        ]
    )
    doc = compose_document("manager", result, verified, facts_, llm)
    assert len(llm.calls) == 2
    retry = llm.calls[1]["messages"]
    assert retry[0]["role"] == "system"
    assert retry[1]["role"] == "user"
    assert retry[2]["role"] == "assistant"
    assert retry[3]["role"] == "user"
    assert "not in cited claims: 9" in retry[3]["content"]
    assert retry[3]["content"].startswith("Fix these problems")
    echoed = json.loads(retry[2]["content"])
    assert echoed["sections"][0]["sentences"][0]["text"] == "TOL is 9."
    assert doc.title == "Fixed"
    assert doc.sections[0].sentences[0].flags == ()
    assert doc.issues == ()


def test_second_bad_reply_is_final_and_keeps_flags() -> None:
    result, verified, facts_, _ = _single_claim()
    long_text = (
        "TOL is 9 and this sentence is deliberately long enough to pass "
        "sixty characters."
    )
    header = "What this notebook does"
    problem = f'{header}: "{long_text[:60]}..." -> not in cited claims: 9'
    llm = FakeLLM(
        [
            _manager_doc(*[sent("TOL is 9.", "F1") for _ in range(11)]),
            _manager_doc(sent(long_text, "F1")),
        ]
    )
    doc = compose_document("manager", result, verified, facts_, llm)
    assert len(llm.calls) == 2
    retry = llm.calls[1]["messages"][3]["content"]
    assert retry.count("\n- ") == 10
    assert doc.sections[0].sentences[0].flags == ("not in cited claims: 9",)
    assert doc.sections[0].sentences[0].text == long_text

    llm = FakeLLM(
        [_manager_doc(sent(long_text, "F1")), _manager_doc(sent(long_text, "F1"))]
    )
    doc = compose_document("manager", result, verified, facts_, llm)
    retry = llm.calls[1]["messages"][3]["content"]
    assert problem in retry
    assert doc.sections[0].sentences[0].flags == ("not in cited claims: 9",)


def test_no_selected_claims_makes_no_call() -> None:
    result, _, facts_, _ = _single_claim()
    flagged = vc(mk("F1", "OBSERVED", "config", "TOL is 0.5.", [c(1)]), "flagged")
    llm = FakeLLM([])
    doc = compose_document("manager", result, ledger(flagged), facts_, llm)
    assert llm.calls == []
    assert doc.title == "Notebook summary"
    assert [section.heading for section in doc.sections] == list(MANAGER_HEADINGS)
    assert all(not section.sentences for section in doc.sections)
    assert doc.used_ids == ()
    assert doc.error is None


@pytest.mark.parametrize(
    "error",
    [
        TruncatedOutput("cut off"),
        StructuredOutputError(3, "bad json", "{}"),
        LLMTransientError("temporary"),
        LLMResponseError("provider shape"),
    ],
)
def test_provider_failure_keeps_static_sections_and_error(
    error: Exception,
) -> None:
    result = small_result()
    facts_ = small_facts()
    claim = vc(mk("F1", "OBSERVED", "config", "TOL is 0.5.", [c(1, 2)]))
    llm = FakeLLM([error])
    doc = compose_document("agent", result, ledger(claim), facts_, llm)
    assert len(llm.calls) == 1
    assert doc.error is not None
    if isinstance(error, TruncatedOutput):
        assert doc.error.endswith("(increase max_output_tokens in config.toml)")
    else:
        assert "max_output_tokens" not in doc.error
    by_heading = {section.heading: section for section in doc.sections}
    assert [s.text for s in by_heading["Unused definitions"].sentences] == [
        "helper is defined and never referenced."
    ]
    assert by_heading[LIMITS_HEADING].sentences
    assert all(not by_heading[heading].sentences for heading in AGENT_MODEL_HEADINGS)
    assert doc.used_ids == ()


def test_failure_on_retry_keeps_the_first_result_out() -> None:
    result, verified, facts_, _ = _single_claim()
    llm = FakeLLM(
        [
            _manager_doc(sent("TOL is 9.", "F1")),
            LLMTransientError("down"),
        ]
    )
    doc = compose_document("manager", result, verified, facts_, llm)
    assert len(llm.calls) == 2
    assert doc.error == "down"
    assert all(not section.sentences for section in doc.sections)
    assert doc.used_ids == ()


@pytest.mark.parametrize(
    "error", [LLMAuthError("bad key"), LLMRequestError("bad request")]
)
def test_authentication_and_request_errors_propagate(error: Exception) -> None:
    result, verified, facts_, _ = _single_claim()
    llm = FakeLLM([error])
    with pytest.raises(type(error)):
        compose_document("manager", result, verified, facts_, llm)


def test_real_notebook_selection_and_prompts() -> None:
    result = real_result()
    verified = real_ledger()
    facts_ = extract_facts(result)

    manager, omitted = select_claims(verified, "manager", 120)
    manager_ids = [item.claim.id for item in manager]
    assert manager_ids == ["F6", "F4"]
    assert omitted == 0

    developer, omitted = select_claims(verified, "developer", 120)
    assert [item.claim.id for item in developer] == [
        "F1",
        "F3",
        "F2",
        "F6",
        "F4",
        "F5",
    ]
    assert omitted == 0

    manager_user = build_messages("manager", manager, facts_)[1]["content"]
    assert "TOL is reassigned" not in manager_user
    assert "CONSTANT NOTES" not in manager_user

    developer_user = build_messages("developer", developer, facts_)[1]["content"]
    assert "F6 [INFERRED, risk]" in developer_user
    assert "CONSTANT NOTES" in developer_user
    assert "TOL: assigned 0.5, then 5.0" in developer_user
    assert "C4" not in developer_user


def test_compose_documents_full_run() -> None:
    result = real_result()
    verified = real_ledger()
    facts_ = extract_facts(result)
    llm = FakeLLM([manager_reply(), developer_reply(), agent_reply()])
    docs = compose_documents(result, verified, facts_, llm)

    assert len(llm.calls) == 3
    assert docs.summary == {
        "total": 6,
        "verified": 6,
        "flagged": 0,
        "dropped": 0,
        "used": 4,
    }
    assert docs.notes == ("note",)
    assert docs.issues == ("developer: TOL mentions only 0.5; also bound: 5.0",)
    assert docs.manager.used_ids == ("F4",)
    assert docs.developer.used_ids == ("F1", "F5")
    assert docs.agent.used_ids == ("F6",)
    assert docs.manager.title == "Manager view"
    assert docs.developer.title == "Developer guide"
    assert docs.agent.title == "Agent brief"

    agent_sections = {section.heading: section for section in docs.agent.sections}
    constants = agent_sections["Constants and where they can change"].sentences
    tol = next(sentence for sentence in constants if sentence.text.startswith("TOL"))
    assert tol.cites == (c(4, 1), c(29, 4))
    risky = agent_sections["Risky areas"].sentences
    assert risky[0].cites == (c(29, 4),)

    developer_sections = {
        section.heading: section for section in docs.developer.sections
    }
    not_covered = developer_sections["Not covered"].sentences
    cells = {
        int(part)
        for part in not_covered[0].text.split(": ", 1)[1].rstrip(".").split(", ")
    }
    assert {4, 29, 43, 45}.isdisjoint(cells)
    assert 1 not in cells
    assert 2 in cells and 47 in cells

    manager_md = render_markdown(docs.manager, verified)
    assert "[C43:L11]" in manager_md
    agent_md = render_markdown(docs.agent, verified)
    assert "[C29:L4]" in agent_md
    serialised = docs.to_dict()
    assert json.loads(json.dumps(serialised)) == serialised

    again = compose_documents(
        result,
        verified,
        facts_,
        FakeLLM([manager_reply(), developer_reply(), agent_reply()]),
    )
    assert again == docs


def test_one_document_failure_does_not_stop_the_others() -> None:
    result = real_result()
    verified = real_ledger()
    facts_ = extract_facts(result)
    llm = FakeLLM([TruncatedOutput("cut off"), developer_reply(), agent_reply()])
    docs = compose_documents(result, verified, facts_, llm)
    assert len(llm.calls) == 3
    assert docs.manager.error is not None
    assert "max_output_tokens" in docs.manager.error
    assert docs.developer.error is None
    assert docs.agent.error is None
    assert docs.summary["used"] == 3
    assert any(
        section.sentences
        for section in docs.developer.sections
        if section.heading == "Parameters"
    )
