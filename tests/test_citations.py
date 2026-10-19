from __future__ import annotations

import pytest
from harness.citations import (
    Citation,
    find_citations,
    format_citation,
    numbered_listing,
    parse_citation,
    validate_citation,
)
from harness.models import CitationError, ParseResult
from harness.parser import parse_text

HEADER = "# Databricks notebook source"
SEP = "# COMMAND ----------"


def small_notebook() -> ParseResult:
    text = (
        "\n".join(
            [
                HEADER,
                "a = 1",
                "b = 2",
                "c = 3",
                SEP,
                "# MAGIC %sql",
                "# MAGIC select 1",
            ]
        )
        + "\n"
    )
    return parse_text(text, "py")


def test_format_citation_range_and_single_line() -> None:
    assert format_citation(Citation(33, 4, 9)) == "[C33:L4-9]"
    assert format_citation(Citation(33, 4, 4)) == "[C33:L4]"


@pytest.mark.parametrize(
    "text",
    ["[C33:L4-9]", "C33:L4-9", "  c33:l4-9  ", "[c33:L4-9]"],
)
def test_parse_citation_accepts_wrappers_case_and_whitespace(text: str) -> None:
    assert parse_citation(text) == Citation(33, 4, 9)


@pytest.mark.parametrize("text", ["[C33:L4]", "C33:L4", " c33:l4 "])
def test_parse_citation_single_line(text: str) -> None:
    assert parse_citation(text) == Citation(33, 4, 4)


def test_citation_round_trip() -> None:
    for citation in (Citation(1, 1, 1), Citation(7, 2, 8), Citation(12, 3, 3)):
        assert parse_citation(format_citation(citation)) == citation


@pytest.mark.parametrize(
    "text",
    [
        "C0:L1",
        "C1:L0",
        "C1:L5-2",
        "C:L1",
        "garbage",
        "[C1:L1",
        "C1:L1]",
        "[C1:L1]]",
        "",
    ],
)
def test_parse_citation_rejects_invalid_forms(text: str) -> None:
    with pytest.raises(CitationError):
        parse_citation(text)


@pytest.mark.parametrize(
    ("cell", "start", "end"),
    [(0, 1, 1), (1, 0, 1), (1, 5, 2), (-1, 1, 1)],
)
def test_citation_constructor_validates(cell: int, start: int, end: int) -> None:
    with pytest.raises(CitationError):
        Citation(cell, start, end)


def test_find_citations_in_prose_skips_invalid_ones() -> None:
    text = "([C1:L1-2]), then [C2:L3]. Also [C3:L4]! Invalid [C0:L1] and [C2:L9-2]."

    found = find_citations(text)

    assert [(c.cell, c.start, c.end) for c, _, _ in found] == [
        (1, 1, 2),
        (2, 3, 3),
        (3, 4, 4),
    ]
    brackets = ["[C1:L1-2]", "[C2:L3]", "[C3:L4]"]
    for (citation, start, end), bracket in zip(found, brackets, strict=True):
        assert text[start:end] == bracket
        assert parse_citation(bracket) == citation


def test_find_citations_returns_empty_list_when_none_exist() -> None:
    assert find_citations("no citations here") == []


def test_validate_citation_returns_contextual_excerpt() -> None:
    result = small_notebook()

    excerpt = validate_citation(result, Citation(1, 2, 3), context=1)

    assert (excerpt.cell_no, excerpt.start, excerpt.end) == (1, 2, 3)
    assert (excerpt.abs_start, excerpt.abs_end) == (3, 4)
    assert [line.raw for line in excerpt.lines] == ["b = 2", "c = 3"]
    assert [line.raw for line in excerpt.before] == ["a = 1"]
    assert excerpt.after == ()


def test_validate_citation_rejects_missing_cell() -> None:
    result = small_notebook()

    with pytest.raises(CitationError, match="cell 3"):
        validate_citation(result, Citation(3, 1, 1))


def test_validate_citation_rejects_range_past_cell_end() -> None:
    result = small_notebook()

    with pytest.raises(CitationError, match="exceeds cell 1"):
        validate_citation(result, Citation(1, 4, 4))


def test_validate_citation_rejects_range_longer_than_max_span() -> None:
    result = small_notebook()

    with pytest.raises(CitationError, match="max_span=2"):
        validate_citation(result, Citation(1, 1, 3), max_span=2)


def test_validate_citation_accepts_span_equal_to_max_span() -> None:
    result = small_notebook()

    excerpt = validate_citation(result, Citation(1, 1, 3), max_span=3)
    assert len(excerpt.lines) == 3


def test_validate_citation_rejects_empty_cell() -> None:
    result = parse_text("\n".join([HEADER, SEP, SEP]) + "\n", "py")

    with pytest.raises(CitationError, match="empty cell"):
        validate_citation(result, Citation(1, 1, 1))


def listing_notebook() -> ParseResult:
    text = (
        "\n".join(
            [
                HEADER,
                "x = 1",
                "y = 2",
                SEP,
                SEP,
                "# MAGIC %sql",
                "# MAGIC select 1",
            ]
        )
        + "\n"
    )
    return parse_text(text, "py")


def test_numbered_listing_exact_format_including_empty_cell() -> None:
    result = listing_notebook()

    assert numbered_listing(result) == (
        "### C1 [python] file lines 2-3\n"
        "L1| x = 1\n"
        "L2| y = 2\n"
        "\n"
        "### C2 [empty]\n"
        "\n"
        "### C3 [sql] file lines 6-7\n"
        "L1| %sql\n"
        "L2| select 1"
    )


def test_numbered_listing_uses_text_not_raw_and_respects_order() -> None:
    result = listing_notebook()

    listing = numbered_listing(result, [3, 1])

    assert listing == (
        "### C3 [sql] file lines 6-7\n"
        "L1| %sql\n"
        "L2| select 1\n"
        "\n"
        "### C1 [python] file lines 2-3\n"
        "L1| x = 1\n"
        "L2| y = 2"
    )
    assert "# MAGIC" not in listing


def test_numbered_listing_unknown_cell_raises() -> None:
    result = listing_notebook()

    with pytest.raises(CitationError, match="cell 9"):
        numbered_listing(result, [9])
