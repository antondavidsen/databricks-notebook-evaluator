from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from harness.models import CitationError, ParseResult
from harness.parser import parse_file, parse_text

HEADER = "# Databricks notebook source"
SEP = "# COMMAND ----------"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cash_application_engine.py"


def real_notebook() -> ParseResult:
    if not FIXTURE.exists():
        pytest.fail(
            "copy the attached notebook to tests/fixtures/cash_application_engine.py"
        )
    return parse_file(FIXTURE)


def basic_notebook() -> ParseResult:
    return parse_text(
        "\n".join([HEADER, "x = 1", SEP, "y = 2", SEP, "z = 3"]) + "\n",
        "py",
    )


def blank_heavy_notebook() -> ParseResult:
    return parse_text(
        "\n".join(
            [
                HEADER,
                "",
                "a = 1",
                "",
                SEP,
                "",
                SEP,
                "",
                "b = 2",
                "",
            ]
        )
        + "\n",
        "py",
    )


def notebook_for(case: str) -> ParseResult:
    if case == "basic":
        return basic_notebook()
    if case == "blank_heavy":
        return blank_heavy_notebook()
    if case == "real":
        return real_notebook()
    raise AssertionError(f"unknown case {case}")


@pytest.mark.parametrize("case", ["basic", "blank_heavy", "real"])
def test_rel_to_abs_and_abs_to_rel_are_inverse_on_every_content_line(
    case: str,
) -> None:
    result = notebook_for(case)
    checked = 0
    for cell in result.cells:
        for line in cell.lines:
            assert result.abs_to_cell_rel(line.abs_no) == (cell.number, line.rel_no)
            assert result.cell_rel_to_abs(cell.number, line.rel_no) == line.abs_no
            assert result.flat_lines[line.abs_no - 1] == line.raw
            checked += 1
    assert checked == sum(len(cell.lines) for cell in result.cells)
    assert checked > 0


def test_every_file_line_is_accounted_for_exactly_once() -> None:
    result = blank_heavy_notebook()

    content = {line.abs_no for cell in result.cells for line in cell.lines}
    separators = {
        abs_no for abs_no, raw in enumerate(result.flat_lines, start=1) if raw == SEP
    }
    header = {1}
    blank_without_cell = {
        abs_no
        for abs_no, raw in enumerate(result.flat_lines, start=1)
        if raw.strip() == "" and abs_no not in content
    }

    assert len(content) + len(header) + len(separators) + len(
        blank_without_cell
    ) == len(result.flat_lines)
    assert content | header | separators | blank_without_cell == set(
        range(1, len(result.flat_lines) + 1)
    )
    for abs_no in header | separators | blank_without_cell:
        assert result.abs_to_cell_rel(abs_no) is None


def test_out_of_range_lookups_raise_citation_error() -> None:
    result = basic_notebook()
    calls: list[Callable[[], object]] = [
        lambda: result.cell(0),
        lambda: result.cell(4),
        lambda: result.line(0, 1),
        lambda: result.line(4, 1),
        lambda: result.line(1, 0),
        lambda: result.line(1, 2),
        lambda: result.cell_rel_to_abs(1, 99),
        lambda: result.abs_to_cell_rel(0),
        lambda: result.abs_to_cell_rel(999),
    ]
    for call in calls:
        with pytest.raises(CitationError):
            call()


def test_lookup_rejects_non_integral_numbers() -> None:
    result = basic_notebook()
    calls: list[Callable[[], object]] = [
        lambda: result.cell(cast(int, "1")),
        lambda: result.cell(True),
        lambda: result.line(1, cast(int, "1")),
        lambda: result.line(1, True),
        lambda: result.abs_to_cell_rel(cast(int, "1")),
        lambda: result.abs_to_cell_rel(True),
    ]

    for call in calls:
        with pytest.raises(CitationError, match="must be an integer"):
            call()


def test_lookup_in_empty_cell_raises() -> None:
    result = parse_text("\n".join([HEADER, SEP, SEP]) + "\n", "py")

    assert result.cells[0].kind.value == "empty"
    with pytest.raises(CitationError, match="empty"):
        result.line(1, 1)


def test_excerpt_context_is_clamped_to_the_same_cell() -> None:
    text = (
        "\n".join(
            [
                HEADER,
                "c1l1",
                "c1l2",
                "c1l3",
                "c1l4",
                "c1l5",
                SEP,
                "c2l1",
                "c2l2",
                "c2l3",
                "c2l4",
                "c2l5",
                SEP,
                "c3l1",
                "c3l2",
                "c3l3",
                "c3l4",
                "c3l5",
            ]
        )
        + "\n"
    )
    result = parse_text(text, "py")

    excerpt = result.excerpt(2, 3, 4, context=2)
    assert (excerpt.cell_no, excerpt.start, excerpt.end) == (2, 3, 4)
    assert (excerpt.abs_start, excerpt.abs_end) == (10, 11)
    assert [line.rel_no for line in excerpt.lines] == [3, 4]
    assert [line.rel_no for line in excerpt.before] == [1, 2]
    assert [line.rel_no for line in excerpt.after] == [5]
    neighbours = excerpt.before + excerpt.lines + excerpt.after
    assert {line.cell_no for line in neighbours} == {2}

    entire = result.excerpt(2, 2, 4, context=99)
    assert [line.rel_no for line in entire.before] == [1]
    assert [line.rel_no for line in entire.after] == [5]

    at_start = result.excerpt(2, 1, 2, context=99)
    assert at_start.before == ()
    assert [line.rel_no for line in at_start.after] == [3, 4, 5]

    at_end = result.excerpt(2, 4, 5, context=99)
    assert [line.rel_no for line in at_end.before] == [1, 2, 3]
    assert at_end.after == ()


def test_excerpt_without_context_has_no_neighbours() -> None:
    result = basic_notebook()

    excerpt = result.excerpt(1, 1, 1)
    assert excerpt.before == ()
    assert excerpt.after == ()
    assert excerpt.lines[0].raw == "x = 1"
    assert excerpt.abs_start == excerpt.abs_end == 2


def test_excerpt_rejects_negative_context() -> None:
    result = basic_notebook()

    with pytest.raises(CitationError, match="context"):
        result.excerpt(1, 1, 1, context=-1)


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 1), (1, 0), (2, 1)],
)
def test_excerpt_rejects_invalid_ranges(start: int, end: int) -> None:
    result = basic_notebook()

    with pytest.raises(CitationError, match=r"expected 1 <= start <= end"):
        result.excerpt(1, start, end)


def test_excerpt_rejects_range_past_cell_end() -> None:
    result = basic_notebook()

    with pytest.raises(CitationError, match=r"exceeds the valid range 1\.\.1"):
        result.excerpt(1, 1, 2)


def test_excerpt_rejects_any_range_in_an_empty_cell() -> None:
    result = parse_text("\n".join([HEADER, SEP, SEP]) + "\n", "py")

    with pytest.raises(CitationError, match="is empty"):
        result.excerpt(1, 1, 1)
