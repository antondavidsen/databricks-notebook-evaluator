"""Citation syntax shared by the LLM prompt, verifier, and line highlighter."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .models import CitationError, Excerpt, ParseResult

_CITATION_RE = re.compile(r"^(\[)?[Cc](\d+):[Ll](\d+)(?:-(\d+))?(\])?$")
_FIND_RE = re.compile(r"\[C(\d+):L(\d+)(?:-(\d+))?\]")


@dataclass(frozen=True, slots=True)
class Citation:
    """A 1-based, inclusive cell-relative range."""

    cell: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.cell < 1:
            raise CitationError(
                f"citation cell must be >= 1, got {self.cell}; "
                "cells are numbered from 1"
            )
        if self.start < 1:
            raise CitationError(
                f"citation start line must be >= 1, got {self.start}; "
                "lines are numbered from 1"
            )
        if self.end < self.start:
            raise CitationError(
                f"citation end line {self.end} is before start line "
                f"{self.start}; expected start <= end"
            )


def format_citation(citation: Citation) -> str:
    """Render the canonical form the prompt asks the model to produce."""
    if citation.start == citation.end:
        return f"[C{citation.cell}:L{citation.start}]"
    return f"[C{citation.cell}:L{citation.start}-{citation.end}]"


def parse_citation(text: str) -> Citation:
    """Parse one citation, tolerating wrappers, case, and whitespace."""
    stripped = text.strip()
    match = _CITATION_RE.fullmatch(stripped)
    if match is None or bool(match.group(1)) != bool(match.group(5)):
        raise CitationError(
            f"invalid citation {text!r}; expected e.g. [C3:L10-12] or C3:L10"
        )
    start = int(match.group(3))
    end = int(match.group(4)) if match.group(4) is not None else start
    try:
        return Citation(cell=int(match.group(2)), start=start, end=end)
    except CitationError as exc:
        raise CitationError(f"invalid citation {text!r}: {exc}") from exc


def find_citations(text: str) -> list[tuple[Citation, int, int]]:
    """Find bracketed citations in prose, skipping invalid ones by design."""
    found: list[tuple[Citation, int, int]] = []
    for match in _FIND_RE.finditer(text):
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) is not None else start
        try:
            citation = Citation(cell=int(match.group(1)), start=start, end=end)
        except CitationError:
            continue
        found.append((citation, match.start(), match.end()))
    return found


def validate_citation(
    result: ParseResult,
    citation: Citation,
    max_span: int | None = None,
    context: int = 0,
) -> Excerpt:
    """Resolve a citation against a notebook before any UI trusts it."""
    cell = result.cell(citation.cell)
    if not cell.lines:
        raise CitationError(
            f"citation {format_citation(citation)} targets empty cell "
            f"{citation.cell}, which has no lines"
        )
    if citation.end > len(cell.lines):
        raise CitationError(
            f"citation {format_citation(citation)} exceeds cell {citation.cell}, "
            f"which has {len(cell.lines)} lines "
            f"(valid range L1-L{len(cell.lines)})"
        )
    span = citation.end - citation.start + 1
    if max_span is not None and span > max_span:
        raise CitationError(
            f"citation {format_citation(citation)} spans {span} lines, "
            f"exceeding max_span={max_span}"
        )
    return result.excerpt(citation.cell, citation.start, citation.end, context=context)


def numbered_listing(
    result: ParseResult, cell_numbers: Iterable[int] | None = None
) -> str:
    """Render cells in the exact shape the prompt and citation grammar share."""
    numbers = range(1, len(result.cells) + 1) if cell_numbers is None else cell_numbers
    blocks: list[str] = []
    for number in numbers:
        cell = result.cell(number)
        if not cell.lines:
            blocks.append(f"### C{number} [{cell.kind.value}]")
            continue
        header = (
            f"### C{number} [{cell.kind.value}] file lines "
            f"{cell.first_abs}-{cell.last_abs}"
        )
        body = [f"L{line.rel_no}| {line.text}" for line in cell.lines]
        blocks.append("\n".join([header, *body]))
    return "\n\n".join(blocks)
