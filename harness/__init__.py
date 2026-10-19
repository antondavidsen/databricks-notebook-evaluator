"""Public API for parsing Databricks Python notebooks and resolving citations."""

from typing import TYPE_CHECKING

from .citations import (
    Citation,
    find_citations,
    format_citation,
    numbered_listing,
    parse_citation,
    validate_citation,
)
from .facts import StaticFacts, extract_facts, render_facts_for_prompt
from .models import (
    Cell,
    CellKind,
    CitationError,
    Excerpt,
    Line,
    NotebookError,
    NotebookFormatError,
    ParseResult,
    UnsupportedFormatError,
)

if TYPE_CHECKING:
    from .parser import parse_file, parse_text


def __getattr__(name: str) -> object:
    """Import the parser lazily so ``python -m harness.parser`` stays clean."""
    if name in ("parse_file", "parse_text"):
        from .parser import parse_file, parse_text

        return {"parse_file": parse_file, "parse_text": parse_text}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Cell",
    "CellKind",
    "Citation",
    "CitationError",
    "Excerpt",
    "Line",
    "NotebookError",
    "NotebookFormatError",
    "ParseResult",
    "StaticFacts",
    "UnsupportedFormatError",
    "extract_facts",
    "find_citations",
    "format_citation",
    "numbered_listing",
    "parse_citation",
    "parse_file",
    "parse_text",
    "render_facts_for_prompt",
    "validate_citation",
]
