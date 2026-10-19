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
from .compose import (
    Document,
    DocumentSet,
    compose_document,
    compose_documents,
    render_markdown,
)
from .config import load_config, load_dotenv
from .extract import Claim, Ledger, extract_claims, plan_chunks
from .facts import StaticFacts, extract_facts, render_facts_for_prompt
from .llm import FakeLLM, LLMClient, make_client
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
from .verify import VerifiedClaim, VerifiedLedger, verify_claim, verify_ledger

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
    "Claim",
    "Document",
    "DocumentSet",
    "Excerpt",
    "FakeLLM",
    "LLMClient",
    "Ledger",
    "Line",
    "NotebookError",
    "NotebookFormatError",
    "ParseResult",
    "StaticFacts",
    "UnsupportedFormatError",
    "VerifiedClaim",
    "VerifiedLedger",
    "compose_document",
    "compose_documents",
    "extract_claims",
    "extract_facts",
    "find_citations",
    "format_citation",
    "load_config",
    "load_dotenv",
    "make_client",
    "numbered_listing",
    "parse_citation",
    "parse_file",
    "parse_text",
    "plan_chunks",
    "render_facts_for_prompt",
    "render_markdown",
    "validate_citation",
    "verify_claim",
    "verify_ledger",
]
