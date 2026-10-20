"""Lazy public facade for the Databricks notebook harness.

Importing ``harness`` must stay cheap: a leaf module such as ``harness.models``
or the ``python -m harness.parser`` CLI should not pay for Pydantic, a provider
adapter, or the extraction/verification/composition stack. Each public name is
therefore resolved on first access and only loads the module that defines it.

The facade is the supported entry point for consumers. Stage modules are public
for callers that need their prompts, wire formats, or checks; names with a
leading underscore are internal.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    # Models and parser
    "Cell": ".models",
    "CellKind": ".models",
    "CitationError": ".models",
    "Excerpt": ".models",
    "Line": ".models",
    "NotebookError": ".models",
    "NotebookFormatError": ".models",
    "ParseResult": ".models",
    "UnsupportedFormatError": ".models",
    "parse_file": ".parser",
    "parse_text": ".parser",
    # Citations
    "Citation": ".citations",
    "find_citations": ".citations",
    "format_citation": ".citations",
    "numbered_listing": ".citations",
    "parse_citation": ".citations",
    "validate_citation": ".citations",
    # Static facts
    "StaticFacts": ".facts",
    "extract_facts": ".facts",
    "render_facts_for_prompt": ".facts",
    # Claim extraction
    "Chunk": ".extract",
    "Claim": ".extract",
    "FailedRegion": ".extract",
    "Kind": ".extract",
    "Ledger": ".extract",
    "Tag": ".extract",
    "extract_claims": ".extract",
    "plan_chunks": ".extract",
    # Verification
    "Judge": ".verify",
    "Status": ".verify",
    "VerifiedClaim": ".verify",
    "VerifiedLedger": ".verify",
    "verify_claim": ".verify",
    "verify_ledger": ".verify",
    # Composition
    "Audience": ".compose",
    "Document": ".compose",
    "DocumentSet": ".compose",
    "compose_document": ".compose",
    "compose_documents": ".compose",
    "render_markdown": ".compose",
    # Configuration
    "ConfigError": ".config",
    "LLMConfig": ".config",
    "Provider": ".config",
    "Strategy": ".config",
    "load_config": ".config",
    "load_dotenv": ".config",
    # LLM client and testing utility
    "FakeLLM": ".llm",
    "LLMClient": ".llm",
    "LLMError": ".llm",
    "make_client": ".llm",
}

if TYPE_CHECKING:
    from .citations import (
        Citation,
        find_citations,
        format_citation,
        numbered_listing,
        parse_citation,
        validate_citation,
    )
    from .compose import (
        Audience,
        Document,
        DocumentSet,
        compose_document,
        compose_documents,
        render_markdown,
    )
    from .config import (
        ConfigError,
        LLMConfig,
        Provider,
        Strategy,
        load_config,
        load_dotenv,
    )
    from .extract import (
        Chunk,
        Claim,
        FailedRegion,
        Kind,
        Ledger,
        Tag,
        extract_claims,
        plan_chunks,
    )
    from .facts import StaticFacts, extract_facts, render_facts_for_prompt
    from .llm import FakeLLM, LLMClient, LLMError, make_client
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
    from .parser import parse_file, parse_text
    from .verify import (
        Judge,
        Status,
        VerifiedClaim,
        VerifiedLedger,
        verify_claim,
        verify_ledger,
    )

__all__ = [
    "Audience",
    "Cell",
    "CellKind",
    "Chunk",
    "Citation",
    "CitationError",
    "Claim",
    "ConfigError",
    "Document",
    "DocumentSet",
    "Excerpt",
    "FailedRegion",
    "FakeLLM",
    "Judge",
    "Kind",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "Ledger",
    "Line",
    "NotebookError",
    "NotebookFormatError",
    "ParseResult",
    "Provider",
    "StaticFacts",
    "Status",
    "Strategy",
    "Tag",
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


def __getattr__(name: str) -> object:
    """Resolve one public name on first access, then cache it in the module."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: object = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List lazy exports for introspection without importing their modules."""
    return sorted({*globals(), *__all__})
