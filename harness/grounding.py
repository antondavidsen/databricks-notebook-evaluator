"""Shared lexical grounding rules and deterministic claim ordering.

Extraction, verification, composition, and the app must agree on what counts as
a checkable number, identifier, or quote, and on the order claims are processed
and displayed. Keeping those decisions here makes the agreement explicit.

This module is a leaf: it imports only ``CellKind`` and the standard library.
Stages that need grounding import it; it never imports a stage, so no cycle is
possible.
"""

from __future__ import annotations

import re
import secrets

from .models import CellKind

__all__ = [
    "DOCUMENTABLE_KINDS",
    "IDENT_RE",
    "LOWER_UPPER_RE",
    "NUMBER_RE",
    "QUOTED_RE",
    "SKIPPED_NUMBERS",
    "claim_id_key",
    "claim_number",
    "is_checked_identifier",
    "is_skipped_number",
    "marker_token",
    "numbers_in",
    "to_number",
]

# A number must not be part of an identifier (``v3``, ``F2``) or a dotted name.
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
IDENT_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
QUOTED_RE = re.compile(r'"([^"\n]+)"|`([^`\n]+)`')
LOWER_UPPER_RE = re.compile(r"[a-z][A-Z]")

# Cells whose text can support a claim. Empty cells and markdown cannot.
DOCUMENTABLE_KINDS = frozenset(
    {CellKind.PYTHON, CellKind.SQL, CellKind.RUN, CellKind.MAGIC_OTHER}
)
# 0, 1, and 2 are too common in prose to prove anything.
SKIPPED_NUMBERS = frozenset({0.0, 1.0, 2.0})


def marker_token(text: str) -> str:
    """Return a token that cannot already occur in the fenced text."""
    token = secrets.token_hex(4)
    while token in text:
        token = secrets.token_hex(4)
    return token


def to_number(raw: str) -> float:
    """Parse a ``NUMBER_RE`` match, accepting thousands separators."""
    return float(raw.replace(",", ""))


def numbers_in(text: str) -> tuple[float, ...]:
    """Extract comparable numbers from text, dropping grouping commas."""
    return tuple(to_number(match.group()) for match in NUMBER_RE.finditer(text))


def is_skipped_number(value: float) -> bool:
    """Small counts are too common in prose to prove grounding."""
    return value.is_integer() and value in SKIPPED_NUMBERS


def is_checked_identifier(token: str) -> bool:
    """Keep code-shaped tokens and ignore plain words and bare acronyms."""
    if "_" in token:
        return True
    if "." in token and max(len(part) for part in token.split(".")) >= 3:
        return True
    if LOWER_UPPER_RE.search(token):
        return True
    return any(char.isalpha() for char in token) and any(
        char.isdigit() for char in token
    )


def claim_number(claim_id: str) -> int:
    """Return the first integer in an id, or 0 when it has none.

    The app uses this to display ``F2`` before ``F10`` and to place ids without
    a number before numeric ones. Composition uses it only as a position
    tie-break, so two claims starting on the same line stay stable.
    """
    match = re.search(r"\d+", claim_id)
    return int(match.group()) if match is not None else 0


def claim_id_key(claim_id: str) -> tuple[int, str]:
    """Canonical claim-id order: ids without a number first, then numeric order.

    The whole id is the tie-break, so the order is total for arbitrary ids and
    does not depend on input order. ``F2`` precedes ``F10``; ``note`` precedes
    both because it has no number.
    """
    return (claim_number(claim_id), claim_id)
