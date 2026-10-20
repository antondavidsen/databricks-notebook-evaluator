"""Non-obedient injection detector.

Nothing in this module changes a prompt or a verdict.  It scans the parsed
notebook for instruction-like text, records where it is, and lets the run
report it.  The pipeline already fences notebook text as DATA for extraction
and judging; this makes a hostile notebook visible in ``metrics.json`` instead
of silently passing through as ordinary code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ParseResult

# Order matters: the first match on a line decides the rule name.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore-instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget)\b.{0,40}"
            r"\b(?:previous|prior|above|all|earlier)\b.{0,40}"
            r"\b(?:instructions?|prompts?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role-override",
        re.compile(
            r"\byou are now\b|\bfrom now on\b|"
            r"\bact as (?:the )?(?:system|developer|admin)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt-exfiltration",
        re.compile(
            r"\b(?:reveal|print|show|output|repeat|exfiltrate|send|leak|steal)\b"
            r".{0,40}\b(?:system prompt|instructions?|api[ _-]?key|secret|token|"
            r"password)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tool-instruction",
        re.compile(
            r"\b(?:run|execute|call|invoke)\b.{0,30}"
            r"\b(?:shell|command|script|https?|url|email|webhook)\b",
            re.IGNORECASE,
        ),
    ),
)

_TEXT_LIMIT = 200


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One line that reads like an instruction to the model."""

    cell: int
    line: int
    abs_line: int
    rule: str
    text: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view for metrics.json and the UI."""
        return {
            "cell": self.cell,
            "line": self.line,
            "abs_line": self.abs_line,
            "rule": self.rule,
            "text": self.text,
        }


def scan_injection(result: ParseResult) -> tuple[InjectionFinding, ...]:
    """Return instruction-like lines in source order; never raise on content."""
    findings: list[InjectionFinding] = []
    for cell in result.cells:
        for line in cell.lines:
            text = line.text.strip()
            if not text:
                continue
            for rule, pattern in _RULES:
                if pattern.search(text):
                    findings.append(
                        InjectionFinding(
                            cell=cell.number,
                            line=line.rel_no,
                            abs_line=line.abs_no,
                            rule=rule,
                            text=text[:_TEXT_LIMIT],
                        )
                    )
                    break
    return tuple(findings)
