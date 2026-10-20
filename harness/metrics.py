"""Per-run metrics: one printable line and one JSON object.

The line is emitted at the end of every pipeline run and the object is written
to ``metrics.json``.  Counts are cheap to compare across prompt or model
changes; the per-document claim-ID coverage shows whether a document actually
used the verified material it was given.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .compose import Document, DocumentSet
from .extract import Ledger
from .injection import InjectionFinding
from .verify import VerifiedLedger


@dataclass(frozen=True, slots=True)
class DocumentMetrics:
    """Coverage of one audience document's model-written prose."""

    claims_used: int
    verified_claims: int
    sentences: int
    sentences_with_ids: int
    static_sentences: int = 0

    @property
    def claim_coverage(self) -> float:
        """Share of verified claims this document cited; 0.0 with no claims."""
        if self.verified_claims == 0:
            return 0.0
        return self.claims_used / self.verified_claims

    @property
    def sentence_coverage(self) -> float:
        """Share of model sentences with a claim id; static text is excluded."""
        if self.sentences == 0:
            return 0.0
        return self.sentences_with_ids / self.sentences

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view with stable rounding."""
        return {
            "claims_used": self.claims_used,
            "verified_claims": self.verified_claims,
            "claim_coverage": round(self.claim_coverage, 4),
            "sentences": self.sentences,
            "sentences_with_ids": self.sentences_with_ids,
            "sentence_coverage": round(self.sentence_coverage, 4),
            "static_sentences": self.static_sentences,
        }


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Everything one run should report about its own quality and cost."""

    strategy: str
    repairs: int
    chunks: int
    extracted: int
    dropped_foreign: int
    dropped_duplicate: int
    verified: int
    flagged: int
    dropped: int
    documents: dict[str, DocumentMetrics]
    injection: tuple[InjectionFinding, ...]

    def line(self) -> str:
        """Return the single human-readable line printed after every run."""
        parts = [
            f"metrics: chunks={self.chunks} extracted={self.extracted}",
            "ledger_drops="
            f"foreign:{self.dropped_foreign},duplicate:{self.dropped_duplicate}",
            f"verified={self.verified} flagged={self.flagged} dropped={self.dropped}",
            f"repairs={self.repairs} strategy={self.strategy}",
        ]
        for audience, doc in self.documents.items():
            parts.append(
                f"{audience} ids={doc.claims_used}/{doc.verified_claims} "
                f"sentences={doc.sentences_with_ids}/{doc.sentences}"
            )
        parts.append(f"injection={len(self.injection)}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON artifact written next to the documents."""
        return {
            "line": self.line(),
            "strategy": self.strategy,
            "repairs": self.repairs,
            "ledger": {
                "chunks": self.chunks,
                "extracted": self.extracted,
                "dropped_foreign": self.dropped_foreign,
                "dropped_duplicate": self.dropped_duplicate,
            },
            "verification": {
                "verified": self.verified,
                "flagged": self.flagged,
                "dropped": self.dropped,
            },
            "documents": {
                audience: doc.to_dict() for audience, doc in self.documents.items()
            },
            "injection": [finding.to_dict() for finding in self.injection],
        }


def build_metrics(
    ledger: Ledger,
    verified: VerifiedLedger,
    documents: DocumentSet,
    *,
    repairs: int,
    strategy: str,
    injection: Sequence[InjectionFinding] = (),
) -> RunMetrics:
    """Assemble the metrics for one finished pipeline run."""
    verified_total = len(verified.ids("verified"))
    by_audience = {
        "manager": documents.manager,
        "developer": documents.developer,
        "agent": documents.agent,
    }
    return RunMetrics(
        strategy=strategy,
        repairs=repairs,
        chunks=ledger.n_chunks,
        extracted=len(ledger.claims),
        dropped_foreign=ledger.n_dropped_foreign,
        dropped_duplicate=ledger.n_dropped_duplicate,
        verified=verified_total,
        flagged=len(verified.ids("flagged")),
        dropped=len(verified.ids("dropped")),
        documents={
            audience: _document_metrics(document, verified_total)
            for audience, document in by_audience.items()
        },
        injection=tuple(injection),
    )


def _document_metrics(document: Document, verified_total: int) -> DocumentMetrics:
    """Count ids and model sentences without trusting the document's prose."""
    sentences = [
        sentence for section in document.sections for sentence in section.sentences
    ]
    model = [sentence for sentence in sentences if sentence.source != "static"]
    return DocumentMetrics(
        claims_used=len(document.used_ids),
        verified_claims=verified_total,
        sentences=len(model),
        sentences_with_ids=sum(1 for sentence in model if sentence.ids),
        static_sentences=len(sentences) - len(model),
    )
