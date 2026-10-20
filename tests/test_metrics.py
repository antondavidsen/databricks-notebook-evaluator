"""per-run metrics include repairs, ledger drops and document coverage."""

from __future__ import annotations

from harness.citations import Citation
from harness.compose import DocSection, DocSentence, Document, DocumentSet
from harness.extract import Ledger
from harness.injection import InjectionFinding
from harness.metrics import DocumentMetrics, build_metrics
from harness.verify import VerifiedLedger

EMPTY_LEDGER = Ledger(
    claims=(),
    failed_regions=(),
    uncovered_cells=(),
    n_chunks=0,
    n_dropped_foreign=0,
    n_dropped_duplicate=0,
)
EMPTY_VERIFIED = VerifiedLedger(
    items=(), failed_regions=(), uncovered_cells=(), notes=()
)
EMPTY_DOC = Document("manager", "Title", (), (), None, ())
EMPTY_SET = DocumentSet(EMPTY_DOC, EMPTY_DOC, EMPTY_DOC, (), {"total": 0}, ())


def test_zero_counts_do_not_divide_by_zero_or_hide_audiences() -> None:
    metrics = build_metrics(
        EMPTY_LEDGER, EMPTY_VERIFIED, EMPTY_SET, repairs=0, strategy="prompt"
    )

    assert metrics.documents["manager"].claim_coverage == 0.0
    assert metrics.documents["manager"].sentence_coverage == 0.0
    assert list(metrics.documents) == ["manager", "developer", "agent"]
    assert metrics.line() == (
        "metrics: chunks=0 extracted=0 | "
        "ledger_drops=foreign:0,duplicate:0 | "
        "verified=0 flagged=0 dropped=0 | repairs=0 strategy=prompt | "
        "manager ids=0/0 sentences=0/0 | "
        "developer ids=0/0 sentences=0/0 | "
        "agent ids=0/0 sentences=0/0 | "
        "injection=0"
    )


def test_document_metrics_round_to_four_places() -> None:
    metrics = DocumentMetrics(
        claims_used=1, verified_claims=3, sentences=3, sentences_with_ids=1
    )

    assert metrics.to_dict() == {
        "claims_used": 1,
        "verified_claims": 3,
        "claim_coverage": 0.3333,
        "sentences": 3,
        "sentences_with_ids": 1,
        "sentence_coverage": 0.3333,
        "static_sentences": 0,
    }


def test_static_sentences_are_reported_but_not_counted_as_model_prose() -> None:
    model = DocSentence(
        "The rate is 0.2.",
        ("F1",),
        (Citation(1, 1, 1),),
        "claims",
        (),
    )
    static = DocSentence("CONSTANT TOL can change.", (), (), "static", ())
    doc = Document(
        "developer",
        "Title",
        (DocSection("Heading", (model, static)),),
        (),
        None,
        ("F1",),
    )
    documents = DocumentSet(doc, doc, doc, (), {"total": 0}, ())

    metrics = build_metrics(
        EMPTY_LEDGER, EMPTY_VERIFIED, documents, repairs=0, strategy="prompt"
    )

    assert metrics.documents["developer"].sentences == 1
    assert metrics.documents["developer"].sentences_with_ids == 1
    assert metrics.documents["developer"].static_sentences == 1
    assert "developer ids=1/0 sentences=1/1" in metrics.line()


def test_injection_findings_and_ledger_drops_reach_the_artifact() -> None:
    finding = InjectionFinding(2, 1, 6, "ignore-instructions", "ignore this")
    ledger = Ledger((), (), (), 2, 3, 4)

    metrics = build_metrics(
        ledger,
        EMPTY_VERIFIED,
        EMPTY_SET,
        repairs=2,
        strategy="json_mode",
        injection=(finding,),
    )
    payload = metrics.to_dict()

    assert payload["repairs"] == 2
    assert payload["strategy"] == "json_mode"
    assert payload["ledger"] == {
        "chunks": 2,
        "extracted": 0,
        "dropped_foreign": 3,
        "dropped_duplicate": 4,
    }
    assert payload["injection"] == [finding.to_dict()]
    assert "ledger_drops=foreign:3,duplicate:4" in metrics.line()
    assert "injection=1" in metrics.line()
