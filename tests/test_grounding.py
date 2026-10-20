from __future__ import annotations

import pytest
from harness.grounding import (
    DOCUMENTABLE_KINDS,
    NUMBER_RE,
    SKIPPED_NUMBERS,
    claim_id_key,
    claim_number,
    is_checked_identifier,
    is_skipped_number,
    marker_token,
    numbers_in,
    to_number,
)
from harness.models import CellKind


def test_documentable_kinds_are_source_cells_only() -> None:
    assert set(DOCUMENTABLE_KINDS) == {
        CellKind.PYTHON,
        CellKind.SQL,
        CellKind.RUN,
        CellKind.MAGIC_OTHER,
    }
    assert CellKind.MARKDOWN not in DOCUMENTABLE_KINDS
    assert CellKind.EMPTY not in DOCUMENTABLE_KINDS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nothing here", ()),
        ("TOL is 0.5", (0.5,)),
        ("LIMIT 1,250 and 1,234,567.89", (1250.0, 1234567.89)),
        ("100%", (100.0,)),
        # Identifiers and dotted names do not contribute their digits.
        ("v3 F2 df.col2 x_1", ()),
        (".5 stays a fraction without a leading digit", ()),
    ],
)
def test_numbers_in_ignores_identifier_digits(
    text: str, expected: tuple[float, ...]
) -> None:
    assert numbers_in(text) == expected


def test_number_re_matches_grouped_and_decimal_numbers() -> None:
    assert [match.group() for match in NUMBER_RE.finditer("a 1,000 b 2.50")] == [
        "1,000",
        "2.50",
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234", 1234.0), ("0.125", 0.125), ("10", 10.0)],
)
def test_to_number_accepts_grouping_commas(raw: str, expected: float) -> None:
    assert to_number(raw) == expected


@pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
def test_small_integers_are_skipped(value: float) -> None:
    assert is_skipped_number(value)


@pytest.mark.parametrize("value", [2.5, 3.0, -1.5])
def test_other_numbers_are_not_skipped(value: float) -> None:
    assert not is_skipped_number(value)


def test_skipped_numbers_are_small_integers() -> None:
    assert set(SKIPPED_NUMBERS) == {0.0, 1.0, 2.0}


@pytest.mark.parametrize(
    "token",
    ["MAX_WO", "df.column", "camelCase", "v3", "F2", "a1"],
)
def test_code_shaped_tokens_are_checked(token: str) -> None:
    assert is_checked_identifier(token)


@pytest.mark.parametrize("token", ["a.b", "lower", "MAXWO", "UPPER", "x"])
def test_plain_words_and_bare_acronyms_are_not_checked(token: str) -> None:
    assert not is_checked_identifier(token)


def test_marker_token_regenerates_until_it_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter(("deadbeef", "cafebabe"))
    monkeypatch.setattr("harness.grounding.secrets.token_hex", lambda _n: next(values))

    assert marker_token("prefix deadbeef suffix") == "cafebabe"


def test_marker_token_accepts_a_fresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harness.grounding.secrets.token_hex", lambda _n: "0badf00d")

    assert marker_token("no collision") == "0badf00d"


@pytest.mark.parametrize(
    ("claim_id", "expected"),
    [("F10", 10), ("F2", 2), ("F-12", 12), ("F2a10", 2), ("note", 0), ("", 0)],
)
def test_claim_number_uses_the_first_integer(claim_id: str, expected: int) -> None:
    assert claim_number(claim_id) == expected


def test_claim_ids_sort_naturally_with_numberless_ids_first() -> None:
    ids = ["F10", "note", "F2", "F1", "F02"]

    assert sorted(ids, key=claim_id_key) == ["note", "F1", "F02", "F2", "F10"]


def test_claim_id_key_is_total_for_duplicate_numbers() -> None:
    assert sorted(["F2b", "F2a", "F2"], key=claim_id_key) == ["F2", "F2a", "F2b"]
