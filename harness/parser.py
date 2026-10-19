"""Parse Databricks Python notebooks (``.py``) into exact line coordinates.

The splitter mirrors Databricks itself, including its known limitation: a
separator-looking line inside a string literal still splits. Only the
Databricks ``.py`` source format is supported; other formats are rejected.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from .models import (
    Cell,
    CellKind,
    Line,
    NotebookError,
    NotebookFormatError,
    ParseResult,
    UnsupportedFormatError,
)

_SUPPORTED_SUFFIXES = (".py",)
_HEADER = "# Databricks notebook source"
_MAGIC_PREFIX = "# MAGIC"
_SEPARATOR_RE = re.compile(r"^# COMMAND -{3,}\s*$")
_DBTITLE_RE = re.compile(r"^# DBTITLE \d+,(.*)$")
_MAGIC_COMMAND_RE = re.compile(r"^%([A-Za-z][\w-]*)(?:\s+(.*))?$")


def _label(path: str | None) -> str:
    return path if path is not None else "<text>"


def _split_lines(text: str) -> list[str]:
    """Split like an editor: normalize CRLF/CR, then split only on LF."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if normalized.endswith("\n") and lines and lines[-1] == "":
        lines.pop()
    return lines


def _trim(segment: list[tuple[int, str]]) -> list[tuple[int, str]]:
    start = 0
    end = len(segment)
    while start < end and segment[start][1].strip() == "":
        start += 1
    while end > start and segment[end - 1][1].strip() == "":
        end -= 1
    return segment[start:end]


def _is_magic_line(raw: str) -> bool:
    return raw == _MAGIC_PREFIX or raw.startswith(_MAGIC_PREFIX + " ")


def _strip_magic(raw: str) -> str:
    return raw[len(_MAGIC_PREFIX) :].removeprefix(" ")


def _magic_command(text: str) -> tuple[str | None, str | None]:
    match = _MAGIC_COMMAND_RE.match(text)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def _kind_for_magic(name: str) -> CellKind:
    if name in ("md", "md-sandbox"):
        return CellKind.MARKDOWN
    if name == "sql":
        return CellKind.SQL
    if name == "run":
        return CellKind.RUN
    if name == "python":
        return CellKind.PYTHON
    return CellKind.MAGIC_OTHER


def _build_lines(
    number: int,
    segment: list[tuple[int, str]],
    strip: bool,
) -> tuple[Line, ...]:
    lines: list[Line] = []
    for rel_no, (abs_no, raw) in enumerate(segment, start=1):
        text = raw
        if strip and _is_magic_line(raw):
            text = _strip_magic(raw)
        lines.append(Line(number, rel_no, abs_no, raw, text))
    return tuple(lines)


def _build_source_cell(
    number: int,
    segment: list[tuple[int, str]],
    warnings: list[str],
) -> Cell:
    trimmed = _trim(segment)
    if not trimmed:
        return Cell(number, CellKind.EMPTY, None, None, None, ())

    title_match = _DBTITLE_RE.match(trimmed[0][1])
    title = title_match.group(1) if title_match is not None else None
    body = trimmed[1:] if title_match is not None else trimmed
    body_is_magic = [_is_magic_line(raw) for _, raw in body]

    if body and all(body_is_magic):
        lines = _build_lines(number, trimmed, strip=True)
        first_text = lines[0 if title is None else 1].text
        name, args = _magic_command(first_text)
        if name is None:
            warnings.append(
                f"cell {number}: magic cell has no %command; "
                "classified as magic without a name"
            )
            return Cell(number, CellKind.MAGIC_OTHER, None, None, title, lines)
        return Cell(number, _kind_for_magic(name), name, args, title, lines)

    if any(body_is_magic):
        warnings.append(
            f"cell {number}: only some lines use the magic prefix; treated as "
            f"{CellKind.PYTHON.value} with unstripped text"
        )
    lines = _build_lines(number, trimmed, strip=False)
    return Cell(number, CellKind.PYTHON, None, None, title, lines)


def _source_segments(
    lines: list[str], warnings: list[str]
) -> list[list[tuple[int, str]]]:
    segments: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    last_separator: int | None = None
    for index in range(1, len(lines)):
        abs_no = index + 1
        raw = lines[index]
        if _SEPARATOR_RE.match(raw):
            segments.append(current)
            current = []
            last_separator = abs_no
        else:
            current.append((abs_no, raw))

    if last_separator is not None and not any(raw.strip() for _, raw in current):
        warnings.append(
            f"line {last_separator}: dropped the empty final cell after a "
            "trailing separator"
        )
    else:
        segments.append(current)
    return segments


def _parse_python(text: str, path: str | None) -> ParseResult:
    if text == "":
        raise NotebookFormatError(
            f"{_label(path)}: empty file; expected at least a notebook header "
            f"or one line of content"
        )
    lines = _split_lines(text)
    warnings: list[str] = []
    if lines and lines[0].strip() == _HEADER:
        segments = _source_segments(lines, warnings)
    else:
        warnings.append("no Databricks header; treated as a single cell")
        segments = [[(index + 1, raw) for index, raw in enumerate(lines)]]

    cells = tuple(
        _build_source_cell(number, segment, warnings)
        for number, segment in enumerate(segments, start=1)
    )
    return ParseResult(
        path=path,
        format="py",
        cells=cells,
        flat_lines=tuple(lines),
        warnings=tuple(warnings),
    )


def parse_text(text: str, fmt: str, path: str | None = None) -> ParseResult:
    """Parse notebook text when the caller already knows its format.

    Only Databricks ``.py`` notebooks are supported. The signature stays
    format-aware so unsupported input fails with a typed, actionable error
    instead of a generic argument error.
    """
    normalized = fmt.lower().lstrip(".")
    if normalized != "py":
        raise UnsupportedFormatError(
            f"unsupported format {fmt!r}; only the Databricks .py format is supported"
        )
    return _parse_python(text, path)


def parse_file(path: str | os.PathLike[str]) -> ParseResult:
    """Parse a notebook selected by its file suffix."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"{file_path}: unsupported notebook suffix {suffix or '<none>'!r}; "
            f"supported suffix is {_SUPPORTED_SUFFIXES[0]}"
        )

    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise NotebookError(f"{file_path}: cannot read file: {exc}") from exc
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NotebookFormatError(
            f"{file_path}: not valid UTF-8 near byte {exc.start}"
        ) from exc
    return parse_text(text, suffix[1:], str(file_path))


def render_debug_table(result: ParseResult) -> str:
    """Render a stable plain-text table for the module's debug entry point."""
    rows: list[str] = []
    for cell in result.cells:
        abs_range = (
            "none" if cell.first_abs is None else f"{cell.first_abs}-{cell.last_abs}"
        )
        first = cell.lines[0].text if cell.lines else ""
        rows.append(
            f"C{cell.number:<3} {cell.kind.value:<12} "
            f"abs {abs_range:<10} lines {len(cell.lines):<3} {first[:60]}"
        )
    if result.warnings:
        rows.append("")
        rows.extend(f"warning: {warning}" for warning in result.warnings)
    return "\n".join(rows) + ("\n" if rows else "")


def main(argv: Sequence[str] | None = None) -> int:
    """Print the debug table; kept thin because I/O belongs at the edge."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.parser",
        description="Print one row per Databricks Python notebook cell.",
    )
    parser.add_argument("path", help="Databricks Python notebook file (.py)")
    args = parser.parse_args(argv)
    try:
        result = parse_file(args.path)
    except NotebookError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(render_debug_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
