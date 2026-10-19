"""Immutable notebook model and citation errors.

All numbering in this package is 1-based: cell numbers, cell-relative line
numbers (``rel_no``), and physical file line numbers (``abs_no``). Zero and
negative values are never valid cell or line numbers anywhere in the API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class NotebookError(Exception):
    """Base class for every expected notebook-processing failure."""


class NotebookFormatError(NotebookError):
    """The file is not a notebook this parser can understand."""


class UnsupportedFormatError(NotebookError):
    """The requested file suffix or format name is not supported."""


class CitationError(NotebookError):
    """A citation or a line lookup does not address an existing line."""


class CellKind(StrEnum):
    """Cell kinds kept deliberately small; unknown magics stay MAGIC_OTHER."""

    PYTHON = "python"
    MARKDOWN = "markdown"
    SQL = "sql"
    RUN = "run"
    MAGIC_OTHER = "magic"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class Line:
    """One physical line with both coordinate systems attached.

    ``raw`` is never cleaned because the UI edits from it; ``text`` is the
    display/source form and only differs when a magic prefix was stripped.
    """

    cell_no: int
    rel_no: int
    abs_no: int
    raw: str
    text: str


@dataclass(frozen=True, slots=True)
class Cell:
    """A numbered cell whose lines may start after trimmed blank lines."""

    number: int
    kind: CellKind
    magic: str | None
    magic_args: str | None
    title: str | None
    lines: tuple[Line, ...]

    @property
    def first_abs(self) -> int | None:
        """Return the first physical file line, or None for an empty cell."""
        return self.lines[0].abs_no if self.lines else None

    @property
    def last_abs(self) -> int | None:
        """Return the last physical file line, or None for an empty cell."""
        return self.lines[-1].abs_no if self.lines else None

    @property
    def source(self) -> str:
        """Return the magic-stripped source, which is what tools should show."""
        return "\n".join(line.text for line in self.lines)

    @property
    def run_target(self) -> str | None:
        """Return the ``%run`` target without the quotes Databricks accepts."""
        if self.kind is not CellKind.RUN or self.magic_args is None:
            return None
        args = self.magic_args.strip()
        if not args:
            return None
        if args[0] in "\"'":
            closing = args.find(args[0], 1)
            if closing != -1:
                return args[1:closing]
        return args.split(maxsplit=1)[0]


@dataclass(frozen=True, slots=True)
class Excerpt:
    """A validated citation range plus same-cell context for the UI."""

    cell_no: int
    start: int
    end: int
    abs_start: int
    abs_end: int
    lines: tuple[Line, ...]
    before: tuple[Line, ...]
    after: tuple[Line, ...]


@dataclass(frozen=True, slots=True)
class ParseResult:
    """A parsed notebook and the single source of truth for line lookup."""

    path: str | None
    format: str
    cells: tuple[Cell, ...]
    flat_lines: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    _abs_index: Mapping[int, tuple[int, int] | None] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        index: dict[int, tuple[int, int] | None] = {
            abs_no: None for abs_no in range(1, len(self.flat_lines) + 1)
        }
        for cell in self.cells:
            for line in cell.lines:
                index[line.abs_no] = (cell.number, line.rel_no)
        object.__setattr__(self, "_abs_index", MappingProxyType(index))

    def cell(self, value: int) -> Cell:
        """Return a cell by 1-based number, or explain the valid range."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise CitationError(f"cell number must be an integer, got {value!r}")
        if not 1 <= value <= len(self.cells):
            raise CitationError(
                f"cell {value} does not exist; notebook has {len(self.cells)} "
                f"cells (valid range 1..{len(self.cells)})"
            )
        return self.cells[value - 1]

    def line(self, cell_no: int, rel_no: int) -> Line:
        """Return a cell-relative line, or explain the cell's valid range."""
        cell = self.cell(cell_no)
        if not isinstance(rel_no, int) or isinstance(rel_no, bool):
            raise CitationError(
                f"line number must be an integer, got {rel_no!r} for cell {cell_no}"
            )
        if not 1 <= rel_no <= len(cell.lines):
            if not cell.lines:
                raise CitationError(
                    f"cell {cell_no} has no lines (it is empty); "
                    f"line {rel_no} cannot be resolved"
                )
            raise CitationError(
                f"cell {cell_no} has {len(cell.lines)} lines; "
                f"line {rel_no} is outside the valid range 1..{len(cell.lines)}"
            )
        return cell.lines[rel_no - 1]

    def cell_rel_to_abs(self, cell_no: int, rel_no: int) -> int:
        """Translate cell-relative numbering to physical file numbering."""
        return self.line(cell_no, rel_no).abs_no

    def abs_to_cell_rel(self, abs_no: int) -> tuple[int, int] | None:
        """Translate physical file numbering, or None for non-cell lines."""
        if not isinstance(abs_no, int) or isinstance(abs_no, bool):
            raise CitationError(f"absolute line must be an integer, got {abs_no!r}")
        if not 1 <= abs_no <= len(self.flat_lines):
            raise CitationError(
                f"absolute line {abs_no} is outside the notebook; valid range "
                f"is 1..{len(self.flat_lines)}"
            )
        return self._abs_index[abs_no]

    def excerpt(self, cell_no: int, start: int, end: int, context: int = 0) -> Excerpt:
        """Build a citation excerpt, clamping context inside one cell."""
        cell = self.cell(cell_no)
        if not isinstance(context, int) or isinstance(context, bool) or context < 0:
            raise CitationError(
                f"context must be a non-negative integer, got {context!r} "
                f"for cell {cell_no}"
            )
        if start < 1 or end < 1 or start > end:
            raise CitationError(
                f"invalid line range L{start}-L{end} in cell {cell_no}; "
                "expected 1 <= start <= end"
            )
        if end > len(cell.lines):
            if not cell.lines:
                raise CitationError(
                    f"cell {cell_no} is empty; L{start}-L{end} cannot be resolved"
                )
            raise CitationError(
                f"cell {cell_no} has {len(cell.lines)} lines; "
                f"L{start}-L{end} exceeds the valid range 1..{len(cell.lines)}"
            )

        selected = cell.lines[start - 1 : end]
        before = cell.lines[max(0, start - 1 - context) : start - 1]
        after = cell.lines[end : min(len(cell.lines), end + context)]
        return Excerpt(
            cell_no=cell_no,
            start=start,
            end=end,
            abs_start=selected[0].abs_no,
            abs_end=selected[-1].abs_no,
            lines=tuple(selected),
            before=tuple(before),
            after=tuple(after),
        )
