from __future__ import annotations

import codecs
from pathlib import Path

import pytest
from harness.models import (
    CellKind,
    NotebookError,
    NotebookFormatError,
    ParseResult,
    UnsupportedFormatError,
)
from harness.parser import main, parse_file, parse_text, render_debug_table

HEADER = "# Databricks notebook source"
SEP = "# COMMAND ----------"


def parse_py(text: str) -> ParseResult:
    return parse_text(text, "py")


def source(*lines: str, trailing_newline: bool = True) -> str:
    text = "\n".join(lines)
    return text + "\n" if trailing_newline else text


def test_basic_three_cell_python_notebook() -> None:
    text = source(HEADER, "x = 1", SEP, "y = 2", SEP, "z = 3")
    result = parse_text(text, "py")

    assert result.path is None
    assert result.format == "py"
    assert result.warnings == ()
    assert [cell.number for cell in result.cells] == [1, 2, 3]
    assert [cell.kind for cell in result.cells] == [CellKind.PYTHON] * 3
    assert [(cell.first_abs, cell.last_abs) for cell in result.cells] == [
        (2, 2),
        (4, 4),
        (6, 6),
    ]
    assert [result.cell_rel_to_abs(n, 1) for n in (1, 2, 3)] == [2, 4, 6]
    assert result.flat_lines == (HEADER, "x = 1", SEP, "y = 2", SEP, "z = 3")

    # The header and separators are real file lines but belong to no cell.
    assert result.abs_to_cell_rel(1) is None
    assert result.abs_to_cell_rel(3) is None
    assert result.abs_to_cell_rel(5) is None


def test_leading_and_trailing_blanks_are_trimmed_but_keep_abs_numbers() -> None:
    text = source(
        HEADER,
        "",
        "   ",
        "a = 1",
        "",
        "b = 2",
        "\t",
        SEP,
        "",
        "c = 3",
        "   ",
    )
    result = parse_text(text, "py")

    first, second = result.cells
    assert [line.abs_no for line in first.lines] == [4, 5, 6]
    assert [line.rel_no for line in first.lines] == [1, 2, 3]
    assert [line.raw for line in first.lines] == ["a = 1", "", "b = 2"]
    assert first.first_abs == 4
    assert first.last_abs == 6
    assert [(line.abs_no, line.raw) for line in second.lines] == [(10, "c = 3")]

    for abs_no in (1, 2, 3, 7, 8, 9, 11):
        assert result.abs_to_cell_rel(abs_no) is None


def test_two_consecutive_separators_create_an_empty_cell() -> None:
    text = source(HEADER, "a = 1", SEP, SEP, "b = 2")
    result = parse_text(text, "py")

    assert len(result.cells) == 3
    assert [cell.number for cell in result.cells] == [1, 2, 3]
    assert result.cells[1].kind is CellKind.EMPTY
    assert result.cells[1].lines == ()
    assert result.cells[1].first_abs is None
    assert result.cells[1].last_abs is None
    assert result.cells[2].lines[0].abs_no == 5


def test_trailing_separator_is_dropped_with_one_warning() -> None:
    text = source(HEADER, "a = 1", SEP)
    result = parse_text(text, "py")

    assert len(result.cells) == 1
    assert len(result.warnings) == 1
    assert "line 3" in result.warnings[0]
    assert "trailing separator" in result.warnings[0]


def test_missing_header_treats_the_whole_file_as_one_cell() -> None:
    text = source("x = 1", SEP, "", "y = 2")
    result = parse_text(text, "py")

    assert len(result.cells) == 1
    assert len(result.warnings) == 1
    assert result.warnings[0] == "no Databricks header; treated as a single cell"
    cell = result.cells[0]
    assert cell.kind is CellKind.PYTHON
    assert [line.abs_no for line in cell.lines] == [1, 2, 3, 4]
    assert [line.rel_no for line in cell.lines] == [1, 2, 3, 4]
    assert cell.lines[1].raw == SEP
    assert cell.lines[2].raw == ""
    assert result.abs_to_cell_rel(1) == (1, 1)
    assert result.abs_to_cell_rel(2) == (1, 2)


def signature(result: ParseResult) -> tuple[object, ...]:
    return (
        result.format,
        tuple(result.flat_lines),
        tuple(result.warnings),
        tuple(
            (
                cell.number,
                cell.kind,
                cell.magic,
                cell.magic_args,
                cell.title,
                tuple(
                    (line.cell_no, line.rel_no, line.abs_no, line.raw, line.text)
                    for line in cell.lines
                ),
            )
            for cell in result.cells
        ),
    )


@pytest.mark.parametrize(
    "variant",
    ["lf", "lf_no_trailing_newline", "crlf", "cr", "bom"],
)
def test_newline_and_bom_variants_have_identical_numbering(
    tmp_path: Path, variant: str
) -> None:
    logical = [HEADER, "x = 1", SEP, "", "y = 2"]
    payloads = {
        "lf": "\n".join(logical) + "\n",
        "lf_no_trailing_newline": "\n".join(logical),
        "crlf": "\r\n".join(logical) + "\r\n",
        "cr": "\r".join(logical) + "\r",
    }
    if variant == "bom":
        data = codecs.BOM_UTF8 + ("\n".join(logical) + "\n").encode("utf-8")
    else:
        data = payloads[variant].encode("utf-8")

    path = tmp_path / f"{variant}.py"
    path.write_bytes(data)
    result = parse_file(path)

    assert signature(result) == signature(parse_text(payloads["lf"], "py"))
    assert result.flat_lines == (HEADER, "x = 1", SEP, "", "y = 2")
    assert len(result.flat_lines) == 5
    assert result.cells[0].first_abs == 2
    assert result.cells[1].first_abs == 5
    assert result.path == str(path)


def test_form_feed_and_unicode_line_separators_do_not_split_lines() -> None:
    form_feed_line = "y = 'a\x0cb'"
    unicode_sep_line = "z = 'a\u2028b'"
    text = source(
        HEADER,
        "x = 1",
        form_feed_line,
        unicode_sep_line,
        SEP,
        "w = 1",
    )
    result = parse_text(text, "py")

    assert len(result.flat_lines) == 6
    assert result.flat_lines[2] == form_feed_line
    assert result.flat_lines[3] == unicode_sep_line
    assert result.cells[0].lines[1].raw == form_feed_line
    assert result.cells[0].lines[2].raw == unicode_sep_line
    assert result.cells[1].first_abs == 6


def test_tabs_and_non_ascii_are_preserved_verbatim() -> None:
    line = 's = "\t\u00e9\u4e2d\U0001f600"'
    result = parse_text(source(HEADER, line), "py")

    parsed = result.cells[0].lines[0]
    assert parsed.raw == line
    assert parsed.text == line


@pytest.mark.parametrize(
    ("command", "kind", "magic", "args"),
    [
        ("%md", CellKind.MARKDOWN, "md", None),
        ("%md-sandbox", CellKind.MARKDOWN, "md-sandbox", None),
        ("%md A title", CellKind.MARKDOWN, "md", "A title"),
        ("%sql", CellKind.SQL, "sql", None),
        ("%run ../shared/util", CellKind.RUN, "run", "../shared/util"),
        ('%run "../shared/util"', CellKind.RUN, "run", '"../shared/util"'),
        ("%sh", CellKind.MAGIC_OTHER, "sh", None),
        ("%pip install x", CellKind.MAGIC_OTHER, "pip", "install x"),
        ("%python", CellKind.PYTHON, "python", None),
    ],
)
def test_magic_classification_table(
    command: str, kind: CellKind, magic: str, args: str | None
) -> None:
    text = source(HEADER, f"# MAGIC {command}", "# MAGIC body")
    result = parse_text(text, "py")

    cell = result.cells[0]
    assert cell.kind is kind
    assert cell.magic == magic
    assert cell.magic_args == args
    assert cell.lines[0].raw == f"# MAGIC {command}"
    assert cell.lines[0].text == command
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ('%run "../shared/fin_utils"', "../shared/fin_utils"),
        ("%run '../shared/fin_utils'", "../shared/fin_utils"),
        ('%run "../shared/my utils"', "../shared/my utils"),
        ("%run ../shared/fin_utils", "../shared/fin_utils"),
        ("%run", None),
        ("%run ", None),
        ('%run "unterminated', '"unterminated'),
    ],
)
def test_run_target_removes_matching_quotes(command: str, target: str | None) -> None:
    result = parse_text(source(HEADER, f"# MAGIC {command}"), "py")

    assert result.cells[0].kind is CellKind.RUN
    assert result.cells[0].run_target == target


def test_magic_cell_without_a_percent_command_warns() -> None:
    result = parse_text(source(HEADER, "# MAGIC hello"), "py")

    cell = result.cells[0]
    assert cell.kind is CellKind.MAGIC_OTHER
    assert cell.magic is None
    assert cell.magic_args is None
    assert cell.lines[0].text == "hello"
    assert len(result.warnings) == 1
    assert result.warnings[0] == (
        "cell 1: magic cell has no %command; classified as magic without a name"
    )


def test_bare_magic_line_inside_markdown_cell_stays_in_place() -> None:
    text = source(
        HEADER,
        "# MAGIC %md",
        "# MAGIC hello",
        "# MAGIC",
        "# MAGIC world",
    )
    result = parse_text(text, "py")

    cell = result.cells[0]
    assert cell.kind is CellKind.MARKDOWN
    assert cell.magic == "md"
    assert [line.rel_no for line in cell.lines] == [1, 2, 3, 4]
    assert cell.lines[2].raw == "# MAGIC"
    assert cell.lines[2].text == ""
    assert result.warnings == ()


def test_dbtitle_is_a_real_line_and_cell_stays_magic() -> None:
    text = source(
        HEADER,
        "# DBTITLE 1,My setup",
        "# MAGIC %md",
        "# MAGIC hello",
    )
    result = parse_text(text, "py")

    cell = result.cells[0]
    assert cell.title == "My setup"
    assert cell.kind is CellKind.MARKDOWN
    assert cell.magic == "md"
    assert cell.lines[0].rel_no == 1
    assert cell.lines[0].raw == "# DBTITLE 1,My setup"
    assert cell.lines[0].text == "# DBTITLE 1,My setup"
    assert cell.lines[1].text == "%md"
    assert result.warnings == ()


def test_mixed_magic_and_code_uses_default_kind_with_warning() -> None:
    text = source(HEADER, "# MAGIC %md", "# MAGIC hello", "print(1)")
    result = parse_text(text, "py")

    cell = result.cells[0]
    assert cell.kind is CellKind.PYTHON
    assert cell.magic is None
    assert cell.lines[0].text == "# MAGIC %md"
    assert len(result.warnings) == 1
    assert "cell 1" in result.warnings[0]


def test_markdown_magic_first_line_is_rel_one_and_text_is_stripped() -> None:
    result = parse_text(source(HEADER, "# MAGIC %md", "# MAGIC hello"), "py")

    cell = result.cells[0]
    first = cell.lines[0]
    assert first.rel_no == 1
    assert first.abs_no == 2
    assert first.raw == "# MAGIC %md"
    assert first.text == "%md"
    assert cell.source == "%md\nhello"


def test_header_only_file_is_one_empty_cell() -> None:
    result = parse_text(source(HEADER), "py")

    assert result.flat_lines == (HEADER,)
    assert len(result.cells) == 1
    assert result.cells[0].kind is CellKind.EMPTY
    assert result.cells[0].lines == ()
    assert result.cells[0].first_abs is None
    assert result.warnings == ()


def test_empty_python_file_raises() -> None:
    with pytest.raises(NotebookFormatError, match="empty file"):
        parse_text("", "py")


def test_unreadable_file_raises_notebook_error(tmp_path: Path) -> None:
    path = tmp_path / "is-a-directory.py"
    path.mkdir()

    with pytest.raises(NotebookError, match="cannot read file"):
        parse_file(path)


@pytest.mark.parametrize("suffix", [".txt", ".sql", ".ipynb", ""])
def test_unsupported_suffix_names_the_one_supported_format(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"notebook{suffix}"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError) as excinfo:
        parse_file(path)
    message = str(excinfo.value)
    assert message == (
        f"{path}: unsupported notebook suffix {suffix or '<none>'!r}; "
        "supported suffix is .py"
    )


@pytest.mark.parametrize("fmt", ["sql", "ipynb", "txt"])
def test_unsupported_text_format_names_the_one_supported_format(fmt: str) -> None:
    with pytest.raises(UnsupportedFormatError) as excinfo:
        parse_text("hello", fmt)
    message = str(excinfo.value)
    assert message == (
        f"unsupported format {fmt!r}; only the Databricks .py format is supported"
    )


@pytest.mark.parametrize("fmt", ["py", ".py", "PY"])
def test_python_format_aliases_are_accepted(fmt: str) -> None:
    result = parse_text(source(HEADER, "x = 1"), fmt)

    assert result.format == "py"
    assert result.cells[0].lines[0].text == "x = 1"


def test_invalid_utf8_file_raises_with_byte_offset(tmp_path: Path) -> None:
    path = tmp_path / "latin1.py"
    path.write_bytes(b"# Databricks notebook source\n\xff\n")

    with pytest.raises(NotebookFormatError, match="not valid UTF-8 near byte 29"):
        parse_file(path)


def test_render_debug_table_is_stable_and_appends_warnings() -> None:
    result = parse_text(
        source(HEADER, "a = 1", SEP, "# MAGIC %sql", "# MAGIC select 1"),
        "py",
    )

    assert render_debug_table(result) == (
        "C1   python       abs 2-2        lines 1   a = 1\n"
        "C2   sql          abs 4-5        lines 2   %sql\n"
    )

    warned = parse_text(source(HEADER, "a = 1", SEP), "py")
    assert render_debug_table(warned) == (
        "C1   python       abs 2-2        lines 1   a = 1\n"
        "\n"
        "warning: line 3: dropped the empty final cell after a trailing "
        "separator\n"
    )


def test_main_prints_the_debug_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plain.py"
    path.write_text(source(HEADER, "x = 1"), encoding="utf-8")

    assert main([str(path)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == render_debug_table(parse_file(path))


def test_main_reports_parse_errors_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "plain.txt"
    path.write_text("hello", encoding="utf-8")

    assert main([str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "unsupported notebook suffix" in captured.err


def test_package_reexports_the_parser_without_eager_import() -> None:
    import harness

    assert harness.parse_file is parse_file
    assert harness.parse_text is parse_text
    missing = "missing"
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        getattr(harness, missing)
