from __future__ import annotations

from pathlib import Path

import pytest
from harness.citations import numbered_listing
from harness.models import CellKind, ParseResult
from harness.parser import parse_file

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cash_application_engine.py"


def load_real_notebook() -> ParseResult:
    if not FIXTURE.exists():
        pytest.fail(
            "copy the attached notebook to tests/fixtures/cash_application_engine.py"
        )
    return parse_file(FIXTURE)


def test_real_notebook_shape_and_cell_anchors() -> None:
    result = load_real_notebook()

    assert result.format == "py"
    assert result.path == str(FIXTURE)
    assert len(result.flat_lines) == 551
    assert len(result.cells) == 47
    assert result.flat_lines[0] == "# Databricks notebook source"
    assert result.flat_lines[-1] == (
        'dbutils.notebook.exit(json.dumps({"status": "ok", "rcpt": len(rp), '
        '"lines": len(ln), "bad": len(bad)}))'
    )

    cell1, cell4, cell26, cell29 = (
        result.cell(1),
        result.cell(4),
        result.cell(26),
        result.cell(29),
    )
    cell45, cell46, cell47 = result.cell(45), result.cell(46), result.cell(47)

    assert (cell1.first_abs, cell1.last_abs) == (2, 4)
    assert cell1.kind is CellKind.MARKDOWN
    assert cell1.magic == "md"
    assert cell1.source == (
        "%md\n## cash app v3\nnew logic from Q2 - dont use v2 notebook anymore"
    )

    assert (cell4.first_abs, cell4.last_abs) == (33, 40)
    assert cell4.kind is CellKind.PYTHON
    assert cell4.lines[0].raw == "TOL = 0.5"
    assert cell4.lines[-1].raw == 'spark.conf.set("spark.sql.shuffle.partitions", 8)'

    assert (cell26.first_abs, cell26.last_abs) == (286, 341)
    assert cell26.lines[30].abs_no == 316
    assert result.cell_rel_to_abs(26, 31) == 316
    assert result.abs_to_cell_rel(316) == (26, 31)

    assert (cell29.first_abs, cell29.last_abs) == (358, 362)
    assert result.abs_to_cell_rel(361) == (29, 4)
    assert cell29.lines[3].raw == "TOL = 5.0"

    assert (cell45.first_abs, cell45.last_abs) == (540, 541)
    assert cell45.kind is CellKind.SQL
    assert (cell46.first_abs, cell46.last_abs) == (545, 547)
    assert cell46.kind is CellKind.SQL
    assert (cell47.first_abs, cell47.last_abs) == (551, 551)
    assert cell47.kind is CellKind.PYTHON


def test_real_notebook_every_content_line_round_trips() -> None:
    result = load_real_notebook()

    for cell in result.cells:
        for line in cell.lines:
            assert result.cell_rel_to_abs(cell.number, line.rel_no) == line.abs_no
            assert result.abs_to_cell_rel(line.abs_no) == (cell.number, line.rel_no)
            assert result.flat_lines[line.abs_no - 1] == line.raw


def test_real_notebook_numbered_listing_for_c4_and_c29() -> None:
    result = load_real_notebook()

    assert numbered_listing(result, [4, 29]) == (
        "### C4 [python] file lines 33-40\n"
        "L1| TOL = 0.5\n"
        "L2| MAX_WO = 250.0\n"
        "L3| EX_TOL = 0.01\n"
        'L4| bk = {"DB-4471": "DK01", "HDFC-0932": "IN02", '
        '"DBS-1180": "SG01", "JPM-2290": "US01"}\n'
        'L5| EXCL = ["C10077", "C10412"]\n'
        'L6| tgt = "fin_cash.alloc_lines"\n'
        "L7| t0 = time.time()\n"
        'L8| spark.conf.set("spark.sql.shuffle.partitions", 8)\n'
        "\n"
        "### C29 [python] file lines 358-362\n"
        "L1| tot_in = rp.amt_usd.sum()\n"
        "L2| tot_open = ip.open_usd.sum()\n"
        "L3| print(round(tot_in, 2), round(tot_open, 2), "
        "round(tot_in / tot_open, 3))\n"
        "L4| TOL = 5.0\n"
        "L5| print(len(rp[rp.refs.apply(lambda z: z is not None and len(z) > 0)]))"
    )


def test_real_notebook_separator_lines_are_not_citable() -> None:
    result = load_real_notebook()

    separators = [
        abs_no
        for abs_no, raw in enumerate(result.flat_lines, start=1)
        if raw == "# COMMAND ----------"
    ]
    assert separators
    for abs_no in separators:
        assert result.abs_to_cell_rel(abs_no) is None
