from __future__ import annotations

from pathlib import Path

import pytest
from harness import extract_facts, parse_text, render_facts_for_prompt
from harness.facts import ConstantFact, StaticFacts
from harness.models import CellKind, ParseResult
from harness.parser import parse_file

HEADER = "# Databricks notebook source"
SEP = "# COMMAND ----------"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cash_application_engine.py"


def notebook(*cells: str) -> ParseResult:
    body = f"\n\n{SEP}\n\n".join(cells)
    return parse_text(f"{HEADER}\n\n{body}\n", "py")


def facts(*cells: str) -> StaticFacts:
    return extract_facts(notebook(*cells))


def constant(found: StaticFacts, name: str) -> ConstantFact:
    return next(item for item in found.constants if item.name == name)


def load_real_result() -> ParseResult:
    if not FIXTURE.exists():
        pytest.fail(
            "copy the attached notebook to tests/fixtures/cash_application_engine.py"
        )
    return parse_file(FIXTURE)


def load_real_notebook() -> StaticFacts:
    return extract_facts(load_real_result())


# -- bindings and constants ----------------------------------------------


def test_literal_assignments_become_constants_with_citations() -> None:
    found = facts(
        """
MAX = 10
ratio: float = 0.5
untyped: int
cfg = {"a": [1, 2]}
nothing = None
""",
        """
MAX = 20
ratio += 1
""",
    )

    max_fact = constant(found, "MAX")
    assert max_fact.reassigned is True
    assert max_fact.values_differ is True
    assert [b.cite.cell for b in max_fact.bindings] == [1, 2]
    assert [b.cite.start for b in max_fact.bindings] == [1, 1]
    assert [b.literal for b in max_fact.bindings] == [True, True]
    assert [b.value_repr for b in max_fact.bindings] == ["10", "20"]

    ratio = constant(found, "ratio")
    assert ratio.reassigned is True
    assert ratio.values_differ is False
    assert [b.kind for b in ratio.bindings] == ["assign", "augassign"]
    assert [b.value_repr for b in ratio.bindings] == ["0.5", None]

    assert constant(found, "cfg").bindings[0].value_repr == "{'a': [1, 2]}"
    assert constant(found, "nothing").bindings[0].value_repr == "None"
    assert "untyped" not in {item.name for item in found.constants}


def test_only_single_name_assignments_are_treated_as_literal_bindings() -> None:
    found = facts(
        """
a = b = 1
p, q = 1, 2
[first, *rest] = [1, 2, 3]
computed = 1 + 2
called = len("abc")
joined = "a" + "b"
""",
    )

    names = {item.name for item in found.constants}
    assert names.isdisjoint({"a", "b", "p", "q", "first", "rest", "computed", "called"})
    assert "joined" not in names


def test_module_bindings_are_found_through_compound_statements() -> None:
    found = facts(
        """
if True:
    BRANCH = 1
for looped in [1]:
    LOOPED = 2
while False:
    WHILE_NAME = 3
with open("x") as handle:
    WITH_NAME = 4
try:
    TRY_NAME = 5
    raise ValueError("x")
except Exception as exc:
    EXC_NAME = 6
else:
    ELSE_NAME = 7
finally:
    FINAL_NAME = 8
class Hidden:
    CLASS_NAME = 9
    def method(self):
        return undefined_in_class()
""",
    )

    names = {item.name for item in found.constants}
    assert {
        "BRANCH",
        "LOOPED",
        "WHILE_NAME",
        "WITH_NAME",
        "TRY_NAME",
        "EXC_NAME",
        "ELSE_NAME",
        "FINAL_NAME",
    } <= names
    assert "CLASS_NAME" not in names
    # Class bodies are not analysed at all, per the documented limitation.
    assert found.unresolved_calls == ()


def test_module_scan_handles_optional_compound_statement_bindings() -> None:
    found = facts(
        """
with open("x"):
    INSIDE_WITH = 1
try:
    raise ValueError()
except Exception:
    IN_HANDLER = 2
obj.attr += 1
arr[0] += 1
(WALRUS := 3)
WALRUS()
""",
    )

    assert [item.name for item in found.constants] == ["INSIDE_WITH", "IN_HANDLER"]
    # A walrus target is a module binding, so the later bare-name call is known.
    assert found.unresolved_calls == ()


def test_never_read_tracks_loads_anywhere_in_the_notebook() -> None:
    found = facts(
        """
UNUSED_VALUE = 1
USED_VALUE = 2
GLOBAL_READ = 3
EMPTY = 4
EMPTY = 4
print(USED_VALUE)
def show():
    return GLOBAL_READ
def shadow(UNUSED_VALUE):
    return UNUSED_VALUE
""",
    )

    assert constant(found, "UNUSED_VALUE").never_read is True
    assert constant(found, "USED_VALUE").never_read is False
    assert constant(found, "GLOBAL_READ").never_read is False
    empty = constant(found, "EMPTY")
    assert empty.reassigned is True
    assert empty.values_differ is False


def test_constants_are_ordered_by_their_first_binding() -> None:
    found = facts("B = 2\nA = 1", "C = 3")

    assert [item.name for item in found.constants] == ["B", "A", "C"]
    assert [item.bindings[0].cite.cell for item in found.constants] == [1, 1, 2]


# -- functions -----------------------------------------------------------


def test_function_facts_exclude_decorators_and_include_every_parameter() -> None:
    found = facts(
        """
def keep(decorator):
    return decorator

@keep
@keep
def decorated(a, /, b, *args, c=1, **kwargs):
    return a

async def worker(x):
    return x
""",
    )

    by_name = {item.name: item for item in found.functions}
    assert list(by_name) == ["keep", "decorated", "worker"]
    decorated = by_name["decorated"]
    assert decorated.cite.cell == 1
    assert (decorated.cite.start, decorated.cite.end) == (6, 7)
    assert decorated.params == ("a", "b", "args", "c", "kwargs")
    assert by_name["worker"].params == ("x",)
    assert by_name["keep"].params == ("decorator",)


def test_unused_functions_cover_recursion_decorated_and_nested_definitions() -> None:
    found = facts(
        """
def decorator(fn):
    return fn

@decorator
def decorated():
    return 1

def helper(value):
    return value

def caller():
    return helper(1)

def recursive(n):
    return recursive(n - 1)

def never_called():
    def nested():
        return 1
    return nested

def uses_local_lambda():
    action = lambda: 1
    return action()
""",
    )

    assert found.unused_functions == (
        "caller",
        "never_called",
        "recursive",
        "uses_local_lambda",
    )


def test_function_locals_and_builtins_are_not_invisible_dependencies() -> None:
    found = facts(
        """
def outer(a, /, b, *args, c=1, **kwargs):
    import json
    from math import ceil as ceiling
    try:
        local = json.dumps({})
    except ValueError as err:
        local = str(err)
    def nested(z):
        return z
    action = lambda w: w
    comp = [key for key in range(3)]
    text = ceiling(a) + b + len(args) + c + len(kwargs)
    return missing(local, nested, special_expr(), action, comp, text)

def special_expr():
    return unknown + 1
""",
    )

    calls = {item.name: item for item in found.unresolved_calls}
    assert set(calls) == {"missing"}
    assert calls["missing"].sites[0].cell == 1
    assert calls["missing"].sites[0].start == 13
    assert calls["missing"].run_targets == ()


def test_function_local_scan_skips_classes_and_star_imports() -> None:
    found = facts(
        """
def outer():
    from math import *
    class Hidden:
        value = 1
    return missing(ceil(1))
""",
    )

    assert {item.name for item in found.unresolved_calls} == {"ceil", "missing"}


def test_global_declarations_are_module_bindings_not_function_locals() -> None:
    found = facts(
        """
counter = {"n": 0}
def bump(step):
    global counter
    counter["n"] += step
    global created
    created = step
    return created
""",
    )

    assert [item.name for item in found.constants] == ["counter"]
    assert [(item.name, item.how) for item in found.mutations] == [
        ("counter", "subscript_augassign")
    ]

    # ``created`` is assigned through a global declaration, so a later call to
    # the same name is not an invisible dependency.
    later = facts(
        "def make():\n    global created\n    created = lambda: 1",
        "make()\ncreated()",
    )
    assert later.unresolved_calls == ()


# -- unresolved calls, %run and mutations --------------------------------


def test_unresolved_calls_ignore_bindings_builtins_and_platform_objects() -> None:
    found = facts(
        """
import pandas as pd
from math import sqrt as root

def local_fn():
    return 1

local_fn()
pd.DataFrame()
root(4)
print("ok")
display(spark.table("t"))
StringType()
StructField("a", IntegerType())
dbutils.notebook.exit("x")
""",
    )

    assert found.unresolved_calls == ()


def test_unresolved_calls_deduplicate_sites_and_attach_preceding_runs() -> None:
    found = facts(
        "# MAGIC %run ../shared/fin_utils",
        "get_gl_map(get_gl_map(1))\nnotify_ar('x')",
        "log_run('name')",
    )

    assert [
        (
            item.name,
            [(cite.cell, cite.start) for cite in item.sites],
            item.run_targets,
        )
        for item in found.unresolved_calls
    ] == [
        ("get_gl_map", [(2, 1)], ("../shared/fin_utils",)),
        ("notify_ar", [(2, 2)], ("../shared/fin_utils",)),
        ("log_run", [(3, 1)], ("../shared/fin_utils",)),
    ]


def test_a_run_after_the_call_is_not_an_origin_candidate() -> None:
    found = facts("unknown_one()", "# MAGIC %run ../later", "unknown_two()")

    calls = {item.name: item for item in found.unresolved_calls}
    assert calls["unknown_one"].run_targets == ()
    assert calls["unknown_two"].run_targets == ("../later",)


def test_mutations_only_track_module_names_from_inside_functions() -> None:
    found = facts(
        """
store = {}
items = []
name = "module"
total = 0
class Box:
    pass
box = Box()
store["module"] = 1

def touch(arg):
    store["a"] = 1
    store["a"] += 2
    del store["a"]
    items.append(1)
    items.sort()
    name = "local"
    arg["x"] = 1
    box.value = 2
    box.value += 1
    factory()["x"] = 3
    undeclared["x"] = 3
    total += 1

def shadow():
    store = {}
    store["a"] = 1
    return store
""",
    )

    assert [
        (item.function, item.name, item.how, item.cite.start)
        for item in found.mutations
    ] == [
        ("touch", "store", "subscript_store", 11),
        ("touch", "store", "subscript_augassign", 12),
        ("touch", "store", "subscript_del", 13),
        ("touch", "items", "method:append", 14),
        ("touch", "items", "method:sort", 15),
        ("touch", "box", "attr_store", 18),
        ("touch", "box", "attr_store", 19),
    ]


# -- widgets -------------------------------------------------------------


def test_widget_facts_read_positional_keyword_and_missing_values() -> None:
    found = facts(
        """
dbutils.widgets.text("src", "/mnt/in")
dbutils.widgets.text("blank")
dbutils.widgets.dropdown("co", "ALL", ["ALL", "DK"], "Country")
dbutils.widgets.combobox(name="combo", defaultValue="A", choices=("A", "B"))
dbutils.widgets.multiselect("multi", "x", ["x", "y"])
dbutils.widgets.text(dynamic_name, dynamic_default)
dbutils.widgets.get("src")
other.widgets.text("not", "a", "widget")
dbutils.text("plain")
dbutils.widgets.text("kw", "d", extra=1)
dbutils.widgets.text(name=dynamic_name)
""",
    )

    assert [
        (item.op, item.name, item.default, item.choices, item.cite.start)
        for item in found.widgets
    ] == [
        ("text", "src", "/mnt/in", None, 1),
        ("text", "blank", None, None, 2),
        ("dropdown", "co", "ALL", ("ALL", "DK"), 3),
        ("combobox", "combo", "A", ("A", "B"), 4),
        ("multiselect", "multi", "x", ("x", "y"), 5),
        ("text", None, None, None, 6),
        ("text", "kw", "d", None, 10),
        ("text", None, None, None, 11),
    ]


def test_widgets_only_have_choices_when_the_value_is_a_sequence() -> None:
    found = facts(
        'dbutils.widgets.dropdown("co", "ALL", SOME_CHOICES)',
        'dbutils.widgets.dropdown("co", "ALL", "not-a-list")',
    )

    assert [item.choices for item in found.widgets] == [None, None]


# -- tables --------------------------------------------------------------


def test_spark_reads_and_writes_are_resolved_from_literals_and_bindings() -> None:
    found = facts(
        """
TARGET = "main.gold.alloc"
PATH = "/mnt/out"
df = spark.read.format("delta").load("main.bronze.raw")
one = spark.table("main.gold.dim")
two = spark.read.table("main.gold.dim")
""",
        """
df.write.format("delta").mode("overwrite").option("x", "y").saveAsTable(TARGET)
df.write.mode("append").insertInto("main.gold.copy")
""",
        """
df.write.save(PATH)
df.write.csv(path_var)
df.write.partitionBy("dt").saveAsTable("main.gold.partitioned")
spark.read.load()
df.write.mode(mode_var).saveAsTable("main.gold.dynamic_mode")
""",
    )

    tables = [
        (
            item.direction,
            item.api,
            item.target,
            item.target_expr,
            item.mode,
            item.cite.cell,
            item.cite.start,
        )
        for item in found.tables
    ]
    assert tables == [
        ("read", "spark.read.load", "main.bronze.raw", None, None, 1, 3),
        ("read", "spark.table", "main.gold.dim", None, None, 1, 4),
        ("read", "spark.read.table", "main.gold.dim", None, None, 1, 5),
        ("write", "write.saveAsTable", "main.gold.alloc", None, "overwrite", 2, 1),
        ("write", "write.insertInto", "main.gold.copy", None, "append", 2, 2),
        ("write", "write.save", "/mnt/out", None, None, 3, 1),
        ("write", "write.csv", None, "path_var", None, 3, 2),
        (
            "write",
            "write.saveAsTable",
            "main.gold.partitioned",
            None,
            None,
            3,
            3,
        ),
        ("read", "spark.read.load", None, None, None, 3, 4),
        (
            "write",
            "write.saveAsTable",
            "main.gold.dynamic_mode",
            None,
            None,
            3,
            5,
        ),
    ]
    assert found.tables[3].resolved_from is not None
    assert found.tables[3].resolved_from.cell == 1
    assert found.tables[3].resolved_from.start == 1


def test_a_reassigned_or_computed_table_name_falls_back_to_the_expression() -> None:
    found = facts(
        """
TABLE = "main.one"
TABLE = "main.two"
COMPUTED = build_name()
spark.table(TABLE)
spark.table(COMPUTED)
spark.table("literal" + suffix)
""",
    )

    assert [(item.target, item.target_expr) for item in found.tables] == [
        (None, "TABLE"),
        (None, "COMPUTED"),
        (None, "'literal' + suffix"),
    ]
    assert all(item.resolved_from is None for item in found.tables)


def test_unrelated_read_and_write_calls_are_ignored() -> None:
    found = facts(
        """
other.read.csv("x")
spark.readStream.csv("x")
spark.read.csv
reader.saveAsTable("x")
spark.something.writeTable("x")
""",
    )

    assert found.tables == ()


# -- SQL -----------------------------------------------------------------


def test_sql_reads_joins_and_writes_preserve_citations_and_ignore_comments() -> None:
    found = facts(
        """
# MAGIC %sql
# MAGIC -- select * from hidden.line_comment
# MAGIC /* from hidden.block_comment */
# MAGIC SELECT s.id
# MAGIC FROM main.sales s
# MAGIC JOIN `catalog`.`schema`.`customers` c ON c.id = s.id
""",
        """
# MAGIC %sql
# MAGIC INSERT OVERWRITE TABLE main.gold.target
# MAGIC SELECT * FROM (SELECT * FROM main.bronze.source) x
""",
        """
# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMP VIEW recent AS
# MAGIC SELECT * FROM main.gold.target
""",
    )

    assert [
        (item.direction, item.api, item.target, item.cite.cell, item.cite.start)
        for item in found.tables
    ] == [
        ("read", "%sql", "main.sales", 1, 5),
        ("read", "%sql", "catalog.schema.customers", 1, 6),
        ("write", "%sql", "main.gold.target", 2, 2),
        ("read", "%sql", "main.bronze.source", 2, 3),
        ("write", "%sql", "recent", 3, 2),
        ("read", "%sql", "main.gold.target", 3, 3),
    ]


def test_sql_block_comments_span_lines_without_hiding_later_queries() -> None:
    found = facts(
        """
# MAGIC %sql
# MAGIC /* hidden
# MAGIC FROM hidden.table
# MAGIC */
# MAGIC SELECT *
# MAGIC FROM real.table
""",
    )

    assert [
        (item.direction, item.target, item.cite.start) for item in found.tables
    ] == [("read", "real.table", 6)]


def test_sql_insert_and_create_variants() -> None:
    found = facts(
        """
# MAGIC %sql
# MAGIC INSERT INTO t1 VALUES (1);
# MAGIC INSERT INTO TABLE t2 SELECT * FROM s2;
# MAGIC CREATE TABLE IF NOT EXISTS t3 (x INT);
# MAGIC CREATE OR REPLACE VIEW t4 AS SELECT * FROM s4;
""",
    )

    assert [
        (item.direction, item.target, item.cite.start) for item in found.tables
    ] == [
        ("write", "t1", 2),
        ("read", "s2", 3),
        ("write", "t2", 3),
        ("write", "t3", 4),
        ("read", "s4", 5),
        ("write", "t4", 5),
    ]


def test_a_python_cell_that_starts_with_a_magic_is_not_scanned_as_sql() -> None:
    found = facts("%sql\nSELECT * FROM not_a_sql_cell")

    assert found.tables == ()
    assert [(item.cell, item.line) for item in found.parse_issues] == [(1, 2)]


# -- parse issues and rendering -----------------------------------------


def test_unparseable_cells_are_reported_without_hiding_other_cells() -> None:
    found = facts(
        "VALUE = 1",
        "def broken(:\n    pass",
        "%python\nBAD =",
        "LATER = 2",
    )

    assert [(item.cell, item.line) for item in found.parse_issues] == [(2, 1), (3, 2)]
    assert all(item.message for item in found.parse_issues)
    assert [item.name for item in found.constants] == ["VALUE", "LATER"]
    assert "invalid syntax" in found.parse_issues[0].message


def _render_fixture() -> StaticFacts:
    return facts(
        """
TOL = 1
TOL = 2
MAX_WO = 3
TARGET = "db.in"
""",
        """
def unused():
    missing()
""",
        """
bal = {}
def mutate():
    bal["x"] = 1
""",
        """
dbutils.widgets.text("src", "/in")
dbutils.widgets.dropdown(dynamic, dynamic, dynamic)
spark.table(TARGET)
out.write.mode("append").saveAsTable("db.out")
spark.read.load()
""",
        "# MAGIC %run ../lib",
        "broken =",
    )


def test_render_facts_for_prompt_covers_every_fact_kind() -> None:
    text = render_facts_for_prompt(_render_fixture())

    assert (
        "FACT constant TOL: reassigned, values differ, never read, upper-case name"
        in text
    )
    assert "bindings [C1:L1]=1, [C1:L2]=2 [C1:L1] [C1:L2]" in text
    assert "FACT constant MAX_WO: never read, upper-case name" in text
    assert (
        "FACT unused_function unused: no reference outside its own body [C2:L1-2]"
        in text
    )
    assert "FACT unresolved_call missing: called at [C2:L2]; no %run before it" in text
    assert "FACT mutation bal: subscript_store in mutate [C3:L3]" in text
    assert "FACT widget text src: default '/in' [C4:L1]" in text
    assert "FACT widget dropdown ?: default None [C4:L2]" in text
    assert (
        "FACT table read spark.table: target 'db.in' (resolved from [C1:L4]) "
        "[C4:L3]" in text
    )
    assert (
        "FACT table write write.saveAsTable: target 'db.out', mode append [C4:L4]"
        in text
    )
    assert "FACT table read spark.read.load: target ? [C4:L5]" in text
    assert "FACT run ../lib: %run [C5:L1]" in text
    assert "FACT parse_issue C6: invalid syntax [C6:L1]" in text
    assert text.count("LIMIT ") == 4


def test_render_facts_for_prompt_filters_by_cell() -> None:
    found = _render_fixture()

    only_cell_two = render_facts_for_prompt(found, [2])
    assert "FACT unused_function unused" in only_cell_two
    assert "FACT unresolved_call missing" in only_cell_two
    assert "FACT constant" not in only_cell_two
    assert "FACT mutation" not in only_cell_two
    assert only_cell_two.count("LIMIT ") == 4

    only_cell_five = render_facts_for_prompt(found, (5,))
    assert only_cell_five.splitlines()[:1] == ["FACT run ../lib: %run [C5:L1]"]
    assert "FACT constant" not in only_cell_five

    only_parse_issue = render_facts_for_prompt(found, [6])
    assert only_parse_issue.splitlines()[:1] == [
        "FACT parse_issue C6: invalid syntax [C6:L1]"
    ]

    nothing = render_facts_for_prompt(found, [])
    assert len(nothing.splitlines()) == 4
    assert all(line.startswith("LIMIT ") for line in nothing.splitlines())


def test_render_names_preceding_runs_and_literal_widget_choices() -> None:
    found = facts(
        "# MAGIC %run ../lib",
        "missing()",
        'dbutils.widgets.dropdown("co", "ALL", ["A", "B"])',
    )

    text = render_facts_for_prompt(found)
    assert (
        "FACT unresolved_call missing: called at [C2:L1]; possible from %run ../lib"
        in text
    )
    assert "FACT widget dropdown co: default 'ALL', choices 'A', 'B' [C3:L1]" in text


def test_render_shows_question_mark_for_a_non_literal_rebinding() -> None:
    found = facts("VALUE = 1", "VALUE = call()")

    text = render_facts_for_prompt(found)
    assert "bindings [C1:L1]=1, [C2:L1]=?" in text


def test_render_omits_an_empty_constant_flag_list() -> None:
    text = render_facts_for_prompt(facts("lower = 'x'", "print(lower)"))

    assert "FACT constant lower: bindings [C1:L1]='x' [C1:L1]" in text
    assert ": ;" not in text


# -- the real cash application notebook ---------------------------------


def test_real_notebook_functions_constants_and_unused_code() -> None:
    result = load_real_result()
    found = extract_facts(result)

    assert sum(cell.kind is CellKind.PYTHON for cell in result.cells) == 43
    assert [
        (item.name, item.cite.cell, item.cite.start, item.cite.end)
        for item in found.functions
    ] == [
        ("nrm", 10, 1, 9),
        ("get_refs", 10, 13, 16),
        ("alloc_by_aging", 23, 1, 20),
        ("aging_rpt", 23, 22, 26),
        ("apply_ref", 26, 1, 15),
        ("apply_exact", 26, 17, 29),
        ("apply_fifo", 26, 31, 43),
        ("do_wo", 26, 45, 50),
        ("check_bal", 26, 52, 56),
        ("_gl", 33, 4, 6),
    ]
    assert found.functions[2].params == ("pdf_inv", "amt", "asof", "buckets")
    assert found.unused_functions == ("alloc_by_aging",)

    tol = constant(found, "TOL")
    assert [
        (binding.cite.cell, binding.cite.start, binding.value_repr)
        for binding in tol.bindings
    ] == [(4, 1, "0.5"), (29, 4, "5.0")]
    assert (tol.reassigned, tol.values_differ, tol.never_read) == (True, True, False)

    max_wo = constant(found, "MAX_WO")
    assert (max_wo.never_read, max_wo.bindings[0].value_repr) == (True, "250.0")
    assert len(found.constants) == 11


def test_real_notebook_unresolved_calls_mutations_and_widgets() -> None:
    found = load_real_notebook()

    assert [
        (
            item.name,
            [(cite.cell, cite.start) for cite in item.sites],
            item.run_targets,
        )
        for item in found.unresolved_calls
    ] == [
        ("get_gl_map", [(33, 3)], ("../shared/fin_utils",)),
        ("notify_ar", [(43, 11)], ("../shared/fin_utils",)),
        ("log_run", [(44, 1)], ("../shared/fin_utils",)),
    ]

    assert [
        (item.function, item.name, item.how, item.cite.cell, item.cite.start)
        for item in found.mutations
    ] == [
        ("apply_ref", "bal", "subscript_store", 26, 12),
        ("apply_exact", "bal", "subscript_store", 26, 26),
        ("apply_fifo", "bal", "subscript_store", 26, 40),
        ("do_wo", "bal", "subscript_store", 26, 50),
    ]

    assert [
        (item.op, item.name, item.default, item.choices, item.cite.start)
        for item in found.widgets
    ] == [
        ("text", "src", "/mnt/fin-raw/cashapp/", None, 1),
        ("text", "run_dt", "", None, 2),
        ("dropdown", "co", "ALL", ("ALL", "DK01", "IN02", "SG01", "US01"), 3),
        ("dropdown", "dry", "Y", ("Y", "N"), 4),
    ]
    assert {item.cite.cell for item in found.widgets} == {3}


def test_real_notebook_spark_and_sql_tables_and_run() -> None:
    found = load_real_notebook()

    assert len(found.tables) == 10
    reads = [item for item in found.tables if item.direction == "read"]
    assert len(reads) == 7
    assert [(item.api, item.cite.cell, item.cite.start) for item in reads[:5]] == [
        ("spark.read.csv", 6, 1),
        ("spark.read.csv", 8, 1),
        ("spark.read.csv", 9, 1),
        ("spark.read.csv", 13, 8),
        ("spark.read.csv", 16, 1),
    ]

    writes = [item for item in found.tables if item.direction == "write"]
    assert len(writes) == 3
    first_write = writes[0]
    assert (first_write.api, first_write.target, first_write.mode) == (
        "write.saveAsTable",
        "fin_cash.alloc_lines",
        "overwrite",
    )
    assert first_write.resolved_from is not None
    assert (first_write.resolved_from.cell, first_write.resolved_from.start) == (4, 6)
    assert (first_write.cite.cell, first_write.cite.start) == (37, 2)
    assert [(item.target, item.mode) for item in writes[1:]] == [
        ("fin_cash.open_items_after", "overwrite"),
        ("fin_cash.ar_aging_snap", "append"),
    ]

    sql_reads = [item for item in reads if item.api == "%sql"]
    assert [(item.target, item.cite.cell, item.cite.start) for item in sql_reads] == [
        ("fin_cash.alloc_lines", 45, 2),
        ("fin_cash.alloc_lines", 46, 3),
    ]

    assert [(item.target, item.cite.cell, item.cite.start) for item in found.runs] == [
        ("../shared/fin_utils", 5, 1)
    ]
    assert found.parse_issues == ()
