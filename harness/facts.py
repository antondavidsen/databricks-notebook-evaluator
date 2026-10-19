"""Static, LLM-free facts about a Databricks Python notebook.

Module-level bindings are found by walking statements through
``if``/``for``/``while``/``try``/``with`` bodies and stopping at ``def`` and
``class``. A function's locals are approximated as its parameters plus every
name stored anywhere below it (nested defs, lambdas, comprehensions, imports),
minus names declared ``global``. That one-scope approximation is stated in the
limitations rather than hidden.
"""

from __future__ import annotations

import ast
import builtins
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .citations import Citation, format_citation
from .models import Cell, CellKind, ParseResult

PYSPARK_TYPES = frozenset(
    {
        "StringType",
        "IntegerType",
        "LongType",
        "DoubleType",
        "FloatType",
        "BooleanType",
        "DateType",
        "TimestampType",
        "DecimalType",
        "ArrayType",
        "MapType",
        "StructType",
        "StructField",
        "ShortType",
        "ByteType",
        "BinaryType",
        "NullType",
        "DataType",
    }
)

_EXEMPT_CALLS = frozenset(
    {"spark", "sc", "sqlContext", "dbutils", "display", "displayHTML"}
)
_MUTATING_METHODS = frozenset(
    {
        "append",
        "extend",
        "insert",
        "pop",
        "remove",
        "clear",
        "update",
        "add",
        "discard",
        "setdefault",
        "sort",
        "reverse",
        "popitem",
    }
)
_WIDGET_OPS = frozenset({"text", "dropdown", "combobox", "multiselect"})
_READ_METHODS = frozenset({"csv", "parquet", "json", "orc", "load", "table"})
_WRITE_METHODS = frozenset(
    {"saveAsTable", "insertInto", "save", "csv", "parquet", "json", "orc"}
)
_LIMITATIONS = (
    "Dynamic access (eval, exec, globals(), getattr) is not analysed.",
    "Globals passed into functions as arguments are aliases and are not tracked.",
    "Classes are ignored and nested scopes are approximated.",
    "Names from %run notebooks or star imports other than pyspark.sql.types are "
    "reported as candidates, not facts; SQL CTEs, MERGE, UPDATE and DELETE are "
    "not scanned.",
)

_NO_VALUE = object()


@dataclass(frozen=True, slots=True)
class Binding:
    """One module-level binding of a name."""

    name: str
    kind: str
    cite: Citation
    literal: bool
    value_repr: str | None


@dataclass(frozen=True, slots=True)
class FunctionFact:
    """A module-level function; the citation excludes decorators."""

    name: str
    cite: Citation
    params: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConstantFact:
    """A module-level name with at least one literal binding."""

    name: str
    bindings: tuple[Binding, ...]
    reassigned: bool
    values_differ: bool
    never_read: bool


@dataclass(frozen=True, slots=True)
class WidgetFact:
    """A dbutils widget declaration with literal values where available."""

    op: str
    name: str | None
    default: object
    choices: tuple[object, ...] | None
    cite: Citation


@dataclass(frozen=True, slots=True)
class TableFact:
    """A read or write of a table, resolved from literals where possible."""

    direction: str
    api: str
    target: str | None
    target_expr: str | None
    resolved_from: Citation | None
    mode: str | None
    cite: Citation


@dataclass(frozen=True, slots=True)
class UnresolvedCall:
    """A bare-name call that has no local, module or builtin definition."""

    name: str
    sites: tuple[Citation, ...]
    run_targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Mutation:
    """A mutation of a module-level name from inside a function."""

    function: str
    name: str
    how: str
    cite: Citation


@dataclass(frozen=True, slots=True)
class RunFact:
    """A ``%run`` cell."""

    target: str | None
    cite: Citation


@dataclass(frozen=True, slots=True)
class ParseIssue:
    """A cell that could not be parsed, addressed by cell and relative line."""

    cell: int
    message: str
    line: int | None


@dataclass(frozen=True, slots=True)
class StaticFacts:
    """Every fact stage 2 can establish without an LLM."""

    functions: tuple[FunctionFact, ...]
    constants: tuple[ConstantFact, ...]
    widgets: tuple[WidgetFact, ...]
    tables: tuple[TableFact, ...]
    runs: tuple[RunFact, ...]
    unresolved_calls: tuple[UnresolvedCall, ...]
    unused_functions: tuple[str, ...]
    mutations: tuple[Mutation, ...]
    parse_issues: tuple[ParseIssue, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Bound:
    """A public binding plus the evaluator's literal value, if any."""

    binding: Binding
    value: object


@dataclass(frozen=True, slots=True)
class _Function:
    """A module-level function plus its approximated local names."""

    fact: FunctionFact
    locals: frozenset[str]
    decorated: bool


def _cite(cell: int, node: ast.AST) -> Citation:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", None) or start
    return Citation(cell, start, end)


def _literal(node: ast.expr) -> tuple[bool, object]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return False, None


def _target_names(target: ast.expr) -> Iterator[ast.Name]:
    if isinstance(target, ast.Name):
        yield target
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _arg_names(args: ast.arguments) -> tuple[str, ...]:
    names = [arg.arg for arg in args.posonlyargs + args.args]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return tuple(names)


def _function_locals(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    local = set(_arg_names(node.args))
    declared_global: set[str] = set()
    pending: list[ast.AST] = list(node.body)
    while pending:
        item = pending.pop()
        if isinstance(item, ast.ClassDef):
            continue
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local.add(item.name)
            local.update(_arg_names(item.args))
            pending.extend(item.body)
            continue
        if isinstance(item, ast.Lambda):
            local.update(_arg_names(item.args))
            pending.append(item.body)
            continue
        if isinstance(item, ast.Global):
            declared_global.update(item.names)
            continue
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            local.add(item.id)
        elif isinstance(item, ast.Import):
            for alias in item.names:
                local.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(item, ast.ImportFrom):
            for alias in item.names:
                if alias.name != "*":
                    local.add(alias.asname or alias.name)
        elif isinstance(item, ast.ExceptHandler) and item.name is not None:
            local.add(item.name)
        pending.extend(ast.iter_child_nodes(item))
    return frozenset(local - declared_global)


def _module_stmts(stmts: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield module-level statements, descending through compound statements."""
    for stmt in stmts:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            children = getattr(stmt, field, ())
            yield from _module_stmts(
                child for child in children if isinstance(child, ast.stmt)
            )
        for handler in getattr(stmt, "handlers", ()):
            yield from _module_stmts(handler.body)


def _expr_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Visit expression children of one statement without entering its body."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.stmt, ast.Lambda)):
            continue
        yield child
        yield from _expr_nodes(child)


def _chain(call: ast.Call) -> tuple[ast.expr, list[str], list[ast.Call]]:
    """Walk ``call.func`` down to its root, keeping attributes and calls."""
    attributes: list[str] = []
    calls: list[ast.Call] = []
    node: ast.expr = call.func
    while True:
        if isinstance(node, ast.Attribute):
            attributes.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Call):
            calls.append(node)
            node = node.func
        else:
            attributes.reverse()
            calls.reverse()
            return node, attributes, calls


def _widget_op(func: ast.expr) -> str | None:
    if not isinstance(func, ast.Attribute) or func.attr not in _WIDGET_OPS:
        return None
    widgets = func.value
    if not isinstance(widgets, ast.Attribute) or widgets.attr != "widgets":
        return None
    root = widgets.value
    if isinstance(root, ast.Name) and root.id == "dbutils":
        return func.attr
    return None


def _mode_in(calls: Iterable[ast.Call]) -> str | None:
    for call in calls:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "mode" and call.args:
            ok, value = _literal(call.args[0])
            if ok and isinstance(value, str):
                return value
    return None


_SQL_IDENT = r"(?:`[^`\n]+`|[A-Za-z_][\w$]*)"
_SQL_NAME = rf"{_SQL_IDENT}(?:\s*\.\s*{_SQL_IDENT})*"


class _Collector:
    def __init__(self) -> None:
        self.bindings: dict[str, list[_Bound]] = {}
        self.functions: dict[int, _Function] = {}
        self.function_order: list[_Function] = []
        self.loads: list[tuple[str, str | None]] = []
        self.unresolved: dict[str, list[Citation]] = {}
        self.mutations: list[Mutation] = []
        self.widgets: list[WidgetFact] = []
        self.tables: list[TableFact] = []
        self.runs: list[RunFact] = []
        self.issues: list[ParseIssue] = []
        self.skipped_mutations: set[int] = set()

    def add_cell(self, cell: Cell) -> None:
        cell_no = cell.number
        if cell.kind is CellKind.PYTHON:
            lines = cell.source.split("\n")
            if lines and (lines[0].startswith("%") or cell.magic == "python"):
                lines[0] = ""
            try:
                tree = ast.parse("\n".join(lines))
            except (SyntaxError, ValueError) as exc:
                line = exc.lineno if isinstance(exc, SyntaxError) else None
                message = exc.msg if isinstance(exc, SyntaxError) else str(exc)
                self.issues.append(ParseIssue(cell_no, message, line))
                return
            self._collect_module(cell_no, tree)
            self._analyse_module(cell_no, tree)
        elif cell.kind is CellKind.RUN and cell.run_target is not None:
            self.runs.append(RunFact(cell.run_target, Citation(cell_no, 1, 1)))
        elif cell.kind is CellKind.SQL:
            self._scan_sql(cell_no, cell.source)

    # -- phase 2: module bindings --------------------------------------
    def _add(
        self, name: str, kind: str, cite: Citation, value: object = _NO_VALUE
    ) -> None:
        literal = value is not _NO_VALUE
        binding = Binding(
            name,
            kind,
            cite,
            literal,
            repr(value)[:80] if literal else None,
        )
        self.bindings.setdefault(name, []).append(_Bound(binding, value))

    def _add_name(
        self, target: ast.Name, kind: str, cell_no: int, value: object = _NO_VALUE
    ) -> None:
        self._add(target.id, kind, _cite(cell_no, target), value)

    def _collect_module(self, cell_no: int, tree: ast.Module) -> None:
        for stmt in _module_stmts(tree.body):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cite = _cite(cell_no, stmt)
                self._add(stmt.name, "def", cite)
                fact = FunctionFact(stmt.name, cite, _arg_names(stmt.args))
                info = _Function(
                    fact, _function_locals(stmt), bool(stmt.decorator_list)
                )
                self.functions[id(stmt)] = info
                self.function_order.append(info)
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Global):
                        for name in node.names:
                            self._add(name, "global", _cite(cell_no, node))
            elif not isinstance(stmt, ast.ClassDef):
                self._collect_stmt(cell_no, stmt)

    def _collect_stmt(self, cell_no: int, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            single_name = len(stmt.targets) == 1 and isinstance(
                stmt.targets[0], ast.Name
            )
            value = self._literal_value(stmt.value) if single_name else _NO_VALUE
            for target in stmt.targets:
                self._collect_assign_target(cell_no, target, value)
        elif isinstance(stmt, ast.AnnAssign):
            value = _NO_VALUE if stmt.value is None else self._literal_value(stmt.value)
            self._collect_assign_target(cell_no, stmt.target, value)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                self._add_name(stmt.target, "augassign", cell_no)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._collect_loop_target(cell_no, stmt.target)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    self._collect_other_target(cell_no, item.optional_vars)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                name = alias.asname or alias.name.split(".")[0]
                self._add(name, "import", _cite(cell_no, alias))
        elif isinstance(stmt, ast.ImportFrom):
            cite = _cite(cell_no, stmt)
            for alias in stmt.names:
                if alias.name == "*":
                    self._add(stmt.module or "", "star_import", cite)
                else:
                    self._add(alias.asname or alias.name, "import", cite)
        elif isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                if handler.name is not None:
                    self._add(handler.name, "other", _cite(cell_no, handler))
        for node in _expr_nodes(stmt):
            if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
                self._add_name(node.target, "other", cell_no)

    def _literal_value(self, node: ast.expr) -> object:
        ok, value = _literal(node)
        return value if ok else _NO_VALUE

    def _collect_assign_target(
        self, cell_no: int, target: ast.expr, value: object
    ) -> None:
        if isinstance(target, ast.Name):
            self._add_name(target, "assign", cell_no, value)
        else:
            self._collect_other_target(cell_no, target)

    def _collect_loop_target(self, cell_no: int, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self._add_name(target, "for", cell_no)
        else:
            self._collect_other_target(cell_no, target)

    def _collect_other_target(self, cell_no: int, target: ast.expr) -> None:
        for name in _target_names(target):
            self._add_name(name, "other", cell_no)

    # -- phase 3: loads, calls, mutations, widgets, tables --------------
    def _analyse_module(self, cell_no: int, tree: ast.Module) -> None:
        self.skipped_mutations = set()
        self._visit(tree, None, frozenset(), cell_no)

    def _visit(
        self,
        node: ast.AST,
        function: str | None,
        local_names: frozenset[str],
        cell_no: int,
    ) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if function is None:
                info = self.functions[id(node)]
                for decorator in node.decorator_list:
                    self._visit(decorator, None, frozenset(), cell_no)
                for stmt in node.body:
                    self._visit(stmt, info.fact.name, info.locals, cell_no)
            else:
                for stmt in node.body:
                    self._visit(stmt, function, local_names, cell_no)
            return
        if isinstance(node, ast.ClassDef):
            return
        if isinstance(node, ast.Lambda):
            lambda_names = local_names | frozenset(_arg_names(node.args))
            for child in ast.iter_child_nodes(node):
                self._visit(child, function, lambda_names, cell_no)
            return
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            comp_names = set(local_names)
            for generator in node.generators:
                comp_names.update(name.id for name in _target_names(generator.target))
            frozen = frozenset(comp_names)
            for child in ast.iter_child_nodes(node):
                self._visit(child, function, frozen, cell_no)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.loads.append((node.id, function))
        if isinstance(node, ast.Call):
            self._handle_call(node, function, local_names, cell_no)
        if function is not None:
            self._handle_mutation(node, function, local_names, cell_no)
        for child in ast.iter_child_nodes(node):
            self._visit(child, function, local_names, cell_no)

    def _handle_call(
        self,
        call: ast.Call,
        function: str | None,
        local_names: frozenset[str],
        cell_no: int,
    ) -> None:
        op = _widget_op(call.func)
        if op is not None:
            self._handle_widget(op, call, cell_no)
        self._handle_table(call, cell_no)
        if isinstance(call.func, ast.Name):
            name = call.func.id
            if (
                name not in local_names
                and name not in self.bindings
                and not hasattr(builtins, name)
                and name not in _EXEMPT_CALLS
                and name not in PYSPARK_TYPES
            ):
                sites = self.unresolved.setdefault(name, [])
                cite = _cite(cell_no, call)
                if cite not in sites:
                    sites.append(cite)

    def _handle_widget(self, op: str, call: ast.Call, cell_no: int) -> None:
        keys = (
            ("name", "defaultValue")
            if op == "text"
            else ("name", "defaultValue", "choices")
        )
        values: dict[str, object] = {}
        for key, arg in zip(keys, call.args, strict=False):
            ok, value = _literal(arg)
            if ok:
                values[key] = value
        for keyword in call.keywords:
            if keyword.arg in ("name", "defaultValue", "choices"):
                ok, value = _literal(keyword.value)
                if ok:
                    values[keyword.arg] = value
        name = values.get("name")
        choices = values.get("choices")
        self.widgets.append(
            WidgetFact(
                op,
                name if isinstance(name, str) else None,
                values.get("defaultValue"),
                tuple(choices) if isinstance(choices, (list, tuple)) else None,
                _cite(cell_no, call),
            )
        )

    def _handle_table(self, call: ast.Call, cell_no: int) -> None:
        root, attributes, chain_calls = _chain(call)
        if not attributes:
            return
        method = attributes[-1]
        direction: str | None = None
        api: str | None = None
        if (
            method in _READ_METHODS
            and isinstance(root, ast.Name)
            and root.id == "spark"
        ):
            if "read" in attributes:
                direction, api = "read", f"spark.read.{method}"
            elif method == "table" and attributes == ["table"]:
                direction, api = "read", "spark.table"
        elif method in _WRITE_METHODS and "write" in attributes:
            direction, api = "write", f"write.{method}"
        if direction is None or api is None:
            return
        target, target_expr, resolved = self._table_target(call)
        self.tables.append(
            TableFact(
                direction,
                api,
                target,
                target_expr,
                resolved,
                _mode_in(chain_calls),
                _cite(cell_no, call),
            )
        )

    def _table_target(
        self, call: ast.Call
    ) -> tuple[str | None, str | None, Citation | None]:
        if not call.args:
            return None, None, None
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value, None, None
        if isinstance(argument, ast.Name):
            bounds = self.bindings.get(argument.id, [])
            if (
                len(bounds) == 1
                and isinstance(bounds[0].value, str)
                and bounds[0].binding.literal
            ):
                return bounds[0].value, None, bounds[0].binding.cite
        return None, ast.unparse(argument), None

    def _handle_mutation(
        self,
        node: ast.AST,
        function: str,
        local_names: frozenset[str],
        cell_no: int,
    ) -> None:
        if isinstance(node, ast.AugAssign):
            self.skipped_mutations.add(id(node.target))
            if isinstance(node.target, ast.Subscript):
                self._mutation(
                    function,
                    node.target,
                    "subscript_augassign",
                    local_names,
                    cell_no,
                )
            elif isinstance(node.target, ast.Attribute):
                self._mutation(
                    function, node.target, "attr_store", local_names, cell_no
                )
        elif isinstance(node, ast.Subscript):
            if id(node) not in self.skipped_mutations and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                how = (
                    "subscript_store"
                    if isinstance(node.ctx, ast.Store)
                    else "subscript_del"
                )
                self._mutation(function, node, how, local_names, cell_no)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and id(node) not in self.skipped_mutations
        ):
            self._mutation(function, node, "attr_store", local_names, cell_no)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATING_METHODS
        ):
            self._mutation(
                function,
                node.func.value,
                f"method:{node.func.attr}",
                local_names,
                cell_no,
            )

    def _mutation(
        self,
        function: str,
        target: ast.expr,
        how: str,
        local_names: frozenset[str],
        cell_no: int,
    ) -> None:
        root = target
        while isinstance(root, (ast.Subscript, ast.Attribute)):
            root = root.value
        if not isinstance(root, ast.Name):
            return
        if root.id in local_names or root.id not in self.bindings:
            return
        self.mutations.append(Mutation(function, root.id, how, _cite(cell_no, root)))

    # -- SQL -----------------------------------------------------------
    def _scan_sql(self, cell_no: int, source: str) -> None:
        lines = source.split("\n")
        lines[0] = ""
        text = _strip_sql_comments("\n".join(lines))
        patterns = (
            (
                "read",
                "%sql",
                re.compile(rf"\b(?:FROM|JOIN)\s+(?!\()({_SQL_NAME})", re.IGNORECASE),
            ),
            (
                "write",
                "%sql",
                re.compile(
                    r"\bINSERT\s+(?:INTO(?:\s+TABLE)?|OVERWRITE(?:\s+TABLE)?)"
                    rf"\s+({_SQL_NAME})",
                    re.IGNORECASE,
                ),
            ),
            (
                "write",
                "%sql",
                re.compile(
                    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?"
                    r"(?:(?:TEMP|TEMPORARY)\s+)?(?:TABLE|VIEW)\s+"
                    rf"(?:IF\s+NOT\s+EXISTS\s+)?({_SQL_NAME})",
                    re.IGNORECASE,
                ),
            ),
        )
        for direction, api, pattern in patterns:
            for match in pattern.finditer(text):
                name = _clean_sql_name(match.group(1))
                line = text.count("\n", 0, match.start(1)) + 1
                cite = Citation(cell_no, line, line)
                self.tables.append(
                    TableFact(direction, api, name, None, None, None, cite)
                )

    # -- result --------------------------------------------------------
    def build(self) -> StaticFacts:
        function_locals = {info.fact.name: info.locals for info in self.function_order}
        loaded = {
            name
            for name, owner in self.loads
            if owner is None or name not in function_locals.get(owner, frozenset())
        }
        constants = tuple(
            sorted(
                (
                    ConstantFact(
                        name,
                        tuple(bound.binding for bound in bounds),
                        len(bounds) > 1,
                        len(
                            {
                                bound.binding.value_repr
                                for bound in bounds
                                if bound.binding.literal
                            }
                        )
                        > 1,
                        name not in loaded,
                    )
                    for name, bounds in self.bindings.items()
                    if any(bound.binding.literal for bound in bounds)
                ),
                key=lambda fact: (
                    fact.bindings[0].cite.cell,
                    fact.bindings[0].cite.start,
                    fact.name,
                ),
            )
        )
        functions = tuple(
            sorted(
                (info.fact for info in self.function_order),
                key=lambda fact: (fact.cite.cell, fact.cite.start, fact.name),
            )
        )
        unused = tuple(
            sorted(
                info.fact.name
                for info in self.function_order
                if not info.decorated
                and not any(
                    name == info.fact.name and owner != info.fact.name
                    for name, owner in self.loads
                )
            )
        )
        unresolved = tuple(
            sorted(
                (
                    UnresolvedCall(name, tuple(sorted(sites, key=_cite_key)), targets)
                    for name, sites in self.unresolved.items()
                    for targets in (self._run_targets(sites),)
                ),
                key=lambda fact: (
                    fact.sites[0].cell,
                    fact.sites[0].start,
                    fact.name,
                ),
            )
        )
        return StaticFacts(
            functions=functions,
            constants=constants,
            widgets=tuple(sorted(self.widgets, key=_widget_key)),
            tables=tuple(sorted(self.tables, key=_table_key)),
            runs=tuple(sorted(self.runs, key=lambda fact: _cite_key(fact.cite))),
            unresolved_calls=unresolved,
            unused_functions=unused,
            mutations=tuple(
                sorted(
                    self.mutations,
                    key=lambda fact: (*_cite_key(fact.cite), fact.name),
                )
            ),
            parse_issues=tuple(
                sorted(
                    self.issues,
                    key=lambda issue: (issue.cell, issue.line or 0, issue.message),
                )
            ),
            limitations=_LIMITATIONS,
        )

    def _run_targets(self, sites: list[Citation]) -> tuple[str, ...]:
        targets: list[str] = []
        for run in self.runs:
            if (
                run.target is not None
                and any(run.cite.cell < site.cell for site in sites)
                and run.target not in targets
            ):
                targets.append(run.target)
        return tuple(targets)


def _strip_sql_comments(text: str) -> str:
    """Blank SQL comments without moving any character or newline."""
    out = list(text)
    index = 0
    in_line = False
    in_block = False
    while index < len(text):
        char = text[index]
        pair = text[index : index + 2]
        if in_line:
            if char == "\n":
                in_line = False
            else:
                out[index] = " "
        elif in_block:
            if pair == "*/":
                out[index] = out[index + 1] = " "
                index += 1
                in_block = False
            elif char != "\n":
                out[index] = " "
        elif pair == "--":
            out[index] = out[index + 1] = " "
            index += 1
            in_line = True
        elif pair == "/*":
            out[index] = out[index + 1] = " "
            index += 1
            in_block = True
        index += 1
    return "".join(out)


def _clean_sql_name(name: str) -> str:
    return re.sub(r"\s*\.\s*", ".", name).replace("`", "").strip()


def _cite_key(cite: Citation) -> tuple[int, int]:
    return cite.cell, cite.start


def _widget_key(fact: WidgetFact) -> tuple[int, int, str]:
    return fact.cite.cell, fact.cite.start, fact.name or ""


def _table_key(fact: TableFact) -> tuple[int, int, str]:
    return fact.cite.cell, fact.cite.start, fact.target or fact.target_expr or ""


def extract_facts(result: ParseResult) -> StaticFacts:
    """Extract every deterministic fact from a parsed notebook."""
    collector = _Collector()
    for cell in result.cells:
        collector.add_cell(cell)
    return collector.build()


def _allowed(citations: Iterable[Citation], cells: frozenset[int] | None) -> bool:
    return cells is None or any(cite.cell in cells for cite in citations)


def render_facts_for_prompt(
    facts: StaticFacts, cell_numbers: Iterable[int] | None = None
) -> str:
    """Render one prompt line per fact, optionally limited to given cells."""
    cells = None if cell_numbers is None else frozenset(cell_numbers)
    lines: list[str] = []

    for constant in facts.constants:
        flags = [
            flag
            for flag, on in (
                ("reassigned", constant.reassigned),
                ("values differ", constant.values_differ),
                ("never read", constant.never_read),
                ("upper-case name", constant.name.isupper()),
            )
            if on
        ]
        citations = [binding.cite for binding in constant.bindings]
        if not _allowed(citations, cells):
            continue
        shown = ", ".join(
            f"{format_citation(binding.cite)}="
            f"{binding.value_repr if binding.value_repr is not None else '?'}"
            for binding in constant.bindings
        )
        cite_text = " ".join(format_citation(cite) for cite in citations)
        label = ", ".join(flags)
        lines.append(
            f"FACT constant {constant.name}: "
            f"{label + '; ' if label else ''}"
            f"bindings {shown} {cite_text}"
        )

    for name in facts.unused_functions:
        unused = next(item for item in facts.functions if item.name == name)
        if _allowed([unused.cite], cells):
            lines.append(
                f"FACT unused_function {name}: no reference outside its own "
                f"body {format_citation(unused.cite)}"
            )

    for call in facts.unresolved_calls:
        if not _allowed(call.sites, cells):
            continue
        sites = " ".join(format_citation(cite) for cite in call.sites)
        if call.run_targets:
            origin = f"possible from %run {', '.join(call.run_targets)}"
        else:
            origin = "no %run before it"
        lines.append(f"FACT unresolved_call {call.name}: called at {sites}; {origin}")

    for mutation in facts.mutations:
        if _allowed([mutation.cite], cells):
            lines.append(
                f"FACT mutation {mutation.name}: {mutation.how} in "
                f"{mutation.function} {format_citation(mutation.cite)}"
            )

    for widget in facts.widgets:
        if not _allowed([widget.cite], cells):
            continue
        details = f"default {widget.default!r}"
        if widget.choices is not None:
            details += ", choices " + ", ".join(repr(item) for item in widget.choices)
        lines.append(
            f"FACT widget {widget.op} {widget.name or '?'}: {details} "
            f"{format_citation(widget.cite)}"
        )

    for table in facts.tables:
        if not _allowed([table.cite], cells):
            continue
        if table.resolved_from is not None:
            target = (
                f"{table.target!r} (resolved from "
                f"{format_citation(table.resolved_from)})"
            )
        elif table.target is not None:
            target = repr(table.target)
        else:
            target = table.target_expr or "?"
        details = f"target {target}"
        if table.mode is not None:
            details += f", mode {table.mode}"
        lines.append(
            f"FACT table {table.direction} {table.api}: {details} "
            f"{format_citation(table.cite)}"
        )

    for run in facts.runs:
        if _allowed([run.cite], cells):
            lines.append(
                f"FACT run {run.target or '?'}: %run {format_citation(run.cite)}"
            )

    for issue in facts.parse_issues:
        if cells is not None and issue.cell not in cells:
            continue
        line = issue.line or 1
        lines.append(
            f"FACT parse_issue C{issue.cell}: {issue.message} "
            f"{format_citation(Citation(issue.cell, line, line))}"
        )

    lines.extend(f"LIMIT {limitation}" for limitation in facts.limitations)
    return "\n".join(lines)
