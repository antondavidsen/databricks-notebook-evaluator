from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import harness
from harness import _EXPORTS

ROOT = Path(__file__).resolve().parents[1]


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the test interpreter in a fresh process rooted at the project."""
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )


def loaded_modules(setup: str) -> list[str]:
    """Return harness stages and Pydantic imported by a small setup snippet."""
    code = (
        f"{setup}\n"
        "import json, sys\n"
        "print(json.dumps(sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.startswith('harness.') or name == 'pydantic'\n"
        ")))\n"
    )
    completed = run_python("-c", code)
    assert completed.returncode == 0, completed.stderr
    modules: list[str] = json.loads(completed.stdout)
    return modules


def test_declared_exports_and_resolution_table_agree() -> None:
    assert harness.__all__ == sorted(harness.__all__)
    assert set(harness.__all__) == set(_EXPORTS)


def test_every_export_is_the_object_from_its_defining_module() -> None:
    for name, module_name in _EXPORTS.items():
        defining_module = import_module(module_name, "harness")

        value = getattr(harness, name)

        assert value is getattr(defining_module, name)
        assert name in vars(harness)


def test_dir_lists_lazy_exports_without_importing_them() -> None:
    assert set(harness.__all__) <= set(dir(harness))


def test_star_import_exposes_exactly_the_declared_names() -> None:
    namespace: dict[str, object] = {}

    exec("from harness import *", namespace)

    assert set(namespace) - {"__builtins__"} == set(harness.__all__)


def test_importing_the_package_loads_no_stage_or_pydantic() -> None:
    assert loaded_modules("import harness") == []


def test_importing_models_loads_only_models() -> None:
    assert loaded_modules("import harness.models") == ["harness.models"]


def test_importing_the_parser_stays_independent_of_pydantic() -> None:
    assert loaded_modules("import harness.parser") == [
        "harness.models",
        "harness.parser",
    ]


def test_grounding_is_a_leaf_module() -> None:
    assert loaded_modules("import harness.grounding") == [
        "harness.grounding",
        "harness.models",
    ]


def test_accessing_a_lazy_name_loads_only_its_module() -> None:
    assert loaded_modules("import harness\nharness.Cell") == ["harness.models"]
    assert loaded_modules("import harness\nharness.parse_text") == [
        "harness.models",
        "harness.parser",
    ]


def test_module_cli_help_does_not_warn_about_eager_imports() -> None:
    completed = run_python("-m", "harness.parser", "--help")

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "found in sys.modules" not in completed.stderr
    assert "RuntimeWarning" not in completed.stderr
