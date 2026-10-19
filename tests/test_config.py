from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from harness.config import ConfigError, LLMConfig, load_config, load_dotenv


def document(**settings: object) -> str:
    """Render a TOML document; JSON scalars are valid TOML scalars."""
    body = "\n".join(f"{key} = {json.dumps(value)}" for key, value in settings.items())
    return f"[llm]\n{body}\n"


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("provider", "base_url", "key_env", "strategy"),
    [
        (
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            "GEMINI_API_KEY",
            "json_mode",
        ),
        ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "json_mode"),
        ("anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "prompt"),
        ("local", "http://localhost:11434/v1", None, "prompt"),
    ],
)
def test_provider_defaults_fill_every_missing_field(
    tmp_path: Path,
    provider: str,
    base_url: str,
    key_env: str | None,
    strategy: str,
) -> None:
    path = write_config(tmp_path, document(provider=provider, model="test-model"))

    cfg = load_config(path)

    assert cfg == LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        model="test-model",
        api_key_env=key_env,
        base_url=base_url,
        strategy=strategy,  # type: ignore[arg-type]
        temperature=None,
        max_output_tokens=4096,
        timeout_s=120.0,
        max_retries=3,
        repair_retries=2,
        log_path="runs/llm_calls.jsonl",
    )


def test_explicit_values_are_normalized(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        document(
            provider="openai",
            model="  gpt-test  ",
            api_key_env="CUSTOM_KEY",
            base_url="https://example.test/v1///",
            structured="prompt",
            temperature=1,
            max_output_tokens=17,
            timeout_s=2,
            max_retries=0,
            repair_retries=0,
            log_path="runs/custom.jsonl",
        ),
    )

    cfg = load_config(str(path))

    assert cfg.provider == "openai"
    assert cfg.model == "gpt-test"
    assert cfg.api_key_env == "CUSTOM_KEY"
    assert cfg.base_url == "https://example.test/v1"
    assert cfg.strategy == "prompt"
    assert cfg.temperature == 1.0
    assert cfg.max_output_tokens == 17
    assert cfg.timeout_s == 2.0
    assert cfg.max_retries == 0
    assert cfg.repair_retries == 0
    assert cfg.log_path == "runs/custom.jsonl"


def test_empty_optional_strings_fall_back_to_provider_defaults(
    tmp_path: Path,
) -> None:
    path = write_config(
        tmp_path,
        document(
            provider="openai",
            model="test-model",
            api_key_env="",
            base_url="",
            log_path="",
        ),
    )

    cfg = load_config(path)

    assert cfg.api_key_env == "OPENAI_API_KEY"
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.log_path == "runs/llm_calls.jsonl"


def test_auto_resolves_to_prompt_for_local(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        document(provider="local", model="test-model", structured="auto"),
    )

    assert load_config(path).strategy == "prompt"


def test_harness_config_environment_variable_selects_the_file(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, document(provider="gemini", model="test-model"))

    cfg = load_config(env={"HARNESS_CONFIG": str(path)})

    assert cfg.model == "test-model"


def test_default_path_is_config_toml_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_config(tmp_path, document(provider="gemini", model="test-model"))
    monkeypatch.chdir(tmp_path)

    assert load_config(env={}).model == "test-model"


def test_environment_defaults_to_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path, document(provider="gemini", model="test-model"))
    monkeypatch.setenv("HARNESS_CONFIG", str(path))

    assert load_config().provider == "gemini"


def test_missing_file_names_the_example(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"config\.example\.toml"):
        load_config(tmp_path / "missing.toml")


def test_invalid_utf8_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_invalid_toml_is_a_config_error(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[llm\nprovider = 1")

    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(path)


def test_missing_llm_table_is_a_config_error(tmp_path: Path) -> None:
    path = write_config(tmp_path, '[other]\nprovider = "gemini"\n')

    with pytest.raises(ConfigError, match="missing \\[llm\\]"):
        load_config(path)


def test_llm_must_be_a_table(tmp_path: Path) -> None:
    path = write_config(tmp_path, "llm = 1\n")

    with pytest.raises(ConfigError, match="missing \\[llm\\]"):
        load_config(path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (document(model="m", provider="gemini", extra=1), "unknown key"),
        (document(model="m"), "unknown provider"),
        (document(model="m", provider=1), "unknown provider"),
        (document(model="m", provider="bogus"), "unknown provider"),
        (document(provider="gemini"), "model is required"),
        (document(provider="gemini", model=1), "model is required"),
        (document(provider="gemini", model="   "), "model is required"),
        (document(provider="gemini", model="REPLACE_ME"), "REPLACE_ME"),
        (
            document(provider="gemini", model="m", structured=1),
            "structured must be",
        ),
        (
            document(provider="gemini", model="m", structured="sometimes"),
            "structured must be",
        ),
        (
            document(provider="anthropic", model="m", structured="json_mode"),
            "anthropic has no JSON mode",
        ),
        (
            document(provider="gemini", model="m", api_key_env=1),
            "api_key_env must be a string",
        ),
        (
            document(provider="gemini", model="m", base_url=1),
            "base_url must be a string",
        ),
        (
            document(provider="gemini", model="m", temperature=True),
            "temperature must be a number",
        ),
        (
            document(provider="gemini", model="m", temperature="hot"),
            "temperature must be a number",
        ),
        (
            document(provider="gemini", model="m", max_output_tokens=0),
            "max_output_tokens must be a positive integer",
        ),
        (
            document(provider="gemini", model="m", max_output_tokens=True),
            "max_output_tokens must be a positive integer",
        ),
        (
            document(provider="gemini", model="m", max_output_tokens="10"),
            "max_output_tokens must be a positive integer",
        ),
        (
            document(provider="gemini", model="m", timeout_s=0),
            "timeout_s must be greater than 0",
        ),
        (
            document(provider="gemini", model="m", timeout_s=True),
            "timeout_s must be greater than 0",
        ),
        (
            document(provider="gemini", model="m", timeout_s="slow"),
            "timeout_s must be greater than 0",
        ),
        (
            document(provider="gemini", model="m", max_retries=-1),
            "max_retries must be >= 0",
        ),
        (
            document(provider="gemini", model="m", repair_retries=-1),
            "repair_retries must be >= 0",
        ),
    ],
)
def test_invalid_settings_raise_config_error(
    tmp_path: Path, body: str, match: str
) -> None:
    path = write_config(tmp_path, body)

    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_load_dotenv_parses_quotes_comments_and_exports(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\ufeff# a comment\n"
        "\n"
        "export FIRST=one\n"
        'SECOND="two # kept"  # ignored\n'
        "THIRD='three \\n literal'\n"
        "FOURTH=four # ignored\n"
        "FIFTH=a=b\n"
        "EMPTY=\n"
        "HASH=# comment becomes an empty value\n"
        'DOUBLE="line\\nbreak\\tend \\"q\\" \\\\ done"\n'
        "exportFOO=bar\n"
        "FIRST=final\n",
        encoding="utf-8",
    )
    env: dict[str, str] = {}

    assert load_dotenv(path, env=env) is True

    assert env == {
        "FIRST": "final",
        "SECOND": "two # kept",
        "THIRD": "three \\n literal",
        "FOURTH": "four",
        "FIFTH": "a=b",
        "EMPTY": "",
        "HASH": "",
        "DOUBLE": 'line\nbreak\tend "q" \\ done',
        "exportFOO": "bar",
    }


def test_load_dotenv_does_not_override_shell_values_unless_asked(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".env"
    path.write_text("FIRST=file\nSECOND=file\n", encoding="utf-8")
    env = {"FIRST": "shell"}

    assert load_dotenv(path, env=env) is True
    assert env == {"FIRST": "shell", "SECOND": "file"}

    assert load_dotenv(path, env=env, override=True) is True
    assert env == {"FIRST": "file", "SECOND": "file"}


def test_load_dotenv_missing_file_is_not_an_error(tmp_path: Path) -> None:
    env = {"UNCHANGED": "yes"}

    assert load_dotenv(tmp_path / "missing.env", env=env) is False
    assert env == {"UNCHANGED": "yes"}


def test_load_dotenv_defaults_to_the_working_directory_and_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DOTENV_SHELL_WINS=file\n", encoding="utf-8")
    monkeypatch.setenv("DOTENV_SHELL_WINS", "shell")

    assert load_dotenv() is True
    assert os.environ["DOTENV_SHELL_WINS"] == "shell"


def test_load_dotenv_reads_a_directory_as_an_error(tmp_path: Path) -> None:
    directory = tmp_path / ".env"
    directory.mkdir()

    with pytest.raises(ConfigError, match="cannot read environment file"):
        load_dotenv(directory, env={})


def test_load_dotenv_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_dotenv(path, env={})


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("NO_EQUALS\n", "expected KEY=VALUE"),
        ("=value\n", "invalid environment variable name"),
        ("1BAD=value\n", "invalid environment variable name"),
        ('A="unterminated\n', "unterminated"),
        ('A="bad\\q"\n', r"unsupported escape \\q"),
        ('A="value" trailing\n', "unexpected text"),
        ('A="bad\\\n', "unterminated escape"),
    ],
)
def test_load_dotenv_rejects_malformed_lines(
    tmp_path: Path, body: str, match: str
) -> None:
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ConfigError, match=match):
        load_dotenv(path, env={})


def test_load_dotenv_errors_do_not_echo_secret_values(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("sk-do-not-print-me\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_dotenv(path, env={})

    assert "sk-do-not-print-me" not in str(caught.value)
    assert "expected KEY=VALUE" in str(caught.value)


def test_load_dotenv_validates_every_line_before_setting_anything(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".env"
    path.write_text("GOOD=value\nBROKEN\n", encoding="utf-8")
    env: dict[str, str] = {}

    with pytest.raises(ConfigError, match="expected KEY=VALUE"):
        load_dotenv(path, env=env)

    assert env == {}
