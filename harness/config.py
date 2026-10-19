"""Load and validate the single ``[llm]`` configuration table.

The file is untrusted input, so every setting is checked here rather than in the
client. Secrets deliberately have no field: the config stores only the *name*
of an environment variable, which keeps a config file safe to share.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Provider = Literal["gemini", "openai", "anthropic", "local"]
Strategy = Literal["json_mode", "prompt"]

_VALID_KEYS = frozenset(
    {
        "provider",
        "model",
        "api_key_env",
        "base_url",
        "structured",
        "temperature",
        "max_output_tokens",
        "timeout_s",
        "max_retries",
        "repair_retries",
        "log_path",
    }
)
_VALID_PROVIDERS: tuple[Provider, ...] = ("gemini", "openai", "anthropic", "local")
_VALID_STRUCTURED = ("auto", "json_mode", "prompt")

_PROVIDER_DEFAULTS: dict[Provider, tuple[str, str | None, Strategy]] = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta",
        "GEMINI_API_KEY",
        "json_mode",
    ),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "json_mode"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "prompt"),
    "local": ("http://localhost:11434/v1", None, "prompt"),
}


class ConfigError(Exception):
    """The config file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """A fully resolved config: no defaults and never ``structured = "auto"``."""

    provider: Provider
    model: str
    api_key_env: str | None
    base_url: str
    strategy: Strategy
    temperature: float | None
    max_output_tokens: int
    timeout_s: float
    max_retries: int
    repair_retries: int
    log_path: str


def _positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"llm.{key} must be a positive integer, got {value!r}")
    return value


def _positive_number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"llm.{key} must be greater than 0, got {value!r}")
    return float(value)


def _non_negative_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"llm.{key} must be >= 0, got {value!r}")
    return value


def _string_setting(data: Mapping[str, object], key: str, default: str) -> str:
    value = data.get(key, default)
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise ConfigError(f"llm.{key} must be a string, got {value!r}")
    return value


def _key_env_setting(data: Mapping[str, object], default: str | None) -> str | None:
    value = data.get("api_key_env", default)
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise ConfigError(f"llm.api_key_env must be a string, got {value!r}")
    return value


def _temperature(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"llm.temperature must be a number, got {value!r}")
    return float(value)


_DOTENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTENV_ESCAPES = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    '"': '"',
    "\\": "\\",
}


def _dotenv_value(raw: str, path: Path, line: int) -> str:
    """Parse one value: quoted, unquoted, or empty, with inline comments."""
    value = raw.strip()
    if not value:
        return ""
    quote = value[0]
    if quote in {'"', "'"}:
        characters: list[str] = []
        index = 1
        while index < len(value):
            character = value[index]
            if character == quote:
                rest = value[index + 1 :].strip()
                if rest and not rest.startswith("#"):
                    raise ConfigError(
                        f"{path}:{line}: unexpected text after the closing quote"
                    )
                return "".join(characters)
            if quote == '"' and character == "\\":
                index += 1
                if index == len(value):
                    raise ConfigError(f"{path}:{line}: unterminated escape")
                escaped = value[index]
                if escaped not in _DOTENV_ESCAPES:
                    raise ConfigError(f"{path}:{line}: unsupported escape \\{escaped}")
                characters.append(_DOTENV_ESCAPES[escaped])
            else:
                characters.append(character)
            index += 1
        raise ConfigError(f"{path}:{line}: unterminated {quote} quote")
    for index, character in enumerate(value):
        if character == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def _parse_dotenv(text: str, path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines without shell expansion or command execution."""
    values: dict[str, str] = {}
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export"):
            remainder = line[len("export") :]
            if remainder[:1].isspace():
                line = remainder.lstrip()
        key, separator, raw_value = line.partition("=")
        if not separator:
            raise ConfigError(f"{path}:{number}: expected KEY=VALUE")
        key = key.strip()
        if not _DOTENV_KEY.fullmatch(key):
            raise ConfigError(f"{path}:{number}: invalid environment variable name")
        values[key] = _dotenv_value(raw_value, path, number)
    return values


def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    *,
    env: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> bool:
    """Load ``.env`` into the process environment and report whether it existed.

    Existing variables win by default, so a real shell export always beats a
    file.  ``env`` is injectable for tests; values are never expanded, and a
    missing file is intentionally not an error.
    """
    environ: MutableMapping[str, str] = os.environ if env is None else env
    dotenv_path = Path(".env" if path is None else path)
    try:
        raw = dotenv_path.read_bytes()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ConfigError(
            f"{dotenv_path}: cannot read environment file ({exc})"
        ) from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{dotenv_path}: environment file is not valid UTF-8"
        ) from exc

    for key, value in _parse_dotenv(text, dotenv_path).items():
        if override or key not in environ:
            environ[key] = value
    return True


def load_config(
    path: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> LLMConfig:
    """Read one config file and return resolved values, never raw defaults.

    ``env`` is injectable so tests (and embedders) never depend on the process
    environment, and the key value itself is never read here.
    """
    environ = os.environ if env is None else env
    if path is None:
        path = Path(environ.get("HARNESS_CONFIG") or "config.toml")
    config_path = Path(path)

    try:
        raw = config_path.read_bytes()
    except OSError as exc:
        raise ConfigError(
            f"{config_path}: cannot read config ({exc}); "
            "copy config.example.toml to config.toml"
        ) from exc
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{config_path}: invalid TOML: {exc}") from exc

    table = document.get("llm")
    if not isinstance(table, dict):
        raise ConfigError(
            f"{config_path}: missing [llm] table; "
            "copy config.example.toml to config.toml"
        )
    data: Mapping[str, object] = table

    unknown = sorted(set(data) - _VALID_KEYS)
    if unknown:
        raise ConfigError(
            f"{config_path}: unknown key(s) in [llm]: {', '.join(unknown)}; "
            f"valid keys are {', '.join(sorted(_VALID_KEYS))}"
        )

    provider_value = data.get("provider")
    if not isinstance(provider_value, str) or provider_value not in _VALID_PROVIDERS:
        raise ConfigError(
            f"{config_path}: unknown provider {provider_value!r}; "
            f"valid providers are {', '.join(_VALID_PROVIDERS)}"
        )
    provider: Provider = provider_value
    default_url, default_key_env, default_strategy = _PROVIDER_DEFAULTS[provider]

    model = data.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(
            f"{config_path}: llm.model is required; check the provider's current "
            "model list (names change)"
        )
    model = model.strip()
    if model == "REPLACE_ME":
        raise ConfigError(
            f"{config_path}: llm.model is still REPLACE_ME; set it to a real model "
            "name after checking the provider's current model list"
        )

    structured = data.get("structured", "auto")
    if not isinstance(structured, str) or structured not in _VALID_STRUCTURED:
        raise ConfigError(
            f"{config_path}: llm.structured must be one of "
            f"{', '.join(_VALID_STRUCTURED)}, got {structured!r}"
        )
    strategy: Strategy = (
        default_strategy if structured == "auto" else cast(Strategy, structured)
    )
    if provider == "anthropic" and strategy == "json_mode":
        raise ConfigError(f"{config_path}: anthropic has no JSON mode; use prompt")

    api_key_env = _key_env_setting(data, default_key_env)
    base_url = _string_setting(data, "base_url", default_url).rstrip("/")
    temperature = _temperature(data.get("temperature"))
    max_output_tokens = _positive_int(
        data.get("max_output_tokens", 4096), "max_output_tokens"
    )
    timeout_s = _positive_number(data.get("timeout_s", 120), "timeout_s")
    max_retries = _non_negative_int(data.get("max_retries", 3), "max_retries")
    repair_retries = _non_negative_int(data.get("repair_retries", 2), "repair_retries")
    log_path = _string_setting(data, "log_path", "runs/llm_calls.jsonl")

    return LLMConfig(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        strategy=strategy,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
        max_retries=max_retries,
        repair_retries=repair_retries,
        log_path=log_path,
    )
