"""A tiny synchronous LLM layer for Gemini, OpenAI, Anthropic, and local servers.

Everything provider-specific lives in a four-method adapter and everything
provider-neutral lives here exactly once: status mapping, retry/backoff,
key scrubbing, structured-output repair, and call logging.  That keeps a
provider switch a config change and keeps tests entirely off the network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

from .config import ConfigError, LLMConfig, Strategy, load_config, load_dotenv

Message = dict[str, str]
JsonBody = dict[str, Any]
Transport = Callable[
    [str, dict[str, str], JsonBody, float],
    tuple[int, dict[str, str], JsonBody],
]

_USER_AGENT = "notebook-harness/0.1"


class LLMError(Exception):
    """Base class for every expected LLM-layer failure."""


class LLMAuthError(LLMError):
    """The provider rejected the API key (401/403)."""


class LLMRequestError(LLMError):
    """The provider rejected the request and retrying would not help."""


class LLMTransientError(LLMError):
    """A retryable failure survived every configured retry."""


class LLMResponseError(LLMError):
    """The reply was valid HTTP but not the shape this provider promises."""


class TruncatedOutput(LLMError):
    """The model hit its output limit; the caller must send less input."""


class StructuredOutputError(LLMError):
    """No validated object arrived even after the configured repairs."""

    def __init__(self, attempts: int, last_error: str, last_text: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.last_text = last_text
        super().__init__(
            f"structured output failed after {attempts} attempts: {last_error}"
        )


class TransportError(Exception):
    """The HTTP call could not produce a status (timeout or network fault)."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One provider reply plus the metadata a caller needs to judge it."""

    text: str
    finish_reason: str | None
    truncated: bool
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    provider: str
    model: str


def _lower_headers(headers: Any) -> dict[str, str]:
    """Normalise urllib's email-style headers; retry logic needs one casing."""
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _decode_body(raw: bytes) -> JsonBody:
    text = raw.decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return {"_raw": text[:500]}
    if isinstance(body, dict):
        return cast(JsonBody, body)
    return {"_raw": text[:500]}


def urllib_transport(
    url: str,
    headers: dict[str, str],
    json_body: JsonBody,
    timeout: float,
) -> tuple[int, dict[str, str], JsonBody]:
    """POST once and return HTTP errors instead of raising them.

    Returning the status keeps the retry policy in ``LLMClient``; only failures
    without a status (DNS, refused connection, timeout) become TransportError.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(json_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
            response_headers = _lower_headers(response.headers)
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
        response_headers = _lower_headers(exc.headers)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError(f"network error talking to {url}: {exc}") from exc
    return status, response_headers, _decode_body(raw)


@dataclass(frozen=True, slots=True)
class _Reply:
    """Provider-neutral parts of a successful reply, before Completion."""

    text: str
    finish_reason: str | None
    truncated: bool
    input_tokens: int | None
    output_tokens: int | None


def _split_system(messages: Sequence[Message]) -> tuple[str, list[Message]]:
    """Pull system text out because Gemini and Anthropic have no system role."""
    system_parts = [
        message["content"] for message in messages if message.get("role") == "system"
    ]
    chat = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message.get("role") != "system"
    ]
    return "\n\n".join(system_parts), chat


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "default",
        "examples",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)


def _strict_json_schema(node: Any) -> Any:
    """Copy a Pydantic JSON Schema into the subset strict mode accepts.

    Strict structured output rejects validation keywords such as ``maxLength``
    and requires every object to set ``additionalProperties: false`` and to
    list every property as required.  The constraints still live in the
    Pydantic model, which validates the reply locally.
    """
    if isinstance(node, list):
        return [_strict_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    strict = {
        key: _strict_json_schema(value)
        for key, value in node.items()
        if key not in _STRICT_UNSUPPORTED_KEYWORDS
    }
    properties = strict.get("properties")
    if strict.get("type") == "object" and isinstance(properties, dict):
        strict["additionalProperties"] = False
        strict["required"] = list(properties)
    return strict


def _openai_response_format(schema: type[BaseModel] | None) -> JsonBody:
    """Ask for JSON, preferring the provider-native schema.

    OpenAI-compatible gateways increasingly reject the older
    ``{"type": "json_object"}`` shape and require ``json_schema`` with strict
    mode on.  Pydantic's constraints are not valid strict-mode keywords, so the
    schema is reduced to strict's subset; the local validator still enforces
    the full model.
    """
    if schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": _strict_json_schema(schema.model_json_schema()),
        },
    }


class _Adapter(Protocol):
    def build_request(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool,
        schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict[str, str], JsonBody]: ...

    def parse_response(self, body: JsonBody) -> _Reply: ...


@dataclass(frozen=True, slots=True)
class _GeminiAdapter:
    cfg: LLMConfig
    api_key: str

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool,
        schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict[str, str], JsonBody]:
        system_text, chat = _split_system(messages)
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in chat
        ]
        body: JsonBody = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        generation: JsonBody = {"maxOutputTokens": self.cfg.max_output_tokens}
        if self.cfg.temperature is not None:
            generation["temperature"] = self.cfg.temperature
        if json_mode:
            generation["responseMimeType"] = "application/json"
        body["generationConfig"] = generation
        url = f"{self.cfg.base_url}/models/{self.cfg.model}:generateContent"
        return url, {"x-goog-api-key": self.api_key}, body

    def parse_response(self, body: JsonBody) -> _Reply:
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            feedback = body.get("promptFeedback")
            detail = ""
            if feedback is not None:
                detail = f"; promptFeedback: {json.dumps(feedback, ensure_ascii=False)}"
            raise LLMResponseError(f"gemini returned no candidates{detail}")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise LLMResponseError("gemini returned a malformed candidate")
        content = candidate.get("content")
        if not isinstance(content, dict):
            raise LLMResponseError("gemini returned a candidate without content")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise LLMResponseError("gemini returned content without parts")
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(cast(str, part["text"]))
        finish_reason = _optional_str(candidate.get("finishReason"))
        usage = body.get("usageMetadata")
        input_tokens = (
            _optional_int(usage.get("promptTokenCount"))
            if isinstance(usage, dict)
            else None
        )
        output_tokens = (
            _optional_int(usage.get("candidatesTokenCount"))
            if isinstance(usage, dict)
            else None
        )
        return _Reply(
            text="".join(texts),
            finish_reason=finish_reason,
            truncated=finish_reason == "MAX_TOKENS",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@dataclass(frozen=True, slots=True)
class _OpenAIAdapter:
    """OpenAI and local OpenAI-compatible servers differ only in naming/auth."""

    cfg: LLMConfig
    api_key: str | None
    token_field: str

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool,
        schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict[str, str], JsonBody]:
        body: JsonBody = {
            "model": self.cfg.model,
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
            self.token_field: self.cfg.max_output_tokens,
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        if json_mode:
            body["response_format"] = _openai_response_format(schema)
        headers: dict[str, str] = {}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return f"{self.cfg.base_url}/chat/completions", headers, body

    def parse_response(self, body: JsonBody) -> _Reply:
        provider = self.cfg.provider
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError(f"{provider} returned no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMResponseError(f"{provider} returned a malformed choice")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseError(f"{provider} returned a choice without a message")
        text = message.get("content")
        if not isinstance(text, str):
            raise LLMResponseError(f"{provider} returned a message without text")
        finish_reason = _optional_str(choice.get("finish_reason"))
        usage = body.get("usage")
        input_tokens = (
            _optional_int(usage.get("prompt_tokens"))
            if isinstance(usage, dict)
            else None
        )
        output_tokens = (
            _optional_int(usage.get("completion_tokens"))
            if isinstance(usage, dict)
            else None
        )
        return _Reply(
            text=text,
            finish_reason=finish_reason,
            truncated=finish_reason == "length",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@dataclass(frozen=True, slots=True)
class _AnthropicAdapter:
    cfg: LLMConfig
    api_key: str

    def build_request(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool,
        schema: type[BaseModel] | None = None,
    ) -> tuple[str, dict[str, str], JsonBody]:
        if json_mode:
            raise ConfigError("anthropic has no JSON mode; use prompt")
        system_text, chat = _split_system(messages)
        body: JsonBody = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if system_text:
            body["system"] = system_text
        body["messages"] = chat
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        return f"{self.cfg.base_url}/messages", headers, body

    def parse_response(self, body: JsonBody) -> _Reply:
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise LLMResponseError("anthropic returned no content blocks")
        texts: list[str] = []
        for block in blocks:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                texts.append(cast(str, block["text"]))
        if not texts:
            raise LLMResponseError("anthropic returned no text content")
        finish_reason = _optional_str(body.get("stop_reason"))
        usage = body.get("usage")
        input_tokens = (
            _optional_int(usage.get("input_tokens"))
            if isinstance(usage, dict)
            else None
        )
        output_tokens = (
            _optional_int(usage.get("output_tokens"))
            if isinstance(usage, dict)
            else None
        )
        return _Reply(
            text="".join(texts),
            finish_reason=finish_reason,
            truncated=finish_reason == "max_tokens",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _adapter(cfg: LLMConfig, api_key: str | None) -> _Adapter:
    """Choose exactly one adapter; no provider SDK is ever imported."""
    if cfg.provider == "gemini":
        return _GeminiAdapter(cfg, cast(str, api_key))
    if cfg.provider == "anthropic":
        return _AnthropicAdapter(cfg, cast(str, api_key))
    token_field = "max_tokens" if cfg.provider == "local" else "max_completion_tokens"
    return _OpenAIAdapter(cfg, api_key, token_field)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def _limit(text: str, length: int = 500) -> str:
    return text[:length]


def _provider_message(body: JsonBody) -> str:
    """Extract the provider's own error text from its three common shapes."""
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    if isinstance(error, str):
        return error
    raw = body.get("_raw")
    if isinstance(raw, str):
        return raw
    return json.dumps(body, ensure_ascii=False)


class LLMClient:
    """One configured provider, with retries and structured output."""

    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None,
        *,
        transport: Transport,
        sleep: Callable[[float], None],
    ) -> None:
        self._cfg = cfg
        self._api_key = api_key
        self._adapter = _adapter(cfg, api_key)
        self._transport = transport
        self._sleep = sleep
        # Count schema repairs across a run; only the caller knows the baseline.
        self.repair_attempts = 0

    def _note_repair(self) -> None:
        """Record one malformed or schema-invalid reply that was retried."""
        self.repair_attempts += 1

    @property
    def provider(self) -> str:
        return self._cfg.provider

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def strategy(self) -> Strategy:
        return self._cfg.strategy

    def _scrub(self, text: str) -> str:
        """Remove the key from anything that can reach a log or an exception."""
        if self._api_key:
            return text.replace(self._api_key, "***")
        return text

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        """Send one conversation and return a parsed Completion."""
        return self._complete(messages, json_mode=json_mode, attempt=1, schema=schema)

    def generate[ModelT: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[ModelT],
    ) -> ModelT:
        """Return a validated instance, repairing bad JSON a bounded number of times."""
        structured_attempt = 0

        def call(next_messages: Sequence[Message], json_mode: bool) -> Completion:
            nonlocal structured_attempt
            structured_attempt += 1
            return self._complete(
                next_messages,
                json_mode=json_mode,
                attempt=structured_attempt,
                schema=schema,
            )

        return _generate_structured(
            call,
            messages,
            schema,
            repair_retries=self._cfg.repair_retries,
            strategy=self.strategy,
            scrub=self._scrub,
            on_repair=self._note_repair,
        )

    def _complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool,
        attempt: int,
        schema: type[BaseModel] | None = None,
    ) -> Completion:
        url, headers, body = self._adapter.build_request(
            messages, json_mode=json_mode, schema=schema
        )
        total_attempts = self._cfg.max_retries + 1
        last_error = "request failed"

        for request_number in range(1, total_attempts + 1):
            started = time.monotonic()
            try:
                status, response_headers, response_body = self._transport(
                    url, headers, body, self._cfg.timeout_s
                )
            except TransportError as exc:
                latency = _elapsed_ms(started)
                error = self._scrub(f"transport error: {exc}")
                self._log(attempt, None, messages, latency, error)
                last_error = error
                if request_number < total_attempts:
                    self._sleep(self._retry_delay(request_number, {}))
                continue

            latency = _elapsed_ms(started)
            if 200 <= status < 300:
                try:
                    reply = self._adapter.parse_response(response_body)
                except LLMResponseError as exc:
                    error = self._scrub(str(exc))
                    self._log(attempt, None, messages, latency, error)
                    raise LLMResponseError(error) from exc
                completion = Completion(
                    text=reply.text,
                    finish_reason=reply.finish_reason,
                    truncated=reply.truncated,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    latency_ms=latency,
                    provider=self.provider,
                    model=self.model,
                )
                self._log(attempt, completion, messages, latency, None)
                return completion

            if status in (401, 403):
                hint = self._cfg.api_key_env or "the provider's API key"
                error = self._scrub(
                    f"{self.provider} authentication failed (HTTP {status}); "
                    f"check {hint}"
                )
                self._log(attempt, None, messages, latency, error)
                raise LLMAuthError(error)

            provider_error = self._scrub(_limit(_provider_message(response_body)))
            error = self._scrub(
                f"{self.provider} request failed (HTTP {status}): {provider_error}"
            )
            self._log(attempt, None, messages, latency, error)
            if status != 429 and status < 500:
                raise LLMRequestError(error)
            last_error = error
            if request_number < total_attempts:
                self._sleep(self._retry_delay(request_number, response_headers))

        raise LLMTransientError(
            self._scrub(
                f"{self.provider} request failed after {total_attempts} "
                f"attempts: {last_error}"
            )
        )

    def _retry_delay(self, retry_number: int, headers: Mapping[str, str]) -> float:
        value = headers.get("retry-after")
        if value is not None:
            try:
                seconds = float(value)
            except ValueError:
                pass
            else:
                return max(0.0, min(seconds, 30.0))
        return float(min(2 ** (retry_number - 1), 30))

    def _log(
        self,
        attempt: int,
        completion: Completion | None,
        request_messages: Sequence[Message],
        latency_ms: int,
        error: str | None,
    ) -> None:
        """Append one JSON line per provider call; logging never breaks a run."""
        if not self._cfg.log_path:
            return
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "attempt": attempt,
            "latency_ms": latency_ms,
            "input_tokens": completion.input_tokens if completion else None,
            "output_tokens": completion.output_tokens if completion else None,
            "finish_reason": completion.finish_reason if completion else None,
            "truncated": completion.truncated if completion else False,
            "ok": completion is not None,
            "error": error,
            "request_messages": list(request_messages),
            "response_text": completion.text if completion else "",
        }
        line = self._scrub(json.dumps(record, ensure_ascii=False, default=str))
        try:
            path = Path(self._cfg.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            # Logging is diagnostic only; a bad path must not fail the model call.
            return


def make_client(
    cfg: LLMConfig,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> LLMClient:
    """Build a client and resolve its one secret from the environment."""
    environ = os.environ if env is None else env
    api_key: str | None = None
    name = cfg.api_key_env
    if name is not None:
        value = environ.get(name)
        if value:
            api_key = value
        elif cfg.provider != "local":
            raise ConfigError(f"{name} is not set; set {name}")
    elif cfg.provider != "local":
        raise ConfigError(f"no api_key_env configured for provider {cfg.provider}")
    return LLMClient(
        cfg,
        api_key,
        transport=transport if transport is not None else urllib_transport,
        sleep=sleep,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys; last-wins would silently hide a bad reply."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate key "{key}" in JSON object')
        result[key] = value
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    """Find the first JSON object in a reply, tolerating fences and prose.

    Starting at each ``{`` and asking ``raw_decode`` for one complete value is
    what rejects junk like ``{ref}`` before a later, real object.  A greedy
    regex cannot recover from that case (and can swallow trailing prose).
    Duplicate keys are rejected rather than resolved last-wins: a model that
    emits the same field twice is ambiguous and must repair the reply.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    last_error: json.JSONDecodeError | None = None
    skip_until = -1
    for index, character in enumerate(text):
        if character != "{" or index <= skip_until:
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            # Do not fall back to an object nested inside a malformed outer one:
            # ``{"claims": [{"kind": "OBSERVED"}],}`` must repair, not accept the
            # inner claim as the whole reply.
            last_error = exc
            skip_until = index + exc.pos
            continue
        return cast(dict[str, Any], value)
    if last_error is None:
        raise ValueError("no JSON object found")
    raise ValueError(f"no JSON object found: {last_error.msg}")


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Reply with ONE JSON object only, no prose and no code fences, that "
        "matches this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema())}"
    )


def _error_summary(exc: ValueError) -> str:
    """Keep validation feedback short and point at the exact failing field."""
    if isinstance(exc, ValidationError):
        entries: list[str] = []
        for error in exc.errors()[:10]:
            location = ".".join(str(part) for part in error["loc"]) or "model"
            entries.append(f"{location}: {error['msg']}")
        return "; ".join(entries)
    return str(exc)


def _noop() -> None:
    """Default repair callback: counting is optional for direct callers."""


def _generate_structured[ModelT: BaseModel](
    complete: Callable[[Sequence[Message], bool], Completion],
    messages: Sequence[Message],
    schema: type[ModelT],
    *,
    repair_retries: int,
    strategy: Strategy,
    scrub: Callable[[str], str],
    on_repair: Callable[[], None] = _noop,
) -> ModelT:
    """Shared structured path so the fake and the real client behave alike."""
    instruction = _schema_instruction(schema)
    conversation = [dict(message) for message in messages]
    for index, message in enumerate(conversation):
        if message.get("role") == "system":
            conversation[index] = {
                "role": "system",
                "content": f"{message['content']}\n\n{instruction}",
            }
            break
    else:
        conversation.insert(0, {"role": "system", "content": instruction})

    repairs_used = 0
    while True:
        completion = complete(conversation, strategy == "json_mode")
        if completion.truncated:
            raise TruncatedOutput(scrub("model hit its output limit; send less input"))
        try:
            obj = extract_json_object(completion.text)
            return schema.model_validate(obj)
        except ValueError as exc:
            last_error = scrub(_error_summary(exc))
            last_text = scrub(completion.text)
            if repairs_used >= repair_retries:
                raise StructuredOutputError(
                    attempts=repairs_used + 1,
                    last_error=last_error,
                    last_text=last_text,
                ) from exc
            conversation = [
                *conversation,
                {"role": "assistant", "content": completion.text},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply was invalid: {last_error}. "
                        "Return only the corrected JSON object."
                    ),
                },
            ]
            repairs_used += 1
            on_repair()


class FakeLLM:
    """Deterministic stand-in that exercises the real structured-output code."""

    provider = "fake"
    model = "fake"

    def __init__(
        self,
        responses: Sequence[str | Exception | Completion],
        strategy: Strategy = "prompt",
    ) -> None:
        self._responses = list(responses)
        self.strategy = strategy
        self.calls: list[dict[str, Any]] = []
        self.repair_attempts = 0

    def _note_repair(self) -> None:
        """Mirror the real client so tests can assert on repair counts."""
        self.repair_attempts += 1

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
    ) -> Completion:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "json_mode": json_mode,
            }
        )
        if not self._responses:
            raise AssertionError("FakeLLM has no more responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, Completion):
            return item
        return Completion(
            text=item,
            finish_reason="stop",
            truncated=False,
            input_tokens=None,
            output_tokens=None,
            latency_ms=0,
            provider=self.provider,
            model=self.model,
        )

    def generate[ModelT: BaseModel](
        self,
        messages: Sequence[Message],
        schema: type[ModelT],
    ) -> ModelT:
        return _generate_structured(
            lambda next_messages, json_mode: self.complete(
                next_messages, json_mode=json_mode
            ),
            messages,
            schema,
            repair_retries=2,
            strategy=self.strategy,
            scrub=lambda text: text,
            on_repair=self._note_repair,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """One-shot CLI: the quickest way to prove a provider key works."""
    parser = argparse.ArgumentParser(
        prog="python -m harness.llm",
        description="Send one prompt through the configured provider.",
    )
    parser.add_argument("prompt", help="user prompt to send")
    args = parser.parse_args(argv)
    try:
        load_dotenv()
        client = make_client(load_config())
        completion = client.complete([{"role": "user", "content": args.prompt}])
    except (ConfigError, LLMError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"provider: {completion.provider}")
    print(f"model: {completion.model}")
    print(f"strategy: {client.strategy}")
    print("reply:")
    print(completion.text)
    print(f"tokens: input={completion.input_tokens} output={completion.output_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
