from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from email.message import Message
from pathlib import Path
from typing import cast

import pytest
from harness import llm
from harness.config import ConfigError, LLMConfig, Provider, Strategy
from harness.llm import (
    Completion,
    FakeLLM,
    JsonBody,
    LLMAuthError,
    LLMClient,
    LLMRequestError,
    LLMResponseError,
    LLMTransientError,
    StructuredOutputError,
    TransportError,
    TruncatedOutput,
    extract_json_object,
    make_client,
    urllib_transport,
)
from pydantic import BaseModel, Field

Reply = tuple[int, dict[str, str], JsonBody]


def make_config(
    *,
    provider: Provider = "gemini",
    model: str = "test-model",
    api_key_env: str | None = "TEST_KEY",
    base_url: str = "https://example.test/v1",
    strategy: Strategy = "prompt",
    temperature: float | None = 0.0,
    max_output_tokens: int = 1024,
    timeout_s: float = 5.0,
    max_retries: int = 0,
    repair_retries: int = 2,
    log_path: str = "",
) -> LLMConfig:
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


class FakeTransport:
    """A transport that replays replies and records every request."""

    def __init__(self, *replies: Reply | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, dict[str, str], JsonBody, float]] = []

    def __call__(
        self,
        url: str,
        headers: dict[str, str],
        body: JsonBody,
        timeout: float,
    ) -> Reply:
        self.calls.append((url, dict(headers), body, timeout))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_test_client(
    cfg: LLMConfig,
    replies: Sequence[Reply | Exception],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[LLMClient, FakeTransport, list[float]]:
    transport = FakeTransport(*replies)
    sleeps: list[float] = []
    client = make_client(
        cfg,
        env={"TEST_KEY": "SECRET123"} if env is None else env,
        transport=transport,
        sleep=sleeps.append,
    )
    return client, transport, sleeps


def user(text: str = "hi") -> dict[str, str]:
    return {"role": "user", "content": text}


def system(text: str = "be brief") -> dict[str, str]:
    return {"role": "system", "content": text}


def openai_reply(
    text: str = "ok",
    *,
    finish_reason: str | None = "stop",
    usage: bool = True,
) -> Reply:
    body: JsonBody = {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}]
    }
    if usage:
        body["usage"] = {"prompt_tokens": 3, "completion_tokens": 4}
    return 200, {}, body


def gemini_reply(
    text: str = "ok",
    *,
    finish_reason: str | None = "STOP",
    usage: bool = True,
) -> Reply:
    body: JsonBody = {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ]
    }
    if usage:
        body["usageMetadata"] = {
            "promptTokenCount": 5,
            "candidatesTokenCount": 6,
        }
    return 200, {}, body


def anthropic_reply(
    text: str = "ok",
    *,
    stop_reason: str | None = "end_turn",
    usage: bool = True,
) -> Reply:
    body: JsonBody = {
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
    }
    if usage:
        body["usage"] = {"input_tokens": 7, "output_tokens": 8}
    return 200, {}, body


class Verdict(BaseModel):
    answer: str
    score: int


# -- provider request/response shapes ---------------------------------------


def test_gemini_request_and_response() -> None:
    cfg = make_config(provider="gemini", strategy="json_mode", temperature=None)
    reply: Reply = (
        200,
        {},
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "hello "},
                            {"text": "world"},
                            {"inlineData": {"mimeType": "text/plain"}},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 6,
            },
        },
    )
    client, transport, sleeps = make_test_client(cfg, [reply])

    result = client.complete(
        [system(), user("hi"), {"role": "assistant", "content": "hey"}],
        json_mode=True,
    )

    assert (client.provider, client.model, client.strategy) == (
        "gemini",
        "test-model",
        "json_mode",
    )
    assert result.text == "hello world"
    assert result.finish_reason == "STOP"
    assert result.truncated is False
    assert result.input_tokens == 5
    assert result.output_tokens == 6
    assert result.provider == "gemini"
    assert result.model == "test-model"
    assert sleeps == []

    url, headers, body, timeout = transport.calls[0]
    assert url == ("https://example.test/v1/models/test-model:generateContent")
    assert headers == {"x-goog-api-key": "SECRET123"}
    assert timeout == 5.0
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hey"}]},
    ]
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert body["generationConfig"] == {
        "maxOutputTokens": 1024,
        "responseMimeType": "application/json",
    }


def test_gemini_without_system_and_with_temperature() -> None:
    cfg = make_config(temperature=0.25)
    client, transport, _ = make_test_client(
        cfg, [gemini_reply("cut", finish_reason="MAX_TOKENS", usage=False)]
    )

    result = client.complete([user()])

    body = transport.calls[0][2]
    assert "systemInstruction" not in body
    assert body["generationConfig"] == {
        "maxOutputTokens": 1024,
        "temperature": 0.25,
    }
    assert result.truncated is True
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_gemini_ignores_non_text_parts() -> None:
    client, _, _ = make_test_client(
        make_config(),
        [
            (
                200,
                {},
                {"candidates": [{"content": {"parts": [{"text": 1}, "not a part"]}}]},
            )
        ],
    )

    result = client.complete([user()])

    assert result.text == ""
    assert result.finish_reason is None
    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            {"promptFeedback": {"blockReason": "SAFETY"}},
            "no candidates; promptFeedback",
        ),
        ({"candidates": []}, "no candidates"),
        ({"candidates": [1]}, "malformed candidate"),
        ({"candidates": [{}]}, "without content"),
        ({"candidates": [{"content": {}}]}, "without parts"),
    ],
)
def test_gemini_response_shape_errors(body: JsonBody, match: str) -> None:
    client, _, _ = make_test_client(make_config(), [(200, {}, body)])

    with pytest.raises(LLMResponseError, match=match):
        client.complete([user()])


def test_openai_request_and_response() -> None:
    cfg = make_config(provider="openai", strategy="json_mode", temperature=0.5)
    client, transport, _ = make_test_client(
        cfg, [openai_reply("cut", finish_reason="length")]
    )

    result = client.complete([system(), user()], json_mode=True)

    url, headers, body, _ = transport.calls[0]
    assert url == "https://example.test/v1/chat/completions"
    assert headers == {"Authorization": "Bearer SECRET123"}
    assert body["messages"] == [system(), user()]
    assert body["max_completion_tokens"] == 1024
    assert body["temperature"] == 0.5
    assert body["response_format"] == {"type": "json_object"}
    assert result.text == "cut"
    assert result.truncated is True
    assert result.input_tokens == 3
    assert result.output_tokens == 4


def test_openai_prompt_mode_omits_optional_fields() -> None:
    cfg = make_config(provider="openai", temperature=None)
    client, transport, _ = make_test_client(
        cfg,
        [
            (
                200,
                {},
                {
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": None,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": True,
                        "completion_tokens": False,
                    },
                },
            )
        ],
    )

    result = client.complete([user()])

    body = transport.calls[0][2]
    assert "temperature" not in body
    assert "response_format" not in body
    assert result.finish_reason is None
    assert result.truncated is False
    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({}, "no choices"),
        ({"choices": []}, "no choices"),
        ({"choices": "bad"}, "no choices"),
        ({"choices": [1]}, "malformed choice"),
        ({"choices": [{}]}, "without a message"),
        ({"choices": [{"message": {}}]}, "without text"),
        ({"choices": [{"message": {"content": 1}}]}, "without text"),
    ],
)
def test_openai_response_shape_errors(body: JsonBody, match: str) -> None:
    client, _, _ = make_test_client(make_config(provider="openai"), [(200, {}, body)])

    with pytest.raises(LLMResponseError, match=match):
        client.complete([user()])


def test_local_provider_needs_no_key_and_uses_max_tokens(
    tmp_path: Path,
) -> None:
    cfg = make_config(
        provider="local",
        api_key_env=None,
        base_url="http://localhost:11434/v1",
        log_path=str(tmp_path / "calls.jsonl"),
    )
    client, transport, _ = make_test_client(cfg, [openai_reply("local")], env={})

    result = client.complete([user("SECRET123")])

    assert result.text == "local"
    url, headers, body, _ = transport.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert headers == {}
    assert body["max_tokens"] == 1024
    log_text = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "SECRET123" in log_text


def test_anthropic_request_and_response() -> None:
    cfg = make_config(provider="anthropic", temperature=0.3)
    reply: Reply = (
        200,
        {},
        {
            "content": [
                {"type": "text", "text": "a"},
                {"type": "redacted", "data": "x"},
                {"type": "text", "text": "b"},
            ],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 7, "output_tokens": 8},
        },
    )
    client, transport, _ = make_test_client(cfg, [reply])

    result = client.complete([system(), user()])

    url, headers, body, _ = transport.calls[0]
    assert url == "https://example.test/v1/messages"
    assert headers == {
        "x-api-key": "SECRET123",
        "anthropic-version": "2023-06-01",
    }
    assert body["system"] == "be brief"
    assert body["messages"] == [user()]
    assert body["max_tokens"] == 1024
    assert body["temperature"] == 0.3
    assert "response_format" not in body
    assert result.text == "ab"
    assert result.truncated is True
    assert result.input_tokens == 7
    assert result.output_tokens == 8


def test_anthropic_without_system_or_optional_fields() -> None:
    cfg = make_config(provider="anthropic", temperature=None)
    client, transport, _ = make_test_client(cfg, [anthropic_reply(usage=False)])

    result = client.complete([user()])

    body = transport.calls[0][2]
    assert "system" not in body
    assert "temperature" not in body
    assert result.finish_reason == "end_turn"
    assert result.truncated is False
    assert result.input_tokens is None
    assert result.output_tokens is None


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("not a list", "no content blocks"),
        ([], "no text content"),
        (
            [1, {"type": "tool_use"}, {"type": "text", "text": 1}],
            "no text content",
        ),
    ],
)
def test_anthropic_response_shape_errors(content: object, match: str) -> None:
    client, _, _ = make_test_client(
        make_config(provider="anthropic"),
        [(200, {}, {"content": content, "stop_reason": "end_turn"})],
    )

    with pytest.raises(LLMResponseError, match=match):
        client.complete([user()])


def test_anthropic_rejects_json_mode_before_calling_the_transport() -> None:
    cfg = make_config(provider="anthropic")
    client, transport, _ = make_test_client(cfg, [])

    with pytest.raises(ConfigError, match="no JSON mode"):
        client.complete([user()], json_mode=True)

    assert transport.calls == []


# -- HTTP status mapping, retries, and key handling -------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failures_name_the_environment_variable(
    status: int,
) -> None:
    client, transport, sleeps = make_test_client(
        make_config(max_retries=3),
        [(status, {}, {"error": {"message": "bad key"}})],
    )

    with pytest.raises(LLMAuthError, match="TEST_KEY") as caught:
        client.complete([user()])

    assert f"HTTP {status}" in str(caught.value)
    assert len(transport.calls) == 1
    assert sleeps == []


def test_non_retryable_request_error_includes_provider_text() -> None:
    client, transport, sleeps = make_test_client(
        make_config(max_retries=3),
        [(400, {}, {"error": {"message": "bad request"}})],
    )

    with pytest.raises(LLMRequestError, match="bad request") as caught:
        client.complete([user()])

    assert "HTTP 400" in str(caught.value)
    assert len(transport.calls) == 1
    assert sleeps == []


def test_redirect_status_is_a_request_error() -> None:
    client, _, _ = make_test_client(
        make_config(), [(302, {"location": "/elsewhere"}, {})]
    )

    with pytest.raises(LLMRequestError, match="HTTP 302"):
        client.complete([user()])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"error": {"message": "nested"}}, "nested"),
        ({"error": "plain"}, "plain"),
        ({"error": {}, "_raw": "raw text"}, "raw text"),
        ({"error": {}}, '"error"'),
    ],
)
def test_provider_error_shapes_are_preserved(body: JsonBody, message: str) -> None:
    client, transport, _ = make_test_client(
        make_config(max_retries=2), [(400, {}, body)]
    )

    with pytest.raises(LLMRequestError, match="HTTP 400") as caught:
        client.complete([user()])

    assert message in str(caught.value)
    assert len(transport.calls) == 1


def test_long_provider_errors_are_truncated() -> None:
    client, _, _ = make_test_client(
        make_config(),
        [(400, {}, {"error": {"message": "x" * 600}})],
    )

    with pytest.raises(LLMRequestError) as caught:
        client.complete([user()])

    assert "x" * 500 in str(caught.value)
    assert "x" * 501 not in str(caught.value)


def test_429_retries_with_exponential_backoff() -> None:
    client, transport, sleeps = make_test_client(
        make_config(provider="openai", max_retries=1),
        [(429, {}, {"error": "slow down"}), openai_reply("finally")],
    )

    result = client.complete([user()])

    assert result.text == "finally"
    assert sleeps == [1.0]
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"retry-after": "soon"}, 1.0),
        ({"retry-after": "-2"}, 0.0),
        ({"retry-after": "99"}, 30.0),
    ],
)
def test_retry_after_is_honored_and_clamped(
    headers: dict[str, str], expected: float
) -> None:
    client, _, sleeps = make_test_client(
        make_config(provider="openai", max_retries=1),
        [(503, headers, {}), openai_reply()],
    )

    client.complete([user()])

    assert sleeps == [expected]


def test_exponential_backoff_caps_at_thirty_seconds() -> None:
    replies: list[Reply] = [(503, {}, {}) for _ in range(7)]
    client, transport, sleeps = make_test_client(make_config(max_retries=6), replies)

    with pytest.raises(LLMTransientError, match="after 7 attempts"):
        client.complete([user()])

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
    assert len(transport.calls) == 7


def test_transport_errors_retry_and_scrub_the_key() -> None:
    cfg = make_config(max_retries=2)
    client, transport, sleeps = make_test_client(
        cfg,
        [
            TransportError("SECRET123 down"),
            TransportError("SECRET123 still down"),
            TransportError("SECRET123 offline"),
        ],
    )

    with pytest.raises(LLMTransientError, match="after 3 attempts") as caught:
        client.complete([user("SECRET123 prompt")])

    assert "SECRET123" not in str(caught.value)
    assert "***" in str(caught.value)
    assert sleeps == [1.0, 2.0]
    assert len(transport.calls) == 3


def test_transport_error_then_success() -> None:
    client, _, sleeps = make_test_client(
        make_config(provider="openai", max_retries=1),
        [TransportError("temporary"), openai_reply()],
    )

    assert client.complete([user()]).text == "ok"
    assert sleeps == [1.0]


def test_request_error_after_a_transient_failure_is_not_retried() -> None:
    client, transport, sleeps = make_test_client(
        make_config(max_retries=3),
        [(500, {}, {}), (400, {}, {"error": "bad"})],
    )

    with pytest.raises(LLMRequestError):
        client.complete([user()])

    assert sleeps == [1.0]
    assert len(transport.calls) == 2


# -- client construction and logging ---------------------------------------


def test_make_client_rejects_a_missing_remote_key() -> None:
    with pytest.raises(ConfigError, match="TEST_KEY is not set"):
        make_client(make_config(), env={})


def test_make_client_rejects_a_remote_config_without_a_key_name() -> None:
    with pytest.raises(ConfigError, match="no api_key_env"):
        make_client(make_config(api_key_env=None), env={})


def test_local_client_may_have_a_key_name_but_no_key() -> None:
    cfg = make_config(
        provider="local", api_key_env="LOCAL_KEY", base_url="http://local/v1"
    )
    transport = FakeTransport(openai_reply())

    client = make_client(cfg, env={}, transport=transport)
    client.complete([user()])

    assert transport.calls[0][1] == {}


def test_local_client_uses_a_key_when_one_is_configured() -> None:
    cfg = make_config(
        provider="local", api_key_env="LOCAL_KEY", base_url="http://local/v1"
    )
    transport = FakeTransport(openai_reply())

    client = make_client(cfg, env={"LOCAL_KEY": "local-secret"}, transport=transport)
    client.complete([user()])

    assert transport.calls[0][1] == {"Authorization": "Bearer local-secret"}


def test_make_client_reads_os_environ_when_env_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_KEY", "SECRET123")
    transport = FakeTransport(openai_reply())

    client = make_client(make_config(provider="openai"), transport=transport)
    client.complete([user()])

    assert transport.calls[0][1] == {"Authorization": "Bearer SECRET123"}


def test_make_client_defaults_to_the_urllib_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(openai_reply())
    monkeypatch.setattr(llm, "urllib_transport", transport)

    client = make_client(
        make_config(provider="openai"),
        env={"TEST_KEY": "SECRET123"},
    )

    assert client.complete([user()]).text == "ok"
    assert len(transport.calls) == 1


def test_call_logging_writes_scrubbed_json_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "calls.jsonl"
    cfg = make_config(max_retries=1, log_path=str(log_path))
    client, _, _ = make_test_client(
        cfg,
        [
            (429, {}, {"error": {"message": "bad SECRET123"}}),
            (
                200,
                {},
                {"candidates": [{"content": {"parts": [{"text": "SECRET123 ok"}]}}]},
            ),
        ],
    )

    assert client.complete([user("SECRET123 prompt")]).text == "SECRET123 ok"

    raw = log_path.read_text(encoding="utf-8")
    assert "SECRET123" not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == 2
    assert records[0]["ok"] is False
    assert records[0]["finish_reason"] is None
    assert records[1]["ok"] is True
    assert records[1]["attempt"] == 1
    assert records[1]["request_messages"] == [{"role": "user", "content": "*** prompt"}]
    assert records[1]["response_text"] == "*** ok"
    assert "***" in records[0]["error"]


def test_response_shape_errors_are_logged(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.jsonl"
    client, _, _ = make_test_client(
        make_config(log_path=str(log_path)), [(200, {}, {})]
    )

    with pytest.raises(LLMResponseError, match="no candidates"):
        client.complete([user()])

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["ok"] is False
    assert record["error"] == "gemini returned no candidates"


def test_logging_failure_does_not_break_a_call(tmp_path: Path) -> None:
    cfg = make_config(provider="openai", log_path=str(tmp_path))
    client, _, _ = make_test_client(cfg, [openai_reply("still ok")])

    assert client.complete([user()]).text == "still ok"


# -- JSON extraction and structured output ----------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "a", "score": 1}', {"answer": "a", "score": 1}),
        (
            '```json\n{"answer": "a", "score": 1}\n```',
            {"answer": "a", "score": 1},
        ),
        (
            'Here you go: {"answer": "a", "score": 1} hope it helps',
            {"answer": "a", "score": 1},
        ),
        (
            'cite {ref} then {"answer": "a", "score": 1}',
            {"answer": "a", "score": 1},
        ),
        (
            '{"outer": {"answer": "a"}, "score": 1} tail',
            {"outer": {"answer": "a"}, "score": 1},
        ),
        (
            '{"answer": "a", "score": 1}{"answer": "b"}',
            {"answer": "a", "score": 1},
        ),
    ],
)
def test_extract_json_object_finds_the_first_real_object(
    text: str, expected: JsonBody
) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.parametrize(
    "text", ["", "no json here", "{not json}", "[1, 2]", "{'single': 'quotes'}"]
)
def test_extract_json_object_rejects_text_without_an_object(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object(text)


def test_generate_inserts_a_schema_instruction() -> None:
    client, transport, _ = make_test_client(
        make_config(provider="openai"),
        [openai_reply('{"answer": "yes", "score": 9}')],
    )

    verdict = client.generate([user("grade this")], Verdict)

    assert verdict == Verdict(answer="yes", score=9)
    messages = transport.calls[0][2]["messages"]
    assert messages[0]["role"] == "system"
    assert "JSON Schema" in messages[0]["content"]
    assert '"title": "Verdict"' in messages[0]["content"]
    assert messages[1] == user("grade this")
    assert "response_format" not in transport.calls[0][2]


def test_generate_extends_an_existing_system_message() -> None:
    client, transport, _ = make_test_client(
        make_config(provider="openai", strategy="json_mode"),
        [openai_reply('{"answer": "yes", "score": 9}')],
    )

    client.generate([system(), user()], Verdict)

    messages = transport.calls[0][2]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("be brief\n\nReply with ONE")
    assert messages[1] == user()
    response_format = transport.calls[0][2]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Verdict"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema == {
        "additionalProperties": False,
        "properties": {
            "answer": {"title": "Answer", "type": "string"},
            "score": {"title": "Score", "type": "integer"},
        },
        "required": ["answer", "score"],
        "title": "Verdict",
        "type": "object",
    }


def test_openai_json_mode_prunes_keywords_strict_mode_rejects() -> None:
    class Inner(BaseModel):
        value: int = Field(ge=1, le=5)

    class Outer(BaseModel):
        title: str = Field(default="x", min_length=1, max_length=10)
        items: list[Inner] = Field(default_factory=list, max_length=3)

    client, transport, _ = make_test_client(
        make_config(provider="openai", strategy="json_mode"),
        [openai_reply('{"title": "a", "items": [{"value": 2}]}')],
    )

    assert client.generate([user()], Outer) == Outer(title="a", items=[Inner(value=2)])

    payload = transport.calls[0][2]["response_format"]["json_schema"]
    assert payload["name"] == "Outer"
    assert payload["strict"] is True
    schema = payload["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["title", "items"]
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["value"]
    assert inner["properties"]["value"]["type"] == "integer"
    text = json.dumps(schema)
    for rejected in (
        "maxLength",
        "minLength",
        "maxItems",
        "minimum",
        "maximum",
        "default",
    ):
        assert rejected not in text


def test_generate_repairs_an_invalid_reply() -> None:
    client, transport, _ = make_test_client(
        make_config(provider="openai"),
        [
            openai_reply("not json"),
            openai_reply('{"answer": "yes", "score": 9}'),
        ],
    )

    assert client.generate([user()], Verdict) == Verdict(answer="yes", score=9)
    assert client.repair_attempts == 1

    repair = transport.calls[1][2]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "not json"}
    assert repair[-1]["role"] == "user"
    assert "no JSON object found" in repair[-1]["content"]
    assert "corrected JSON object" in repair[-1]["content"]


def test_generate_reports_the_failing_field_on_validation_errors() -> None:
    client, transport, _ = make_test_client(
        make_config(provider="openai", repair_retries=1),
        [
            openai_reply('{"answer": "yes", "score": "high"}'),
            openai_reply('{"answer": "yes", "score": 4}'),
        ],
    )

    assert client.generate([user()], Verdict).score == 4
    repair = transport.calls[1][2]["messages"][-1]["content"]
    assert "score" in repair
    assert "valid integer" in repair


def test_generate_gives_up_after_the_configured_repairs() -> None:
    client, transport, _ = make_test_client(
        make_config(provider="openai", repair_retries=1),
        [openai_reply("still wrong"), openai_reply("wrong again")],
    )

    with pytest.raises(StructuredOutputError) as caught:
        client.generate([user()], Verdict)

    error = caught.value
    assert error.attempts == 2
    assert error.last_error == "no JSON object found"
    assert error.last_text == "wrong again"
    assert "after 2 attempts" in str(error)
    assert len(transport.calls) == 2
    assert client.repair_attempts == 1


def test_generate_raises_truncated_output_without_repairing() -> None:
    client, transport, _ = make_test_client(
        make_config(provider="openai"),
        [openai_reply("partial", finish_reason="length")],
    )

    with pytest.raises(TruncatedOutput, match="output limit"):
        client.generate([user()], Verdict)

    assert len(transport.calls) == 1


def test_fake_llm_replays_strings_completions_and_exceptions() -> None:
    fake = FakeLLM(
        [
            Completion(
                text="exact",
                finish_reason="stop",
                truncated=False,
                input_tokens=1,
                output_tokens=2,
                latency_ms=0,
                provider="fake",
                model="fake",
            ),
            "plain",
            RuntimeError("boom"),
        ]
    )

    assert fake.complete([user()]).text == "exact"
    assert fake.complete([user()], json_mode=True).text == "plain"
    with pytest.raises(RuntimeError, match="boom"):
        fake.complete([user()])

    assert fake.calls == [
        {"messages": [user()], "json_mode": False},
        {"messages": [user()], "json_mode": True},
        {"messages": [user()], "json_mode": False},
    ]
    assert (fake.provider, fake.model, fake.strategy) == (
        "fake",
        "fake",
        "prompt",
    )


def test_fake_llm_asserts_when_exhausted() -> None:
    fake = FakeLLM([])

    with pytest.raises(AssertionError, match="no more responses"):
        fake.complete([user()])


def test_fake_llm_generates_with_the_shared_structured_path() -> None:
    fake = FakeLLM(['{"answer": "yes", "score": 2}'], strategy="json_mode")

    assert fake.generate([user()], Verdict) == Verdict(answer="yes", score=2)
    assert fake.calls[0]["json_mode"] is True
    assert fake.calls[0]["messages"][1] == user()


# -- stdlib urllib transport ------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, headers: Mapping[str, str], raw: bytes) -> None:
        self.status = status
        self.headers = dict(headers)
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_urllib_transport_posts_json_and_decodes_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = {key.lower(): value for key, value in request.header_items()}
        seen["body"] = json.loads(cast(bytes, request.data or b"{}"))
        seen["timeout"] = timeout
        return FakeResponse(200, {"Content-Type": "application/json"}, b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    status, headers, body = urllib_transport(
        "https://api.test/v1/chat",
        {"Authorization": "Bearer secret"},
        {"model": "m"},
        2.5,
    )

    assert (status, headers, body) == (
        200,
        {"content-type": "application/json"},
        {"ok": True},
    )
    assert seen == {
        "url": "https://api.test/v1/chat",
        "method": "POST",
        "headers": {
            "content-type": "application/json",
            "user-agent": "notebook-harness/0.1",
            "authorization": "Bearer secret",
        },
        "body": {"model": "m"},
        "timeout": 2.5,
    }


def test_urllib_transport_returns_http_errors_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = Message()
    headers["Retry-After"] = "3"

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            headers,
            io.BytesIO(b'{"error": {"message": "slow down"}}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    status, response_headers, body = urllib_transport(
        "https://api.test/v1", {}, {}, 1.0
    )

    assert status == 429
    assert response_headers == {"retry-after": "3"}
    assert body == {"error": {"message": "slow down"}}


@pytest.mark.parametrize(
    "raw",
    [b"not json at all", b"[1, 2]", b"\xff\xfe"],
)
def test_urllib_transport_keeps_unparseable_bodies(
    raw: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(500, {}, raw),
    )

    status, _, body = urllib_transport("https://api.test", {}, {}, 1.0)

    assert status == 500
    assert set(body) == {"_raw"}


def test_urllib_transport_truncates_raw_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(500, {}, b"x" * 900),
    )

    _, _, body = urllib_transport("https://api.test", {}, {}, 1.0)

    assert body["_raw"] == "x" * 500


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("dns"),
        TimeoutError("slow"),
        OSError("refused"),
    ],
)
def test_urllib_transport_wraps_network_failures(
    failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise failure

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(TransportError, match="network error talking to") as caught:
        urllib_transport("https://api.test", {}, {}, 1.0)

    assert caught.value.__cause__ is failure


def test_local_client_without_a_key_does_not_scrub_errors() -> None:
    cfg = make_config(
        provider="local",
        api_key_env=None,
        base_url="http://local/v1",
        max_retries=0,
    )
    client, _, _ = make_test_client(cfg, [TransportError("connection refused")], env={})

    with pytest.raises(LLMTransientError, match="connection refused") as caught:
        client.complete([user()])

    assert "***" not in str(caught.value)


# -- CLI -------------------------------------------------------------------


def test_main_prints_the_completion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeLLM(
        [
            Completion(
                text="hello from the model",
                finish_reason="stop",
                truncated=False,
                input_tokens=3,
                output_tokens=4,
                latency_ms=0,
                provider="fake",
                model="fake",
            )
        ]
    )
    order: list[str] = []

    def fake_load_dotenv() -> None:
        order.append("dotenv")

    def fake_load_config() -> LLMConfig:
        order.append("config")
        return make_config()

    monkeypatch.setattr(llm, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(llm, "load_config", fake_load_config)
    monkeypatch.setattr(llm, "make_client", lambda cfg: fake)

    assert llm.main(["say hi"]) == 0

    assert order == ["dotenv", "config"]

    out = capsys.readouterr().out
    assert "provider: fake" in out
    assert "model: fake" in out
    assert "strategy: prompt" in out
    assert "hello from the model" in out
    assert "tokens: input=3 output=4" in out
    assert fake.calls[0]["messages"] == [user("say hi")]


@pytest.mark.parametrize(
    "failure",
    [ConfigError("config.toml is missing"), LLMAuthError("bad key")],
)
def test_main_reports_expected_failures_on_stderr(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_failure(*args: object, **kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setattr(llm, "load_config", lambda: make_config())
    monkeypatch.setattr(llm, "make_client", raise_failure)

    assert llm.main(["hi"]) == 1

    captured = capsys.readouterr()
    assert str(failure) in captured.err
    assert captured.out == ""


def test_main_requires_a_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        llm.main([])

    assert caught.value.code == 2
    assert "prompt" in capsys.readouterr().err
