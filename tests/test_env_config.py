import os

from unittest.mock import Mock, patch

from src.llm_client import LLMClient


def test_llm_client_strips_quotes_and_whitespace_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", " groq ")
    monkeypatch.setenv("LLM_MODEL", " openai/gpt-oss-120b ")
    monkeypatch.setenv("GROQ_API_KEY", ' "real-key-value" ')

    client = LLMClient()

    assert client.provider == "groq"
    assert client.model == "openai/gpt-oss-120b"
    assert client.api_key == "real-key-value"


def test_groq_retries_transient_http_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_API_KEY", "real-key-value")

    unavailable = Mock(status_code=503, text="temporarily unavailable")
    success = Mock(
        status_code=200,
        text="",
        json=lambda: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "ok"},
                }
            ]
        },
    )

    with patch("src.llm_client.requests.post", side_effect=[unavailable, success]):
        with patch("src.llm_client.time.sleep") as sleep:
            result = LLMClient().complete("prompt", retries=2)

    assert result == "ok"
    sleep.assert_called_once_with(2)


def test_groq_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_API_KEY", "real-key-value")

    bad_request = Mock(status_code=400, text="bad request")

    with patch("src.llm_client.requests.post", return_value=bad_request) as post:
        with patch("src.llm_client.time.sleep") as sleep:
            try:
                LLMClient().complete("prompt", retries=2)
            except RuntimeError as error:
                assert "HTTP 400" in str(error)
            else:
                raise AssertionError("Expected the client error to be raised")

    post.assert_called_once()
    sleep.assert_not_called()


def test_groq_uses_chat_completions_request(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GROQ_API_KEY", ' "real-key-value" ')

    response = Mock(
        status_code=200,
        text="",
        json=lambda: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "ok"},
                }
            ]
        },
    )

    with patch("src.llm_client.requests.post", return_value=response) as post:
        result = LLMClient().complete("prompt", json_mode=True)

    assert result == "ok"
    request = post.call_args
    assert request.args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert request.kwargs["headers"]["Authorization"] == "Bearer real-key-value"
    assert request.kwargs["json"]["model"] == "openai/gpt-oss-120b"
    assert request.kwargs["json"]["response_format"] == {"type": "json_object"}
