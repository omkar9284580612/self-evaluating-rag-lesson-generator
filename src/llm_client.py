"""
llm_client.py
-------------
Provider-agnostic LLM wrapper.

Supported:
- Gemini
- Anthropic
- OpenAI
"""

import json
import os
import time

import requests


class LLMClient:

    def __init__(
        self,
        provider: str = None,
        model: str = None,
    ):

        self.provider = (
            provider
            or os.getenv(
                "LLM_PROVIDER",
                "gemini",
            )
        ).lower()

        self.model = (
            model
            or os.getenv(
                "LLM_MODEL",
                self._default_model(),
            )
        )

        self.api_key = os.getenv(
            self._key_env_name()
        )

        if not self.api_key:
            raise RuntimeError(
                f"Missing API key. "
                f"Set {self._key_env_name()} "
                f"in your .env file."
            )

    def _default_model(self):

        models = {
            "gemini": "gemini-3.6-flash",
            "anthropic": "claude-sonnet-4-6",
            "openai": "gpt-4o-mini",
        }

        if self.provider not in models:
            raise ValueError(
                f"Unsupported provider: "
                f"{self.provider}"
            )

        return models[self.provider]

    def _key_env_name(self):

        keys = {
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

        if self.provider not in keys:
            raise ValueError(
                f"Unsupported provider: "
                f"{self.provider}"
            )

        return keys[self.provider]

    def complete(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 2000,
        retries: int = 3,
        json_mode: bool = False,
    ) -> str:

        for attempt in range(
            1,
            retries + 1,
        ):

            try:

                if self.provider == "gemini":
                    return self._call_gemini(
                        prompt,
                        temperature,
                        max_tokens,
                        json_mode,
                    )

                if self.provider == "anthropic":
                    return self._call_anthropic(
                        prompt,
                        temperature,
                        max_tokens,
                    )

                if self.provider == "openai":
                    return self._call_openai(
                        prompt,
                        temperature,
                        max_tokens,
                    )

                raise ValueError(
                    f"Unknown provider: "
                    f"{self.provider}"
                )

            except requests.RequestException as error:

                if attempt == retries:
                    raise

                wait = 2 ** attempt

                print(
                    f"[LLMClient] call failed "
                    f"({error}); retrying "
                    f"in {wait}s..."
                )

                time.sleep(wait)

    def _call_gemini(
        self,
        prompt,
        temperature,
        max_tokens,
        json_mode=False,
    ):

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        generation_config = {
            "maxOutputTokens": max_tokens,
            "responseMimeType": (
                "application/json"
                if json_mode
                else "text/plain"
            ),
        }

        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": generation_config,
        }

        response = requests.post(
            url,
            json=body,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        candidates = data.get(
            "candidates",
            [],
        )

        if not candidates:
            raise ValueError(
                f"Gemini returned no candidates: "
                f"{data}"
            )

        candidate = candidates[0]

        finish_reason = candidate.get(
            "finishReason",
            "UNKNOWN",
        )

        print(
            f"[Gemini] finishReason: "
            f"{finish_reason}"
        )

        parts = (
            candidate
            .get("content", {})
            .get("parts", [])
        )

        text = "".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        if not text:
            raise ValueError(
                "Gemini returned empty output."
            )

        return text

    def _call_anthropic(
        self,
        prompt,
        temperature,
        max_tokens,
    ):

        url = (
            "https://api.anthropic.com/"
            "v1/messages"
        )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return "".join(
            block.get("text", "")
            for block in data.get(
                "content",
                [],
            )
        ).strip()

    def _call_openai(
        self,
        prompt,
        temperature,
        max_tokens,
    ):

        url = (
            "https://api.openai.com/"
            "v1/chat/completions"
        )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return (
            data["choices"][0]["message"]["content"]
            .strip()
        )


def extract_json(text: str) -> dict:

    text = text.strip()

    if "```json" in text:

        text = (
            text
            .split("```json", 1)[1]
            .split("```", 1)[0]
            .strip()
        )

    elif "```" in text:

        parts = text.split("```")

        if len(parts) >= 2:
            text = parts[1].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in evaluator output."
        )

    json_text = text[
        start:end + 1
    ]

    try:
        result = json.loads(
            json_text
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Invalid evaluator JSON: "
            f"{error}\n\n"
            f"Raw output:\n{text}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "Evaluator JSON must be an object."
        )

    return result