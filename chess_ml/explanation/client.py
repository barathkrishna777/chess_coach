"""Provider adapters for grounded move explanations."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from chess_ml.explanation.models import ExplanationProvider
from chess_ml.explanation.prompt import BuiltPrompt

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
DEFAULT_CODEX_MODEL = "Codex-opus-4-6"
DEFAULT_TIMEOUT_SECONDS = 8.0


class ProviderError(RuntimeError):
    """Raised when an explanation provider request fails."""


@dataclass(frozen=True)
class ClientResponse:
    """Raw provider text plus response metadata."""

    content: str
    response_json: dict[str, Any]
    provider: ExplanationProvider
    model: str


class ExplanationClient(Protocol):
    """Minimal async interface used by the explanation service."""

    provider: ExplanationProvider
    model: str

    async def complete(self, prompt: BuiltPrompt) -> ClientResponse:
        """Return one raw model response for a grounded explanation prompt."""


class AnthropicExplanationClient:
    """Anthropic Messages API adapter using the existing repo key convention."""

    provider: ExplanationProvider = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("CHESS_ML_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.timeout_seconds = timeout_seconds

    async def complete(self, prompt: BuiltPrompt) -> ClientResponse:
        body = {
            "model": self.model,
            "max_tokens": 220,
            "temperature": 0.2,
            "system": [
                {
                    "type": "text",
                    "text": prompt.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": prompt.user_prompt}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }
        data = await asyncio.to_thread(
            _post_json,
            ANTHROPIC_MESSAGES_URL,
            headers,
            body,
            self.timeout_seconds,
        )
        return ClientResponse(
            content=_anthropic_text(data),
            response_json=data,
            provider=self.provider,
            model=self.model,
        )


class CodexExplanationClient:
    """OpenAI Responses-compatible adapter for the Codex provider path."""

    provider: ExplanationProvider = "codex"

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("CHESS_ML_CODEX_MODEL", DEFAULT_CODEX_MODEL)
        self.timeout_seconds = timeout_seconds

    async def complete(self, prompt: BuiltPrompt) -> ClientResponse:
        body = {
            "model": self.model,
            "instructions": prompt.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt.user_prompt}],
                }
            ],
            "max_output_tokens": 220,
            "temperature": 0.2,
            "store": False,
            "prompt_cache_key": "chess-ml-grounded-coach-v1",
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        data = await asyncio.to_thread(
            _post_json,
            OPENAI_RESPONSES_URL,
            headers,
            body,
            self.timeout_seconds,
        )
        return ClientResponse(
            content=_openai_output_text(data),
            response_json=data,
            provider=self.provider,
            model=self.model,
        )


def client_from_env() -> ExplanationClient | None:
    """Create a provider client from local env, or None when disabled/missing."""

    load_dotenv()
    provider = os.environ.get("CHESS_ML_EXPLANATION_PROVIDER", "auto").strip().lower()
    if provider == "disabled":
        return None
    timeout = _env_float("CHESS_ML_EXPLANATION_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    codex_key = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if provider == "anthropic":
        return (
            AnthropicExplanationClient(api_key=anthropic_key, timeout_seconds=timeout)
            if anthropic_key
            else None
        )
    if provider == "codex":
        return (
            CodexExplanationClient(api_key=codex_key, timeout_seconds=timeout)
            if codex_key
            else None
        )
    if provider != "auto":
        return None
    if anthropic_key:
        return AnthropicExplanationClient(api_key=anthropic_key, timeout_seconds=timeout)
    if codex_key:
        return CodexExplanationClient(api_key=codex_key, timeout_seconds=timeout)
    return None


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs into os.environ without adding a dependency."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise TimeoutError("Explanation provider request timed out.") from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(f"Explanation provider returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Explanation provider request failed: {exc.reason}") from exc

    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ProviderError("Explanation provider returned a non-object JSON response.")
    return parsed


def _anthropic_text(response: dict[str, Any]) -> str:
    content = response.get("content")
    if not isinstance(content, list):
        raise ProviderError("Anthropic response did not include content.")
    parts: list[str] = []
    for item in content:
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            parts.append(item["text"])
    if not parts:
        raise ProviderError("Anthropic response did not include text content.")
    return "\n".join(parts)


def _openai_output_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = response.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") in {"output_text", "text"}
                    and isinstance(content_item.get("text"), str)
                ):
                    parts.append(content_item["text"])
        if parts:
            return "\n".join(parts)

    raise ProviderError("Codex response did not include output text.")


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)
