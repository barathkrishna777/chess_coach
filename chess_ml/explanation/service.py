"""High-level explanation orchestration: prompt, cache, provider, validation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, cast

from chess_ml.explanation.cache import ExplanationCache, cache_key_for_facts
from chess_ml.explanation.client import (
    DEFAULT_TIMEOUT_SECONDS,
    ExplanationClient,
    LocalProviderUnavailableError,
    ProviderError,
    SelectionReason,
    select_client_from_env,
)
from chess_ml.explanation.models import ExplanationProvider, ExplanationRequest, MoveExplanation
from chess_ml.explanation.prompt import (
    InvalidExplanationResponseError,
    build_prompt,
    validate_provider_response,
)


@dataclass(frozen=True)
class ExplanationServiceStatus:
    """Non-probing explanation provider status for the UI."""

    enabled: bool
    configured: bool
    provider: ExplanationProvider | None
    model: str | None
    timeout_seconds: float
    availability: Literal["not_checked"] = "not_checked"
    reason: SelectionReason | None = None


class ExplanationService:
    """Generate or retrieve grounded coaching explanations for flagged moves."""

    def __init__(
        self,
        *,
        cache: ExplanationCache,
        client: ExplanationClient | None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        enabled: bool = True,
        configured: bool | None = None,
        provider: ExplanationProvider | None = None,
        model: str | None = None,
        status_reason: SelectionReason | None = None,
    ) -> None:
        self.cache = cache
        self.client = client
        self.timeout_seconds = timeout_seconds
        self._status = ExplanationServiceStatus(
            enabled=enabled,
            configured=(client is not None if configured is None else configured),
            provider=client.provider if client is not None else provider,
            model=client.model if client is not None else model,
            timeout_seconds=timeout_seconds,
            reason=status_reason,
        )

    def status(self) -> ExplanationServiceStatus:
        """Return explanation config without probing a local or hosted provider."""

        return self._status

    async def explain(self, request: ExplanationRequest) -> MoveExplanation | None:
        if not request.motifs:
            return None
        if self.client is None:
            return MoveExplanation(
                status="unavailable",
                text=None,
                source=None,
                provider=self._status.provider,
                model=self._status.model,
                reason="api_key_missing",
                timeout_seconds=self.timeout_seconds,
                retryable=False,
            )

        prompt = build_prompt(request)
        cache_key = cache_key_for_facts(prompt.facts)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return MoveExplanation(
                status="ok",
                text=cached.text,
                source="cache",
                provider=cast(ExplanationProvider, cached.provider),
                model=cached.model,
                reason=None,
                timeout_seconds=self.timeout_seconds,
                retryable=False,
            )

        try:
            raw_response = await asyncio.wait_for(
                self.client.complete(prompt),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            return MoveExplanation(
                status="error",
                text=None,
                source=None,
                provider=self.client.provider,
                model=self.client.model,
                reason="timeout",
                timeout_seconds=self.timeout_seconds,
                retryable=True,
            )
        except LocalProviderUnavailableError:
            return MoveExplanation(
                status="unavailable",
                text=None,
                source=None,
                provider=self.client.provider,
                model=self.client.model,
                reason="local_model_unavailable",
                timeout_seconds=self.timeout_seconds,
                retryable=True,
            )
        except (ProviderError, OSError, ValueError):
            return MoveExplanation(
                status="error",
                text=None,
                source=None,
                provider=self.client.provider,
                model=self.client.model,
                reason="provider_error",
                timeout_seconds=self.timeout_seconds,
                retryable=True,
            )

        try:
            validated = validate_provider_response(raw_response.content, prompt)
        except InvalidExplanationResponseError:
            return MoveExplanation(
                status="error",
                text=None,
                source=None,
                provider=raw_response.provider,
                model=raw_response.model,
                reason="invalid_response",
                timeout_seconds=self.timeout_seconds,
                retryable=True,
            )

        self.cache.set(
            cache_key=cache_key,
            provider=raw_response.provider,
            model=raw_response.model,
            text=validated.text,
            request_json=prompt.facts,
            response_json=validated.response_json,
        )
        return MoveExplanation(
            status="ok",
            text=validated.text,
            source="llm",
            provider=raw_response.provider,
            model=raw_response.model,
            reason=None,
            timeout_seconds=self.timeout_seconds,
            retryable=False,
        )


async def explain_many(
    service: ExplanationService,
    requests: list[ExplanationRequest],
) -> list[MoveExplanation | None]:
    """Explain flagged moves sequentially to keep local cost and rate limits predictable."""

    results: list[MoveExplanation | None] = []
    for request in requests:
        results.append(await service.explain(request))
        await asyncio.sleep(0)
    return results


def service_from_env() -> ExplanationService:
    """Build the default local explanation service."""

    selection = select_client_from_env()
    return ExplanationService(
        cache=ExplanationCache(),
        client=selection.client,
        timeout_seconds=selection.timeout_seconds,
        enabled=selection.enabled,
        configured=selection.configured,
        provider=selection.provider,
        model=selection.model,
        status_reason=selection.reason,
    )
