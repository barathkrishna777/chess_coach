"""High-level explanation orchestration: prompt, cache, provider, validation."""

from __future__ import annotations

import asyncio
from typing import cast

from chess_ml.explanation.cache import ExplanationCache, cache_key_for_facts
from chess_ml.explanation.client import ExplanationClient, ProviderError, client_from_env
from chess_ml.explanation.models import ExplanationProvider, ExplanationRequest, MoveExplanation
from chess_ml.explanation.prompt import (
    InvalidExplanationResponseError,
    build_prompt,
    validate_provider_response,
)


class ExplanationService:
    """Generate or retrieve grounded coaching explanations for flagged moves."""

    def __init__(
        self,
        *,
        cache: ExplanationCache,
        client: ExplanationClient | None,
    ) -> None:
        self.cache = cache
        self.client = client

    async def explain(self, request: ExplanationRequest) -> MoveExplanation | None:
        if not request.motifs:
            return None
        if self.client is None:
            return MoveExplanation(
                status="unavailable",
                text=None,
                source=None,
                provider=None,
                model=None,
                reason="api_key_missing",
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
            )

        try:
            raw_response = await self.client.complete(prompt)
        except TimeoutError:
            return MoveExplanation(
                status="error",
                text=None,
                source=None,
                provider=self.client.provider,
                model=self.client.model,
                reason="timeout",
            )
        except (ProviderError, OSError, ValueError):
            return MoveExplanation(
                status="error",
                text=None,
                source=None,
                provider=self.client.provider,
                model=self.client.model,
                reason="provider_error",
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

    return ExplanationService(cache=ExplanationCache(), client=client_from_env())
