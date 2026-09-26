"""Shared request, response, and model interfaces for generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    temperature: float = 0.0
    max_output_tokens: int = 512


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    model: str
    latency_ms: int = 0
    finish_reason: str = "stop"


class LanguageModel(Protocol):
    model: str

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
