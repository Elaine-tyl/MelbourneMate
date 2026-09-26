from melbourne_mate.generation.citations import (
    CitationReport,
    check_citations,
    extract_citations,
)
from melbourne_mate.generation.contracts import (
    GenerationRequest,
    GenerationResponse,
    LanguageModel,
)
from melbourne_mate.generation.gate import EvidenceGate, GateDecision
from melbourne_mate.generation.ollama import OllamaClient, OllamaError
from melbourne_mate.generation.prompts import FALLBACK_TEXT, build_prompt

__all__ = [
    "FALLBACK_TEXT",
    "CitationReport",
    "EvidenceGate",
    "GateDecision",
    "GenerationRequest",
    "GenerationResponse",
    "LanguageModel",
    "OllamaClient",
    "OllamaError",
    "build_prompt",
    "check_citations",
    "extract_citations",
]
