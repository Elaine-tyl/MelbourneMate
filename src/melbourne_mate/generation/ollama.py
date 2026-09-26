"""Local Ollama client with fixed settings and model identity tracking."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from time import perf_counter
from typing import Any

from melbourne_mate.config import CONFIG
from melbourne_mate.generation.contracts import GenerationRequest, GenerationResponse

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Talk to Ollama through HTTP or an injected test transport."""

    def __init__(
        self,
        model: str | None = None,
        host: str = DEFAULT_HOST,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        get_transport: Callable[[str], dict[str, Any]] | None = None,
        timeout: int = 180,
        seed: int | None = None,
        stream_transport: Callable[[dict[str, Any]], Iterator[dict[str, Any]]]
        | None = None,
    ) -> None:
        self.model = model or CONFIG.generation.model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = CONFIG.generation.seed if seed is None else seed
        self._transport = transport or self._post
        self._get_transport = get_transport or self._get
        self._stream_transport = stream_transport or self._stream_post
        self.digest = ""
        self.last_response: GenerationResponse | None = None

    def _post(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover
        import requests

        try:
            resp = requests.post(
                f"{self.host}{path}", json=payload, timeout=self.timeout
            )
        except Exception as exc:  # requests can fail before a response exists
            raise OllamaError(
                f"cannot reach Ollama at {self.host}. Start it with `ollama serve`, "
                f"and pull the model with `ollama pull {self.model}`."
            ) from exc
        if resp.status_code == 404:
            raise OllamaError(
                f"model {self.model!r} is not installed. Run `ollama pull {self.model}`."
            )
        if resp.status_code != 200:
            raise OllamaError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _get(self, path: str) -> dict[str, Any]:  # pragma: no cover
        import requests

        try:
            resp = requests.get(f"{self.host}{path}", timeout=self.timeout)
        except Exception as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}") from exc
        if resp.status_code != 200:
            raise OllamaError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def resolve_digest(self) -> str:
        """Resolve the local model digest."""
        try:
            body = self._transport("/api/show", {"model": self.model})
        except OllamaError:
            return ""
        digest = str(body.get("digest") or body.get("details", {}).get("digest") or "")
        if not digest:
            try:
                tags = self._get_transport("/api/tags")
            except OllamaError:
                tags = {}
            for item in tags.get("models", []):
                if item.get("name") == self.model or item.get("model") == self.model:
                    digest = str(item.get("digest") or "")
                    break
        self.digest = digest[:19]
        return self.digest

    def _payload(self, request: GenerationRequest, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": request.prompt,
            "stream": stream,
            "options": {
                "temperature": request.temperature,
                "seed": self.seed,
                "num_predict": request.max_output_tokens,
            },
        }

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """Stream text and keep the final response."""
        self.last_response: GenerationResponse | None = None
        started = perf_counter()
        pieces: list[str] = []
        finish_reason = "stop"

        for event in self._stream_transport(self._payload(request, stream=True)):
            piece = str(event.get("response", ""))
            if piece:
                pieces.append(piece)
                yield piece
            if event.get("done"):
                finish_reason = str(event.get("done_reason") or "stop")

        text = "".join(pieces).strip()
        self.last_response = GenerationResponse(
            text=text,
            model=f"{self.model}@{self.digest}" if self.digest else self.model,
            latency_ms=int((perf_counter() - started) * 1000),
            finish_reason=finish_reason,
        )

    def _stream_post(
        self, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:  # pragma: no cover
        import requests

        try:
            with requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as resp:
                if resp.status_code != 200:
                    raise OllamaError(
                        f"ollama returned {resp.status_code}: {resp.text[:200]}"
                    )
                for line in resp.iter_lines():
                    if line:
                        yield json.loads(line)
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = {
            "model": self.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "seed": self.seed,
                "num_predict": request.max_output_tokens,
            },
        }
        started = perf_counter()
        body = self._transport("/api/generate", payload)
        ms = int((perf_counter() - started) * 1000)

        text = str(body.get("response", "")).strip()
        if not text:
            raise OllamaError("ollama returned an empty answer")
        return GenerationResponse(
            text=text,
            model=f"{self.model}@{self.digest}" if self.digest else self.model,
            latency_ms=ms,
            finish_reason=str(body.get("done_reason") or "stop"),
        )


def spike_report(client: OllamaClient, evidence_text: str) -> dict[str, Any]:
    """Run the four model checks used by the first project gate."""
    from melbourne_mate.generation.citations import extract_citations
    from melbourne_mate.generation.prompts import (
        FALLBACK_TEXT,
        EvidenceItem,
        build_prompt,
    )

    item = EvidenceItem(
        passage_id="P001",
        heading="Work limits while you study",
        text=evidence_text,
        organisation="Department of Home Affairs",
        url="https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500",
    )

    grounded = client.generate(
        GenerationRequest(
            build_prompt("How many hours can I work while studying?", [item])
        )
    )
    refusal = client.generate(
        GenerationRequest(
            build_prompt("Which suburb has the cheapest rent right now?", [item])
        )
    )

    cited = extract_citations(grounded.text)
    fallback_marker = " ".join(FALLBACK_TEXT.lower().split())[:48]

    return {
        "model": grounded.model,
        "cites_in_format": cited == ["P001"],
        "falls_back_verbatim": fallback_marker
        in " ".join(refusal.text.lower().split()),
        "keeps_organisation_name": "Home Affairs" in grounded.text,
        "latency_ms": grounded.latency_ms,
        "latency_acceptable": grounded.latency_ms <= 20_000,
        "grounded_answer": grounded.text,
        "refusal_answer": refusal.text,
    }
