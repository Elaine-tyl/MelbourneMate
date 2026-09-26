"""Small test models used by unit and integration tests."""

from __future__ import annotations

from melbourne_mate.generation.contracts import GenerationRequest, GenerationResponse


class EchoModel:
    """Returns a fixed reply and records the prompts it was given."""

    model = "echo"

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.prompts.append(request.prompt)
        return GenerationResponse(
            text=self.reply or "(no answer configured)", model=self.model
        )


class ScriptedModel:
    """Return prepared responses in order."""

    model = "scripted"

    def __init__(self, responses: list[GenerationResponse]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.prompts.append(request.prompt)
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of prepared responses")
        return self.responses.pop(0)
