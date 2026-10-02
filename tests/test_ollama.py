from melbourne_mate.generation.contracts import GenerationRequest
from melbourne_mate.generation.ollama import OllamaClient


def test_generate_keeps_model_loaded_for_follow_up_questions():
    sent = {}

    def transport(path, payload):
        sent["path"] = path
        sent["payload"] = payload
        return {"response": "OK", "done_reason": "stop"}

    client = OllamaClient(transport=transport)
    client.generate(GenerationRequest("Reply only OK."))

    assert sent["path"] == "/api/generate"
    assert sent["payload"]["keep_alive"] == "30m"
    assert sent["payload"]["stream"] is False


def test_stream_keeps_model_loaded_for_follow_up_questions():
    sent = {}

    def stream_transport(payload):
        sent["payload"] = payload
        yield {"response": "O", "done": False}
        yield {"response": "K", "done": True, "done_reason": "stop"}

    client = OllamaClient(stream_transport=stream_transport)
    pieces = list(client.stream(GenerationRequest("Reply only OK.")))

    assert pieces == ["O", "K"]
    assert sent["payload"]["keep_alive"] == "30m"
    assert sent["payload"]["stream"] is True
    assert client.last_response is not None
    assert client.last_response.text == "OK"
