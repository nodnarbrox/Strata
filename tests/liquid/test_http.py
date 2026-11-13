import json
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from strata_cli.liquid_cmd import make_server


@pytest.fixture
def endpoint():
    class Generator:
        engine = SimpleNamespace(context=4096)
        calls = 0
        def generate(self, messages, max_tokens, **kwargs):
            self.calls += 1
            assert messages == [{"role": "user", "content": "hello"}]
            return dict(content="orchid", reasoning_content="checked", finish_reason="stop", prompt_tokens=3, completion_tokens=4,
                        timings={"predicted_n": 3, "predicted_ms": 150, "predicted_per_second": 20.0})
    generator = Generator()
    server = make_server(generator, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try: yield f"http://127.0.0.1:{server.server_port}", generator
    finally: server.shutdown(); server.server_close(); worker.join()


def request(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def test_http_discovery_response_and_sse(endpoint):
    base, generator = endpoint
    status = json.load(urllib.request.urlopen(base + "/v1/status"))
    assert status["service"] == "strata" and status["backend"] == "strata-liquid-native-cpu"
    assert status["loaded"] and status["context"]["native"] == 4096
    payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 20}
    with request(base, payload) as response: body = json.load(response)
    assert body["choices"][0]["message"]["content"] == "orchid"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert body["timings"]["predicted_per_second"] == 20.0
    with request(base, {**payload, "stream": True}) as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        wire = response.read().decode()
    chunks = [line.removeprefix("data: ") for line in wire.splitlines() if line.startswith("data: ")]
    assert chunks[-1] == "[DONE]"
    assert json.loads(chunks[0])["choices"][0]["delta"]["reasoning_content"] == "checked"
    assert json.loads(chunks[1])["choices"][0]["finish_reason"] == "stop"
    assert json.loads(chunks[1])["usage"]["completion_tokens"] == 4
    assert json.loads(chunks[1])["timings"]["predicted_per_second"] == 20.0
    assert generator.calls == 2


@pytest.mark.parametrize("extra", [{"temperature": .7}, {"tools": [{"type": "function"}]}, {"n": 2}, {"model": "other"}, {"response_format": {"type": "json_object"}}])
def test_unsupported_controls_do_not_silently_generate(endpoint, extra):
    base, generator = endpoint
    with pytest.raises(urllib.error.HTTPError) as error:
        request(base, {"messages": [{"role": "user", "content": "hello"}], **extra})
    assert error.value.code == 400
    assert generator.calls == 0
