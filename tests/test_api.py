"""End-to-end tests for the HTTP and WebSocket surface against a fake model pool."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

from turkish_tts.api import app as api_app
from turkish_tts.api import routes_http, routes_ws
from turkish_tts.engine import PoolBusy
from tests.test_pool import make_pool

SAMPLE_RATE = 48000
CHUNKS = 4
CHUNK_BYTES = 200
WAV_HEADER_BYTES = 44

recorded_voices = []


def fake_model():
    """Builds a stand-in model exposing only the output sample rate."""
    return SimpleNamespace(tts_model=SimpleNamespace(sample_rate=SAMPLE_RATE))


async def fake_synthesize_stream(model, text, voice=None):
    """Yields four PCM16 chunks, flagging the last as sentence final."""
    recorded_voices.append(voice)
    for index in range(CHUNKS):
        yield SimpleNamespace(
            data=b"\x01\x00" * (CHUNK_BYTES // 2),
            sample_rate=SAMPLE_RATE,
            chunk_index=index,
            sentence_index=0,
            is_final=index == CHUNKS - 1,
            is_sentence_final=index == CHUNKS - 1,
        )


@pytest.fixture
def pool(monkeypatch):
    """Replaces the global pool with a one-model fake wherever it is referenced."""
    fake_pool = make_pool()
    fake_pool._idle.get_nowait()
    fake_pool._idle.put_nowait(fake_model())

    for module in (api_app, routes_http, routes_ws):
        monkeypatch.setattr(module, "pool", fake_pool, raising=True)
    for module in (routes_http, routes_ws):
        monkeypatch.setattr(module, "synthesize_stream", fake_synthesize_stream, raising=True)
    return fake_pool


def request(method: str, url: str, **kwargs) -> httpx.Response:
    """Performs one HTTP round trip against the ASGI app."""

    async def run():
        """Opens a client bound to the app and sends the request."""
        transport = httpx.ASGITransport(app=api_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_health_is_immediate(pool):
    """Liveness answers without touching the model."""
    response = request("GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_pool_state(pool):
    """Readiness flips to 503 when the pool is not loaded."""
    assert request("GET", "/ready").status_code == 200

    pool._ready = False
    response = request("GET", "/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_synthesize_returns_wav_and_releases_model(pool):
    """A complete synthesis returns a valid WAV and frees the model."""
    response = request("POST", "/v1/tts", json={"text": "Merhaba dünya."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"
    assert len(response.content) == WAV_HEADER_BYTES + CHUNKS * CHUNK_BYTES
    assert pool.idle_count == 1


def test_empty_text_is_rejected(pool):
    """Missing or blank text fails validation."""
    assert request("POST", "/v1/tts", json={"text": ""}).status_code == 422
    assert request("POST", "/v1/tts", json={}).status_code == 422


def test_busy_pool_answers_503(pool, monkeypatch):
    """A busy pool produces 503 rather than hanging."""

    class BusyPool:
        """Pool stub that is always busy."""

        is_ready, size, idle_count = True, 1, 0

        def acquire(self):
            """Always reports the pool as busy."""
            raise PoolBusy("test: havuz dolu")

    monkeypatch.setattr(routes_http, "pool", BusyPool(), raising=True)
    assert request("POST", "/v1/tts", json={"text": "Merhaba."}).status_code == 503


def test_streaming_response_releases_model(pool):
    """A fully consumed stream frees the model."""

    async def run():
        """Reads the whole streaming response body."""
        transport = httpx.ASGITransport(app=api_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("POST", "/v1/tts/stream", json={"text": "Merhaba."}) as r:
                assert r.status_code == 200
                return await r.aread()

    body = asyncio.run(run())
    assert body[:4] == b"RIFF"
    assert len(body) == WAV_HEADER_BYTES + CHUNKS * CHUNK_BYTES
    assert pool.idle_count == 1


class FakeWebSocket:
    """Minimal WebSocket stand-in so the router can be driven directly."""

    def __init__(self, incoming, disconnect_after_bytes=None):
        """Queues the client messages and the optional disconnect point."""
        self._incoming = list(incoming)
        self._disconnect_after_bytes = disconnect_after_bytes
        self._bytes_sent = 0
        self.sent = []
        self.closed_with = None
        self.client_state = WebSocketState.CONNECTED

    async def accept(self):
        """Accepts the handshake."""

    async def receive_json(self):
        """Returns the next queued message, then simulates a disconnect."""
        if not self._incoming:
            self.client_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(1000)
        return self._incoming.pop(0)

    async def send_json(self, data):
        """Records an outgoing control frame."""
        self.sent.append(("json", data))

    async def send_bytes(self, data):
        """Records an outgoing audio frame, dropping the client if configured."""
        self._bytes_sent += 1
        if self._disconnect_after_bytes and self._bytes_sent > self._disconnect_after_bytes:
            self.client_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(1006)
        self.sent.append(("bytes", data))

    async def close(self, code=1000):
        """Records the close code."""
        self.closed_with = code
        self.client_state = WebSocketState.DISCONNECTED

    def events(self):
        """Returns the names of every control frame sent."""
        return [payload["event"] for kind, payload in self.sent if kind == "json"]

    def audio(self):
        """Returns every audio frame joined together."""
        return b"".join(payload for kind, payload in self.sent if kind == "bytes")

    def event(self, name):
        """Returns the first control frame with the given event name."""
        return next(p for k, p in self.sent if k == "json" and p["event"] == name)


def drive(ws: FakeWebSocket) -> FakeWebSocket:
    """Runs the WebSocket endpoint against a fake client to completion."""
    asyncio.run(routes_ws.websocket_tts(ws))
    return ws


def test_websocket_streams_and_accepts_multiple_requests(pool):
    """One connection serves several synthesis requests in sequence."""
    ws = drive(
        FakeWebSocket(
            [
                {"text": "Birinci cümle."},
                {"text": "İkinci cümle."},
                {"event": "close"},
            ]
        )
    )

    assert ws.events() == [
        "stream_start", "sentence_complete", "stream_end",
        "stream_start", "sentence_complete", "stream_end",
    ]
    assert len(ws.audio()) == 2 * CHUNKS * CHUNK_BYTES
    assert ws.closed_with == 1000
    assert pool.idle_count == 1


def test_websocket_stream_start_reports_real_sample_rate(pool):
    """stream_start advertises the model's real 48 kHz output."""
    ws = drive(FakeWebSocket([{"text": "Merhaba."}, {"event": "close"}]))
    start = ws.event("stream_start")

    assert start["sample_rate"] == SAMPLE_RATE
    assert start["format"] == "pcm16_le"


def test_websocket_error_always_carries_error_type(pool):
    """Error frames include error_type, which integration clients read."""
    ws = drive(FakeWebSocket([{"text": "   "}]))
    error = ws.event("error")

    assert error["error_type"] == "ValidationError"
    assert error["message"]
    assert ws.closed_with == 1008


def test_websocket_disconnect_midstream_releases_model(pool):
    """The original deadlock, reproduced through the real endpoint."""
    ws = drive(FakeWebSocket([{"text": "Merhaba dünya."}], disconnect_after_bytes=1))

    assert "stream_end" not in ws.events()
    assert pool.idle_count == 1


def test_websocket_busy_pool_closes_with_try_again_later(pool, monkeypatch):
    """A busy pool closes the socket with 1013 instead of hanging."""

    class BusyPool:
        """Pool stub that is always busy."""

        size = 1

        def acquire(self):
            """Always reports the pool as busy."""
            raise PoolBusy("test: havuz dolu")

    monkeypatch.setattr(routes_ws, "pool", BusyPool(), raising=True)
    ws = drive(FakeWebSocket([{"text": "Merhaba."}]))
    error = ws.event("error")

    assert error["error_type"] == "PoolBusy"
    assert ws.closed_with == 1013


# --- ses klonlama -----------------------------------------------------------


def test_voices_endpoint_lists_the_reference_voices(pool):
    """The voices listing reports each voice and whether it has a transcript."""
    response = request("GET", "/v1/voices")

    assert response.status_code == 200
    names = [v["name"] for v in response.json()["voices"]]
    assert "kadın" in names
    assert all("has_transcript" in v for v in response.json()["voices"])


def test_synthesize_accepts_a_known_voice(pool):
    """A known voice name is resolved and handed to the generator."""
    recorded_voices.clear()
    response = request("POST", "/v1/tts", json={"text": "Merhaba.", "voice": "kadın"})

    assert response.status_code == 200
    assert recorded_voices and recorded_voices[-1] is not None
    assert recorded_voices[-1].name == "kadın"


def test_synthesize_without_voice_passes_none(pool):
    """Omitting the voice leaves generation on the default speaker."""
    recorded_voices.clear()
    assert request("POST", "/v1/tts", json={"text": "Merhaba."}).status_code == 200

    assert recorded_voices == [None]


def test_unknown_voice_is_rejected_with_422(pool):
    """An unknown voice name fails validation and names the available ones."""
    response = request("POST", "/v1/tts", json={"text": "Merhaba.", "voice": "yokboyle"})

    assert response.status_code == 422
    assert "yokboyle" in response.json()["detail"]


def test_websocket_accepts_a_voice_per_message(pool):
    """Each WebSocket request may pick its own voice."""
    recorded_voices.clear()
    drive(FakeWebSocket([{"text": "Merhaba.", "voice": "kadın"}, {"event": "close"}]))

    assert recorded_voices and recorded_voices[-1].name == "kadın"


def test_websocket_unknown_voice_reports_voice_not_found(pool):
    """An unknown voice closes the socket with a named error."""
    ws = drive(FakeWebSocket([{"text": "Merhaba.", "voice": "yokboyle"}]))
    error = ws.event("error")

    assert error["error_type"] == "VoiceNotFound"
    assert ws.closed_with == 1008
