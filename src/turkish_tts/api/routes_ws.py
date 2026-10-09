"""WebSocket endpoint streaming JSON control frames alongside binary PCM16 audio."""

import asyncio
from contextlib import aclosing

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..engine import PoolBusy, pool, sample_rate_of, synthesize_stream
from ..settings import settings
from ..voices import VoiceNotFound, resolve_voice
from ..logger import get_logger

router = APIRouter()
logger = get_logger()

CLOSE_NORMAL = 1000
CLOSE_POLICY_VIOLATION = 1008
CLOSE_INTERNAL_ERROR = 1011
CLOSE_TRY_AGAIN_LATER = 1013


async def _send_error(websocket: WebSocket, error_type: str, message: str) -> None:
    """Sends an error frame that always carries error_type."""
    await websocket.send_json({"event": "error", "error_type": error_type, "message": message})


async def _synthesize_one(websocket: WebSocket, text: str, voice=None) -> None:
    """Runs one synthesis, holding a pooled model only for its duration."""
    async with pool.acquire() as model:
        await websocket.send_json(
            {
                "event": "stream_start",
                "format": "pcm16_le",
                "sample_rate": sample_rate_of(model),
            }
        )
        async with aclosing(synthesize_stream(model, text, voice)) as chunks:
            async for chunk in chunks:
                if chunk.data:
                    await websocket.send_bytes(chunk.data)
                if chunk.is_sentence_final:
                    await websocket.send_json(
                        {"event": "sentence_complete", "sentence_index": chunk.sentence_index}
                    )
    await websocket.send_json({"event": "stream_end"})


@router.websocket("/v1/tts/stream")
async def websocket_tts(websocket: WebSocket) -> None:
    """Serves any number of {"text": ...} requests over one connection."""
    await websocket.accept()
    idle_timeout = settings.system.ws_idle_timeout_s
    logger.info("WebSocket istemcisi bağlandı.")

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), idle_timeout)
            except asyncio.TimeoutError:
                await _send_error(
                    websocket, "IdleTimeout", f"{idle_timeout} saniye boyunca mesaj gelmedi."
                )
                await websocket.close(code=CLOSE_NORMAL)
                return

            if message.get("event") == "close":
                await websocket.close(code=CLOSE_NORMAL)
                return

            text = (message.get("text") or "").strip()
            if not text:
                await _send_error(websocket, "ValidationError", "text alanı zorunlu.")
                await websocket.close(code=CLOSE_POLICY_VIOLATION)
                return

            try:
                voice = resolve_voice(message.get("voice"))
            except VoiceNotFound as exc:
                await _send_error(websocket, "VoiceNotFound", str(exc))
                await websocket.close(code=CLOSE_POLICY_VIOLATION)
                return

            try:
                await _synthesize_one(websocket, text, voice)
            except PoolBusy as exc:
                await _send_error(websocket, "PoolBusy", str(exc))
                await websocket.close(code=CLOSE_TRY_AGAIN_LATER)
                return

    except WebSocketDisconnect:
        logger.info("WebSocket istemcisi bağlantıyı kapattı.")
    except Exception as exc:
        logger.exception(f"WebSocket akışı hata ile sonlandı: {exc}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await _send_error(websocket, type(exc).__name__, str(exc))
            await websocket.close(code=CLOSE_INTERNAL_ERROR)
