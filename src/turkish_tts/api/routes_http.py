"""HTTP endpoints for health, readiness and speech synthesis."""

import io
import wave
from contextlib import AsyncExitStack, aclosing

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ..engine import PoolBusy, pool, sample_rate_of, synthesize_stream
from ..voices import VoiceNotFound, list_voices, resolve_voice

router = APIRouter()

BYTES_PER_SAMPLE = 2
CHANNELS = 1


class TTSRequest(BaseModel):
    """Synthesis request body."""

    text: str = Field(min_length=1, max_length=5000)
    voice: str | None = Field(default=None, max_length=200)


def _voice_or_422(name: str | None):
    """Resolves a voice name, turning an unknown one into a 422."""
    try:
        return resolve_voice(name)
    except VoiceNotFound as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wraps finished PCM16-LE data in a complete WAV container."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()


def _wav_header(sample_rate: int) -> bytes:
    """Builds a WAV header for a stream whose length is not yet known."""
    unknown_size = 0x7FFFFFFF
    block_align = CHANNELS * BYTES_PER_SAMPLE
    return (
        b"RIFF" + unknown_size.to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + CHANNELS.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + (sample_rate * block_align).to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + (BYTES_PER_SAMPLE * 8).to_bytes(2, "little")
        + b"data" + unknown_size.to_bytes(4, "little")
    )


@router.get("/health")
async def health() -> dict:
    """Answers immediately, even while the model is still loading."""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> JSONResponse:
    """Reports 503 until the model pool has finished loading."""
    return JSONResponse(
        content={"ready": pool.is_ready, "pool_size": pool.size, "idle": pool.idle_count},
        status_code=200 if pool.is_ready else 503,
    )


@router.get("/v1/voices")
async def voices() -> dict:
    """Lists the reference voices available for cloning."""
    return {
        "voices": [
            {"name": voice.name, "has_transcript": voice.has_transcript}
            for voice in list_voices()
        ]
    }


@router.post("/v1/tts")
async def synthesize(request: TTSRequest) -> Response:
    """Synthesizes the whole text and returns one complete WAV file."""
    voice = _voice_or_422(request.voice)
    try:
        async with pool.acquire() as model:
            sample_rate = sample_rate_of(model)
            pcm = bytearray()
            async with aclosing(synthesize_stream(model, request.text, voice)) as chunks:
                async for chunk in chunks:
                    if chunk.data:
                        pcm += chunk.data
    except PoolBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not pcm:
        raise HTTPException(status_code=500, detail="Model ses üretmedi.")

    return Response(content=_wav_bytes(bytes(pcm), sample_rate), media_type="audio/wav")


@router.post("/v1/tts/stream")
async def synthesize_streaming(request: TTSRequest) -> StreamingResponse:
    """Streams a WAV header followed by PCM16-LE frames as they are generated."""
    voice = _voice_or_422(request.voice)
    stack = AsyncExitStack()
    try:
        model = await stack.enter_async_context(pool.acquire())
    except PoolBusy as exc:
        await stack.aclose()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def body():
        """Yields the header and then every generated audio chunk."""
        async with stack:
            yield _wav_header(sample_rate_of(model))
            async with aclosing(synthesize_stream(model, request.text, voice)) as chunks:
                async for chunk in chunks:
                    if chunk.data:
                        yield chunk.data

    return StreamingResponse(body(), media_type="audio/wav")
