"""Regression tests proving a pooled model is returned on every exit path."""

import asyncio
from contextlib import aclosing

import pytest
import torch

from turkish_tts.engine import PoolBusy, VoxCPMPool, _resolve_devices
from turkish_tts._vendor.voxcpm.streaming_async import async_generate_stream
from turkish_tts.settings import settings


def make_pool(size: int = 1) -> VoxCPMPool:
    """Builds a pool primed with dummy models, bypassing GPU initialization."""
    pool = VoxCPMPool()
    pool._devices = [f"cuda:{i}" for i in range(size)]
    for i in range(size):
        pool._idle.put_nowait(f"model-{i}")
    pool._ready = True
    return pool


async def fake_stream(model):
    """Yields chunks forever until the consumer closes the generator."""
    index = 0
    while True:
        yield f"{model}-chunk-{index}"
        index += 1


def test_model_returned_after_normal_use():
    """A completed request leaves the pool full."""

    async def scenario():
        """Acquires and releases one model."""
        pool = make_pool()
        async with pool.acquire() as model:
            assert model == "model-0"
            assert pool.idle_count == 0
        assert pool.idle_count == 1

    asyncio.run(scenario())


def test_model_returned_when_body_raises():
    """A failing request still returns its model."""

    async def scenario():
        """Raises inside the acquire context."""
        pool = make_pool()
        with pytest.raises(ValueError):
            async with pool.acquire():
                raise ValueError("sentez patladi")
        assert pool.idle_count == 1

    asyncio.run(scenario())


def test_model_returned_when_consumer_abandons_stream():
    """The original deadlock: a consumer that stops iterating halfway."""

    async def scenario():
        """Breaks out of the audio stream after one chunk."""
        pool = make_pool()
        async with pool.acquire() as model:
            async with aclosing(fake_stream(model)) as chunks:
                async for _chunk in chunks:
                    break
        assert pool.idle_count == 1

    asyncio.run(scenario())


def test_pool_exhaustion_raises_pool_busy():
    """An exhausted pool raises PoolBusy instead of waiting forever."""

    async def scenario():
        """Acquires the only model, then asks for a second one."""
        pool = make_pool()
        original = settings.system.pool_acquire_timeout_s
        settings.system.pool_acquire_timeout_s = 0.05
        try:
            async with pool.acquire():
                with pytest.raises(PoolBusy):
                    async with pool.acquire():
                        pass
        finally:
            settings.system.pool_acquire_timeout_s = original
        assert pool.idle_count == 1

    asyncio.run(scenario())


def test_every_model_is_handed_out_before_reuse():
    """Concurrent callers get distinct model instances."""

    async def scenario():
        """Acquires all three models at once."""
        pool = make_pool(size=3)
        async with pool.acquire() as a, pool.acquire() as b, pool.acquire() as c:
            assert {a, b, c} == {"model-0", "model-1", "model-2"}
        assert pool.idle_count == 3

    asyncio.run(scenario())


def test_no_gpu_fails_with_a_clear_message(monkeypatch):
    """A machine without CUDA fails loudly instead of waiting on an empty pool."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="CUDA cihazı bulunamadı"):
        _resolve_devices()


def test_invalid_gpu_ids_fail_with_a_clear_message(monkeypatch):
    """Configured GPU ids that do not exist fail loudly."""
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    original = settings.system.voxcpm_gpu_ids
    settings.system.voxcpm_gpu_ids = [3, 7]
    try:
        with pytest.raises(RuntimeError, match="geçersiz"):
            _resolve_devices()
    finally:
        settings.system.voxcpm_gpu_ids = original


def test_async_bridge_closes_the_inner_generator():
    """The async bridge closes the sync generator so its cleanup runs."""
    closed = []

    def sync_gen():
        """Yields two values and records that it was closed."""
        try:
            yield "a"
            yield "b"
        finally:
            closed.append(True)

    async def scenario():
        """Consumes one chunk and abandons the bridge."""
        async with aclosing(async_generate_stream(sync_gen)) as chunks:
            async for _chunk in chunks:
                break

    asyncio.run(scenario())
    assert closed == [True]


def test_max_length_default_cannot_overflow_the_kv_cache():
    """The default KV window is at least as large as the generation loop's own cap."""
    import inspect

    from turkish_tts._vendor.voxcpm.core import VoxCPM
    from turkish_tts.settings import Settings

    ar_cap = inspect.signature(VoxCPM.generate_stream).parameters["max_len"].default
    assert Settings().model.max_length >= ar_cap, (
        f"max_length={Settings().model.max_length} < uretim siniri {ar_cap}: "
        "StaticKVCache 'KV cache is full' hatasi verebilir"
    )


def test_max_length_reaches_the_vendored_loader():
    """The loader accepts the override the settings provide."""
    import inspect

    from turkish_tts._vendor.voxcpm.core import VoxCPM
    from turkish_tts._vendor.voxcpm.model.voxcpm2 import VoxCPM2Model

    assert "max_length" in inspect.signature(VoxCPM2Model.from_local).parameters
    assert "max_length" in inspect.signature(VoxCPM.__init__).parameters
