"""Bridges a synchronous chunk generator into the asyncio event loop."""

import asyncio
from typing import Any, AsyncGenerator, Callable

from .streaming import AudioChunk


async def async_generate_stream(
    generator_func: Callable[..., Any],
    *args,
    **kwargs,
) -> AsyncGenerator[AudioChunk, None]:
    """Yields chunks from a sync generator, closing it even if the consumer stops early."""
    gen = generator_func(*args, **kwargs)
    try:
        while True:
            chunk = await asyncio.to_thread(next, gen, None)
            if chunk is None:
                break
            yield chunk
    finally:
        gen.close()
