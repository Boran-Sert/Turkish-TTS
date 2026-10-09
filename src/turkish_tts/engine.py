"""Pool of VoxCPM model instances, one per GPU, rented out with a timeout."""

import asyncio
from contextlib import aclosing, asynccontextmanager
from typing import AsyncGenerator, AsyncIterator, List, Optional

import torch

from ._vendor.voxcpm import (
    StreamingTextSource,
    TextBuffer,
    VoxCPM,
    async_generate_stream,
)
from ._vendor.voxcpm.streaming import AudioChunk

from .logger import get_logger
from .settings import settings
from .voices import Voice

logger = get_logger()


class PoolBusy(Exception):
    """Raised when no model instance became free within the acquire timeout."""


def _resolve_devices() -> List[str]:
    """Maps the configured GPU ids onto devices that actually exist."""
    requested = settings.system.voxcpm_gpu_ids
    present = torch.cuda.device_count()

    if present == 0:
        raise RuntimeError(
            "CUDA cihazı bulunamadı. Bu servis GPU gerektirir. "
            "torch CPU sürümüyle kurulmuş olabilir; CUDA tekerleği için "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121 kullanın."
        )

    usable = [gpu_id for gpu_id in requested if gpu_id < present]
    if not usable:
        raise RuntimeError(
            f"tts_config.json içindeki voxcpm_gpu_ids={requested} geçersiz: "
            f"makinede {present} GPU var (geçerli id'ler 0..{present - 1})."
        )

    return [f"cuda:{gpu_id}" for gpu_id in usable]


class VoxCPMPool:
    """Holds one loaded VoxCPM instance per GPU and hands them out one at a time."""

    def __init__(self) -> None:
        """Creates an empty pool; models are loaded by initialize()."""
        self._idle: asyncio.Queue = asyncio.Queue()
        self._devices: List[str] = []
        self._ready = False
        self._init_lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Number of model instances the pool was built with."""
        return len(self._devices)

    @property
    def idle_count(self) -> int:
        """Number of model instances currently free."""
        return self._idle.qsize()

    @property
    def is_ready(self) -> bool:
        """Whether every model instance has finished loading."""
        return self._ready

    async def initialize(self) -> None:
        """Loads one model per configured device, raising if no usable GPU exists."""
        async with self._init_lock:
            if self._ready:
                return

            self._devices = _resolve_devices()
            for device in self._devices:
                logger.info(f"Model yükleniyor: {device}")
                model = await asyncio.to_thread(
                    VoxCPM.from_pretrained,
                    hf_model_id=settings.model.model_path,
                    device=device,
                    optimize=settings.system.use_torch_compile,
                    load_denoiser=False,
                    max_length=settings.model.max_length,
                )
                self._idle.put_nowait(model)

            self._ready = True
            logger.info(f"Havuz hazır: {self.size} model ({', '.join(self._devices)}).")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[VoxCPM]:
        """Rents an idle model and always returns it when the context exits."""
        timeout = settings.system.pool_acquire_timeout_s
        try:
            model = await asyncio.wait_for(self._idle.get(), timeout)
        except asyncio.TimeoutError as exc:
            raise PoolBusy(
                f"Tüm modeller meşgul ({self.size} adet), {timeout} saniyede boşalmadı."
            ) from exc

        try:
            yield model
        finally:
            self._idle.put_nowait(model)


def sample_rate_of(model: VoxCPM) -> int:
    """Returns the model's output sample rate, 48 kHz for VoxCPM2."""
    return model.tts_model.sample_rate


def sentence_source(text: str) -> StreamingTextSource:
    """Wraps a complete text in the sentence-splitting source the model consumes."""
    buffer = TextBuffer(
        sentence_delimiters=settings.text_processing.sentence_delimiters,
        flush_timeout_ms=settings.text_processing.flush_timeout_ms,
        min_chars=settings.text_processing.min_chars,
    )
    source = StreamingTextSource(buffer)
    source.push_text(text)
    source.finish()
    return source


def prompt_arguments(voice: Optional[Voice], continuation: bool = False) -> dict:
    """Builds the model's prompt arguments for a voice, keeping it stable across sentences."""
    if voice is None:
        return {}

    if continuation and settings.streaming.lookbehind_mode == "strict":
        logger.warning(
            "lookbehind_mode='strict' devam modundaki sesi ilk cumleden sonra bozar; "
            "referans moduna dusuluyor."
        )
        continuation = False
    return voice.as_prompt(continuation=continuation)


async def synthesize_stream(
    model: VoxCPM,
    text: str,
    voice: Optional[Voice] = None,
) -> AsyncGenerator[AudioChunk, None]:
    """Splits text into sentences and streams the synthesized audio chunks."""
    source = sentence_source(text)
    stream = async_generate_stream(
        model.generate_stream_from_text_source, source, **prompt_arguments(voice)
    )
    async with aclosing(stream) as chunks:
        async for chunk in chunks:
            yield chunk


pool = VoxCPMPool()
