"""Turkish text-to-speech built on VoxCPM2 with Trendyol's fine-tuned weights."""

import wave
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .finetune import FinetuneConfig
from .finetune import check_dataset as _check_dataset
from .finetune import finetune as _finetune
from .settings import Settings, load_settings, settings
from .voices import Voice, VoiceNotFound, list_voices, resolve_voice, voices_dir

__version__ = "1.0.0"

BYTES_PER_SAMPLE = 2
CHANNELS = 1

VoiceLike = Union[str, Path, Voice, None]


class TurkishTTS:
    """Loads the model once and synthesizes Turkish speech, optionally in a cloned voice."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        optimize: Optional[bool] = None,
        voice: VoiceLike = None,
    ):
        """Loads the model from a local directory or a Hugging Face repo id."""
        from ._vendor.voxcpm import VoxCPM

        self.default_voice = resolve_voice(voice)
        self._model = VoxCPM.from_pretrained(
            hf_model_id=model_path or settings.model.model_path,
            device=device,
            optimize=settings.system.use_torch_compile if optimize is None else optimize,
            load_denoiser=False,
            max_length=settings.model.max_length,
        )

    # --- bilgi ---------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Output sample rate, 48 kHz for VoxCPM2."""
        return self._model.tts_model.sample_rate

    def info(self) -> dict:
        """Returns what was loaded and the settings that shape generation."""
        return {
            "model_path": settings.model.model_path,
            "device": str(self._model.tts_model.device),
            "sample_rate": self.sample_rate,
            "cfg_value": settings.model.cfg_value,
            "inference_timesteps": settings.model.inference_timesteps,
            "max_length": settings.model.max_length,
            "chunk_duration_ms": settings.streaming.chunk_duration_ms,
            "default_voice": self.default_voice.name if self.default_voice else None,
        }

    # --- sentez --------------------------------------------------------------

    def stream(self, text: str, voice: VoiceLike = None) -> Iterator[bytes]:
        """Yields PCM16-LE audio chunks as each sentence is synthesized."""
        from .engine import prompt_arguments, sentence_source

        chosen = resolve_voice(voice) or self.default_voice
        generator = self._model.generate_stream_from_text_source(
            sentence_source(text), **prompt_arguments(chosen)
        )
        for chunk in generator:
            if chunk.data:
                yield chunk.data

    def synthesize(self, text: str, voice: VoiceLike = None) -> bytes:
        """Returns the whole utterance as PCM16-LE bytes."""
        return b"".join(self.stream(text, voice=voice))

    def save(self, text: str, path: Union[str, Path], voice: VoiceLike = None) -> Path:
        """Synthesizes the text and writes it to a WAV file."""
        destination = Path(path)
        if destination.parent != Path(""):
            destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as out:
            out.setnchannels(CHANNELS)
            out.setsampwidth(BYTES_PER_SAMPLE)
            out.setframerate(self.sample_rate)
            out.writeframes(self.synthesize(text, voice=voice))
        return destination

    def clone(self, text: str, reference: Union[str, Path, Voice]) -> bytes:
        """Synthesizes the text in the voice of a reference recording."""
        return self.synthesize(text, voice=reference)

    def clone_to_file(
        self, text: str, reference: Union[str, Path, Voice], path: Union[str, Path]
    ) -> Path:
        """Synthesizes the text in a reference voice and writes it to a WAV file."""
        return self.save(text, path, voice=reference)

    # --- sesler --------------------------------------------------------------

    @staticmethod
    def list_voices() -> List[Voice]:
        """Lists the reference voices available for cloning."""
        return list_voices()

    @staticmethod
    def get_voice(name: str) -> Voice:
        """Returns one reference voice by name."""
        return resolve_voice(name)

    @staticmethod
    def voices_directory() -> Path:
        """Returns the directory reference voices are read from."""
        return voices_dir()

    # --- ince ayar -----------------------------------------------------------

    @staticmethod
    def finetune(dataset: str, output_dir: str, **overrides) -> FinetuneConfig:
        """Fine-tunes a LoRA adapter on a JSONL dataset of audio and text pairs."""
        return _finetune(dataset, output_dir, **overrides)

    @staticmethod
    def check_dataset(dataset: str, **options):
        """Validates a fine-tuning dataset and reports its statistics."""
        return _check_dataset(dataset, **options)

    # --- ayarlar -------------------------------------------------------------

    @staticmethod
    def settings() -> Settings:
        """Returns the settings currently in effect."""
        return settings

    @staticmethod
    def reload_settings() -> Settings:
        """Re-reads the configuration file and environment, returning fresh settings."""
        return load_settings()


__all__ = [
    "TurkishTTS",
    "FinetuneConfig",
    "Settings",
    "Voice",
    "VoiceNotFound",
    "check_dataset",
    "finetune",
    "list_voices",
    "load_settings",
    "resolve_voice",
    "settings",
    "voices_dir",
    "__version__",
]

check_dataset = _check_dataset
finetune = _finetune
