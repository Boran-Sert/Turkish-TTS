"""Runtime settings, read from TTS_ environment variables over an optional JSON file."""

import json
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

CONFIG_ENV_VAR = "TTS_CONFIG_FILE"
DEFAULT_CONFIG_NAME = "tts_config.json"


class ModelSettings(BaseModel):
    """Model location and sampling quality."""

    model_config = {"protected_namespaces": ()}

    model_path: str = Field(default="Trendyol/Trendyol-TTS")
    inference_timesteps: int = Field(default=8)
    adaptive_timesteps: bool = Field(default=False)
    adaptive_timesteps_short: int = Field(default=6)
    adaptive_timesteps_long: int = Field(default=12)
    cfg_value: float = Field(default=2.0)
    max_length: int = Field(default=4096)


class StreamingSettings(BaseModel):
    """How generated audio is chunked and encoded."""

    chunk_duration_ms: int = Field(default=200)
    output_format: str = Field(default="pcm16_le")
    enable_lookbehind: bool = Field(default=True)
    lookbehind_mode: str = Field(default="anchor")


class TextProcessingSettings(BaseModel):
    """How incoming text is split into sentences."""

    sentence_delimiters: str = Field(default=".?!…\n")
    flush_timeout_ms: float = Field(default=300.0)
    min_chars: int = Field(default=30)


class SystemSettings(BaseModel):
    """Device pool and request admission."""

    voxcpm_gpu_ids: List[int] = Field(default_factory=lambda: [0])
    use_torch_compile: bool = Field(default=False)
    pool_acquire_timeout_s: float = Field(default=30.0)
    ws_idle_timeout_s: float = Field(default=300.0)


class ApiSettings(BaseModel):
    """CORS policy for the HTTP surface."""

    cors_allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_allowed_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_allowed_headers: List[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseModel):
    """Every runtime setting, grouped by concern."""

    voices_dir: str = Field(default="voices")
    model: ModelSettings = Field(default_factory=ModelSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    text_processing: TextProcessingSettings = Field(default_factory=TextProcessingSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)


_SECTION_PREFIXES = {
    "MODEL_": "model",
    "STREAMING_": "streaming",
    "TEXT_": "text_processing",
    "SYSTEM_": "system",
    "API_": "api",
}


def _config_path() -> Path | None:
    """Returns the JSON config path from the environment or the working directory."""
    explicit = os.getenv(CONFIG_ENV_VAR)
    if explicit:
        return Path(explicit)
    local = Path.cwd() / DEFAULT_CONFIG_NAME
    return local if local.is_file() else None


def _from_file() -> dict:
    """Reads the JSON config file, or returns an empty mapping when absent."""
    path = _config_path()
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"{CONFIG_ENV_VAR}={path} okunamadı: dosya yok.")
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce(raw: str):
    """Parses an environment value as JSON, falling back to the plain string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _from_env() -> dict:
    """Collects TTS_<SECTION>_<FIELD> variables into a nested mapping."""
    overrides: dict = {}
    for name, raw in os.environ.items():
        if not name.startswith("TTS_"):
            continue
        remainder = name[len("TTS_"):]
        for prefix, section in _SECTION_PREFIXES.items():
            if remainder.startswith(prefix):
                field = remainder[len(prefix):].lower()
                overrides.setdefault(section, {})[field] = _coerce(raw)
                break
    return overrides


def load_settings() -> Settings:
    """Builds settings from defaults, then the JSON file, then the environment."""
    merged = _from_file()
    for section, values in _from_env().items():
        merged.setdefault(section, {}).update(values)
    return Settings(**merged)


settings = load_settings()
